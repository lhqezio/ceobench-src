"""Baseline LLM agent for SaaS Bench.

This agent uses an LLM to make decisions. Supports both OpenAI-compatible APIs
(OpenAI, xAI) and Anthropic APIs (direct, Bedrock).
It maintains conversation context and refreshes it after calling next_day.
"""

import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..base import BaseAgent
from ...environment import Action, StepResult


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class BaselineAgent(BaseAgent):
    """Baseline LLM agent for decision making.

    Supports both OpenAI-compatible APIs (OpenAI, xAI) and Anthropic APIs
    (direct, Bedrock). Detects the client type automatically.

    Features:
    - Maintains conversation context within a day
    - Refreshes context after calling next_day (new day = fresh context)
    - Manages its own memory (not environment-side)
    - Uses function calling to select tools
    """

    def __init__(
        self,
        tool_descriptions: List[Dict[str, Any]],
        client,  # OpenAI or anthropic.Anthropic or AnthropicBedrock
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        max_turns_per_day: int = 100,
        response_callback: Optional[callable] = None,
        reasoning_effort: Optional[str] = None,
        tool_result_callback: Optional[callable] = None,
    ):
        """Initialize the baseline agent.

        Args:
            tool_descriptions: Tool descriptions from environment
            client: API client (OpenAI, Anthropic, or AnthropicBedrock)
            model: Model to use (default gpt-4o)
            system_prompt: Custom system prompt (uses default if None)
            max_turns_per_day: Max tool calls per day before forcing next_day
            response_callback: Optional callback for logging raw responses
            reasoning_effort: Reasoning effort level for GPT-5.2+ (none, low, medium, high, xhigh)
            tool_result_callback: Optional callback for logging tool results (turn, day, tool_name, args, result)
        """
        super().__init__(tool_descriptions)
        self.client = client
        self.model = model
        self.max_turns_per_day = max_turns_per_day
        self.response_callback = response_callback
        self.reasoning_effort = reasoning_effort
        self.tool_result_callback = tool_result_callback

        # Detect client type
        client_type = type(client).__name__
        self.use_anthropic = client_type in ('Anthropic', 'AnthropicBedrock')

        # Build system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Agent state
        self.conversation: List[Message] = []
        self.memory: List[str] = []  # Persistent notes
        self.current_day: int = 0
        self.turns_today: int = 0
        self._pending_tool_calls: List[Dict] = []
        self._last_observation: str = ""
        self.total_turns: int = 0  # Total turns across all days

    def _default_system_prompt(self) -> str:
        """Build the default system prompt template.

        Loads the baseline-specific template from system_prompt.md and fills in
        {simulator_instructions} from the shared simulator_instructions.md file.
        The {memory} placeholder is left unfilled — it gets filled at render time
        by _get_system_with_memory().
        """
        from pathlib import Path
        base_dir = Path(__file__).parent

        from ...tools import get_tool_summary_table
        simulator_file = base_dir.parent / "simulator_instructions.md"
        with open(simulator_file, 'r') as f:
            simulator_instructions = f.read().format(tool_list=get_tool_summary_table())

        template_file = base_dir / "system_prompt.md"
        with open(template_file, 'r') as f:
            template = f.read()

        # Only fill simulator_instructions; leave {memory} for render time
        return template.format(simulator_instructions=simulator_instructions, memory="{memory}")

    def reset(self):
        """Reset agent state for a new episode."""
        self.conversation = []
        self.memory = []
        self.current_day = 0
        self.turns_today = 0
        self._pending_tool_calls = []
        self._last_observation = ""

    def act(self, observation: str, reward: float, done: bool, info: Dict[str, Any]) -> Optional[Action]:
        """Choose an action based on the observation.

        The agent processes tool outputs and decides the next action.
        After calling next_day, context is refreshed for the new day.

        Args:
            observation: Tool output or dashboard string
            reward: Reward from previous action
            done: Whether episode is finished
            info: Additional info from environment

        Returns:
            Action to take, or None if done
        """
        if done:
            return None

        self._last_observation = observation

        # Check if this is a new day (context refresh)
        current_day = info.get('day', 0)
        if current_day > self.current_day:
            self._refresh_context(observation, current_day)
            self.current_day = current_day
            self.turns_today = 0

        # Safety: force next_day if too many turns
        if self.turns_today >= self.max_turns_per_day:
            return Action(tool='next_day')

        # If we have pending tool call results to process, add them
        if self._pending_tool_calls:
            if self.use_anthropic:
                # Anthropic format: tool results are user messages with content blocks
                partial_results = self._pending_tool_calls[0].get('_partial_results', [])
                tool_results = list(partial_results)
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': self._pending_tool_calls[0]['id'],
                    'content': observation,
                })
                self.conversation.append(Message(
                    role='user',
                    content=tool_results,
                ))
            else:
                # OpenAI format: tool results are separate messages
                for tc in self._pending_tool_calls:
                    self.conversation.append(Message(
                        role='tool',
                        content=observation,
                        tool_call_id=tc['id'],
                        name=tc['name']
                    ))
            self._pending_tool_calls = []
        else:
            # Add observation as user message (e.g., initial dashboard)
            self.conversation.append(Message(
                role='user',
                content=observation
            ))

        # Call LLM
        action = self._call_llm()
        self.turns_today += 1

        return action

    def _get_system_with_memory(self) -> str:
        """Render the system prompt with current memory filled in."""
        if self.memory:
            memory_text = "=== YOUR NOTES ===\n" + '\n'.join(
                f"{i+1}. {line}" for i, line in enumerate(self.memory)
            )
        else:
            memory_text = ""
        return self.system_prompt.format(memory=memory_text)

    def _refresh_context(self, dashboard: str, new_day: int):
        """Refresh conversation context for a new day.

        Keeps system prompt and memory, but clears conversation history.
        For OpenAI: system prompt is added as a system message.
        For Anthropic: system prompt is passed separately in the API call.
        """
        self.conversation = []
        # Clear pending tool calls to avoid stale tool_call_id references on new day
        self._pending_tool_calls = []

        # Log memory contents to JSONL so external monitor can display them
        if self.tool_result_callback:
            if self.memory:
                memory_display = '\n'.join(f"{i+1}. {line}" for i, line in enumerate(self.memory))
            else:
                memory_display = "(empty)"
            self.tool_result_callback(0, new_day, '_memory', {}, memory_display)

        if not self.use_anthropic:
            # OpenAI: system prompt goes in messages
            self.conversation.append(Message(
                role='system',
                content=self._get_system_with_memory()
            ))

    def _call_llm(self) -> Optional[Action]:
        """Call the LLM and parse the response into an action.

        Dispatches to the appropriate API format based on client type.
        """
        if self.use_anthropic:
            return self._call_anthropic()
        else:
            return self._call_openai()

    def _call_openai(self) -> Optional[Action]:
        """Call OpenAI-compatible API and parse the response."""
        # Build messages for API
        messages = []
        for msg in self.conversation:
            m = {'role': msg.role, 'content': msg.content or ''}
            if msg.tool_call_id:
                m['tool_call_id'] = msg.tool_call_id
            if msg.name:
                m['name'] = msg.name
            if msg.tool_calls:
                m['tool_calls'] = msg.tool_calls
            messages.append(m)

        # Build tools for API (OpenAI format) — env tools + memory tools
        all_tool_defs = list(self.tool_descriptions) + list(self._MEMORY_TOOL_DEFS)
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': t['name'],
                    'description': t['description'],
                    'parameters': t['parameters']
                }
            }
            for t in all_tool_defs
        ]

        try:
            # Build API call kwargs
            api_kwargs = {
                'model': self.model,
                'messages': messages,
                'tools': tools,
                'tool_choice': 'auto',
                'max_tokens': 16384,
            }

            # Add reasoning effort if specified (for GPT-5.2+)
            if self.reasoning_effort:
                api_kwargs['reasoning_effort'] = self.reasoning_effort

            response = self.client.chat.completions.create(**api_kwargs)

            self.total_turns += 1
            self._consecutive_errors = 0

            # Log raw response if callback provided
            if self.response_callback:
                self.response_callback(
                    turn=self.total_turns,
                    day=self.current_day,
                    messages=messages,
                    raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                )

            # Process response
            assistant_msg = response.choices[0].message

            # Log reasoning_content if present (e.g. GLM-5 reasoning model)
            # Try attribute first, then model_extra dict (OpenAI SDK stores unknown fields there)
            reasoning_content = getattr(assistant_msg, 'reasoning_content', None)
            if not reasoning_content:
                extras = getattr(assistant_msg, 'model_extra', {}) or {}
                reasoning_content = extras.get('reasoning_content')
            if reasoning_content and self.tool_result_callback:
                self.tool_result_callback(
                    self.total_turns, self.current_day, '_reasoning', {},
                    reasoning_content
                )

            # Add assistant message to conversation
            tool_calls_data = None
            if assistant_msg.tool_calls:
                tool_calls_data = [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    }
                    for tc in assistant_msg.tool_calls
                ]

            self.conversation.append(Message(
                role='assistant',
                content=assistant_msg.content or '',
                tool_calls=tool_calls_data
            ))

            # If no tool calls, return None (let agent think more)
            if not assistant_msg.tool_calls:
                return None

            # Handle ALL tool calls (OpenAI can return multiple parallel tool calls)
            # First, handle any memory tools (agent-side)
            env_tool_calls = []
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                if tool_name.startswith('memory_'):
                    # Handle memory tool immediately
                    result = self._handle_memory_tool(tool_name, args)
                    self.conversation.append(Message(
                        role='tool',
                        content=result,
                        tool_call_id=tc.id,
                        name=tool_name
                    ))
                else:
                    # Queue environment tool for execution
                    env_tool_calls.append({'id': tc.id, 'name': tool_name, 'args': args})

            # If only memory tools were called, recurse
            if not env_tool_calls:
                return self._call_llm()

            # For environment tools, we can only execute one at a time
            # Execute the first one, and add dummy responses for the rest
            first_tool = env_tool_calls[0]

            # For any additional parallel tool calls, add placeholder responses
            for extra_tc in env_tool_calls[1:]:
                self.conversation.append(Message(
                    role='tool',
                    content=f"[Skipped - only one tool can be executed per turn. Please call {extra_tc['name']} again if needed.]",
                    tool_call_id=extra_tc['id'],
                    name=extra_tc['name']
                ))

            # Track the first tool call for result matching
            self._pending_tool_calls = [{'id': first_tool['id'], 'name': first_tool['name']}]

            return Action(tool=first_tool['name'], arguments=first_tool['args'])

        except Exception as e:
            import traceback
            print(f"OpenAI LLM call error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # On error, try to advance day
            return Action(tool='next_day')

    def _call_anthropic(self) -> Optional[Action]:
        """Call Anthropic/Bedrock API and parse the response."""
        # Build messages for Anthropic API (user/assistant only, no system in messages)
        messages = []
        for msg in self.conversation:
            if msg.role == 'system':
                continue  # System prompt is passed separately
            messages.append({'role': msg.role, 'content': msg.content})

        # Build system prompt with memory as structured content blocks for caching
        system_text = self._get_system_with_memory()
        system_content = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Build tools for Anthropic API format — env tools + memory tools
        all_tool_defs = list(self.tool_descriptions) + list(self._MEMORY_TOOL_DEFS)
        tools = [
            {
                'name': t['name'],
                'description': t['description'],
                'input_schema': t['parameters']
            }
            for t in all_tool_defs
        ]
        # Mark the last tool with cache_control so all tool definitions are cached
        if tools:
            tools[-1]['cache_control'] = {"type": "ephemeral"}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_content,
                messages=messages,
                tools=tools,
            )

            self.total_turns += 1
            self._consecutive_errors = 0

            # Log raw response if callback provided
            if self.response_callback:
                self.response_callback(
                    turn=self.total_turns,
                    day=self.current_day,
                    messages=messages,
                    raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                )

            # Process response - find tool_use blocks
            assistant_content = response.content

            # Add assistant message to conversation
            self.conversation.append(Message(
                role='assistant',
                content=assistant_content
            ))

            # Find ALL tool_use blocks in response
            tool_use_blocks = [block for block in assistant_content if block.type == 'tool_use']

            if not tool_use_blocks:
                return None

            # Separate memory tools from env tools
            env_tool_blocks = []
            for block in tool_use_blocks:
                if block.name.startswith('memory_'):
                    # Handle memory tool immediately
                    result = self._handle_memory_tool(block.name, block.input or {})
                    # Add tool result to conversation (Anthropic format)
                    self.conversation.append(Message(
                        role='user',
                        content=[{
                            'type': 'tool_result',
                            'tool_use_id': block.id,
                            'content': result,
                        }]
                    ))
                else:
                    env_tool_blocks.append(block)

            # If only memory tools were called, recurse
            if not env_tool_blocks:
                return self._call_llm()

            # Execute the first env tool; skip extras
            first_tool = env_tool_blocks[0]

            # Build partial tool_results for skipped extra tools
            partial_results = []
            for extra in env_tool_blocks[1:]:
                partial_results.append({
                    'type': 'tool_result',
                    'tool_use_id': extra.id,
                    'content': f"[Skipped - only one tool can be executed per turn. Please call {extra.name} again if needed.]",
                })

            # Store pending info (reuse _pending_tool_calls for the ID + partial results)
            self._pending_tool_calls = [{'id': first_tool.id, 'name': first_tool.name, '_partial_results': partial_results}]

            return Action(tool=first_tool.name, arguments=first_tool.input or {})

        except Exception as e:
            import traceback
            error_msg = f"Anthropic LLM call error: {e}"
            tb = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"ERROR in BaselineAgent._call_anthropic()")
            print(f"{'='*60}")
            print(error_msg)
            print(f"Traceback:\n{tb}")
            print(f"{'='*60}\n")

            # Retry with exponential backoff (3 attempts)
            if not hasattr(self, '_consecutive_errors'):
                self._consecutive_errors = 0
            self._consecutive_errors += 1

            if self._consecutive_errors <= 3:
                wait = 2 ** self._consecutive_errors
                print(f"  Retrying in {wait}s (attempt {self._consecutive_errors}/3)...")
                time.sleep(wait)
                return self._call_anthropic()

            raise RuntimeError(
                f"LLM failed {self._consecutive_errors} consecutive times. "
                f"Last error: {e}"
            ) from e

    # Memory tool definitions (shared data, formatted per API)

    _MEMORY_TOOL_DEFS = [
        {
            'name': 'memory_add',
            'description': 'Add a note to your persistent memory. Notes persist across days.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'note': {'type': 'string', 'description': 'The note to add'}
                },
                'required': ['note']
            }
        },
        {
            'name': 'memory_edit',
            'description': 'Edit an existing note by index (1-indexed). Replaces the note content.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer', 'description': 'Note index to edit (1-indexed)'},
                    'note': {'type': 'string', 'description': 'The new note content'}
                },
                'required': ['index', 'note']
            }
        },
        {
            'name': 'memory_clear',
            'description': 'Clear all notes from memory.',
            'parameters': {'type': 'object', 'properties': {}}
        },
        {
            'name': 'memory_remove',
            'description': 'Remove a note by index (1-indexed).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer', 'description': 'Note index to remove (1-indexed)'}
                },
                'required': ['index']
            }
        },
    ]

    def _handle_memory_tool(self, tool_name: str, args: Dict) -> str:
        """Handle memory tools (agent-side)."""
        if tool_name == 'memory_add':
            note = args.get('note', '')
            self.memory.append(note)
            result = f"Added note. Memory now has {len(self.memory)} notes."

        elif tool_name == 'memory_clear':
            self.memory = []
            result = "Memory cleared."

        elif tool_name == 'memory_edit':
            idx = args.get('index', 0)
            new_note = args.get('note', '')
            if 1 <= idx <= len(self.memory):
                old_note = self.memory[idx - 1]
                self.memory[idx - 1] = new_note
                result = f"Edited note {idx}: '{old_note[:30]}...' → '{new_note[:30]}...'"
            else:
                result = f"Invalid index {idx}. Memory has {len(self.memory)} notes."

        elif tool_name == 'memory_remove':
            idx = args.get('index', 0)
            if 1 <= idx <= len(self.memory):
                removed = self.memory.pop(idx - 1)
                result = f"Removed note {idx}: '{removed[:50]}...'"
            else:
                result = f"Invalid index {idx}. Memory has {len(self.memory)} notes."

        else:
            result = "Unknown memory tool."

        # Log to tool_results JSONL via callback
        if self.tool_result_callback:
            self.tool_result_callback(self.total_turns, self.current_day, tool_name, args, result)

        return result

    def on_episode_end(self, final_info: Dict[str, Any]):
        """Called when episode ends."""
        final_cash = final_info.get('cash', 0)
        final_day = final_info.get('day', 0)
        print(f"Episode ended: Day {final_day}, Final Cash: ${final_cash:,.0f}")


def run_agent_loop(
    env,
    agent: BaselineAgent,
    max_episodes: int = 1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run the agent in a loop with the environment.

    Args:
        env: SaaSBenchEnv instance
        agent: BaselineAgent instance
        max_episodes: Number of episodes to run
        verbose: Print progress

    Returns:
        List of episode results
    """
    results = []

    for episode in range(max_episodes):
        if verbose:
            print(f"\n{'='*60}")
            print(f"EPISODE {episode + 1}")
            print(f"{'='*60}")

        # Reset
        obs, info = env.reset()
        agent.reset()

        total_reward = 0
        step_count = 0

        while True:
            # Get action from agent
            action = agent.act(obs, 0, False, info)

            if action is None:
                # Agent returned None, force next_day
                action = Action(tool='next_day')

            # Execute action
            result = env.step(action)

            obs = result.observation
            total_reward += result.reward
            step_count += 1

            if verbose and action.tool == 'next_day':
                day = result.info.get('day', 0)
                cash = result.info.get('cash', 0)
                print(f"Day {day}: Cash=${cash:,.0f}")

            if result.done or result.truncated:
                agent.on_episode_end(result.info)
                results.append({
                    'episode': episode + 1,
                    'total_reward': total_reward,
                    'steps': step_count,
                    'final_day': result.info.get('day', 0),
                    'final_cash': result.info.get('cash', 0),
                    'bankruptcy': result.info.get('bankruptcy', False),
                })
                break

    return results
