"""Bash agent for SaaS Bench.

A Claude Code-style agent that uses bash and file tools to interact with the
NovaMind SaaS simulator via the novamind_api Python library and CLI.

Supports OpenAI-compatible APIs (OpenAI, xAI) and Anthropic APIs (direct, Bedrock).
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..base import BaseAgent
from ...environment import Action


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Any  # str or list (Anthropic content blocks)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


# Regex to detect dashboard in bash output (day advancement)
_DASHBOARD_RE = re.compile(r'=== Day (\d+) Dashboard ===')


class BashAgent(BaseAgent):
    """Bash agent for SaaS Bench — Claude Code-style.

    Uses bash, read_file, write_file, edit_file, search_files, glob_files
    tools. Interacts with the simulator via novamind_api Python library
    and novamind-operation CLI.

    After calling `novamind-operation next-day`, context is refreshed:
    the conversation is cleared and rebuilt with system prompt + MEMORY.md
    contents + the new dashboard.
    """

    def __init__(
        self,
        tool_descriptions: List[Dict[str, Any]],
        client,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        max_turns_per_day: int = 100,
        response_callback: Optional[callable] = None,
        reasoning_effort: Optional[str] = None,
        tool_result_callback: Optional[callable] = None,
        workspace_path: Optional[Path] = None,
        total_days: int = 3650,
    ):
        super().__init__(tool_descriptions)
        self.client = client
        self.model = model
        self.max_turns_per_day = max_turns_per_day
        self.response_callback = response_callback
        self.reasoning_effort = reasoning_effort
        self.tool_result_callback = tool_result_callback
        self.workspace_path = workspace_path or Path('.')
        self.total_days = total_days

        # Detect client type
        client_type = type(client).__name__
        self.use_anthropic = client_type in ('Anthropic', 'AnthropicBedrock')

        # Build system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Agent state
        self.conversation: List[Message] = []
        self.current_day: int = 0
        self.turns_today: int = 0
        self._pending_tool_calls: List[Dict] = []
        self._last_observation: str = ""
        self.total_turns: int = 0
        self._day_advanced: bool = False
        self._new_dashboard: str = ""
        self._consecutive_errors: int = 0

    def _default_system_prompt(self) -> str:
        """Build the default system prompt.

        Loads the bash_agent system_prompt.md and fills in
        {simulator_instructions} and {total_days}.
        """
        base_dir = Path(__file__).parent

        # Load simulator instructions — strip the {tool_list} placeholder
        # (bash_agent has its own tool reference in system_prompt.md)
        simulator_file = base_dir.parent / "simulator_instructions.md"
        with open(simulator_file, 'r') as f:
            sim_text = f.read()

        sim_text = sim_text.replace('{tool_list}\n', '')
        sim_text = sim_text.replace('{tool_list}', '')

        template_file = base_dir / "system_prompt.md"
        with open(template_file, 'r') as f:
            template = f.read()

        prompt = template.replace('{simulator_instructions}', sim_text)

        # Replace {total_days} placeholder with actual value
        total_years = self.total_days / 365
        years_str = f"{total_years:.0f}" if total_years == int(total_years) else f"{total_years:.1f}"
        prompt = prompt.replace('{total_days}', str(self.total_days))
        prompt = prompt.replace('{total_years}', years_str)
        return prompt

    def _get_system_prompt_with_memory(self) -> str:
        """Return system prompt with MEMORY.md contents appended.

        MEMORY.md is always injected into the system prompt so the agent
        has its persistent notes available without needing to read the file.
        """
        prompt = self.system_prompt
        memory_path = self.workspace_path / 'MEMORY.md'
        if memory_path.exists():
            try:
                memory_content = memory_path.read_text().strip()
                if memory_content:
                    max_memory_chars = 40_000
                    if len(memory_content) > max_memory_chars:
                        memory_content = memory_content[:max_memory_chars] + (
                            "\n\n--- MEMORY.md TRUNCATED ---\n"
                            f"Showing first {max_memory_chars:,} of {len(memory_content):,} characters. "
                            "Use the read_file tool to see the full contents if needed."
                        )
                    prompt += (
                        "\n\n## Your MEMORY.md (auto-loaded)\n\n"
                        "The following is the contents of your MEMORY.md file. "
                        "This is automatically loaded into your context at the start of every day.\n\n"
                        f"{memory_content}"
                    )
            except Exception:
                pass
        return prompt

    def reset(self):
        """Reset agent state for a new episode."""
        self.conversation = []
        self.current_day = 0
        self.turns_today = 0
        self._pending_tool_calls = []
        self._last_observation = ""
        self._day_advanced = False
        self._new_dashboard = ""

    def _refresh_context(self, dashboard: str, new_day: int):
        """Refresh conversation context for a new day.

        Clears conversation, inserts system prompt + dashboard.
        The agent reads its own files via tools when it needs context.
        """
        self.conversation = []
        self._pending_tool_calls = []

        if not self.use_anthropic:
            # OpenAI: system prompt goes in messages
            self.conversation.append(Message(
                role='system',
                content=self._get_system_prompt_with_memory(),
            ))

    def check_day_advanced(self, bash_output: str) -> bool:
        """Check if bash output contains a dashboard (day advanced).

        Returns True if a new day dashboard was detected.
        Stores the dashboard text for context refresh.
        """
        match = _DASHBOARD_RE.search(bash_output)
        if match:
            new_day = int(match.group(1))
            if new_day > self.current_day:
                self._day_advanced = True
                # Extract dashboard from the output (everything from === Day N ===)
                dashboard_start = bash_output.index(f"=== Day {new_day} Dashboard ===")
                self._new_dashboard = bash_output[dashboard_start:]
                return True
        return False

    @property
    def day_advanced(self) -> bool:
        """Whether the last bash command advanced the day."""
        return self._day_advanced

    @property
    def new_dashboard(self) -> str:
        """The dashboard text from the last day advancement."""
        return self._new_dashboard

    def clear_day_advanced(self):
        """Clear the day-advanced flag after the runner processes it."""
        self._day_advanced = False
        self._new_dashboard = ""

    def act(self, observation: str, reward: float, done: bool, info: Dict[str, Any]) -> Optional[Action]:
        """Choose an action based on the observation.

        The agent processes tool outputs and decides the next action.
        After day advancement is detected, context is refreshed.
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
            return Action(tool='bash', arguments={'command': 'novamind-operation next-day'})

        # If we have pending tool call results to process, add them
        if self._pending_tool_calls:
            if self.use_anthropic:
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

    def _call_llm(self) -> Optional[Action]:
        """Call the LLM and parse the response into an action."""
        if self.use_anthropic:
            return self._call_anthropic()
        else:
            return self._call_openai()

    def _call_openai(self) -> Optional[Action]:
        """Call OpenAI-compatible API and parse the response."""
        import time as _time
        import traceback
        import openai

        while True:
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

            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t['description'],
                        'parameters': t['parameters']
                    }
                }
                for t in self.tool_descriptions
            ]

            try:
                api_kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'tools': tools,
                    'tool_choice': 'auto',
                    'max_tokens': 16384,
                    'temperature': 1.0,
                }
                if self.reasoning_effort:
                    api_kwargs['reasoning_effort'] = self.reasoning_effort

                response = self.client.chat.completions.create(**api_kwargs)
                self.total_turns += 1
                self._consecutive_errors = 0

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=messages,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                assistant_msg = response.choices[0].message

                # Log reasoning_content if present (e.g. GLM-5 reasoning model)
                reasoning_content = getattr(assistant_msg, 'reasoning_content', None)
                if not reasoning_content:
                    extras = getattr(assistant_msg, 'model_extra', {}) or {}
                    reasoning_content = extras.get('reasoning_content')
                if reasoning_content and self.tool_result_callback:
                    self.tool_result_callback(
                        self.total_turns, self.current_day, '_reasoning', {},
                        reasoning_content
                    )

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

                if not assistant_msg.tool_calls:
                    return None

                # Handle tool calls — execute first, skip rest
                first_tc = assistant_msg.tool_calls[0]
                try:
                    args = json.loads(first_tc.function.arguments) if first_tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                # Skip extra parallel tool calls
                for extra_tc in assistant_msg.tool_calls[1:]:
                    self.conversation.append(Message(
                        role='tool',
                        content=f"[Skipped - only one tool per turn. Call {extra_tc.function.name} again if needed.]",
                        tool_call_id=extra_tc.id,
                        name=extra_tc.function.name
                    ))

                self._pending_tool_calls = [{'id': first_tc.id, 'name': first_tc.function.name}]
                return Action(tool=first_tc.function.name, arguments=args)

            except Exception as e:
                status = getattr(e, 'status_code', 0) or 0
                is_retryable = isinstance(e, openai.APIStatusError) and (status >= 500 or status == 429)
                if not is_retryable:
                    is_retryable = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError))
                if not is_retryable:
                    is_retryable = any(code in str(e) for code in ('429', '500', '502', '503', '504', '529'))
                print(f"OpenAI LLM call error (retryable={is_retryable}, status={status}): {e}")
                if is_retryable:
                    # Retry with exponential backoff — keep trying forever until
                    # the endpoint comes back. Never fall back to next-day.
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(120, 10 * (2 ** min(self._consecutive_errors - 1, 3)))
                    print(f"  Server error ({self._consecutive_errors}), retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    # Free memory before retry (messages/tools rebuilt at top of loop)
                    del messages, tools
                    continue  # Loop back to retry
                else:
                    print(f"Traceback: {traceback.format_exc()}")
                    return Action(tool='bash', arguments={'command': 'novamind-operation next-day'})

    def _call_anthropic(self) -> Optional[Action]:
        """Call Anthropic/Bedrock API and parse the response."""
        messages = []
        for msg in self.conversation:
            if msg.role == 'system':
                continue
            messages.append({'role': msg.role, 'content': msg.content})

        system_text = self._get_system_prompt_with_memory()
        system_content = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        from .tools import get_bash_agent_anthropic_tools
        tools = get_bash_agent_anthropic_tools()
        if tools:
            tools[-1]['cache_control'] = {"type": "ephemeral"}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16384,
                system=system_content,
                messages=messages,
                tools=tools,
            )

            self.total_turns += 1
            self._consecutive_errors = 0

            if self.response_callback:
                self.response_callback(
                    turn=self.total_turns,
                    day=self.current_day,
                    messages=messages,
                    raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                )

            assistant_content = response.content
            self.conversation.append(Message(
                role='assistant',
                content=assistant_content
            ))

            tool_use_blocks = [block for block in assistant_content if block.type == 'tool_use']
            if not tool_use_blocks:
                return None

            first_tool = tool_use_blocks[0]

            # Skip extra parallel tool calls
            partial_results = []
            for extra in tool_use_blocks[1:]:
                partial_results.append({
                    'type': 'tool_result',
                    'tool_use_id': extra.id,
                    'content': f"[Skipped - only one tool per turn. Call {extra.name} again if needed.]",
                })

            self._pending_tool_calls = [{'id': first_tool.id, 'name': first_tool.name, '_partial_results': partial_results}]
            return Action(tool=first_tool.name, arguments=first_tool.input or {})

        except Exception as e:
            import traceback
            error_msg = f"Anthropic LLM call error: {e}"
            tb = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"ERROR in BashAgent._call_anthropic()")
            print(f"{'='*60}")
            print(error_msg)
            print(f"Traceback:\n{tb}")
            print(f"{'='*60}\n")

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
