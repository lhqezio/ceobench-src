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
    and ./novamind-operation CLI.

    After calling `./novamind-operation next-week`, context is refreshed:
    the conversation is cleared and rebuilt with system prompt + MEMORY.md
    contents + the new dashboard.
    """

    def __init__(
        self,
        tool_descriptions: List[Dict[str, Any]],
        client,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        max_turns_per_day: int = 0,  # 0 = no limit
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

        # Detect if the endpoint supports OpenAI Responses API.
        # Google's OpenAI-compat endpoint only supports chat.completions (not /responses).
        base_url = str(getattr(client, 'base_url', '') or '')
        self.supports_responses_api = 'generativelanguage.googleapis.com' not in base_url

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

        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_cached_tokens: int = 0
        self.last_reasoning_tokens: int = 0

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

        # Safety: force next_week if too many turns (0 = no limit)
        if self.max_turns_per_day > 0 and self.turns_today >= self.max_turns_per_day:
            return Action(tool='bash', arguments={'command': './novamind-operation next-week'})

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
        elif self.reasoning_effort and self.supports_responses_api:
            return self._call_openai_responses()
        else:
            return self._call_openai()

    def _call_openai(self) -> Optional[Action]:
        """Call OpenAI-compatible API and parse the response."""
        import time as _time
        import traceback
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = 600  # 10min hard wall-clock limit per LLM call

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

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
                    'max_completion_tokens': 16384,
                    'temperature': 1.0,
                }
                if self.reasoning_effort:
                    api_kwargs['reasoning_effort'] = self.reasoning_effort

                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                self.total_turns += 1
                self._consecutive_errors = 0

                # Capture token usage (OpenAI chat completions format)
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                    self.last_output_tokens = getattr(usage, 'completion_tokens', 0) or 0
                    # Cache and reasoning details
                    ptd = getattr(usage, 'prompt_tokens_details', None)
                    self.last_cached_tokens = getattr(ptd, 'cached_tokens', 0) or 0 if ptd else 0
                    ctd = getattr(usage, 'completion_tokens_details', None)
                    self.last_reasoning_tokens = getattr(ctd, 'reasoning_tokens', 0) or 0 if ctd else 0
                else:
                    self.last_input_tokens = 0
                    self.last_output_tokens = 0
                    self.last_cached_tokens = 0
                    self.last_reasoning_tokens = 0
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_cached_tokens += self.last_cached_tokens
                self.total_reasoning_tokens += self.last_reasoning_tokens

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

                # Validate tool_call arguments are parseable JSON BEFORE appending
                # to conversation. Storing a tool_call with invalid-JSON args poisons
                # the history: some OpenAI-compat servers (e.g. Together) reject every
                # subsequent request with 400 "Input validation error" on replay.
                json_validation_error = None
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        if tc.function.arguments:
                            try:
                                json.loads(tc.function.arguments)
                            except json.JSONDecodeError as je:
                                json_validation_error = (
                                    tc.function.name,
                                    str(je),
                                    tc.function.arguments[:300],
                                )
                                break

                if json_validation_error:
                    name, err, preview = json_validation_error
                    print(f"  Invalid JSON in tool_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                tool_calls_data = None
                if assistant_msg.tool_calls:
                    tool_calls_data = []
                    for tc in assistant_msg.tool_calls:
                        tc_dict = {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        }
                        # Preserve Gemini thought_signature (required by Gemini
                        # OpenAI-compat endpoint — must be echoed back on replay).
                        tc_extras = getattr(tc, 'model_extra', None) or {}
                        extra_content = tc_extras.get('extra_content')
                        if extra_content:
                            tc_dict['extra_content'] = extra_content
                        tool_calls_data.append(tc_dict)

                self.conversation.append(Message(
                    role='assistant',
                    content=assistant_msg.content or '',
                    tool_calls=tool_calls_data
                ))

                if not assistant_msg.tool_calls:
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no tool_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            "You must call a tool to proceed. If you have nothing else to do this week, "
                            "call `./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash to advance."
                        )
                    ))
                    continue

                # Handle tool calls — execute first, skip rest
                first_tc = assistant_msg.tool_calls[0]
                # Safe to parse — we already validated above.
                args = json.loads(first_tc.function.arguments) if first_tc.function.arguments else {}

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
                    is_retryable = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, LLMTimeoutError))
                if not is_retryable:
                    is_retryable = any(code in str(e) for code in ('429', '500', '502', '503', '504', '529'))
                print(f"OpenAI LLM call error (retryable={is_retryable}, status={status}): {e}")
                if is_retryable:
                    # Retry with exponential backoff — keep trying forever until
                    # the endpoint comes back. Never fall back to next-week.
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(120, 10 * (2 ** min(self._consecutive_errors - 1, 3)))
                    print(f"  Server error ({self._consecutive_errors}), retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    # Free memory before retry (messages/tools rebuilt at top of loop)
                    del messages, tools
                    continue  # Loop back to retry
                else:
                    # Non-retryable error (4xx other than 429). No next-week fallback —
                    # feed the error message back to the LLM as a user turn and regenerate.
                    print(f"  Non-retryable error — feeding back to LLM for regeneration.")
                    print(f"Traceback: {traceback.format_exc()}")
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(60, 5 * self._consecutive_errors)
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"The previous API request failed with a non-retryable error:\n"
                            f"{type(e).__name__}: {e}\n\n"
                            f"Please re-emit your response. If the error mentions input validation, "
                            f"check your tool_call arguments are valid JSON. "
                            f"If the error mentions context length, produce a shorter response."
                        )
                    ))
                    _time.sleep(wait_time)
                    del messages, tools
                    continue

    def _call_openai_responses(self) -> Optional[Action]:
        """Call OpenAI Responses API (required for reasoning models with tools)."""
        import time as _time
        import traceback
        import signal
        import openai

        LLM_WALL_CLOCK_TIMEOUT = 600

        class LLMTimeoutError(Exception):
            pass

        def _llm_timeout_handler(signum, frame):
            raise LLMTimeoutError(f"LLM call exceeded {LLM_WALL_CLOCK_TIMEOUT}s wall-clock timeout")

        while True:
            # Build input array from conversation
            input_items = []
            for msg in self.conversation:
                if msg.role == 'system':
                    continue  # System prompt goes in instructions parameter
                elif msg.role == 'user':
                    input_items.append({'role': 'user', 'content': msg.content or ''})
                elif msg.role == 'assistant':
                    if isinstance(msg.content, list):
                        # Raw response.output items from previous Responses API call
                        input_items.extend(msg.content)
                    else:
                        input_items.append({'role': 'assistant', 'content': msg.content or ''})
                elif msg.role == 'tool':
                    input_items.append({
                        'type': 'function_call_output',
                        'call_id': msg.tool_call_id,
                        'output': msg.content or '',
                    })

            # Build tools (Responses API format — no nested function wrapper)
            tools = [
                {
                    'type': 'function',
                    'name': t['name'],
                    'description': t['description'],
                    'parameters': t['parameters'],
                }
                for t in self.tool_descriptions
            ]

            try:
                api_kwargs = {
                    'model': self.model,
                    'input': input_items,
                    'tools': tools,
                    'tool_choice': 'auto',
                    'max_output_tokens': 16384,
                    'instructions': self._get_system_prompt_with_memory(),
                }
                if self.reasoning_effort:
                    api_kwargs['reasoning'] = {'effort': self.reasoning_effort}

                # Set hard wall-clock timeout via signal.alarm
                old_handler = signal.signal(signal.SIGALRM, _llm_timeout_handler)
                signal.alarm(LLM_WALL_CLOCK_TIMEOUT)
                try:
                    response = self.client.responses.create(**api_kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                self.total_turns += 1
                self._consecutive_errors = 0

                # Capture token usage (Responses API uses input_tokens/output_tokens)
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_input_tokens = getattr(usage, 'input_tokens', 0) or 0
                    self.last_output_tokens = getattr(usage, 'output_tokens', 0) or 0
                    # Cache and reasoning details
                    itd = getattr(usage, 'input_tokens_details', None)
                    self.last_cached_tokens = getattr(itd, 'cached_tokens', 0) or 0 if itd else 0
                    otd = getattr(usage, 'output_tokens_details', None)
                    self.last_reasoning_tokens = getattr(otd, 'reasoning_tokens', 0) or 0 if otd else 0
                else:
                    self.last_input_tokens = 0
                    self.last_output_tokens = 0
                    self.last_cached_tokens = 0
                    self.last_reasoning_tokens = 0
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_cached_tokens += self.last_cached_tokens
                self.total_reasoning_tokens += self.last_reasoning_tokens

                if self.response_callback:
                    self.response_callback(
                        turn=self.total_turns,
                        day=self.current_day,
                        messages=input_items,
                        raw_response=response.model_dump() if hasattr(response, 'model_dump') else str(response),
                    )

                # Log reasoning content if present
                for item in response.output:
                    if getattr(item, 'type', '') == 'reasoning' and self.tool_result_callback:
                        reasoning_text = ''
                        for summary in getattr(item, 'summary', []) or []:
                            reasoning_text += getattr(summary, 'text', '') + '\n'
                        if reasoning_text.strip():
                            self.tool_result_callback(
                                self.total_turns, self.current_day, '_reasoning', {},
                                reasoning_text.strip()
                            )

                # Find function_call items
                function_calls = [item for item in response.output
                                  if getattr(item, 'type', '') == 'function_call']

                # Validate each function_call's arguments JSON BEFORE storing. An
                # invalid-JSON tool_call poisons the conversation (server-side
                # validators reject every subsequent request on replay).
                json_validation_error = None
                for fc in function_calls:
                    if fc.arguments:
                        try:
                            json.loads(fc.arguments)
                        except json.JSONDecodeError as je:
                            json_validation_error = (
                                fc.name,
                                str(je),
                                fc.arguments[:300],
                            )
                            break

                if json_validation_error:
                    name, err, preview = json_validation_error
                    print(f"  Invalid JSON in function_call `{name}`: {err}. Feeding error back to LLM and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
                            f"JSON decode error: {err}\n"
                            f"Arguments started with: {preview}...\n\n"
                            f"Valid JSON escape sequences are limited to: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. "
                            f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
                            f"Please re-emit the tool call with valid JSON."
                        )
                    ))
                    continue

                # Store raw output items for conversation history reconstruction
                self.conversation.append(Message(
                    role='assistant',
                    content=list(response.output),
                ))

                if not function_calls:
                    # LLM emitted no tool_call — feed feedback and retry.
                    print("  LLM returned no function_call. Feeding feedback and regenerating.")
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            "You must call a tool to proceed. If you have nothing else to do this week, "
                            "call `./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash to advance."
                        )
                    ))
                    continue

                # Handle tool calls — execute first, skip rest
                first_fc = function_calls[0]
                # Safe to parse — we already validated above.
                args = json.loads(first_fc.arguments) if first_fc.arguments else {}

                # Skip extra parallel tool calls
                for extra_fc in function_calls[1:]:
                    self.conversation.append(Message(
                        role='tool',
                        content=f"[Skipped - only one tool per turn. Call {extra_fc.name} again if needed.]",
                        tool_call_id=extra_fc.call_id,
                        name=extra_fc.name
                    ))

                self._pending_tool_calls = [{'id': first_fc.call_id, 'name': first_fc.name}]
                return Action(tool=first_fc.name, arguments=args)

            except Exception as e:
                status = getattr(e, 'status_code', 0) or 0
                is_retryable = isinstance(e, openai.APIStatusError) and (status >= 500 or status == 429)
                if not is_retryable:
                    is_retryable = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, LLMTimeoutError))
                if not is_retryable:
                    is_retryable = any(code in str(e) for code in ('429', '500', '502', '503', '504', '529'))
                print(f"OpenAI Responses API error (retryable={is_retryable}, status={status}): {e}")
                if is_retryable:
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(120, 10 * (2 ** min(self._consecutive_errors - 1, 3)))
                    print(f"  Server error ({self._consecutive_errors}), retrying in {wait_time}s...")
                    _time.sleep(wait_time)
                    del input_items, tools
                    continue  # Loop back to retry
                else:
                    # Non-retryable error. No next-week fallback —
                    # feed the error message back to the LLM and regenerate.
                    print(f"  Non-retryable error — feeding back to LLM for regeneration.")
                    print(f"Traceback: {traceback.format_exc()}")
                    self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                    wait_time = min(60, 5 * self._consecutive_errors)
                    self.conversation.append(Message(
                        role='user',
                        content=(
                            f"The previous API request failed with a non-retryable error:\n"
                            f"{type(e).__name__}: {e}\n\n"
                            f"Please re-emit your response. If the error mentions input validation, "
                            f"check your tool_call arguments are valid JSON. "
                            f"If the error mentions context length, produce a shorter response."
                        )
                    ))
                    _time.sleep(wait_time)
                    del input_items, tools
                    continue

    def _call_anthropic(self) -> Optional[Action]:
        """Call Anthropic/Bedrock API and parse the response."""
        import copy
        messages = []
        for msg in self.conversation:
            if msg.role == 'system':
                continue
            messages.append({'role': msg.role, 'content': copy.deepcopy(msg.content)})

        # Strip any leftover cache_control from previous messages, then add
        # a single breakpoint on the last message.  Combined with the system
        # prompt and tools breakpoints this stays within the 4-breakpoint limit.
        def _strip_cache_control(content):
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and 'cache_control' in block:
                        del block['cache_control']

        for msg in messages:
            _strip_cache_control(msg.get('content'))

        # Add cache_control to the last message so the entire conversation
        # prefix is cached between consecutive turns.
        if messages:
            last_msg = messages[-1]
            content = last_msg.get('content', '')
            if isinstance(content, str) and content:
                last_msg['content'] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    last_block['cache_control'] = {"type": "ephemeral"}

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

        # Claude 4.x thinking API (adaptive effort). Opus 4.7 requires this
        # shape; 4.6 accepts it too. Old {'type':'enabled','budget_tokens': N}
        # is rejected by 4.7.
        _VALID_EFFORTS = {'low', 'medium', 'high', 'xhigh', 'max'}
        api_kwargs = {
            'model': self.model,
            'max_tokens': 32768,
            'system': system_content,
            'messages': messages,
            'tools': tools,
        }
        use_streaming = False
        if self.reasoning_effort and self.reasoning_effort in _VALID_EFFORTS:
            api_kwargs['thinking'] = {'type': 'adaptive'}
            api_kwargs['output_config'] = {'effort': self.reasoning_effort}
            # Stream to avoid SDK's non-streaming 10-minute budget refusal
            # when thinking can produce long outputs.
            use_streaming = True

        try:
            if use_streaming:
                with self.client.messages.stream(**api_kwargs) as stream:
                    response = stream.get_final_message()
            else:
                response = self.client.messages.create(**api_kwargs)

            self.total_turns += 1
            self._consecutive_errors = 0

            # Capture token usage (Anthropic format)
            usage = getattr(response, 'usage', None)
            if usage:
                self.last_input_tokens = getattr(usage, 'input_tokens', 0) or 0
                self.last_output_tokens = getattr(usage, 'output_tokens', 0) or 0
                # Anthropic cache tracking: cache_creation_input_tokens + cache_read_input_tokens
                self.last_cached_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
                self.last_reasoning_tokens = 0  # Anthropic doesn't expose reasoning tokens separately
            else:
                self.last_input_tokens = 0
                self.last_output_tokens = 0
                self.last_cached_tokens = 0
                self.last_reasoning_tokens = 0
            self.total_input_tokens += self.last_input_tokens
            self.total_output_tokens += self.last_output_tokens
            self.total_cached_tokens += self.last_cached_tokens
            self.total_reasoning_tokens += self.last_reasoning_tokens

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
