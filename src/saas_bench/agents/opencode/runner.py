"""OpenCode Runner for SaaS Bench.

This module manages the execution of OpenCode as the agent for SaaS Bench,
handling session management, workspace isolation, and simulation interaction.

OpenCode is an open-source AI coding agent CLI that supports:
- AGENTS.md for custom instructions (equivalent to CLAUDE.md)
- opencode.json for model and provider configuration
- MCP (Model Context Protocol) for tool integration
- Non-interactive execution via `opencode run`
"""

import hashlib
import json
import os
import subprocess
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from numpy.random import Generator, PCG64

from ...config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from ...database import init_database, get_cash, get_mrr, get_active_subscriber_count, get_config
from ...environment import build_daily_dashboard
from ...simulation import Simulator, DayResult
from ...tools import AgentTools
from ...shocks import ShockManager
from ...event_logger import EventLogger


@dataclass
class RationaleEntry:
    """A single rationale/thinking log entry."""
    timestamp: str
    day: int
    rationale: str
    context: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for the OpenCode agent."""
    model: str = "openai/gpt-5.2"  # OpenCode model format: provider/model
    reasoning_effort: str = "medium"  # none, minimal, low, medium, high, xhigh
    seed: int = 42
    scenario: str = "default"
    total_days: int = 3650
    initial_cash: float = 1_000_000.0
    budget_limit_usd: float = 50.0  # API budget limit
    max_turns_per_day: int = 50  # Max tool calls per day


@dataclass
class RunResult:
    """Result from running the OpenCode agent."""
    run_id: str
    seed: int
    scenario: str
    final_cash: float
    days_run: int
    outcome: str  # 'completed', 'bankrupt', 'budget_exceeded', 'interrupted'
    rationales: List[Dict[str, Any]] = field(default_factory=list)
    log_file: Optional[str] = None
    workspace_dir: Optional[str] = None


class OpenCodeRunner:
    """Manages OpenCode agent execution for SaaS Bench."""

    def __init__(self, config: AgentConfig, workspace_base: Optional[Path] = None):
        """Initialize the runner.

        Args:
            config: Agent configuration
            workspace_base: Base directory for workspaces (each run gets a subdirectory)
        """
        self.config = config
        self.workspace_base = (workspace_base or Path('./opencode_runs')).resolve()

        # Generate unique run ID
        self.run_id = self._generate_run_id()

        # Create isolated workspace for this run
        self.workspace_dir = self.workspace_base / f"run_{self.run_id}"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.agent_workspace = self.workspace_dir / "agent"  # Agent's scratchpad
        self.agent_workspace.mkdir(exist_ok=True)
        self.logs_dir = self.workspace_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # OpenCode config directory
        self.opencode_config_dir = self.agent_workspace / ".opencode"
        self.opencode_config_dir.mkdir(exist_ok=True)

        # Database path
        self.db_path = self.workspace_dir / "world.db"

        # Initialize RNG
        self.rng = Generator(PCG64(config.seed))

        # Components (initialized in setup)
        self.conn: Optional[sqlite3.Connection] = None
        self.simulator: Optional[Simulator] = None
        self.shock_manager: Optional[ShockManager] = None
        self.tools: Optional[AgentTools] = None
        self.event_logger: Optional[EventLogger] = None

        # State
        self.current_day = 0
        self.rationales: List[RationaleEntry] = []
        self.game_ended = False
        self.game_outcome: Optional[str] = None

        # Agent conversation log (for incremental logging)
        self.agent_log_file = self.logs_dir / f"agent_conversation_{self.run_id}.jsonl"
        self.agent_turn_count = 0

        # File change tracking
        self.file_diffs_log = self.logs_dir / f"file_diffs_{self.run_id}.jsonl"
        self._last_file_snapshot: Dict[str, str] = {}  # path -> content hash

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())[:8]

    def _now(self) -> str:
        """Get current UTC timestamp."""
        return datetime.utcnow().isoformat() + "Z"

    def _snapshot_workspace_files(self) -> Dict[str, Tuple[str, str]]:
        """Snapshot all files in agent workspace with their content hashes and content.

        Returns:
            Dict mapping relative path -> (content_hash, content)
        """
        snapshot = {}
        # Skip hidden directories and common non-code files
        skip_dirs = {'.opencode', 'node_modules', '__pycache__', '.git', '.venv'}
        skip_extensions = {'.pyc', '.pyo', '.so', '.o', '.lock'}

        for path in self.agent_workspace.rglob('*'):
            if path.is_file():
                # Skip files in hidden/vendor directories
                rel_path = path.relative_to(self.agent_workspace)
                if any(part in skip_dirs for part in rel_path.parts):
                    continue
                if path.suffix in skip_extensions:
                    continue

                try:
                    content = path.read_text(encoding='utf-8', errors='replace')
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    snapshot[str(rel_path)] = (content_hash, content)
                except Exception:
                    # Skip files that can't be read as text
                    pass
        return snapshot

    def _compute_file_diffs(
        self,
        before: Dict[str, Tuple[str, str]],
        after: Dict[str, Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Compute diffs between two workspace snapshots.

        Returns:
            List of diff entries with type, path, and content changes
        """
        diffs = []
        before_paths = set(before.keys())
        after_paths = set(after.keys())

        # New files
        for path in after_paths - before_paths:
            _, content = after[path]
            diffs.append({
                'type': 'added',
                'path': path,
                'content': content
            })

        # Deleted files
        for path in before_paths - after_paths:
            _, content = before[path]
            diffs.append({
                'type': 'deleted',
                'path': path,
                'previous_content': content
            })

        # Modified files
        for path in before_paths & after_paths:
            before_hash, before_content = before[path]
            after_hash, after_content = after[path]
            if before_hash != after_hash:
                diffs.append({
                    'type': 'modified',
                    'path': path,
                    'previous_content': before_content,
                    'new_content': after_content
                })

        return diffs

    def _log_file_diffs(self, day: int, diffs: List[Dict[str, Any]]) -> None:
        """Log file diffs for a day to the diffs log file."""
        if not diffs:
            return

        entry = {
            'timestamp': self._now(),
            'day': day,
            'diffs': diffs
        }

        with open(self.file_diffs_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def setup(self):
        """Initialize the simulation environment."""
        # Get scenario
        scenario = SCENARIO_PACKS.get(self.config.scenario, ScenarioPack(
            name='Default',
            description='Balanced scenario'
        ))

        # Create benchmark config
        bench_config = BenchmarkConfig(
            seed=self.config.seed,
            total_days=self.config.total_days,
            initial_cash=self.config.initial_cash,
            budget_limit_usd=self.config.budget_limit_usd
        )

        # Initialize database
        self.conn = init_database(self.db_path)

        # Initialize components
        self.simulator = Simulator(self.conn, bench_config, self.rng)
        self.shock_manager = ShockManager(self.conn, self.rng, scenario)
        self.tools = AgentTools(self.conn, 0, self.agent_workspace, self.db_path)

        # Initialize event logger
        self.event_logger = EventLogger(
            run_id=self.run_id,
            output_dir=self.logs_dir,
            seed=self.config.seed,
            scenario=self.config.scenario,
            config={
                'model': self.config.model,
                'reasoning_effort': self.config.reasoning_effort,
                'seed': self.config.seed,
                'scenario': self.config.scenario,
                'total_days': self.config.total_days,
                'initial_cash': self.config.initial_cash,
                'agent_type': 'opencode',
            }
        )

        # Connect event logger to components
        self.simulator.set_event_logger(self.event_logger)
        self.tools.set_event_logger(self.event_logger)

        # Initialize simulation
        self.simulator.initialize()
        self.event_logger.log_run_start()

        # Create AGENTS.md with system prompt
        self._create_agents_md()

        # Create opencode.json config
        self._create_opencode_config()

        # Create MCP config
        self._create_mcp_config()

        # Save initial config
        self._save_config()

    def _save_config(self):
        """Save configuration to workspace."""
        config_file = self.workspace_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'model': self.config.model,
                'reasoning_effort': self.config.reasoning_effort,
                'seed': self.config.seed,
                'scenario': self.config.scenario,
                'total_days': self.config.total_days,
                'initial_cash': self.config.initial_cash,
                'agent_type': 'opencode',
                'created_at': self._now(),
                'opencode_settings': {
                    'compaction': {
                        'enabled': True,
                        'threshold': 0.85  # Compact at 85% instead of default 95%
                    },
                    'disable_default_plugins': True
                }
            }, f, indent=2)

    def _create_agents_md(self):
        """Create AGENTS.md file with system prompt for the agent."""
        system_prompt = self._get_system_prompt()
        agents_md_path = self.agent_workspace / "AGENTS.md"
        with open(agents_md_path, 'w') as f:
            f.write(system_prompt)

    def _get_system_prompt(self) -> str:
        """Generate the system prompt for the agent by loading from shared template."""
        # Load simulator instructions (tool_list filled dynamically from TOOL_DOCS)
        from ...tools import get_tool_summary_table
        simulator_file = Path(__file__).parent.parent / "simulator_instructions.md"
        with open(simulator_file, 'r') as f:
            simulator_instructions = f.read().format(tool_list=get_tool_summary_table())

        # Load from shared template
        template_file = Path(__file__).parent.parent / "agent_template.md"
        with open(template_file, 'r') as f:
            template_content = f.read()

        # Format the template with run-specific values
        return template_content.format(
            total_days=self.config.total_days,
            run_id=self.run_id,
            model=self.config.model,
            initial_cash=self.config.initial_cash,
            agent_workspace=str(self.agent_workspace),
            simulator_instructions=simulator_instructions
        )

    def _create_opencode_config(self):
        """Create opencode.json configuration file."""
        # Parse model to determine provider
        # Format: provider/model (e.g., "openai/gpt-5.2", "xai/grok-4-1-fast-reasoning", "bedrock/claude-opus-4-5")
        model_parts = self.config.model.split("/", 1)
        if len(model_parts) == 2:
            provider_name, model_name = model_parts
        else:
            provider_name = "openai"
            model_name = self.config.model

        # Build provider config based on detected provider
        provider_config = {}

        if provider_name == "xai":
            # xAI/Grok provider - uses OpenAI-compatible API
            provider_config["xai"] = {
                "models": {
                    model_name: {
                        "options": {
                            "reasoningEffort": self.config.reasoning_effort,
                            "textVerbosity": "low",
                            "reasoningSummary": "auto"
                        }
                    }
                },
                "options": {
                    "apiKey": "{env:XAI_API_KEY}",
                    "baseUrl": "https://api.x.ai/v1",
                    "timeout": 600000
                }
            }
        elif provider_name in ("bedrock", "amazon-bedrock"):
            # AWS Bedrock provider - supports Anthropic Claude models
            # Accept both "bedrock/" and "amazon-bedrock/" prefixes for user convenience
            # Map friendly model names to Bedrock model IDs WITH global. prefix
            # Using global. prefix avoids OpenCode's bug of double-prefixing (us.us.)
            # when AWS_REGION is set
            bedrock_model_map = {
                "claude-opus-4-5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
                "claude-opus-4.5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
                "claude-sonnet-4": "global.anthropic.claude-sonnet-4-20250514-v1:0",
                "claude-sonnet-4.5": "global.anthropic.claude-sonnet-4-5-20250423-v1:0",
                "claude-haiku-3.5": "global.anthropic.claude-haiku-3-5-20241022-v1:0",
            }

            # Get the actual Bedrock model ID
            bedrock_model_id = bedrock_model_map.get(model_name, model_name)

            # If user provided a model ID without prefix, add global. prefix
            # to avoid OpenCode's double-prefix bug
            region_prefixes = ["us.", "eu.", "global.", "apac.", "au."]
            has_prefix = any(bedrock_model_id.startswith(p) for p in region_prefixes)
            if not has_prefix and bedrock_model_id.startswith("anthropic."):
                bedrock_model_id = f"global.{bedrock_model_id}"

            # Get AWS region from environment or default
            aws_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))

            # Store the OpenCode model string with global. prefix
            self._opencode_model = f"amazon-bedrock/{bedrock_model_id}"

            provider_config["amazon-bedrock"] = {
                "models": {
                    bedrock_model_id: {
                        "options": {}
                    }
                },
                "options": {
                    "region": aws_region,
                    "timeout": 600000
                }
            }

            # If AWS_PROFILE is set, use it; otherwise rely on default credential chain
            if os.environ.get("AWS_PROFILE"):
                provider_config["amazon-bedrock"]["options"]["profile"] = os.environ.get("AWS_PROFILE")
        else:
            # Default to OpenAI provider
            provider_config["openai"] = {
                "models": {
                    model_name: {
                        "options": {
                            "reasoningEffort": self.config.reasoning_effort,
                            "textVerbosity": "low",
                            "reasoningSummary": "auto"
                        }
                    }
                },
                "options": {
                    "apiKey": "{env:OPENAI_API_KEY}",
                    "timeout": 600000
                }
            }

        # OpenCode config
        # For Bedrock, use the full model ID; otherwise use config model
        effective_model = getattr(self, '_opencode_model', self.config.model)
        opencode_config = {
            "$schema": "https://opencode.ai/config.json",
            "model": effective_model,
            "provider": provider_config,
            "default_agent": "build",
            "autoupdate": False,
            "share": "disabled",
            # Context compaction settings - compact earlier to preserve game state
            "compaction": {
                "enabled": True,
                "threshold": 0.85  # Compact at 85% instead of default 95%
            }
        }

        config_path = self.agent_workspace / "opencode.json"
        with open(config_path, 'w') as f:
            json.dump(opencode_config, f, indent=2)

    def _create_mcp_config(self):
        """Create MCP server configuration in opencode.json."""
        # Get the src directory path for PYTHONPATH
        src_dir = Path(__file__).parent.parent.parent.parent.resolve()

        # MCP server script path (reuse Claude's serve_mcp.py)
        mcp_server_path = Path(__file__).parent.parent / "claude_code" / "serve_mcp.py"

        # Read existing config
        config_path = self.agent_workspace / "opencode.json"
        with open(config_path, 'r') as f:
            opencode_config = json.load(f)

        # Add MCP server configuration (OpenCode format)
        # OpenCode requires: type="local", command as array, environment instead of env
        opencode_config["mcp"] = {
            "saas-bench": {
                "type": "local",
                "command": [sys.executable, str(mcp_server_path)],
                "enabled": True,
                "environment": {
                    "SAAS_BENCH_WORKSPACE": str(self.workspace_dir),
                    "SAAS_BENCH_RUN_ID": self.run_id,
                    "SAAS_BENCH_DB_PATH": str(self.db_path),
                    "SAAS_BENCH_TOTAL_DAYS": str(self.config.total_days),
                    "SAAS_BENCH_SEED": str(self.config.seed),
                    "SAAS_BENCH_SCENARIO": self.config.scenario,
                    "PYTHONPATH": str(src_dir)
                }
            }
        }

        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(opencode_config, f, indent=2)

    def _build_daily_dashboard(self, day: int, last_result: Optional[DayResult] = None) -> str:
        """Build the daily dashboard. Delegates to the shared build_daily_dashboard()."""
        inbox = self.shock_manager.get_inbox_items(day)
        return build_daily_dashboard(self.conn, day, last_result, inbox_items=inbox)

    def _fix_jose_dependency(self):
        """Fix the jose module corruption issue in OpenCode's cache.

        OpenCode's Bun-based plugin system corrupts the jose module.
        We fix it by reinstalling with npm before each call.
        See: https://github.com/anomalyco/opencode/issues/7249
        """
        cache_dir = Path.home() / ".cache" / "opencode"
        if cache_dir.exists():
            try:
                # Remove corrupted modules and reinstall with npm
                subprocess.run(
                    ["npm", "install", "jose@5.9.6", "--save"],
                    cwd=str(cache_dir),
                    capture_output=True,
                    timeout=30
                )
            except Exception:
                pass  # Best effort - continue even if fix fails

    def _run_opencode_headless(self, prompt: str) -> Dict[str, Any]:
        """Run OpenCode in headless mode.

        Args:
            prompt: The prompt to send to OpenCode

        Returns:
            Dict with response and any tool calls
        """
        # Fix jose dependency before each call (workaround for OpenCode bug)
        self._fix_jose_dependency()

        # Build command
        # opencode run <prompt> --model <model> --variant <reasoning_effort> --format json
        # For Bedrock, use the full model ID (amazon-bedrock/model-id)
        effective_model = getattr(self, '_opencode_model', self.config.model)
        cmd = [
            "opencode",
            "run",
            prompt,
            "--model", effective_model,
            "--variant", self.config.reasoning_effort,  # Reasoning effort level
            "--format", "json",  # Get JSON output with tool calls
        ]

        # Set working directory to agent workspace (where opencode.json is)
        env = os.environ.copy()
        env["PWD"] = str(self.agent_workspace)

        # Disable OpenCode's default plugins to avoid jose/bun dependency issues
        env["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"

        # Ensure API keys are set from .env if not already in environment
        env_file = Path(__file__).parent.parent.parent.parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in env:
                        env["OPENAI_API_KEY"] = line.split("=", 1)[1]
                    elif line.startswith("XAI_API_KEY=") and "XAI_API_KEY" not in env:
                        env["XAI_API_KEY"] = line.split("=", 1)[1]
                    # AWS Bedrock credentials
                    elif line.startswith("AWS_ACCESS_KEY_ID=") and "AWS_ACCESS_KEY_ID" not in env:
                        env["AWS_ACCESS_KEY_ID"] = line.split("=", 1)[1]
                    elif line.startswith("AWS_SECRET_ACCESS_KEY=") and "AWS_SECRET_ACCESS_KEY" not in env:
                        env["AWS_SECRET_ACCESS_KEY"] = line.split("=", 1)[1]
                    elif line.startswith("AWS_REGION=") and "AWS_REGION" not in env:
                        env["AWS_REGION"] = line.split("=", 1)[1]
                    elif line.startswith("AWS_DEFAULT_REGION=") and "AWS_DEFAULT_REGION" not in env:
                        env["AWS_DEFAULT_REGION"] = line.split("=", 1)[1]
                    elif line.startswith("AWS_PROFILE=") and "AWS_PROFILE" not in env:
                        env["AWS_PROFILE"] = line.split("=", 1)[1]

        print(f"  [DEBUG] Running OpenCode: {' '.join(cmd[:5])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.agent_workspace),
                env=env
                # No timeout - let it run as long as needed
            )

            # Log raw output size for debugging
            stdout_size = len(result.stdout) if result.stdout else 0
            stderr_size = len(result.stderr) if result.stderr else 0
            print(f"  [DEBUG] OpenCode output: stdout={stdout_size} bytes, stderr={stderr_size} bytes, exit={result.returncode}")

            if result.returncode != 0:
                print(f"  [DEBUG] OpenCode stderr: {result.stderr[:1000] if result.stderr else 'empty'}")
                print(f"  [DEBUG] OpenCode stdout: {result.stdout[:1000] if result.stdout else 'empty'}")
                return {
                    "error": f"OpenCode exited with code {result.returncode}",
                    "stderr": result.stderr,
                    "raw_output": result.stdout
                }

            # Parse output - OpenCode outputs JSONL with events
            output = {"raw_output": result.stdout, "tool_calls": []}

            # Parse all JSONL lines for tool calls and final result
            lines = result.stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)

                    # Capture tool calls from event stream
                    if parsed.get("type") == "tool_call" or parsed.get("event") == "tool_start":
                        tool_data = parsed.get("data", parsed)
                        output["tool_calls"].append({
                            "name": tool_data.get("name", tool_data.get("tool")),
                            "arguments": tool_data.get("arguments", tool_data.get("input", tool_data.get("args"))),
                            "id": tool_data.get("id")
                        })

                    # Capture tool results
                    elif parsed.get("type") == "tool_result" or parsed.get("event") == "tool_end":
                        tool_data = parsed.get("data", parsed)
                        tool_id = tool_data.get("tool_use_id", tool_data.get("id"))
                        for tc in output["tool_calls"]:
                            if tc.get("id") == tool_id:
                                tc["result"] = tool_data.get("content", tool_data.get("output"))
                                break

                    # Capture final result
                    elif 'result' in parsed or 'message' in parsed:
                        output["result"] = parsed.get("result", parsed.get("message"))

                except json.JSONDecodeError:
                    # Not JSON, might be plain text output
                    continue

            # Log tool call summary
            if output["tool_calls"]:
                print(f"  [DEBUG] Captured {len(output['tool_calls'])} tool calls")

            return output

        except Exception as e:
            import traceback
            print(f"  [DEBUG] Exception: {traceback.format_exc()}")
            return {"error": f"Failed to run OpenCode: {str(e)}"}

    def _log_rationale(self, rationale: str, context: Optional[str] = None):
        """Log a rationale entry."""
        entry = RationaleEntry(
            timestamp=self._now(),
            day=self.current_day,
            rationale=rationale,
            context=context
        )
        self.rationales.append(entry)

        # Also log to event logger using proper EventLogEntry
        if self.event_logger:
            from ...event_logger import EventLogEntry
            self.event_logger._write_event(EventLogEntry(
                timestamp=entry.timestamp,
                day=entry.day,
                event_type="agent_rationale",
                category=context or "general",
                details={"rationale": rationale}
            ))

    def _save_rationales(self):
        """Save all rationales to a JSON file."""
        rationales_file = self.logs_dir / f"rationales_{self.run_id}.json"
        with open(rationales_file, 'w') as f:
            json.dump([asdict(r) for r in self.rationales], f, indent=2)

    def _log_agent_turn(self, prompt: str, response: Dict[str, Any], tool_results: Optional[List[Dict]] = None):
        """Log an agent turn to the incremental JSONL file.

        Each line is a complete JSON object representing one agent turn.
        This allows easy streaming and merging with env logs.
        """
        self.agent_turn_count += 1

        # Extract tool calls from response for detailed logging
        tool_calls = self._extract_tool_calls(response)

        turn_entry = {
            "turn": self.agent_turn_count,
            "timestamp": self._now(),
            "day": self.current_day,
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": {
                "result": response.get("result", response.get("message", "")),
                "error": response.get("error"),
                "raw_output": response.get("raw_output", "")[:2000] if response.get("raw_output") else None,
            },
            "tool_calls": tool_calls,  # All tool calls with args and results
            "tool_results": tool_results or []
        }

        # Append to JSONL file (one JSON object per line)
        with open(self.agent_log_file, 'a') as f:
            f.write(json.dumps(turn_entry) + "\n")

        # Also print tool calls for live monitoring
        if tool_calls:
            for tc in tool_calls:
                print(f"    🔧 Tool: {tc.get('name', 'unknown')}")
                if tc.get('arguments'):
                    args_str = json.dumps(tc['arguments'])[:200]
                    print(f"       Args: {args_str}")
                if tc.get('result'):
                    result_str = str(tc['result'])[:200]
                    print(f"       Result: {result_str}")

    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all tool calls from OpenCode response.

        OpenCode outputs tool calls in various formats depending on the output mode.
        This method handles all known formats.
        """
        tool_calls = []

        # Check for explicit tool_calls field
        if "tool_calls" in response:
            tool_calls.extend(response["tool_calls"])

        # Parse raw_output for tool call patterns
        raw_output = response.get("raw_output", "")
        if raw_output:
            # Try to parse JSONL lines for tool events
            for line in raw_output.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    # OpenCode tool call format
                    if parsed.get("type") == "tool_call":
                        tool_calls.append({
                            "name": parsed.get("name", parsed.get("tool")),
                            "arguments": parsed.get("arguments", parsed.get("args", parsed.get("input"))),
                            "result": parsed.get("result", parsed.get("output"))
                        })
                    # Alternative format with tool_use
                    elif parsed.get("type") == "tool_use":
                        tool_calls.append({
                            "name": parsed.get("name"),
                            "arguments": parsed.get("input"),
                            "result": None
                        })
                    # Tool result format
                    elif parsed.get("type") == "tool_result":
                        # Try to match with previous tool call
                        tool_id = parsed.get("tool_use_id")
                        for tc in tool_calls:
                            if tc.get("id") == tool_id:
                                tc["result"] = parsed.get("content")
                                break
                    # OpenCode event stream format
                    elif "event" in parsed and parsed["event"] in ["tool_start", "tool_end"]:
                        if parsed["event"] == "tool_start":
                            tool_calls.append({
                                "name": parsed.get("data", {}).get("tool"),
                                "arguments": parsed.get("data", {}).get("input"),
                                "result": None,
                                "id": parsed.get("data", {}).get("id")
                            })
                        elif parsed["event"] == "tool_end":
                            tool_id = parsed.get("data", {}).get("id")
                            for tc in tool_calls:
                                if tc.get("id") == tool_id:
                                    tc["result"] = parsed.get("data", {}).get("output")
                                    break
                except json.JSONDecodeError:
                    # Not JSON, try regex patterns
                    pass

            # Also look for function call patterns in text
            import re
            # Pattern: [Tool: name] or Tool: name or function_name(args)
            tool_pattern = r'\[Tool:\s*(\w+)\]|\bTool:\s*(\w+)|(\w+)\(([^)]*)\)'
            matches = re.findall(tool_pattern, raw_output)
            # Known false positives to exclude (code keywords, SQL functions, common functions)
            exclude_names = {
                # Python keywords
                'if', 'for', 'while', 'print', 'return', 'def', 'class', 'import', 'from',
                'try', 'except', 'finally', 'with', 'as', 'raise', 'assert', 'yield',
                'lambda', 'pass', 'break', 'continue', 'global', 'nonlocal', 'del',
                # SQL functions
                'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
                'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ROUND', 'COALESCE', 'CAST',
                'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT', 'AS',
                # Common Python functions
                'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                'open', 'read', 'write', 'close', 'append', 'extend', 'pop', 'remove',
                'get', 'keys', 'values', 'items', 'update', 'format', 'join', 'split',
                'strip', 'replace', 'lower', 'upper', 'startswith', 'endswith',
                # Common library functions
                'execute', 'fetchone', 'fetchall', 'commit', 'rollback', 'cursor',
                'connect', 'read_sql', 'to_sql', 'json', 'loads', 'dumps',
                # Likely output fragments (prefixed with 'n' from newline)
                'nshow', 'nprint', 'nyesterday_started', 'ntoday', 'nresult',
            }
            for match in matches:
                tool_name = match[0] or match[1] or match[2]
                if tool_name and tool_name not in exclude_names and tool_name.lower() not in exclude_names:
                    # Avoid false positives from code
                    existing_names = [tc.get('name') for tc in tool_calls]
                    if tool_name not in existing_names:
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": match[3] if match[3] else None,
                            "result": None,
                            "source": "text_pattern"
                        })

        return tool_calls

    def run(self, verbose: bool = True) -> RunResult:
        """Run the full simulation with OpenCode as the agent.

        Args:
            verbose: Whether to print progress

        Returns:
            RunResult with final outcome and metrics
        """
        self.setup()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting OpenCode Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.config.model}")
            print(f"Reasoning Effort: {self.config.reasoning_effort}")
            print(f"Scenario: {self.config.scenario}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        # State file for MCP server communication
        state_file = self.workspace_dir / ".mcp_state.json"

        last_result: Optional[DayResult] = None

        for day in range(1, self.config.total_days + 1):
            self.current_day = day
            self.tools.set_current_day(day)
            self.event_logger.set_day(day)

            # Snapshot workspace files at start of day
            workspace_snapshot_before = self._snapshot_workspace_files()

            # Update state file
            with open(state_file, 'w') as f:
                json.dump({
                    'current_day': day,
                    'day_ended': False,
                    'last_updated': self._now()
                }, f)

            if verbose:
                print(f"\n{'='*40}")
                print(f"DAY {day}")
                print(f"{'='*40}")

            # Check for shocks
            new_shocks = self.shock_manager.check_and_generate_shocks(day)
            for shock in new_shocks:
                self.event_logger.log_shock(shock.shock_type, shock.details)
                if verbose:
                    print(f"  ⚡ Shock: {shock.shock_type}")

            # Build dashboard
            dashboard = self._build_daily_dashboard(day, last_result)

            if verbose:
                print(dashboard[:500])  # Print truncated dashboard

            # Run agent turn
            prompt = f"Day {day} has started.\n\n{dashboard}\n\nReview the situation and take actions. Call next_day when you're done with today's decisions."

            # Multi-turn loop for the day
            day_ended = False
            turns = 0

            while not day_ended and turns < self.config.max_turns_per_day:
                turns += 1

                response = self._run_opencode_headless(prompt)

                # Log agent turn (incremental)
                self._log_agent_turn(prompt, response)

                if "error" in response:
                    if verbose:
                        print(f"  ❌ Error: {response['error']}")
                    break

                # Check for next_day tool call or game completion signals
                result_text = str(response.get("result", response.get("raw_output", "")))
                if "NEXT_DAY_SIGNAL" in result_text or "SIMULATION COMPLETE" in result_text or "GAME OVER" in result_text:
                    day_ended = True
                    if verbose:
                        print(f"  → Day ended (turns: {turns})")

                # Extract any rationales from response
                self._extract_rationales(response)

                # Update prompt for continuation
                prompt = "Continue with your actions for today, or call next_day when done."

            # Check if step_day was already called by the MCP next_day tool
            # by checking if the day record already exists in service_day
            day_already_stepped = self.conn.execute(
                "SELECT 1 FROM service_day WHERE day = ?", (day,)
            ).fetchone() is not None

            # Run simulation step only if not already done via MCP
            if day_already_stepped:
                # MCP next_day tool already ran step_day, just fetch the result
                # by getting data from the existing day record
                if verbose:
                    print(f"  (step_day already run via MCP)")
                # Get the last result from database state (get_cash, get_mrr imported at top)
                cash = get_cash(self.conn)
                mrr = get_mrr(self.conn)
                # Create a minimal DayResult for logging
                last_result = DayResult(
                    day=day,
                    cash=cash,
                    mrr=mrr,
                    total_usage=0,  # Not critical for logging
                    overload=False,
                    outage=False,
                    downtime_minutes=0,
                    new_subscribers=0,
                    cancellations=0,
                    upgrades=0,
                    downgrades=0
                )
            else:
                last_result = self.simulator.step_day()

            # Log daily state
            self.event_logger.log_daily_state(
                cash=last_result.cash,
                mrr=last_result.mrr,
                subscribers=get_active_subscriber_count(self.conn),
                usage=last_result.total_usage,
                overload=last_result.overload,
                outage=last_result.outage,
                group_reputations=self._get_group_reputations(),
                group_awareness=self._get_group_awareness()
            )

            # Log outage if occurred
            if last_result.outage:
                self.event_logger.log_outage(
                    last_result.downtime_minutes,
                    last_result.overload
                )

            # Save incrementally
            self.event_logger.save_incremental()

            # Compute and log file diffs for this day
            workspace_snapshot_after = self._snapshot_workspace_files()
            file_diffs = self._compute_file_diffs(workspace_snapshot_before, workspace_snapshot_after)
            if file_diffs:
                self._log_file_diffs(day, file_diffs)
                if verbose:
                    print(f"  📁 File changes: {len(file_diffs)} files modified")

            if verbose:
                print(f"  📊 End of day: Cash=${last_result.cash:,.0f}")

            # Check for bankruptcy
            if self.simulator.shutdown_mode:
                self.game_ended = True
                self.game_outcome = 'bankrupt'
                if verbose:
                    print(f"\n💀 BANKRUPT at day {day}! Cash: ${last_result.cash:,.0f}")
                break

        # Determine final outcome
        if not self.game_outcome:
            self.game_outcome = 'completed'

        # Finalize logging
        final_cash = last_result.cash if last_result else 0.0
        self.event_logger.log_run_end(final_cash, self.current_day, self.game_outcome)
        self.event_logger.save()
        self._save_rationales()

        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Final Cash: ${final_cash:,.0f}")
            print(f"Days Run: {self.current_day}")
            print(f"Outcome: {self.game_outcome}")
            print(f"Rationales logged: {len(self.rationales)}")
            print(f"{'='*60}\n")

        return RunResult(
            run_id=self.run_id,
            seed=self.config.seed,
            scenario=self.config.scenario,
            final_cash=final_cash,
            days_run=self.current_day,
            outcome=self.game_outcome,
            rationales=[asdict(r) for r in self.rationales],
            log_file=str(self.event_logger.log_file),
            workspace_dir=str(self.workspace_dir)
        )

    def _extract_rationales(self, response: Dict[str, Any]) -> None:
        """Extract any rationale tool calls from the response."""
        # Check for tool_calls in various formats
        tool_calls = response.get("tool_calls", [])

        for call in tool_calls:
            if call.get("name") == "log_rationale":
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                self._log_rationale(
                    args.get("rationale", ""),
                    args.get("context")
                )

    def _get_group_reputations(self) -> Dict[str, float]:
        """Get all group reputations."""
        from ...database import get_all_group_reputations
        return get_all_group_reputations(self.conn)

    def _get_group_awareness(self) -> Dict[str, float]:
        """Get all group awareness levels."""
        from ...database import get_all_group_awareness
        return get_all_group_awareness(self.conn)

    @classmethod
    def from_config_file(cls, config_path: Path, workspace_base: Optional[Path] = None) -> 'OpenCodeRunner':
        """Create a runner from a configuration file.

        Args:
            config_path: Path to JSON config file
            workspace_base: Base directory for workspaces

        Returns:
            OpenCodeRunner instance
        """
        with open(config_path) as f:
            config_dict = json.load(f)

        config = AgentConfig(
            model=config_dict.get('model', 'openai/gpt-5.2'),
            reasoning_effort=config_dict.get('reasoning_effort', 'medium'),
            seed=config_dict.get('seed', 42),
            scenario=config_dict.get('scenario', 'default'),
            total_days=config_dict.get('total_days', 3650),
            initial_cash=config_dict.get('initial_cash', 1_000_000.0),
            budget_limit_usd=config_dict.get('budget_limit_usd', 50.0),
            max_turns_per_day=config_dict.get('max_turns_per_day', 50),
        )

        return cls(config, workspace_base)


def run_opencode_agent(
    config_path: Optional[Path] = None,
    model: str = "openai/gpt-5.2",
    reasoning_effort: str = "medium",
    seed: int = 42,
    scenario: str = "default",
    workspace_base: Optional[Path] = None,
    verbose: bool = True
) -> RunResult:
    """Convenience function to run the OpenCode agent.

    Args:
        config_path: Optional path to config file (overrides other args if provided)
        model: OpenCode model to use (format: provider/model)
        reasoning_effort: Reasoning effort level (none, minimal, low, medium, high, xhigh)
        seed: Random seed
        scenario: Scenario name
        workspace_base: Base directory for workspaces
        verbose: Print progress

    Returns:
        RunResult
    """
    if config_path:
        runner = OpenCodeRunner.from_config_file(config_path, workspace_base)
    else:
        config = AgentConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            seed=seed,
            scenario=scenario
        )
        runner = OpenCodeRunner(config, workspace_base)

    return runner.run(verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenCode agent for SaaS Bench")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--model", default="openai/gpt-5.2", help="OpenCode model (provider/model)")
    parser.add_argument("--reasoning-effort", default="medium",
                       choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                       help="Reasoning effort level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--workspace", type=Path, help="Workspace base directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    result = run_opencode_agent(
        config_path=args.config,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        scenario=args.scenario,
        workspace_base=args.workspace,
        verbose=not args.quiet
    )

    print(f"\nResult: {result.outcome}")
    print(f"Final Cash: ${result.final_cash:,.0f}")
    print(f"Workspace: {result.workspace_dir}")
