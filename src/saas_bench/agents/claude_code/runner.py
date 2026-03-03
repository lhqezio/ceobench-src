"""Claude Code Runner for SaaS Bench.

This module manages the execution of Claude Code as the agent for SaaS Bench,
handling session management, workspace isolation, and simulation interaction.
"""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from numpy.random import Generator, PCG64

from ...config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from ...database import init_database, get_cash, get_active_subscriber_count, get_config
from ...environment import build_daily_dashboard
from ...simulation import Simulator, DayResult
from ...tools import AgentTools
from ...shocks import ShockManager
from ...event_logger import EventLogger
from .mcp_server import AgentState, RationaleEntry


@dataclass
class AgentConfig:
    """Configuration for the Claude Code agent."""
    model: str = "claude-sonnet-4-20250514"  # Claude model to use
    seed: int = 42
    scenario: str = "default"
    total_days: int = 3650
    initial_cash: float = 1_000_000.0
    budget_limit_usd: float = 50.0  # API budget limit
    max_turns_per_day: int = 50  # Max tool calls per day
    web_access: bool = True  # Enable web access for agent


@dataclass
class RunResult:
    """Result from running the Claude Code agent."""
    run_id: str
    session_id: Optional[str]  # Claude Code session ID for resumption
    seed: int
    scenario: str
    final_cash: float
    days_run: int
    outcome: str  # 'completed', 'bankrupt', 'budget_exceeded', 'interrupted'
    rationales: List[Dict[str, Any]] = field(default_factory=list)
    log_file: Optional[str] = None
    workspace_dir: Optional[str] = None


class ClaudeCodeRunner:
    """Manages Claude Code agent execution for SaaS Bench."""

    def __init__(self, config: AgentConfig, workspace_base: Optional[Path] = None):
        """Initialize the runner.

        Args:
            config: Agent configuration
            workspace_base: Base directory for workspaces (each run gets a subdirectory)
        """
        self.config = config
        self.workspace_base = (workspace_base or Path('./claude_code_runs')).resolve()

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

        # Database path
        self.db_path = self.workspace_dir / "world.db"

        # Session tracking
        self.session_id: Optional[str] = None
        self.session_file = self.workspace_dir / ".session_id"

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
        skip_dirs = {'.claude', 'node_modules', '__pycache__', '.git', '.venv'}
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
                'seed': self.config.seed,
                'scenario': self.config.scenario,
                'total_days': self.config.total_days,
                'initial_cash': self.config.initial_cash,
            }
        )

        # Connect event logger to components
        self.simulator.set_event_logger(self.event_logger)
        self.tools.set_event_logger(self.event_logger)

        # Initialize simulation
        self.simulator.initialize()
        self.event_logger.log_run_start()

        # Create CLAUDE.md with system prompt
        self._create_claude_md()

        # Save initial config
        self._save_config()

    def _save_config(self):
        """Save configuration to workspace."""
        config_file = self.workspace_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'model': self.config.model,
                'seed': self.config.seed,
                'scenario': self.config.scenario,
                'total_days': self.config.total_days,
                'initial_cash': self.config.initial_cash,
                'created_at': self._now()
            }, f, indent=2)

    def _create_claude_md(self):
        """Create CLAUDE.md file with system prompt for the agent."""
        system_prompt = self._get_system_prompt()
        claude_md_path = self.agent_workspace / "CLAUDE.md"
        with open(claude_md_path, 'w') as f:
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

    def _build_daily_dashboard(self, day: int, last_result: Optional[DayResult] = None) -> str:
        """Build the daily dashboard. Delegates to the shared build_daily_dashboard()."""
        inbox = self.shock_manager.get_inbox_items(day)
        return build_daily_dashboard(self.conn, day, last_result, inbox_items=inbox)

    def _create_mcp_config(self) -> Path:
        """Create MCP server configuration file."""
        # Get the src directory path for PYTHONPATH
        src_dir = Path(__file__).parent.parent.parent.parent.resolve()

        mcp_config = {
            "mcpServers": {
                "saas-bench": {
                    "command": "python",
                    "args": [
                        "-m", "saas_bench.agents.claude_code.mcp_server"
                    ],
                    "env": {
                        "SAAS_BENCH_WORKSPACE": str(self.workspace_dir),
                        "SAAS_BENCH_RUN_ID": self.run_id,
                        "SAAS_BENCH_DB_PATH": str(self.db_path),
                        "PYTHONPATH": str(src_dir)
                    }
                }
            }
        }

        config_path = self.workspace_dir / "mcp_config.json"
        with open(config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        return config_path

    def _run_claude_headless(self, prompt: str, resume_session: bool = False) -> Dict[str, Any]:
        """Run Claude Code in headless mode.

        Args:
            prompt: The prompt to send to Claude
            resume_session: Whether to resume the previous session

        Returns:
            Dict with response, session_id, and any tool calls
        """
        # Build command
        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        # Add model specification
        cmd.extend(["--model", self.config.model])

        # Add MCP config
        mcp_config_path = self._create_mcp_config()
        cmd.extend(["--mcp-config", str(mcp_config_path)])

        # Auto-approve all tools
        cmd.extend(["--allowedTools", "*"])

        # Resume session if requested
        if resume_session and self.session_id:
            cmd.extend(["--resume", self.session_id])
        elif resume_session and self.session_file.exists():
            self.session_id = self.session_file.read_text().strip()
            cmd.extend(["--resume", self.session_id])

        # CLAUDE.md is auto-loaded from the cwd (agent_workspace)
        # No need to pass it via --append-system-prompt

        # Note: web_access is controlled via WebSearch/WebFetch tools being available
        # No special flag needed - removing invalid --permission flag

        # Set working directory to agent workspace
        env = os.environ.copy()
        env["PWD"] = str(self.agent_workspace)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.agent_workspace),
                env=env
                # No timeout - let it run as long as needed
            )

            if result.returncode != 0:
                print(f"  [DEBUG] Claude Code stderr: {result.stderr[:500] if result.stderr else 'empty'}")
                print(f"  [DEBUG] Claude Code stdout: {result.stdout[:500] if result.stdout else 'empty'}")
                return {
                    "error": f"Claude Code exited with code {result.returncode}",
                    "stderr": result.stderr
                }

            # Parse JSON output
            output = json.loads(result.stdout)

            # Save session ID for resumption
            if "session_id" in output:
                self.session_id = output["session_id"]
                self.session_file.write_text(self.session_id)

            return output

        except json.JSONDecodeError:
            return {"error": "Failed to parse Claude Code output", "raw": result.stdout}

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

        # Claude Code JSON output uses 'result' not 'output'
        turn_entry = {
            "turn": self.agent_turn_count,
            "timestamp": self._now(),
            "day": self.current_day,
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": {
                "result": response.get("result", ""),
                "num_turns": response.get("num_turns", 0),
                "total_cost_usd": response.get("total_cost_usd", 0),
                "session_id": response.get("session_id", ""),
                "error": response.get("error"),
            },
            "tool_results": tool_results or []
        }

        # Append to JSONL file (one JSON object per line)
        with open(self.agent_log_file, 'a') as f:
            f.write(json.dumps(turn_entry) + "\n")

    def run(self, verbose: bool = True) -> RunResult:
        """Run the full simulation with Claude Code as the agent.

        Args:
            verbose: Whether to print progress

        Returns:
            RunResult with final outcome and metrics
        """
        self.setup()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Claude Code Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.config.model}")
            print(f"Scenario: {self.config.scenario}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        last_result: Optional[DayResult] = None

        for day in range(1, self.config.total_days + 1):
            self.current_day = day
            self.tools.set_current_day(day)
            self.event_logger.set_day(day)

            # Snapshot workspace files at start of day
            workspace_snapshot_before = self._snapshot_workspace_files()

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

                response = self._run_claude_headless(prompt, resume_session=(turns > 1))

                # Log agent turn (incremental)
                self._log_agent_turn(prompt, response)

                if "error" in response:
                    if verbose:
                        print(f"  ❌ Error: {response['error']}")
                    break

                # Check for next_day tool call
                if self._check_for_next_day(response):
                    day_ended = True
                    if verbose:
                        print(f"  → Agent called next_day (turns: {turns})")

                # Extract any rationales from response
                self._extract_rationales(response)

                # Update prompt for continuation
                prompt = "Continue with your actions for today, or call next_day when done."

            # Run simulation step
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
            session_id=self.session_id,
            seed=self.config.seed,
            scenario=self.config.scenario,
            final_cash=final_cash,
            days_run=self.current_day,
            outcome=self.game_outcome,
            rationales=[asdict(r) for r in self.rationales],
            log_file=str(self.event_logger.log_file),
            workspace_dir=str(self.workspace_dir)
        )

    def _check_for_next_day(self, response: Dict[str, Any]) -> bool:
        """Check if the response contains a next_day tool call."""
        # This depends on Claude Code's output format
        # Look for tool calls in the response
        if "tool_calls" in response:
            for call in response["tool_calls"]:
                if call.get("name") == "next_day":
                    return True
        if "output" in response:
            # Check text output for signals
            if "NEXT_DAY_SIGNAL" in str(response["output"]):
                return True
        return False

    def _extract_rationales(self, response: Dict[str, Any]) -> None:
        """Extract any rationale tool calls from the response."""
        if "tool_calls" in response:
            for call in response["tool_calls"]:
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
    def from_config_file(cls, config_path: Path, workspace_base: Optional[Path] = None) -> 'ClaudeCodeRunner':
        """Create a runner from a configuration file.

        Args:
            config_path: Path to JSON config file
            workspace_base: Base directory for workspaces

        Returns:
            ClaudeCodeRunner instance
        """
        with open(config_path) as f:
            config_dict = json.load(f)

        config = AgentConfig(
            model=config_dict.get('model', 'claude-sonnet-4-20250514'),
            seed=config_dict.get('seed', 42),
            scenario=config_dict.get('scenario', 'default'),
            total_days=config_dict.get('total_days', 3650),
            initial_cash=config_dict.get('initial_cash', 1_000_000.0),
            budget_limit_usd=config_dict.get('budget_limit_usd', 50.0),
            max_turns_per_day=config_dict.get('max_turns_per_day', 50),
            web_access=config_dict.get('web_access', True)
        )

        return cls(config, workspace_base)

    @classmethod
    def resume(cls, workspace_dir: Path) -> 'ClaudeCodeRunner':
        """Resume a previous run from its workspace directory.

        Args:
            workspace_dir: Path to the existing workspace

        Returns:
            ClaudeCodeRunner instance configured for resumption
        """
        config_file = workspace_dir / "config.json"
        if not config_file.exists():
            raise ValueError(f"No config.json found in {workspace_dir}")

        with open(config_file) as f:
            saved = json.load(f)

        config = AgentConfig(
            model=saved.get('model', 'claude-sonnet-4-20250514'),
            seed=saved['seed'],
            scenario=saved['scenario'],
            total_days=saved.get('total_days', 3650),
            initial_cash=saved.get('initial_cash', 1_000_000.0),
        )

        runner = cls(config, workspace_dir.parent)
        runner.run_id = saved['run_id']
        runner.workspace_dir = workspace_dir
        runner.agent_workspace = workspace_dir / "agent"
        runner.logs_dir = workspace_dir / "logs"
        runner.db_path = workspace_dir / "world.db"
        runner.session_file = workspace_dir / ".session_id"

        # Load session ID if exists
        if runner.session_file.exists():
            runner.session_id = runner.session_file.read_text().strip()

        return runner


def run_claude_code_agent(
    config_path: Optional[Path] = None,
    model: str = "claude-sonnet-4-20250514",
    seed: int = 42,
    scenario: str = "default",
    workspace_base: Optional[Path] = None,
    verbose: bool = True
) -> RunResult:
    """Convenience function to run the Claude Code agent.

    Args:
        config_path: Optional path to config file (overrides other args if provided)
        model: Claude model to use
        seed: Random seed
        scenario: Scenario name
        workspace_base: Base directory for workspaces
        verbose: Print progress

    Returns:
        RunResult
    """
    if config_path:
        runner = ClaudeCodeRunner.from_config_file(config_path, workspace_base)
    else:
        config = AgentConfig(
            model=model,
            seed=seed,
            scenario=scenario
        )
        runner = ClaudeCodeRunner(config, workspace_base)

    return runner.run(verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Claude Code agent for SaaS Bench")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--workspace", type=Path, help="Workspace base directory")
    parser.add_argument("--resume", type=Path, help="Resume from workspace directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.resume:
        runner = ClaudeCodeRunner.resume(args.resume)
        result = runner.run(verbose=not args.quiet)
    else:
        result = run_claude_code_agent(
            config_path=args.config,
            model=args.model,
            seed=args.seed,
            scenario=args.scenario,
            workspace_base=args.workspace,
            verbose=not args.quiet
        )

    print(f"\nResult: {result.outcome}")
    print(f"Final Cash: ${result.final_cash:,.0f}")
    print(f"Workspace: {result.workspace_dir}")
