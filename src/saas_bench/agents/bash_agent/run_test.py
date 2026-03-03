#!/usr/bin/env python3
"""Test runner for Bash Agent with SaaS Bench.

This script runs a simulation using the bash_agent with any supported LLM provider.
The agent uses bash/file tools and interacts with the simulator via
novamind_api (Python library) and novamind-operation (CLI).

Supports OpenAI, xAI/Grok, Anthropic (direct and Bedrock).
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from numpy.random import Generator, PCG64

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from openai import OpenAI

try:
    import anthropic
    from anthropic import AnthropicBedrock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from saas_bench.config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from saas_bench.database import init_database, get_cash, get_active_subscriber_count, get_all_group_reputations, get_all_group_awareness
import sqlite3 as _sqlite3
from saas_bench.simulation import Simulator
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.tools import AgentTools
from saas_bench.shocks import ShockManager
from saas_bench.event_logger import EventLogger
from saas_bench.environment import Action, build_daily_dashboard, get_thread_inbox_items
from saas_bench.api_server import NovaMindAPIServer
from saas_bench.docs_generator import initialize_workspace


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_env_file(env_path: Path) -> Dict[str, str]:
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars


class BashAgentRunner:
    """Runner for bash_agent with SaaS Bench."""

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: int = 42,
        scenario: str = "default",
        total_days: int = 3650,
        initial_cash: float = 1_000_000.0,
        workspace_base: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
        continue_from: Optional[Path] = None,
    ):
        self.model = model
        self.provider = provider
        self.seed = seed
        self.scenario = scenario
        self.total_days = total_days
        self.initial_cash = initial_cash
        self.reasoning_effort = reasoning_effort
        self.continue_from = continue_from

        if continue_from:
            self.workspace_dir = Path(continue_from).resolve()
            if not self.workspace_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {self.workspace_dir}")
            config_file = self.workspace_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    old_config = json.load(f)
                self.run_id = old_config['run_id']
            else:
                self.run_id = self.workspace_dir.name.replace('run_', '')
            self.workspace_base = self.workspace_dir.parent
        else:
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or Path('./bash_agent_runs')).resolve()
            self.workspace_dir = self.workspace_base / f"run_{self.run_id}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Agent working directory (inside the run directory)
        self.agent_workspace = self.workspace_dir / "agent_workspace"

        # Logs directory
        self.logs_dir = self.workspace_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Database path
        self.db_path = self.workspace_dir / "world.db"

        # Log file for raw responses
        self.response_log_file = self.logs_dir / f"raw_responses_{self.run_id}.jsonl"

        # Timing log — fine-grained per-turn and per-day timing data
        self.timing_log_file = self.logs_dir / f"timing_{self.run_id}.jsonl"

        # CEOBench dashboard URL for live timing push (set via env var)
        self._dashboard_url = os.environ.get("CEOBENCH_DASHBOARD_URL", "")
        self._timing_queue = None
        if self._dashboard_url:
            import queue, threading
            self._timing_queue = queue.Queue(maxsize=500)
            def _timing_poster():
                import urllib.request
                batch = []
                while True:
                    try:
                        item = self._timing_queue.get(timeout=5)
                        if item is None:
                            break
                        batch.append(item)
                        # Drain up to 20 more without blocking
                        for _ in range(20):
                            try:
                                batch.append(self._timing_queue.get_nowait())
                            except queue.Empty:
                                break
                    except queue.Empty:
                        pass
                    if batch:
                        try:
                            data = json.dumps(batch).encode()
                            req = urllib.request.Request(
                                self._dashboard_url.rstrip('/') + '/ingest',
                                data=data,
                                headers={'Content-Type': 'application/json'},
                                method='POST',
                            )
                            urllib.request.urlopen(req, timeout=10)
                        except Exception:
                            pass  # Non-critical — dashboard may be down
                        batch = []
            self._timing_thread = threading.Thread(target=_timing_poster, daemon=True)
            self._timing_thread.start()

        # Load API key
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)

        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_SESSION_TOKEN']:
            if key in env_vars and key not in os.environ:
                os.environ[key] = env_vars[key]

        self.use_anthropic = provider in ("anthropic", "bedrock")

        if api_key:
            self.api_key = api_key
        elif provider == "xai":
            self.api_key = env_vars.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY")
        elif provider == "anthropic":
            self.api_key = env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "bedrock":
            self.api_key = None
        elif provider == "modal":
            self.api_key = env_vars.get("MODAL_API_KEY") or os.environ.get("MODAL_API_KEY")
        else:
            self.api_key = env_vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key and provider not in ("bedrock",):
            raise ValueError(f"No API key found for provider {provider}")

        if base_url:
            self.base_url = base_url
        elif provider == "xai":
            self.base_url = "https://api.x.ai/v1"
        elif provider == "modal":
            self.base_url = "https://corporate--glm5-serving-server.us-east.modal.direct/v1"
        else:
            self.base_url = None

        # Create client
        if provider == "bedrock":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required for Bedrock")
            self.client = AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                aws_region=os.environ.get("AWS_REGION", "us-east-2"),
            )
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            import httpx
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            # Infinite timeout — large prompts can take minutes to prefill on overloaded endpoints.
            client_kwargs["timeout"] = httpx.Timeout(None)
            self.client = OpenAI(**client_kwargs)

        # Initialize RNG
        self.rng = Generator(PCG64(seed))

        # Components (initialized in setup)
        self.conn = None
        self.simulator = None
        self.shock_manager = None
        self.tools = None
        self.event_logger = None
        self.agent = None
        self.api_server = None
        self.tool_executor = None

    def _log_response(self, turn: int, day: int, messages: List[Dict], raw_response: Any):
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "messages_count": len(messages),
            "raw_response": raw_response,
        }
        with open(self.response_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_tool_result(self, turn: int, day: int, tool_name: str, arguments: Dict, result: str):
        tool_results_file = self.logs_dir / f"tool_results_{self.run_id}.jsonl"
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        }
        with open(tool_results_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_timing(self, event: str, day: int, turn: int = 0, **kwargs):
        """Log a timing event to the timing JSONL file and push to dashboard."""
        entry = {
            "timestamp": now(),
            "run_id": self.run_id,
            "event": event,
            "day": day,
            "turn": turn,
            **kwargs,
        }
        with open(self.timing_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        # Push to ceobench dashboard (non-blocking)
        if self._timing_queue is not None:
            try:
                self._timing_queue.put_nowait(entry)
            except Exception:
                pass

    def _build_dashboard(self, day: int, last_result=None) -> str:
        """Build the daily dashboard."""
        inbox = self.shock_manager.get_inbox_items(day)
        inbox.extend(get_thread_inbox_items(self.conn, day))
        # Run daily scripts from the agent workspace
        calc_outputs = self._run_daily_scripts()
        return build_daily_dashboard(self.conn, day, last_result, calc_outputs, inbox)

    def _run_daily_scripts(self) -> Dict[str, str]:
        """Run registered daily scripts from server-side snapshots.

        Uses immutable content snapshots stored in the API server at registration
        time. Even if the agent modifies the source file or daily_scripts/ copy,
        the registered snapshot executes unchanged.
        """
        import subprocess
        import tempfile

        # Get snapshots from server (immutable copies stored at registration time)
        snapshots = self.api_server.get_daily_scripts()
        if not snapshots:
            return {}

        outputs = {}
        env = {
            'PATH': f'{os.path.dirname(sys.executable)}:/usr/local/bin:/usr/bin:/bin',
            'HOME': str(self.agent_workspace),
            'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
            'PYTHONPATH': str(self.agent_workspace),
            'NOVAMIND_API_PORT': str(self.api_server.port),
            'NOVAMIND_WORKSPACE': str(self.agent_workspace),
        }

        for name in sorted(snapshots.keys()):
            content = snapshots[name]
            # Write snapshot to a temp file for execution
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', prefix=f'daily_{name}_',
                dir=str(self.agent_workspace), delete=False
            )
            try:
                tmp.write(content)
                tmp.close()
                result = subprocess.run(
                    [sys.executable, '-u', tmp.name],
                    capture_output=True, text=True,
                    cwd=str(self.agent_workspace),
                    env=env, timeout=60,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr] {result.stderr}"
                outputs[name] = output
            except subprocess.TimeoutExpired:
                outputs[name] = "[Error: Script timed out after 60s]"
            except Exception as e:
                outputs[name] = f"[Error: {e}]"
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        return outputs

    def _save_checkpoint(self, day: int):
        """Save checkpoint for resume capability."""
        checkpoint = {
            'day': day,
            'run_id': self.run_id,
            'model': self.model,
            'provider': self.provider,
            'seed': self.seed,
            'scenario': self.scenario,
            'agent_total_turns': self.agent.total_turns if self.agent else 0,
            'daily_scripts': self.api_server.get_daily_scripts() if self.api_server else {},
        }
        checkpoint_file = self.workspace_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from disk."""
        checkpoint_file = self.workspace_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """Restore state from checkpoint."""
        cp_day = checkpoint['day']

        # Clean up any partial data from days beyond checkpoint
        # This handles the case where a previous run crashed mid-day
        if self.conn:
            day_tables = [
                'daily_usage', 'service_day', 'ledger', 'config_history',
                'ad_channel_leads', 'events', 'reputation_history',
                'api_costs', 'social_media_posts', 'notifications',
                'issues', 'dividends', 'funding_rounds',
            ]
            for table in day_tables:
                try:
                    self.conn.execute(f"DELETE FROM {table} WHERE day > ?", (cp_day,))
                except Exception:
                    pass  # Table may not exist in older schemas
            # Clean up subscriptions that started after checkpoint
            self.conn.execute("DELETE FROM subscriptions WHERE start_day > ?", (cp_day,))
            # Revert subscriptions that were cancelled/ended after checkpoint
            # Exclude plan='pending' leads — they should stay as lost, not revert to subscribed
            self.conn.execute(
                "UPDATE subscriptions SET status='subscribed', end_day=NULL WHERE end_day > ? AND plan != 'pending'",
                (cp_day,)
            )
            # Clean up enterprise turns created after checkpoint
            self.conn.execute("DELETE FROM enterprise_turns WHERE day > ?", (cp_day,))
            # Clean up vc turns created after checkpoint
            self.conn.execute("DELETE FROM vc_turns WHERE day > ?", (cp_day,))
            self.conn.commit()
            print(f"  Cleaned partial data for days > {cp_day}")

        # Truncate JSONL logs to remove entries from days beyond checkpoint
        for log_file in [
            self.logs_dir / f"tool_results_{self.run_id}.jsonl",
            self.logs_dir / f"raw_responses_{self.run_id}.jsonl",
        ]:
            if log_file.exists():
                kept_lines = []
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            entry_day = entry.get('day', 0)
                            if entry_day <= cp_day:
                                kept_lines.append(line)
                        except json.JSONDecodeError:
                            kept_lines.append(line)
                with open(log_file, 'w') as f:
                    for line in kept_lines:
                        f.write(line + "\n")
                print(f"  Trimmed {log_file.name}: kept entries for days <= {cp_day}")

        # Re-generate the checkpoint day's dashboard with correct DB data.
        # Previous resumes may have written a dashboard with 0 subs (before the
        # DB-fallback fix). Appending a fresh one ensures the monitor shows
        # correct values on startup.
        tool_log = self.logs_dir / f"tool_results_{self.run_id}.jsonl"
        if self.conn and tool_log.exists():
            fresh_dashboard = build_daily_dashboard(self.conn, cp_day)
            entry = {
                "timestamp": now(),
                "turn": 0,
                "day": cp_day,
                "tool": "_dashboard",
                "arguments": {},
                "result": fresh_dashboard,
            }
            with open(tool_log, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  Appended fresh dashboard for Day {cp_day} to JSONL")

        # Set simulator's current_day so step_day() increments to the right day
        if self.simulator:
            self.simulator.current_day = cp_day

        # Set tools' current_day
        if self.tools:
            self.tools.set_current_day(cp_day)

        if self.agent:
            self.agent.total_turns = checkpoint.get('agent_total_turns', 0)
        # Restore daily script snapshots
        if self.api_server and 'daily_scripts' in checkpoint:
            self.api_server.set_daily_scripts(checkpoint['daily_scripts'])

    def setup(self):
        """Initialize the simulation environment."""
        from .agent import BashAgent
        from .tools import get_bash_agent_tool_descriptions, BashAgentToolExecutor

        scenario_pack = SCENARIO_PACKS.get(self.scenario, ScenarioPack(
            name='Default', description='Balanced scenario'
        ))

        bench_config = BenchmarkConfig(
            seed=self.seed,
            total_days=self.total_days,
            initial_cash=self.initial_cash,
        )

        self.conn = init_database(self.db_path)
        # Allow cross-thread access for the API server (runs in a daemon thread)
        # The API server uses a lock for serialization, so this is safe.
        self.conn.execute("SELECT 1")  # ensure connection is initialized
        # Re-open with check_same_thread=False for API server compatibility
        self.conn.close()
        self.conn = _sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = _sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        customer_sim = CustomerSimulator(client=None, conn=self.conn, config=bench_config)
        self.simulator = Simulator(self.conn, bench_config, self.rng, customer_simulator=customer_sim)
        self.shock_manager = ShockManager(self.conn, self.rng, scenario_pack)
        self.tools = AgentTools(self.conn, 0, self.agent_workspace, self.db_path)

        self.event_logger = EventLogger(
            run_id=self.run_id,
            output_dir=self.logs_dir,
            seed=self.seed,
            scenario=self.scenario,
            config={
                'model': self.model,
                'provider': self.provider,
                'seed': self.seed,
                'scenario': self.scenario,
                'total_days': self.total_days,
                'initial_cash': self.initial_cash,
                'agent_type': 'bash_agent',
            }
        )

        self.simulator.set_event_logger(self.event_logger)
        self.tools.set_event_logger(self.event_logger)

        if not self.continue_from:
            self.simulator.initialize()
            self.event_logger.log_run_start()
        else:
            self.event_logger.log_run_start()

        # Initialize agent workspace with docs
        initialize_workspace(self.agent_workspace)

        # Start the API server
        self.api_server = NovaMindAPIServer(
            tools=self.tools,
            simulator=self.simulator,
            conn=self.conn,
            dashboard_callback=self._build_dashboard,
        )
        self.api_server.start()

        # Find the venv bin directory for novamind CLI access
        import sysconfig
        venv_bin = os.path.dirname(sys.executable)

        # Create tool executor with API server port in environment
        self.tool_executor = BashAgentToolExecutor(
            workspace_path=self.agent_workspace,
            env={
                'NOVAMIND_API_PORT': str(self.api_server.port),
                'NOVAMIND_WORKSPACE': str(self.agent_workspace),
                'PYTHONPATH': str(self.agent_workspace),
                'PATH': f'{venv_bin}:/usr/local/bin:/usr/bin:/bin',
            },
        )

        # Get bash agent tool descriptions
        tool_descriptions = get_bash_agent_tool_descriptions()

        # Create agent
        self.agent = BashAgent(
            tool_descriptions=tool_descriptions,
            client=self.client,
            model=self.model,
            max_turns_per_day=100,
            response_callback=self._log_response,
            reasoning_effort=self.reasoning_effort,
            tool_result_callback=self._log_tool_result,
            workspace_path=self.agent_workspace,
            total_days=self.total_days,
        )

        # Save run config
        config = {
            'run_id': self.run_id,
            'model': self.model,
            'provider': self.provider,
            'seed': self.seed,
            'scenario': self.scenario,
            'total_days': self.total_days,
            'initial_cash': self.initial_cash,
            'agent_type': 'bash_agent',
            'api_server_port': self.api_server.port,
        }
        with open(self.workspace_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a bash_agent tool."""
        result = self.tool_executor.execute(tool_name, arguments)

        # Check if bash output contains a day advancement
        if tool_name == 'bash':
            self.agent.check_day_advanced(result)

        return result

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full simulation."""
        self.setup()

        start_day = 1
        if self.continue_from:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_day = checkpoint['day'] + 1
                self._restore_from_checkpoint(checkpoint)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"RESUMING Bash Agent Run from Day {start_day}")
                    print(f"Run ID: {self.run_id}")
                    print(f"Model: {self.model}")
                    print(f"Checkpoint: Day {checkpoint['day']}")
                    print(f"Workspace: {self.workspace_dir}")
                    print(f"{'='*60}\n")
            else:
                print(f"WARNING: No checkpoint found, starting from Day 1")

        if start_day == 1 and verbose:
            print(f"\n{'='*60}")
            print(f"Starting Bash Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.model}")
            print(f"Provider: {self.provider}")
            print(f"Seed: {self.seed}")
            print(f"API Server Port: {self.api_server.port}")
            print(f"Agent Workspace: {self.agent_workspace}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        current_day = start_day - 1
        game_ended = False
        game_outcome = None
        last_result = None

        import time as _time

        for day in range(start_day, self.total_days + 1):
            _day_start = _time.monotonic()
            current_day = day
            self.tools.set_current_day(day)
            self.event_logger.set_day(day)

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

            # Build dashboard (timed)
            _t0 = _time.monotonic()
            dashboard = self._build_dashboard(day, last_result)
            _dashboard_elapsed = _time.monotonic() - _t0
            self._log_tool_result(0, day, '_dashboard', {}, dashboard)
            self._log_timing("dashboard", day, elapsed_s=round(_dashboard_elapsed, 3))

            # Agent loop for this day
            observation = dashboard
            info = {'day': day, 'cash': get_cash(self.conn)}
            turns_today = 0
            day_ended = False
            _day_llm_total = 0.0
            _day_tool_total = 0.0

            while not day_ended and turns_today < 100:
                turns_today += 1

                # LLM call (timed)
                _t0 = _time.monotonic()
                action = self.agent.act(observation, 0, False, info)
                _llm_elapsed = _time.monotonic() - _t0
                _day_llm_total += _llm_elapsed

                if action is None:
                    # No action — force next-day via bash
                    action = Action(tool='bash', arguments={'command': 'novamind-operation next-day'})

                tool_name = action.tool
                tool_args_preview = ""
                if tool_name == 'bash':
                    tool_args_preview = (action.arguments or {}).get('command', '')[:120]
                else:
                    tool_args_preview = json.dumps(action.arguments or {})[:120]

                self._log_timing("llm_call", day, turn=turns_today,
                                 elapsed_s=round(_llm_elapsed, 2),
                                 tool=tool_name, tool_preview=tool_args_preview)

                # Execute action (timed)
                if verbose:
                    if tool_name == 'bash':
                        print(f"    [Turn {turns_today}] bash: {tool_args_preview[:100]}")
                    else:
                        print(f"    [Turn {turns_today}] {tool_name}({tool_args_preview[:100]})")

                _t0 = _time.monotonic()
                result = self._execute_tool(action.tool, action.arguments or {})
                _tool_elapsed = _time.monotonic() - _t0
                _day_tool_total += _tool_elapsed
                observation = result if isinstance(result, str) else json.dumps(result)

                self._log_timing("tool_exec", day, turn=turns_today,
                                 elapsed_s=round(_tool_elapsed, 3),
                                 tool=tool_name, tool_preview=tool_args_preview)

                # Log tool result
                self._log_tool_result(
                    self.agent.total_turns, day,
                    action.tool, action.arguments or {},
                    observation  # Full result in JSONL (tool already caps at 50K)
                )

                if verbose:
                    print(f"      → {observation[:200]}")
                    print(f"      ⏱ llm={_llm_elapsed:.1f}s tool={_tool_elapsed:.1f}s")

                # Check if the agent detected a day advancement
                if self.agent.day_advanced:
                    day_ended = True
                    self.agent.clear_day_advanced()

                # Check if step_day timed out (via API server)
                if self.api_server._step_day_timed_out:
                    print(f"\n⚠️  step_day timed out on day {day} (>{self.api_server.STEP_DAY_TIMEOUT}s)")
                    print(f"Auto-quitting to prevent runaway timeouts. Saving checkpoint...")
                    self._save_checkpoint(day - 1)  # Save checkpoint at PREVIOUS day (current day incomplete)
                    self.event_logger.save_incremental()
                    game_ended = True
                    game_outcome = 'timeout'
                    break

                info = {'day': day, 'cash': get_cash(self.conn)}

            if game_ended:
                break

            # If day didn't end through agent action, step simulation
            if not day_ended:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                _step_start = _time.monotonic()
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self.simulator.step_day)
                try:
                    day_result = future.result(timeout=1200)
                except FuturesTimeoutError:
                    _step_elapsed = _time.monotonic() - _step_start
                    print(f"\n⚠️  step_day took {_step_elapsed:.1f}s on day {day} (>1200s timeout)")
                    print(f"Auto-quitting. Saving checkpoint at day {day - 1}...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    self._save_checkpoint(day - 1)
                    self.event_logger.save_incremental()
                    game_ended = True
                    game_outcome = 'timeout'
                    break
                executor.shutdown(wait=False)
                _step_elapsed = _time.monotonic() - _step_start
                last_result = day_result
            else:
                # Day was advanced by the API server — get the result
                last_result = self.api_server._last_day_result
                _step_elapsed = getattr(self.api_server, '_last_step_elapsed', 0)

            # Log step_day timing
            self._log_timing("step_day", day, elapsed_s=round(_step_elapsed, 2))

            # Log slow step_day as warning (but don't auto-quit)
            if _step_elapsed > 300:
                print(f"\n⚠️  step_day took {_step_elapsed:.1f}s on day {day} (>300s) — continuing")

            # Log daily state
            _subs = 0
            if last_result:
                _subs = get_active_subscriber_count(self.conn)
                self.event_logger.log_daily_state(
                    cash=last_result.cash,
                    mrr=last_result.mrr,
                    subscribers=_subs,
                    usage=last_result.total_usage,
                    overload=last_result.overload,
                    outage=last_result.outage,
                    group_reputations=get_all_group_reputations(self.conn),
                    group_awareness=get_all_group_awareness(self.conn),
                    total_dividends=last_result.total_dividends_paid,
                    founder_dividends=last_result.founder_cumulative_dividends,
                )

            # Per-day timing summary
            _day_elapsed = _time.monotonic() - _day_start
            _day_other = _day_elapsed - _day_llm_total - _day_tool_total - _step_elapsed - _dashboard_elapsed
            self._log_timing("day_summary", day,
                             elapsed_s=round(_day_elapsed, 1),
                             llm_total_s=round(_day_llm_total, 1),
                             tool_total_s=round(_day_tool_total, 1),
                             step_day_s=round(_step_elapsed, 1),
                             dashboard_s=round(_dashboard_elapsed, 2),
                             other_s=round(max(_day_other, 0), 1),
                             turns=turns_today,
                             subs=_subs,
                             cash=last_result.cash if last_result else 0,
                             dividends=last_result.founder_cumulative_dividends if last_result else 0)

            # Print per-day timing summary to stderr (visible in logs)
            import sys as _sys
            _pct_llm = (_day_llm_total / _day_elapsed * 100) if _day_elapsed > 0 else 0
            _pct_step = (_step_elapsed / _day_elapsed * 100) if _day_elapsed > 0 else 0
            _pct_tool = (_day_tool_total / _day_elapsed * 100) if _day_elapsed > 0 else 0
            print(f"\n⏱ DAY {day} TIMING: total={_day_elapsed:.0f}s | "
                  f"llm={_day_llm_total:.0f}s ({_pct_llm:.0f}%) | "
                  f"step_day={_step_elapsed:.0f}s ({_pct_step:.0f}%) | "
                  f"tools={_day_tool_total:.0f}s ({_pct_tool:.0f}%) | "
                  f"dashboard={_dashboard_elapsed:.1f}s | "
                  f"turns={turns_today}", file=_sys.stderr, flush=True)

            if verbose and last_result:
                print(f"  📊 End of day: Cash=${last_result.cash:,.0f}, IndSubs={last_result.total_individual_subscribers}, EntSeats={last_result.total_enterprise_subscription_seats}")

            # Save checkpoint
            self._save_checkpoint(day)
            self.event_logger.save_incremental()

            # Check bankruptcy
            if self.simulator.shutdown_mode:
                game_ended = True
                game_outcome = 'bankrupt'
                if verbose:
                    print(f"\n💀 BANKRUPT at day {day}!")
                break

        if not game_outcome:
            game_outcome = 'completed'

        # Stop API server
        if self.api_server:
            self.api_server.stop()

        final_cash = get_cash(self.conn)
        self.event_logger.log_run_end(final_cash, current_day, game_outcome)
        self.event_logger.save()

        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Final Cash: ${final_cash:,.0f}")
            print(f"Days Run: {current_day}")
            print(f"Outcome: {game_outcome}")
            print(f"Total Turns: {self.agent.total_turns}")
            print(f"{'='*60}\n")

        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'scenario': self.scenario,
            'final_cash': final_cash,
            'days_run': current_day,
            'outcome': game_outcome,
            'total_turns': self.agent.total_turns,
            'workspace_dir': str(self.workspace_dir),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run bash agent for SaaS Bench")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "xai", "anthropic", "bedrock", "modal"],
                        help="API provider")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--days", type=int, default=3650, help="Total simulation days")
    parser.add_argument("--workspace", type=Path, help="Workspace base directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--reasoning-effort",
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for GPT-5.2+ models")
    parser.add_argument("--continue-from", type=Path,
                        help="Path to previous run directory to resume from")

    args = parser.parse_args()

    runner = BashAgentRunner(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        seed=args.seed,
        scenario=args.scenario,
        total_days=args.days,
        workspace_base=args.workspace,
        reasoning_effort=args.reasoning_effort,
        continue_from=args.continue_from,
    )

    result = runner.run(verbose=not args.quiet)
    print(f"\nResult: {result['outcome']}")
    print(f"Final Cash: ${result['final_cash']:,.0f}")
    print(f"Workspace: {result['workspace_dir']}")


if __name__ == "__main__":
    main()
