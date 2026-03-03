#!/usr/bin/env python3
"""Test runner for Baseline LLM agent with SaaS Bench.

This script runs a simulation using the baseline agent with any OpenAI-compatible API.
Supports OpenAI, xAI/Grok, Anthropic (direct and Bedrock), and other compatible providers.
"""

import json
import os
import shutil
import sqlite3
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

# Optional imports for Anthropic
try:
    import anthropic
    from anthropic import AnthropicBedrock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from saas_bench.config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from saas_bench.database import init_database, get_cash, get_active_subscriber_count, get_config, get_all_group_reputations, get_all_group_awareness
from saas_bench.simulation import Simulator
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.tools import AgentTools, get_tool_descriptions
from saas_bench.shocks import ShockManager
from saas_bench.event_logger import EventLogger
from saas_bench.environment import Action, build_daily_dashboard, get_thread_inbox_items


def now() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars


class BaselineRunner:
    """Runner for baseline LLM agent with SaaS Bench."""

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
        restart_from_day: Optional[int] = None,
        snapshot_interval: int = 50,
    ):
        """Initialize the runner.

        Args:
            model: Model name to use
            provider: Provider name (openai, xai, etc.)
            base_url: Custom API base URL (for xAI, etc.)
            api_key: API key (or loaded from env)
            seed: Random seed
            scenario: Scenario name
            total_days: Total simulation days
            initial_cash: Starting cash
            workspace_base: Base directory for workspaces
            reasoning_effort: Reasoning effort for GPT-5.2+ (none, low, medium, high, xhigh)
            continue_from: Path to a previous run directory to resume from
            restart_from_day: Day number to restart from (forks into a new run)
            snapshot_interval: Save DB snapshots every N days (default: 50)
        """
        self.model = model
        self.provider = provider
        self.seed = seed
        self.scenario = scenario
        self.total_days = total_days
        self.initial_cash = initial_cash
        self.reasoning_effort = reasoning_effort
        self.continue_from = continue_from
        self.restart_from_day = restart_from_day
        self.snapshot_interval = snapshot_interval

        if restart_from_day is not None and continue_from is None:
            raise ValueError("--restart-from-day requires --continue-from to specify the source run")

        if restart_from_day is not None:
            # Fork: create a new run directory from an existing run's snapshot
            source_dir = Path(continue_from).resolve()
            if not source_dir.exists():
                raise FileNotFoundError(f"Source run directory not found: {source_dir}")
            # Load source run config
            config_file = source_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    old_config = json.load(f)
                source_run_id = old_config['run_id']
            else:
                source_run_id = source_dir.name.replace('run_', '')
            # Create new run directory
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or source_dir.parent).resolve()
            self.workspace_dir = self.workspace_base / f"run_{self.run_id}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._source_dir = source_dir
            self._source_run_id = source_run_id
            # Don't use continue_from logic — we handle restart separately
            self.continue_from = None
        elif continue_from:
            # Resume: reuse existing run directory
            self.workspace_dir = Path(continue_from).resolve()
            if not self.workspace_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {self.workspace_dir}")
            # Load run_id from config
            config_file = self.workspace_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    old_config = json.load(f)
                self.run_id = old_config['run_id']
            else:
                # Extract from directory name: run_XXXXXXXX
                self.run_id = self.workspace_dir.name.replace('run_', '')
            self.workspace_base = self.workspace_dir.parent
            self._source_dir = None
        else:
            # Fresh run: generate new ID and directory
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or Path('./baseline_runs')).resolve()
            self.workspace_dir = self.workspace_base / f"run_{self.run_id}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._source_dir = None

        # Logs directory
        self.logs_dir = self.workspace_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Checkpoints directory (per-day JSONs + periodic DB snapshots)
        self.checkpoints_dir = self.workspace_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Database path
        self.db_path = self.workspace_dir / "world.db"

        # Log file for raw responses
        self.response_log_file = self.logs_dir / f"raw_responses_{self.run_id}.jsonl"

        # Load API key from env if not provided
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)

        # Export AWS credentials to os.environ for Bedrock
        # (AnthropicBedrock reads from os.environ, not from passed args)
        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_SESSION_TOKEN']:
            if key in env_vars and key not in os.environ:
                os.environ[key] = env_vars[key]

        # Determine if using Anthropic/Bedrock
        self.use_anthropic = provider in ("anthropic", "bedrock")

        if api_key:
            self.api_key = api_key
        elif provider == "xai":
            self.api_key = env_vars.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY")
        elif provider == "anthropic":
            self.api_key = env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "bedrock":
            # Bedrock uses AWS credentials, not API key
            self.api_key = None
        elif provider == "modal":
            self.api_key = env_vars.get("MODAL_API_KEY") or os.environ.get("MODAL_API_KEY")
        else:
            self.api_key = env_vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key and provider not in ("bedrock",):
            raise ValueError(f"No API key found for provider {provider}")

        # Set up base URL
        if base_url:
            self.base_url = base_url
        elif provider == "xai":
            self.base_url = "https://api.x.ai/v1"
        elif provider == "modal":
            self.base_url = "https://api.us-west-2.modal.direct/v1"
        else:
            self.base_url = None  # Use default

        # Create client based on provider
        if provider == "bedrock":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required for Bedrock. Install with: pip install anthropic")
            # Pass AWS credentials explicitly (boto3 session resolution is flaky)
            self.client = AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                aws_region=os.environ.get("AWS_REGION", "us-east-2"),
            )
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            # OpenAI-compatible client
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
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

    def _log_response(self, turn: int, day: int, messages: List[Dict], raw_response: Any):
        """Log raw API response to JSONL file."""
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
        """Log tool execution result to separate JSONL file."""
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

    def setup(self):
        """Initialize the simulation environment."""
        from .agent import BaselineAgent

        # If restart_from_day, prepare the forked run (copies DB, logs, etc.)
        if self.restart_from_day is not None:
            self._prepare_restart()

        # Get scenario
        scenario_pack = SCENARIO_PACKS.get(self.scenario, ScenarioPack(
            name='Default',
            description='Balanced scenario'
        ))

        # Create benchmark config
        bench_config = BenchmarkConfig(
            seed=self.seed,
            total_days=self.total_days,
            initial_cash=self.initial_cash,
        )

        # Initialize database (creates tables if needed, connects to existing if present)
        self.conn = init_database(self.db_path)

        # Initialize customer simulator (LLM-generated social posts via Bedrock Haiku 4.5)
        customer_sim = CustomerSimulator(client=None, conn=self.conn, config=bench_config)

        # Initialize components
        self.simulator = Simulator(self.conn, bench_config, self.rng, customer_simulator=customer_sim)
        self.shock_manager = ShockManager(self.conn, self.rng, scenario_pack)
        self.tools = AgentTools(self.conn, 0, self.workspace_dir, self.db_path)

        # Initialize event logger
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
                'agent_type': 'baseline',
            }
        )

        # Connect event logger
        self.simulator.set_event_logger(self.event_logger)
        self.tools.set_event_logger(self.event_logger)

        if self.restart_from_day is not None:
            # Forked run: DB already prepared by _prepare_restart()
            self.event_logger.log_run_start()
        elif not self.continue_from:
            # Fresh run: initialize simulation from scratch
            self.simulator.initialize()
            self.event_logger.log_run_start()
        else:
            # Resuming: DB already has state, just log continuation
            self.event_logger.log_run_start()  # Append to existing log

        # Get tool descriptions (standalone function)
        tool_descriptions = get_tool_descriptions()

        # Create agent with response callback
        self.agent = BaselineAgent(
            tool_descriptions=tool_descriptions,
            client=self.client,
            model=self.model,
            max_turns_per_day=100,
            response_callback=self._log_response,
            reasoning_effort=self.reasoning_effort,
            tool_result_callback=self._log_tool_result,
        )

        # Save config
        self._save_config()

    def _save_config(self):
        """Save run configuration."""
        config_file = self.workspace_dir / "config.json"
        config = {
            'run_id': self.run_id,
            'model': self.model,
            'provider': self.provider,
            'base_url': self.base_url,
            'seed': self.seed,
            'scenario': self.scenario,
            'total_days': self.total_days,
            'initial_cash': self.initial_cash,
            'agent_type': 'baseline',
            'snapshot_interval': self.snapshot_interval,
            'created_at': now(),
        }
        if self._source_dir is not None:
            config['forked_from'] = {
                'run_id': self._source_run_id,
                'run_dir': str(self._source_dir),
                'restart_day': self.restart_from_day,
            }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, day: int):
        """Save checkpoint after each completed day.

        - Per-day JSON checkpoint saved to checkpoints/day_NNN.json (every day)
        - DB snapshot saved to checkpoints/day_NNN.db (every snapshot_interval days)
        - Also saves checkpoint.json for backward compatibility with --continue-from
        """
        checkpoint = {
            'day': day,
            'timestamp': now(),
            'agent_memory': self.agent.memory if self.agent else [],
            'agent_total_turns': self.agent.total_turns if self.agent else 0,
            'daily_calculations': self.tools.get_daily_calculations() if self.tools else {},
            'scripts': self.tools.get_scripts() if self.tools else {},
            'rng_state': {
                'bit_generator': self.rng.bit_generator.state['bit_generator'],
                'state': {
                    'state': int(self.rng.bit_generator.state['state']['state']),
                    'inc': int(self.rng.bit_generator.state['state']['inc']),
                },
                'has_uint32': int(self.rng.bit_generator.state['has_uint32']),
                'uinteger': int(self.rng.bit_generator.state['uinteger']),
            },
        }

        # 1. Save per-day checkpoint JSON (every day)
        day_checkpoint_file = self.checkpoints_dir / f"day_{day:04d}.json"
        tmp_file = self.checkpoints_dir / f"day_{day:04d}.json.tmp"
        with open(tmp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        tmp_file.rename(day_checkpoint_file)

        # 2. Save backward-compatible checkpoint.json (overwritten each day)
        compat_file = self.workspace_dir / "checkpoint.json"
        compat_tmp = self.workspace_dir / "checkpoint.json.tmp"
        with open(compat_tmp, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        compat_tmp.rename(compat_file)

        # 3. Save DB snapshot at snapshot_interval boundaries
        if day % self.snapshot_interval == 0:
            db_snapshot_file = self.checkpoints_dir / f"day_{day:04d}.db"
            if not db_snapshot_file.exists():
                self.conn.execute(f"VACUUM INTO '{db_snapshot_file}'")

    def _load_checkpoint(self, day: Optional[int] = None) -> Optional[Dict]:
        """Load checkpoint, optionally for a specific day.

        Args:
            day: If specified, load the checkpoint for this exact day.
                 If None, load the latest checkpoint (checkpoint.json).
        """
        if day is not None:
            # Load per-day checkpoint
            day_file = self.checkpoints_dir / f"day_{day:04d}.json"
            if day_file.exists():
                with open(day_file) as f:
                    return json.load(f)
            # Fall back: check if the backward-compatible checkpoint matches
            checkpoint_file = self.workspace_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    cp = json.load(f)
                if cp.get('day') == day:
                    return cp
            return None
        else:
            # Load latest checkpoint (backward compatible)
            checkpoint_file = self.workspace_dir / "checkpoint.json"
            if not checkpoint_file.exists():
                return None
            with open(checkpoint_file) as f:
                return json.load(f)

    def _find_nearest_db_snapshot(self, target_day: int, source_dir: Path) -> Optional[Path]:
        """Find the nearest DB snapshot at or before target_day in the source run's checkpoints."""
        checkpoints_dir = source_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        # Find all DB snapshots
        snapshots = sorted(checkpoints_dir.glob("day_*.db"))
        best = None
        for snap in snapshots:
            # Extract day number from filename: day_0050.db -> 50
            snap_day = int(snap.stem.split('_')[1])
            if snap_day <= target_day:
                best = snap
            else:
                break
        return best

    def _prepare_restart(self):
        """Prepare a forked run by restoring DB snapshot and rolling forward to target day.

        This is called during setup() when restart_from_day is set.
        Copies the nearest DB snapshot, rolls back day-indexed tables to the target day,
        copies JSONL logs truncated to target day, and loads the per-day checkpoint JSON.
        """
        target_day = self.restart_from_day
        source_dir = self._source_dir
        source_run_id = self._source_run_id

        print(f"Forking run {source_run_id} from day {target_day} → new run {self.run_id}")

        # 1. Find the checkpoint JSON for the target day
        source_checkpoints = source_dir / "checkpoints"
        source_checkpoint_file = source_checkpoints / f"day_{target_day:04d}.json"
        if not source_checkpoint_file.exists():
            # Fall back to source's checkpoint.json
            source_checkpoint_file = source_dir / "checkpoint.json"
            if source_checkpoint_file.exists():
                with open(source_checkpoint_file) as f:
                    cp = json.load(f)
                if cp.get('day') != target_day:
                    raise FileNotFoundError(
                        f"No checkpoint JSON found for day {target_day} in {source_dir}. "
                        f"Available: latest checkpoint is day {cp.get('day')}. "
                        f"Per-day checkpoints must exist in {source_checkpoints}/"
                    )
            else:
                raise FileNotFoundError(f"No checkpoints found in {source_dir}")

        # Load the target checkpoint
        with open(source_checkpoint_file) as f:
            self._restart_checkpoint = json.load(f)

        # 2. Find nearest DB snapshot <= target_day
        db_snapshot = self._find_nearest_db_snapshot(target_day, source_dir)
        if db_snapshot is None:
            # No snapshots at all — fall back to copying the full world.db
            # and rolling back (works but loses some state accuracy for stateful tables)
            source_db = source_dir / "world.db"
            if not source_db.exists():
                raise FileNotFoundError(f"No world.db found in {source_dir}")
            print(f"  WARNING: No DB snapshots found. Copying full world.db and rolling back.")
            print(f"  Stateful tables (customer_state, etc.) may not reflect exact day-{target_day} state.")
            shutil.copy2(source_db, self.db_path)
        else:
            snapshot_day = int(db_snapshot.stem.split('_')[1])
            print(f"  Using DB snapshot from day {snapshot_day}: {db_snapshot.name}")
            shutil.copy2(db_snapshot, self.db_path)

        # 3. Roll back the DB to target_day (clean up anything after target_day)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        day_tables = [
            'daily_usage', 'service_day', 'ledger', 'config_history',
            'ad_channel_leads', 'events', 'reputation_history',
            'api_costs', 'social_media_posts', 'notifications',
            'issues', 'dividends', 'funding_rounds',
        ]
        for table in day_tables:
            try:
                conn.execute(f"DELETE FROM {table} WHERE day > ?", (target_day,))
            except Exception:
                pass
        # Clean up subscriptions created after target_day
        conn.execute("DELETE FROM subscriptions WHERE start_day > ?", (target_day,))
        # Revert subscriptions that ended after target_day (exclude pending leads)
        conn.execute(
            "UPDATE subscriptions SET status='subscribed', end_day=NULL WHERE end_day > ? AND plan != 'pending'",
            (target_day,)
        )
        # Clean up customers created after target_day
        conn.execute("DELETE FROM customers WHERE created_day > ?", (target_day,))
        # Clean up customer_state for deleted customers
        conn.execute("DELETE FROM customer_state WHERE customer_id NOT IN (SELECT customer_id FROM customers)")
        # Clean up enterprise/vc turns after target_day
        conn.execute("DELETE FROM enterprise_turns WHERE day > ?", (target_day,))
        try:
            conn.execute("DELETE FROM vc_turns WHERE day > ?", (target_day,))
        except Exception:
            pass
        # Clean up research projects started after target_day
        try:
            conn.execute("DELETE FROM research_projects WHERE started_day > ?", (target_day,))
            conn.execute(
                "UPDATE research_projects SET status='in_progress' WHERE status='completed' AND expected_completion_day > ?",
                (target_day,)
            )
        except Exception:
            pass
        # Clean up pending group research
        try:
            conn.execute("DELETE FROM pending_group_research WHERE started_day > ?", (target_day,))
            conn.execute(
                "UPDATE pending_group_research SET status='in_progress' WHERE status='completed' AND expected_completion_day > ?",
                (target_day,)
            )
        except Exception:
            pass
        # Clean up segment discovery after target_day
        try:
            conn.execute("DELETE FROM segment_discovery WHERE day > ?", (target_day,))
        except Exception:
            pass
        # Clean up competitor events after target_day
        try:
            conn.execute("DELETE FROM competitor_events WHERE day > ?", (target_day,))
        except Exception:
            pass
        # Clean up macroeconomic conditions after target_day
        try:
            conn.execute("DELETE FROM macroeconomic_conditions WHERE day > ?", (target_day,))
        except Exception:
            pass
        conn.commit()
        conn.close()

        # 4. Copy and truncate JSONL logs
        source_logs = source_dir / "logs"
        if source_logs.exists():
            for log_file in source_logs.glob(f"*_{source_run_id}.jsonl"):
                # Determine new filename with new run_id
                new_name = log_file.name.replace(source_run_id, self.run_id)
                dest_file = self.logs_dir / new_name
                # Copy only entries with day <= target_day
                kept_lines = []
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get('day', 0) <= target_day:
                                kept_lines.append(line)
                        except json.JSONDecodeError:
                            kept_lines.append(line)
                with open(dest_file, 'w') as f:
                    for line in kept_lines:
                        f.write(line + "\n")
            # Also copy the main event log
            for log_file in source_logs.glob(f"run_{source_run_id}.jsonl"):
                new_name = log_file.name.replace(source_run_id, self.run_id)
                dest_file = self.logs_dir / new_name
                shutil.copy2(log_file, dest_file)

        # Update response log file path for new run_id
        self.response_log_file = self.logs_dir / f"raw_responses_{self.run_id}.jsonl"

        print(f"  Forked run ready at: {self.workspace_dir}")
        print(f"  Restarting from day {target_day + 1}")

    def _restore_agent_state(self, checkpoint: Dict):
        """Restore only agent, tools, and RNG state from checkpoint.

        Used by restart-from-day where the DB is already in the correct state
        (prepared by _prepare_restart). Does NOT touch the DB or JSONL logs.
        """
        cp_day = checkpoint['day']

        # Restore agent memory and turns
        if self.agent:
            self.agent.memory = checkpoint.get('agent_memory', [])
            self.agent.total_turns = checkpoint.get('agent_total_turns', 0)
            self.agent.current_day = cp_day

        # Restore daily calculations
        if self.tools and 'daily_calculations' in checkpoint:
            self.tools.set_daily_calculations(checkpoint['daily_calculations'])

        # Restore named scripts
        if self.tools and 'scripts' in checkpoint:
            self.tools.set_scripts(checkpoint['scripts'])

        # Restore RNG state
        rng_state = checkpoint.get('rng_state')
        if rng_state:
            self.rng.bit_generator.state = {
                'bit_generator': rng_state['bit_generator'],
                'state': {
                    'state': rng_state['state']['state'],
                    'inc': rng_state['state']['inc'],
                },
                'has_uint32': rng_state['has_uint32'],
                'uinteger': rng_state['uinteger'],
            }

        # Restore simulator day counter
        if self.simulator:
            self.simulator.current_day = cp_day

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """Restore agent and RNG state from checkpoint."""
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

        # Truncate JSONL logs to remove entries from days beyond checkpoint
        for log_file in [
            self.logs_dir / f"tool_results_{self.run_id}.jsonl",
            self.response_log_file,
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

        # Restore agent memory and turns
        if self.agent:
            self.agent.memory = checkpoint.get('agent_memory', [])
            self.agent.total_turns = checkpoint.get('agent_total_turns', 0)
            self.agent.current_day = cp_day

        # Restore daily calculations
        if self.tools and 'daily_calculations' in checkpoint:
            self.tools.set_daily_calculations(checkpoint['daily_calculations'])

        # Restore named scripts
        if self.tools and 'scripts' in checkpoint:
            self.tools.set_scripts(checkpoint['scripts'])

        # Restore RNG state
        rng_state = checkpoint.get('rng_state')
        if rng_state:
            self.rng.bit_generator.state = {
                'bit_generator': rng_state['bit_generator'],
                'state': {
                    'state': rng_state['state']['state'],
                    'inc': rng_state['state']['inc'],
                },
                'has_uint32': rng_state['has_uint32'],
                'uinteger': rng_state['uinteger'],
            }

        # Restore simulator day counter (step_day increments by 1, so set to last completed day)
        if self.simulator:
            self.simulator.current_day = cp_day

    def _build_dashboard(self, day: int, last_result=None) -> str:
        """Build the daily dashboard. Delegates to the shared build_daily_dashboard()."""
        inbox = self.shock_manager.get_inbox_items(day)
        # Add thread inbox items (new threads + new messages on existing threads)
        inbox.extend(get_thread_inbox_items(self.conn, day))
        calc_outputs = self.tools.run_daily_calculations()
        return build_daily_dashboard(self.conn, day, last_result, calc_outputs, inbox)

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by dispatching to the appropriate AgentTools method."""
        # Map tool names to methods
        tool_methods = {
            'set_prices': lambda args: self.tools.set_prices({k: v for k, v in args.items() if v is not None}),
            'set_model_tiers': lambda args: self.tools.set_model_tiers({k: v for k, v in args.items() if v is not None}),
            'set_daily_spend': lambda args: self.tools.set_daily_spend({k: v for k, v in args.items() if v is not None}),
            'set_ad_channel_spend': lambda args: self.tools.set_ad_channel_spend({k: v for k, v in args.items() if v is not None}),
            'set_targeted_ad_spend': lambda args: self.tools.set_targeted_ad_spend(args.get('targeted_spend', args)),
            'set_capacity_tier': lambda args: self.tools.set_capacity_tier(args.get('tier', args.get('capacity_tier', 0))),
            'set_usage_quotas': lambda args: self.tools.set_usage_quotas(args),
            'send_enterprise_deal': lambda args: self.tools.send_enterprise_deal(deals=args.get('deals', [])),
            'python_exec': lambda args: self.tools.python_exec(args.get('code', '')),
            'get_social_posts': lambda args: self.tools.get_social_posts(args.get('days', 7), args.get('limit', 50)),

            'get_cost_info': lambda args: self.tools.get_cost_info(),
            'register_daily_calculation': lambda args: self.tools.register_daily_calculation(args.get('name'), args.get('code', '')),
            'remove_daily_calculation': lambda args: self.tools.remove_daily_calculation(args.get('name')),
            'list_daily_calculations': lambda args: self.tools.list_daily_calculations(),
            # Named scripts
            'register_script': lambda args: self.tools.register_script(args.get('name', ''), args.get('code', '')),
            'run_script': lambda args: self.tools.run_script(args.get('name', '')),
            'list_scripts': lambda args: self.tools.list_scripts(),
            'delete_script': lambda args: self.tools.delete_script(args.get('name', '')),
            'log_rationale': lambda args: self._log_rationale(args.get('rationale', args.get('text', ''))),
            'get_tool_documentation': lambda args: self.tools.get_tool_documentation(args.get('tool_names')),
            # R&D Research Projects
            'start_research_project': lambda args: self.tools.start_research_project(args.get('project_id', '')),
            'list_research_projects': lambda args: self.tools.list_research_projects(),
            # Enterprise negotiation
            'reject_enterprise_deal': lambda args: self.tools.reject_enterprise_deal(deals=args.get('deals', [])),
            # VC tools
            'list_potential_vcs': lambda args: self.tools.list_potential_vcs(),
            'send_vc_deal': lambda args: self.tools.send_vc_deal(deals=args.get('deals', [])),
            'reject_vc_deal': lambda args: self.tools.reject_vc_deal(deals=args.get('deals', [])),
            'get_cap_table_info': lambda args: self.tools.get_cap_table_info(),
            'settle_investments': lambda args: self.tools.settle_investments(),
            'declare_dividend': lambda args: self.tools.declare_dividend(args.get('amount', 0)),
            # Market discovery
            'research_market': lambda args: self.tools.research_market(),
            'research_group': lambda args: self.tools.research_group(args.get('group_id', '')),
            'get_market_overview': lambda args: self.tools.get_market_overview(),
            'get_group_insights': lambda args: self.tools.get_group_insights(args.get('group_id', '')),
            # Targeted spend
            'set_targeted_ops_spend': lambda args: self.tools.set_targeted_ops_spend(args.get('targeted_spend', args)),
            'set_targeted_dev_spend': lambda args: self.tools.set_targeted_dev_spend(args.get('targeted_spend', args)),
            # Database exploration
            'list_all_tables': lambda args: self.tools.list_all_tables(),
            'describe_tables': lambda args: self.tools.describe_tables(args.get('table_names')),
        }

        if tool_name in tool_methods:
            result = tool_methods[tool_name](arguments)
            # Handle ToolResult objects (has success, message, data attributes)
            if hasattr(result, 'message'):
                if hasattr(result, 'data') and result.data:
                    return f"{result.message}\n\nData: {json.dumps(result.data, default=str)}"
                return result.message
            elif hasattr(result, 'output'):
                return result.output
            return str(result) if result is not None else "Done"
        else:
            return f"Unknown tool: {tool_name}"

    def _log_rationale(self, rationale: str) -> str:
        """Log agent rationale."""
        # Log as a special agent action
        if self.event_logger:
            self.event_logger.log_agent_action(
                tool_name='log_rationale',
                arguments={'rationale': rationale},
                result={'logged': True},
                success=True
            )
        return f"Rationale logged: {rationale}"

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full simulation."""
        self.setup()

        # Determine start day
        start_day = 1
        if self.restart_from_day is not None:
            # Forked run: restore agent state from the checkpoint prepared in _prepare_restart
            checkpoint = self._restart_checkpoint
            start_day = checkpoint['day'] + 1
            self._restore_agent_state(checkpoint)
            if verbose:
                print(f"\n{'='*60}")
                print(f"FORKED Baseline Agent Run from Day {start_day}")
                print(f"New Run ID: {self.run_id}")
                print(f"Source: {self._source_run_id} (day {checkpoint['day']})")
                print(f"Model: {self.model}")
                print(f"Agent Memory: {len(checkpoint.get('agent_memory', []))} notes")
                print(f"Daily Calcs: {len(checkpoint.get('daily_calculations', {}))} registered")
                print(f"Scripts: {len(checkpoint.get('scripts', {}))} registered")
                print(f"Snapshot Interval: every {self.snapshot_interval} days")
                print(f"Workspace: {self.workspace_dir}")
                print(f"{'='*60}\n")
        elif self.continue_from:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_day = checkpoint['day'] + 1
                self._restore_from_checkpoint(checkpoint)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"RESUMING Baseline Agent Run from Day {start_day}")
                    print(f"Run ID: {self.run_id}")
                    print(f"Model: {self.model}")
                    print(f"Checkpoint: Day {checkpoint['day']}, Turns: {checkpoint.get('agent_total_turns', 0)}")
                    print(f"Agent Memory: {len(checkpoint.get('agent_memory', []))} notes")
                    print(f"Daily Calcs: {len(checkpoint.get('daily_calculations', {}))} registered")
                    print(f"Scripts: {len(checkpoint.get('scripts', {}))} registered")
                    print(f"Workspace: {self.workspace_dir}")
                    print(f"{'='*60}\n")
            else:
                print(f"WARNING: No checkpoint found in {self.workspace_dir}, starting from Day 1")

        if start_day == 1 and verbose:
            print(f"\n{'='*60}")
            print(f"Starting Baseline Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.model}")
            print(f"Provider: {self.provider}")
            print(f"Base URL: {self.base_url or 'default'}")
            print(f"Seed: {self.seed}")
            print(f"Snapshot Interval: every {self.snapshot_interval} days")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        current_day = start_day - 1
        game_ended = False
        game_outcome = None
        last_result = None

        for day in range(start_day, self.total_days + 1):
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

            # Build dashboard
            dashboard = self._build_dashboard(day, last_result)

            # Log the dashboard to tool_results JSONL so external monitors can read the exact agent view
            self._log_tool_result(0, day, '_dashboard', {}, dashboard)

            # Agent loop for this day
            observation = dashboard
            info = {'day': day, 'cash': get_cash(self.conn)}
            turns_today = 0
            day_ended = False

            while not day_ended and turns_today < 100:
                turns_today += 1

                # Get action from agent
                action = self.agent.act(observation, 0, False, info)

                if action is None:
                    # No action, force next_day
                    action = Action(tool='next_day')

                # Execute action
                if action.tool == 'next_day':
                    day_ended = True
                    observation = "Day ended. Moving to next day..."
                    # Log next_day tool result
                    self._log_tool_result(self.agent.total_turns, day, 'next_day', {}, observation)
                    if verbose:
                        print(f"    [Turn {turns_today}] next_day")
                else:
                    # Execute tool by calling the appropriate method on AgentTools
                    if verbose:
                        print(f"    [Turn {turns_today}] {action.tool}({action.arguments})")
                    result = self._execute_tool(action.tool, action.arguments or {})
                    observation = result if isinstance(result, str) else json.dumps(result)
                    # Log tool result
                    self._log_tool_result(self.agent.total_turns, day, action.tool, action.arguments or {}, observation)
                    if verbose:
                        print(f"      → {observation}")

                info = {'day': day, 'cash': get_cash(self.conn)}

            # Run simulation step (with timeout detection)
            import time as _time
            _step_start = _time.monotonic()
            day_result = self.simulator.step_day()
            _step_elapsed = _time.monotonic() - _step_start
            last_result = day_result

            if _step_elapsed > 300:
                # step_day took >5 minutes — save checkpoint and quit
                print(f"\n⚠️  step_day took {_step_elapsed:.1f}s on day {day} (>300s threshold)")
                print(f"Auto-quitting to prevent runaway timeouts. Saving checkpoint...")
                self._save_checkpoint(day)
                self.event_logger.save_incremental()
                game_ended = True
                game_outcome = 'timeout'
                break

            # Log daily state
            self.event_logger.log_daily_state(
                cash=day_result.cash,
                mrr=day_result.mrr,
                subscribers=get_active_subscriber_count(self.conn),
                usage=day_result.total_usage,
                overload=day_result.overload,
                outage=day_result.outage,
                group_reputations=get_all_group_reputations(self.conn),
                group_awareness=get_all_group_awareness(self.conn),
                total_dividends=day_result.total_dividends_paid,
                founder_dividends=day_result.founder_cumulative_dividends,
            )

            if verbose:
                print(f"  📊 End of day: Cash=${day_result.cash:,.0f}, IndSubs={day_result.total_individual_subscribers}, EntSeats={day_result.total_enterprise_subscription_seats}")

            # Save checkpoint after each completed day
            self._save_checkpoint(day)
            self.event_logger.save_incremental()

            # Check for bankruptcy
            if self.simulator.shutdown_mode:
                game_ended = True
                game_outcome = 'bankrupt'
                if verbose:
                    print(f"\n💀 BANKRUPT at day {day}!")
                break

        # Determine final outcome
        if not game_outcome:
            game_outcome = 'completed'

        # Finalize
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

    parser = argparse.ArgumentParser(description="Run baseline LLM agent for SaaS Bench")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "xai", "anthropic", "bedrock", "modal"], help="API provider")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--days", type=int, default=3650, help="Total days to simulate (10 years)")
    parser.add_argument("--workspace", type=Path, help="Workspace base directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for GPT-5.2+ models")
    parser.add_argument("--continue-from", type=Path,
                        help="Path to a previous run directory to resume from (e.g., baseline_runs/run_e87347ea)")
    parser.add_argument("--restart-from-day", type=int,
                        help="Fork a run from this day number (requires --continue-from to specify source run)")
    parser.add_argument("--snapshot-interval", type=int, default=50,
                        help="Save DB snapshots every N days (default: 50)")

    args = parser.parse_args()

    runner = BaselineRunner(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        seed=args.seed,
        scenario=args.scenario,
        total_days=args.days,
        workspace_base=args.workspace,
        reasoning_effort=args.reasoning_effort,
        continue_from=args.continue_from,
        restart_from_day=args.restart_from_day,
        snapshot_interval=args.snapshot_interval,
    )

    result = runner.run(verbose=not args.quiet)

    print(f"\nResult: {result['outcome']}")
    print(f"Final Cash: ${result['final_cash']:,.0f}")
    print(f"Workspace: {result['workspace_dir']}")


if __name__ == "__main__":
    main()
