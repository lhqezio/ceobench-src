#!/usr/bin/env python3
"""Test runner for Codex agent WITHOUT MCP - uses CLI tools instead.

This is a simpler approach that lets Codex call bash commands directly.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from numpy.random import Generator, PCG64

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from saas_bench.config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from saas_bench.database import init_database, get_cash, get_active_subscriber_count, get_config
from saas_bench.simulation import Simulator
from saas_bench.shocks import ShockManager
from saas_bench.event_logger import EventLogger


def now() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def run_codex_command(
    cmd: list,
    cwd: str,
    env: dict,
    verbose: bool = True,
    timeout: int = 300  # 5 minute timeout
) -> Tuple[int, str, str]:
    """Run a Codex command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -2, "", str(e)


def run_test(
    total_days: int = 10,
    seed: int = 42,
    scenario: str = "default",
    model: str = "gpt-5.2",
    workspace_base: Optional[Path] = None,
    verbose: bool = True
):
    """Run a test simulation with Codex using CLI tools (no MCP).

    Args:
        total_days: Number of days to simulate
        seed: Random seed
        scenario: Scenario name
        model: Codex model to use
        workspace_base: Base directory for workspace
        verbose: Print progress
    """
    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Set up workspace - default to results directory in saas-bench repo
    default_results_dir = Path(__file__).parent.parent.parent.parent.parent / "results" / "codex-runs"
    workspace_base = (workspace_base or default_results_dir).resolve()
    workspace_dir = workspace_base / f"run_{run_id}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    agent_workspace = workspace_dir / "agent"
    agent_workspace.mkdir(exist_ok=True)

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    db_path = workspace_dir / "world.db"

    if verbose:
        print(f"\n{'='*60}")
        print(f"SaaS Bench - Codex Test (No MCP)")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Model: {model}")
        print(f"Seed: {seed}")
        print(f"Days: {total_days}")
        print(f"Workspace: {workspace_dir}")
        print(f"{'='*60}\n")

    # Initialize RNG
    rng = Generator(PCG64(seed))

    # Get scenario
    scenario_pack = SCENARIO_PACKS.get(scenario, ScenarioPack(
        name='Default',
        description='Balanced scenario'
    ))

    # Create benchmark config
    bench_config = BenchmarkConfig(
        seed=seed,
        total_days=total_days,
        initial_cash=1_000_000.0,
    )

    # Initialize database
    conn = init_database(db_path)

    # Initialize components
    simulator = Simulator(conn, bench_config, rng)
    simulator.initialize()

    # Initialize state file
    state_file = workspace_dir / ".mcp_state.json"
    with open(state_file, 'w') as f:
        json.dump({
            'current_day': 1,
            'day_ended': False,
            'last_updated': now()
        }, f)

    # Get initial dashboard
    cash = get_cash(conn)
    subscribers = get_active_subscriber_count(conn)
    config = get_config(conn, 1)

    dashboard_lines = [
        "=== DAY 1 DASHBOARD ===",
        "",
        f"CASH: ${cash:,.0f}",
        f"SUBSCRIBERS: {subscribers}",
        "",
    ]
    if config:
        dashboard_lines.extend([
            "CURRENT CONFIG:",
            f"  - Prices: A=${config['price_A']}, B=${config['price_B']}, C=${config['price_C']}",
            f"  - Model tiers: A={config['tier_A']}, B={config['tier_B']}, C={config['tier_C']}",
            f"  - Daily spend: ads=${config['spend_advertising']}, ops=${config['spend_operations']}, dev=${config['spend_development']}",
            f"  - Capacity tier: {config['capacity_tier']}",
        ])
    dashboard_lines.extend(["", "========================="])
    dashboard = "\n".join(dashboard_lines)

    if verbose:
        print(dashboard)

    # Path to CLI tool
    cli_tool = Path(__file__).parent / "cli_tool.py"
    python_path = sys.executable

    # Create AGENTS.md with CLI instructions
    agents_md = f"""# SaaS Bench - CLI Agent Instructions

## Role
You are the COO of NovaMind AI. Maximize cash by day {total_days}.

## Current Status
{dashboard}

## Available CLI Commands

Run these bash commands to control the simulation:

```bash
# Check current status
python {cli_tool} status

# Advance to next day (REQUIRED to progress)
python {cli_tool} next_day

# Set prices for plans A, B, C
python {cli_tool} set_prices --A 29 --B 79 --C 199

# Set daily spending
python {cli_tool} set_daily_spend --ads 1000 --ops 500 --dev 500

# Set capacity tier (0-3)
python {cli_tool} set_capacity --tier 1

# Log your reasoning
python {cli_tool} log_rationale "My strategy is..."
```

## Strategy Tips
- Start with marketing spend to get customers
- Balance quality (model tiers) with costs
- Monitor cash carefully - bankruptcy ends the game

## Your Task
1. Analyze the current state
2. Make strategic decisions using the CLI commands above
3. Call `next_day` to advance
4. Repeat until day {total_days} or bankruptcy

START NOW - use the bash commands above!
"""

    (agent_workspace / "AGENTS.md").write_text(agents_md)

    # Build prompt for Codex
    prompt = f"""You are playing a business simulation game. You have {total_days} days to maximize cash.

IMPORTANT: Use bash commands to interact with the game. The CLI tool is at:
{cli_tool}

Example commands:
python {cli_tool} status
python {cli_tool} set_daily_spend --ads 5000 --ops 1000 --dev 1000
python {cli_tool} next_day

Current status:
{dashboard}

Your goal: Maximize cash by day {total_days}.

Strategy suggestion: Start by setting advertising spend to attract customers, then call next_day repeatedly.

BEGIN NOW - run commands to play the game!"""

    # Environment for Codex
    env = {
        **os.environ,
        "SAAS_BENCH_WORKSPACE": str(workspace_dir),
        "SAAS_BENCH_DB_PATH": str(db_path),
        "SAAS_BENCH_RUN_ID": run_id,
        "SAAS_BENCH_TOTAL_DAYS": str(total_days),
        "SAAS_BENCH_SEED": str(seed),
        "SAAS_BENCH_SCENARIO": scenario,
        "PYTHONPATH": str(package_root),
    }

    # Build Codex command
    cmd = [
        "codex",
        "exec",
        prompt,
        "--json",
        "--full-auto",
        "--skip-git-repo-check",
        "--model", model,
    ]

    if verbose:
        print(f"\nRunning Codex with model {model}...")
        print(f"Command: {' '.join(cmd[:6])}...")

    # Run Codex
    agent_log_file = logs_dir / f"agent_conversation_{run_id}.jsonl"

    returncode, stdout, stderr = run_codex_command(
        cmd=cmd,
        cwd=str(agent_workspace),
        env=env,
        verbose=verbose,
        timeout=600  # 10 minute timeout
    )

    # Log result
    with open(agent_log_file, 'w') as f:
        f.write(json.dumps({
            "timestamp": now(),
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }) + "\n")

    # Check final state
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            final_day = state.get('current_day', 1)
    else:
        final_day = 1

    final_cash = get_cash(conn)
    outcome = 'completed' if final_day >= total_days else ('bankrupt' if final_cash <= 0 else 'incomplete')

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Return code: {returncode}")
        print(f"Final Day: {final_day}")
        print(f"Final Cash: ${final_cash:,.0f}")
        print(f"Outcome: {outcome}")
        if stderr:
            print(f"Stderr: {stderr[:500]}")
        print(f"{'='*60}\n")

    conn.close()

    return {
        'run_id': run_id,
        'final_day': final_day,
        'final_cash': final_cash,
        'outcome': outcome,
        'returncode': returncode,
        'workspace': str(workspace_dir),
        'stdout_preview': stdout[:2000] if stdout else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Codex agent test (no MCP)")
    parser.add_argument("--days", type=int, default=10, help="Number of days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--model", default="gpt-5.2", help="Codex model")
    parser.add_argument("--workspace", type=Path, help="Workspace directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    result = run_test(
        total_days=args.days,
        seed=args.seed,
        scenario=args.scenario,
        model=args.model,
        workspace_base=args.workspace,
        verbose=not args.quiet
    )

    print(f"\nResult: {json.dumps(result, indent=2)}")
