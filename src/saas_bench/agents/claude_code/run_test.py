#!/usr/bin/env python3
"""Test runner for Claude Code agent with SaaS Bench.

This script runs a test simulation using Claude Code in headless mode.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
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
from saas_bench.tools import AgentTools
from saas_bench.shocks import ShockManager
from saas_bench.event_logger import EventLogger


def now() -> str:
    """Get current UTC timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def extract_session_id(output: str) -> Optional[str]:
    """Extract session_id from Claude Code JSON output."""
    match = re.search(r'"session_id"\s*:\s*"([^"]+)"', output)
    if match:
        return match.group(1)
    return None


def run_claude_command(
    cmd: list,
    cwd: str,
    env: dict,
    verbose: bool = True
) -> Tuple[int, str, str]:
    """Run a Claude command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -2, "", str(e)


def run_test(
    total_days: int = 10,
    seed: int = 42,
    scenario: str = "default",
    model: str = "claude-sonnet-4-20250514",
    workspace_base: Optional[Path] = None,
    verbose: bool = True
):
    """Run a test simulation with Claude Code.

    Args:
        total_days: Number of days to simulate
        seed: Random seed
        scenario: Scenario name
        model: Claude model to use
        workspace_base: Base directory for workspace
        verbose: Print progress
    """
    # Generate run ID
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Set up workspace (ensure absolute path)
    workspace_base = (workspace_base or Path("/tmp/saas_bench_test")).resolve()
    workspace_dir = workspace_base / f"run_{run_id}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    agent_workspace = workspace_dir / "agent"
    agent_workspace.mkdir(exist_ok=True)

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    db_path = workspace_dir / "world.db"

    if verbose:
        print(f"\n{'='*60}")
        print(f"SaaS Bench - Claude Code Test Run")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Model: {model}")
        print(f"Seed: {seed}")
        print(f"Scenario: {scenario}")
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
    shock_manager = ShockManager(conn, rng, scenario_pack)
    tools = AgentTools(conn, 0, agent_workspace, db_path)

    # Initialize event logger
    event_logger = EventLogger(
        run_id=run_id,
        output_dir=logs_dir,
        seed=seed,
        scenario=scenario,
        config={
            'model': model,
            'seed': seed,
            'scenario': scenario,
            'total_days': total_days,
        }
    )

    # Connect event logger
    simulator.set_event_logger(event_logger)
    tools.set_event_logger(event_logger)

    # Initialize simulation
    simulator.initialize()
    event_logger.log_run_start()

    # Load simulator instructions (tool_list filled dynamically from TOOL_DOCS)
    from saas_bench.tools import get_tool_summary_table
    simulator_file = Path(__file__).parent.parent / "simulator_instructions.md"
    with open(simulator_file, 'r') as f:
        simulator_instructions = f.read().format(tool_list=get_tool_summary_table())

    # Load CLAUDE.md template from shared file
    template_file = Path(__file__).parent / "agent_claude_template.md"
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Format the template with run-specific values
    system_prompt = template_content.format(
        total_days=total_days,
        run_id=run_id,
        model=model,
        initial_cash=bench_config.initial_cash,
        agent_workspace=str(agent_workspace),
        simulator_instructions=simulator_instructions
    )
    (agent_workspace / "CLAUDE.md").write_text(system_prompt)

    # Create MCP config for Claude Code
    mcp_server_path = Path(__file__).parent / "serve_mcp.py"
    python_path = sys.executable

    mcp_config = {
        "mcpServers": {
            "saas-bench": {
                "command": python_path,
                "args": [str(mcp_server_path)],
                "env": {
                    "SAAS_BENCH_WORKSPACE": str(workspace_dir),
                    "SAAS_BENCH_RUN_ID": run_id,
                    "SAAS_BENCH_DB_PATH": str(db_path),
                    "SAAS_BENCH_TOTAL_DAYS": str(total_days),
                    "SAAS_BENCH_SEED": str(seed),
                    "SAAS_BENCH_SCENARIO": scenario,
                    "PYTHONPATH": str(package_root)
                }
            }
        }
    }

    mcp_config_path = workspace_dir / "mcp_config.json"
    with open(mcp_config_path, 'w') as f:
        json.dump(mcp_config, f, indent=2)

    if verbose:
        print(f"MCP config written to: {mcp_config_path}")

    # State file for MCP server communication
    state_file = workspace_dir / ".mcp_state.json"

    # Initialize state file
    with open(state_file, 'w') as f:
        json.dump({
            'current_day': 1,
            'day_ended': False,
            'last_updated': now()
        }, f)

    # Build Day 1 dashboard
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

    # Build single prompt for entire simulation
    prompt = f"""You are the COO of NovaMind AI. Your goal is to maximize cash by day {total_days}.

{dashboard}

INSTRUCTIONS:
1. Review the dashboard above
2. Use tools to adjust pricing, spending, capacity, etc.
3. Use log_rationale to record your thinking
4. Call next_day when done - it will run the simulation and return the next day's dashboard
5. Continue until you complete all {total_days} days or go bankrupt

The next_day tool returns the dashboard for the following day, so keep calling it to progress through the simulation.

Available tools: set_prices, set_model_tiers, set_daily_spend, set_ad_channel_spend, set_targeted_ad_spend, set_capacity_tier, set_usage_quotas, python_exec, log_rationale, next_day, and more.

Start by analyzing Day 1 and making your first decisions!"""

    # Run Claude Code headless with session resume loop
    # The agent will run until it reaches total_days or goes bankrupt
    # If Claude exits early (context limit), we resume the session

    session_id = None
    agent_log_file = logs_dir / f"agent_conversation_{run_id}.jsonl"
    iteration = 0
    max_iterations = 50  # Safety limit to prevent infinite loops

    if verbose:
        print(f"\n  Running Claude Code for {total_days}-day simulation...")

    while iteration < max_iterations:
        iteration += 1

        # Read current state
        current_day = 1
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                current_day = state.get('current_day', 1)

        current_cash = get_cash(conn)

        # Check if game is over
        if current_day >= total_days:
            if verbose:
                print(f"  ✅ Completed all {total_days} days!")
            break
        if current_cash <= 0:
            if verbose:
                print(f"  💀 Bankrupt at day {current_day}!")
            break

        # Build command - use full path to claude binary
        claude_bin = os.environ.get("CLAUDE_BIN", "/home/hc5019/.local/bin/claude")
        if session_id:
            # Resume existing session
            resume_prompt = f"Continue managing NovaMind AI. Current day: {current_day}, Cash: ${current_cash:,.0f}. Keep calling next_day until you reach day {total_days} or go bankrupt."
            cmd = [
                claude_bin,
                "-p", resume_prompt,
                "--output-format", "json",
                "--model", model,
                "--mcp-config", str(mcp_config_path),
                "--allowedTools", "*",
                "--resume", session_id,
            ]
            if verbose:
                print(f"\n  [Iteration {iteration}] Resuming session {session_id[:8]}... (Day {current_day}, ${current_cash:,.0f})")
        else:
            # First call - no resume
            cmd = [
                claude_bin,
                "-p", prompt,
                "--output-format", "json",
                "--model", model,
                "--mcp-config", str(mcp_config_path),
                "--allowedTools", "*",
            ]
            if verbose:
                print(f"\n  [Iteration {iteration}] Starting new session...")

        # Run Claude (no timeout - let it run until completion or context limit)
        returncode, stdout, stderr = run_claude_command(
            cmd=cmd,
            cwd=str(agent_workspace),
            env={**os.environ, "SAAS_BENCH_DAY": str(current_day)},
            verbose=verbose
        )

        # Log result
        with open(agent_log_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": now(),
                "iteration": iteration,
                "session_id": session_id,
                "current_day": current_day,
                "returncode": returncode,
                "stdout_preview": stdout[:2000] if stdout else None,
                "stderr_preview": stderr[:1000] if stderr else None,
            }) + "\n")

        # Extract session_id from output if we don't have one
        if not session_id and stdout:
            session_id = extract_session_id(stdout)
            if session_id and verbose:
                print(f"  📍 Captured session ID: {session_id[:8]}...")

        # Check result
        if returncode != 0:
            if verbose:
                print(f"  ⚠️ Claude exited with code {returncode}, will resume...")
                if stderr:
                    print(f"     stderr: {stderr[:200]}")
            continue
        else:
            # Success - but check if game is over
            if verbose:
                try:
                    output = json.loads(stdout)
                    print(f"  ✓ Iteration completed: {str(output.get('result', ''))[:200]}")
                except:
                    print(f"  ✓ Iteration completed")

            # Check state again after successful run
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    current_day = state.get('current_day', 1)
            current_cash = get_cash(conn)

            if current_day >= total_days or current_cash <= 0:
                break  # Game over

            # Continue to next iteration
            if verbose:
                print(f"  🔄 Game not over yet (Day {current_day}/{total_days}, ${current_cash:,.0f}), resuming...")

    if iteration >= max_iterations:
        if verbose:
            print(f"  ⚠️ Reached maximum iterations ({max_iterations})")

    # Read final state from MCP server
    final_day = 1
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            final_day = state.get('current_day', 1)

    final_cash = get_cash(conn)

    # Final outcome
    outcome = 'completed' if final_day >= total_days else ('bankrupt' if final_cash <= 0 else 'incomplete')

    # Log run end
    event_logger.log_run_end(final_cash, final_day, outcome)
    event_logger.save()

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Final Cash: ${final_cash:,.0f}")
        print(f"Days Run: {final_day}")
        print(f"Outcome: {outcome}")
        print(f"Log file: {event_logger.log_file}")
        print(f"{'='*60}\n")

    return {
        'run_id': run_id,
        'final_cash': final_cash,
        'days_run': final_day,
        'outcome': outcome,
        'workspace': str(workspace_dir),
        'log_file': str(event_logger.log_file)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Claude Code agent test")
    parser.add_argument("--days", type=int, default=10, help="Number of days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model")
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
