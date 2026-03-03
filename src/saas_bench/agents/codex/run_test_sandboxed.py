#!/usr/bin/env python3
"""Test runner for Codex agent with bubblewrap sandbox and MCP tools.

Uses bwrap to restrict Codex to only read/write within the agent workspace.
MCP server runs OUTSIDE the sandbox to provide tools.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

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
    return datetime.now(timezone.utc).isoformat()


def build_bwrap_command(
    workspace_dir: Path,
    inner_cmd: List[str],
    env: dict,
    saas_bench_root: Optional[Path] = None,
) -> List[str]:
    """Build a bubblewrap command that sandboxes filesystem access.

    The sandbox:
    - Read-only access to system directories (for python, codex, libs)
    - Read-write access ONLY to workspace_dir
    - Network access allowed (for MCP communication)
    """
    # Get home directory for codex auth
    home_dir = Path.home()
    codex_home = home_dir / ".codex"

    # Get nvm/node directories if they exist
    nvm_dir = home_dir / ".nvm"

    bwrap_cmd = [
        "bwrap",
        # Mount system directories read-only
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", "/opt", "/opt",
        # Device and proc
        "--dev", "/dev",
        "--proc", "/proc",
        # Tmp directory (some tools need it)
        "--tmpfs", "/tmp",
        # Mount workspace with read-write access
        "--bind", str(workspace_dir), str(workspace_dir),
        # Mount codex home directory with read-write (for session storage)
        "--bind", str(codex_home), str(codex_home),
        # Set working directory
        "--chdir", str(workspace_dir / "agent"),
    ]

    # Mount nvm if it exists (for node/codex)
    if nvm_dir.exists():
        bwrap_cmd.extend(["--ro-bind", str(nvm_dir), str(nvm_dir)])

    # Mount Python and related directories (read-only)
    python_prefix = sys.prefix
    bwrap_cmd.extend(["--ro-bind", python_prefix, python_prefix])

    # Mount conda/miniconda if used (read-only)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        bwrap_cmd.extend(["--ro-bind", conda_prefix, conda_prefix])

    # Mount saas_bench package root for MCP server access (read-only)
    # This allows Codex to spawn the MCP server which needs to import saas_bench
    if saas_bench_root and saas_bench_root.exists():
        bwrap_cmd.extend(["--ro-bind", str(saas_bench_root), str(saas_bench_root)])

    # Add environment variables
    for key, value in env.items():
        bwrap_cmd.extend(["--setenv", key, value])

    # Add the inner command
    bwrap_cmd.extend(inner_cmd)

    return bwrap_cmd


def run_sandboxed_codex(
    cmd: list,
    workspace_dir: Path,
    env: dict,
    saas_bench_root: Optional[Path] = None,
    verbose: bool = True,
    timeout: int = 3600  # 1 hour
) -> Tuple[int, str, str]:
    """Run Codex inside bubblewrap sandbox."""
    try:
        bwrap_cmd = build_bwrap_command(workspace_dir, cmd, env, saas_bench_root)

        if verbose:
            print(f"  Sandbox command: bwrap ... {' '.join(cmd[:4])}...")

        result = subprocess.run(
            bwrap_cmd,
            capture_output=True,
            text=True,
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
    reasoning_effort: str = "medium",
    workspace_base: Optional[Path] = None,
    verbose: bool = True
):
    """Run a sandboxed test simulation with Codex using MCP tools.

    Args:
        total_days: Number of days to simulate
        seed: Random seed
        scenario: Scenario name
        model: Codex model to use
        reasoning_effort: Reasoning effort level (low, medium, high)
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
        print(f"SaaS Bench - Codex Test (Sandboxed + MCP)")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Model: {model}")
        print(f"Reasoning Effort: {reasoning_effort}")
        print(f"Seed: {seed}")
        print(f"Days: {total_days}")
        print(f"Workspace: {workspace_dir}")
        print(f"Sandbox: bubblewrap (filesystem isolation)")
        print(f"Tools: MCP server (runs outside sandbox)")
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
            'agent_type': 'codex_sandboxed',
        }
    )

    # Connect event logger
    simulator.set_event_logger(event_logger)
    tools.set_event_logger(event_logger)

    # Initialize simulation
    simulator.initialize()
    event_logger.log_run_start()

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

    # Set up MCP server using `codex mcp add` with environment variables
    # MCP server runs OUTSIDE sandbox, so it can access saas_bench
    mcp_server_path = Path(__file__).parent.parent / "claude_code" / "serve_mcp.py"
    python_path = sys.executable

    # Remove any existing saas-bench MCP server entry
    subprocess.run(["codex", "mcp", "remove", "saas-bench"], capture_output=True)

    # Add MCP server with env vars for this run
    mcp_add_cmd = [
        "codex", "mcp", "add", "saas-bench",
        "--env", f"SAAS_BENCH_WORKSPACE={workspace_dir}",
        "--env", f"SAAS_BENCH_RUN_ID={run_id}",
        "--env", f"SAAS_BENCH_DB_PATH={db_path}",
        "--env", f"SAAS_BENCH_TOTAL_DAYS={total_days}",
        "--env", f"SAAS_BENCH_SEED={seed}",
        "--env", f"SAAS_BENCH_SCENARIO={scenario}",
        "--env", f"PYTHONPATH={package_root}",
        "--",
        python_path, str(mcp_server_path)
    ]
    result = subprocess.run(mcp_add_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Failed to add MCP server: {result.stderr}")

    if verbose:
        print(f"MCP server registered with codex mcp add")

    # Load AGENTS.md template from shared file
    template_file = Path(__file__).parent.parent / "agent_template.md"
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Load simulator instructions (tool_list filled dynamically from TOOL_DOCS)
    from saas_bench.tools import get_tool_summary_table
    simulator_instructions_file = Path(__file__).parent.parent / "simulator_instructions.md"
    with open(simulator_instructions_file, 'r') as f:
        simulator_instructions = f.read().format(tool_list=get_tool_summary_table())

    # Format the template with run-specific values
    system_prompt = template_content.format(
        total_days=total_days,
        run_id=run_id,
        model=model,
        initial_cash=bench_config.initial_cash,
        agent_workspace=str(agent_workspace),
        simulator_instructions=simulator_instructions
    )
    (agent_workspace / "AGENTS.md").write_text(system_prompt)

    if verbose:
        print(f"AGENTS.md written to: {agent_workspace / 'AGENTS.md'}")

    # Build prompt for Codex
    prompt = f"""You are the COO of NovaMind AI. Your goal is to maximize cash by day {total_days}.

{dashboard}

INSTRUCTIONS:
1. Review the dashboard above
2. Use MCP tools to adjust pricing, spending, capacity, etc.
3. Use log_rationale to record your thinking
4. Call next_day when done - it will run the simulation and return the next day's dashboard
5. Continue until you complete all {total_days} days or go bankrupt

The next_day tool returns the dashboard for the following day, so keep calling it to progress through the simulation.

Available MCP tools: set_prices, set_model_tiers, set_daily_spend, set_ad_channel_spend, set_targeted_ad_spend, set_capacity_tier, set_usage_quotas, python_exec, log_rationale, next_day, and more.

Start by analyzing Day 1 and making your first decisions!"""

    # Environment for Codex (inside sandbox)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "SAAS_BENCH_DAY": "1",
    }

    # Build Codex command
    codex_cmd = [
        "codex",
        "exec",
        prompt,
        "--json",
        "--full-auto",
        "--skip-git-repo-check",
        "--model", model,
    ]

    if verbose:
        print(f"\nRunning sandboxed Codex with model {model} and MCP tools...")

    # Run Codex in sandbox
    agent_log_file = logs_dir / f"agent_conversation_{run_id}.jsonl"
    iteration = 0
    max_iterations = 50

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

        # Build prompt
        if iteration == 1:
            current_prompt = prompt
        else:
            current_prompt = f"Continue managing NovaMind AI. Current day: {current_day}, Cash: ${current_cash:,.0f}. Keep calling next_day until you reach day {total_days} or go bankrupt."

        codex_cmd = [
            "codex",
            "exec",
            current_prompt,
            "--json",
            "--full-auto",
            "--skip-git-repo-check",
            "--model", model,
            "-c", f'reasoning_effort="{reasoning_effort}"',
        ]

        if verbose:
            if iteration == 1:
                print(f"\n  [Iteration {iteration}] Starting Codex session...")
            else:
                print(f"\n  [Iteration {iteration}] Continuing... (Day {current_day}, ${current_cash:,.0f})")

        # Update env with current day
        env["SAAS_BENCH_DAY"] = str(current_day)

        returncode, stdout, stderr = run_sandboxed_codex(
            cmd=codex_cmd,
            workspace_dir=workspace_dir,
            env=env,
            saas_bench_root=package_root,
            verbose=verbose,
            timeout=3600  # 1 hour
        )

        # Log result
        with open(agent_log_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": now(),
                "iteration": iteration,
                "current_day": current_day,
                "returncode": returncode,
                "stdout_preview": stdout[:2000] if stdout else None,
                "stderr_preview": stderr[:1000] if stderr else None,
            }) + "\n")

        if returncode != 0:
            if verbose:
                print(f"  ⚠️ Codex exited with code {returncode}")
                if stderr:
                    print(f"     stderr: {stderr[:300]}")
            # Check if we made progress
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    new_day = state.get('current_day', 1)
                if new_day > current_day:
                    if verbose:
                        print(f"     But made progress to day {new_day}, continuing...")
                    continue
            break
        else:
            if verbose:
                print(f"  ✓ Iteration completed")

    # Read final state
    final_day = 1
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            final_day = state.get('current_day', 1)

    final_cash = get_cash(conn)
    outcome = 'completed' if final_day >= total_days else ('bankrupt' if final_cash <= 0 else 'incomplete')

    # Log run end
    event_logger.log_run_end(final_cash, final_day, outcome)
    event_logger.save()

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Final Day: {final_day}")
        print(f"Final Cash: ${final_cash:,.0f}")
        print(f"Outcome: {outcome}")
        print(f"Iterations: {iteration}")
        print(f"Log file: {event_logger.log_file}")
        print(f"{'='*60}\n")

    conn.close()

    return {
        'run_id': run_id,
        'final_day': final_day,
        'final_cash': final_cash,
        'outcome': outcome,
        'iterations': iteration,
        'workspace': str(workspace_dir),
        'log_file': str(event_logger.log_file)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sandboxed Codex agent test with MCP")
    parser.add_argument("--days", type=int, default=10, help="Number of days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scenario", default="default", help="Scenario name")
    parser.add_argument("--model", default="gpt-5.2", help="Codex model")
    parser.add_argument("--reasoning-effort", default="medium",
                       choices=["low", "medium", "high"],
                       help="Reasoning effort level")
    parser.add_argument("--workspace", type=Path, help="Workspace directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    result = run_test(
        total_days=args.days,
        seed=args.seed,
        scenario=args.scenario,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        workspace_base=args.workspace,
        verbose=not args.quiet
    )

    print(f"\nResult: {json.dumps(result, indent=2)}")
