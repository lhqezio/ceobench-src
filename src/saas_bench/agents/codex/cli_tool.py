#!/usr/bin/env python3
"""CLI tool for SaaS Bench - allows Codex to interact via bash commands.

Usage:
    saas-bench-cli next_day
    saas-bench-cli set_prices --A 29 --B 79 --C 199
    saas-bench-cli set_daily_spend --ads 1000 --ops 500 --dev 500
    saas-bench-cli log_rationale "My reasoning here"
    saas-bench-cli status
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))


def get_env_config():
    """Get configuration from environment variables."""
    workspace = os.environ.get('SAAS_BENCH_WORKSPACE')
    db_path = os.environ.get('SAAS_BENCH_DB_PATH')
    run_id = os.environ.get('SAAS_BENCH_RUN_ID')
    total_days = int(os.environ.get('SAAS_BENCH_TOTAL_DAYS', 3650))
    seed = int(os.environ.get('SAAS_BENCH_SEED', 42))

    if not workspace or not db_path:
        print("Error: SAAS_BENCH_WORKSPACE and SAAS_BENCH_DB_PATH must be set", file=sys.stderr)
        sys.exit(1)

    return {
        'workspace': Path(workspace),
        'db_path': Path(db_path),
        'run_id': run_id,
        'total_days': total_days,
        'seed': seed,
    }


def load_state(workspace: Path):
    """Load current state from state file."""
    state_file = workspace / ".mcp_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {'current_day': 1, 'day_ended': False}


def save_state(workspace: Path, state: dict):
    """Save state to state file."""
    state_file = workspace / ".mcp_state.json"
    state['last_updated'] = datetime.now(timezone.utc).isoformat()
    with open(state_file, 'w') as f:
        json.dump(state, f)


def cmd_status(args):
    """Show current status/dashboard."""
    from saas_bench.database import get_cash, get_active_subscriber_count, get_config
    import sqlite3

    config = get_env_config()
    state = load_state(config['workspace'])

    conn = sqlite3.connect(config['db_path'])
    conn.row_factory = sqlite3.Row

    current_day = state.get('current_day', 1)
    cash = get_cash(conn)
    subscribers = get_active_subscriber_count(conn)
    cfg = get_config(conn, current_day)

    print(f"=== DAY {current_day} DASHBOARD ===")
    print()
    print(f"CASH: ${cash:,.0f}")
    print(f"SUBSCRIBERS: {subscribers}")
    print()
    if cfg:
        print("CURRENT CONFIG:")
        print(f"  - Prices: A=${cfg['price_A']}, B=${cfg['price_B']}, C=${cfg['price_C']}")
        print(f"  - Model tiers: A={cfg['tier_A']}, B={cfg['tier_B']}, C={cfg['tier_C']}")
        print(f"  - Daily spend: ads=${cfg['spend_advertising']}, ops=${cfg['spend_operations']}, dev=${cfg['spend_development']}")
        print(f"  - Capacity tier: {cfg['capacity_tier']}")
    print()
    print(f"Total days: {config['total_days']}")
    print("=========================")

    conn.close()


def cmd_next_day(args):
    """Advance to next day and show new dashboard."""
    from saas_bench.database import get_cash, get_active_subscriber_count, get_config, init_database
    from saas_bench.simulation import Simulator
    from saas_bench.shocks import ShockManager
    from saas_bench.config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
    from numpy.random import Generator, PCG64
    import sqlite3

    config = get_env_config()
    state = load_state(config['workspace'])
    current_day = state.get('current_day', 1)

    # Connect to database
    conn = sqlite3.connect(config['db_path'])
    conn.row_factory = sqlite3.Row

    # Check if game over
    cash = get_cash(conn)
    if cash <= 0:
        print(f"GAME OVER - Bankrupt on day {current_day}")
        conn.close()
        return

    if current_day > config['total_days']:
        print(f"GAME COMPLETE - Finished all {config['total_days']} days!")
        print(f"Final cash: ${cash:,.0f}")
        conn.close()
        return

    # Initialize simulator
    rng = Generator(PCG64(config['seed'] + current_day))
    scenario = os.environ.get('SAAS_BENCH_SCENARIO', 'default')
    scenario_pack = SCENARIO_PACKS.get(scenario, ScenarioPack(name='Default', description='Balanced'))

    bench_config = BenchmarkConfig(
        seed=config['seed'],
        total_days=config['total_days'],
        initial_cash=1_000_000.0,
    )

    simulator = Simulator(conn, bench_config, rng)
    shock_manager = ShockManager(conn, rng, scenario_pack)

    # Run one day
    simulator.run_day(current_day)
    shocks = shock_manager.process_day(current_day)

    # Update state
    new_day = current_day + 1
    state['current_day'] = new_day
    state['day_ended'] = True
    save_state(config['workspace'], state)

    # Show new dashboard
    cash = get_cash(conn)
    subscribers = get_active_subscriber_count(conn)
    cfg = get_config(conn, new_day)

    print(f"=== DAY {new_day} DASHBOARD ===")
    print()
    print(f"CASH: ${cash:,.0f}")
    print(f"SUBSCRIBERS: {subscribers}")
    print()
    if cfg:
        print("CURRENT CONFIG:")
        print(f"  - Prices: A=${cfg['price_A']}, B=${cfg['price_B']}, C=${cfg['price_C']}")
        print(f"  - Model tiers: A={cfg['tier_A']}, B={cfg['tier_B']}, C={cfg['tier_C']}")
        print(f"  - Daily spend: ads=${cfg['spend_advertising']}, ops=${cfg['spend_operations']}, dev=${cfg['spend_development']}")
        print(f"  - Capacity tier: {cfg['capacity_tier']}")
    print()
    if shocks:
        print(f"SHOCKS: {len(shocks)} events occurred")
        for shock in shocks:
            print(f"  - {shock}")
    print(f"Days remaining: {config['total_days'] - new_day + 1}")
    print("=========================")

    conn.close()


def cmd_set_prices(args):
    """Set prices for plans A, B, C."""
    from saas_bench.tools import AgentTools
    import sqlite3

    config = get_env_config()
    state = load_state(config['workspace'])

    conn = sqlite3.connect(config['db_path'])
    conn.row_factory = sqlite3.Row

    tools = AgentTools(conn, state.get('current_day', 1), config['workspace'] / 'agent', config['db_path'])

    result = tools.set_prices({'A': args.A, 'B': args.B, 'C': args.C})
    print(result.message)

    conn.close()


def cmd_set_daily_spend(args):
    """Set daily spending."""
    from saas_bench.tools import AgentTools
    import sqlite3

    config = get_env_config()
    state = load_state(config['workspace'])

    conn = sqlite3.connect(config['db_path'])
    conn.row_factory = sqlite3.Row

    tools = AgentTools(conn, state.get('current_day', 1), config['workspace'] / 'agent', config['db_path'])

    result = tools.set_daily_spend({
        'advertising': args.ads,
        'operations': args.ops,
        'development': args.dev
    })
    print(result.message)

    conn.close()


def cmd_set_capacity(args):
    """Set capacity tier."""
    from saas_bench.tools import AgentTools
    import sqlite3

    config = get_env_config()
    state = load_state(config['workspace'])

    conn = sqlite3.connect(config['db_path'])
    conn.row_factory = sqlite3.Row

    tools = AgentTools(conn, state.get('current_day', 1), config['workspace'] / 'agent', config['db_path'])

    result = tools.set_capacity_tier(args.tier)
    print(result.message)

    conn.close()


def cmd_log_rationale(args):
    """Log reasoning/rationale."""
    config = get_env_config()
    state = load_state(config['workspace'])

    log_file = config['workspace'] / 'rationale.log'
    with open(log_file, 'a') as f:
        timestamp = datetime.now(timezone.utc).isoformat()
        day = state.get('current_day', 1)
        f.write(f"[Day {day}] [{timestamp}] {args.text}\n")

    print(f"Logged rationale for day {state.get('current_day', 1)}")


def main():
    parser = argparse.ArgumentParser(description='SaaS Bench CLI Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # status
    subparsers.add_parser('status', help='Show current dashboard')

    # next_day
    subparsers.add_parser('next_day', help='Advance to next day')

    # set_prices
    prices_parser = subparsers.add_parser('set_prices', help='Set plan prices')
    prices_parser.add_argument('--A', type=float, required=True, help='Price for Plan A')
    prices_parser.add_argument('--B', type=float, required=True, help='Price for Plan B')
    prices_parser.add_argument('--C', type=float, required=True, help='Price for Plan C')

    # set_daily_spend
    spend_parser = subparsers.add_parser('set_daily_spend', help='Set daily spending')
    spend_parser.add_argument('--ads', type=float, default=0, help='Advertising spend')
    spend_parser.add_argument('--ops', type=float, default=0, help='Operations spend')
    spend_parser.add_argument('--dev', type=float, default=0, help='Development spend')

    # set_capacity
    capacity_parser = subparsers.add_parser('set_capacity', help='Set capacity tier')
    capacity_parser.add_argument('--tier', type=int, required=True, help='Capacity tier (0-3)')

    # log_rationale
    rationale_parser = subparsers.add_parser('log_rationale', help='Log your reasoning')
    rationale_parser.add_argument('text', help='Rationale text')

    args = parser.parse_args()

    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'next_day':
        cmd_next_day(args)
    elif args.command == 'set_prices':
        cmd_set_prices(args)
    elif args.command == 'set_daily_spend':
        cmd_set_daily_spend(args)
    elif args.command == 'set_capacity':
        cmd_set_capacity(args)
    elif args.command == 'log_rationale':
        cmd_log_rationale(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
