#!/usr/bin/env python3
"""Standalone MCP server for SaaS Bench.

This script runs the MCP server that exposes SaaS Bench tools to Claude Code.
It reads configuration from environment variables set by the runner.

Environment variables:
    SAAS_BENCH_WORKSPACE: Path to the workspace directory
    SAAS_BENCH_RUN_ID: Run ID for this session
    SAAS_BENCH_DB_PATH: Path to the SQLite database
    SAAS_BENCH_DAY: Current simulation day (updated by runner)
"""

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Add the package to path if running standalone
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

from saas_bench.tools import AgentTools, get_mcp_tool_definitions
from saas_bench.database import init_database, get_cash, get_active_subscriber_count, get_config
from saas_bench.simulation import Simulator
from saas_bench.config import BenchmarkConfig, ScenarioPack, SCENARIO_PACKS
from saas_bench.shocks import ShockManager
from numpy.random import Generator, PCG64


@dataclass
class RationaleEntry:
    """A single rationale/thinking log entry."""
    timestamp: str
    day: int
    rationale: str
    context: Optional[str] = None


class SaaSBenchMCPServer:
    """MCP Server for SaaS Bench tools."""

    def __init__(self):
        # Read config from environment
        self.workspace_dir = Path(os.environ.get('SAAS_BENCH_WORKSPACE', './workspace'))
        self.run_id = os.environ.get('SAAS_BENCH_RUN_ID', 'unknown')
        self.db_path = Path(os.environ.get('SAAS_BENCH_DB_PATH', self.workspace_dir / 'world.db'))
        self.current_day = int(os.environ.get('SAAS_BENCH_DAY', '1'))
        self.total_days = int(os.environ.get('SAAS_BENCH_TOTAL_DAYS', '3650'))
        self.seed = int(os.environ.get('SAAS_BENCH_SEED', '42'))

        # Get scenario from environment or use default
        scenario_name = os.environ.get('SAAS_BENCH_SCENARIO', 'default')
        self.scenario = SCENARIO_PACKS.get(scenario_name, ScenarioPack(
            name='Default',
            description='Balanced scenario'
        ))

        # State file for communication with runner
        self.state_file = self.workspace_dir / '.mcp_state.json'
        self.rationales_file = self.workspace_dir / 'logs' / f'rationales_{self.run_id}.json'
        self.tool_calls_file = self.workspace_dir / 'logs' / f'tool_calls_{self.run_id}.jsonl'
        self.file_changes_file = self.workspace_dir / 'logs' / f'file_changes_{self.run_id}.jsonl'

        # Track file states for change detection
        self._file_snapshots: Dict[str, Dict[str, Any]] = {}  # path -> {mtime, size, hash}

        # Initialize database connection
        self.conn: Optional[sqlite3.Connection] = None
        self.tools: Optional[AgentTools] = None

        # Simulation components (initialized lazily)
        self.simulator: Optional[Simulator] = None
        self.shock_manager: Optional[ShockManager] = None
        self.rng: Optional[Generator] = None
        self.rng: Optional[Generator] = None
        self.last_result = None  # Track last simulation result for dashboard

        # Load existing rationales from file (persists across MCP server restarts)
        self.rationales: List[RationaleEntry] = self._load_rationales()

        # Track if next_day was called
        self.day_ended = False

        # MCP server
        self.server = Server("saas-bench")
        self._setup_handlers()

    def _now(self) -> str:
        """Get current UTC timestamp."""
        return datetime.utcnow().isoformat() + "Z"

    def _load_state(self):
        """Load state from file (updated by runner each day)."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.current_day = state.get('current_day', self.current_day)
                self.day_ended = state.get('day_ended', False)

    def _save_state(self):
        """Save state to file for runner to read."""
        state = {
            'current_day': self.current_day,
            'day_ended': self.day_ended,
            'last_updated': self._now()
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def _load_rationales(self) -> List[RationaleEntry]:
        """Load existing rationales from file."""
        if self.rationales_file.exists():
            try:
                with open(self.rationales_file) as f:
                    data = json.load(f)
                    return [
                        RationaleEntry(
                            timestamp=r['timestamp'],
                            day=r['day'],
                            rationale=r['rationale'],
                            context=r.get('context')
                        )
                        for r in data
                    ]
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save_rationales(self):
        """Save rationales to file."""
        self.rationales_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rationales_file, 'w') as f:
            json.dump([{
                'timestamp': r.timestamp,
                'day': r.day,
                'rationale': r.rationale,
                'context': r.context
            } for r in self.rationales], f, indent=2)

    def _log_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                       result: Optional[str], error: Optional[str]):
        """Log a tool call to the JSONL file."""
        self.tool_calls_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": self._now(),
            "day": self.current_day,
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:5000] if result else None,  # Limit result size
            "error": error[:2000] if error else None
        }
        with open(self.tool_calls_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get file info for change detection."""
        import hashlib
        try:
            stat = path.stat()
            # Read first 10KB for hash (enough to detect changes)
            with open(path, 'rb') as f:
                content_sample = f.read(10240)
            content_hash = hashlib.md5(content_sample).hexdigest()
            return {
                'mtime': stat.st_mtime,
                'size': stat.st_size,
                'hash': content_hash
            }
        except Exception:
            return None

    def _scan_workspace_files(self) -> Dict[str, Dict[str, Any]]:
        """Scan agent workspace for all files."""
        agent_workspace = self.workspace_dir / 'agent'
        files = {}
        if agent_workspace.exists():
            for path in agent_workspace.rglob('*'):
                if path.is_file():
                    rel_path = str(path.relative_to(agent_workspace))
                    info = self._get_file_info(path)
                    if info:
                        files[rel_path] = info
        return files

    def _detect_and_log_file_changes(self):
        """Detect file changes in agent workspace and log them."""
        current_files = self._scan_workspace_files()

        changes = []

        # Check for new or modified files
        for path, info in current_files.items():
            if path not in self._file_snapshots:
                # New file
                changes.append({
                    'type': 'created',
                    'path': path,
                    'size': info['size']
                })
            elif info['hash'] != self._file_snapshots[path]['hash']:
                # Modified file
                changes.append({
                    'type': 'modified',
                    'path': path,
                    'old_size': self._file_snapshots[path]['size'],
                    'new_size': info['size']
                })

        # Check for deleted files
        for path in self._file_snapshots:
            if path not in current_files:
                changes.append({
                    'type': 'deleted',
                    'path': path
                })

        # Log changes
        if changes:
            self.file_changes_file.parent.mkdir(parents=True, exist_ok=True)
            for change in changes:
                entry = {
                    'timestamp': self._now(),
                    'day': self.current_day,
                    **change
                }
                # Read file content for created/modified files (first 5000 chars)
                if change['type'] in ('created', 'modified'):
                    agent_workspace = self.workspace_dir / 'agent'
                    file_path = agent_workspace / change['path']
                    try:
                        with open(file_path, 'r', errors='replace') as f:
                            content = f.read(5000)
                        entry['content_preview'] = content
                    except Exception:
                        pass
                with open(self.file_changes_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')

        # Update snapshots
        self._file_snapshots = current_files

    def _init_tools(self):
        """Initialize database, tools, and simulator."""
        if self.conn is None:
            if not self.db_path.exists():
                raise RuntimeError(f"Database not found at {self.db_path}")
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

        # Initialize RNG first (needed for tools and simulator)
        if self.rng is None:
            self.rng = Generator(PCG64(self.seed))

        if self.tools is None:
            agent_workspace = self.workspace_dir / 'agent'
            agent_workspace.mkdir(parents=True, exist_ok=True)
            self.tools = AgentTools(self.conn, self.current_day, agent_workspace, self.db_path, self.rng)

        # Initialize simulator if not already done
        # NOTE: The runner (run_test.py) already calls simulator.initialize() to set up Day 1.
        # We should NOT call initialize() again here, just set up the Simulator object.
        if self.simulator is None:
            bench_config = BenchmarkConfig(
                seed=self.seed,
                total_days=self.total_days,
                initial_cash=1_000_000.0,
            )
            self.simulator = Simulator(self.conn, bench_config, self.rng)

        # Always sync simulator's current_day with what's actually in the database
        # This prevents issues when next_day is called multiple times in a session
        max_day_row = self.conn.execute("SELECT COALESCE(MAX(day), 0) FROM service_day").fetchone()
        db_current_day = max_day_row[0] if max_day_row else 0
        self.simulator.current_day = db_current_day

        # Initialize shock manager if not already done
        if self.shock_manager is None:
            # Use a separate RNG for shock manager to avoid state conflicts
            shock_rng = Generator(PCG64(self.seed + 1000))
            self.shock_manager = ShockManager(self.conn, shock_rng, self.scenario)

        # Update current day
        self._load_state()
        self.tools.set_current_day(self.current_day)

    def _setup_handlers(self):
        """Set up MCP handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools.

            Tool definitions (names, descriptions, schemas) are loaded dynamically
            from get_mcp_tool_definitions() which derives them from TOOL_DOCS.
            Only log_rationale is defined here since it's MCP-agent-specific.
            """
            # Dynamically load tool definitions from the canonical source
            tool_defs = get_mcp_tool_definitions()
            tools = [
                Tool(
                    name=td["name"],
                    description=td["description"],
                    inputSchema=td["inputSchema"],
                )
                for td in tool_defs
            ]

            # Add MCP-only tools not in the shared tool set
            tools.append(
                Tool(
                    name="log_rationale",
                    description="Log your thinking, rationale, or reasoning for decisions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string", "description": "Your thinking or reasoning"},
                            "context": {"type": "string", "description": "Optional context"}
                        },
                        "required": ["rationale"]
                    }
                ),
            )
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                self._init_tools()
                self._load_state()  # Get current day

                # Take initial file snapshot if not done yet
                if not self._file_snapshots:
                    self._file_snapshots = self._scan_workspace_files()

                result = await self._execute_tool(name, arguments)

                # Log the tool call and result
                self._log_tool_call(name, arguments, result, None)

                # Detect and log any file changes in agent workspace
                self._detect_and_log_file_changes()

                return CallToolResult(
                    content=[TextContent(type="text", text=result)]
                )
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                self._log_tool_call(name, arguments, None, error_msg)
                return CallToolResult(
                    content=[TextContent(type="text", text=error_msg)]
                )

    def _build_dispatch(self) -> Dict[str, Any]:
        """Build the tool dispatch table.

        Maps tool names to handler callables. Each handler receives the
        `arguments` dict and returns a ToolResult.
        """
        tools = self.tools

        return {
            # === Cost Information ===
            "get_cost_info": lambda args: tools.get_cost_info(),

            # === Business Configuration ===
            "set_prices": lambda args: tools.set_prices(args),
            "set_model_tiers": lambda args: tools.set_model_tiers(args),
            "set_daily_spend": lambda args: tools.set_daily_spend(args),
            "set_ad_channel_spend": lambda args: tools.set_ad_channel_spend(args),
            "set_targeted_ad_spend": lambda args: tools.set_targeted_ad_spend(args.get("targeted_spend", args)),
            "set_targeted_ops_spend": lambda args: tools.set_targeted_ops_spend(args.get("targeted_spend", args)),
            "set_targeted_dev_spend": lambda args: tools.set_targeted_dev_spend(args.get("targeted_spend", args)),
            "set_capacity_tier": lambda args: tools.set_capacity_tier(args["tier"]),
            "set_usage_quotas": lambda args: tools.set_usage_quotas(args),

            # === Customer Communication ===
            "send_enterprise_deal": lambda args: tools.send_enterprise_deal(deals=args.get("deals", [])),
            "reject_enterprise_deal": lambda args: tools.reject_enterprise_deal(deals=args.get("deals", [])),

            # === Analytics ===
            "python_exec": lambda args: tools.python_exec(args["code"]),

            # === Daily Calculations ===
            "register_daily_calculation": lambda args: tools.register_daily_calculation(args["name"], args["code"]),
            "remove_daily_calculation": lambda args: tools.remove_daily_calculation(args["name"]),
            "list_daily_calculations": lambda args: tools.list_daily_calculations(),

            # === Named Scripts ===
            "register_script": lambda args: tools.register_script(args.get("name", ""), args.get("code", "")),
            "run_script": lambda args: tools.run_script(args.get("name", "")),
            "list_scripts": lambda args: tools.list_scripts(),
            "delete_script": lambda args: tools.delete_script(args.get("name", "")),

            # === Social Media & Notifications ===
            "get_social_posts": lambda args: tools.get_social_posts(
                days=args.get("days", 7),
                limit=args.get("limit", 50),
            ),
            # === Documentation ===
            "get_tool_documentation": lambda args: tools.get_tool_documentation(args.get("tool_names")),

            # === R&D Research Projects ===
            "start_research_project": lambda args: tools.start_research_project(args.get("project_id", "")),
            "list_research_projects": lambda args: tools.list_research_projects(),

            # === VC Negotiation & Equity ===
            "list_potential_vcs": lambda args: tools.list_potential_vcs(),
            "send_vc_deal": lambda args: tools.send_vc_deal(deals=args.get("deals", [])),
            "reject_vc_deal": lambda args: tools.reject_vc_deal(deals=args.get("deals", [])),
            "get_cap_table": lambda args: tools.get_cap_table_info(),
            "settle_investments": lambda args: tools.settle_investments(),
            "declare_dividend": lambda args: tools.declare_dividend(args.get("amount", 0)),

            # === Market Discovery ===
            "research_market": lambda args: tools.research_market(),
            "research_group": lambda args: tools.research_group(args.get("group_id", "")),
            "get_market_overview": lambda args: tools.get_market_overview(),
            "get_group_insights": lambda args: tools.get_group_insights(args.get("group_id", "")),

            # === Database Exploration ===
            "list_all_tables": lambda args: tools.list_all_tables(),
            "describe_tables": lambda args: tools.describe_tables(args.get("table_names")),

            # === Memory Management ===
            "memory_insert": lambda args: tools.memory_insert(args["line"], args["content"]),
            "memory_delete": lambda args: tools.memory_delete(args["start"], args["end"]),
            "memory_edit": lambda args: tools.memory_edit(args["line"], args["content"]),
        }

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result.

        Tool dispatch is centralized in _build_dispatch(). Special cases
        (next_day, log_rationale) are handled inline since they have
        custom logic beyond just calling AgentTools.
        """
        tools = self.tools

        # === Special case: next_day (runs simulation step) ===
        if name == "next_day":
            # Run simulation step for current day
            self.last_result = self.simulator.step_day()

            # Check for bankruptcy
            if self.simulator.shutdown_mode:
                return f"GAME OVER - BANKRUPT!\n\nYou ran out of cash on day {self.current_day}.\nFinal cash: ${self.last_result.cash:,.0f}"

            # Advance to next day
            self.current_day += 1
            self.tools.set_current_day(self.current_day)

            # Check if simulation complete
            if self.current_day > self.total_days:
                return f"SIMULATION COMPLETE!\n\nYou successfully managed NovaMind AI for {self.total_days} days.\nFinal cash: ${self.last_result.cash:,.0f}"

            # Check for shocks on new day
            new_shocks = self.shock_manager.check_and_generate_shocks(self.current_day)

            # Build dashboard for new day
            cash = get_cash(self.conn)
            subscribers = get_active_subscriber_count(self.conn)
            config = get_config(self.conn, self.current_day)

            # Get open issues count
            open_issues = self.conn.execute("""
                SELECT COUNT(*) FROM customer_state cs
                JOIN subscriptions s ON cs.customer_id = s.customer_id
                WHERE s.status = 'subscribed' AND s.end_day IS NULL
                  AND cs.open_issue_days > 0
            """).fetchone()[0]

            dashboard_lines = [
                f"=== DAY {self.current_day} DASHBOARD ===",
                "",
                f"CASH: ${cash:,.0f}",
                f"SUBSCRIBERS: {subscribers}",
                f"OPEN ISSUES: {open_issues}",
                "",
                "YESTERDAY'S METRICS:",
                f"  - Usage: {self.last_result.total_usage:,} units",
                f"  - New Individual Leads: {self.last_result.new_individual_leads} | New Enterprise Leads: {self.last_result.new_enterprise_leads}",
                f"  - New Individual Subscribers: {self.last_result.new_individual_subscribers} | New Enterprise Subscribed Seats: {self.last_result.new_enterprise_subscribers_seats}",
                f"  - Cancellations: {self.last_result.cancellations}",
                f"  - Upgrades: {self.last_result.upgrades}",
                f"  - Downgrades: {self.last_result.downgrades}",
                f"  - Overload: {self.last_result.overload:.1%}",
                f"  - Outage: {'Yes' if self.last_result.outage else 'No'}",
                "",
            ]

            # Add shocks if any
            if new_shocks:
                dashboard_lines.append("!! ALERTS:")
                for shock in new_shocks:
                    dashboard_lines.append(f"  - {shock.shock_type}: {shock.details.get('description', '')[:60]}")
                dashboard_lines.append("")

            if config:
                dashboard_lines.extend([
                    "CURRENT CONFIG:",
                    f"  - Prices: A=${config['price_A']}, B=${config['price_B']}, C=${config['price_C']}",
                    f"  - Model tiers: A={config['tier_A']}, B={config['tier_B']}, C={config['tier_C']}",
                    f"  - Daily spend: ads=${config['spend_advertising']}, ops=${config['spend_operations']}, dev=${config['spend_development']}",
                    f"  - Capacity tier: {config['capacity_tier']}",
                ])

            # Add inbox
            inbox = self.shock_manager.get_inbox_items(self.current_day)
            if inbox:
                dashboard_lines.extend([
                    "",
                    f"INBOX ({len(inbox)} messages):",
                ])
                for item in inbox[:5]:
                    # Show thread_id prominently for thread_waiting items
                    if item.get('type') == 'thread_waiting':
                        thread_id = item.get('thread_id', '?')
                        state = item.get('state', 'unknown')
                        dashboard_lines.append(f"  - [Thread #{thread_id}] Enterprise - {state}")
                    else:
                        subject = item.get('subject', item.get('message', item.get('type', 'message')))
                        dashboard_lines.append(f"  - {subject[:60]}")

            # Run registered daily calculations
            calc_outputs = tools.run_daily_calculations()
            if calc_outputs:
                dashboard_lines.extend(["", "DAILY CALCULATIONS:"])
                for calc_name, calc_output in calc_outputs.items():
                    dashboard_lines.append(f"  [{calc_name}]")
                    # Show full output (no truncation)
                    for line in str(calc_output).split('\n'):
                        dashboard_lines.append(f"    {line}")

            dashboard_lines.extend(["", "========================="])

            # Save state
            self.day_ended = False
            self._save_state()

            return "\n".join(dashboard_lines)

        # === Special case: log_rationale (MCP-only, persists to file) ===
        if name == "log_rationale":
            # Always load fresh state to get current day from runner
            self._load_state()
            entry = RationaleEntry(
                timestamp=self._now(),
                day=self.current_day,
                rationale=arguments["rationale"],
                context=arguments.get("context")
            )
            self.rationales.append(entry)
            self._save_rationales()
            return f"Rationale logged at day {self.current_day}"

        # === Standard tools via dispatch table ===
        dispatch = self._build_dispatch()
        handler = dispatch.get(name)
        if handler is None:
            return f"Unknown tool: {name}"

        result = handler(arguments)
        if hasattr(result, 'data') and result.data:
            return f"{result.message}\n\nData: {json.dumps(result.data, default=str)}"
        return result.message

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point for the MCP server."""
    server = SaaSBenchMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
