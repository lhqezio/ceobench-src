"""OpenAI Gym-style environment for SaaS Bench.

This module provides a standardized interface for agents to interact with
the SaaS business simulation. The environment follows the Gym API pattern:
- reset() -> initial observation
- step(action) -> (observation, reward, done, truncated, info)

Actions are tool calls (tool name + arguments).
Observations are tool outputs (including daily dashboard from next_day).
"""

import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from numpy.random import Generator, default_rng

from .config import BenchmarkConfig, MODEL_TIERS, CAPACITY_TIERS
from .database import init_database, get_config, get_cash, get_mrr, get_active_subscriber_count, get_founder_cumulative_dividends
from .simulation import Simulator, DayResult
from .tools import AgentTools, ToolResult


def build_daily_dashboard(
    conn: sqlite3.Connection,
    day: int,
    day_result: Optional[DayResult] = None,
    calc_outputs: Optional[Dict[str, str]] = None,
    inbox_items: Optional[list] = None,
) -> str:
    """Build the daily dashboard string — SINGLE SOURCE OF TRUTH.

    This is the canonical dashboard format returned by the next_day tool.
    All agent runners and the environment should call this function instead
    of maintaining their own dashboard builders.

    Args:
        conn: Database connection
        day: Current day number
        day_result: Results from the previous day's simulation (None on Day 1)
        calc_outputs: Dict mapping calculation name to output string
        inbox_items: List of inbox items (notifications, threads) for today

    Returns:
        Formatted dashboard string
    """
    config = get_config(conn, day) or {}
    cash = get_cash(conn)
    sub_count = get_active_subscriber_count(conn)

    # Get open issues count
    open_issues = conn.execute("""
        SELECT COUNT(*) FROM customer_state cs
        JOIN subscriptions s ON cs.customer_id = s.customer_id
        WHERE s.status = 'subscribed' AND s.end_day IS NULL
          AND cs.open_issue_days > 0
    """).fetchone()[0]

    founder_divs = get_founder_cumulative_dividends(conn)

    # Use day_result if available, otherwise fall back to DB queries
    # (day_result is None on the first day after a resume)
    if day_result:
        ind_subs = day_result.total_individual_subscribers
        ent_seats = day_result.total_enterprise_subscription_seats
        mrr = day_result.mrr
    else:
        # Fall back to DB
        ind_subs = conn.execute("""
            SELECT COUNT(*) FROM subscriptions s
            JOIN customers c ON s.customer_id = c.customer_id
            WHERE s.status = 'subscribed' AND s.end_day IS NULL
              AND c.customer_type = 'small'
        """).fetchone()[0]
        ent_seats = conn.execute("""
            SELECT COALESCE(SUM(CAST(c.seat_count AS INTEGER)), 0) FROM subscriptions s
            JOIN customers c ON s.customer_id = c.customer_id
            WHERE s.status = 'subscribed' AND s.end_day IS NULL
              AND c.customer_type = 'large'
        """).fetchone()[0]
        mrr = get_mrr(conn)

    lines = [
        f"=== Day {day} Dashboard ===",
        "",
        f"Cash: ${cash:,.0f}",
        f"Founder Dividends (Cumulative): ${founder_divs:,.0f}",
        f"Individual Subscribers: {ind_subs}",
        f"Enterprise Subscribed Seats: {ent_seats}",
        f"MRR: ${mrr:,.0f} (sum of all active effective_price; accounts for promotions and negotiated prices)",
        f"Open Issues: {open_issues}",
    ]

    # Yesterday's metrics (only after Day 1)
    if day_result:
        lines.extend([
            "",
            "--- Yesterday's Metrics ---",
            f"Usage: {day_result.total_usage:,} units",
            f"New Individual Leads: {day_result.new_individual_leads} | New Enterprise Leads: {day_result.new_enterprise_leads}",
            f"New Individual Subscribers: {day_result.new_individual_subscribers} | New Enterprise Subscribed Seats: {day_result.new_enterprise_subscribers_seats}",
            f"Cancellations: {day_result.cancellations}",
            f"Upgrades: {day_result.upgrades} | Downgrades: {day_result.downgrades}",
            f"Overload: {day_result.overload:.1%}" if day_result.overload > 0 else "Overload: None",
            f"Outage: {'YES (' + str(day_result.downtime_minutes) + ' min)' if day_result.outage else 'No'}",
            f"P95 Latency: {day_result.p95_ms:.0f}ms | Error Rate: {day_result.error_rate:.2%}",
            f"Revenue: ${day_result.payments_received:,.0f} | Costs: ${day_result.total_costs:,.0f}",
        ])

    # Equity & Funding
    if day_result:
        lines.extend([
            "",
            "--- Equity & Funding ---",
            f"Founder Ownership: {day_result.founder_ownership_pct:.1f}%",
            f"Total Shares: {day_result.total_shares:,.0f}",
            f"Active VC Negotiations: {day_result.vc_deals_pending}",
            f"Total Dividends Paid: ${day_result.total_dividends_paid:,.0f} (Founder: ${day_result.founder_cumulative_dividends:,.0f})",
            f"Retained Earnings: ${day_result.retained_earnings:,.0f}",
        ])

    # Current Config
    lines.extend([
        "",
        "--- Current Config ---",
        f"Prices: A=${config.get('price_A', 0):.0f}, B=${config.get('price_B', 0):.0f}, C=${config.get('price_C', 0):.0f}",
        f"Model Tiers: A={config.get('tier_A', 1)}, B={config.get('tier_B', 2)}, C={config.get('tier_C', 3)}",
        f"Quotas: A={config.get('quota_A', 100)}, B={config.get('quota_B', 500)}, C={config.get('quota_C', 2000)} units/day",
        f"Capacity: Tier {config.get('capacity_tier', 0)}",
        f"Daily Spend: Ads=${config.get('spend_advertising', 0):.0f}, Ops=${config.get('spend_operations', 0):.0f}, Dev=${config.get('spend_development', 0):.0f}",
    ])

    # Quality info
    q_shared_bonus_row = conn.execute("SELECT value FROM global_state WHERE key = 'q_shared_bonus'").fetchone()
    q_shared_bonus = float(q_shared_bonus_row['value']) if q_shared_bonus_row else 0.0

    # Compute delivered quality per plan: (base_product_quality + q_shared_bonus) × tier_multiplier
    base_pq = BenchmarkConfig.base_product_quality  # Class default (0.50)
    tier_a = config.get('tier_A', 1)
    tier_b = config.get('tier_B', 2)
    tier_c = config.get('tier_C', 3)
    q_a = (base_pq + q_shared_bonus) * MODEL_TIERS[tier_a].quality_multiplier
    q_b = (base_pq + q_shared_bonus) * MODEL_TIERS[tier_b].quality_multiplier
    q_c = (base_pq + q_shared_bonus) * MODEL_TIERS[tier_c].quality_multiplier

    lines.append(f"Product Quality: A={q_a:.3f}, B={q_b:.3f}, C={q_c:.3f}")

    # Daily calculation outputs
    if calc_outputs:
        lines.append("")
        lines.append("--- Daily Calculations ---")
        for name, output in calc_outputs.items():
            lines.append(f"[{name}]")
            lines.append(output[:500])  # Truncate long outputs

    # Inbox
    lines.append("")
    lines.append("--- Inbox ---")
    if inbox_items:
        for item in inbox_items:
            lines.append(f"  • {item}")
    else:
        lines.append("  (No new messages)")

    return '\n'.join(lines)


def get_thread_inbox_items(conn: sqlite3.Connection, day: int) -> List[str]:
    """Get inbox summary items — counts of new enterprise messages today.

    Standalone function so both the environment class and run_test.py can use it.

    Returns list of formatted summary strings for the inbox section.
    """
    items = []

    # Count new enterprise threads created today
    new_threads = conn.execute("""
        SELECT COUNT(*) as cnt,
               COALESCE(SUM(CAST(c.seat_count AS INTEGER)), 0) as total_seats
        FROM enterprise_turns et
        JOIN customers c ON et.customer_id = c.customer_id
        WHERE et.turn_number = 0 AND et.day = ?
    """, (day,)).fetchone()

    if new_threads['cnt'] > 0:
        items.append(f"📨 {new_threads['cnt']} new enterprise leads today ({new_threads['total_seats']:,} total seats)")

    # Count new enterprise replies today (customer replies on existing threads)
    new_replies = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM enterprise_turns et
        WHERE et.day = ? AND et.turn_number > 0 AND et.sender = 'customer'
    """, (day,)).fetchone()

    if new_replies['cnt'] > 0:
        items.append(f"✉️ {new_replies['cnt']} new enterprise replies today")

    # Count total open threads awaiting agent response
    awaiting = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM enterprise_turns et
        WHERE et.message_id = (
            SELECT MAX(et2.message_id) FROM enterprise_turns et2
            WHERE et2.thread_id = et.thread_id
        )
        AND et.closed = 0
        AND et._internal_status IS NULL
        AND et.sender = 'customer'
    """).fetchone()

    if awaiting['cnt'] > 0:
        items.append(f"⏳ {awaiting['cnt']} enterprise threads awaiting your response")

    # Count new VC messages today
    new_vc = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM vc_turns
        WHERE day = ? AND sender != 'agent'
    """, (day,)).fetchone()

    if new_vc['cnt'] > 0:
        items.append(f"🏦 {new_vc['cnt']} new VC messages today")

    return items


@dataclass
class Action:
    """An action in the environment = a tool call."""
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of taking a step in the environment."""
    observation: str  # Tool output (text)
    reward: float  # Reward signal
    done: bool  # Episode finished (bankruptcy or max days)
    truncated: bool  # Episode truncated (max steps)
    info: Dict[str, Any] = field(default_factory=dict)  # Additional info


class SaaSBenchEnv:
    """OpenAI Gym-style environment for the SaaS business simulation.

    The agent interacts with the environment through tool calls.
    Each step executes one tool and returns the output.
    The `next_day` tool advances the simulation and returns the daily dashboard.

    Memory tools are NOT included - agents manage their own memory/context.
    """

    # Environment tools (no memory tools - those are agent-side)
    ENV_TOOLS = [
        'set_prices',
        'set_model_tiers',
        'set_daily_spend',
        'set_ad_channel_spend',
        'set_targeted_ad_spend',
        'set_targeted_ops_spend',
        'set_targeted_dev_spend',
        'set_capacity_tier',
        'set_usage_quotas',
        'send_enterprise_deal',
        'python_exec',
        'register_daily_calculation',
        'remove_daily_calculation',
        'list_daily_calculations',
        'get_social_posts',

        'get_cost_info',
        'get_tool_documentation',
        'next_day',
        # V2: VC negotiation & equity tools
        'list_potential_vcs',
        'send_vc_deal',
        'reject_vc_deal',
        'reject_enterprise_deal',
        'get_cap_table_info',
        'settle_investments',
        'declare_dividend',
        # V2: Database exploration
        'list_all_tables',
        'describe_tables',
    ]

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        db_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
        seed: Optional[int] = None,
        max_days: int = 3650,
    ):
        """Initialize the environment.

        Args:
            config: Benchmark configuration (uses default if None)
            db_path: Path to database file (uses temp if None)
            workspace_path: Path for agent workspace (uses temp if None)
            seed: Random seed for reproducibility
            max_days: Maximum simulation days before truncation
        """
        self.config = config or BenchmarkConfig()
        self.db_path = db_path or Path('/tmp/saas_bench.db')
        self.workspace_path = workspace_path or Path('/tmp/saas_bench_workspace')
        self.seed = seed
        self.max_days = max_days

        # State (initialized in reset)
        self.conn: Optional[sqlite3.Connection] = None
        self.simulator: Optional[Simulator] = None
        self.tools: Optional[AgentTools] = None
        self.rng: Optional[Generator] = None
        self.current_day: int = 0
        self.last_day_result: Optional[DayResult] = None
        self._done: bool = False

        # Daily calculations storage
        self._daily_calculations: Dict[str, str] = {}

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional seed override

        Returns:
            (initial_observation, info) - Initial dashboard and metadata
        """
        # Use provided seed or instance seed
        actual_seed = seed if seed is not None else self.seed
        self.rng = default_rng(actual_seed)

        # Clean up previous connection
        if self.conn:
            self.conn.close()

        # Remove old database if exists
        if self.db_path.exists():
            self.db_path.unlink()

        # Initialize database (init_database creates tables and returns connection)
        self.conn = init_database(self.db_path)

        # Create workspace
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize simulator
        self.simulator = Simulator(self.conn, self.config, self.rng)
        self.simulator.initialize()  # Set up initial state (config, cash, etc.)

        # Initialize tools (V2: pass config for VC negotiation)
        self.tools = AgentTools(
            self.conn,
            self.simulator.current_day,
            self.workspace_path,
            self.db_path.resolve(),
            self.rng,
            config=self.config,
        )

        # Reset state
        self.current_day = self.simulator.current_day
        self.last_day_result = None
        self._done = False
        self._daily_calculations = {}

        # Build initial observation (Day 1 dashboard)
        initial_obs = self._build_initial_dashboard()

        info = {
            'day': self.current_day,
            'cash': get_cash(self.conn),
            'seed': actual_seed,
        }

        return initial_obs, info

    def step(self, action: Action) -> StepResult:
        """Execute an action (tool call) and return the result.

        Args:
            action: The tool call to execute

        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        if self._done:
            return StepResult(
                observation="Episode has ended. Call reset() to start a new episode.",
                reward=0.0,
                done=True,
                truncated=False,
                info={'error': 'episode_ended'}
            )

        # Validate tool
        if action.tool not in self.ENV_TOOLS:
            return StepResult(
                observation=f"Invalid tool: {action.tool}. Valid tools: {self.ENV_TOOLS}",
                reward=0.0,
                done=False,
                truncated=False,
                info={'error': 'invalid_tool'}
            )

        # Execute the tool
        if action.tool == 'next_day':
            return self._handle_next_day()
        else:
            return self._handle_tool_call(action)

    def _handle_tool_call(self, action: Action) -> StepResult:
        """Handle a non-next_day tool call."""
        tool_name = action.tool
        args = action.arguments

        try:
            # Get the tool method
            tool_method = getattr(self.tools, tool_name, None)
            if tool_method is None:
                return StepResult(
                    observation=f"Tool not found: {tool_name}",
                    reward=0.0,
                    done=False,
                    truncated=False,
                    info={'error': 'tool_not_found'}
                )

            # Handle tools that accept partial dicts (filter out None values)
            if tool_name == 'set_prices':
                d = {k: args[k] for k in ('A', 'B', 'C') if k in args and args[k] is not None}
                result = tool_method(d)
            elif tool_name == 'set_model_tiers':
                d = {k: args[k] for k in ('A', 'B', 'C') if k in args and args[k] is not None}
                result = tool_method(d)
            elif tool_name == 'set_daily_spend':
                d = {k: args[k] for k in ('advertising', 'operations', 'development') if k in args and args[k] is not None}
                result = tool_method(d)
            elif tool_name == 'set_ad_channel_spend':
                d = {k: args[k] for k in ('social_media', 'search_ads', 'linkedin', 'content_marketing', 'referral_program') if k in args and args[k] is not None}
                result = tool_method(d)
            elif tool_name == 'set_targeted_ad_spend':
                result = tool_method(args.get('targeted_spend', args))
            elif tool_name == 'set_targeted_ops_spend':
                result = tool_method(args.get('targeted_spend', args))
            elif tool_name == 'set_targeted_dev_spend':
                result = tool_method(args.get('targeted_spend', args))
            elif tool_name == 'set_ads_strength':
                result = tool_method(
                    global_strength=args.get('global_strength'),
                    by_group=args.get('by_group'),
                    by_customer=args.get('by_customer'),
                )
            elif tool_name == 'set_lead_promotion':
                result = tool_method(
                    global_promotion=args.get('global_promotion'),
                    by_group=args.get('by_group'),
                    by_channel=args.get('by_channel'),
                    by_channel_group=args.get('by_channel_group'),
                )
            elif tool_name == 'set_promotion':
                result = tool_method(
                    global_promotion=args.get('global_promotion'),
                    by_group=args.get('by_group'),
                    by_customer=args.get('by_customer'),
                    by_group_plan=args.get('by_group_plan'),
                )
            elif tool_name == 'set_usage_quotas':
                result = tool_method({'A': args.get('A'), 'B': args.get('B'), 'C': args.get('C')})
            elif tool_name in ('send_enterprise_deal', 'send_vc_deal', 'reject_vc_deal', 'reject_enterprise_deal'):
                # List-based tools: pass deals parameter
                result = tool_method(deals=args.get('deals', []))
            elif tool_name == 'get_tool_documentation':
                # Handle string, list, or None for tool_names
                tool_names = args.get('tool_names', args.get('tools', None))
                result = tool_method(tool_names)
            else:
                # Standard tool call
                result = tool_method(**args)

            observation = result.message
            if result.data:
                observation += f"\n\nData: {json.dumps(result.data, indent=2, default=str)}"

            return StepResult(
                observation=observation,
                reward=0.0,  # No reward for non-next_day actions
                done=False,
                truncated=False,
                info={'success': result.success, 'data': result.data}
            )

        except Exception as e:
            return StepResult(
                observation=f"Tool execution error: {str(e)}",
                reward=0.0,
                done=False,
                truncated=False,
                info={'error': str(e)}
            )

    def _handle_next_day(self) -> StepResult:
        """Handle the next_day action - advance simulation and return dashboard."""
        # Run daily calculations first
        calc_outputs = self._run_daily_calculations()

        # Advance simulation
        day_result = self.simulator.step_day()
        self.last_day_result = day_result
        self.current_day = day_result.day

        # Update tools with new day
        self.tools.set_current_day(self.current_day)

        # Check termination conditions
        cash = get_cash(self.conn)
        done = False
        truncated = False

        if cash < 0:
            done = True  # Bankruptcy
        elif self.current_day >= self.max_days:
            truncated = True  # Max days reached

        self._done = done or truncated

        # Build daily dashboard
        dashboard = self._build_daily_dashboard(day_result, calc_outputs)

        # Calculate reward (change in cash + some MRR bonus)
        reward = self._calculate_reward(day_result)

        info = {
            'day': self.current_day,
            'cash': cash,
            'mrr': day_result.mrr,
            'day_result': day_result,
            'bankruptcy': cash < 0,
        }

        return StepResult(
            observation=dashboard,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info
        )

    def _build_initial_dashboard(self) -> str:
        """Build the initial dashboard for Day 1. Delegates to build_daily_dashboard()."""
        dashboard = build_daily_dashboard(self.conn, self.current_day)
        dashboard += "\n\nWelcome to NovaMind! Your AI SaaS company is ready to launch."
        dashboard += "\nUse tools to configure your business, then call next_day to advance."
        return dashboard

    def _build_daily_dashboard(self, result: DayResult, calc_outputs: Dict[str, str]) -> str:
        """Build the daily dashboard from day results. Delegates to build_daily_dashboard()."""
        inbox_items = self._get_inbox_items()
        return build_daily_dashboard(
            self.conn, self.current_day, result, calc_outputs, inbox_items
        )

    def _get_inbox_items(self) -> List[str]:
        """Get inbox items (today's notifications, new threads, and new messages)."""
        items = []

        # Get all notifications for today (no read/unread tracking)
        notifications = self.conn.execute("""
            SELECT notification_id, type, title
            FROM notifications
            WHERE day = ?
            ORDER BY notification_id
            LIMIT 10
        """, (self.current_day,)).fetchall()

        for n in notifications:
            items.append(f"[{n['notification_id']}] {n['title']}")

        # Add thread inbox items (new threads + new messages on existing threads)
        items.extend(get_thread_inbox_items(self.conn, self.current_day))

        return items

    def _run_daily_calculations(self) -> Dict[str, str]:
        """Run registered daily calculations."""
        results = {}
        for name, code in self._daily_calculations.items():
            result = self.tools.python_exec(code)
            if result.success:
                results[name] = result.message
            else:
                results[name] = f"ERROR: {result.message}"
        return results

    def _calculate_reward(self, result: DayResult) -> float:
        """Calculate reward for the day.

        Simple reward: net profit (revenue - costs) + MRR growth bonus
        """
        net_profit = result.payments_received - result.total_costs
        mrr_bonus = result.mrr * 0.01  # Small bonus for MRR

        # Penalty for outages
        outage_penalty = -1000 if result.outage else 0

        return net_profit + mrr_bonus + outage_penalty

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get tool descriptions for the agent (environment tools only)."""
        from .tools import get_tool_descriptions
        all_tools = get_tool_descriptions()

        # Filter to only environment tools
        env_tools = [t for t in all_tools if t['name'] in self.ENV_TOOLS]
        return env_tools

    def close(self):
        """Clean up resources."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
