"""V2: VC negotiation system for SaaS Bench.

This module handles multi-turn negotiations with VC investors.
It provides a programmatic chassis for equity negotiations while allowing
the agent to make strategic decisions about dilution vs. cash infusion.

Key concepts:
- VCs approach the company with a randomly selected equity % from their range
- Check size is derived from equity % and the VC's valuation of the company
- Agent proposes share_pct → if >= VC's target, accept; else counter with target
- Agent can accept, counter, or reject. Rejection terminates permanently.
- Accepted deals must be settled via settle_investments() before expiry
- At settlement, all deals must use the same price_per_share

Direction of negotiation:
- VC samples equity_pct from [equity_pct_min, equity_pct_max]
- Check size = pct / (1 - pct) × valuation
- Agent offers share_pct → VC accepts if >= target, else counters with target pct
"""

import sqlite3
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from numpy.random import Generator

from .config import BenchmarkConfig


@dataclass
class VCNegotiationState:
    """Current state of a VC negotiation."""
    shareholder_id: int
    state: str  # 'pending', 'negotiating', 'accepted', 'agent_rejected', 'expired', 'timeout'
    negotiation_turn: int
    current_offer_share_pct: Optional[float]
    current_offer_amount: Optional[float]
    next_reply_day: Optional[int]
    replied: int  # 0=not replied, 1=replied
    created_day: int
    expiry_day: Optional[int]
    # From shareholders table
    vc_name: str
    target_share_pct: float  # Base equity % from valuation (check / (valuation + check))
    investment_amount: float  # How much $ the VC wants to invest
    equity_pct_min: float     # Minimum equity % target
    equity_pct_max: float     # Maximum equity % target
    reply_delay_mean: float
    reply_delay_std: float


def get_vc_negotiation_state(conn: sqlite3.Connection, shareholder_id: int) -> Optional[VCNegotiationState]:
    """Get the current state of a VC negotiation.

    Reads from vc_turns (latest turn = current state).
    """
    row = conn.execute("""
        SELECT vt.*, s.name as vc_name, s.target_share_pct, s.investment_amount,
               s.equity_pct_min, s.equity_pct_max,
               s.reply_delay_mean, s.reply_delay_std
        FROM vc_turns vt
        JOIN shareholders s ON vt.shareholder_id = s.shareholder_id
        WHERE vt.shareholder_id = ?
        ORDER BY vt.turn_number DESC
        LIMIT 1
    """, (shareholder_id,)).fetchone()

    if not row:
        return None

    # Derive state from closed/close_reason
    if row['closed']:
        reason = row['close_reason'] or 'closed'
        state = reason  # 'accepted', 'agent_rejected', 'settled'
    else:
        # Open negotiation: check if agent has replied (sender='agent' on latest turn)
        state = 'negotiating' if row['sender'] == 'agent' else 'pending'

    # Determine replied: agent has replied if latest sender is agent
    replied = 1 if row['sender'] == 'agent' else 0

    # Get the first turn's day as created_day
    first_turn = conn.execute("""
        SELECT day FROM vc_turns WHERE shareholder_id = ? ORDER BY turn_number ASC LIMIT 1
    """, (shareholder_id,)).fetchone()
    created_day = first_turn['day'] if first_turn else row['day']

    return VCNegotiationState(
        shareholder_id=row['shareholder_id'],
        state=state,
        negotiation_turn=row['turn_number'],
        current_offer_share_pct=row['current_offer_share_pct'],
        current_offer_amount=row['current_offer_amount'],
        next_reply_day=row['next_reply_day'],
        replied=replied,
        created_day=created_day,
        expiry_day=row['expiry_day'],
        vc_name=row['vc_name'],
        target_share_pct=row['target_share_pct'],
        investment_amount=row['investment_amount'],
        equity_pct_min=row['equity_pct_min'],
        equity_pct_max=row['equity_pct_max'],
        reply_delay_mean=row['reply_delay_mean'],
        reply_delay_std=row['reply_delay_std'],
    )


def compute_check_size_from_pct(share_pct: float, valuation: float) -> float:
    """Compute the implied check size for a given share_pct and pre-money valuation.

    Formula: check = share_pct / (1 - share_pct) × valuation
    This is the post-money algebra: share_pct = check / (valuation + check)
    """
    if share_pct <= 0 or share_pct >= 1.0:
        return 0.0
    return (share_pct / (1.0 - share_pct)) * valuation


def compute_pct_from_check(check: float, valuation: float) -> float:
    """Compute share_pct from a check size and pre-money valuation.

    Formula: pct = check / (valuation + check)  (post-money %)
    """
    if valuation + check <= 0:
        return 0.0
    return check / (valuation + check)


def evaluate_agent_vc_offer(
    state: VCNegotiationState,
    agent_offer_share_pct: float,
    config: BenchmarkConfig,
    valuation: float,
    term_sheet_adjustment: float = 0.0,
) -> Tuple[str, float, float]:
    """Evaluate an equity offer made by the agent to a VC.

    The agent offers share_pct → if it meets the VC's target, accept.
    Otherwise counter with the VC's target pct.

    V2.2: term_sheet_adjustment modifies the effective valuation.
    Positive adjustment = agent proposing more VC-friendly terms → higher valuation →
    VC accepts at lower equity.

    Args:
        state: Current VC negotiation state
        agent_offer_share_pct: The share % the agent is offering to the VC
        config: Benchmark configuration
        valuation: VC's current fair valuation of the company (pre-money)
        term_sheet_adjustment: Friendliness delta (agent - VC original). Positive = VC-friendlier.

    Returns:
        Tuple of (decision, final_pct, final_check)
        decision: 'accept' or 'counter'
        final_pct: The share % for the deal
        final_check: The check size for the deal
    """
    # Adjust valuation for term sheet friendliness
    effective_valuation = valuation * (1.0 + term_sheet_adjustment)
    effective_valuation = max(effective_valuation, 100_000)  # Floor

    # Compute implied check from agent's offered %
    implied_check = compute_check_size_from_pct(agent_offer_share_pct, effective_valuation)
    implied_check = round(implied_check, -3)  # Round to nearest $1K

    # Accept if agent's offer meets or exceeds VC's target equity %
    if agent_offer_share_pct >= state.target_share_pct:
        final_pct = round(agent_offer_share_pct, 6)
        return ('accept', final_pct, implied_check)

    # Otherwise counter with VC's target pct
    counter_pct = round(state.target_share_pct, 6)
    counter_check = round(
        compute_check_size_from_pct(counter_pct, effective_valuation), -3
    )
    return ('counter', counter_pct, counter_check)


def compute_vc_reply_delay(state: VCNegotiationState, rng: Generator) -> int:
    """Compute how many days until VC replies."""
    delay = rng.normal(state.reply_delay_mean, state.reply_delay_std)
    return max(1, int(round(delay)))


def schedule_vc_reply(conn: sqlite3.Connection, shareholder_id: int,
                      current_day: int, rng: Generator):
    """Schedule when the VC will reply to the agent's message.

    Updates next_reply_day on the latest turn for this VC.
    """
    state = get_vc_negotiation_state(conn, shareholder_id)
    if not state:
        return

    delay = compute_vc_reply_delay(state, rng)
    next_reply_day = current_day + delay

    conn.execute("""
        UPDATE vc_turns SET next_reply_day = ?
        WHERE shareholder_id = ? AND message_id = (
            SELECT MAX(message_id) FROM vc_turns WHERE shareholder_id = ?
        )
    """, (next_reply_day, shareholder_id, shareholder_id))
    conn.commit()


def get_vcs_needing_reply(conn: sqlite3.Connection, current_day: int) -> List[int]:
    """Get VC shareholder_ids where VC reply is due today.

    Finds VCs where the latest turn has next_reply_day = current_day
    and the negotiation is not in a terminal status.
    """
    rows = conn.execute("""
        SELECT vt.shareholder_id FROM vc_turns vt
        WHERE vt.message_id = (
            SELECT MAX(vt2.message_id) FROM vc_turns vt2 WHERE vt2.shareholder_id = vt.shareholder_id
        )
        AND vt.next_reply_day = ?
        AND vt.closed = 0
        AND vt._internal_status IS NULL
    """, (current_day,)).fetchall()
    return [row['shareholder_id'] for row in rows]


def get_vcs_awaiting_agent(conn: sqlite3.Connection, current_day: int,
                            timeout_days: int = 5) -> List[dict]:
    """Get VC negotiations where agent hasn't responded within timeout_days.

    Returns VCs where the latest turn has sender='vc' and days_waiting >= timeout_days.
    """
    rows = conn.execute("""
        SELECT vt.shareholder_id, s.name as vc_name,
               vt.day as last_message_day,
               (? - vt.day) as days_waiting
        FROM vc_turns vt
        JOIN shareholders s ON vt.shareholder_id = s.shareholder_id
        WHERE vt.message_id = (
            SELECT MAX(vt2.message_id) FROM vc_turns vt2 WHERE vt2.shareholder_id = vt.shareholder_id
        )
        AND vt.closed = 0
        AND vt._internal_status IS NULL
        AND vt.sender = 'vc'
        AND (? - vt.day) >= ?
    """, (current_day, current_day, timeout_days)).fetchall()
    return [dict(row) for row in rows]


def compute_vc_satisfaction(conn: sqlite3.Connection, shareholder_id: int,
                            current_price_per_share: float) -> float:
    """Compute a VC's satisfaction based on their investment returns.

    Formula: satisfaction = Σ(shares_i × (current_price - buy_price_i)) / total_shares_held

    This represents the paper profit/loss per share relative to purchase prices.
    Positive = happy (investment appreciating), negative = unhappy.

    Args:
        conn: Database connection
        shareholder_id: The VC shareholder ID
        current_price_per_share: Current implied price per share

    Returns:
        Satisfaction score (can be negative for underwater investments)
    """
    # Get all funding rounds for this investor
    rounds = conn.execute("""
        SELECT shares_issued, price_per_share
        FROM funding_rounds
        WHERE investor_shareholder_id = ?
    """, (shareholder_id,)).fetchall()

    if not rounds:
        return 0.0

    total_shares = sum(r['shares_issued'] for r in rounds)
    if total_shares <= 0:
        return 0.0

    weighted_gain = sum(
        r['shares_issued'] * (current_price_per_share - r['price_per_share'])
        for r in rounds
    )
    return weighted_gain / total_shares


def generate_vc_name(rng: Generator) -> str:
    """Generate a random VC firm name."""
    prefixes = [
        "Apex", "Summit", "Horizon", "Pinnacle", "Atlas", "Meridian",
        "Vanguard", "Nexus", "Catalyst", "Forge", "Beacon", "Vertex",
        "Zenith", "Accel", "Bridge", "Crest", "Dawn", "Eagle", "Frontier",
        "Gateway", "Harbor", "Iron", "Keystone", "Landmark", "Matrix",
    ]
    suffixes = [
        "Capital", "Ventures", "Partners", "Fund", "Equity",
        "Investment Group", "Holdings", "Associates",
    ]
    prefix = prefixes[rng.integers(0, len(prefixes))]
    suffix = suffixes[rng.integers(0, len(suffixes))]
    return f"{prefix} {suffix}"
