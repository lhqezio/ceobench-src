"""No-LLM tests for `set_targeted_ops_spend` with mixed targeting (group / plan / group+plan / customer).

Strategy: build synthetic customers directly in the DB (bypass acquisition), then
drive `_process_issues` in isolation and verify that each scope resolves the
target pool at the correct *per-group* Poisson rate.

Per-group semantics (v3.4h):
  - Every pool (global + 4 targeted scopes) is partitioned by customer group.
  - For each group g in a pool of size P with n_g members:
        mean_g = scale_g × spend × (n_g / P)
    where scale_g = 0.25 for S*/D_S* (individual) and 0.05 for E*/D_E* (enterprise).
  - Pure-group pool collapses to scale_g × spend.
  - Mixed pool (by_plan spanning groups) yields composition-weighted rate.

Covers:
  (A) Handler validation — backward-compat legacy `targeted_spend`, new kwargs, rejection paths
  (B) Ledger — daily 'Targeted ops spend' entry sums all four scopes
  (C) Engine — each scope's per-group Poisson means match theory (CLT over K trials)
  (D) Isolation — non-target subscribers are never touched by a targeted scope
"""
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pytest
from numpy.random import Generator, PCG64

from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.simulation import Simulator
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.tools import AgentTools


SEED = 42


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers — synthetic DB state
# ══════════════════════════════════════════════════════════════════════════

def _make_sim(seed: int = SEED, total_days: int = 90):
    """Fresh headless simulation (no LLM) with seed + tool handle.

    Global resolution is DISABLED (base_rate=0, spend_operations=0) so any
    resolutions observed come from targeted pools only.
    """
    config = BenchmarkConfig(seed=seed, total_days=total_days, initial_cash=1_000_000.0)
    # Disable global resolution entirely — we want to isolate targeted pools.
    config.issue_resolution_base_rate = 0.0
    # Disable relationship decay noise (irrelevant to resolution count, just reduces noise in logs)
    config.relationship_decay_per_unresolved_day = 0.0

    rng = Generator(PCG64(seed))
    conn = init_database(":memory:")
    conn.row_factory = sqlite3.Row

    customer_sim = CustomerSimulator(client=None, conn=conn, config=config)
    simulator = Simulator(conn, config, rng, customer_simulator=customer_sim)
    simulator.initialize()

    tmpdir = Path(tempfile.mkdtemp())
    tools = AgentTools(conn, 0, tmpdir, tmpdir / "world.db", rng=rng, config=config, seed=seed)

    return conn, simulator, tools, config


def _insert_synthetic_customer(
    conn: sqlite3.Connection,
    customer_id: int,
    group_id: str,
    plan: str,
    *,
    has_issue: bool = True,
    seat_count: int = 1,
) -> None:
    """Insert a fully-formed synthetic customer: customers + subscriptions + customer_state."""
    conn.execute(
        """INSERT INTO customers (
               customer_id, customer_type, group_id, created_day,
               steepness_left, steepness_right, c_max,
               q_max, q_min, usage_demand,
               quality_sensitivity, price_sensitivity, willingness_to_pay,
               usage_scale, patience, seat_count
           ) VALUES (?, 'small', ?, 0,
                    10.0, 10.0, 200.0,
                    0.75, 0.25, 1.0,
                    0.5, 0.5, 100.0,
                    1.0, 0.5, ?)""",
        (customer_id, group_id, seat_count),
    )
    conn.execute(
        """INSERT INTO subscriptions (
               customer_id, plan, listed_price, effective_price, start_day, status,
               billing_day_mod30, daily_usage_rate, seat_count, contract_months, first_billing_done
           ) VALUES (?, ?, 50.0, 50.0, 0, 'subscribed', 0, 1.0, ?, 1, 1)""",
        (customer_id, plan, seat_count),
    )
    conn.execute(
        "INSERT INTO customer_state (customer_id, satisfaction, open_issue_days, relationship) VALUES (?, 0.5, ?, 0.5)",
        (customer_id, 1 if has_issue else 0),
    )


def _populate_grid(conn: sqlite3.Connection, spec: List[Tuple[str, str, int]], *, start_cid: int = 10_000) -> Dict[Tuple[str, str], List[int]]:
    """Insert `n` customers for each (group, plan) in spec. Return {(group, plan): [cids]}.

    All inserted customers have open_issue_days=1 (in the resolution pool).
    """
    cid = start_cid
    cells: Dict[Tuple[str, str], List[int]] = {}
    for group, plan, n in spec:
        cell_cids: List[int] = []
        for _ in range(n):
            _insert_synthetic_customer(conn, cid, group, plan, has_issue=True)
            cell_cids.append(cid)
            cid += 1
        cells[(group, plan)] = cell_cids
    conn.commit()
    return cells


def _count_with_issues(conn: sqlite3.Connection, cids: Iterable[int]) -> int:
    """Count how many of the given customer_ids currently have open_issue_days > 0."""
    cids = list(cids)
    if not cids:
        return 0
    placeholders = ",".join("?" * len(cids))
    return conn.execute(
        f"SELECT COUNT(*) FROM customer_state WHERE open_issue_days > 0 AND customer_id IN ({placeholders})",
        cids,
    ).fetchone()[0]


def _run_one_isolated_resolution(simulator) -> None:
    """Invoke `_process_issues` once with spend_operations=0 (pure targeted-pool test).

    We bypass full step_day so there's no marketing, no billing, no new customer
    acquisition — just the resolution mechanic on the current DB state. The
    only side effects we care about are updates to `customer_state.open_issue_days`
    and `issues.status`.
    """
    # Reset cache so _process_issues reads fresh DB state via its fallback query
    if hasattr(simulator, "_cached_all_subscribers"):
        delattr(simulator, "_cached_all_subscribers")
    fake_config = {"spend_operations": 0.0}
    simulator._process_issues(fake_config, outage=False)


# ══════════════════════════════════════════════════════════════════════════
# (A) Handler validation
# ══════════════════════════════════════════════════════════════════════════

class TestHandlerValidation:

    def test_legacy_targeted_spend_still_works(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend({"E1": 300, "E2": 200})
        assert result.success, result.message
        assert config.targeted_ops_spend == {"E1": 300.0, "E2": 200.0}
        assert result.data['by_group'] == {"E1": 300.0, "E2": 200.0}
        assert result.data['targeted_spend'] == {"E1": 300.0, "E2": 200.0}
        assert result.data['total_extra_per_day'] == 500.0

    def test_by_group_kwarg(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend(by_group={"E1": 400})
        assert result.success
        assert config.targeted_ops_spend == {"E1": 400.0}

    def test_by_plan(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend(by_plan={"A": 200, "B": 100})
        assert result.success
        assert config.targeted_ops_spend_by_plan == {"A": 200.0, "B": 100.0}

    def test_by_group_plan(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend(by_group_plan={"E1": {"A": 150, "B": 50}})
        assert result.success
        assert config.targeted_ops_spend_by_group_plan == {"E1": {"A": 150.0, "B": 50.0}}

    def test_by_customer(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend(by_customer={"42": 75})
        assert result.success
        assert config.targeted_ops_spend_by_customer == {42: 75.0}
        assert result.data['by_customer'] == {"42": 75.0}

    def test_all_scopes_at_once(self):
        _, _, tools, config = _make_sim()
        result = tools.set_targeted_ops_spend(
            by_group={"E1": 300},
            by_plan={"A": 200},
            by_group_plan={"E2": {"B": 100}},
            by_customer={"42": 50},
        )
        assert result.success
        assert result.data['total_extra_per_day'] == 650.0

    def test_empty_dict_clears_scope(self):
        _, _, tools, config = _make_sim()
        tools.set_targeted_ops_spend(by_group={"E1": 300}, by_plan={"A": 100})
        result = tools.set_targeted_ops_spend(by_group={})
        assert result.success
        assert config.targeted_ops_spend == {}
        assert config.targeted_ops_spend_by_plan == {"A": 100.0}

    def test_none_preserves_scope(self):
        _, _, tools, config = _make_sim()
        tools.set_targeted_ops_spend(by_group={"E1": 300}, by_plan={"A": 100})
        tools.set_targeted_ops_spend(by_customer={"1": 25})
        assert config.targeted_ops_spend == {"E1": 300.0}
        assert config.targeted_ops_spend_by_plan == {"A": 100.0}
        assert config.targeted_ops_spend_by_customer == {1: 25.0}

    def test_rejects_legacy_plus_by_group(self):
        _, _, tools, _ = _make_sim()
        result = tools.set_targeted_ops_spend(targeted_spend={"E1": 100}, by_group={"E2": 100})
        assert not result.success
        assert "not both" in result.message

    def test_rejects_invalid_group(self):
        _, _, tools, _ = _make_sim()
        result = tools.set_targeted_ops_spend(by_group={"ZZZ": 100})
        assert not result.success
        assert "Invalid group IDs" in result.message

    def test_rejects_invalid_plan(self):
        _, _, tools, _ = _make_sim()
        result = tools.set_targeted_ops_spend(by_plan={"D": 100})
        assert not result.success
        assert "Invalid plans" in result.message

    def test_rejects_invalid_group_plan_plan(self):
        _, _, tools, _ = _make_sim()
        result = tools.set_targeted_ops_spend(by_group_plan={"E1": {"Z": 100}})
        assert not result.success
        assert "Invalid plans for group E1" in result.message

    def test_rejects_non_integer_customer_id(self):
        _, _, tools, _ = _make_sim()
        result = tools.set_targeted_ops_spend(by_customer={"not-an-int": 100})
        assert not result.success
        assert "must be an integer" in result.message

    def test_rejects_negative_amount(self):
        _, _, tools, _ = _make_sim()
        for scope in [
            {"by_group": {"E1": -1}},
            {"by_plan": {"A": -1}},
            {"by_group_plan": {"E1": {"A": -1}}},
            {"by_customer": {"1": -1}},
        ]:
            result = tools.set_targeted_ops_spend(**scope)
            assert not result.success, f"Should reject negative in {scope}"


# ══════════════════════════════════════════════════════════════════════════
# (B) Ledger sums all four scopes
# ══════════════════════════════════════════════════════════════════════════

class TestLedgerIntegration:

    def test_ledger_sums_all_scopes(self):
        """After one sim day, 'Targeted ops spend' ledger entry must include all four scope totals."""
        conn, simulator, tools, _ = _make_sim()
        # Insert one synthetic customer so by_customer scope has a valid cid (10_000+ to avoid the
        # sentinel customer inserted by simulator.initialize())
        _insert_synthetic_customer(conn, 10_001, "S1", "A", has_issue=True)
        conn.commit()

        tools.set_targeted_ops_spend(
            by_group={"S1": 100},
            by_plan={"A": 50},
            by_group_plan={"S2": {"B": 30}},
            by_customer={"10001": 20},
        )
        expected_per_day = 100 + 50 + 30 + 20

        simulator.step_day()

        rows = conn.execute(
            """SELECT amount FROM ledger
               WHERE day = ? AND category = 'operations' AND note = 'Targeted ops spend'""",
            (simulator.current_day,),
        ).fetchall()
        assert len(rows) == 1, f"Expected 1 'Targeted ops spend' entry, got {len(rows)}"
        assert rows[0]['amount'] == pytest.approx(-expected_per_day)


# ══════════════════════════════════════════════════════════════════════════
# (C) Engine — Poisson rate verification on synthetic customers
#
# Each test:
#   1. Build K fresh sims with 200 customers per cell (covers target + non-target)
#   2. For each sim: apply ONE scope's targeted spend, run ONE resolution step
#   3. Record how many customers in each cell had their issue resolved
#   4. Assert sample mean for target cell ≈ Poisson mean = ops_scale * spend
#      (±3σ around sqrt(K * mean))
#   5. Assert non-target cells saw 0 resolutions (no leakage)
# ══════════════════════════════════════════════════════════════════════════

K_TRIALS = 40  # number of independent sims per test
N_PER_CELL = 200  # synthetic customers per (group, plan) cell
INDIVIDUAL_OPS_SCALE = 0.25  # must match BenchmarkConfig.individual_ops_scale default
ENTERPRISE_OPS_SCALE = 0.05  # must match BenchmarkConfig.enterprise_ops_scale default


def _run_trial(
    seed: int,
    grid_spec: List[Tuple[str, str, int]],
    scope_call: Dict,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], List[int]]]:
    """Build a fresh sim with the grid, apply one scope, run resolution once.

    Returns: (resolved_per_cell, cell_cids)
    """
    conn, sim, tools, config = _make_sim(seed=seed)
    cells = _populate_grid(conn, grid_spec)
    result = tools.set_targeted_ops_spend(**scope_call)
    assert result.success, f"Scope call failed: {result.message}"

    # Pre-snapshot: every cell customer has an issue
    pre_counts = {cell: _count_with_issues(conn, cids) for cell, cids in cells.items()}
    assert all(v == len(cells[cell]) for cell, v in pre_counts.items())

    _run_one_isolated_resolution(sim)

    # Post: resolved = those with open_issue_days == 0 now
    resolved_per_cell = {
        cell: len(cids) - _count_with_issues(conn, cids)
        for cell, cids in cells.items()
    }
    return resolved_per_cell, cells


def _poisson_interval_3sigma(mean_per_trial: float, k: int) -> Tuple[float, float]:
    """3σ CLT interval on the sample mean of K Poisson(mean_per_trial) draws."""
    se = math.sqrt(mean_per_trial / k) if mean_per_trial > 0 else 0.0
    return mean_per_trial - 3.0 * se, mean_per_trial + 3.0 * se


class TestEnginePoissonRates:

    def test_by_group_rate_individual(self):
        """by_group targeting an individual group (S1) → mean = 0.25 × spend."""
        spend = 400  # mean per trial = 0.25 * 400 = 100, pool = 200 → no cap
        expected_mean = INDIVIDUAL_OPS_SCALE * spend
        grid = [
            ("S1", "A", N_PER_CELL),
            ("S2", "A", N_PER_CELL),  # non-target same plan
            ("E1", "A", N_PER_CELL),  # non-target different group
        ]

        target_cell = ("S1", "A")
        resolved_totals: Dict[Tuple[str, str], int] = {(g, p): 0 for g, p, _ in grid}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=1000 + trial,
                grid_spec=grid,
                scope_call={"by_group": {"S1": spend}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        sample_mean = resolved_totals[target_cell] / K_TRIALS
        lo, hi = _poisson_interval_3sigma(expected_mean, K_TRIALS)
        assert lo <= sample_mean <= hi, (
            f"by_group S1: sample mean {sample_mean:.2f} outside Poisson 3σ "
            f"interval [{lo:.2f}, {hi:.2f}] (expected {expected_mean})"
        )
        # Non-target cells: exactly 0 resolutions (no leakage)
        assert resolved_totals[("S2", "A")] == 0, (
            f"by_group leaked into S2/A: {resolved_totals[('S2', 'A')]}"
        )
        assert resolved_totals[("E1", "A")] == 0, (
            f"by_group leaked into E1/A: {resolved_totals[('E1', 'A')]}"
        )

    def test_by_group_rate_enterprise(self):
        """by_group targeting an enterprise group (E1) → mean = 0.05 × spend (5× slower per $)."""
        spend = 400  # mean per trial = 0.05 * 400 = 20, pool = 200 → no cap
        expected_mean = ENTERPRISE_OPS_SCALE * spend
        grid = [
            ("E1", "A", N_PER_CELL),   # target
            ("E2", "A", N_PER_CELL),   # non-target enterprise
            ("S1", "A", N_PER_CELL),   # non-target individual
        ]

        target_cell = ("E1", "A")
        resolved_totals: Dict[Tuple[str, str], int] = {(g, p): 0 for g, p, _ in grid}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=1500 + trial,
                grid_spec=grid,
                scope_call={"by_group": {"E1": spend}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        sample_mean = resolved_totals[target_cell] / K_TRIALS
        lo, hi = _poisson_interval_3sigma(expected_mean, K_TRIALS)
        assert lo <= sample_mean <= hi, (
            f"by_group E1 (enterprise): sample mean {sample_mean:.2f} outside Poisson 3σ "
            f"interval [{lo:.2f}, {hi:.2f}] (expected {expected_mean})"
        )
        assert resolved_totals[("E2", "A")] == 0, f"leaked into E2/A: {resolved_totals[('E2', 'A')]}"
        assert resolved_totals[("S1", "A")] == 0, f"leaked into S1/A: {resolved_totals[('S1', 'A')]}"

    def test_by_plan_pure_individual_rate(self):
        """by_plan A with only individual groups in plan A → mean = 0.25 × spend."""
        spend = 400
        expected_mean = INDIVIDUAL_OPS_SCALE * spend
        grid = [
            ("S1", "A", N_PER_CELL),
            ("S2", "A", N_PER_CELL),
            ("S1", "B", N_PER_CELL),  # non-target plan
        ]
        target_cells = [("S1", "A"), ("S2", "A")]
        non_target_cells = [("S1", "B")]
        resolved_totals = {c: 0 for c in target_cells + non_target_cells}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=2000 + trial,
                grid_spec=grid,
                scope_call={"by_plan": {"A": spend}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        target_sum = sum(resolved_totals[c] for c in target_cells)
        sample_mean = target_sum / K_TRIALS
        lo, hi = _poisson_interval_3sigma(expected_mean, K_TRIALS)
        assert lo <= sample_mean <= hi, (
            f"by_plan A (pure individual): sample mean {sample_mean:.2f} outside Poisson 3σ "
            f"interval [{lo:.2f}, {hi:.2f}] (expected {expected_mean})"
        )
        assert resolved_totals[("S1", "B")] == 0, (
            f"by_plan leaked into S1/B: {resolved_totals[('S1', 'B')]}"
        )

    def test_by_plan_mixed_pool_composition_weighted(self):
        """by_plan A with a mixed pool of individual + enterprise subscribers.

        Pool = 200 (S1) + 200 (E1) = 400.
          mean_S1 = 0.25 × spend × 200/400 = 0.25 × spend / 2
          mean_E1 = 0.05 × spend × 200/400 = 0.05 × spend / 2
          total  = (0.25 + 0.05) × spend / 2 = 0.15 × spend
        """
        spend = 400
        expected_mean_S1 = INDIVIDUAL_OPS_SCALE * spend * 0.5
        expected_mean_E1 = ENTERPRISE_OPS_SCALE * spend * 0.5
        expected_total = expected_mean_S1 + expected_mean_E1  # = 60 when spend=400

        grid = [
            ("S1", "A", N_PER_CELL),
            ("E1", "A", N_PER_CELL),
            ("S1", "B", N_PER_CELL),  # non-target plan
        ]
        target_cells = [("S1", "A"), ("E1", "A")]
        non_target_cells = [("S1", "B")]
        resolved_totals = {c: 0 for c in target_cells + non_target_cells}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=2500 + trial,
                grid_spec=grid,
                scope_call={"by_plan": {"A": spend}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        # Check per-cell means (each cell is Poisson-distributed with its per-group mean)
        sample_mean_S1 = resolved_totals[("S1", "A")] / K_TRIALS
        lo_S1, hi_S1 = _poisson_interval_3sigma(expected_mean_S1, K_TRIALS)
        assert lo_S1 <= sample_mean_S1 <= hi_S1, (
            f"by_plan mixed — S1/A cell mean {sample_mean_S1:.2f} outside 3σ "
            f"[{lo_S1:.2f}, {hi_S1:.2f}] (expected {expected_mean_S1})"
        )

        sample_mean_E1 = resolved_totals[("E1", "A")] / K_TRIALS
        lo_E1, hi_E1 = _poisson_interval_3sigma(expected_mean_E1, K_TRIALS)
        assert lo_E1 <= sample_mean_E1 <= hi_E1, (
            f"by_plan mixed — E1/A cell mean {sample_mean_E1:.2f} outside 3σ "
            f"[{lo_E1:.2f}, {hi_E1:.2f}] (expected {expected_mean_E1})"
        )

        # Total across target cells
        target_sum = sum(resolved_totals[c] for c in target_cells)
        sample_total = target_sum / K_TRIALS
        lo_t, hi_t = _poisson_interval_3sigma(expected_total, K_TRIALS)
        assert lo_t <= sample_total <= hi_t, (
            f"by_plan mixed — total {sample_total:.2f} outside 3σ "
            f"[{lo_t:.2f}, {hi_t:.2f}] (expected {expected_total})"
        )
        assert resolved_totals[("S1", "B")] == 0, (
            f"by_plan leaked into S1/B: {resolved_totals[('S1', 'B')]}"
        )

    def test_by_group_plan_rate_individual(self):
        """by_group_plan S1/A → mean = 0.25 × spend (S1 is individual)."""
        spend = 400
        expected_mean = INDIVIDUAL_OPS_SCALE * spend
        grid = [
            ("S1", "A", N_PER_CELL),   # target (intersection)
            ("S1", "B", N_PER_CELL),   # non-target (same group different plan)
            ("E1", "A", N_PER_CELL),   # non-target (same plan different group)
        ]
        target_cell = ("S1", "A")
        non_target_cells = [("S1", "B"), ("E1", "A")]
        resolved_totals = {c: 0 for c in [target_cell] + non_target_cells}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=3000 + trial,
                grid_spec=grid,
                scope_call={"by_group_plan": {"S1": {"A": spend}}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        sample_mean = resolved_totals[target_cell] / K_TRIALS
        lo, hi = _poisson_interval_3sigma(expected_mean, K_TRIALS)
        assert lo <= sample_mean <= hi, (
            f"by_group_plan S1/A: sample mean {sample_mean:.2f} outside Poisson "
            f"3σ interval [{lo:.2f}, {hi:.2f}] (expected {expected_mean})"
        )
        for cell in non_target_cells:
            assert resolved_totals[cell] == 0, (
                f"by_group_plan leaked into {cell}: {resolved_totals[cell]}"
            )

    def test_by_group_plan_rate_enterprise(self):
        """by_group_plan E1/A → mean = 0.05 × spend (E1 is enterprise)."""
        spend = 400
        expected_mean = ENTERPRISE_OPS_SCALE * spend
        grid = [
            ("E1", "A", N_PER_CELL),   # target
            ("E1", "B", N_PER_CELL),   # non-target (same group different plan)
            ("S1", "A", N_PER_CELL),   # non-target (same plan different group)
        ]
        target_cell = ("E1", "A")
        non_target_cells = [("E1", "B"), ("S1", "A")]
        resolved_totals = {c: 0 for c in [target_cell] + non_target_cells}

        for trial in range(K_TRIALS):
            resolved_per_cell, _ = _run_trial(
                seed=3500 + trial,
                grid_spec=grid,
                scope_call={"by_group_plan": {"E1": {"A": spend}}},
            )
            for cell, n in resolved_per_cell.items():
                resolved_totals[cell] += n

        sample_mean = resolved_totals[target_cell] / K_TRIALS
        lo, hi = _poisson_interval_3sigma(expected_mean, K_TRIALS)
        assert lo <= sample_mean <= hi, (
            f"by_group_plan E1/A (enterprise): sample mean {sample_mean:.2f} outside 3σ "
            f"[{lo:.2f}, {hi:.2f}] (expected {expected_mean})"
        )
        for cell in non_target_cells:
            assert resolved_totals[cell] == 0, (
                f"by_group_plan leaked into {cell}: {resolved_totals[cell]}"
            )

    def test_by_customer_rate_individual(self):
        """by_customer on an S1 (individual) customer: pool size = 1, scale = 0.25.

        P(resolved in trial) = 1 - exp(-0.25 × spend). With spend=4, mean=1,
        P(res) = 1 - e^{-1} ≈ 0.632.
        """
        spend = 4
        mean_rate = INDIVIDUAL_OPS_SCALE * spend  # = 1.0
        p_resolved = 1.0 - math.exp(-mean_rate)   # ≈ 0.632
        expected_successes = K_TRIALS * p_resolved
        se = math.sqrt(K_TRIALS * p_resolved * (1 - p_resolved))
        lo, hi = expected_successes - 3.0 * se, expected_successes + 3.0 * se

        grid = [("S1", "A", N_PER_CELL)]
        successes = 0
        leakage_total = 0
        for trial in range(K_TRIALS):
            conn, sim, tools, _ = _make_sim(seed=4000 + trial)
            cells = _populate_grid(conn, grid)
            target_cid = cells[("S1", "A")][0]
            non_target_cids = cells[("S1", "A")][1:]

            result = tools.set_targeted_ops_spend(by_customer={str(target_cid): spend})
            assert result.success

            _run_one_isolated_resolution(sim)

            target_still_open = conn.execute(
                "SELECT open_issue_days FROM customer_state WHERE customer_id = ?",
                (target_cid,),
            ).fetchone()["open_issue_days"]
            if target_still_open == 0:
                successes += 1

            still_open_non_target = _count_with_issues(conn, non_target_cids)
            leakage_total += (len(non_target_cids) - still_open_non_target)

        assert lo <= successes <= hi, (
            f"by_customer (S1): {successes}/{K_TRIALS} successes outside Bernoulli 3σ "
            f"interval [{lo:.2f}, {hi:.2f}] (expected {expected_successes:.2f})"
        )
        assert leakage_total == 0, f"by_customer leaked to non-target customers: {leakage_total}"

    def test_by_customer_rate_enterprise(self):
        """by_customer on an E1 (enterprise) customer: scale = 0.05.

        With spend=20, mean = 0.05 × 20 = 1.0, P(res) = 1 - e^{-1} ≈ 0.632.
        Same Bernoulli as individual case but spend is 5× higher to reach mean=1.
        """
        spend = 20
        mean_rate = ENTERPRISE_OPS_SCALE * spend  # = 1.0
        p_resolved = 1.0 - math.exp(-mean_rate)
        expected_successes = K_TRIALS * p_resolved
        se = math.sqrt(K_TRIALS * p_resolved * (1 - p_resolved))
        lo, hi = expected_successes - 3.0 * se, expected_successes + 3.0 * se

        grid = [("E1", "A", N_PER_CELL)]
        successes = 0
        leakage_total = 0
        for trial in range(K_TRIALS):
            conn, sim, tools, _ = _make_sim(seed=4500 + trial)
            cells = _populate_grid(conn, grid)
            target_cid = cells[("E1", "A")][0]
            non_target_cids = cells[("E1", "A")][1:]

            result = tools.set_targeted_ops_spend(by_customer={str(target_cid): spend})
            assert result.success

            _run_one_isolated_resolution(sim)

            target_still_open = conn.execute(
                "SELECT open_issue_days FROM customer_state WHERE customer_id = ?",
                (target_cid,),
            ).fetchone()["open_issue_days"]
            if target_still_open == 0:
                successes += 1

            still_open_non_target = _count_with_issues(conn, non_target_cids)
            leakage_total += (len(non_target_cids) - still_open_non_target)

        assert lo <= successes <= hi, (
            f"by_customer (E1 enterprise): {successes}/{K_TRIALS} successes outside Bernoulli "
            f"3σ interval [{lo:.2f}, {hi:.2f}] (expected {expected_successes:.2f})"
        )
        assert leakage_total == 0, f"by_customer leaked to non-target customers: {leakage_total}"

    def test_by_customer_spend_5x_parity_individual_vs_enterprise(self):
        """Sanity: 5× spend on an enterprise customer matches 1× spend on an individual.

        At spend=4 on S1 → mean = 0.25 × 4 = 1.0.
        At spend=20 on E1 → mean = 0.05 × 20 = 1.0.
        Both should yield ~63.2% resolution rate.
        """
        # This is implicitly covered by the two tests above sharing p_resolved ≈ 0.632.
        assert abs(INDIVIDUAL_OPS_SCALE * 4 - ENTERPRISE_OPS_SCALE * 20) < 1e-12

    def test_zero_spend_resolves_nothing(self):
        """Sanity: every targeted-spend scope with spend=0 must resolve zero issues."""
        for scope_call in [
            {"by_group": {"S1": 0}},
            {"by_plan": {"A": 0}},
            {"by_group_plan": {"S1": {"A": 0}}},
            {"by_customer": {"10000": 0}},
        ]:
            conn, sim, tools, _ = _make_sim(seed=9999)
            _populate_grid(conn, [("S1", "A", 50)])  # inserts cids 10000..10049
            result = tools.set_targeted_ops_spend(**scope_call)
            assert result.success
            _run_one_isolated_resolution(sim)
            still_open = conn.execute(
                "SELECT COUNT(*) FROM customer_state WHERE open_issue_days > 0"
            ).fetchone()[0]
            assert still_open == 50, (
                f"Zero spend on {scope_call} resolved some issues: {50 - still_open}"
            )


# ══════════════════════════════════════════════════════════════════════════
# (D) Multi-scope combined Poisson rates
# ══════════════════════════════════════════════════════════════════════════

class TestMultiScopeCombined:

    def test_two_scopes_cover_same_pool_stack_rates(self):
        """If two scopes target the same customer, each contributes its own Poisson pool.

        Effective resolution probability per trial = 1 - exp(-(sum of means))
        because two independent Poisson draws: target is hit if *either* fires.
        With a single-customer pool, each draw is capped at 1.
        """
        # S1 is individual (scale = 0.25).
        # by_group S1 with spend=4 → mean 1.0
        # by_plan  A  with spend=4 → mean 1.0 (pool has one customer, all in group S1)
        # A customer in S1/A is covered by BOTH, so effective rate on that customer
        # is: 1 - exp(-1) × exp(-1) = 1 - exp(-2) ≈ 0.865 per trial.
        mean_total = INDIVIDUAL_OPS_SCALE * 4 + INDIVIDUAL_OPS_SCALE * 4
        p_resolved = 1.0 - math.exp(-mean_total)
        expected = K_TRIALS * p_resolved
        se = math.sqrt(K_TRIALS * p_resolved * (1 - p_resolved))
        lo, hi = expected - 3.0 * se, expected + 3.0 * se

        successes = 0
        target_cid = 10_001
        for trial in range(K_TRIALS):
            conn, sim, tools, _ = _make_sim(seed=5000 + trial)
            _insert_synthetic_customer(conn, target_cid, "S1", "A", has_issue=True)
            conn.commit()

            tools.set_targeted_ops_spend(
                by_group={"S1": 4},
                by_plan={"A": 4},
            )
            _run_one_isolated_resolution(sim)

            if conn.execute(
                "SELECT open_issue_days FROM customer_state WHERE customer_id = ?",
                (target_cid,),
            ).fetchone()["open_issue_days"] == 0:
                successes += 1

        assert lo <= successes <= hi, (
            f"Combined by_group+by_plan: {successes}/{K_TRIALS} outside Bernoulli "
            f"3σ interval [{lo:.2f}, {hi:.2f}] (expected {expected:.2f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
