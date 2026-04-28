"""Tests for the competitor-event drift hook.

Verifies three properties tied to requirements 1-3 of the reactivity feature:

  1. The existing *global* q_bias drift behavior still fires on each competitor
     event (raw boost added to `global_drift_state.global_q_bias_total`).
  2. Each group — including discoverable groups that are NOT yet discovered —
     receives `COMPETITOR_REACTIVITY_Q_BIAS[gid] * boost` added to its
     `group_parameters.drift_q_bias_total` accumulator.
  3. `group_parameters` rows exist for all 26 groups from day 0, so the
     per-group accumulator silently tracks drift before discovery.
"""
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from numpy.random import Generator, PCG64

from saas_bench.config import (
    BenchmarkConfig,
    CUSTOMER_GROUPS,
    COMPETITOR_REACTIVITY_Q_BIAS,
    GROUP_PREFERENCE_DRIFT,
    INITIAL_CUSTOMER_GROUPS,
)
from saas_bench.database import init_database
from saas_bench.simulation import Simulator
from saas_bench.customer_llm import CustomerSimulator


SEED = 42


def _make_sim(total_days: int = 500):
    """Fresh headless sim with forced-trigger competitor-event config.

    Grace period = 0, mean_interval = 1, min_interval = 0, late_cutoff = 0 so
    the stochastic gate in `_process_competitor_events` is guaranteed to fire
    when we override `_competitor_rng` with a fixture that returns 0.0 for
    `.random()`.
    """
    config = BenchmarkConfig(seed=SEED, total_days=total_days, initial_cash=1_000_000.0)
    config.drift_grace_period_days = 0
    config.competitor_event_mean_interval = 1.0
    config.competitor_event_min_interval = 0
    config.competitor_event_late_cutoff_days = 0
    # Fix the magnitude ramp so base_boost == boost (scale == 1.0 regardless of day).
    config.competitor_event_magnitude_scale_min = 1.0
    config.competitor_event_magnitude_scale_max = 1.0

    rng = Generator(PCG64(SEED))
    conn = init_database(":memory:")
    conn.row_factory = sqlite3.Row

    customer_sim = CustomerSimulator(client=None, conn=conn, config=config)
    simulator = Simulator(conn, config, rng, customer_simulator=customer_sim)
    simulator.initialize()
    return conn, simulator, config


class _ForcedRng:
    """Minimal RNG stand-in that always triggers the event with a known boost."""

    def __init__(self, boost: float):
        self._boost = boost

    def random(self):
        return 0.0  # < daily_prob → always trigger

    def lognormal(self, mu, sigma):
        return self._boost  # bypass stochastic sampling

    def uniform(self, lo, hi):
        return 1.0  # neutral feedback multiplier; tests use no improvement so delta=0

    def integers(self, lo, hi):
        return lo  # unused here, but safe default

    def bytes(self, n):
        return b"\x00" * n


# ──────────────────────────────────────────────────────────────────────────
# Requirement #3: discoverable-group accumulators exist + accept writes
# ──────────────────────────────────────────────────────────────────────────

def test_group_parameters_rows_exist_for_all_26_groups_at_day_zero():
    """Every group — initial + discoverable — gets a group_parameters row on init.

    Without this, the per-group shock would silently no-op for discoverable
    groups (UPDATE ... WHERE group_id = ? → 0 rows affected).
    """
    conn, _sim, _cfg = _make_sim()
    rows = {row["group_id"]: dict(row) for row in conn.execute("SELECT * FROM group_parameters")}

    # All 26 expected group IDs present
    assert set(rows.keys()) == set(CUSTOMER_GROUPS.keys())
    assert len(rows) == 26

    # Initial groups (6) + discoverable groups (20) all have 0 drift at day 0
    for gid, row in rows.items():
        assert row["drift_q_bias_total"] == 0.0, f"{gid} should start with 0 q_bias drift"
        assert row["drift_c_max_total"] == 0.0, f"{gid} should start with 0 c_max drift"


# ──────────────────────────────────────────────────────────────────────────
# Requirement #1 + #2: event hook applies global + per-group shock to all 26
# ──────────────────────────────────────────────────────────────────────────

def test_competitor_event_applies_global_and_per_group_shock():
    """Firing one competitor event must:
      - add `boost` to global_q_bias_total (existing behavior, preserved)
      - add `coef[g] * boost` to every group's drift_q_bias_total (new hook),
        including discoverable groups that have NOT been discovered.
    """
    conn, sim, _cfg = _make_sim()
    sim.current_day = 1  # past grace=0

    BOOST = 0.1
    sim._competitor_rng = _ForcedRng(BOOST)
    sim._process_competitor_events(config={})

    # ── (1) global drift accumulator ──
    global_total = conn.execute(
        "SELECT global_q_bias_total FROM global_drift_state WHERE id = 1"
    ).fetchone()["global_q_bias_total"]
    assert global_total == pytest.approx(BOOST, abs=1e-9)

    # ── (2) per-group shock for all 26 groups, incl. discoverable/undiscovered ──
    rows = {r["group_id"]: dict(r) for r in conn.execute("SELECT * FROM group_parameters")}
    for gid, coef in COMPETITOR_REACTIVITY_Q_BIAS.items():
        expected = coef * BOOST
        actual = rows[gid]["drift_q_bias_total"]
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"{gid}: expected coef×boost = {coef}×{BOOST} = {expected}, got {actual}"
        )
        # c_max_drift is unaffected by competitor events
        assert rows[gid]["drift_c_max_total"] == 0.0

    # ── (3) Discoverable groups specifically: accumulator moves even though
    #        these groups have not yet been discovered by the agent. ──
    discoverable_gids = [g for g in rows if g.startswith("D_")]
    assert len(discoverable_gids) == 20, "expected 20 discoverable groups"
    for gid in discoverable_gids:
        coef = COMPETITOR_REACTIVITY_Q_BIAS[gid]
        assert rows[gid]["drift_q_bias_total"] == pytest.approx(coef * BOOST, abs=1e-9)


def test_repeated_events_accumulate_linearly():
    """Firing N events with boost B each → each group drift_q_bias_total = N×coef×B."""
    conn, sim, _cfg = _make_sim()
    sim.current_day = 1

    BOOST = 0.05
    N = 4
    sim._competitor_rng = _ForcedRng(BOOST)
    for _ in range(N):
        sim._process_competitor_events(config={})

    global_total = conn.execute(
        "SELECT global_q_bias_total FROM global_drift_state"
    ).fetchone()["global_q_bias_total"]
    assert global_total == pytest.approx(N * BOOST, abs=1e-9)

    rows = {r["group_id"]: dict(r) for r in conn.execute("SELECT * FROM group_parameters")}
    for gid, coef in COMPETITOR_REACTIVITY_Q_BIAS.items():
        assert rows[gid]["drift_q_bias_total"] == pytest.approx(N * coef * BOOST, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# Config integrity: no silent orphans between the two drift dicts
# ──────────────────────────────────────────────────────────────────────────

def test_reactivity_keys_match_group_preference_drift_keys():
    """COMPETITOR_REACTIVITY_Q_BIAS must cover exactly the same groups as
    GROUP_PREFERENCE_DRIFT — otherwise a group would either be missed by
    the event hook or receive a shock without a corresponding daily-drift
    entry (silent mis-config)."""
    reactivity = set(COMPETITOR_REACTIVITY_Q_BIAS.keys())
    drift = set(GROUP_PREFERENCE_DRIFT.keys())
    initial = set(INITIAL_CUSTOMER_GROUPS.keys())

    # No group is present in one dict but missing from the other
    assert reactivity == drift, (
        f"orphan groups between reactivity and drift dicts: "
        f"only in reactivity={reactivity - drift}, only in drift={drift - reactivity}"
    )
    # Initial groups are fully covered
    assert initial.issubset(reactivity)
    # Coefficients are non-negative (negative reactivity would invert shocks)
    for gid, coef in COMPETITOR_REACTIVITY_Q_BIAS.items():
        assert coef >= 0.0, f"{gid} coef must be non-negative, got {coef}"
