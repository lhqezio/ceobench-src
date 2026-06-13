"""Regression tests for CEOBench customer choice and acceptance rules."""

import math

import pytest

from saas_bench.config import BenchmarkConfig


def test_quality_price_curve_and_acceptance_rule(make_initialized_sim):
    _conn, sim, _config = make_initialized_sim()

    curve_kwargs = {
        "steepness_left": 0.8,
        "steepness_right": 1.6,
        "c_max": 100.0,
        "q_max": 0.8,
        "q_min": 0.2,
    }
    required_at_free = sim._compute_required_quality(cost=0.0, **curve_kwargs)
    required_at_mid = sim._compute_required_quality(cost=50.0, **curve_kwargs)
    required_at_budget = sim._compute_required_quality(cost=100.0, **curve_kwargs)
    required_above_budget = sim._compute_required_quality(cost=101.0, **curve_kwargs)

    assert 0.2 <= required_at_free < required_at_mid < required_at_budget <= 0.8
    assert required_above_budget == pytest.approx(0.8)

    affordable_required_quality = sim._compute_required_quality(
        cost=75.0, **curve_kwargs
    )
    assert sim._plan_acceptable(
        quality=affordable_required_quality, cost=75.0, **curve_kwargs
    )
    assert not sim._plan_acceptable(
        quality=math.nextafter(affordable_required_quality, 0.0),
        cost=75.0,
        **curve_kwargs,
    )
    assert not sim._plan_acceptable(quality=1.0, cost=101.0, **curve_kwargs)

    assert sim._compute_satisfaction(
        quality=affordable_required_quality + 0.05,
        cost=75.0,
        **curve_kwargs,
    ) == pytest.approx(0.05)


def test_new_customer_selects_best_plan_or_no_plan(make_initialized_sim):
    config = BenchmarkConfig(seed=123, base_product_quality=0.5)
    _conn, sim, _config = make_initialized_sim(config=config, seed=123)

    plan_config = {
        "price_A": 10.0,
        "price_B": 25.0,
        "price_C": 40.0,
        "tier_A": 1,
        "tier_B": 3,
        "tier_C": 5,
    }

    assert (
        sim._select_best_plan(
            steepness_left=0.8,
            steepness_right=1.6,
            c_max=100.0,
            config=plan_config,
            overload=0.0,
            outage=False,
            q_max=0.8,
            q_min=0.2,
        )
        == "C"
    )

    unaffordable_config = {
        "price_A": 101.0,
        "price_B": 125.0,
        "price_C": 140.0,
        "tier_A": 1,
        "tier_B": 3,
        "tier_C": 5,
    }

    assert (
        sim._select_best_plan(
            steepness_left=0.8,
            steepness_right=1.6,
            c_max=100.0,
            config=unaffordable_config,
            overload=0.0,
            outage=False,
            q_max=0.8,
            q_min=0.2,
        )
        is None
    )
