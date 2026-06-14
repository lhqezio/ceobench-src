"""Regression tests for CEOBench Arena shared-market primitives."""

import pytest
from numpy.random import default_rng

from saas_bench.arena import (
    ArenaCompanyMarketState,
    ArenaPlanOffer,
    CompanyExposure,
    CustomerChoiceProfile,
    SharedArrival,
    allocate_shared_arrivals,
    choose_company_plan,
    choose_for_shared_arrival,
    compute_group_arrival_rates,
    compute_required_quality,
    filter_offers_for_consideration_set,
    make_company_specs,
    plan_offers_from_company_config,
    sample_shared_arrivals,
)
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.simulation import Simulator


class FakeRng:
    def __init__(self):
        self.poisson_lams = []
        self.choice_probabilities = []
        self.choice_results = [1, 0, 1]
        self.random_results = [1.0, 1.0, 1.0]

    def poisson(self, lam):
        self.poisson_lams.append(lam)
        return 3

    def choice(self, a, p=None):
        self.choice_probabilities.append(tuple(p))
        return self.choice_results.pop(0)

    def random(self):
        return self.random_results.pop(0)


def test_company_specs_are_deterministic_and_keep_novamind_for_single_company():
    assert [(spec.company_id, spec.display_name) for spec in make_company_specs(1)] == [
        ("company_0", "NovaMind")
    ]

    specs = make_company_specs(
        3,
        agent_models=["gpt-5", "claude-opus-4.1", "gemini-2.5-pro"],
    )
    assert [(spec.company_id, spec.display_name) for spec in specs] == [
        ("company_0", "NovaMind"),
        ("company_1", "AsterAI"),
        ("company_2", "LatticeWorks"),
    ]
    assert [spec.agent_model for spec in specs] == [
        "gpt-5",
        "claude-opus-4.1",
        "gemini-2.5-pro",
    ]


def test_arena_choice_matches_single_company_ceobench_plan_choice(tmp_path):
    config = BenchmarkConfig(seed=123, base_product_quality=0.5)
    conn = init_database(tmp_path / "ceobench.db")
    sim = Simulator(conn, config, default_rng(123))
    sim.initialize()
    ceobench_config = {
        "price_A": 10.0,
        "price_B": 25.0,
        "price_C": 40.0,
        "tier_A": 1,
        "tier_B": 3,
        "tier_C": 5,
    }
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )

    ceobench_plan = sim._select_best_plan(
        steepness_left=profile.steepness_left,
        steepness_right=profile.steepness_right,
        c_max=profile.c_max,
        config=ceobench_config,
        overload=0.0,
        outage=False,
        q_max=profile.q_max,
        q_min=profile.q_min,
    )
    arena_choice = choose_company_plan(
        profile,
        plan_offers_from_company_config(
            company_id="company_0",
            display_name="NovaMind",
            config=ceobench_config,
            base_product_quality=config.base_product_quality,
        ),
    )

    assert ceobench_plan == "C"
    assert arena_choice.company_id == "company_0"
    assert arena_choice.plan == ceobench_plan
    assert arena_choice.chose_product


def test_shared_market_choice_selects_best_company_plan_pair():
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )
    offers = [
        ArenaPlanOffer(
            company_id="company_0",
            display_name="NovaMind",
            plan="A",
            price=20.0,
            tier=1,
            base_product_quality=0.5,
        ),
        ArenaPlanOffer(
            company_id="company_1",
            display_name="AsterAI",
            plan="A",
            price=30.0,
            tier=5,
            base_product_quality=0.5,
        ),
    ]

    choice = choose_company_plan(profile, offers)

    assert choice.company_id == "company_1"
    assert choice.display_name == "AsterAI"
    assert choice.plan == "A"


def test_customer_can_choose_no_product():
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )
    offers = [
        ArenaPlanOffer(
            company_id="company_0",
            display_name="NovaMind",
            plan="A",
            price=101.0,
            tier=5,
            base_product_quality=1.0,
        ),
        ArenaPlanOffer(
            company_id="company_1",
            display_name="AsterAI",
            plan="A",
            price=10.0,
            tier=1,
            base_product_quality=0.1,
        ),
    ]

    choice = choose_company_plan(profile, offers)

    assert not choice.chose_product
    assert choice.company_id is None


def test_required_quality_matches_current_simulator_formula(tmp_path):
    config = BenchmarkConfig(seed=123)
    conn = init_database(tmp_path / "ceobench.db")
    sim = Simulator(conn, config, default_rng(123))
    sim.initialize()
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )

    for cost in (0.0, 10.0, 50.0, 75.0, 100.0, 101.0):
        assert compute_required_quality(cost, profile) == pytest.approx(
            sim._compute_required_quality(
                cost=cost,
                steepness_left=profile.steepness_left,
                steepness_right=profile.steepness_right,
                c_max=profile.c_max,
                q_max=profile.q_max,
                q_min=profile.q_min,
            )
        )


def test_shared_arrivals_sum_company_exposure_into_one_pool():
    exposures = [
        CompanyExposure("company_0", "S1", 2.0),
        CompanyExposure("company_1", "S1", 3.0),
    ]
    rng = FakeRng()

    arrivals = sample_shared_arrivals(exposures, rng)

    assert compute_group_arrival_rates(exposures) == {"S1": pytest.approx(5.0)}
    assert rng.poisson_lams == [pytest.approx(5.0)]
    assert rng.choice_probabilities == [
        (pytest.approx(0.4), pytest.approx(0.6)),
        (pytest.approx(0.4), pytest.approx(0.6)),
        (pytest.approx(0.4), pytest.approx(0.6)),
    ]
    assert [arrival.source_company_id for arrival in arrivals] == [
        "company_1",
        "company_0",
        "company_1",
    ]
    assert [arrival.consideration_set for arrival in arrivals] == [
        ("company_1",),
        ("company_0",),
        ("company_1",),
    ]


def test_consideration_set_filters_company_plan_offers():
    config = {
        "price_A": 20.0,
        "price_B": 40.0,
        "price_C": 60.0,
        "tier_A": 1,
        "tier_B": 3,
        "tier_C": 5,
    }
    offers = []
    for company_id, display_name in (
        ("company_0", "NovaMind"),
        ("company_1", "AsterAI"),
    ):
        offers.extend(
            plan_offers_from_company_config(
                company_id=company_id,
                display_name=display_name,
                config=config,
                base_product_quality=0.5,
            )
        )

    filtered = filter_offers_for_consideration_set(offers, {"company_1"})

    assert len(filtered) == 3
    assert {offer.company_id for offer in filtered} == {"company_1"}
    assert [offer.plan for offer in filtered] == ["A", "B", "C"]


def test_company_market_state_adapts_ceobench_abc_config_for_group():
    state = ArenaCompanyMarketState(
        company_id="company_1",
        display_name="AsterAI",
        config={
            "price_A": 20.0,
            "price_B": 40.0,
            "price_C": 60.0,
            "tier_A": 1,
            "tier_B": 3,
            "tier_C": 5,
        },
        base_product_quality=0.5,
        q_shared_bonus=0.02,
        q_group_bonuses={"S1": 0.03},
        lead_promotions_by_group={"S1": 5.0},
    )

    offers = state.offers_for_group("S1")

    assert [offer.plan for offer in offers] == ["A", "B", "C"]
    assert {offer.company_id for offer in offers} == {"company_1"}
    assert offers[0].effective_price == pytest.approx(15.0)
    assert offers[0].delivered_quality == pytest.approx((0.5 + 0.02 + 0.03) * 0.6)


def test_shared_arrival_allocates_to_best_considered_company_plan():
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )
    states = [
        ArenaCompanyMarketState(
            company_id="company_0",
            display_name="NovaMind",
            config={
                "price_A": 20.0,
                "price_B": 40.0,
                "price_C": 60.0,
                "tier_A": 1,
                "tier_B": 1,
                "tier_C": 1,
            },
            base_product_quality=0.5,
        ),
        ArenaCompanyMarketState(
            company_id="company_1",
            display_name="AsterAI",
            config={
                "price_A": 30.0,
                "price_B": 50.0,
                "price_C": 70.0,
                "tier_A": 5,
                "tier_B": 5,
                "tier_C": 5,
            },
            base_product_quality=0.5,
        ),
    ]
    arrival = SharedArrival(
        group_id="S1",
        source_company_id="company_0",
        consideration_set=("company_0", "company_1"),
    )

    choice = choose_for_shared_arrival(arrival, profile, states)

    assert choice.company_id == "company_1"
    assert choice.display_name == "AsterAI"
    assert choice.plan == "A"


def test_shared_arrival_respects_consideration_set_and_no_product_choice():
    profile = CustomerChoiceProfile(
        group_id="S1",
        steepness_left=0.8,
        steepness_right=1.6,
        c_max=100.0,
        q_max=0.8,
        q_min=0.2,
    )
    states = [
        ArenaCompanyMarketState(
            company_id="company_0",
            display_name="NovaMind",
            config={
                "price_A": 20.0,
                "price_B": 40.0,
                "price_C": 60.0,
                "tier_A": 1,
                "tier_B": 1,
                "tier_C": 1,
            },
            base_product_quality=0.1,
        ),
        ArenaCompanyMarketState(
            company_id="company_1",
            display_name="AsterAI",
            config={
                "price_A": 30.0,
                "price_B": 50.0,
                "price_C": 70.0,
                "tier_A": 5,
                "tier_B": 5,
                "tier_C": 5,
            },
            base_product_quality=0.8,
        ),
    ]
    arrival = SharedArrival(
        group_id="S1",
        source_company_id="company_0",
        consideration_set=("company_0",),
    )

    allocations = allocate_shared_arrivals(
        [arrival],
        profiles_by_group={"S1": profile},
        company_states=states,
    )

    assert len(allocations) == 1
    assert allocations[0].arrival == arrival
    assert not allocations[0].choice.chose_product
