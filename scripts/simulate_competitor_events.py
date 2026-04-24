"""Simulate competitor event schedule for a given config without running the full sim.

Reproduces the RNG stream used by SimulationEngine._process_competitor_events.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpy.random import Generator, PCG64
from saas_bench.config import BenchmarkConfig


def simulate(seed: int = 42, total_days: int = 500):
    config = BenchmarkConfig(seed=seed, total_days=total_days)

    # Reproduce simulation.py init draws from main rng (in order)
    rng = Generator(PCG64(seed))
    _macro_seed = int(rng.integers(0, 2**63))          # draw #1 — discarded here
    competitor_seed = int(rng.integers(0, 2**63))       # draw #2 — used for competitor rng

    competitor_rng = Generator(PCG64(competitor_seed ^ 0x434F4D50))  # XOR with 'COMP'

    grace = config.drift_grace_period_days            # 60
    late_cutoff = config.competitor_event_late_cutoff_days  # 30
    mean_base = config.competitor_event_mean_interval       # 3
    min_base = config.competitor_event_min_interval         # 1
    half_sim = max(total_days // 2, 1)                      # 250
    ramp_end_day = max(total_days - late_cutoff, 2)         # 470
    scale_min = config.competitor_event_magnitude_scale_min # 1.0
    scale_max = config.competitor_event_magnitude_scale_max # 4.0 (v3.4f)
    mu = config.competitor_event_boost_mu
    sigma = config.competitor_event_boost_sigma
    b_min = config.competitor_event_boost_min
    b_max = config.competitor_event_boost_max

    last_event_day = -mean_base  # matches simulation.py fallback
    posts_per_day = config.competitor_event_posts_per_day  # 2
    post_days = config.competitor_event_post_days  # 3

    events = []
    cumulative = 0.0
    # Track active events: list of post_end_day values. An event fired on day D has post_end_day = D + post_days
    active_post_end_days = []

    for day in range(1, total_days + 1):
        # --- _process_competitor_events (always called; may early-return before any RNG draw) ---
        fire = False
        can_fire = True
        if day < grace:
            can_fire = False
        if day > total_days - late_cutoff:
            can_fire = False

        if can_fire:
            mean_interval = mean_base
            min_interval = min_base
            if day < half_sim:
                mean_interval *= 1.5
                min_interval *= 1.5

            days_since = day - last_event_day
            if days_since >= min_interval:
                daily_prob = 1.0 / mean_interval
                draw = competitor_rng.random()
                if draw < daily_prob:
                    fire = True
                    raw = float(competitor_rng.lognormal(mu, sigma))
                    base_boost = max(b_min, min(raw, b_max))
                    day_frac = max(0.0, min((day - 1) / max(ramp_end_day - 1, 1), 1.0))
                    magnitude_scale = scale_min + (scale_max - scale_min) * day_frac
                    boost = base_boost * magnitude_scale

                    if boost < 0.03:
                        severity = "minor"
                    elif boost < 0.10:
                        severity = "moderate"
                    elif boost < 0.20:
                        severity = "major"
                    else:
                        severity = "transformative"

                    cumulative += boost
                    events.append({
                        "day": day,
                        "base_boost": base_boost,
                        "magnitude_scale": magnitude_scale,
                        "boost": boost,
                        "cumulative": cumulative,
                        "severity": severity,
                    })
                    last_event_day = day
                    active_post_end_days.append(day + post_days)

        # --- _generate_competitor_event_posts (called every day; draws from _competitor_rng
        # if there's any active event and _market_observer_id is set).
        # Per post: 2 integers + 3 random = 5 draws. Per day: posts_per_day * 5 = 10 draws.
        has_active = any(end >= day for end in active_post_end_days)
        if has_active:
            for _ in range(posts_per_day):
                competitor_rng.integers(0, 5)   # competitor_name choice
                competitor_rng.integers(0, 7)   # perspective choice
                competitor_rng.random()         # views multiplier
                competitor_rng.random()         # likes multiplier
                competitor_rng.random()         # shares multiplier

    return events, cumulative, config


def _fmt(events, cumulative):
    header = f"{'Day':>4}  {'Base':>8}  {'Scale':>6}  {'Boost':>8}  {'Cum':>8}  {'Severity':<14}"
    print(header)
    print("-" * len(header))
    for e in events:
        print(f"{e['day']:>4}  {e['base_boost']:>8.5f}  {e['magnitude_scale']:>6.2f}  {e['boost']:>8.5f}  {e['cumulative']:>8.4f}  {e['severity']:<14}")
    print("-" * len(header))
    print(f"Total events: {len(events)} | Cumulative boost: {cumulative:.4f}")


if __name__ == "__main__":
    events, cumulative, config = simulate(seed=42, total_days=500)
    print(f"Config: v3.4l — scale_max={config.competitor_event_magnitude_scale_max}, mean_interval={config.competitor_event_mean_interval} (0.5× from v3.4k=4.286; 0.35× from v3.4j=3), individual_ops_scale={config.individual_ops_scale}, enterprise_ops_scale={config.enterprise_ops_scale}")
    print(f"Grace: {config.drift_grace_period_days}d | late_cutoff: {config.competitor_event_late_cutoff_days}d | active window: [{config.drift_grace_period_days}, {500 - config.competitor_event_late_cutoff_days}]")
    print(f"Intervals — base mean/min: {config.competitor_event_mean_interval}/{config.competitor_event_min_interval}; first-half (×1.5): {config.competitor_event_mean_interval*1.5}/{config.competitor_event_min_interval*1.5}")
    print()
    _fmt(events, cumulative)
