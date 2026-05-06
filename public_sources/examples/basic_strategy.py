"""Basic CEOBench strategy — a simple example agent.

Run with:
    ./novamind-operation python examples/basic_strategy.py

This script runs a simple strategy for one day:
1. Sets reasonable prices
2. Allocates spending
3. Checks metrics
"""
import novamind_api as nm

print(f"=== Day {nm.vars.current_day} Strategy ===\n")

# Set tiered pricing
nm.pricing.set_prices(A=19, B=59, C=149)
print("✅ Prices set: A=$19, B=$59, C=$149")

# Allocate daily spending
nm.marketing.set_daily_spend(
    advertising=3000,
    operations=2000,
    development=2000,
)
print("✅ Daily spend: ads=$3K, ops=$2K, dev=$2K")

# Set infrastructure capacity
nm.infrastructure.set_capacity_tier(tier=3)
print("✅ Capacity tier: 3")

# Check current subscriber count
result = nm.query(
    "SELECT COUNT(*) as total, "
    "SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active "
    "FROM subscriptions"
)
row = result['rows'][0] if result['rows'] else {'total': 0, 'active': 0}
print(f"\n📊 Subscribers: {row['active']} active / {row['total']} total")

# Check cash
result = nm.query("SELECT SUM(amount) as cash FROM ledger")
cash = result['rows'][0]['cash'] if result['rows'] and result['rows'][0]['cash'] else 0
print(f"💰 Cash: ${cash:,.0f}")

# Rationale is now a required argument of `next-week` itself — no separate
# log_rationale call. After running this script, advance the week with:
#
#   ./novamind-operation next-week \
#       "Day N: Basic strategy — moderate pricing, balanced spending." \
#       <12 cash forecasts>
#
# See docs/cli.md for the full forecast schema.
print("\n✅ Strategy applied. Pass a rationale + 12 forecasts to ./novamind-operation next-week to advance.")
