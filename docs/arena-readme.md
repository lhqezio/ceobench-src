# CEOBench Arena README

CEOBench Arena is the multi-company extension of CEOBench. It lets multiple
agents run separate companies in one shared market while preserving the core
CEOBench benchmark: delayed R&D, noisy daily market dynamics, customer
quality-price curves, pricing, quotas, ad spend, enterprise leads, churn,
forecasting, and the weekly `next-week` interface.

The implementation principle is:

```text
Arena changes market allocation.
CEOBench still evaluates whether each company-plan offer is good enough.
```

Arena should not be read as a replacement benchmark. A one-company Arena run
should behave like ordinary CEOBench except for small implementation
differences. Multi-company Arena generalizes the same company and customer
mechanisms to N companies competing for the same customers.

## What Was Added

### Arena company identities

Arena companies are generated deterministically by `ArenaCompanySpec`:

- `company_0` -> `NovaMind`
- `company_1` -> `AsterAI`
- `company_2` -> `LatticeWorks`
- `company_3` -> `HelioStack`

Additional companies receive deterministic fallback names. A single-company
Arena keeps the ordinary `NovaMind` identity.

### Shared Arena coordinator

Arena adds a coordinator that sits above ordinary CEOBench company servers. Each
company still has its own CEOBench database, public `novamind-operation`
workspace, and agent loop. The coordinator handles shared-market behavior:

- registers company servers
- waits for every live company to submit `next-week`
- retires terminal companies from future barriers
- advances the simulation day by day behind the weekly interface
- samples shared acquisition arrivals
- allocates customers across companies
- writes public competitor snapshots
- applies structured interaction primitives
- keeps hidden allocation and switching audit logs

The agent-facing cadence stays the same: the agent reasons for a week, calls
`next-week`, and receives a weekly dashboard.

### Shared customer acquisition

Ordinary CEOBench computes expected leads from the company's ads, reputation,
network effects, social media, macro state, seasonality, shocks, and saturation.
Arena computes that same style of exposure for every company and segment, then
turns it into one shared market:

```text
arrivals[group] ~ Poisson(sum_company exposure[company, group])
```

Each customer has a source company sampled proportional to exposure. The source
company is always in the consideration set. Rival companies can also enter the
consideration set based on their exposure share, and customer introductions can
add one-use visibility.

### Source-aware customer choice

Arena uses CEOBench's existing offer evaluation. Each considered company exposes
its A/B/C plans, and the simulator evaluates each company-plan offer using the
same ingredients ordinary CEOBench uses:

- delivered product quality
- group-specific quality
- model tier
- plan quota and usage demand
- lead promotions
- customer quality noise
- customer `c_max`, `q_min`, `q_max`, and curve steepness
- effective price
- required quality
- satisfaction

The customer can choose no product. If no considered company-plan offer is
acceptable, the lead is lost.

Arena also adds a first-contact rule. If the source company's best offer is
acceptable, a rival must beat it by a hidden comparison hurdle. If the source
offer is not acceptable, the best acceptable rival can win. This keeps ads and
first exposure meaningful without letting low-quality ads force purchases.

### Shared saturation

Market saturation is shared across companies. Arena uses total active customers
across all companies in a segment when calculating acquisition difficulty, so
one company capturing a segment can make marginal acquisition harder for
everyone.

### Public competitor snapshots

Arena writes visible competitor state to `arena_public_market_snapshots`.
Agents can inspect this public market feed through:

```python
import novamind_api as nm

print(nm.arena.public_market())
```

Snapshots include public prices, tiers, quotas, and public subscriber summaries.
They do not expose hidden simulator internals.

### Interaction primitives

Arena adds a public `novamind_api.arena` module. These primitives allow
companies to interact without making freeform promises magically alter state:

- `nm.arena.get_inbox()`
- `nm.arena.send_email(recipient_company_id, subject, body)`
- `nm.arena.transfer_money(recipient_company_id, amount, memo="")`
- `nm.arena.share_research(artifact_id, scope=..., recipient_company_id=None, group_id=None, memo="")`
- `nm.arena.introduce_customer(recipient_company_id, customer_ref, group_id=None, memo="")`
- `nm.arena.public_market()`

Emails are message-only. Money transfers debit and credit company ledgers with
idempotent hidden application logs. Research sharing can move a recipient at
most one group-information level toward the sender's level for that group; it
does not directly change product quality. Customer introductions affect
consideration-set visibility; they do not force conversion.

### Cross-company switching

Arena can evaluate existing subscribers for cross-company switching at renewal
moments. A subscriber switches only when a rival offer, evaluated with CEOBench
offer terms, beats the current satisfaction by more than the switching hurdle.
The hurdle includes contract lock-in, relationship inertia, and hidden
customer-level noise.

### Enterprise allocation

Enterprise arrivals can be allocated competitively before inserting an ordinary
CEOBench enterprise lead for the winner. The current implementation is a
standing-offer allocation, not a simultaneous multi-turn RFP thread.

### Public CLI support

The public `novamind-operation` bundle now supports Arena setup for external
harnesses:

- `arena-init`
- `arena-start`
- `arena-stop`

Inside an Arena company workspace, the ordinary `next-week` command forwards to
the Arena coordinator when `arena.env` sets:

- `CEOBENCH_ARENA_COMPANY_ID`
- `CEOBENCH_ARENA_DISPLAY_NAME`
- `CEOBENCH_ARENA_COORDINATOR_PORT`

This is what lets external harnesses keep using the ordinary CEOBench contract.

## What Changed In Existing CEOBench Code

The main changed areas are:

- `src/saas_bench/arena/`: company specs, shared market helpers, coordinator,
  and interaction logs.
- `src/saas_bench/agents/bash_agent/arena_runner.py`: multi-company bash-agent
  orchestration and weekly barrier handling.
- `src/saas_bench/agents/bash_agent/run_test.py`: `--arena`,
  `--arena-companies`, `--arena-models`, and Arena resume support.
- `src/saas_bench/_public_cli.py`: public `arena-init`, `arena-start`,
  `arena-stop`, and Arena-aware `next-week` forwarding.
- `src/saas_bench/api_server.py`: hidden Arena endpoints used by the
  coordinator.
- `src/saas_bench/simulation.py`: Arena insertion, offer evaluation, shared
  snapshots, ad attribution, switching candidates, interaction effects, and
  shared competitor-event application.
- `src/saas_bench/database.py`: public Arena snapshot table and hidden Arena
  audit/application tables.
- `src/saas_bench/novamind_api/arena.py`: public Python helpers available to
  agents during Arena runs.
- `scripts/build_public.py`: includes Arena modules in the public bundle.

Ordinary CEOBench paths are intended to keep their existing behavior. Arena
logic is activated by Arena CLI flags or Arena environment variables.

## How To Run Arena With The Bash-Agent Harness

Example three-company run:

```bash
uv run --env-file .env python -m saas_bench.agents.bash_agent.run_test \
  --arena \
  --arena-companies 3 \
  --arena-models anthropic:claude-sonnet-4-6,anthropic:claude-sonnet-4-6,anthropic:claude-sonnet-4-6 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --reasoning-effort none \
  --days 504 \
  --seed 42 \
  --workspace bash_agent_arena_runs
```

`--arena-models` accepts comma-separated provider/model specs. Specs with an
explicit provider use `provider:model`; specs without a provider use the default
`--provider`.

For single-company parity:

```bash
uv run --env-file .env python -m saas_bench.agents.bash_agent.run_test \
  --arena \
  --arena-companies 1 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --days 504
```

## How To Run Arena With External Harnesses

Build the public bundle after source changes:

```bash
uv run python scripts/build_public.py
```

Create an Arena directory:

```bash
./public/novamind-operation arena-init \
  --arena-dir /tmp/ceobench-arena \
  --companies 3 \
  --days 504 \
  --seed 42
```

Start the shared coordinator:

```bash
./public/novamind-operation arena-start \
  --arena-dir /tmp/ceobench-arena
```

Each external agent then runs from its assigned company workspace:

```bash
cd /tmp/ceobench-arena/companies/company_0
source arena.env
./novamind-operation next-week "<rationale>" \
  <cash_1wk_point> <cash_1wk_lower> <cash_1wk_upper> \
  <cash_4wk_point> <cash_4wk_lower> <cash_4wk_upper> \
  <cash_12wk_point> <cash_12wk_lower> <cash_12wk_upper> \
  <cash_26wk_point> <cash_26wk_lower> <cash_26wk_upper>
```

Stop the coordinator:

```bash
./public/novamind-operation arena-stop \
  --arena-dir /tmp/ceobench-arena
```

If an external harness decides a company is terminal, for example because it
bankrupted or timed out, retire that company so the remaining companies can keep
advancing:

```bash
./novamind-operation arena-retire \
  --company-id company_0 \
  --outcome bankrupt
```

The bash-agent Arena runner does this automatically when a company thread
finishes.

## Useful Agent-Side Queries

Inspect the public market:

```bash
./novamind-operation python-c "import novamind_api as nm; print(nm.arena.public_market())"
```

Read the Arena inbox:

```bash
./novamind-operation python-c "import novamind_api as nm; print(nm.arena.get_inbox())"
```

Check local subscriber state:

```bash
./novamind-operation query "SELECT plan, COUNT(*) FROM subscriptions WHERE status='subscribed' AND end_day IS NULL GROUP BY plan"
```

## Hidden Analysis Tables

These tables are hidden from agents during play but useful for post-run
analysis:

- `_hidden_arena_allocation_log`: shared-market arrivals, source company,
  consideration set, chosen company, chosen plan, no-product outcomes, and
  evaluated offers.
- `_hidden_arena_switching_log`: cross-company subscriber switching decisions.
- `_hidden_arena_money_transfer_applications`: idempotent transfer application
  records.
- `_hidden_arena_research_share_applications`: idempotent bounded research-share
  application records.

The visible table `arena_public_market_snapshots` is intentionally available to
agents.

## Tests

Focused Arena tests:

```bash
uv run pytest tests/test_arena_shared_market.py tests/test_arena_coordinator.py tests/test_arena_interactions.py
```

Full local test suite:

```bash
uv run pytest tests
```

Public bundle build:

```bash
uv run python scripts/build_public.py
```

## Current Limitations

- Shared exposure still duplicates parts of ordinary CEOBench lead-exposure
  logic. Long term, ordinary CEOBench and Arena should call one shared helper.
- Competitive enterprise allocation is standing-offer based. Multi-company,
  multi-turn enterprise RFP threads are not implemented.
- Cross-company enterprise renewal and switching are not implemented.
- Actual-company first-mover and catch-up expectation events are not fully
  implemented. Current shared competitor expectation shocks are sampled once by
  the Arena coordinator and applied consistently across companies.
- Explicit category-awareness market expansion is not implemented.
- Interaction primitives are deliberately generic. They create structured state
  only where the mechanism is deterministic: money transfer, bounded research
  sharing, and customer-introduction visibility.

## Design Guardrails

- Do not replace CEOBench's quality-price customer choice with a new aggregate
  "surplus" metric.
- Do not let freeform emails or promises directly change customers, product
  quality, contracts, or market state.
- Do not expose hidden mechanism levers to companies.
- Do not add deterministic collusion or misconduct scoring unless it can be
  measured mechanically.
- Keep single-company Arena behavior as close as possible to ordinary CEOBench.
