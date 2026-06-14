# CEOBench Arena Design

CEOBench Arena is the multi-company extension of CEOBench. The goal is to let
multiple agents run companies in the same market while preserving CEOBench's
core mechanisms: delayed investment, daily simulation dynamics, customer
quality-price curves, noisy demand, distribution shift, R&D, pricing, marketing,
infrastructure, enterprise leads, and forecasting.

The design principle is:

```text
Arena changes the funnel and market structure.
CEOBench satisfaction still decides whether a customer buys.
```

Arena should not become a separate hand-tuned market game. It should generalize
ordinary CEOBench from one company with plans A/B/C to N companies, each with
plans A/B/C, competing for the same customers.

## 1. Inspiration From Existing Arena Benchmarks

Vending Bench Arena is useful because it turns a single-agent business simulator
into a competitive environment where agents can affect one another through a
shared market. CEOBench Arena borrows that high-level structure:

- multiple independent agents run in parallel
- each agent controls one company
- companies share an external market
- outcomes depend on relative performance, not only absolute progress
- runs synchronize at a common time boundary

CEOBench Arena should differ in important ways:

- CEOBench remains a coding and data-analysis benchmark, not only a tool-call
  game.
- Agents still use the ordinary `novamind-operation` interface.
- The weekly agent decision cadence remains intact, but the simulator advances
  day by day internally.
- Customers should be modeled with CEOBench's existing quality-price curve,
  not with an LLM counterparty promise system.
- Freeform messages should not change simulator state unless paired with a
  structured mechanism.
- Market dynamics should be less hackable than aggregate demand because each
  customer has latent parameters and evaluates actual company-plan offers.

The useful Arena idea is shared competition. The CEOBench-specific advantage is
that long-horizon investments, delayed R&D, pricing, service quality, customer
heterogeneity, and distribution shift remain the source of difficulty.

## 2. Current CEOBench Mechanisms And Decision Space

Ordinary CEOBench has one company, NovaMind. The agent makes business decisions
through code and API calls, then advances the simulation with:

```bash
./novamind-operation next-week "<rationale>" <12 cash forecast numbers>
```

From the agent's perspective, CEOBench advances by week. Internally,
`step_week()` calls `step_day()` seven times, so the simulator has daily
dynamics.

### Agent Decisions

The agent controls the company through the public CEOBench API:

- pricing for plans A/B/C
- model tier for each plan
- usage quota for each plan
- operations, development, and infrastructure spending
- ad spend by channel and customer segment
- promotions by group, channel, or customer
- R&D projects
- segment research and discovery
- social media posts
- enterprise negotiations
- scripts and analysis files in the workspace
- weekly cash forecasts and rationale

These should remain the decision space in Arena. Arena should add shared-market
context and optional company-to-company primitives without removing the
ordinary CEOBench controls.

### Customer Acquisition

Current CEOBench does not call this "exposure", but the expected lead
calculation is the single-company version of exposure:

```text
daily_leads[group] =
  reputation_factor[group]
* demand_multiplier[group]
* cycle_multiplier
* macro_lead_multiplier[group]
* social_media_multiplier[group]
* surge_multiplier
* (paid_channel_leads[group] + network_leads[group])
```

Then:

```text
n_new[group] ~ Poisson(daily_leads[group])
```

Paid channel leads come from targeted ad spend and channel productivity.
Network leads come from existing subscribers and the network influence matrix.
Demand multiplier is market saturation against the segment's market cap. Macro,
seasonality, social media influence, and demand surges all multiply the flow.

### Customer Choice

For each new lead, CEOBench samples latent customer parameters including the
quality-price curve:

```text
steepness_left
steepness_right
c_max
q_min
q_max
usage_demand
seat_count
```

For each plan, CEOBench computes an effective price and perceived quality. The
purchase test is:

```text
satisfaction = Q_perceived - Q_required_customer(effective_price)
```

If the best plan is acceptable, an individual customer subscribes. Otherwise,
the lead is lost. Enterprise customers become leads if the quality gate is
viable and then proceed through the enterprise negotiation system.

For existing subscribers, CEOBench already has a richer perceived-quality
calculation:

```text
Q_perceived =
  delivered_quality
+ relationship_bonus
+ stickiness_bonus
- quota_penalty
- issue_penalty
- ads_penalty
```

Arena should reuse this idea. The customer decision mechanism should not be
replaced with a new "surplus" or "attractiveness" score.

### Distribution Shift

CEOBench already has non-stationary dynamics:

- preference drift in customer parameters
- macroeconomic conditions
- social media feedback
- demand surges
- competitor events that raise customer expectations
- market cap growth over time
- reputation changes from customer experience

Arena should preserve these mechanics and generalize the market pieces across
companies.

## 3. CEOBench Arena Features

### Company Identity

Arena companies use deterministic internal IDs and public display names:

```text
company_0 -> NovaMind
company_1 -> AsterAI
company_2 -> LatticeWorks
company_3 -> HelioStack
...
```

Single-company Arena should behave like ordinary CEOBench except for small
implementation differences. In practice, `--arena --arena-companies 1` should
fall back to the ordinary CEOBench path so NovaMind identity and single-player
mechanics are preserved.

### Same Agent Interface

Agents should still see the ordinary CEOBench game:

```text
read docs
start or receive a session
query/analyze data
call CEOBench API tools
call ./novamind-operation next-week
repeat
```

In multi-company Arena, the `next-week` command is intercepted by the Arena
coordinator. The agent still submits one weekly decision. Internally, the
coordinator advances all companies through seven hidden daily simulation steps
and shared market allocations.

### Shared Market

All companies compete in one market. Arena should not run one independent
market per company.

For each simulated day and customer group:

```text
exposure[company, group] =
  reputation[company, group]
* shared_saturation[group]
* cycle_multiplier
* macro_lead_multiplier[group]
* social_media_multiplier[company, group]
* surge_multiplier
* (
    paid_channel_leads[company, group]
  + network_leads[company, group]
  + optional_public_launch_attention[company, group]
  )
```

Then:

```text
N_customers[group] ~ Poisson(sum_company exposure[company, group])
```

Each arriving customer has a source company sampled proportional to exposure.
The source company is always in the consideration set. Other companies may
enter the consideration set based on their relative exposure/visibility.

### Multi-Company Customer Choice

For each arriving customer, Arena evaluates company-plan alternatives:

```text
for company in consideration_set:
  for plan in A, B, C:
    compute Q_perceived[customer, company, plan]
    compute Q_required_customer(price[company, plan])
    satisfaction = Q_perceived - Q_required_customer(...)
```

The customer chooses:

```text
if no company-plan offer is acceptable:
    choose no product
else:
    choose the acceptable company-plan offer with highest satisfaction
```

This is the multi-company version of the existing CEOBench quality-price curve.
It should preserve CEOBench parameters wherever possible.

For new individual leads, relationship/stickiness/issue terms are naturally
neutral. The key terms are:

- delivered quality
- group-specific quality bonus
- plan tier multiplier
- effective price after relevant lead promotion
- quota penalty
- initial perceived-quality noise
- customer curve parameters

For existing subscribers and future switching, the fuller formula applies:

```text
Q_perceived[customer, company, plan] =
  delivered_quality[company, group, plan]
+ relationship_bonus[customer, company]
+ stickiness_bonus[customer, company]
- quota_penalty[customer, company, plan]
- issue_penalty[customer, company]
- ads_penalty[customer, company]
```

### No Product

Customers must be able to choose no product. This preserves the ordinary
CEOBench behavior where leads are lost if no plan clears the participation
curve.

In Arena, no-product should be recorded as a market/customer outcome. It should
not count as a lost lead for every company in the consideration set. The current
implementation records the lost lead against the source company as a practical
v1 approximation.

### Market Saturation

Market saturation should be shared:

```text
total_active_customers[group] =
  sum_company active_customers[company, group]

market_cap_t[group] =
  base_market_cap[group] * (1 + annual_cap_growth_rate[group] * day / 365)

saturation[group] =
  max(0, 1 - (total_active_customers[group] / market_cap_t[group])^2)
```

This means one company can make acquisition harder for everyone by capturing a
segment early.

### First-Mover Advantage

First-mover advantage should emerge from existing CEOBench mechanisms:

- earlier R&D creates higher delivered quality sooner
- better quality raises conversion
- more customers create more network referrals
- good experiences improve reputation and social proof
- enterprise wins create lock-in before rivals catch up
- public launches may create temporary awareness, not direct utility

Arena should avoid adding a direct `first_mover_bonus` to satisfaction.

### Catch-Up Pressure

Catch-up pressure should operate through expectations, not direct penalties.

When a company publicly demonstrates that a feature frontier is possible, the
market can gradually raise expectations for relevant segments. Mechanically,
this should reuse the CEOBench competitor-event style of shifting customer
quality requirements rather than making rivals' R&D cheaper by default.

For v1, actual rival companies do not yet create these expectation events.

### Market Expansion

The first version should rely on summed CEOBench exposure:

```text
all companies' ads + all companies' referrals + all public social attention
  -> larger total arrival flow
```

An optional v2 category-awareness state could multiply arrival volume:

```text
arrival_rate[group] *= category_awareness[group]
```

Category awareness should affect the funnel, not customer utility.

### Public Data

Companies should eventually see public competitor data:

- public company names
- visible plans, prices, tiers, and quotas
- public social posts
- public launch or research-share artifacts
- public market summaries
- public interaction artifacts

Arena v1 should not add a special stealth mechanism. If a company does not
publish data, other companies should only see what ordinary public snapshots or
market outcomes reveal.

### Interaction Primitives

Arena can expose generic primitives that enable collaboration, comarketing, and
other emergent behavior without hard-coding special deals:

- email company
- transfer money
- share research artifact
- introduce customer or lead

Freeform text should not alter market state by itself. Structured primitives
can alter state only when the simulator has an explicit deterministic rule for
that primitive.

Possible future deterministic effects:

- money transfer debits sender ledger and credits recipient ledger
- customer introduction increases the recipient's chance of entering that
  customer's consideration set
- public research share creates a bounded research credit for recipients that
  are behind
- email creates only observable communication

### Antitrust And Collusion

Arena should not try to deterministically classify price collusion,
monopolistic behavior, or deceptive intent in v1. Those judgments are difficult
to make robustly from logs.

The deterministic design should instead:

- avoid freeform promises changing state
- make economic transfers structured and auditable
- keep customer choice mechanistic
- keep public/private data boundaries clear
- avoid hidden rewards for suspicious communication

Any later policy analysis should be post-run analysis, not a hidden simulator
score.

## 4. Mechanisms Extended In Arena

### Weekly Interface, Daily Simulation

Ordinary CEOBench:

```text
agent calls next-week
simulator runs step_week()
step_week() calls step_day() seven times
```

Arena:

```text
each agent calls next-week
coordinator waits for all companies
for each of 7 hidden days:
    each company enters ordinary CEOBench step_day()
    each company reaches the normal customer-acquisition slot
    coordinator computes shared market state
    coordinator samples shared arrivals
    coordinator allocates customers across companies/plans/no-product
    each company receives its own CEOBench generation result
    each company continues the ordinary day with billing, usage, costs, etc.
return one weekly dashboard to each agent
```

This preserves the ordinary weekly decision cadence and the ordinary daily
ordering. Arena changes who owns acquisition, not where acquisition happens in
the CEOBench day.

### Exposure

Arena currently lifts the CEOBench lead-generation ingredients into
`_compute_lead_exposure_by_group`. The intended long-term implementation is to
factor ordinary CEOBench and Arena through one shared exposure helper so the
mechanisms cannot drift.

Arena changes only the saturation count:

```text
ordinary CEOBench saturation uses own subscribers
Arena saturation uses total subscribers across all companies
```

Network referrals remain company-specific because a company's own subscribers
generate that company's referrals.

### Customer Choice

Arena extends CEOBench plan choice from:

```text
plans A/B/C within NovaMind
```

to:

```text
company-plan pairs across the consideration set
```

Arena asks each company simulator to evaluate its own A/B/C offers, then the
coordinator only chooses among those already-evaluated offers:

```text
evaluate_customer_offer(customer, company, plan, context)
```

The implementation now factors the lead-offer terms through
`Simulator._evaluate_lead_plan_offer_terms`, which is also used by the ordinary
CEOBench acquisition path for the selected lead offer. Arena should continue to
avoid owning a separate customer utility formula.

### Enterprise Customers

Arena v1 implements enterprise arrivals as competitive standing-offer RFPs:

```text
enterprise arrival creates a consideration set
considered companies' current A/B/C offers are evaluated by their simulators
customer evaluates offers using CEOBench satisfaction terms
best acceptable offer wins
if none are acceptable, no product
winner receives an ordinary CEOBench enterprise lead/thread
```

This keeps the enterprise negotiation system intact after the winner is chosen.
A later version can add simultaneous multi-turn enterprise threads where all
considered companies submit structured offers. Freeform promises should not
alter contract state unless they become structured offers or contracts.

### Research Sharing

Current CEOBench research improves product quality through ordinary R&D and
group research. Arena should not bypass this with arbitrary quality grants.

Research sharing has a bounded information effect:

```text
if sender_info_level[group] > recipient_info_level[group]:
    recipient_info_level[group] += 1
```

The credit improves the recipient's group research state by at most one level
toward the sender's state. It does not directly set final product quality.

### Money Transfers

Structured money transfer should be a ledger operation:

```text
sender:   arena_transfer_out = -amount
receiver: arena_transfer_in  = +amount
```

Transfer IDs are idempotent so retries cannot duplicate funds. Cash-availability
checks are a future tightening.

### Customer Introductions

Customer introductions should affect visibility, not force purchases. A
recipient company can become part of a customer's consideration set or receive a
structured lead, but the customer still evaluates offers with the CEOBench
quality-price curve.

### Cross-Company Switching

Arena lets existing individual subscribers compare outside offers on billing
renewal days.

The current implementation uses CEOBench satisfaction only: a subscriber can
switch if the best acceptable rival offer has higher satisfaction than the
incumbent's current satisfaction. Relationship/stickiness and enterprise
contract switching are future extensions.

## 5. Implementation Plan

### Current Implementation Status

Implemented:

- multi-company bash-agent Arena runner
- deterministic company IDs and names
- `--arena`, `--arena-companies`, and `--arena-models` for the bash-agent path
- single-company Arena parity by falling back to ordinary CEOBench
- weekly barrier with all companies submitting `next-week`
- hidden daily Arena advancement inside the weekly barrier
- shared acquisition returned inside CEOBench's ordinary `step_day()`
  customer-acquisition slot
- shared market exposure by group
- shared saturation using total subscribers across companies
- daily shared arrivals sampled from summed exposure
- consideration sets with source company plus exposure-based rivals
- multi-company A/B/C plan choice using simulator-evaluated CEOBench offer terms
- no-product option
- shared lead/customer insertion into the winning company's CEOBench DB
- hidden persistent Arena allocation logs for post-run analysis
- public competitor snapshots in `arena_public_market_snapshots`
- rebuilt public `novamind-operation` bundle with hidden Arena endpoints
- in-memory interaction primitive classes and tests
- coordinator-backed live Arena SDK module at `novamind_api.arena`
- live email, money-transfer, research-share, and customer-introduction event
  routing through the Arena coordinator
- deterministic money-transfer ledger effects with idempotent hidden transfer
  application logs
- bounded research-share effects: recipients can gain at most one group-info
  level toward the sender's level for the shared group
- customer introductions add one-use consideration-set visibility
- enterprise arrivals use competitive standing-offer RFP allocation before
  inserting an ordinary CEOBench enterprise lead for the winner
- individual subscribers can switch across companies on renewal days when a
  rival's CEOBench-evaluated offer beats current satisfaction
- `--arena --continue-from` resumes multi-company bash-agent Arena runs
- public `novamind-operation next-week` forwards to an Arena coordinator when
  `CEOBENCH_ARENA_COMPANY_ID` and `CEOBENCH_ARENA_COORDINATOR_PORT` are set,
  so non-bash harnesses can keep using the same company-side CLI contract
- public `arena-init`, `arena-start`, and `arena-stop` commands create ordinary
  CEOBench company workspaces and run the shared Arena coordinator for external
  harnesses
- ordinary CEOBench and Arena share lead-offer term evaluation through
  `Simulator._evaluate_lead_plan_offer_terms`

Partially implemented:

- exposure helper exists but duplicates ordinary CEOBench logic instead of
  being the single shared implementation
- customer choice asks each company simulator to evaluate offers with delivered
  quality, group quality, lead promotions, quota penalties, quality noise,
  required quality, and satisfaction; ordinary CEOBench still keeps its
  historical plan-selection path before using the shared offer-term helper
- competitive enterprise RFPs are standing-offer allocations today; simultaneous
  multi-turn RFP threads remain future work
- customer introductions affect consideration sets but do not create guaranteed
  purchases or persistent relationship state

Missing:

- cross-company enterprise renewals/switching
- actual-company first-mover/catch-up expectation events
- category-awareness market expansion

### Recommended PR Sequence

1. Extract the shared exposure helper.

   Ordinary CEOBench and Arena should call the same functions for lead exposure.
   This prevents drift.

2. Tighten ordinary/Arena offer choice parity.

   Arena offer construction now uses simulator-evaluated terms. The remaining
   parity work is to make ordinary CEOBench and Arena share the same plan-choice
   path end to end, not only the final offer-term helper.

3. Persist Arena market allocation logs.

   Record source company, group, consideration set, chosen company, chosen plan,
   no-product outcomes, and lead insertion results. These logs should be hidden
   from agents during play but useful for analysis.

4. Add public market snapshots.

   Publish visible rival state through a public Arena API:

   ```python
   nm.arena.public_market()
   ```

5. Wire interaction primitives. [implemented]

   Add `nm.arena.send_email`, `nm.arena.transfer_money`,
   `nm.arena.share_research`, and `nm.arena.introduce_customer`, backed by a
   coordinator Arena event store.

6. Add deterministic economic effects for selected primitives. [implemented]

   Money transfer ledger effects and bounded group-info research-share credits
   are implemented. Customer introductions affect consideration-set visibility.

7. Add enterprise RFPs. [partially implemented]

   Standing-offer RFP allocation is implemented. Multi-turn competitive
   enterprise threads remain future work.

8. Add switching and public frontier dynamics. [partially implemented]

   Individual subscribers can compare outside offers at renewal moments. Public
   launches or demonstrated frontiers can raise segment expectations through
   CEOBench-style expectation drift in a later version.

9. Add public Arena packaging. [implemented]

   The public CLI can now create company workspaces, start a standalone Arena
   coordinator, and forward each company's ordinary `next-week` call through
   the coordinator.

## Running Arena Today

Arena support is available through the bash-agent runner:

```bash
uv run --env-file .env python -m saas_bench.agents.bash_agent.run_test \
  --arena \
  --arena-companies 2 \
  --arena-models anthropic:claude-sonnet-4-6,anthropic:claude-sonnet-4-6 \
  --days 500 \
  --workspace bash_agent_arena_runs \
  --seed 42
```

For one company:

```bash
uv run python -m saas_bench.agents.bash_agent.run_test \
  --arena \
  --arena-companies 1 \
  --days 500
```

This should behave like ordinary CEOBench.

External harnesses can use the public CEOBench interface directly:

```bash
./novamind-operation arena-init \
  --arena-dir /tmp/ceobench-arena \
  --companies 3 \
  --days 500 \
  --seed 42

./novamind-operation arena-start \
  --arena-dir /tmp/ceobench-arena

cd /tmp/ceobench-arena/companies/company_0
source arena.env
./novamind-operation next-week "<rationale>" \
  <cash_1wk_point> <cash_1wk_lower> <cash_1wk_upper> \
  <cash_4wk_point> <cash_4wk_lower> <cash_4wk_upper> \
  <cash_12wk_point> <cash_12wk_lower> <cash_12wk_upper> \
  <cash_26wk_point> <cash_26wk_lower> <cash_26wk_upper>
```

Each company workspace is an ordinary CEOBench public bundle. The only Arena
addition is `arena.env`, which identifies the company and coordinator.

## Non-Goals For V1

- Do not replace CEOBench customer choice with a new hand-tuned surplus metric.
- Do not let freeform messages directly alter simulator state.
- Do not add hidden deterministic collusion or misconduct scoring.
- Do not make first-mover advantage a direct satisfaction bonus.
- Do not make research sharing directly set product quality.
- Do not expose hidden simulator internals as public competitor data.

The Arena should remain a CEOBench extension: same core benchmark, same core
customer math, more companies in one market.
