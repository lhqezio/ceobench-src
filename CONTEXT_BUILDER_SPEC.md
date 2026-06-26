# Active Context Manager — Design Spec

## Problem

CEO-Bench agents operate on 500-day horizons with weekly decisions. The current
architecture wipes conversation context every week and relies on the model to
write freeform notes to MEMORY.md. This is the baseline: unstructured,
model-dependent, no causal tracking, no adaptive shaping.

We want a **persistent context construction function** that sits between the
environment and the model, maintains a compressed belief state, and emits the
right context for the current decision — not a memory store, not a retriever.

## Analogy

Kalman filter for LLM context. Takes noisy observations (weekly dashboard +
event stream), maintains a belief state (compressed world model), updates
incrementally each turn, discards noise, emits a context frame sized for the
current decision.

## Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌────────┐
│ environment │────▶│  ActiveContextManager │────▶│ model  │
│  (events,   │     │                      │     │        │
│   dashboard,│     │  belief_state (dict)  │     │  gets  │
│   db access)│     │  ├── company_state    │     │  built │
└─────────────┘     │  ├── market_state     │     │  ctx   │
                    │  ├── active_threads   │     └────────┘
                    │  ├── decision_history │
                    │  └── trajectory       │
                    │                      │
                    │  update() per turn    │
                    │  construct() per turn │
                    └──────────────────────┘
```

## Belief State Components

### 1. company_state (always emit, compressed)
- cash trajectory: [current, 4wk_ago, 12wk_ago, 26wk_ago] + trend (rising/falling/burning)
- MRR + growth rate (slope of last 8 weeks)
- subscriber count + net growth rate
- churn rate (rolling 4-week)
- runway: months until cash = 0 at current burn
- cost structure: ops/dev/ad spend breakdown
- capacity utilization (overload history, p95 latency trend)

### 2. market_state (emit when relevant)
- discovered groups + info level
- per-group: subscriber count, quality gap vs competitor, awareness level, preference drift direction
- competitor quality trajectory (inferred from competitor events)
- macro signal (if shocks detected)

### 3. active_threads (emit when pending)
- enterprise negotiations: thread_id, customer persona, offer history, days since last response, status
- open issues: count, oldest days_open, affected groups
- social media: recent post performance, viral threads

### 4. decision_history (compressed causal chains)
- not a log — a set of (decision, outcome) pairs
- "cut price A from $29 to $19 in week 12 → signups +40% in weeks 13-16, churn +5% in weeks 14-18"
- "raised dev spend on S1 in week 8 → quality bonus +0.03 by week 12"
- "rejected enterprise deal with E1 in week 20 → no further contact from E1"
- old decisions with no downstream effect → flushed
- decisions whose effects are still unfolding → kept with "pending" status

### 5. trajectory (phase detection)
- current phase: launch / growth / optimization / crisis / decline
- phase determines context shape (see below)

## Update Function

Called each week with: new dashboard + new event log entries + db access.

```python
def update(self, dashboard: str, events: list[Event], db_conn):
    # 1. Parse dashboard into structured fields
    # 2. Append to time series (cash, mrr, subs, churn)
    # 3. Process events:
    #    - signups/churn → update rates, update group state
    #    - config_change → record as decision point
    #    - negotiation → update active_threads
    #    - shock → flag in market_state, set recency priority
    #    - social_post → update reputation tracking
    # 4. Update decision_history: check if old decisions now have outcomes
    # 5. Update trajectory/phase
    # 6. Flush: drop resolved threads, saturated groups, stale decisions
```

## Construct Function

Called each week to build the context string the model sees.

```python
def construct(self, budget_tokens: int = 8000) -> str:
    # 1. Always: company_state (compressed, ~500 tokens)
    # 2. Phase-adaptive blocks:
    #    - crisis (runway < 3mo): cash burn breakdown, cost-cutting options,
    #      revenue projections
    #    - growth (subs growing > 5%/wk): acquisition channel performance,
    #      capacity headroom, pricing optimization signals
    #    - optimization (stable growth < 2%/wk): quality gaps per group,
    #      churn analysis, enterprise pipeline
    #    - launch (week 1-4): group discovery status, initial pricing feedback,
    #      early signups by group
    # 3. Active threads (if any pending)
    # 4. Recent decision outcomes (last 3-4 with new data)
    # 5. Anomalies: anything that deviated > 2 sigma from expectation
    # 6. Trim to budget
```

## What Makes This Different From RAG

| RAG | Active Context Manager |
|-----|----------------------|
| query → retrieve similar docs | no query — construct from state |
| semantic similarity | causal relevance |
| static index | incrementally updated belief |
| same shape every query | adaptive shape per phase |
| retrieval = the whole job | retrieval is never done — state is always warm |
| no temporal model | temporal model is core (trends, trajectories, phases) |
| no flushing | explicit flush of stale/irrelevant state |

## Benchmark Structure

### Conditions
1. **Oracle** — full context, no limit (ceiling)
2. **Active Context Manager** — bounded, constructed (ours)
3. **Truncation** — last N turns of raw conversation (naive baseline)
4. **Rolling summary** — compress older turns into a summary block (what a good MEMORY.md agent approximates)
5. **RAG** — semantic retrieval of "relevant" past events/tokens

### Metrics
1. **Decision quality ratio** — score(builder) / score(oracle). Headline number.
2. **Token efficiency** — tokens used vs oracle tokens. Pareto frontier plot.
3. **Causal relevance** — via counterfactual ablation: remove event X from history, re-run oracle, if score drops → X was relevant. Measure precision/recall of builder's inclusion.
4. **Temporal adaptivity** — does context shape change across phases? Measure KL divergence of context distributions across phases.
5. **Flush quality** — are flushed items truly irrelevant? Measure: do scores degrade when flushed items are force-retained? (If not, flush was correct.)

### Hook Point
Replace `_refresh_context()` in `bash_agent/agent.py`:

```python
# Before (baseline):
def _refresh_context(self, dashboard, new_day):
    self.conversation = []
    self.conversation.append(Message(role='system', content=system_prompt + MEMORY.md))

# After (with context manager):
def _refresh_context(self, dashboard, new_day):
    self.context_manager.update(dashboard, events, db_conn)
    constructed = self.context_manager.construct(budget_tokens=8000)
    self.conversation = []
    self.conversation.append(Message(role='system', content=system_prompt + constructed))
```

## Implementation Plan

### Phase 1: Instrument (1-2 days)
- Add event capture hook to bash_agent runner
- Log full event stream per week (already exists via event_logger, just wire it)
- Run baseline: raw Qwen 3.6 with default MEMORY.md agent on 50-day sim
- Run baseline: Claude Sonnet with default agent on 50-day sim
- Collect trajectory data for analysis

### Phase 2: Build belief state (2-3 days)
- Implement company_state, market_state, active_threads parsers
- Implement decision_history with causal chain tracking
- Implement phase detection
- Test against collected trajectory data — does the state accurately reflect what happened?

### Phase 3: Build construct function (2-3 days)
- Implement phase-adaptive context construction
- Token budget management
- Run against CEO-Bench, compare to baselines

### Phase 4: Benchmark (2-3 days)
- Implement all 5 conditions
- Run N=3-5 seeds per condition at 100-day, 250-day, 500-day horizons
- Generate comparison tables and plots
- Counterfactual ablation for causal relevance metric

### Phase 5: Iterate (ongoing)
- Where does the builder lose to oracle? analyze gaps
- Where does it beat truncation/summary/RAG? quantify
- Tune budget, phase detection thresholds, flush rules