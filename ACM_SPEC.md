# Active Context Manager (ACM) — Universal Framework Spec

## Thesis

Context for long-horizon agents is not a growing log to retrieve from. It is a
**maintained belief state** — continuously updated, phase-adaptive, flushed of
stale information, and constructed fresh each turn for the decision at hand.

This is state estimation, not storage. This is construction, not retrieval.

Existing approaches (RAG, conversation truncation, rolling summary, memory
files) all treat context as a storage problem. ACM treats it as an estimation
problem. We will prove this beats every existing approach on standard
long-horizon agent benchmarks.

## What ACM Is

A universal, model-agnostic middleware layer that sits between any agent
environment and any LLM:

```
any environment ──▶ ACM ──▶ any model
(observations,       │        (constructed
 events, actions)    │         context frame)
                     │
              belief_state
              (always warm)
```

- **Plug and play**: one adapter per environment, one adapter per model API
- **Model-agnostic**: works with OpenAI, Anthropic, local Qwen, anything that
  takes text context
- **Environment-agnostic**: works with CEO-Bench, web agents, coding agents,
  any long-horizon task that produces observations over time
- **No training required**: the framework itself is inference-time middleware.
  (A finetuned model can later exploit ACM's structured output, but ACM works
  with any base model first.)

## Universal Abstractions

Every long-horizon agent task has the same structure:

1. **State variables** that change over time
2. **Events** that occur (external, agent-caused)
3. **Decisions** the agent makes (with delayed consequences)
4. **A current situation** that determines what context is needed

ACM captures these in five universal components:

### 1. EntityTracker — "what exists and what's its state"

Tracks named entities and their values over time. Not a key-value store — a
time series with trend detection.

```
entity: cash
values: [1M (w0), 950K (w4), 1.1M (w12), 800K (w24)]
trend: falling (last 8 weeks)
anomaly: -200K drop at w20 (outside 2σ)

entity: subscriber_count_group_S1
values: [0, 12, 45, 89, 102, 98]
trend: plateauing
```

Any environment adapter registers entities it cares about. ACM maintains their
trajectory, computes trends, flags anomalies. The CEO-Bench adapter registers
cash, mrr, subs per group, quality per group, etc. A coding agent adapter
registers files changed, tests passing, build status, etc.

### 2. ActiveQueue — "what needs attention right now"

Items that require action or monitoring this turn. Auto-promoted from events,
auto-demoted when resolved or stale.

```
active:
  - enterprise_thread_42: offer sent, 12 days no reply, E1 group, 50 seats
  - churn_spike_S2: 8 cancellations this week (4σ above baseline)
  - capacity_warning: overload 15% last 2 weeks
flushed:
  - enterprise_thread_15: closed (accepted), removed
  - outage_w18: resolved, removed
```

### 3. DecisionLedger — "what did i do and what happened"

The causal memory. Each decision is tracked with its observed outcomes over
time. Not a log — a set of (decision → outcome) pairs with confidence.

```
decision: cut price A $29→$19 (w12)
outcomes:
  - signups +40% (w13-16, confirmed)
  - churn +5% (w14-18, confirmed)
  - net revenue -8% (w16, confirmed)
status: saturated (no new outcomes emerging)

decision: raised dev spend S1 +$200/day (w8)
outcomes:
  - quality bonus +0.03 (w12, confirmed)
  - subscriber growth S1 +15% (w14, emerging)
status: unfolding
```

Decisions with no observed downstream effect after N weeks → flushed.
Decisions whose effects are still unfolding → kept with higher priority.

### 4. PhaseDetector — "what mode am i in"

Classifies the current situation into a phase that determines context shape.
Phases are domain-general, detected from entity trajectories and active queue.

Universal phases:
- **explore**: early, high uncertainty, gathering information
- **grow**: positive trajectories, scaling up
- **optimize**: stable, fine-tuning, efficiency-focused
- **defend**: negative trajectories, crisis management, resource constrained
- **transition**: phase change detected, reorienting

Each phase weights context components differently:
- explore → entity discovery, early signal, what's unknown
- grow → capacity headroom, resource allocation, what's scaling
- optimize → quality gaps, efficiency metrics, marginal improvements
- defend → burn rate, cost-cutting options, critical path
- transition → what changed, why, what matters now

### 5. RecencyBuffer — "what just happened"

Raw last N observations for tactical detail that hasn't been processed into
the belief state yet. This is the working memory — small, FIFO, unprocessed.

## Environment Adapter Interface

```python
class EnvironmentAdapter(Protocol):
    """One adapter per environment. Tells ACM how to read the world."""

    def parse_observation(self, obs: str) -> dict:
        """Extract structured fields from a raw observation string."""

    def extract_events(self, obs: str, db_access) -> list[Event]:
        """Extract events from the latest observation + environment access."""

    def register_entities(self) -> list[EntitySpec]:
        """Declare which entities to track (name, source, parse_fn)."""

    def classify_action(self, action: dict) -> Decision | None:
        """Classify an agent action as a tracked decision (or None if trivial)."""

    def score(self, db_access) -> dict:
        """Return environment-specific score (cash, tests passed, etc.)."""
```

## Model Adapter Interface

```python
class ModelAdapter(Protocol):
    """One adapter per model API. Handles context injection."""

    def inject_context(self, messages: list, context_frame: str) -> list:
        """Insert the constructed context into the message stream."""

    def count_tokens(self, text: str) -> int:
        """Token counting for budget management."""

    def call(self, messages: list, tools: list) -> dict:
        """Make the LLM API call."""
```

Adapters: OpenAIChatAdapter, AnthropicAdapter, OpenAICompatAdapter (covers
Qwen, vLLM, llama.cpp, Ollama, etc.)

## ACM Core

```python
class ActiveContextManager:
    def __init__(
        self,
        env_adapter: EnvironmentAdapter,
        model_adapter: ModelAdapter,
        budget_tokens: int = 8000,
    ):
        self.entities = EntityTracker()
        self.active = ActiveQueue()
        self.decisions = DecisionLedger()
        self.phase = PhaseDetector()
        self.recency = RecencyBuffer(window=5)
        self.budget = budget_tokens

    def update(self, observation: str, action: dict | None, db_access):
        """Called each turn. Updates belief state from new observation."""
        parsed = self.env_adapter.parse_observation(observation)
        events = self.env_adapter.extract_events(observation, db_access)

        self.entities.update(parsed, events)
        self.active.update(events)
        if action:
            decision = self.env_adapter.classify_action(action)
            if decision:
                self.decisions.add(decision)
        self.decisions.check_outcomes(self.entities)
        self.phase.update(self.entities, self.active)
        self.recency.push(observation)
        self._flush()

    def construct(self) -> str:
        """Called each turn. Builds the context frame for the model."""
        phase = self.phase.current
        sections = []

        # Always: entity state (compressed)
        sections.append(self.entities.summarize())

        # Phase-adaptive
        sections.append(self.active.summarize(phase=phase))

        # Recent decisions with new outcomes
        sections.append(self.decisions.summarize_unfolding())

        # Anomalies (anything > 2σ)
        sections.append(self.entities.summarize_anomalies())

        # Raw recency buffer
        sections.append(self.recency.summarize())

        # Trim to budget
        return self._trim(sections, self.budget)

    def _flush(self):
        """Drop stale/irrelevant state."""
        self.active.flush_resolved()
        self.decisions.flush_saturated()
        self.entities.flush_old_anomalies()
```

## Benchmark — Proving The Thesis

### Testbeds (start with CEO-Bench, expand)

1. **CEO-Bench** — 500-day SaaS simulation, weekly decisions, rich event log
2. **WebArena / VisualWebArena** — long-horizon web tasks
3. **SWE-bench Multi-step** — multi-issue codebase tasks
4. **Custom: multi-session agent** — same agent across N disconnected sessions

### Conditions

1. **Oracle** — full context, no limit (ceiling)
2. **ACM** — ours, bounded belief state construction
3. **Truncation** — last N tokens of raw conversation
4. **Rolling summary** — older turns compressed into a running summary
5. **RAG** — semantic retrieval over past observations/events
6. **MemGPT-style** — tiered memory with model-managed context

### Metrics

1. **Decision quality ratio** — score(method) / score(oracle)
2. **Token efficiency** — tokens used vs oracle. Pareto frontier.
3. **Causal relevance** — counterfactual ablation: remove event X, re-run
   oracle, if score drops → X was causally relevant. Measure precision/recall
   of each method's inclusion of relevant events.
4. **Temporal adaptivity** — KL divergence of context distributions across
   phases. Higher = more adaptive. (Truncation/summary should be ~0, ACM
   should be high.)
5. **Flush precision** — of items flushed by ACM, what % were truly irrelevant
   (no counterfactual impact)? Target: >90%.
6. **Cross-model consistency** — does ACM's advantage hold across models?
   Run with Qwen-35B, Llama-70B, Claude Sonnet, GPT-4o. If ACM helps all
   models, it's a universal framework. If it only helps small models, it's a
   small-model crutch.

### The Headline Result We're Going For

> "ACM preserves 9X% of oracle decision quality at 1X% of the token budget,
> across [N] benchmarks and [M] models, outperforming RAG by [Δ] and rolling
> summary by [Δ]."

That table is the artifact. That's what goes in front of VCs, pilots, and
in the paper.

## Project Structure

```
acm/
├── core/
│   ├── __init__.py
│   ├── manager.py          # ActiveContextManager
│   ├── entity_tracker.py   # EntityTracker
│   ├── active_queue.py     # ActiveQueue
│   ├── decision_ledger.py  # DecisionLedger
│   ├── phase_detector.py   # PhaseDetector
│   ├── recency_buffer.py   # RecencyBuffer
│   └── trim.py             # Token budget management
├── adapters/
│   ├── env/
│   │   ├── base.py          # EnvironmentAdapter protocol
│   │   ├── ceobench.py      # CEO-Bench adapter
│   │   ├── webarena.py      # (future)
│   │   └── swebench.py      # (future)
│   └── model/
│       ├── base.py          # ModelAdapter protocol
│       ├── openai.py        # OpenAI + compatible
│       └── anthropic.py     # Anthropic
├── benchmark/
│   ├── conditions/
│   │   ├── oracle.py
│   │   ├── truncation.py
│   │   ├── rolling_summary.py
│   │   ├── rag.py
│   │   └── acm.py
│   ├── metrics.py
│   └── runner.py
└── tests/
```

## Implementation Order

### Phase 1: Core + CEO-Bench adapter (3-4 days)
- Implement EntityTracker, ActiveQueue, RecencyBuffer (straightforward)
- Implement DecisionLedger (harder — causal chain tracking)
- Implement PhaseDetector (start simple: threshold-based on entity trends)
- Build CEO-Bench environment adapter
- Build OpenAI-compatible model adapter
- Run: raw Qwen with ACM vs raw Qwen without, 50-day sim

### Phase 2: Baseline conditions (2 days)
- Implement truncation, rolling summary, RAG conditions
- Run all conditions on 50-day and 100-day CEO-Bench sims
- Generate first comparison table

### Phase 3: Full benchmark (3-4 days)
- Run N=3-5 seeds × 5 conditions × 2-3 models at 100/250/500-day horizons
- Implement causal relevance metric (counterfactual ablation)
- Generate plots and tables

### Phase 4: Second testbed (3-4 days)
- Port to a second benchmark (WebArena or SWE-bench multi-step)
- Show ACM advantage transfers — this proves universality

### Phase 5: Write up (2 days)
- Paper or technical blog post
- The headline table
- Open-source the framework