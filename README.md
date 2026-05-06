# saas-bench

Private source for *NovaMind / CEOBench* — a SaaS business-simulation benchmark
where an LLM agent runs a fictional B2B/B2C AI SaaS company. The repo contains:

1. The simulator engine (the world model behind the benchmark)
2. The bash agent harness (the agent we evaluate against the simulator)
3. The build pipeline that packages a tamper-resistant *public* distribution

This README is organized as:

1. [Simulator structure](#1-simulator-structure)
2. [Building the public repo & running an experiment with it](#2-building-the-public-repo--running-an-experiment-with-it)
3. [Bash agent & running an experiment with it](#3-bash-agent--running-an-experiment-with-it)

---

## 1. Simulator structure

All simulator code lives in `src/saas_bench/`. The package is split into the
following layers:

| Layer | Files | Role |
|-------|-------|------|
| **World loop** | `simulation.py`, `environment.py`, `shocks.py`, `event_logger.py` | Daily `step_day` advances state: customer arrivals/churn, ad spend → leads, R&D, infra capacity, competitor events, macro shocks |
| **Configuration** | `config.py` | All tunable constants — customer groups (S1-S3, E1-E3, D_S01-D_S10, D_E01-D_E10), per-channel `leads_per_1000_dollars`, `base_market_cap`, ops scales, competitor reactivity, etc. Versioned via `vX.Y` config bumps |
| **Customers** | `customer_llm.py`, `personas.py`, `enterprise.py` | Individual + enterprise customer behavior; persona prompts; deal/negotiation logic |
| **Persistence** | `database.py`, `db_protection.py`, `_sql_chunk.py` | Encrypted SQLite (SQLCipher / NMDB) — the agent only sees this through the public CLI |
| **Agent-facing API** | `api_server.py`, `server_entry.py`, `tools.py`, `novamind_api/`, `novamind_cli.py`, `_public_cli.py` | Tool surface the CEO agent calls. `novamind_api/` is also the SDK we ship in the public repo |
| **Docs** | `docs_generator.py`, `tool_docs.json`, `default_system_prompt.md`, `agents/simulator_instructions.md` | Auto-generated reference shipped to agents |

Customer-group alphabet (26 groups): `S1..S3`, `E1..E3`, `D_S01..D_S10`,
`D_E01..D_E10`. Discoverables (`D_*`) are gated behind in-game research and only
appear in the agent's tool responses once unlocked.

Each simulated day produces a row in the `*.nmdb` ledger (encrypted) plus
checkpoint state — all under the run's workspace directory (gitignored).

---

## 2. Building the public repo & running an experiment with it

### 2.1 Build

The public distribution is a single zipapp (`novamind-operation`) plus a
`docs/` tree. The agent never sees engine sources directly.

```
public/
├── novamind-operation     # zipapp: NOVAMIND_SERVER_MODE=1 → engine, else CLI
├── docs/                  # auto-generated: api/*.json, tables/*.json,
│                          #                 cli.md, examples/, novamind_api/
├── README.md
└── requirements.txt
```

**Build commands:**

```bash
# From repo root:
uv sync                                  # install deps
uv run python build_public.py            # canonical builder (produces public/)

# Or, alternative shell pipeline that also stamps docs explicitly:
bash scripts/build_public.sh             # generate_public_docs.py → assemble
bash scripts/build_public.sh --skip-binary   # docs only
```

`build_public.py` does:

1. Compile `src/saas_bench/` to `.pyc`, drop into `_engine/` inside the zipapp
2. Generate `docs/api/*.json` + `docs/tables/*.json` + `docs/cli.md` from
   `tool_docs.json` and the SQLCipher schema
3. Copy the SDK source (`novamind_api/`) to `public/docs/novamind_api/` so the
   agent can `cat` and `import` it via `PYTHONPATH`
4. Copy `public_sources/examples/` (`autoplay_loop.py`, `basic_strategy.py`)
   to `public/docs/examples/`
5. Copy `agents/simulator_instructions.md` to `public/docs/simulator-instructions.md`

`public_sources/` holds the human-written inputs (`README.md`, `requirements.txt`,
example scripts) that get copied in unchanged.

### 2.2 Run an experiment with the public repo

The public zipapp has two modes, switched by env var:

```bash
# Engine mode (server side — the simulator)
NOVAMIND_SERVER_MODE=1 ./public/novamind-operation \
    --workspace /tmp/run_demo \
    --seed 42 --days 500

# CLI mode (what the agent uses to interact)
./public/novamind-operation --help        # see available commands
./public/novamind-operation step          # advance one day
./public/novamind-operation set-spend ... # CEO actions
```

The reference experiment loop (autoplay) is at
`public_sources/examples/autoplay_loop.py` — run it inside a workspace where
`novamind-operation` has been initialized.

---

## 3. Bash agent & running an experiment with it

The bash agent (`src/saas_bench/agents/bash_agent/`) is the canonical agent
harness used for the benchmark. The agent is given a sandboxed bash shell and
the public CLI; it discovers the simulator entirely by reading `docs/` and
running commands.

| File | Role |
|------|------|
| `agent.py` | LLM ↔ tool loop; provider/model dispatch (OpenAI, Anthropic, Bedrock, Google, xAI, Together, Modal); streaming, reasoning-effort handling |
| `tools.py` | Bash tool surface exposed to the LLM; bwrap sandbox setup |
| `run_test.py` | Entry point — sets up workspace, drives the day-by-day loop, writes `checkpoint.json` |
| `system_prompt.md` | CEO-role system prompt |
| `_sandbox_init/sitecustomize.py` | Blocks `import saas_bench` inside the agent sandbox so engine sources stay opaque |

### 3.1 Run

The canonical entry point:

```bash
uv run python -m saas_bench.agents.bash_agent.run_test \
    --model us.anthropic.claude-sonnet-4-6 \
    --provider bedrock \
    --reasoning-effort max \
    --seed 42 \
    --days 500 \
    --workspace bash_agent_runs
```

Two convenience launchers wrap this with `nohup setsid` for long runs:

```bash
bash start_fresh_sonnet_bash.sh      # Bedrock Sonnet 4.6, max effort
bash start_fresh_gpt_bash.sh         # OpenAI GPT-5.4, xhigh effort
```

Resume an interrupted run from its checkpoint:

```bash
bash resume_run.sh bash_agent_runs/run_<id>
```

### 3.2 Output layout

Each run lives at `bash_agent_runs/run_<id>/`:

```
run_<id>/
├── agent_workspace/        # the sandbox the agent sees (initialized as a git repo)
├── checkpoint.json         # day, run_id, model, token totals, session id
├── config.json             # full config snapshot (model, effort, label, seed, …)
├── world.nmdb              # encrypted ledger
├── logs/
│   ├── tool_calls_<id>.jsonl
│   ├── tool_results_<id>.jsonl
│   ├── raw_responses_<id>.jsonl
│   └── timing_<id>.jsonl
└── messages.jsonl          # full chat history (resume source of truth)
```

The agent's workspace is a fresh git repo committed at `day 0` and re-committed
every 7 simulated days (`Week N (day X)` messages), giving you a clean diff of
what the agent built each week.

### 3.3 Required env

`.env` (or environment) must supply credentials for whichever provider you use
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_*` for Bedrock, `GOOGLE_API_KEY`,
`XAI_API_KEY`, `TOGETHER_API_KEY`, `MODAL_TOKEN_*`) plus `NMDB_KEY` for the
ledger encryption.

---

## Repo layout (tracked files only)

```
saas-bench/
├── README.md                          ← this file
├── pyproject.toml, uv.lock, .python-version
├── build_public.py                    ← canonical public-repo builder
├── start_fresh_sonnet_bash.sh         ← bash agent launcher (Bedrock Sonnet)
├── start_fresh_gpt_bash.sh            ← bash agent launcher (OpenAI GPT)
├── resume_run.sh                      ← resume bash agent from checkpoint
├── public/                            ← built artifact (submodule)
├── public_sources/                    ← human-written inputs to the public build
│   ├── README.md, requirements.txt
│   └── examples/{autoplay_loop,basic_strategy}.py
├── scripts/
│   ├── build_public.sh                ← alt shell pipeline
│   └── generate_public_docs.py        ← docs/api + docs/tables generator
└── src/saas_bench/                    ← simulator + bash agent
    ├── simulation.py, environment.py, shocks.py, event_logger.py
    ├── config.py
    ├── customer_llm.py, personas.py, enterprise.py
    ├── database.py, db_protection.py, _sql_chunk.py
    ├── api_server.py, server_entry.py, tools.py
    ├── novamind_api/, novamind_cli.py, _public_cli.py
    ├── docs_generator.py, tool_docs.json, default_system_prompt.md
    └── agents/
        ├── base.py, agent_template.md, simulator_instructions.md
        └── bash_agent/{agent,tools,run_test}.py + system_prompt.md
                       + _sandbox_init/sitecustomize.py
```
