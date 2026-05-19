<p align="center">
  <img src="assets/mascot.png" alt="CEO-Bench mascot" width="220"/>
</p>

# 🤖 CEO-Bench: Can Agents Play the Long Game?

Source repository for **CEO-Bench**, a long-horizon agent benchmark in which an
LLM agent operates a fictional AI startup for 500 simulated days.



## 📊 Overview

<p align="center">
  <img src="assets/teaser.png" alt="CEO-Bench teaser" width="100%"/>
</p>

CEO-Bench evaluates general long-horizon agent capabilities by simulating a
startup over 500 days in a realistic and challenging environment. The agent
operates through a programmable interface with access to business databases,
company management tools, and social media. Outcomes are driven by a partially
observable, noisy, and evolving market with delayed and coupled consequences.




## 🚀 Running CEO-Bench

### 🔑 Setup: Environment variables

The simulator uses a small Claude model (**Haiku 4.5** by default) to generate
customer-facing social-media content during the simulation. Pick **one**
provider and export the matching credentials:

```bash
# Option A: Amazon Bedrock (default)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-2"

# Option B: Anthropic direct API
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then in `src/saas_bench/config.py`, set the matching provider/model:

```python
social_post_llm_model:    str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
social_post_llm_provider: str = "bedrock"   # "bedrock" | "anthropic" 
```

If you switch to `anthropic`, set `social_post_llm_model` to
`"claude-haiku-4-5-20251001"` (drop the Bedrock prefix/suffix).



### 🎯 Option A: Evaluate any coding agent easily

We built CEO-Bench into a single executable and docs that any coding agent can just download the game and start playing.

The executable is hosted at **[zlab-princeton/run-ceobench](https://github.com/zlab-princeton/run-ceobench)**

If you want to evaluate a coding agent with terminal and internet access, prompt it

```
Download this, read instructions, and finish 500 day gameplay. https://github.com/zlab-princeton/run-ceobench
```



### ⚙️ Option B: Customize the configuration

All tunable simulator constants live in **`src/saas_bench/config.py`**: pricing,
customer groups, ad-channel productivity, R&D speed, competitor difficulty, etc.
After editing, rebuild the public bundle. 

```bash
uv sync                                  # one-time install
uv run python scripts/build_public.py    # rebuild public/ artifact
```

Then generated `public/` directory would play the same role as the same way as **[zlab-princeton/run-ceobench](https://github.com/zlab-princeton/run-ceobench)** in Option A

**Tuning difficulty** You can modify configuration in `config.py` to adjust difficulty.

An important difficulty is competitor strength. Competitor keeps track of a unreleased_dev_bank. Each agent's research and development quality improvement is added to this variable. At each competitor event, competitor draws `u ~ U(competitor_feedback_u_min, competitor_feedback_u_max)`, raises customer expectations by u × unreleased_dev_bank, and subtract this amount from unreleased_dev_bank. Larger competitor_feedback_u_min and competitor_feedback_u_max leads to stronger competitor and higher quality pressure. The default config value is (0.2,0.5). 



### 🤖 Option C: Replicate the bash-agent baseline

The paper's baseline gives an LLM a sandboxed bash shell plus the public CLI and
runs the full 500-day loop with checkpointing and logging. The full process:

**1. Install dependencies** (one-time):

```bash
uv sync
```

**2. Set provider credentials** in a `.env` file at the repo root. Which keys you
need depends on the agent model; for the default Bedrock run:

```bash
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_REGION="us-east-2"
```

Other providers read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`XAI_API_KEY`, `TOGETHER_API_KEY`, or `MODAL_TOKEN_*`. No `NMDB_KEY` is needed:
the SQLCipher key is embedded in the engine.

**3. Run.** `public/` ships prebuilt, so there is no build step:

```bash
uv run python -m saas_bench.agents.bash_agent.run_test \
    --model us.anthropic.claude-sonnet-4-6 \
    --provider bedrock \
    --reasoning-effort max \
    --seed 42 \
    --days 500 \
    --workspace bash_agent_runs
```

`--provider` accepts `openai | anthropic | bedrock | google | xai | together |
modal | ai_sandbox`; `--reasoning-effort` accepts `none | low | medium | high |
xhigh | max`. Pass `--label <tag>` to tag a config variant on the dashboard.

For long unattended runs, the convenience launchers wrap the command with
`nohup setsid`:

```bash
bash scripts/start_fresh_sonnet_bash.sh             # Bedrock Sonnet 4.6, max effort
bash scripts/start_fresh_gpt_bash.sh                # OpenAI GPT-5.5, xhigh effort
bash scripts/resume_run.sh bash_agent_runs/run_<id> # resume from a checkpoint
```

**4. Output.** Each run lands at `bash_agent_runs/run_<id>/` with `world.nmdb`
(encrypted ledger), `messages.jsonl`, `logs/`, `agent_workspace/` (a fresh git
repo with weekly commits), `config.json`, and `checkpoint.json`. To score and
analyze the run, see [docs/analyze_trajectory.md](docs/analyze_trajectory.md).

If you edit `src/saas_bench/config.py`, rebuild the bundle the agent sees with
`uv run python scripts/build_public.py` before launching.



## 📈 Analyzing agent trajectory

Every finished run leaves a single artifact: an encrypted `world.nmdb` ledger
(SQLCipher, page-level AES-256). It is the complete record of the run: cash,
subscriptions, customers, competitor events, and every action the agent took.

The decryption key is fixed and bundled into the published `novamind-operation`
zipapp at build time; see `KEYS.md` in this repo for the value, or import it
from the compiled `saas_bench._embedded_key` module. To decrypt and query:

```bash
KEY=$(grep _NMDB_KEY KEYS.md | head -1 | cut -d'"' -f2)
sqlcipher path/to/world.nmdb \
  "PRAGMA key = '$KEY';" \
  "SELECT day, category, amount FROM ledger ORDER BY day, id LIMIT 10;"
```

For the database schema, analysis recipes, and notes on keeping the agent from
cheating, see **[docs/analyze_trajectory.md](docs/analyze_trajectory.md)**.



## 📁 Repo layout

```
ceobench-src/
├── README.md                          ← this file
├── docs/
│   └── analyze_trajectory.md          ← decrypt, schema + analysis guide
├── public_sources/                    ← human-written inputs to the public build
│   ├── README.md, requirements.txt
│   └── examples/{autoplay_loop,basic_strategy}.py
├── scripts/
│   ├── build_public.py                ← canonical public-repo builder
│   ├── start_fresh_sonnet_bash.sh     ← bash-agent launcher (Bedrock Sonnet)
│   ├── start_fresh_gpt_bash.sh        ← bash-agent launcher (OpenAI GPT)
│   └── resume_run.sh                  ← resume bash agent from checkpoint
└── src/saas_bench/                    ← simulator + bash agent
    ├── simulation.py, environment.py, shocks.py, event_logger.py
    ├── config.py                      ← all tunable constants
    ├── customer_llm.py, personas.py, enterprise.py
    ├── database.py, db_protection.py
    ├── api_server.py, server_entry.py, tools.py
    ├── novamind_api/, novamind_cli.py, _public_cli.py
    └── agents/bash_agent/             ← canonical baseline harness
```



## 📜 Citation

```bibtex
@article{ceobench2026,
  title  = {CEO-Bench: Can Agents Play the Long Game?},
  author = {<authors>},
  year   = {2026},
}
```
