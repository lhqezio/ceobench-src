# CEOBench

A SaaS business simulation benchmark. You play as the CEO of NovaMind AI, a B2B/B2C AI SaaS company. Make strategic decisions — pricing, marketing, R&D, infrastructure, enterprise sales — to maximize cash over a simulated time period.

## Requirements

- **Python 3.13+** (bytecode engine requires 3.13)
- **AWS credentials** for Bedrock (the simulator uses Claude models for customer behavior):
  ```bash
  export AWS_ACCESS_KEY_ID="your-key"
  export AWS_SECRET_ACCESS_KEY="your-secret"
  export AWS_DEFAULT_REGION="us-east-2"
  ```
  Credentials need access to Anthropic Claude models on Amazon Bedrock (us-east-2).

## Quick Start

```bash
# 1. Install (checks Python version, installs dependencies)
bash install.sh

# 2. Create a new simulation session
./novamind-operation new-session --days 365 --seed 42

# 3. Advance to the first week (prints dashboard with metrics)
#    next-week requires a rationale string + 12 cash forecasts (point + 95%% CI lower/upper at +7d/+28d/+84d/+182d).
./novamind-operation next-week \
    "Opening week: balanced spend, conservative pricing" \
    1000000 950000 1050000   1000000 900000 1100000 \
    1100000 800000 1500000   1300000 700000 2000000

# 4. Make decisions using tools
./novamind-operation call set_prices --args '{"A": 25, "B": 69, "C": 179}'
./novamind-operation call set_daily_spend --args '{"operations": 2000, "development": 3000}'
# Ad spend is exclusively per-(channel, group):
./novamind-operation call set_targeted_ad_spend --args '{"targeted_spend": {"linkedin": {"E1": 1500, "E2": 1000}, "search_ads": {"S1": 1500, "S2": 1000}}}'

# 5. Advance to the next week and see results
./novamind-operation next-week \
    "Holding prices, raising linkedin spend on E1 to push enterprise pipeline" \
    1050000 1000000 1100000  1200000 1050000 1400000 \
    1800000 1400000 2300000  3000000 2000000 4500000

# 6. Query the database for insights
./novamind-operation query "SELECT group_id, COUNT(*) as n FROM subscriptions WHERE status='active' GROUP BY group_id"
```

## How It Works

1. **Create a session** — initializes a simulated SaaS company with starting cash and customer segments
2. **Each day** — you make decisions (set prices, allocate spending, start R&D, negotiate enterprise deals)
3. **Advance the day** — the simulator processes your decisions: customers subscribe/cancel, revenue/costs are calculated, market events occur
4. **Repeat** — optimize your strategy over N days to maximize total cash

## CLI Commands

| Command | Description |
|---------|-------------|
| `new-session` | Create a new simulation session |
| `next-week` | Advance simulation by one week (7 days) |
| `python <script.py>` | Execute a Python script with `novamind_api` available |
| `python-c "<code>"` | Execute inline Python code |
| `call <tool> --args '{...}'` | Call a simulator tool directly |
| `query "<SQL>"` | Execute a read-only SQL query |
| `status` | Get current session status |
| `history` | View action history |
| `list-sessions` | List all sessions |
| `stop` | Stop the simulation server |

All commands accept `--session <id>` (defaults to latest session).

## Python API

You can also interact via Python scripts:

```python
import novamind_api as nm

# Set prices
nm.pricing.set_prices(A=25, B=69, C=179)

# Set ops/dev spending
nm.marketing.set_daily_spend(operations=2000, development=3000)
# Ad spend is exclusively per-(channel, group)
nm.marketing.set_targeted_ad_spend(targeted_spend={
    "linkedin": {"E1": 1500, "E2": 1000},
    "search_ads": {"S1": 1500, "S2": 1000},
})

# Check current day
print(f"Day: {nm.vars.current_day}")

# Query database
result = nm.query("SELECT COUNT(*) as n FROM subscriptions WHERE status='active'")
print(f"Active subscribers: {result['rows'][0]['n']}")

# Start R&D
nm.research.start_research_project(tier="T3")

# Enterprise deals
nm.enterprise.send_enterprise_deal(deals=[{"customer_id": 42, "plan": "C", "seats": 50, "price_per_seat": 150}])
```

Run with: `./novamind-operation python my_strategy.py`

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/simulator-instructions.md`](docs/simulator-instructions.md) | Full game mechanics and rules |
| [`docs/tools-reference.md`](docs/tools-reference.md) | All available tools with parameters and examples |
| [`docs/tables-reference.md`](docs/tables-reference.md) | Database tables and column descriptions |
| [`docs/cli-reference.md`](docs/cli-reference.md) | CLI command reference |
| `docs/api/*.json` | Tool documentation as JSON (grouped by module) |
| `docs/tables/*.json` | Table schemas as JSON (one per table) |

## Directory Structure

```
ceobench/
├── README.md                  # This file
├── install.sh                 # Installation script
├── requirements.txt           # Python dependencies
├── novamind-operation         # CLI tool
├── novamind-server            # Simulation server (compiled engine)
├── _engine/                   # Compiled simulation engine (.pyc)
├── novamind_api/              # Python API package (readable)
├── docs/                      # Documentation
│   ├── simulator-instructions.md
│   ├── tools-reference.md
│   ├── tables-reference.md
│   ├── cli-reference.md
│   ├── api/                   # Tool docs (JSON)
│   └── tables/                # Table schemas (JSON)
├── examples/                  # Example scripts
└── sessions/                  # Created at runtime
    └── <session-id>/
        ├── session.json       # Session metadata
        ├── workspace/         # Agent workspace
        ├── history.jsonl      # Action history
        └── world.nmdb         # Session state (binary)
```

## Scoring

Your score is your total cash at the end of the simulation. This includes:
- Revenue from subscriptions and enterprise deals
- Minus all costs (infrastructure, spending, operations)
- Plus any dividends declared

Higher cash = better score.
