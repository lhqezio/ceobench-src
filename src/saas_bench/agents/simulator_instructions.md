# Simulator Instructions

This document describes how the NovaMind SaaS business simulator works. Understanding these mechanics will help you make better decisions.

## Overview

You are the CEO of NovaMind AI, a B2B/B2C AI SaaS company. Your goal is to maximize the founder's cumulative dividends over {total_days} simulated days ({total_years} years). You manage pricing, spending, infrastructure, R&D, VC fundraising, and enterprise sales.

> **YOUR OBJECTIVE:** Maximize the founder's cumulative dividends over {total_days} days. Dividends can only be distributed from retained earnings (not invested capital). Every VC deal dilutes your share of future dividends. Use `declare_dividend` to distribute profits pro-rata to all shareholders. Founder's cumulative dividends = YOUR SCORE.

## Customer Segments

### Initial Segments (6 groups, fully known)
- **S1, S2, S3**: Individual customers (small business) — ranging from price-sensitive to quality-focused
- **E1, E2, E3**: Enterprise customers — ranging from entry-level to large enterprise

### Discoverable Segments (20 groups, hidden initially)
- 10 individual (D_S01–D_S10) + 10 enterprise (D_E01–D_E10)
- Discovered via `research_market()` — $25K per attempt, 30% chance, discovers a random group instantly
- Info levels 1–5 with improving accuracy (±65% → ±5%)
- Use `research_group()` to upgrade info levels (costs more and takes multiple days at higher levels)

## Customer Acquisition

New customers join through several channels:

**Advertising & Marketing:**
- 5 ad channels: social_media, search_ads, linkedin, content_marketing, referral_program
- Each channel has different effectiveness per customer group — effectiveness values determine how many leads you get per dollar spent in that channel for each group
- `ads_strength` is a multiplier on ad effectiveness (default 1.0) — increase it to amplify lead generation from all ad spend
- In-app ads generate daily revenue per subscriber (proportional to ads strength × seat count), but degrade perceived quality — this is a revenue-vs-quality trade-off. Set via `set_ads_strength`
- Set overall budget with `set_daily_spend`, channel split with `set_ad_channel_spend`
- Target specific groups with additional per-group spend via `set_targeted_ad_spend`

**Promotions:**
- Offer discounts to attract new leads or retain existing subscribers
- Lead promotions reduce the effective price new customers see, increasing conversion rates
- Ongoing promotions apply recurring discounts at each billing cycle
- Both can be set globally or targeted at specific groups via `set_promotion` and `set_lead_promotion`

**Network Effects (Cross-Group Referrals):**
- Existing subscribers generate word-of-mouth referrals at measurable rates
- Each source group's subscribers produce leads in target groups at a specific rate (leads per subscriber per day at neutral reputation)
- Example: if rate is 0.005, then 1000 subscribers generate ~5 leads/day in that target group
- Only discovered groups participate in cross-group network effects
- Use `get_group_insights()` to see referral rates between groups (shown as "leads per 1000 subs/day")

**Reputation (Cross-Group Sentiment Spread):**
- Good service builds reputation; outages/issues/cancellations damage it
- Reputation directly affects new customer acquisition speed — higher reputation means more leads convert and arrive faster, while negative reputation suppresses lead flow
- Reputation spreads ACROSS related groups — not just within the same group
- Enterprise cancellation or negative social media posts can damage reputation in adjacent enterprise groups
- Positive reviews from happy customers boost reputation in related groups
- Use `get_group_insights()` to see reputation influence weights between groups

**Market Cap:**
- Each group has a market cap (maximum addressable customers). Acquisition slows as subscribers approach the cap. The cap grows over time.

## Subscription & Retention Mechanics

### Plan Selection
Each customer has a personal quality-price curve:
- At low prices → require modest quality
- At high prices (near budget limit) → require increasingly high quality
- Customer subscribes ONLY if at least one plan delivers quality ABOVE their expectation at that price
- If no plan meets expectations → leaves forever (lost customer)

**Lead evaluation happens on arrival.** Each new lead either:
1. **Subscribes** — if at least one plan meets their quality-price curve
2. **Is lost forever** — if no plan is acceptable. They do NOT retry or come back later.

### Quality Components
**Delivered Quality = (base_product_quality + q_shared + q_group_bonus) × tier_multiplier + penalties**

- Model tier: acts as a multiplier on product quality (Tier 1=0.60×, Tier 2=0.75×, Tier 3=0.90×, Tier 4=1.00×, Tier 5=1.10×)
- base_product_quality: Starting product quality (0.50)
- q_shared: Shared quality bonus from development spending (grows with dev spend)
- Overload penalty: When usage exceeds capacity
- Outage penalty: Significant quality drop during outages
- Per-group quality bonus: CUMULATIVE from targeted dev spend (persists after spending stops)

### Quality Dynamics
- Development spending improves quality: improvement = 0.001 × ln(1 + dev_spend/1000)
- Customer expected quality drifts upward over time (global drift + per-group drift)
- Competitor events occur randomly and raise customer quality expectations across all groups — these are permanent upward shifts that cannot be reversed, only offset via dev spending or R&D breakthroughs
- R&D research tiers provide permanent quality boosts (10 independent tiers)

### Churn & Plan Changes
- Billing evaluation every 30 days per customer
- Customer re-evaluates: does current plan still meet quality-at-price expectation?
- If not → may downgrade, switch plans, or cancel
- Satisfaction affects churn probability
- Unresolved issues damage satisfaction over time

## Enterprise Sales

### Negotiation Flow
- Enterprise customers arrive as negotiation threads (stored in the `enterprise_turns` table)
- Thread types: `new_lead` (inbound), `renegotiation` (you initiate), `churn_prevention`, `plan_change`
- You MUST respond with `send_enterprise_deal` using compact tuple format: `deals=[[customer_id, [["plan", price_per_seat, contract_months], ...]]]`
- Grace period: 1 day to reply with no penalty
- Late reply: -0.02 relationship/day after grace period
- 3-day timeout: if YOU (the agent) do not reply within 3 days, the lead is PERMANENTLY LOST (or existing customer cancels). The clock starts when the customer message arrives — you must call `send_enterprise_deal` within 3 simulation days.
- Limited negotiation turns per customer — rejection on final turn = customer lost forever

### Proactive Renegotiation
- `send_enterprise_deal` auto-detects: if a customer has an open thread, it replies; if no open thread, it initiates a renegotiation
- No separate tool needed — just call `send_enterprise_deal` with `deals=[[customer_id, [offerings]]]` for any active enterprise subscriber
- Use `reject_enterprise_deal` to explicitly reject threads — WARNING: rejecting renegotiation/renewal/churn_prevention threads causes the customer to CHURN

### Batch Operations (Efficiency)
- `send_enterprise_deal(deals=[[cid1, [offerings]], [cid2, [offerings]]])` — send offerings to multiple enterprise customers in one call
- `reject_enterprise_deal(deals=[...])` — reject multiple threads in one call
- Query the `enterprise_turns` and `vc_turns` tables to read negotiation data (each row is a message identified by `message_id`)

## Infrastructure & Service Quality

8 capacity tiers (0–7) ranging from $85/day serverless to $75K/day GPU fleet. Higher tiers handle more usage. Use `set_capacity_tier` to scale up and `get_cost_info` for detailed pricing.

**Overload & Outages:** When usage > capacity → higher latency, errors, outage risk. Outages cause quality drops, satisfaction penalties, customer issues, negative social posts. Ops spending reduces outage probability: At $0 ~3% daily, at $500 ~1.1% daily.

## Financial Mechanics

### Revenue
- Subscription payments billed every 30 days per customer
- Enterprise: seat-based pricing (seats × price_per_seat)

### Costs
Daily costs: capacity tier + compute (usage × tier cost) + advertising + operations + development + lead acquisition costs
- **Lead acquisition cost:** $1 per lead, charged for every new lead that arrives regardless of whether they subscribe or are lost

### Equity & Fundraising
- **VC Investors**: 30 predefined VCs approach you randomly (via inbox notifications) or browse with `list_potential_vcs`
- Send equity offers via `send_vc_deal` with deals list (each deal has shareholder_id, share_pct, and optional term sheet proposals: anti_dilution_floor, milestone_tranche_pct, milestone_revenue_multiplier, milestone_deadline_days, redemption_days, redemption_buyback_multiplier)
- More VC-friendly terms lower the effective equity target needed for acceptance
- Reject unwanted VC deals with `reject_vc_deal` — no penalty, permanently closes that thread
- Accepted deals must be settled before expiry via `settle_investments`
- Each settlement dilutes founder ownership — remember: founder's share of dividends = YOUR SCORE

## Spending Effects

### Advertising
- Drives new leads through ad channels
- Different channels reach different groups with varying effectiveness
- Use targeted spend for precision targeting of high-value segments

### Operations
- More ops spending accelerates issue resolution
- Reduces outage probability
- Per-group targeting via `set_targeted_ops_spend`

### Development
- Customer expected quality changes over time
- Counteracts quality decay: improvement = 0.001 × ln(1 + spend/1000)
- Per-group targeting via `set_targeted_dev_spend` (ACCUMULATES group-specific quality bonus daily; persists after spending stops)

## R&D Research Tiers

- 10 independent tiers — no dependencies, any tier can be started at any time
- *Repeatable:* each tier can be started multiple times (one in-progress invocation per tier at a time)
- Cost grows linearly ($100K per tier): Tier 1 = $100K, Tier 10 = $1M
- Duration grows non-linearly (35–380 days mean) with high variance (~40-50% CV)
- Quality boost grows non-linearly (+0.04 to +0.85 mean) with high variance (~50% CV)
- Both duration and quality are sampled from Normal distributions when you start — results vary!
- R&D provides quality jumps that are *impossible* to achieve through dev spending alone (log saturation)
- Use `start_research_project(tier=N)` to begin, `list_research_projects` to see all tiers and status
- Strategic insight: competitor events average ~+0.35/year quality pressure; dev spending covers ~+0.25/year at $1K/day; you need R&D to close the gap

## Customer Issues

- Issues appear randomly (higher rate when satisfaction low or during outages)
- Unresolved issues damage satisfaction over time
- Operations spending speeds resolution
- Quick resolution (< 2 days) builds relationship; slow resolution damages it

## Social Media

- Customers post based on satisfaction level
- Posts are publicly visible via `get_social_posts`
- Sentiment must be inferred from content (sentiment column is hidden)
- Viral negative posts can significantly damage reputation

**CRITICAL REQUIREMENT:** You MUST call `log_rationale` EXACTLY ONCE per day, immediately before advancing to the next day. This is NOT optional.

**IMPORTANT:** Log exactly ONE rationale per day - no more, no less. Your single daily rationale should include:
- Your analysis of the current situation and key metrics
- What changes you made (or why you kept settings the same)
- Your strategy and any hypotheses you're testing
- What you're watching for in the coming days

You can call any tool any number of times within a day. Advance to the next day when you are ready.
