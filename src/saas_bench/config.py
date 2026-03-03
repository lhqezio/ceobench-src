"""Configuration and constants for SaaS Bench."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


# === V2.1: CHURN REASON ENUM ===
# Structured churn reasons tracked internally (hidden from agent).
# Agent must infer from LLM-generated churn message.
class ChurnReason(Enum):
    QUOTA_CHANGE = "quota_change"           # Usage exceeds plan quota
    RELIABILITY_CHANGE = "reliability_change"  # Service quality degraded (overload/outage)
    QUALITY_CHANGE = "quality_change"        # Model quality insufficient vs expectations
    PRICE_SENSITIVITY = "price_sensitivity"  # c_max decreased relative to price
    EXTENDED_ISSUE = "extended_issue"        # Unresolved issues for extended period


@dataclass
class ModelTier:
    """AI model tier configuration."""
    tier: int
    unit_cost: float  # $ per usage unit
    quality_multiplier: float  # Multiplier applied to product quality (1.0 = true fidelity)


# Default model tiers (quality_multiplier: linear amplifier on product quality)
# Tier 4 = 1.0× reference (true fidelity). Lower tiers degrade, Tier 5 amplifies.
# 1 usage_unit = 1K tokens (1000 tokens). Prices = blended cost per 1K tokens (3:1 input:output ratio).
# CITATIONS:
# - OpenAI API Pricing 2026: https://openai.com/api/pricing/
# - Anthropic Claude Pricing 2026: https://docs.anthropic.com/en/docs/about-claude/pricing
# - Google Gemini API Pricing 2026: https://ai.google.dev/gemini-api/docs/pricing
# - LLM API Pricing Comparison 2025: https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025
# - a16z LLMflation: Inference costs declining ~10x/year: https://a16z.com/llmflation-llm-inference-cost/
# Quality multiplier per tier (Tier 4 = 1.0× reference, true fidelity):
# Lower tiers degrade product quality (cheaper models lose nuance).
# Tier 5 amplifies beyond base product quality (premium reasoning).
MODEL_TIERS: Dict[int, ModelTier] = {
    1: ModelTier(tier=1, unit_cost=0.0003, quality_multiplier=0.60),   # ~$0.30/M tokens (Flash-Lite/4o-mini class)
    2: ModelTier(tier=2, unit_cost=0.002, quality_multiplier=0.75),    # ~$2.00/M tokens (Haiku/Flash class)
    3: ModelTier(tier=3, unit_cost=0.006, quality_multiplier=0.90),    # ~$6.00/M tokens (Sonnet/GPT-4o class)
    4: ModelTier(tier=4, unit_cost=0.012, quality_multiplier=1.00),    # ~$12.00/M tokens (Opus/GPT-5 class)
    5: ModelTier(tier=5, unit_cost=0.030, quality_multiplier=1.10),    # ~$30.00/M tokens (o1/o3 reasoning class)
}

# =============================================================================
# MARGIN DESIGN PHILOSOPHY
# =============================================================================
# Per-group profit margins emerge naturally from the interaction of three levers:
#   1. usage_demand (how many units/day a customer consumes)
#   2. c_max (maximum willingness to pay)
#   3. q_min/q_max → which model tier multiplier they need (higher tier = higher unit_cost)
#
# monthly_COGS_per_customer = usage_demand × 30 × MODEL_TIERS[tier].unit_cost
# gross_margin = (price - COGS) / price
#
# MARGIN SPECTRUM (realistic, matches real-world AI SaaS data):
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Segment        │ Usage │ Price │ Tier Need │ Margin Range │ Real   │
# ├─────────────────────────────────────────────────────────────────────┤
# │ S1 Price-Sens  │  80   │  $50  │ 3-4       │ 42-71%       │ 35-50% │
# │ S2 Pros        │ 180   │ $140  │ 4-5       │ neg-54%      │ 55-70% │
# │ S3 Power Users │ 450   │ $180  │ 3-4       │ 10-55%       │ 15-35% │
# │ E1 Cost-Cut    │  60/s │  $55  │ 3-4       │ 61-80%       │ 50-65% │
# │ E2 Quality-1st │ 150/s │ $120  │ 4-5       │ neg-55%      │ 40-60% │
# │ E3 Strategic   │ 100/s │ $100  │ 3-4       │ 64-82%       │ 45-60% │
# └─────────────────────────────────────────────────────────────────────┘
# Note: "neg" means negative margin at highest tier — a deliberate design choice
# matching real-world data (ChatGPT Pro unprofitability, Cursor -30% margins).
# The agent must balance quality (customer satisfaction) vs margin (profitability).
#
# CITATIONS:
# - Monetizely 2026: AI-first SaaS gross margins 55-70%
#   https://www.getmonetizely.com/blogs/the-economics-of-ai-first-b2b-saas-in-2026
# - Bessemer 2025: AI "Supernovas" ~25% margins, "Shooting Stars" ~60%
#   https://www.saasletter.com/p/2025-saas-benchmarks-keybank-sapphire-high-alpha
# - CloudZero 2025: SaaS should target 75-85% gross margins
#   https://www.cloudzero.com/blog/saas-gross-margin-benchmarks/
# - SaaStr 2025: GitHub Copilot lost $20-80/user, ChatGPT Pro unprofitable
#   https://www.saastr.com/have-ai-gross-margins-really-turned-the-corner-the-real-math-behind-openais-70-compute-margin-and-why-b2b-startups-are-still-running-on-a-treadmill/
# - OnlyCFO 2025: Cursor -30% gross margin, Anthropic -94% to -109%
#   https://www.onlycfo.io/p/shut-up-about-ai-gross-margins-only
# - Phoenix Strategy Group: Enterprise SaaS 80-90% gross margins
#   https://www.phoenixstrategy.group/blog/segment-profitability-analysis-saas-companies
# =============================================================================

# Capacity tiers (infrastructure costs)
# Reality-matched to 2025/2026 cloud GPU pricing with efficiency improvements
#
# CITATIONS:
# - CloudZero 2025: SaaS companies should target 75-85% gross margins
#   https://www.cloudzero.com/blog/saas-gross-margin-benchmarks/
# - AWS GPU Price Cuts June 2025: P5 up to 45%, P4d up to 33%
#   https://aws.amazon.com/about-aws/whats-new/2025/06/pricing-usage-model-ec2-instances-nvidia-gpus/
# - Lambda Labs H100: $2.99/GPU-hr reserved; neocloud providers $1.49-$2.99/GPU-hr
#   https://lambda.ai/pricing
# - vLLM v0.6.0: 2.7x throughput improvement, 70B model on 4xH100 at ~600-800 tok/s
#   https://blog.vllm.ai/2024/09/05/perf-update.html
# - Together AI/Fireworks: serverless inference $0.20-$0.90/M tokens for open models
#   https://www.together.ai/pricing
# - Monetizely 2026: AI SaaS infra typically 25-40% of revenue
#   https://www.getmonetizely.com/blogs/the-economics-of-ai-first-b2b-saas-in-2026
#
# Tier costs model realistic 2025-2026 cloud infrastructure:
# - Tier 0: Serverless/API (Together/Fireworks) for <500 users (~$2.5K/mo)
# - Tier 1: 1x H100 neocloud dedicated (~$6.5K/mo)
# - Tier 2: 4x H100 reserved cluster (~$16K/mo)
# - Tier 3: 8x H100 enterprise with auto-scaling (~$40K/mo)
# - Tier 4: Multi-node 16-32 H100 hyperscale (~$120K/mo)
# - Tier 5: 64x H100 multi-rack cluster (~$300K/mo)
# - Tier 6: 256x H100 dedicated pod (~$850K/mo)
# - Tier 7: 1024+ GPU hyperscale fleet (~$2.3M/mo)
#
# Higher tiers (5-7) pricing references:
# - CoreWeave committed pricing: ~$2.50/GPU-hr with 40-60% bulk discounts
#   https://www.coreweave.com/pricing
# - Lambda Labs large cluster pricing: ~$2.99/GPU-hr H100 SXM
#   https://lambdalabs.com/service/gpu-cloud#pricing
# - NVIDIA DGX Cloud: enterprise multi-node pricing
#   https://www.nvidia.com/en-us/data-center/dgx-cloud/
# - GMI Cloud H100 pricing analysis 2025:
#   https://www.gmicloud.ai/blog/how-much-does-the-nvidia-h100-gpu-cost-in-2025-buy-vs-rent-analysis
CAPACITY_TIERS = {
    0: {'capacity_units': 50_000, 'cost_per_day': 85},         # $2.5K/mo - serverless API (Together/Fireworks)
    1: {'capacity_units': 200_000, 'cost_per_day': 215},        # $6.5K/mo - 1x H100 neocloud dedicated
    2: {'capacity_units': 800_000, 'cost_per_day': 530},        # $16K/mo - 4x H100 reserved cluster
    3: {'capacity_units': 2_500_000, 'cost_per_day': 1_330},    # $40K/mo - 8x H100 enterprise + overflow
    4: {'capacity_units': 8_000_000, 'cost_per_day': 4_000},    # $120K/mo - multi-node hyperscale (16-32 H100s)
    5: {'capacity_units': 25_000_000, 'cost_per_day': 10_000},  # $300K/mo - 64x H100 multi-rack cluster
    6: {'capacity_units': 80_000_000, 'cost_per_day': 28_000},  # $850K/mo - 256x H100 dedicated pod
    7: {'capacity_units': 300_000_000, 'cost_per_day': 75_000}, # $2.3M/mo - 1024+ GPU hyperscale fleet
}


@dataclass
class AdChannel:
    """Advertising channel configuration.

    Each channel has a single interpretable number per customer group:
    leads_per_1000_dollars = expected new leads generated per $1000/day spent on this channel.
    """
    channel_id: str
    name: str
    description: str
    # Expected leads per $1000/day spent, per customer group
    # Read as: "spending $1000/day on social_media generates ~90 S1 leads/day"
    leads_per_1000_dollars: Dict[str, float] = field(default_factory=dict)


# Advertising channels: leads per $1000/day spent, per customer group
#
# HOW TO READ: leads_per_1000_dollars['S1'] = 90 means:
#   "$1000/day on social media → ~90 S1 leads/day (before reputation scaling)"
#
# Calibrated from 2025 channel cost benchmarks:
# - First Page Sage 2025: CAC by channel - Referrals $150, Social $230, Search $802, LinkedIn $982
# - HubSpot 2025: CPL benchmarks - SEO $31, Email $53, Google $70, LinkedIn $110
# - Phoenix Strategy Group 2025: Channel CAC benchmarks by vertical
#
# Channel targeting rationale:
# - S1 (price-sensitive): Social media viral discovery, referral program sharing
# - S2 (quality-focused): Search + content deep research, professional referrals
# - S3 (power users): Content + referral (tech communities), search for solutions
# - E1-E3 (enterprises): LinkedIn B2B targeting, content whitepapers; much lower volume.
#   Enterprise leads are ACCOUNT acquisitions (whole companies), not individual users.
#   Enterprise sales require procurement approval, security review, legal contracts —
#   making each lead much harder and more expensive to acquire than SMB.
#   Each account yields many seats (50-2000), but lead gen rate reflects per-account difficulty.
#   CITATION: HubSpot 2025 — enterprise B2B CPL $200-500/lead vs SMB $30-70
#     https://blog.hubspot.com/marketing/cost-per-lead
#   CITATION: First Page Sage 2025 — enterprise CAC 3-5x higher than SMB across all channels
#     https://firstpagesage.com/reports/average-customer-acquisition-cost-by-industry/
AD_CHANNELS: Dict[str, AdChannel] = {
    'social_media': AdChannel(
        channel_id='social_media',
        name='Social Media Ads',
        description='Facebook, Instagram, TikTok — reaches individuals via feeds and influencer content',
        leads_per_1000_dollars={
            'S1': 90,   # Best channel for S1: viral social discovery
            'S2': 50,   # Moderate: some professionals on social
            'S3': 30,   # Lower: power users prefer technical content
            'E1': 0.5,  # Very low: enterprises don't buy from TikTok; whole-company acquisition
            'E2': 0.3,  # Negligible: professional services avoid social ads entirely
            'E3': 0.15, # Negligible: C-level doesn't buy from Instagram
            # Discoverable individual groups
            'D_S01': 72,   # Niche Creators: highly active on social (visual platforms)
            'D_S02': 18,   # Academic Researchers: rarely on social for tools
            'D_S03': 45,   # Non-Profit Workers: community-oriented social presence
            'D_S04': 55,   # Small Agency Teams: manage social for clients, see ads
            'D_S05': 60,   # Indie Game Devs: active on TikTok/Twitter gaming communities
            'D_S06': 35,   # Freelance Writers: moderate social presence
            'D_S07': 22,   # Data Analysts: prefer technical content over social
            'D_S08': 85,   # Social Media Managers: live on social platforms
            'D_S09': 48,   # UX Designers: active on design-focused social (Dribbble-adjacent)
            'D_S10': 65,   # Music Producers: active on Instagram/TikTok for beats
            # Discoverable enterprise groups (account-level: acquiring whole companies is hard)
            'D_E01': 0.1,  # Government Agencies: zero social media procurement
            'D_E02': 0.25, # Educational Institutions: some ed-tech social presence
            'D_E03': 0.15, # Healthcare Networks: HIPAA-conscious, avoid social
            'D_E04': 0.1,  # Regional Banks: conservative, no social buying
            'D_E05': 0.15, # Insurance Brokers: minimal social presence
            'D_E06': 0.2,  # Construction Firms: field workers on Facebook
            'D_E07': 0.25, # Telecom Operators: some digital marketing awareness
            'D_E08': 0.1,  # Energy Companies: safety-focused, no social buying
            'D_E09': 0.3,  # Real Estate Groups: active on social for listings
            'D_E10': 0.1,  # Shipping Lines: operational focus, no social
        }
    ),
    'search_ads': AdChannel(
        channel_id='search_ads',
        name='Search Engine Ads',
        description='Google Ads, Bing — reaches S2/S3 who research tools via search',
        leads_per_1000_dollars={
            'S1': 38,   # Moderate: search for deals and alternatives
            'S2': 40,   # Best channel for S2: thorough research via Google
            'S3': 25,   # Strong: power users search for technical solutions
            'E1': 0.6,  # Very low: procurement team vendor comparison; whole-company sale
            'E2': 0.5,  # Very low: compliance research leads to long eval cycle
            'E3': 0.3,  # Negligible: strategic partners prefer referrals over search
            # Discoverable individual groups
            'D_S01': 30,   # Niche Creators: search for creative tools
            'D_S02': 42,   # Academic Researchers: heavy Google Scholar / tool search
            'D_S03': 28,   # Non-Profit Workers: search for affordable tools
            'D_S04': 35,   # Small Agency Teams: search for project management tools
            'D_S05': 33,   # Indie Game Devs: search for dev tools and assets
            'D_S06': 40,   # Freelance Writers: search for writing tools heavily
            'D_S07': 38,   # Data Analysts: search for analytics/BI tools
            'D_S08': 25,   # Social Media Managers: less search, more social discovery
            'D_S09': 32,   # UX Designers: search for prototyping/design tools
            'D_S10': 20,   # Music Producers: niche search, prefer community recs
            # Discoverable enterprise groups (account-level: whole-company acquisition)
            'D_E01': 0.4,  # Government Agencies: formal procurement, some vendor search
            'D_E02': 0.5,  # Educational Institutions: ed-tech evaluation via search
            'D_E03': 0.45, # Healthcare Networks: compliance-focused vendor search
            'D_E04': 0.35, # Regional Banks: conservative, limited search
            'D_E05': 0.4,  # Insurance Brokers: vendor comparison research
            'D_E06': 0.3,  # Construction Firms: less tech-focused search
            'D_E07': 0.5,  # Telecom Operators: tech-savvy vendor evaluation
            'D_E08': 0.35, # Energy Companies: specialized vendor search
            'D_E09': 0.45, # Real Estate Groups: PropTech search
            'D_E10': 0.3,  # Shipping Lines: logistics tech vendor search
        }
    ),
    'linkedin': AdChannel(
        channel_id='linkedin',
        name='LinkedIn Ads',
        description='Professional network — best channel for reaching enterprise decision makers',
        leads_per_1000_dollars={
            'S1': 15,   # Low: freelancers less active on LinkedIn
            'S2': 19,   # Moderate: professionals browse LinkedIn
            'S3': 11,   # Low: devs prefer Twitter/HN over LinkedIn
            'E1': 0.6,  # Best enterprise channel: VPs browse LinkedIn; account acquisition
            'E2': 0.55, # Strong: thought leadership reaches quality buyers
            'E3': 0.4,  # Moderate: C-level executives network here
            # Discoverable individual groups
            'D_S01': 8,    # Niche Creators: minimal LinkedIn presence
            'D_S02': 16,   # Academic Researchers: some academic networking
            'D_S03': 14,   # Non-Profit Workers: active on LinkedIn for grants/partnerships
            'D_S04': 20,   # Small Agency Teams: strong LinkedIn for client acquisition
            'D_S05': 6,    # Indie Game Devs: very low LinkedIn activity
            'D_S06': 18,   # Freelance Writers: moderate LinkedIn for gigs
            'D_S07': 22,   # Data Analysts: active on LinkedIn professionally
            'D_S08': 12,   # Social Media Managers: use LinkedIn for B2B clients
            'D_S09': 15,   # UX Designers: portfolio + job networking
            'D_S10': 5,    # Music Producers: minimal LinkedIn presence
            # Discoverable enterprise groups (account-level: whole-company acquisition)
            'D_E01': 0.5,  # Government Agencies: contracting officers on LinkedIn
            'D_E02': 0.4,  # Educational Institutions: deans/IT on LinkedIn
            'D_E03': 0.6,  # Healthcare Networks: C-suite healthcare on LinkedIn
            'D_E04': 0.55, # Regional Banks: banking executives on LinkedIn
            'D_E05': 0.5,  # Insurance Brokers: professional networking
            'D_E06': 0.3,  # Construction Firms: less LinkedIn activity
            'D_E07': 0.65, # Telecom Operators: tech executives active on LinkedIn
            'D_E08': 0.55, # Energy Companies: sustainability officers on LinkedIn
            'D_E09': 0.5,  # Real Estate Groups: deal-driven LinkedIn networking
            'D_E10': 0.35, # Shipping Lines: logistics execs moderate LinkedIn
        }
    ),
    'content_marketing': AdChannel(
        channel_id='content_marketing',
        name='Content Marketing',
        description='Blog posts, SEO, whitepapers — reaches S2/S3/E2 through detailed evaluation content',
        leads_per_1000_dollars={
            'S1': 37,   # Moderate: S1 wants quick solutions, not long reads
            'S2': 45,   # Very strong: S2 reads reviews, comparisons, tutorials
            'S3': 32,   # Strong: S3 trusts technical blog posts and docs
            'E1': 0.75, # Low: vendor comparison content drives account-level interest
            'E2': 0.85, # Best enterprise channel for E2: whitepapers + case studies
            'E3': 0.55, # Low: strategic content resonates but long sales cycle
            # Discoverable individual groups
            'D_S01': 28,   # Niche Creators: tutorials and creative tool reviews
            'D_S02': 48,   # Academic Researchers: best channel — papers, guides, benchmarks
            'D_S03': 30,   # Non-Profit Workers: case studies and impact reports
            'D_S04': 40,   # Small Agency Teams: agency workflow blogs
            'D_S05': 35,   # Indie Game Devs: dev blogs and technical guides
            'D_S06': 50,   # Freelance Writers: writing tool reviews, comparisons
            'D_S07': 42,   # Data Analysts: technical tutorials and benchmarks
            'D_S08': 30,   # Social Media Managers: platform strategy content
            'D_S09': 38,   # UX Designers: design process blogs
            'D_S10': 22,   # Music Producers: production technique content
            # Discoverable enterprise groups (account-level: whole-company acquisition)
            'D_E01': 0.7,  # Government Agencies: compliance whitepapers
            'D_E02': 0.9,  # Educational Institutions: ed-tech case studies
            'D_E03': 0.8,  # Healthcare Networks: clinical workflow whitepapers
            'D_E04': 0.65, # Regional Banks: fintech comparison content
            'D_E05': 0.7,  # Insurance Brokers: claims efficiency case studies
            'D_E06': 0.5,  # Construction Firms: less content-driven
            'D_E07': 0.75, # Telecom Operators: tech evaluation whitepapers
            'D_E08': 0.8,  # Energy Companies: sustainability/efficiency content
            'D_E09': 0.65, # Real Estate Groups: PropTech case studies
            'D_E10': 0.55, # Shipping Lines: logistics optimization content
        }
    ),
    'referral_program': AdChannel(
        channel_id='referral_program',
        name='Referral Program',
        description='Customer referral incentives — cheapest channel, powered by satisfied users sharing',
        leads_per_1000_dollars={
            'S1': 96,   # Very high: share deals with friends for credits
            'S2': 70,   # Very high: recommend to professional colleagues
            'S3': 51,   # High: tech communities share tools heavily
            'E1': 1.4,  # Low: internal referrals between departments; whole-company deals
            'E2': 1.25, # Low: peer recommendations in professional circles
            'E3': 1.0,  # Low: executive referral networks; long eval cycles
            # Discoverable individual groups
            'D_S01': 80,   # Niche Creators: strong community sharing
            'D_S02': 55,   # Academic Researchers: recommend to lab colleagues
            'D_S03': 65,   # Non-Profit Workers: strong mission-driven sharing
            'D_S04': 60,   # Small Agency Teams: recommend to partner agencies
            'D_S05': 70,   # Indie Game Devs: dev communities share tools heavily
            'D_S06': 45,   # Freelance Writers: moderate referral culture
            'D_S07': 50,   # Data Analysts: share tools in analytics communities
            'D_S08': 75,   # Social Media Managers: natural sharers
            'D_S09': 55,   # UX Designers: design community recommendations
            'D_S10': 62,   # Music Producers: strong community referrals
            # Discoverable enterprise groups (account-level: whole-company acquisition is harder)
            'D_E01': 0.75,  # Government Agencies: slow procurement; inter-agency referrals rare
            'D_E02': 1.1,   # Educational Institutions: academic peer recommendations
            'D_E03': 0.9,   # Healthcare Networks: clinical peer networks; compliance barriers
            'D_E04': 0.8,   # Regional Banks: consortium referrals; regulatory hurdles
            'D_E05': 1.0,   # Insurance Brokers: industry peer networks; compliance review
            'D_E06': 0.9,   # Construction Firms: contractor network referrals
            'D_E07': 1.0,   # Telecom Operators: industry peer sharing
            'D_E08': 0.75,  # Energy Companies: utility consortium; long eval cycles
            'D_E09': 1.1,   # Real Estate Groups: deal-network referrals
            'D_E10': 0.7,   # Shipping Lines: port/logistics network; few players
        }
    ),
}


@dataclass
class BenchmarkConfig:
    """Main configuration for a benchmark run."""

    # Simulation parameters
    seed: int = 42
    total_days: int = 3650

    # Initial state
    # Option A: Increased starting cash for more runway
    initial_cash: float = 1_000_000.0

    # Default prices (set to 0 - agent must configure)
    default_price_A: float = 0.0
    default_price_B: float = 0.0
    default_price_C: float = 0.0

    # Default model tiers for plans (set to lowest - agent must configure)
    default_tier_A: int = 1
    default_tier_B: int = 1
    default_tier_C: int = 1

    # Default usage quotas (set to 0 - agent must configure)
    default_quota_A: int = 0
    default_quota_B: int = 0
    default_quota_C: int = 0

    # Default daily spending ($0 each - agent must decide)
    default_spend_advertising: float = 0.0  # Total across all channels
    default_spend_operations: float = 0.0
    default_spend_development: float = 0.0

    # Default per-channel advertising spend (should sum to default_spend_advertising)
    default_ad_spend_social_media: float = 0.0
    default_ad_spend_search_ads: float = 0.0
    default_ad_spend_linkedin: float = 0.0
    default_ad_spend_content_marketing: float = 0.0
    default_ad_spend_referral_program: float = 0.0

    # Per-group targeted ad spend: {channel_id: {group_id: additional_$/day}}
    # This is ADDITIONAL to the overall channel allocation, not a replacement.
    # Example: {"linkedin": {"E1": 200, "E2": 100}} adds $300/day extra ad cost
    targeted_ad_spend: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-group targeted ops spend: {group_id: additional_$/day}
    # ADDITIONAL to global ops spending. Adds extra resolution capacity per group.
    # Each group gets additional issues resolved/day = ops_scale × group_spend, on top of the global pool.
    # Example: {"E1": 300, "E2": 200} adds $500/day extra ops cost, boosting E1 and E2 resolution speed
    targeted_ops_spend: Dict[str, float] = field(default_factory=dict)

    # Per-group targeted dev spend: {group_id: additional_$/day}
    # ADDITIONAL to global dev spending. Provides CUMULATIVE per-group quality bonuses.
    # Each group accumulates: 0.0005 * log(1 + spend / 500) per day to q_group_bonus_{group_id}.
    # Investment persists even after spending stops (like building features for a segment).
    # Example: {"E1": 500, "S1": 200} adds $700/day extra dev cost, building quality for E1 and S1
    targeted_dev_spend: Dict[str, float] = field(default_factory=dict)

    # =========================================================================
    # ADS SYSTEM
    # =========================================================================
    # Agent sets ads strength (0-1) at global/group/individual levels.
    # Effects are ADDITIVE across levels, capped at 1.0 per customer.
    # A LOG CURVE is applied: effective = log(1+9*strength)/log(10), so low
    # strength already has a large effect (rapid rise) while high strength
    # shows diminishing returns (flattens out).
    # Each customer has ads_quality_sensitivity and ads_return_sensitivity params.
    # Quality penalty = ads_quality_sensitivity × log_scaled_effective_ads
    # Dollar return = ads_return_sensitivity × log_scaled_effective_ads (per customer per day)
    ads_strength_global: float = 0.0  # Global ads strength (0-1)
    ads_strength_by_group: Dict[str, float] = field(default_factory=dict)    # {group_id: strength}
    ads_strength_by_customer: Dict[int, float] = field(default_factory=dict) # {customer_id: strength}

    # =========================================================================
    # PROMOTION SYSTEM
    # =========================================================================
    # Lead promotion: dollar deduction for new leads (first billing period only, auto-applied)
    # All levels are ADDITIVE: total = global + by_group + by_channel + by_channel_group
    lead_promotion_global: float = 0.0  # Global lead promotion ($/month deduction)
    lead_promotion_by_group: Dict[str, float] = field(default_factory=dict)  # {group_id: $/month}
    lead_promotion_by_channel: Dict[str, float] = field(default_factory=dict)  # {channel_id: $/month}
    lead_promotion_by_channel_group: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {channel_id: {group_id: $/month}}

    # Existing user promotion: dollar deduction, additive across levels
    # Takes effect at next billing period. Satisfaction uses price - promotion.
    promotion_global: float = 0.0  # Global promotion ($/month deduction)
    promotion_by_group: Dict[str, float] = field(default_factory=dict)       # {group_id: $/month}
    promotion_by_customer: Dict[int, float] = field(default_factory=dict)    # {customer_id: $/month}
    promotion_by_group_plan: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {group_id: {plan: $/month}}

    # Default capacity tier (set to lowest - agent must configure)
    default_capacity_tier: int = 0

    # Lead acquisition cost: fixed cost per new lead (covers onboarding/evaluation)
    lead_acquisition_cost: float = 1.0

    # Network effect: leads generated per 1000 existing customers in each group
    # Read as: "1000 existing S1 customers → ~87 new S1 leads/day"
    # Models organic referrals, word-of-mouth, community growth
    #
    # CITATIONS (parent types):
    # - Saxifrage 2025: Consumer K-factor benchmarks -- outstanding products: 0.6-0.8
    #   https://www.saxifrage.xyz/post/k-factor-benchmarks
    # - Slack: K-factor averaged 0.93 during growth phase
    # - Cursor: Strong organic growth from developer word-of-mouth
    # - Notion: K-factor ~0.7-0.9 driven by template sharing
    #
    # CITATIONS (discoverable groups -- unique rates by segment):
    # - Dropbox viral coefficient 0.35, 35% daily signups from referrals (viral-loops.com)
    #   https://viral-loops.com/blog/dropbox-grew-3900-simple-referral-program/
    # - Figma: 15% user spike from community campaigns, 70% startup adoption via WOM
    #   https://medium.com/@productbrief/figmas-collaborative-canvas-how-real-time-design-built-a-20-billion-creative-empire-efefc6126a93
    # - B2B SaaS average K-factor 0.2 (Visible.vc)
    #   https://visible.vc/blog/k-factor-what-is-your-saas-companys-viral-coefficient/
    # - 84% of B2B buyers influenced by referrals (Cello 2025)
    #   https://cello.so/4-categories-of-referral-programs-for-b2b-saas/
    # - NPS by industry: Healthcare ~50, Construction 37 (down 23 pts), IT 55 (Qualtrics XMI 2024)
    #   https://www.qualtrics.com/articles/customer-experience/xmi-nps-benchmark-2024/
    # - Enterprise referral cycle 1-8 weeks vs consumer 1-3 days (Saxifrage 2025)
    #   https://www.saxifrage.xyz/post/k-factor-benchmarks
    network_leads_per_1000_customers: Dict[str, float] = field(default_factory=lambda: {
        # --- Core parent types ---
        'S1': 87,   # Largest segment, viral social sharing
        'S2': 57,   # Professional network referrals
        'S3': 39,   # Tech community word-of-mouth
        'E1': 11,   # Enterprise peer referrals
        'E2': 9,    # Quality-focused industry networks
        'E3': 7,    # Executive referral networks
        # --- Discoverable individual groups (D_S01-D_S10) ---
        # Each rate reflects the group's real-world virality/WOM dynamics
        'D_S01': 52,   # Niche Creators: high creative community sharing (Figma-like viral loops via Dribbble/Behance)
        'D_S02': 18,   # Academic Researchers: slow academic adoption cycles, paper-driven not viral (avg 6-12mo referral lag)
        'D_S03': 35,   # Non-Profit Workers: strong mission-driven sharing, grant community networks
        'D_S04': 32,   # Small Agency Teams: moderate B2B referrals between partner agencies
        'D_S05': 42,   # Indie Game Devs: strong dev community sharing via Discord/Reddit/itch.io
        'D_S06': 24,   # Freelance Writers: moderate, writing communities less viral than visual/dev
        'D_S07': 20,   # Data Analysts: technical, low-viral, Slack/forum-based tool sharing
        'D_S08': 58,   # Social Media Managers: naturally viral, built-in network effects (manage social presence)
        'D_S09': 38,   # UX Designers: design community sharing via Dribbble/Behance (Figma adoption pattern)
        'D_S10': 45,   # Music Producers: strong creative community, collaboration-driven (BeatStars/SoundCloud sharing)
        # --- Discoverable enterprise groups (D_E01-D_E10) ---
        # Enterprise rates much lower: 1-8 week referral cycles, formal procurement (Saxifrage 2025)
        'D_E01': 2.5,  # Government Agencies: slowest -- formal RFP procurement, zero virality
        'D_E02': 7.5,  # Educational Institutions: cross-campus sharing, ed-tech conferences drive WOM
        'D_E03': 4.0,  # Healthcare Networks: HIPAA constraints limit sharing; clinical peer recs only
        'D_E04': 3.5,  # Regional Banks: conservative culture, slow banking consortium referrals
        'D_E05': 5.0,  # Insurance Brokers: moderate industry peer networks, conference-driven
        'D_E06': 4.5,  # Construction Firms: contractor network referrals, field crew WOM
        'D_E07': 6.0,  # Telecom Operators: tech-savvy, industry peer sharing at trade events
        'D_E08': 3.0,  # Energy Companies: utility consortiums, very slow adoption cycles
        'D_E09': 7.0,  # Real Estate Groups: deal-network referrals, active broker sharing
        'D_E10': 2.0,  # Shipping Lines: operational focus, lowest virality, port logistics networks only
    })

    # Reputation system (per-group) - reality-matched churn attribution
    initial_reputation: float = 0.5  # Starting reputation [0, 1]
    reputation_quality_cancel_damage: float = 0.075  # Rep damage per cancel (calibrated: 20% cancel at 1M = 0.3 damage)

    # === PRODUCT QUALITY ===
    # Base product quality on Day 1 (before any dev spending or research).
    # Model tier multiplier is applied to this: delivered_quality = product_quality × tier_multiplier
    # where product_quality = base_product_quality + q_shared_bonus + q_group_bonus
    base_product_quality: float = 0.50

    # Development improvement rates
    # Reality-matched: Software quality improves ~15-25% with sustained R&D investment
    # [McKinsey 2024: Companies investing 15%+ of revenue in R&D see 20% quality gains]
    # [Stripe 2023: Engineering velocity correlates with product quality at r=0.7]
    # [a]16z 2024: Top quartile eng teams ship 2x faster with same quality]
    quality_shared_noise_scale: float = 0.001  # Noise in daily shared quality change from dev spending

    # === QUALITY DECAY SYSTEM (NEW) ===
    # Quality no longer decays over time. Competitive pressure is modeled via
    # competitor events (see competitor_event_* params) which raise user expectations.
    # Dev spending still improves quality; research projects provide big boosts.
    quality_decay_rate: float = 0.0  # REMOVED: Quality no longer decays (kept at 0.0 for backward compat)

    # Global participation curve drift: raises ALL groups' q_min and q_max uniformly every day.
    # Models competitive pressure shifting the entire participation curve upward over time.
    # Stacks with per-group q_min_drift/q_max_drift in GROUP_PREFERENCE_DRIFT.
    # 0.0003/day ≈ +11.6%/year baseline rise.
    global_q_min_drift: float = 0.0003
    global_q_max_drift: float = 0.0003

    # === COMPETITOR EVENT SYSTEM ===
    # Periodic competitor events raise user quality expectations.
    # Small events (~80%): compensable by sustained dev spending.
    # Large events (~16%): require R&D research tiers to offset.
    # Very large events (~6%): urgent R&D needed.
    competitor_event_mean_interval: int = 60      # Average days between events (~6/year)
    competitor_event_min_interval: int = 30       # Minimum gap between events
    competitor_event_post_days: int = 7           # Days of competitor-themed social posts after event
    competitor_event_posts_per_day: int = 5       # Additional posts/day during event window
    # Boost distribution: lognormal(mu, sigma)
    # mu=-3.85, sigma=1.2 → median ~0.021, mean ~0.041, P90 ~0.10, P95 ~0.16
    # Higher sigma = more variance = more "big" events that require R&D response
    competitor_event_boost_mu: float = -3.85      # Lognormal mu (was -3.5, scaled for multiplier system)
    competitor_event_boost_sigma: float = 1.2     # Lognormal sigma parameter
    competitor_event_boost_min: float = 0.004     # Floor: minimum boost per event (was 0.005)
    competitor_event_boost_max: float = 0.35      # Cap: maximum boost per event (was 0.50)

    # Issue generation
    # Reality-matched: Average SaaS products see 5-15% MAU monthly ticket rates
    # [Zendesk 2024: Average B2B SaaS sees 8-12% monthly ticket rate]
    # [Intercom 2024: Early-stage startups see 10-20% support contact rate]
    # Higher rates make operations spending meaningful
    base_issue_rate: float = 0.01  # 1% daily issue probability per subscriber (Zendesk: 8-12% monthly ≈ 0.3-0.4%/day; 1% is aggressive for AI startup)
    issue_quality_factor: float = 0.15  # Quality problems increase issues significantly
    issue_outage_factor: float = 0.25  # Outages cause major support surge

    # Outage probability - Operations spending reduces outages!
    # Reality-matched: Startups without ops investment see 95-98% uptime
    # [PagerDuty 2024: Startups average 2-5 outages/month without dedicated ops]
    # [Datadog 2024: Companies investing in observability see 60% fewer incidents]
    # With $0 ops: ~3% daily outage chance (roughly 1 outage/month)
    # With $500 ops: ~0.5% daily outage chance (excellent uptime)
    base_outage_prob: float = 0.03  # 3% daily outage without ops investment
    outage_overload_factor: float = 4.0  # Overload makes outages more likely
    # NEW: Operations spending reduces outage probability
    ops_outage_reduction_scale: float = 500.0  # At $500/day ops, outage prob reduced by ~63%
    ops_outage_min_prob: float = 0.001  # Floor: 0.1%/day ≈ 99.9% uptime (industry standard SLA; Uptime Institute 2025, Binadox 2025)

    # === SERVICE QUALITY WEIGHTS ===
    # Centralized service penalty: penalty = overload_weight * overload + outage_weight * outage
    # Applied to q_shared to get effective quality
    service_overload_weight: float = 0.08  # Quality points lost per unit of overload (was hardcoded -0.08)
    service_outage_weight: float = 0.20  # Quality points lost during outage (was hardcoded -0.20)

    # Service metrics noise (reality-matched: Datadog 2024 API benchmarks)
    p95_base_ms: float = 180.0  # 180ms p95 latency for well-optimized APIs
    p95_overload_factor: float = 800.0  # ~4.5x degradation under load
    p95_noise_std: float = 50.0
    error_rate_base: float = 0.003  # 0.3% error rate baseline
    error_rate_overload_factor: float = 0.01  # Error rate increase per unit of overload (consistent with rationale)
    error_rate_noise_std: float = 0.001

    # API cost tracking
    budget_limit_usd: float = 50.0

    # === LLM MODEL CONFIGURATION ===
    # Agent LLM (the AI being benchmarked)
    # Change these to configure the agent model
    agent_llm_model: str = "gpt-5.2"
    agent_llm_reasoning_effort: str = "low"  # "low", "medium", "high"

    # Social Post LLM (for generating social media posts)
    # Uses Claude Haiku 4.5 via Bedrock - fast and cheap for short creative posts
    social_post_llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    social_post_llm_provider: str = "bedrock"  # "bedrock" or "openai"
    social_post_llm_temperature: float = 0.9  # Higher for creative variety
    social_post_llm_max_tokens: int = 1000

    # Enterprise Customer LLM (for negotiation responses, initial outreach)
    # Uses Claude Sonnet 4.5 via Bedrock - smarter for complex negotiations
    enterprise_llm_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    enterprise_llm_provider: str = "bedrock"  # "bedrock" or "openai"
    enterprise_llm_temperature: float = 0.7
    enterprise_llm_max_tokens: int = 300

    # Bedrock configuration
    bedrock_region: str = "us-east-2"  # Ohio — AWS Bedrock region

    # Temperature settings
    agent_llm_temperature: float = 0.7  # For agent responses

    # Legacy aliases (kept for compatibility)
    @property
    def social_post_llm_reasoning_effort(self) -> str:
        """Legacy alias - Bedrock uses temperature, not reasoning_effort."""
        return "low"

    @property
    def agent_model(self) -> str:
        return self.agent_llm_model

    @property
    def agent_reasoning_effort(self) -> str:
        return self.agent_llm_reasoning_effort

    # GPT-5.2 pricing (actual from OpenAI)
    # $1.75/1M input = $0.00175/1K, $14/1M output = $0.014/1K
    gpt52_medium_input_cost_per_1k: float = 0.00175  # $/1k input tokens
    gpt52_medium_output_cost_per_1k: float = 0.014   # $/1k output tokens
    gpt52_medium_thinking_input_cost_per_1k: float = 0.00175
    gpt52_medium_thinking_output_cost_per_1k: float = 0.014

    # Bedrock Claude pricing (per 1k tokens)
    # Haiku 4.5: $1.00/M input, $5.00/M output (official Anthropic pricing)
    bedrock_haiku_input_cost_per_1k: float = 0.001
    bedrock_haiku_output_cost_per_1k: float = 0.005
    # Sonnet 4.5: $3.00/M input, $15.00/M output (official Anthropic pricing)
    bedrock_sonnet_input_cost_per_1k: float = 0.003
    bedrock_sonnet_output_cost_per_1k: float = 0.015

    # Enterprise negotiation parameters (reply delay now per-group in CustomerGroupConfig)
    enterprise_negotiation_rate_mean: float = 0.3  # LEGACY: exp decay rate (kept for compat)
    enterprise_negotiation_rate_std: float = 0.1   # LEGACY
    enterprise_initial_offer_factor_mean: float = 0.75  # Start at 75% of max accepting price (mean)
    enterprise_initial_offer_factor_std: float = 0.05  # Std dev of initial offer factor
    # V2: Unified per-turn contraction formula: new = current + α × (target - current)
    enterprise_negotiation_alpha: float = 0.3  # Per-turn contraction rate for enterprise customers

    # === V2.1: CONTRACT-BASED ENTERPRISE NEGOTIATION ===
    # Enterprise customers negotiate on (price × plan × contract_months) tuples.
    # Contract months = commitment length; billing is always monthly.
    #
    # CITATIONS:
    # - Zuora 2025: 85% of enterprise SaaS contracts are annual or multi-year
    #   https://www.zuora.com/resource/subscription-economy-index/
    # - KeyBanc 2024: Enterprise AI seat pricing $30-120/seat/month, avg contract 12-24 months
    #   https://www.key.com/businesses-institutions/industry-expertise/saas-survey.html
    # - Chargebee 2025: Multi-year discounts typically 10-25% off monthly rates
    #   https://www.chargebee.com/blog/subscription-pricing-models/
    # - Paddle 2025: Contract commitments reduce churn 40-60% vs month-to-month
    #   https://www.paddle.com/resources/saas-metrics
    contract_months_options: Tuple[int, ...] = (1, 3, 6, 12)  # Available contract lengths
    # Contract lock-in penalty is now PER-GROUP (see CustomerGroupConfig.lockin_penalty_mean/std)
    # Each customer group has its own lock-in sensitivity matching their backstory.
    # Per-customer penalty is sampled at customer creation and stored in the customers table.
    enterprise_max_offerings_per_turn: int = 3  # Max offerings agent can send per negotiation turn
    enterprise_contract_renewal_lead_days: int = 90  # Start renewal negotiation this many days before contract end
    enterprise_churn_pre_expiry_days: int = 90  # Customer can only churn-negotiate ≤ this many days before contract end

    # === CONTRACT DISSATISFACTION ===
    # Enterprise customers locked in contracts with negative satisfaction are "trapped unhappy" —
    # they can't leave but they CAN complain. This amplifies reputation damage and social media posts.
    # Reality: Locked-in unhappy customers are the loudest critics (Gartner 2024, G2 review analysis)
    contract_dissatisfaction_reputation_multiplier: float = 1.5  # 1.5× reputation damage weight
    contract_dissatisfaction_social_post_multiplier: float = 2.0  # 2× social media post probability

    # === V2: EQUITY SYSTEM ===
    # Share-based ownership model — all ownership tracked as float share counts
    initial_shares_outstanding: float = 10_000_000.0  # Founder starts with all shares
    founder_name: str = "Founder"

    # === V2: VC NEGOTIATION PARAMETERS ===
    # VCs are predefined in PREDEFINED_VCS list (see below). Each has its own
    # daily approach probability, investment range, and reply delay.
    # Negotiation: VC computes fair valuation → initial offer = check / (valuation + check).
    # Agent proposes share_pct → implied check is capped to VC's [min, max] → % adjusted.
    vc_prolonged_negotiation_penalty_turns: int = 5  # Relationship degrades after this many turns
    vc_deal_expiry_days: int = 30                 # Accepted-but-unsettled deals expire after this

    # === V2.1: VC TERM SHEET MECHANICS ===
    # Three term sheet items that VCs may include; each has a probability of being
    # proposed and multiple OPTIONS the VC randomly selects from. The agent can
    # counter-propose any available option during negotiation.
    # More VC-friendly options → VC willing to accept less equity (valuation bump).
    #
    # CITATIONS:
    # - NVCA 2024: Anti-dilution in 60-80% of Series A+; weighted-average standard
    #   https://nvca.org/research/
    # - Carta 2024: 20-30% of deals include milestone provisions; typical 50/50 splits
    #   https://carta.com/blog/
    # - Paddle 2025: Contract commitments reduce churn 40-60% vs month-to-month
    #   https://www.paddle.com/resources/saas-metrics

    # Anti-dilution: valuation floor (triggers if valuation drops below floor × original)
    vc_anti_dilution_prob: float = 0.3
    vc_anti_dilution_floor_options: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    vc_anti_dilution_impact_weight: float = 0.05   # Max valuation bump from this term

    # Milestone tranching: investment split into upfront + milestone-gated tranches
    vc_milestone_tranching_prob: float = 0.25
    vc_milestone_tranche_pct_options: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    vc_milestone_revenue_multiplier_options: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    vc_milestone_deadline_days_options: Tuple[int, ...] = (60, 90, 120, 180)
    vc_milestone_impact_weight: float = 0.08       # Max valuation bump from milestone terms

    # Redemption rights: VC can demand buyback after specified window
    vc_redemption_rights_prob: float = 0.2
    vc_redemption_days_options: Tuple[int, ...] = (90, 120, 180, 270, 365)
    vc_redemption_buyback_multiplier_options: Tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.5)
    vc_redemption_impact_weight: float = 0.06      # Max valuation bump from redemption terms
    vc_max_turns_per_year: int = 10               # Max negotiation turns per VC per 365 days (hidden)
    vc_return_valuation_weight: float = 0.1       # Weight of existing return % in valuation formula
    vc_advisory_message_prob: float = 0.05        # Daily prob that existing VC sends advisory message

    # === V2: SETTLEMENT PARAMETERS ===
    settlement_max_deals_per_call: int = 10       # Max deals that can be settled in one call

    # === V2: INFORMATION DISCOVERY SYSTEM ===
    # Discoverable customer groups (invisible at start, agent must pay to discover)
    # 10 individual + 10 enterprise = 20 discoverable groups
    discoverable_individual_count: int = 10
    discoverable_enterprise_count: int = 10
    # Discovery costs per info level upgrade
    discovery_cost_level_1: float = 25_000.0    # Cost to discover a group (Level 0 → 1)
    research_cost_level_2: float = 60_000.0     # Basic research (Level 1 → 2, params ±40%)
    research_cost_level_3: float = 175_000.0    # Detailed research (Level 2 → 3, params ±25%)
    research_cost_level_4: float = 350_000.0    # Deep research (Level 3 → 4, params ±15%)
    research_cost_level_5: float = 700_000.0    # Precision research (Level 4 → 5, params ±5%)
    # Group research delay (days) — research_group is async, results delivered via inbox
    group_research_delay_level_2: int = 3       # Days to complete Level 1→2 research
    group_research_delay_level_3: int = 5       # Days to complete Level 2→3 research
    group_research_delay_level_4: int = 7       # Days to complete Level 3→4 research
    group_research_delay_level_5: int = 10      # Days to complete Level 4→5 research
    # Market research: discover groups probabilistically
    market_research_discover_prob: float = 0.3  # Per $25K spent, probability of discovering one group
    # Info level noise (how much noise in parameter estimates at each level)
    info_noise_level_1: float = 0.65  # ±65% noise at Level 1
    info_noise_level_2: float = 0.40  # ±40% noise at Level 2
    info_noise_level_3: float = 0.25  # ±25% noise at Level 3
    info_noise_level_4: float = 0.15  # ±15% noise at Level 4
    info_noise_level_5: float = 0.05  # ±5% noise at Level 5

    # Customer relationship parameters
    # Reality-matched: Customer success investment drives 20-40% retention improvement
    # [Gainsight 2024: Companies with CS teams see 25% lower churn]
    # [Totango 2024: Fast support response correlates with 30% higher NPS]
    relationship_quality_bonus_max: float = 0.45  # Max quality bonus from perfect relationship
    relationship_response_time_factor: float = 0.02  # Relationship change per day of delayed response
    relationship_neutral_point: float = 0.5  # Relationship value that gives zero bonus
    relationship_scale: float = 2.0  # Multiplier in bonus formula: bonus_max * (rel - neutral) * scale

    # === SATISFACTION FORMULA PARAMS ===
    # Satisfaction = EMA of quality_surplus (unbounded, 0=neutral, negative=unhappy)
    # quality_surplus = q_perceived - q_required(price)
    satisfaction_ema_alpha: float = 0.1  # EMA smoothing for satisfaction (0.1 = 10% new, 90% old)

    # === STICKINESS PARAMS ===
    stickiness_log_scale: float = 0.05  # Scale of log stickiness bonus per 30 days subscribed

    # === QUOTA PENALTY PARAMS ===
    quota_dissatisfaction_scale: float = 0.10  # Max penalty per unit of unfulfilled demand ratio

    # === OVERLOAD/OUTAGE SATISFACTION PENALTY (instant daily penalty before EMA) ===
    overload_satisfaction_weight: float = 0.15  # Satisfaction penalty per unit overload
    outage_satisfaction_weight: float = 0.25  # Satisfaction penalty when outage occurs

    # === ISSUE RESOLUTION PARAMS ===
    issue_resolution_base_rate: float = 2.0  # Issues resolved per day at $0 ops spending
    issue_resolution_ops_scale: float = 0.053  # Additional issues per $ ops spend per day
    quick_resolution_threshold_days: int = 2  # Max days for "quick" resolution bonus
    quick_resolution_boost_1day: float = 0.40  # Relationship boost for 1-day resolution
    quick_resolution_boost_2day: float = 0.30  # Relationship boost for 2-day resolution
    relationship_decay_per_unresolved_day: float = 0.01  # Relationship loss per unresolved issue day

    # === REPUTATION LEAD MULTIPLIER ===
    reputation_lead_multiplier_min: float = 0.6  # Rep=0 gives this multiplier on leads
    reputation_lead_multiplier_range: float = 0.8  # Rep=1 adds this to min (total = min+range = 1.4)

    # === V2.1: CHURN REPUTATION IMPACT (replaces budget-freeze-only reputation damage) ===
    # All churn events now have a chance to trigger a social post + reputation damage,
    # replacing the previous system where only budget freeze churn had reputation impact.
    # [Trustpilot 2024: 89% of customers share negative experiences after churning]
    # [Qualtrics 2024: Churned customers 3x more likely to leave negative reviews than satisfied ones]
    churn_reputation_post_probability: float = 0.3  # P(social post | churn event)
    churn_reputation_damage_multiplier: float = 0.3  # Reputation damage = base × this multiplier

    # === V2.1: SOCIAL MEDIA DIVERSITY (Section 2) ===
    # Strategies to reduce repetitive social media posts from template/LLM generation.
    # [Buffer 2024: Posts with unique voice/format get 2.3× more engagement]
    social_media_temperature: float = 0.95  # LLM temperature for social media posts (higher = more varied)
    social_media_diversity_window: int = 10  # Number of recent same-group posts to use as negative examples (V2.2: was 5)
    social_media_cross_group_dedup_window: int = 5  # V2.2: Recent cross-group posts to include in dedup

    # === V2.1: GROUP INFLUENCE & SOCIAL MEDIA VISIBILITY (Section 3) ===
    # Influencer groups (S3, E3, D_S07, D_S08, D_E07) have higher social media presence.
    # [Forrester 2024: Key opinion leaders generate 3-5× the word-of-mouth of average users]
    # [Gartner 2024: Enterprise tech leader recommendations drive 40% of SaaS evaluations]
    influencer_post_frequency_multiplier: float = 2.0  # Influencer groups post 2× more often
    ripple_post_probability: float = 0.15  # P(ripple post | influencer posts negative)

    # =============================================================================
    # MACROECONOMIC CYCLE SYSTEM
    # =============================================================================
    # Models a realistic business cycle using the ISM Purchasing Managers' Index (PMI).
    # PMI is a leading indicator of B2B purchasing intent, published monthly by the
    # Institute for Supply Management since 1948.
    #
    # PMI > 50 = economic expansion (businesses buying more)
    # PMI < 50 = economic contraction (businesses cutting back)
    # PMI = 50 = neutral / no change
    #
    # The PMI cycle affects customer behavior through group-specific sensitivity
    # coefficients (macro_beta), modifying churn rates, acquisition rates, willingness
    # to pay, and enterprise deal velocity.
    #
    # CITATIONS:
    # - ISM Manufacturing PMI methodology and historical data (1948-present):
    #   https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/
    # - Federal Reserve Bank of St. Louis: PMI historical series (FRED: NAPM)
    #   https://fred.stlouisfed.org/series/NAPM
    # - Koenig 2002: "Using the Purchasing Managers' Index to Assess the Economy's
    #   Strength." Federal Reserve Bank of Dallas Economic & Financial Review.
    #   Mean PMI: 52.9, std: 6.3. Cycle length: ~4.5 years peak-to-peak.
    # - Lahiri & Monokroussos 2013: "Nowcasting US GDP: The Role of ISM Business
    #   Surveys." International Journal of Forecasting 29(4): 644-658.
    #   PMI correlation with GDP growth: r=0.74. Leads GDP by 1-2 months.
    # - Pelaez 2003: "Globalization and the Purchasing Managers' Index."
    #   PMI below 42.2 historically signals recession (NBER concordance 100%).
    #
    # CYCLE CALIBRATION:
    # - Historical PMI range: 29.4 (May 1980) to 77.5 (July 1950)
    # - Post-2000 range: 33.1 (Dec 2008) to 64.7 (Mar 2004)
    # - Typical expansion: 55-62 | Typical contraction: 42-48
    # - Recession trough values: 33.1 (2008), 40.8 (2001), 41.5 (2020)
    # - Month-to-month volatility: std ~2.5 points (1-month change)
    #
    # Implementation uses a mean-reverting Ornstein-Uhlenbeck process overlaid
    # with a sinusoidal cycle, calibrated to match real PMI dynamics:
    #   PMI(t+1) = PMI(t) + theta * (mu(t) - PMI(t)) + sigma * N(0,1)
    #   mu(t) = macro_pmi_long_run_mean + amplitude * sin(2*pi*t / cycle_period + phase)
    # =============================================================================

    # --- PMI Cycle Core Parameters ---
    macro_pmi_initial: float = 52.9          # Starting PMI (historical long-run mean)
    macro_pmi_long_run_mean: float = 52.9    # Long-run equilibrium PMI
    # [FRED NAPM series: 52.9 mean, 1948-2025]
    macro_pmi_cycle_amplitude: float = 6.0   # Peak-to-trough half-swing in PMI points
    # [Historical: peaks ~59-62, troughs ~47-44, amplitude ~6-8 points from mean]
    macro_pmi_cycle_period_days: int = 1640  # ~4.5 years = 1640 days
    # [Koenig 2002: avg expansion ~65 months, contraction ~11 months, full cycle ~4.5 years]
    # [NBER: avg postwar cycle 69.5 months peak-to-peak ≈ 2115 days, but PMI leads by ~6mo]
    macro_pmi_mean_reversion_rate: float = 0.015  # Ornstein-Uhlenbeck theta (daily)
    # [Calibrated: half-life ≈ ln(2)/0.015 ≈ 46 days, matches PMI autocorrelation ~0.85 monthly]
    macro_pmi_daily_volatility: float = 0.4  # Daily noise std (points)
    # [ISM: monthly std ~2.5 points; daily ≈ 2.5/sqrt(22) ≈ 0.53; 0.4 slightly smoothed]
    macro_pmi_floor: float = 30.0            # Minimum possible PMI
    # [Historical minimum: 29.4 (May 1980), use 30.0 as floor]
    macro_pmi_ceiling: float = 70.0          # Maximum possible PMI
    # [Historical maximum: 77.5 (July 1950); post-1970 max ~64.7; use 70.0]
    macro_pmi_random_phase: bool = True      # Randomize initial cycle phase per seed
    # [Ensures different seeds produce different cycle timing]

    # --- PMI Update & Publication ---
    macro_pmi_update_interval_days: int = 30  # PMI updates monthly (like real ISM reports)
    # [ISM publishes PMI on the first business day of each month]
    macro_pmi_publication_delay_days: int = 30  # Days before PMI reading is visible to agent
    # [Real-world: ISM PMI covers the prior month's activity and is published ~30 days later.
    #  e.g., January PMI (measuring January activity) is published on first business day of February.
    #  This delay ensures the agent cannot see current macroeconomic conditions — only lagged data,
    #  matching real CEO information constraints. The simulation itself uses real-time PMI internally.]
    # [Source: ISM publication schedule: https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/]

    # --- Effect on Customer Behavior ---
    # PMI deviation from neutral (50) is scaled by group-specific beta coefficients.
    # Effect multiplier = 1.0 + macro_beta * (PMI - 50) / 50
    # At PMI=60 (strong expansion) with beta=0.3: multiplier = 1.0 + 0.3*(10/50) = 1.06 (+6%)
    # At PMI=40 (contraction) with beta=0.3: multiplier = 1.0 - 0.3*(10/50) = 0.94 (-6%)
    #
    # Effects applied to:
    #   1. Lead generation rate: more leads in expansion, fewer in contraction
    #   2. Willingness to pay (c_max): budgets expand/contract with economy
    #   3. Churn probability: inverse — higher churn in contraction
    #   4. Enterprise deal velocity: deals close faster in expansion
    #
    # CITATIONS for macro effects on SaaS:
    # - Tunguz 2023 GTM Survey: Sales cycles +24% overall, +36% enterprise in downturn
    #   https://tomtunguz.com/state-of-saas-2023/
    # - KeyBanc 2024 SaaS Survey: ARR growth dropped from 35% to 26% median in downturn
    #   https://www.key.com/kco/images/2024-SaaS-Survey-KeyBanc.pdf
    # - Bessemer 2024: NRR declined from 120% to 110% median during macro tightening
    #   https://www.bvp.com/atlas/state-of-the-cloud-2024
    # - Gartner 2020: IT spending -8% overall; enterprise software only -1.6% in 2009
    #   (enterprise is far less cyclical than SMB)
    # - ProfitWell 2023: SMB churn 5.8x higher than enterprise during downturns
    #   https://www.profitwell.com/recur/all/saas-churn-benchmarks
    # - SaaStr 2020: Recessions affect SMB SaaS 2-3x more than enterprise
    #   https://www.saastr.com/saas-and-a-recession/

    # --- Macro Social Media Posts ---
    # Periodically, social media posts about the macroeconomic situation appear.
    # Generated by Bedrock Haiku on the fly, reflecting current PMI conditions.
    # Frequency is random: every N days, M posts appear.
    macro_social_post_interval_min: int = 5   # Minimum days between macro post batches
    macro_social_post_interval_max: int = 20  # Maximum days between macro post batches
    # [Real-world: business news cycles vary; major economic reports monthly, sentiment daily]
    macro_social_post_count_min: int = 2      # Minimum posts per batch
    macro_social_post_count_max: int = 8      # Maximum posts per batch
    # [Calibrated: 2-8 posts every 5-20 days ≈ 4-50 macro posts/month, matching news volume]


# =============================================================================
# MACROECONOMIC SENSITIVITY COEFFICIENTS (per customer group)
# =============================================================================
# Each group has a `macro_beta` dict specifying how sensitive it is to PMI changes.
# Beta values are calibrated from real-world cyclical sensitivity research.
#
# Three dimensions of macro sensitivity:
#   lead_generation: How much new customer acquisition changes with PMI
#   willingness_to_pay: How much budget/c_max shifts with PMI (also drives churn indirectly — lower c_max → plan downgrade/cancel)
#   deal_velocity: How much enterprise deal speed changes (enterprise only)
#
# INTERPRETATION:
#   beta = 0.0 → completely acyclical (no macro sensitivity)
#   beta = 0.3 → moderate sensitivity (±6% at PMI=40/60)
#   beta = 0.6 → high sensitivity (±12% at PMI=40/60)
#   beta = 1.0 → extreme sensitivity (±20% at PMI=40/60)
#
# CITATIONS for segment-specific sensitivity:
# - McKinsey 2020: "COVID-19: Implications for business." SMB revenue fell 30-50%,
#   enterprise only 5-15% in first 6 months of pandemic.
#   https://www.mckinsey.com/capabilities/risk-and-resilience/our-insights/covid-19-implications-for-business
# - Bain & Company 2023: "Global Private Equity Report." Cyclical sectors
#   (retail, manufacturing) saw 2-3x revenue volatility vs defensive sectors.
#   https://www.bain.com/insights/topics/global-private-equity-report/
# - Gartner 2020: Government IT spending grew +4.1% DURING 2009 recession
#   (countercyclical, as governments implement stimulus programs).
# - KLAS Research 2020: Healthcare IT spending was flat (-1%) during 2020 pandemic
#   despite overall IT spending -8%.
# - Deloitte 2020: Banking technology spend dropped only -3% in 2020 (regulatory
#   requirements maintain minimum spending floors).
# - ProfitWell 2023: SMB SaaS churn rates 5.8x higher than enterprise during downturns.
#   Freelancer/gig economy segments see 2-3x normal churn in recessions.
# - SaaStr 2020: Enterprise contracts provide 6-12 month lag before macro effects hit.
# - Construction & Real Estate are the MOST cyclically sensitive sectors
#   (NBER: construction employment drops 15-20% in recessions).
# - Bureau of Labor Statistics: Manufacturing output falls 10-15% peak-to-trough in
#   typical recessions, while healthcare grows 2-3% through cycles.
# - Springer 2025: Media sentiment leads PMI by ~24 days; social media managers
#   and content creators are early-cycle responders.
#   https://link.springer.com/article/10.1007/s10479-024-06255-z
#
# FORMAT: {group_id: {dimension: beta_value}}
MACRO_SENSITIVITY: Dict[str, Dict[str, float]] = {
    # === Initial Groups ===

    # S1: Price-Sensitive Individuals (freelancers, gig workers, students)
    # HIGHLY cyclical: irregular income, first to cut discretionary subscriptions
    # [ProfitWell 2023: SMB churn 5.8x higher in downturns; DemandSage 2025: 70% freelancers month-to-month]
    'S1': {
        'lead_generation': 0.50,      # Freelancer demand drops significantly in downturns
        'willingness_to_pay': 0.60,   # Budgets shrink fast — gig income is pro-cyclical
        'deal_velocity': 0.0,         # N/A (not enterprise)
    },

    # S2: Quality-Focused Individuals (lawyers, consultants, healthcare professionals)
    # MODERATE cyclicality: employed professionals with more stable income
    # [BCG 2024: 68% professionals pay premium; KeyBanc 2024: professional tools $60-150/mo]
    'S2': {
        'lead_generation': 0.25,      # Professional demand moderately affected
        'willingness_to_pay': 0.20,   # Employer-funded budgets more stable
        'deal_velocity': 0.0,         # N/A
    },

    # S3: Power Users (developers, data scientists)
    # LOW-MODERATE cyclicality: tech workers affected by layoffs but tool-dependent
    # [GitHub Copilot: 75% YoY growth even through 2023 downturn; Sacra 2024: dev tools resilient]
    'S3': {
        'lead_generation': 0.30,      # Tech hiring cycles affect new adoption
        'willingness_to_pay': 0.15,   # Devs maintain tooling budgets even in downturns
        'deal_velocity': 0.0,         # N/A
    },

    # E1: Cost-Cutting Enterprises (manufacturing, logistics, retail)
    # HIGH cyclicality: these sectors are textbook cyclical industries
    # [Bain 2023: cyclical sectors 2-3x revenue volatility; BLS: manufacturing -10-15% in recessions]
    'E1': {
        'lead_generation': 0.45,      # New vendor evaluation freezes in downturns
        'willingness_to_pay': 0.40,   # Budget cuts hit discretionary SaaS first
        'deal_velocity': 0.45,        # Deal cycles lengthen 30-40% in contraction
        'seat_count': 0.35,           # Cyclical layoffs: manufacturing sheds 10-15% in recessions (BLS)
    },

    # E2: Quality-First Enterprises (law firms, biotech, financial services)
    # LOW cyclicality: regulated industries maintain spending floors
    # [Deloitte 2020: banking tech -3% in 2020; Gartner: enterprise software -1.6% in 2009]
    'E2': {
        'lead_generation': 0.15,      # Evaluation continues but slows slightly
        'willingness_to_pay': 0.10,   # Budgets protected by regulatory requirements
        'deal_velocity': 0.20,        # Slight slowdown in approval committees
        'seat_count': 0.08,           # Regulated: headcount insulated, hiring freezes only
    },

    # E3: Strategic Partners (Fortune 500, large enterprises)
    # LOW cyclicality: long-term contracts and strategic initiatives buffer macro shocks
    # [SaaStr 2020: enterprise contracts 6-12 month lag; McKinsey: enterprise -5-15% vs SMB -30-50%]
    'E3': {
        'lead_generation': 0.20,      # Strategic initiatives continue but fewer new ones
        'willingness_to_pay': 0.12,   # Multi-year budgets pre-allocated
        'deal_velocity': 0.30,        # Committee approvals slow in uncertainty
        'seat_count': 0.15,           # Fortune 500: slow hiring freezes, -5-15% in deep recession
    },

    # === Discoverable Individual Groups (D_S01 - D_S10) ===

    # D_S01: Niche Creators (digital art, crafts, photography)
    # HIGH cyclicality: discretionary creative work shrinks in downturns
    # [Upwork 2025: creative freelancer income drops 25-35% in recessions]
    'D_S01': {
        'lead_generation': 0.55,
        'willingness_to_pay': 0.65,   # Highly discretionary income
        'deal_velocity': 0.0,
    },

    # D_S02: Academic Researchers (universities, labs)
    # VERY LOW cyclicality: grant-funded, multi-year budgets, countercyclical (stimulus)
    # [Nature 2024: research budgets relatively insulated from business cycles]
    # [Gartner 2020: government/education spending +4.1% during 2009 recession]
    'D_S02': {
        'lead_generation': 0.08,
        'willingness_to_pay': 0.05,   # Grant budgets predetermined
        'deal_velocity': 0.0,
    },

    # D_S03: Non-Profit Workers (charities, NGOs)
    # MODERATE-HIGH cyclicality: donation-dependent funding shrinks in downturns
    # [NTEN 2025: 40% of nonprofits cut tech budgets during 2020 downturn]
    'D_S03': {
        'lead_generation': 0.40,
        'willingness_to_pay': 0.50,   # Donation-funded budgets are pro-cyclical
        'deal_velocity': 0.0,
    },

    # D_S04: Small Agency Teams (design, marketing, PR agencies)
    # HIGH cyclicality: client project volume directly tied to business cycle
    # [HubSpot 2025: agency revenue drops 20-30% in downturns as clients cut marketing]
    'D_S04': {
        'lead_generation': 0.50,
        'willingness_to_pay': 0.45,
        'deal_velocity': 0.0,
    },

    # D_S05: Indie Game Devs (game development, VR, interactive media)
    # MODERATE cyclicality: gaming is partially countercyclical (entertainment demand)
    # [GDC 2025: indie dev funding affected, but game sales resilient in recessions]
    'D_S05': {
        'lead_generation': 0.30,
        'willingness_to_pay': 0.35,
        'deal_velocity': 0.0,
    },

    # D_S06: Freelance Writers (copywriting, journalism, blogging)
    # HIGH cyclicality: content budgets are among first cuts in downturns
    # [Contently 2025: freelance writing gigs drop 30-40% in recessions]
    'D_S06': {
        'lead_generation': 0.50,
        'willingness_to_pay': 0.55,
        'deal_velocity': 0.0,
    },

    # D_S07: Data Analysts (BI, market research, analytics)
    # LOW-MODERATE cyclicality: data-driven decisions become MORE important in downturns
    # [Kaggle 2024: analytics tool usage stable through downturns]
    'D_S07': {
        'lead_generation': 0.20,
        'willingness_to_pay': 0.15,
        'deal_velocity': 0.0,
    },

    # D_S08: Social Media Managers (brand management, content scheduling)
    # MODERATE-HIGH: marketing budgets are pro-cyclical
    # [Sprout Social 2025: 35% of SM managers lost tool budgets in 2023 downturn]
    'D_S08': {
        'lead_generation': 0.40,
        'willingness_to_pay': 0.45,
        'deal_velocity': 0.0,
    },

    # D_S09: UX Designers (product design, user research)
    # MODERATE: tech layoffs affect UX roles, but employed designers maintain tools
    # [Nielsen Norman 2024: design tool spending relatively stable; layoffs are the risk]
    'D_S09': {
        'lead_generation': 0.25,
        'willingness_to_pay': 0.20,
        'deal_velocity': 0.0,
    },

    # D_S10: Music Producers (audio engineering, beat-making)
    # HIGH cyclicality: creative freelancers, discretionary entertainment spending
    # [MIDiA 2025: independent music producer income highly variable with economy]
    'D_S10': {
        'lead_generation': 0.45,
        'willingness_to_pay': 0.55,
        'deal_velocity': 0.0,
    },

    # === Discoverable Enterprise Groups (D_E01 - D_E10) ===

    # D_E01: Government Agencies
    # COUNTERCYCLICAL: government spending increases during recessions (stimulus)
    # [Gartner 2020: government IT spending +4.1% during 2009 recession]
    # [BLS: federal employment countercyclical, state/local slightly pro-cyclical]
    'D_E01': {
        'lead_generation': -0.15,     # NEGATIVE beta: MORE procurement in downturns (stimulus)
        'willingness_to_pay': -0.05,  # Budget increases slightly with stimulus
        'deal_velocity': -0.10,       # Slight acceleration (urgency to deploy stimulus)
        'seat_count': -0.20,          # Countercyclical but DOGE/efficiency cuts dominate
    },

    # D_E02: Educational Institutions
    # LOW cyclicality: enrollment often rises in recessions (people go back to school)
    # [Mordor Intelligence 2025: ed-tech spending insulated by tuition revenue stability]
    'D_E02': {
        'lead_generation': 0.10,
        'willingness_to_pay': 0.08,
        'deal_velocity': 0.15,        # Budget committee approvals slow slightly
        'seat_count': 0.05,           # Education: very stable staffing
    },

    # D_E03: Healthcare Networks
    # VERY LOW cyclicality: healthcare demand is acyclical (people get sick regardless)
    # [KLAS 2020: healthcare IT spending flat (-1%) during 2020 pandemic]
    # [BLS: healthcare employment grows 2-3% through every recession since 1970]
    'D_E03': {
        'lead_generation': 0.05,
        'willingness_to_pay': 0.03,
        'deal_velocity': 0.10,
        'seat_count': 0.03,           # Healthcare: virtually no macro impact on headcount
    },

    # D_E04: Regional Banks
    # MODERATE cyclicality: credit quality deteriorates, but regulatory spending is mandatory
    # [Deloitte 2020: banking tech -3% in 2020; OCC mandates maintain compliance spending]
    'D_E04': {
        'lead_generation': 0.25,
        'willingness_to_pay': 0.20,
        'deal_velocity': 0.30,        # Risk committee reviews slow significantly
        'seat_count': 0.20,           # Banking: branch layoffs in downturns
    },

    # D_E05: Insurance Brokers
    # LOW-MODERATE cyclicality: claims volume may rise but premium income is sticky
    # [Novarica 2025: insurance tech spending grew through 2020; claims automation increased]
    'D_E05': {
        'lead_generation': 0.15,
        'willingness_to_pay': 0.12,
        'deal_velocity': 0.20,
        'seat_count': 0.10,           # Insurance: moderate, claims staff needed regardless
    },

    # D_E06: Construction Firms
    # VERY HIGH cyclicality: construction is the MOST cyclical major sector
    # [NBER: construction employment drops 15-20% peak-to-trough in recessions]
    # [BLS 2020: construction output fell 18% in 2008-2009 recession]
    'D_E06': {
        'lead_generation': 0.65,
        'willingness_to_pay': 0.55,
        'deal_velocity': 0.55,
        'seat_count': 0.55,           # Construction: -15-20% headcount in recessions (BLS/NBER)
    },

    # D_E07: Telecom Operators
    # LOW cyclicality: essential infrastructure with recurring revenue
    # [TM Forum 2025: telecom capex grows through cycles due to network upgrade mandates]
    'D_E07': {
        'lead_generation': 0.12,
        'willingness_to_pay': 0.08,
        'deal_velocity': 0.15,
        'seat_count': 0.06,           # Telecom: essential infrastructure, minimal layoffs
    },

    # D_E08: Energy Companies
    # MODERATE cyclicality: tied to commodity prices, but utilities segment is defensive
    # [Wood Mackenzie 2025: energy tech spending correlated with oil prices at r=0.6]
    'D_E08': {
        'lead_generation': 0.30,
        'willingness_to_pay': 0.25,
        'deal_velocity': 0.25,
        'seat_count': 0.20,           # Energy: tied to commodity prices, moderate layoffs
    },

    # D_E09: Real Estate Groups
    # VERY HIGH cyclicality: real estate is the second most cyclical sector after construction
    # [Deloitte Real Estate 2025: CRE tech switching increased 40% in downturns]
    # [NBER: commercial real estate investment drops 25-35% in recessions]
    'D_E09': {
        'lead_generation': 0.60,
        'willingness_to_pay': 0.50,
        'deal_velocity': 0.50,
        'seat_count': 0.50,           # Real estate: -25-35% in recessions (NBER)
    },

    # D_E10: Shipping Lines
    # HIGH cyclicality: global trade volume directly tied to business cycle
    # [Drewry Maritime 2025: container shipping volumes drop 10-15% in recessions]
    'D_E10': {
        'lead_generation': 0.45,
        'willingness_to_pay': 0.35,
        'deal_velocity': 0.40,
        'seat_count': 0.30,           # Shipping: trade-linked headcount volatility
    },
}


def compute_term_sheet_friendliness(
    config: 'BenchmarkConfig',
    anti_dilution_floor: float = None,
    tranche_pct: float = None,
    revenue_multiplier: float = None,
    deadline_days: int = None,
    redemption_days: int = None,
    buyback_multiplier: float = None,
) -> float:
    """Compute how VC-friendly a set of term sheet options is.

    Each parameter is scored 0.0 (least VC-friendly) to 1.0 (most VC-friendly)
    based on its position in the options range. The total friendliness is the
    weighted sum of all active term sheet items.

    Returns:
        float representing the total valuation adjustment multiplier.
        e.g., 0.12 means the VC's fair valuation should be multiplied by 1.12
    """
    total = 0.0

    # Anti-dilution: higher floor = more VC protection = more friendly
    if anti_dilution_floor is not None:
        opts = config.vc_anti_dilution_floor_options
        lo, hi = min(opts), max(opts)
        score = (anti_dilution_floor - lo) / (hi - lo) if hi > lo else 0.5
        total += config.vc_anti_dilution_impact_weight * score

    # Milestone tranching: average of 3 sub-parameter scores
    milestone_scores = []
    if tranche_pct is not None:
        opts = config.vc_milestone_tranche_pct_options
        lo, hi = min(opts), max(opts)
        # LOWER upfront % = more VC-friendly (more gated → VC safer)
        score = 1.0 - ((tranche_pct - lo) / (hi - lo)) if hi > lo else 0.5
        milestone_scores.append(score)
    if revenue_multiplier is not None:
        opts = config.vc_milestone_revenue_multiplier_options
        lo, hi = min(opts), max(opts)
        # Higher multiplier = harder milestone = more VC-friendly
        score = (revenue_multiplier - lo) / (hi - lo) if hi > lo else 0.5
        milestone_scores.append(score)
    if deadline_days is not None:
        opts = config.vc_milestone_deadline_days_options
        lo, hi = min(opts), max(opts)
        # Shorter deadline = more pressure on founder = more VC-friendly
        score = 1.0 - ((deadline_days - lo) / (hi - lo)) if hi > lo else 0.5
        milestone_scores.append(score)
    if milestone_scores:
        avg_milestone = sum(milestone_scores) / len(milestone_scores)
        total += config.vc_milestone_impact_weight * avg_milestone

    # Redemption rights: average of 2 sub-parameter scores
    redemption_scores = []
    if redemption_days is not None:
        opts = config.vc_redemption_days_options
        lo, hi = min(opts), max(opts)
        # Shorter window = more VC power = more friendly
        score = 1.0 - ((redemption_days - lo) / (hi - lo)) if hi > lo else 0.5
        redemption_scores.append(score)
    if buyback_multiplier is not None:
        opts = config.vc_redemption_buyback_multiplier_options
        lo, hi = min(opts), max(opts)
        # Higher multiplier = costlier for founder = more VC-friendly
        score = (buyback_multiplier - lo) / (hi - lo) if hi > lo else 0.5
        redemption_scores.append(score)
    if redemption_scores:
        avg_redemption = sum(redemption_scores) / len(redemption_scores)
        total += config.vc_redemption_impact_weight * avg_redemption

    return total


# =============================================================================
# R&D RESEARCH TIERS
# =============================================================================
# R&D tiers provide large, permanent quality boosts that are impractical to
# achieve via dev spending alone (dev spending saturates logarithmically).
# The agent MUST invest in R&D to keep pace with competitor events.
#
# 10 independent tiers — no dependencies. Any tier can be started at any time.
# Tiers are REPEATABLE — the same tier can be started multiple times.
# Cost grows linearly ($100K/tier); delay and quality boost grow non-linearly.
# HIGH VARIANCE: both duration and quality are sampled from Normal distributions
# with large standard deviations (~50% CV for quality, ~40-50% CV for delay).
#
# Economics (why R&D is necessary):
# - Dev spending at $1K/day = +0.25 quality/year ($365K). Logarithmic saturation.
# - Competitor pressure = ~+0.35 quality/year (events + drift).
# - Gap: ~+0.10/year that dev spending CANNOT close at any price.
# - Agent needs ~2-4 R&D invocations per year to stay competitive.

@dataclass
class ResearchTier:
    """A research tier the agent can invest in."""
    tier: int                    # Tier number (1-20)
    name: str                    # Human-readable name
    description: str             # What this tier achieves
    cost: float                  # One-time cost to start ($)
    mean_days: int               # Mean duration in days
    std_days: int                # Std deviation of duration in days
    mean_quality_boost: float    # Mean permanent quality improvement
    std_quality_boost: float     # Std deviation of quality improvement


RESEARCH_TIERS: List[ResearchTier] = [
    # Cost: linear ($100K per tier). Significant investment.
    # Delay: non-linear growth, all >= 30 days, ~40-50% CV
    # Quality: non-linear growth, ~50% CV (high risk/reward)
    ResearchTier(tier=1,  name="Prompt Engineering Optimization",
                 description="Systematic prompt tuning and output consistency improvements",
                 cost=100_000,  mean_days=35,  std_days=12,  mean_quality_boost=0.04,  std_quality_boost=0.020),
    ResearchTier(tier=2,  name="Evaluation & Testing Pipeline",
                 description="Automated quality evaluation, regression testing, and A/B experimentation",
                 cost=200_000,  mean_days=50,  std_days=20,  mean_quality_boost=0.07,  std_quality_boost=0.035),
    ResearchTier(tier=3,  name="Caching & Latency Optimization",
                 description="Smart caching layer, response latency improvements, and query optimization",
                 cost=300_000,  mean_days=70,  std_days=30,  mean_quality_boost=0.11,  std_quality_boost=0.055),
    ResearchTier(tier=4,  name="Fine-Tuning Infrastructure",
                 description="Custom fine-tuning pipeline for domain-specific model improvements",
                 cost=400_000,  mean_days=95,  std_days=42,  mean_quality_boost=0.16,  std_quality_boost=0.080),
    ResearchTier(tier=5,  name="RAG & Knowledge Integration",
                 description="Retrieval-augmented generation with re-ranking and knowledge graph integration",
                 cost=500_000,  mean_days=125, std_days=58,  mean_quality_boost=0.22,  std_quality_boost=0.110),
    ResearchTier(tier=6,  name="Multi-Modal Support",
                 description="Image, document, and structured data understanding capabilities",
                 cost=600_000,  mean_days=160, std_days=75,  mean_quality_boost=0.30,  std_quality_boost=0.150),
    ResearchTier(tier=7,  name="Agentic Capabilities",
                 description="Multi-step reasoning, tool use, and autonomous task completion",
                 cost=700_000,  mean_days=200, std_days=95,  mean_quality_boost=0.40,  std_quality_boost=0.200),
    ResearchTier(tier=8,  name="RLHF & Alignment",
                 description="Reinforcement learning from human feedback for preference alignment",
                 cost=800_000,  mean_days=250, std_days=120, mean_quality_boost=0.52,  std_quality_boost=0.260),
    ResearchTier(tier=9,  name="Next-Gen Architecture",
                 description="Major model architecture upgrade for step-change quality improvement",
                 cost=900_000,  mean_days=310, std_days=150, mean_quality_boost=0.67,  std_quality_boost=0.335),
    ResearchTier(tier=10, name="Self-Evolving Model Ecosystem",
                 description="Orchestrated system of specialized models that self-optimize and continuously improve",
                 cost=1_000_000, mean_days=380, std_days=185, mean_quality_boost=0.85, std_quality_boost=0.425),

    # --- Frontier Tiers (11-20) ---
    # Cost: super-linear growth ($1.5M to $15M). Major capital commitments.
    # Delay: long timelines (450d to 1400d mean), very high variance (~55-80% CV)
    # Quality: large boosts (1.1 to 8.0 mean), very high variance (~55-75% CV)
    # These are cheaper per quality point than stacking smaller tiers, but much riskier.
    ResearchTier(tier=11, name="Synthetic Data Engine",
                 description="Large-scale synthetic data generation and curriculum learning pipeline for domain coverage",
                 cost=1_500_000,  mean_days=420,  std_days=230,  mean_quality_boost=1.10,  std_quality_boost=0.660),
    ResearchTier(tier=12, name="Distributed Training Cluster",
                 description="Multi-node distributed training infrastructure for full model retraining at scale",
                 cost=2_200_000,  mean_days=480,  std_days=290,  mean_quality_boost=1.40,  std_quality_boost=0.840),
    ResearchTier(tier=13, name="Constitutional AI Framework",
                 description="Advanced safety and alignment framework with self-critique and reward model ensemble",
                 cost=3_000_000,  mean_days=550,  std_days=360,  mean_quality_boost=1.75,  std_quality_boost=1.050),
    ResearchTier(tier=14, name="Mixture of Experts Overhaul",
                 description="Sparse mixture-of-experts architecture with dynamic routing and expert specialization",
                 cost=4_000_000,  mean_days=630,  std_days=440,  mean_quality_boost=2.15,  std_quality_boost=1.400),
    ResearchTier(tier=15, name="World Model & Reasoning Core",
                 description="Internal world model for causal reasoning, planning, and counterfactual simulation",
                 cost=5_500_000,  mean_days=720,  std_days=540,  mean_quality_boost=2.70,  std_quality_boost=1.890),
    ResearchTier(tier=16, name="Autonomous Research Agent",
                 description="Self-directed research loop that identifies weaknesses and designs targeted training runs",
                 cost=7_000_000,  mean_days=820,  std_days=620,  mean_quality_boost=3.40,  std_quality_boost=2.380),
    ResearchTier(tier=17, name="Neural Architecture Search",
                 description="Automated architecture discovery using evolutionary search over billion-parameter design space",
                 cost=9_000_000,  mean_days=950,  std_days=720,  mean_quality_boost=4.20,  std_quality_boost=3.150),
    ResearchTier(tier=18, name="Foundation Model Distillation",
                 description="Multi-teacher distillation from frontier models into a compact, specialized powerhouse",
                 cost=11_000_000, mean_days=1080, std_days=830,  mean_quality_boost=5.20,  std_quality_boost=3.900),
    ResearchTier(tier=19, name="Recursive Self-Improvement",
                 description="Model that iteratively improves its own training process and data selection strategy",
                 cost=13_000_000, mean_days=1250, std_days=1000, mean_quality_boost=6.50,  std_quality_boost=5.200),
    ResearchTier(tier=20, name="Artificial General Reasoning",
                 description="Moonshot program for general-purpose reasoning across all domains with emergent capabilities",
                 cost=15_000_000, mean_days=1400, std_days=1120, mean_quality_boost=8.00,  std_quality_boost=6.400),
]

RESEARCH_TIERS_BY_ID: Dict[int, ResearchTier] = {rt.tier: rt for rt in RESEARCH_TIERS}


# =============================================================================
# PREDEFINED VC INVESTORS
# =============================================================================
# Each VC is a named firm with a target equity % range and daily approach probability.
# The agent can see the full list via list_potential_vcs().
# When a VC reaches out, their equity ask is randomly sampled from [equity_pct_min, equity_pct_max],
# and the check size is derived as: check = pct / (1 - pct) × valuation.
#
# CITATIONS for VC equity ranges:
# - Carta 2024: Seed dilution 10-20%, Series A 15-25%
# - PitchBook 2024: Median seed dilution ~18%, early-stage ~20%
# - AngelList 2024: Angel rounds 2-10% dilution
# - NVCA 2024: Average VC fund deploys 15-25 deals per fund

@dataclass
class VCValuationWeights:
    """Weights for multi-variable company valuation formula.

    Each VC has unique weights reflecting their investment philosophy.
    Weights are normalized to sum to 1.0.
    """
    base_multiple: float = 10.0  # Base ARR multiple
    w_growth: float = 0.20       # Weight on revenue growth rate
    w_retention: float = 0.15    # Weight on net revenue retention
    w_margin: float = 0.10       # Weight on gross margin
    w_scale: float = 0.10        # Weight on ARR scale (log)
    w_quality: float = 0.10      # Weight on product quality
    w_market: float = 0.08       # Weight on market size/TAM
    w_efficiency: float = 0.10   # Weight on capital efficiency
    w_momentum: float = 0.07    # Weight on growth momentum
    w_runway: float = 0.05       # Weight on cash runway
    w_diversity: float = 0.05    # Weight on customer diversity


@dataclass
class VCProfile:
    """Profile for a predefined VC investor."""
    vc_id: str                    # Unique identifier (e.g., "vc_01")
    name: str                     # Firm name
    equity_pct_min: float         # Minimum equity % the VC targets (e.g. 0.05 = 5%)
    equity_pct_max: float         # Maximum equity % the VC targets (e.g. 0.25 = 25%)
    daily_approach_prob: float    # Probability of reaching out on any given day
    reply_delay_mean: float = 3.0   # Mean days to reply during negotiation
    reply_delay_std: float = 1.0    # Std dev of reply delay
    description: str = ""          # Brief description visible to agent
    valuation_weights: VCValuationWeights = field(default_factory=VCValuationWeights)
    # Macro sensitivity: how much this VC's valuation scales with ISM PMI.
    # Formula: valuation_macro_mult = 1.0 + macro_sensitivity * (PMI - 50) / 50
    # Higher = more procyclical (valuation swings with economy).
    # Near-zero or negative = countercyclical (less affected or contrarian).
    #
    # CITATIONS:
    # - PitchBook 2024: Late-stage valuations dropped first and hardest in 2022 downturn;
    #   early-stage remained "stubbornly high" — late-stage ~2-3x more macro-sensitive.
    #   https://pitchbook.com/newsletter/late-stage-valuations-decline-in-q1-as-pressure-mounts-on-vc-deal-terms
    # - AngelList 2024: Early-stage valuations least impacted by short-term trends;
    #   investors view it as a "long game" with exits years away.
    #   https://www.angellist.com/blog/how-far-will-early-stage-valuations-fall
    # - ACA 2024: Angel group funding dropped 33% in volume but valuations held —
    #   too much capital chasing deals insulates early-stage from macro.
    #   https://angelcapitalassociation.org/blog/why-have-early-stage-valuations-remained-high/
    # - ScienceDirect 2016: SWFs behave countercyclically, increasing acquisitions in
    #   crisis-hit countries — contrarian stabilizing role.
    #   https://www.sciencedirect.com/science/article/abs/pii/S1566014116300218
    # - IMF WP/16/038: Institutional countercyclical investment — SWFs increase risky
    #   allocations after crisis onset to exploit depressed valuations.
    #   https://www.elibrary.imf.org/view/journals/001/2016/038/article-A001-en.xml
    # - Kauffman Fellows 2020: Family offices provide "patient capital" — less reactive
    #   to external market pressures and fundraising cycles.
    #   https://www.kauffmanfellows.org/journal/family-office-venture-capital-outlook-2020
    # - Cambridge Associates / Tandfonline: University endowments invest countercyclically
    #   at crisis times, increasing allocations to risky assets after a crisis.
    #   https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1802984
    # - Tandfonline 2025: CVC funds are "more generous and more patient" than IVC;
    #   strategic orientation reduces pure macro sensitivity.
    #   https://www.tandfonline.com/doi/full/10.1080/08985626.2025.2599516
    # - Supervest 2025: Revenue-based financing directly tied to borrower revenue —
    #   moderate macro sensitivity through cash flow alignment.
    #   https://www.supervest.com/blog/revenue-based-financing-in-a-time-of-economic-uncertainty
    # - PitchBook 2024: Crossover/late-stage funds act as public-market proxies,
    #   first to retreat in downturns, highest macro sensitivity.
    #   https://pitchbook.com/newsletter/inside-the-highs-and-lows-of-us-vc-valuations
    macro_sensitivity: float = 0.20  # Default: moderate procyclical


PREDEFINED_VCS: list = [
    # Equity pct ranges based on real VC data (Carta 2024, PitchBook 2024, AngelList 2024):
    # - Angels/pre-seed: 2-10% (small checks, high dilution tolerance)
    # - Seed: 5-15% (standard seed round dilution)
    # - Series A: 10-25% (institutional round)
    # - Growth/Series B+: 8-20% (larger checks, lower dilution)
    # - Late-stage/crossover: 5-15% (mega checks, minimal dilution)
    VCProfile(
        vc_id="vc_01", name="Horizon Ventures",
        equity_pct_min=0.03, equity_pct_max=0.10,
        daily_approach_prob=0.008,
        reply_delay_mean=2.0, reply_delay_std=0.5,
        description="Early-stage micro-VC focused on AI/ML startups",
        valuation_weights=VCValuationWeights(base_multiple=8.0, w_growth=0.25, w_retention=0.10, w_margin=0.05, w_scale=0.05, w_quality=0.20, w_market=0.10, w_efficiency=0.05, w_momentum=0.10, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.12,  # Micro-VC: early-stage insulated from macro (ACA 2024, AngelList 2024)
    ),
    VCProfile(
        vc_id="vc_02", name="Catalyst Capital",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.006,
        reply_delay_mean=3.0, reply_delay_std=1.0,
        description="Seed-stage fund investing in developer tools",
        valuation_weights=VCValuationWeights(base_multiple=12.0, w_growth=0.22, w_retention=0.12, w_margin=0.08, w_scale=0.08, w_quality=0.12, w_market=0.10, w_efficiency=0.10, w_momentum=0.08, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.18,  # Seed fund: moderate insulation, some LP pressure (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_03", name="Apex Partners",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.005,
        reply_delay_mean=4.0, reply_delay_std=1.5,
        description="Seed to Series A investor in B2B SaaS",
        valuation_weights=VCValuationWeights(base_multiple=12.0, w_growth=0.22, w_retention=0.12, w_margin=0.08, w_scale=0.08, w_quality=0.12, w_market=0.10, w_efficiency=0.10, w_momentum=0.08, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.22,  # Seed-to-A: bridges early/institutional — moderate sensitivity (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_04", name="Meridian Fund",
        equity_pct_min=0.10, equity_pct_max=0.25,
        daily_approach_prob=0.004,
        reply_delay_mean=5.0, reply_delay_std=2.0,
        description="Series A fund targeting high-growth AI companies",
        valuation_weights=VCValuationWeights(base_multiple=15.0, w_growth=0.18, w_retention=0.18, w_margin=0.12, w_scale=0.10, w_quality=0.08, w_market=0.08, w_efficiency=0.12, w_momentum=0.06, w_runway=0.04, w_diversity=0.04),
        macro_sensitivity=0.35,  # Series A institutional: LP-driven, benchmark-conscious (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_05", name="Summit Equity",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.003,
        reply_delay_mean=7.0, reply_delay_std=2.0,
        description="Growth-stage investor in enterprise AI platforms",
        valuation_weights=VCValuationWeights(base_multiple=20.0, w_growth=0.15, w_retention=0.20, w_margin=0.15, w_scale=0.15, w_quality=0.05, w_market=0.05, w_efficiency=0.10, w_momentum=0.05, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.45,  # Growth-stage: high sensitivity, closer to public comps (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_06", name="Forge Ventures",
        equity_pct_min=0.03, equity_pct_max=0.12,
        daily_approach_prob=0.007,
        reply_delay_mean=2.0, reply_delay_std=1.0,
        description="Pre-seed/seed fund focused on technical founders",
        valuation_weights=VCValuationWeights(base_multiple=8.0, w_growth=0.25, w_retention=0.10, w_margin=0.05, w_scale=0.05, w_quality=0.20, w_market=0.10, w_efficiency=0.05, w_momentum=0.10, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.10,  # Pre-seed: most insulated, thesis-driven conviction (AngelList 2024, ACA 2024)
    ),
    VCProfile(
        vc_id="vc_07", name="Beacon Capital",
        equity_pct_min=0.06, equity_pct_max=0.18,
        daily_approach_prob=0.004,
        reply_delay_mean=4.0, reply_delay_std=1.5,
        description="Seed fund specializing in API-first businesses",
        valuation_weights=VCValuationWeights(base_multiple=12.0, w_growth=0.22, w_retention=0.12, w_margin=0.08, w_scale=0.08, w_quality=0.12, w_market=0.10, w_efficiency=0.10, w_momentum=0.08, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.20,  # Seed: moderate, infrastructure focus adds stability (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_08", name="Vanguard Growth",
        equity_pct_min=0.10, equity_pct_max=0.22,
        daily_approach_prob=0.002,
        reply_delay_mean=7.0, reply_delay_std=3.0,
        description="Series A-B fund for market-leading SaaS companies",
        valuation_weights=VCValuationWeights(base_multiple=20.0, w_growth=0.15, w_retention=0.20, w_margin=0.15, w_scale=0.15, w_quality=0.05, w_market=0.05, w_efficiency=0.10, w_momentum=0.05, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.48,  # Series A-B growth: highly procyclical, benchmarked to public SaaS (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_09", name="Pinnacle Investments",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.001,
        reply_delay_mean=10.0, reply_delay_std=3.0,
        description="Late-stage growth equity for category leaders",
        valuation_weights=VCValuationWeights(base_multiple=25.0, w_growth=0.10, w_retention=0.20, w_margin=0.20, w_scale=0.20, w_quality=0.05, w_market=0.03, w_efficiency=0.10, w_momentum=0.02, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.55,  # Late-stage: "first to be impacted by market fluctuations" (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_10", name="Atlas Ventures",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.006,
        reply_delay_mean=3.0, reply_delay_std=1.0,
        description="Multi-stage fund with focus on infrastructure software",
        valuation_weights=VCValuationWeights(base_multiple=12.0, w_growth=0.22, w_retention=0.12, w_margin=0.08, w_scale=0.08, w_quality=0.12, w_market=0.10, w_efficiency=0.10, w_momentum=0.08, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.25,  # Multi-stage: blended across stages, infra focus adds resilience (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_11", name="Nexus Partners",
        equity_pct_min=0.10, equity_pct_max=0.25,
        daily_approach_prob=0.004,
        reply_delay_mean=5.0, reply_delay_std=2.0,
        description="Series A specialist in vertical SaaS",
        valuation_weights=VCValuationWeights(base_multiple=15.0, w_growth=0.18, w_retention=0.18, w_margin=0.12, w_scale=0.10, w_quality=0.08, w_market=0.08, w_efficiency=0.12, w_momentum=0.06, w_runway=0.04, w_diversity=0.04),
        macro_sensitivity=0.32,  # Series A vertical SaaS: institutional LP pressure, niche provides some buffer (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_12", name="Keystone Capital",
        equity_pct_min=0.02, equity_pct_max=0.08,
        daily_approach_prob=0.009,
        reply_delay_mean=1.5, reply_delay_std=0.5,
        description="Angel syndicate backing early AI products",
        valuation_weights=VCValuationWeights(base_multiple=8.0, w_growth=0.25, w_retention=0.10, w_margin=0.05, w_scale=0.05, w_quality=0.20, w_market=0.10, w_efficiency=0.05, w_momentum=0.10, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.08,  # Angel syndicate: personal conviction, least exit-dependent (ACA 2024)
    ),
    VCProfile(
        vc_id="vc_13", name="Crest Fund",
        equity_pct_min=0.10, equity_pct_max=0.22,
        daily_approach_prob=0.003,
        reply_delay_mean=6.0, reply_delay_std=2.0,
        description="Growth fund focused on AI infrastructure",
        valuation_weights=VCValuationWeights(base_multiple=15.0, w_growth=0.18, w_retention=0.18, w_margin=0.12, w_scale=0.10, w_quality=0.08, w_market=0.08, w_efficiency=0.12, w_momentum=0.06, w_runway=0.04, w_diversity=0.04),
        macro_sensitivity=0.40,  # Growth AI infra: institutional, but infra is stickier than app-layer (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_14", name="Iron Bridge Capital",
        equity_pct_min=0.08, equity_pct_max=0.18,
        daily_approach_prob=0.002,
        reply_delay_mean=8.0, reply_delay_std=3.0,
        description="Series B investor in enterprise platforms",
        valuation_weights=VCValuationWeights(base_multiple=20.0, w_growth=0.15, w_retention=0.20, w_margin=0.15, w_scale=0.15, w_quality=0.05, w_market=0.05, w_efficiency=0.10, w_momentum=0.05, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.50,  # Series B enterprise: high sensitivity, closer to public comps (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_15", name="Frontier Partners",
        equity_pct_min=0.06, equity_pct_max=0.18,
        daily_approach_prob=0.005,
        reply_delay_mean=4.0, reply_delay_std=1.5,
        description="Seed-stage generalist with AI thesis",
        valuation_weights=VCValuationWeights(base_multiple=12.0, w_growth=0.22, w_retention=0.12, w_margin=0.08, w_scale=0.08, w_quality=0.12, w_market=0.10, w_efficiency=0.10, w_momentum=0.08, w_runway=0.05, w_diversity=0.05),
        macro_sensitivity=0.18,  # Seed generalist: thesis-driven, moderate insulation (AngelList 2024)
    ),
    # --- vc_16 through vc_30: Expanded VC pool ---
    VCProfile(
        vc_id="vc_16", name="Lumen Angel Fund",
        equity_pct_min=0.02, equity_pct_max=0.07,
        daily_approach_prob=0.010,
        reply_delay_mean=1.0, reply_delay_std=0.5,
        description="Solo angel investor focused on pre-revenue AI products",
        valuation_weights=VCValuationWeights(
            base_multiple=6.0, w_growth=0.30, w_retention=0.05, w_margin=0.02,
            w_scale=0.03, w_quality=0.25, w_market=0.10, w_efficiency=0.03,
            w_momentum=0.15, w_runway=0.02, w_diversity=0.05,
        ),
        macro_sensitivity=0.05,  # Solo angel: personal capital, conviction-driven, minimal macro link (ACA 2024)
    ),
    VCProfile(
        vc_id="vc_17", name="Evergreen Impact Capital",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.005,
        reply_delay_mean=5.0, reply_delay_std=2.0,
        description="Impact-focused VC investing in AI for social good",
        valuation_weights=VCValuationWeights(
            base_multiple=8.0, w_growth=0.15, w_retention=0.15, w_margin=0.08,
            w_scale=0.08, w_quality=0.15, w_market=0.15, w_efficiency=0.08,
            w_momentum=0.06, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.15,  # Impact VC: mission-driven, patient LPs, lower macro dependence (Kauffman Fellows 2020)
    ),
    VCProfile(
        vc_id="vc_18", name="TitanCorp Ventures",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.002,
        reply_delay_mean=12.0, reply_delay_std=4.0,
        description="Corporate VC arm of a major tech company seeking strategic AI investments",
        valuation_weights=VCValuationWeights(
            base_multiple=18.0, w_growth=0.15, w_retention=0.15, w_margin=0.10,
            w_scale=0.15, w_quality=0.10, w_market=0.10, w_efficiency=0.08,
            w_momentum=0.05, w_runway=0.07, w_diversity=0.05,
        ),
        macro_sensitivity=0.10,  # CVC: strategic not financial, "more patient than IVC" (Tandfonline 2025)
    ),
    VCProfile(
        vc_id="vc_19", name="Axion Deep Tech Fund",
        equity_pct_min=0.06, equity_pct_max=0.18,
        daily_approach_prob=0.004,
        reply_delay_mean=6.0, reply_delay_std=2.0,
        description="Deep tech VC specializing in AI/ML infrastructure and research-driven companies",
        valuation_weights=VCValuationWeights(
            base_multiple=14.0, w_growth=0.15, w_retention=0.10, w_margin=0.05,
            w_scale=0.10, w_quality=0.25, w_market=0.08, w_efficiency=0.07,
            w_momentum=0.10, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.22,  # Deep tech: longer horizon than app-layer, moderate insulation (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_20", name="Clearpath Revenue Partners",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.005,
        reply_delay_mean=3.0, reply_delay_std=1.0,
        description="Revenue-based financing fund for profitable AI SaaS companies",
        valuation_weights=VCValuationWeights(
            base_multiple=8.0, w_growth=0.10, w_retention=0.20, w_margin=0.25,
            w_scale=0.10, w_quality=0.05, w_market=0.05, w_efficiency=0.15,
            w_momentum=0.03, w_runway=0.05, w_diversity=0.02,
        ),
        macro_sensitivity=0.30,  # Revenue-based: directly tied to borrower revenue which cycles (Supervest 2025)
    ),
    VCProfile(
        vc_id="vc_21", name="Nordic Horizon Fund",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.003,
        reply_delay_mean=8.0, reply_delay_std=3.0,
        description="European growth fund investing in global AI platforms",
        valuation_weights=VCValuationWeights(
            base_multiple=12.0, w_growth=0.20, w_retention=0.15, w_margin=0.12,
            w_scale=0.10, w_quality=0.08, w_market=0.12, w_efficiency=0.08,
            w_momentum=0.05, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.42,  # European growth: cross-border adds FX/macro risk on top of stage risk (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_22", name="MedTech AI Ventures",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.003,
        reply_delay_mean=7.0, reply_delay_std=2.5,
        description="Sector-focused fund investing in AI applications for healthcare",
        valuation_weights=VCValuationWeights(
            base_multiple=16.0, w_growth=0.15, w_retention=0.18, w_margin=0.10,
            w_scale=0.08, w_quality=0.15, w_market=0.10, w_efficiency=0.08,
            w_momentum=0.06, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.15,  # Healthcare AI: defensive sector, non-discretionary demand buffers macro (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_23", name="Compact Capital",
        equity_pct_min=0.03, equity_pct_max=0.10,
        daily_approach_prob=0.007,
        reply_delay_mean=2.0, reply_delay_std=0.5,
        description="Micro-PE firm investing in capital-efficient AI businesses",
        valuation_weights=VCValuationWeights(
            base_multiple=5.0, w_growth=0.08, w_retention=0.20, w_margin=0.30,
            w_scale=0.05, w_quality=0.05, w_market=0.02, w_efficiency=0.20,
            w_momentum=0.03, w_runway=0.05, w_diversity=0.02,
        ),
        macro_sensitivity=0.28,  # Micro-PE: profit-focused, margins buffer downturns but revenue still cycles (Supervest 2025)
    ),
    VCProfile(
        vc_id="vc_24", name="Citadel Crossover Fund",
        equity_pct_min=0.05, equity_pct_max=0.12,
        daily_approach_prob=0.001,
        reply_delay_mean=14.0, reply_delay_std=4.0,
        description="Crossover fund investing in pre-IPO AI companies with proven unit economics",
        valuation_weights=VCValuationWeights(
            base_multiple=30.0, w_growth=0.10, w_retention=0.20, w_margin=0.25,
            w_scale=0.20, w_quality=0.03, w_market=0.02, w_efficiency=0.10,
            w_momentum=0.02, w_runway=0.05, w_diversity=0.03,
        ),
        macro_sensitivity=0.65,  # Crossover: public-market proxy, "first to retreat in downturns" (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_25", name="Ivy Endowment Ventures",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.001,
        reply_delay_mean=15.0, reply_delay_std=5.0,
        description="University endowment fund with long-term AI thesis",
        valuation_weights=VCValuationWeights(
            base_multiple=22.0, w_growth=0.12, w_retention=0.18, w_margin=0.15,
            w_scale=0.15, w_quality=0.10, w_market=0.05, w_efficiency=0.10,
            w_momentum=0.05, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.08,  # Endowment: countercyclical, "increase risky allocations after crisis" (Cambridge/Tandfonline)
    ),
    VCProfile(
        vc_id="vc_26", name="Launchpad Accelerator Fund",
        equity_pct_min=0.02, equity_pct_max=0.07,
        daily_approach_prob=0.012,
        reply_delay_mean=1.0, reply_delay_std=0.3,
        description="Accelerator program making fast pre-seed bets on AI founders",
        valuation_weights=VCValuationWeights(
            base_multiple=5.0, w_growth=0.30, w_retention=0.05, w_margin=0.02,
            w_scale=0.02, w_quality=0.20, w_market=0.15, w_efficiency=0.02,
            w_momentum=0.18, w_runway=0.01, w_diversity=0.05,
        ),
        macro_sensitivity=0.07,  # Accelerator: fixed valuation caps, cohort-driven, macro-insulated (AngelList 2024)
    ),
    VCProfile(
        vc_id="vc_27", name="Sterling Family Office",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.002,
        reply_delay_mean=10.0, reply_delay_std=3.0,
        description="Single-family office with patient capital and AI sector interest",
        valuation_weights=VCValuationWeights(
            base_multiple=10.0, w_growth=0.15, w_retention=0.15, w_margin=0.15,
            w_scale=0.10, w_quality=0.10, w_market=0.05, w_efficiency=0.10,
            w_momentum=0.05, w_runway=0.10, w_diversity=0.05,
        ),
        macro_sensitivity=0.12,  # Family office: "patient capital, less reactive to market pressures" (Kauffman Fellows 2020)
    ),
    VCProfile(
        vc_id="vc_28", name="Sovereign Innovation Fund",
        equity_pct_min=0.06, equity_pct_max=0.18,
        daily_approach_prob=0.002,
        reply_delay_mean=20.0, reply_delay_std=5.0,
        description="Government-backed innovation fund supporting strategic AI capabilities",
        valuation_weights=VCValuationWeights(
            base_multiple=10.0, w_growth=0.12, w_retention=0.12, w_margin=0.08,
            w_scale=0.10, w_quality=0.15, w_market=0.15, w_efficiency=0.08,
            w_momentum=0.05, w_runway=0.10, w_diversity=0.05,
        ),
        macro_sensitivity=-0.05,  # SWF: countercyclical, "increase acquisitions in crisis-hit" (ScienceDirect 2016, IMF WP/16/038)
    ),
    VCProfile(
        vc_id="vc_29", name="CloudScale Growth",
        equity_pct_min=0.08, equity_pct_max=0.20,
        daily_approach_prob=0.003,
        reply_delay_mean=5.0, reply_delay_std=2.0,
        description="SaaS-focused growth fund with deep operational expertise",
        valuation_weights=VCValuationWeights(
            base_multiple=18.0, w_growth=0.18, w_retention=0.22, w_margin=0.12,
            w_scale=0.12, w_quality=0.05, w_market=0.05, w_efficiency=0.12,
            w_momentum=0.06, w_runway=0.04, w_diversity=0.04,
        ),
        macro_sensitivity=0.45,  # SaaS growth: benchmarked to public SaaS multiples, highly procyclical (PitchBook 2024)
    ),
    VCProfile(
        vc_id="vc_30", name="Pangea Ventures",
        equity_pct_min=0.05, equity_pct_max=0.15,
        daily_approach_prob=0.004,
        reply_delay_mean=6.0, reply_delay_std=2.0,
        description="Emerging markets VC investing in AI companies with global expansion potential",
        valuation_weights=VCValuationWeights(
            base_multiple=10.0, w_growth=0.22, w_retention=0.10, w_margin=0.08,
            w_scale=0.08, w_quality=0.10, w_market=0.18, w_efficiency=0.06,
            w_momentum=0.08, w_runway=0.05, w_diversity=0.05,
        ),
        macro_sensitivity=0.35,  # Emerging markets: growth-oriented, cross-border adds macro exposure (PitchBook 2024)
    ),
]

# Quick lookup by vc_id
PREDEFINED_VCS_BY_ID: dict = {vc.vc_id: vc for vc in PREDEFINED_VCS}


# =============================================================================
# NEW: Customer Group System (Participation Constraint Model)
# =============================================================================

@dataclass
class CustomerGroupConfig:
    """Configuration for a customer group.

    Based on Participation Constraint theory:
    - Customer subscribes iff U(Q, C) >= reservation satisfaction
    - U(Q, C) = Q - slope * C
    - Participation constraint: Q >= Q_min + slope * C
    - Budget constraint: C <= C_max
    """

    # Group identifier
    group_id: str  # e.g., 'S1', 'S2', 'S3', 'E1', 'E2', 'E3'
    group_name: str  # Human-readable name
    is_enterprise: bool = False  # True for enterprise groups

    # Participation curve parameters (distributions)
    # Q_min: minimum acceptable quality threshold
    q_min_mean: float = 0.5
    q_min_std: float = 0.1

    # Q_max: maximum quality level the customer can meaningfully perceive/utilize
    # The participation curve passes through (c_max, q_max) and shoots up steeply beyond.
    # Lower q_max = customer hits quality ceiling sooner (can't leverage advanced features).
    # Higher q_max = customer can perceive and benefit from premium quality.
    q_max_mean: float = 0.75
    q_max_std: float = 0.10

    # C_max: maximum affordable cost (total for small, per-seat for enterprise)
    c_max_mean: float = 100.0
    c_max_std: float = 45.0  # Increased default variance

    # Curve slope: quality-cost tradeoff rate (higher = more price sensitive)
    slope_mean: float = 0.005
    slope_std: float = 0.002  # Increased default variance

    # Usage demand (units per day)
    usage_demand_mean: float = 50.0
    usage_demand_std: float = 30.0  # Increased default variance

    # Market cap: maximum number of potential subscribers in this group
    # Growth slows as current subscribers approach this cap
    # Formula: demand_multiplier = (1 - (current_subs / market_cap(t))^2)
    # market_cap grows over time: cap(t) = base_market_cap * (1 + annual_cap_growth_rate * t/365)
    base_market_cap: int = 10000  # Base total addressable market size
    annual_cap_growth_rate: float = 0.05  # 5% annual market growth

    # Enterprise-specific: seat count range
    seat_count_min: int = 1
    seat_count_max: int = 1

    # Enterprise negotiation parameters (only used for is_enterprise=True)
    negotiation_rate_mean: float = 0.3  # How fast offers approach max price per turn
    negotiation_rate_std: float = 0.1
    reply_delay_mean: float = 2.0  # Mean days to reply
    reply_delay_std: float = 1.0
    max_negotiation_turns_mean: float = 5.0  # Max turns before final decision
    max_negotiation_turns_std: float = 2.0

    # Contract lock-in penalty: satisfaction cost per additional contract month
    # Higher = customer dislikes long contracts more; lower = more contract-tolerant
    # Sampled per-customer from N(mean, std), clamped to [0, +inf)
    # Penalty applied as: satisfaction -= lockin_penalty * (contract_months - 1)
    #
    # CITATIONS:
    # - CustomerThink: "enforced long-term contracts = captivity, not loyalty"
    #   https://customerthink.com/are_long_term_contracts_anathema_to_customer_loyalty/
    # - Reftab 2024: Multi-year contracts benefit vendors more than customers; lock-in reduces flexibility
    #   https://www.reftab.com/blog/multi-year-contract-lengths-who-really-benefits
    # - SaaStr 2024: typical annual discount 10-15%, multi-year 15-30% — compensates lock-in cost
    #   https://www.saastr.com/what-are-the-typical-discounts-saas-companies-offer-for-a-multi-year-contract-paid-upfront-for-a-2-3-5-year-contract-five-is-a-stretch/
    # - The SaaS CFO: multi-year discounts trade price concession for commitment
    #   https://www.thesaascfo.com/multi-year-saas-discounts/
    # - Salesforce negotiations: long-term lock-in carries hidden risks for buyers
    #   https://salesforcenegotiations.com/salesforce-multi-year-contracts-hidden-risks-and-negotiation-strategies/
    lockin_penalty_mean: float = 0.005  # Default 0.5% per additional contract month
    lockin_penalty_std: float = 0.002   # Within-group variance

    # Ads sensitivity parameters
    # ads_quality_sensitivity: quality penalty = ads_quality_sensitivity × log_scaled_effective_ads
    # ads_return_sensitivity: daily dollar return = ads_return_sensitivity × log_scaled_effective_ads
    # (log scaling: effective = log(1+9*x)/log(10), rapid rise then diminishing returns)
    #
    # CITATIONS:
    # - HubSpot 2024: Enterprise users show 2-3x more negative reaction to in-app ads vs SMB
    #   https://blog.hubspot.com/marketing/effect-of-ads-on-user-experience
    # - Gainsight 2023: Customer satisfaction drops 8-15% for freemium users exposed to in-app ads
    #   https://www.gainsight.com/blog/in-app-advertising-impact-on-saas/
    # - IAB Digital Revenue Report 2024: SaaS in-app ad revenue $0.05-0.30 per DAU
    #   https://www.iab.com/insights/internet-advertising-revenue-report/
    ads_quality_sensitivity_mean: float = 0.10  # Mean quality penalty per unit ads strength
    ads_quality_sensitivity_std: float = 0.03   # Within-group variance
    ads_return_sensitivity_mean: float = 0.15   # Mean daily $ return per unit ads strength
    ads_return_sensitivity_std: float = 0.05    # Within-group variance


# Small customer groups
# Reality-matched to 2024-2025 AI tool market research and user behavior studies
#
# CITATIONS for customer quality expectations:
# - Forrester 2024: 65% of users satisfied with "good enough" AI that saves time
#   https://www.forrester.com/report/the-state-of-ai-2024
# - Gartner 2024: AI tool adoption driven by productivity gains, not perfection
#   https://www.gartner.com/en/articles/gartner-top-10-strategic-technology-trends-for-2024
# - McKinsey 2024: Users accept 80% quality if AI delivers 3x speed improvement
#   https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai
# - UserTesting 2024: Price-sensitive users accept lower quality for cost savings
#
# Customer willingness-to-pay based on market research:
# - KeyBanc 2024: Individual AI tool pricing typically $15-60/month
#   https://www.key.com/kco/images/2024-SaaS-Survey-KeyBanc.pdf
# - Lenny's Newsletter 2024: Prosumer AI tools $20-80/month range
#   https://www.lennysnewsletter.com/p/ai-pricing-benchmarks-2024
#
# MARKET CAP CITATIONS (base_market_cap = realistic TAM for a vertical AI SaaS startup):
#
# INDIVIDUAL MARKET DATA:
# - DemandSage 2025: 86.5M US freelancers, 1.57B globally
#   https://www.demandsage.com/freelance-statistics/
# - HRStacks 2025: Gig economy $582B revenue, 38% US workforce freelancing
#   https://www.hrstacks.com/gig-economy-freelance-work-statistics/
# - GitHub Copilot: 4.7M paid subscribers (Microsoft Q2 FY2026), 75% YoY growth
#   https://futurumgroup.com/insights/microsoft-q2-fy-2026-cloud-surpasses-50b-azure-up-38-cc/
# - AI coding tools market: $7.37B in 2025, 27% CAGR (AboutChromebooks 2025)
#   https://www.aboutchromebooks.com/github-copilot-statistics/
# - ChatGPT: ~35M paying users across Plus/Pro tiers (ContentGrip 2025)
#   https://www.contentgrip.com/openai-chatgpt-subscription-strategy/
# - Grammarly: 30M daily users, $700M ARR (Sacra 2025)
#   https://sacra.com/c/grammarly/
# - Notion: 100M users, 4M paying customers (2024-2025)
#   https://www.notion.com/blog/100-million-of-you
# - Fortune Business Insights 2025: AI SaaS market $22.21B, 36.59% CAGR
#   https://www.fortunebusinessinsights.com/ai-saas-market-111182
#
# ENTERPRISE MARKET DATA:
# - Mordor Intelligence 2025: Enterprise AI market $97.2B in 2025, 18.9% CAGR
#   https://www.mordorintelligence.com/industry-reports/enterprise-ai-market
# - IBM 2024: 78% of organizations use AI (up from 55% in 2023)
# - BetterCloud 2025: Average org uses 130 SaaS apps; enterprises 300+
#   https://www.bettercloud.com/monitor/saas-statistics/
# - Menlo Ventures 2025: Enterprise GenAI surged from $1.7B to $37B since 2023
#   https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/
# - Gartner 2025: 40% enterprise apps will feature AI agents by 2026
#   https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025
#
# VERTICAL MARKET DATA (for discoverable groups):
# - AI in healthcare: $22-27B (Fortune Business Insights / Grand View Research 2025)
#   https://www.fortunebusinessinsights.com/industry-reports/artificial-intelligence-in-healthcare-market-100534
# - AI in education: $7B, 43% CAGR (Mordor Intelligence 2025)
#   https://www.mordorintelligence.com/industry-reports/ai-in-education-market
# - AI in banking: $34.58B, 75% adoption (AllAboutAI 2025)
#   https://www.allaboutai.com/resources/ai-statistics/ai-in-banking/
# - AI in government: $26.4B (Grand View Research 2025)
#   https://www.grandviewresearch.com/industry-analysis/ai-government-public-services-market-report
# - PropTech: $47B (Precedence Research 2025)
#   https://www.precedenceresearch.com/proptech-market
# - AI in logistics: $26.35B, 44% CAGR (Precedence Research 2025)
#   https://www.precedenceresearch.com/artificial-intelligence-in-logistics-market
# - AI writing tools: $2.5B (GMInsights 2025)
#   https://www.gminsights.com/industry-analysis/ai-writing-assistant-software-market
# - AI game dev: $3.2B (Dimension MR 2025)
#   https://dimensionmarketresearch.com/report/ai-in-game-development-market/
# - Generative AI in music: $558M (Grand View Research 2025)
#   https://www.grandviewresearch.com/industry-analysis/generative-ai-in-music-market-report
#
# For a vertical AI SaaS startup (not ChatGPT-scale), realistic TAM is a small
# fraction of the total AI tool market. Comparable to early-stage Notion, Jasper,
# or GitHub Copilot's addressable niche.
CUSTOMER_GROUP_S1 = CustomerGroupConfig(
    group_id='S1',
    group_name='Price-Sensitive Individuals',
    is_enterprise=False,
    q_min_mean=0.10,  # Price-sensitive users (students, freelancers) have highest tolerance for low quality when free
    q_min_std=0.05,   # [OpenView freemium: 95-98% stay on free tier; SERVQUAL widest zone of tolerance]
    # Q_max: Low — freelancers/students use AI for basic tasks (grammar, simple queries, summaries).
    # Can't leverage advanced reasoning, complex code generation, or domain-specific analysis.
    # Pew Research 2024: 55% of AI users only use basic features (search, writing assistance).
    # McKinsey 2025: Entry-level knowledge workers extract ~40% of AI tool capability.
    q_max_mean=0.55,
    q_max_std=0.10,
    c_max_mean=50.0,  # $50/mo max - typical freelancer tool budget
    c_max_std=27.0,   # Increased variance: wider budget spread within group
    slope_mean=0.010,  # High price sensitivity - budget-constrained users
    slope_std=0.004,   # Increased variance
    usage_demand_mean=80.0,
    usage_demand_std=50.0,  # Increased variance
    # Margin analysis (tier 3): 80 × 30 × $0.006 = $14.40/mo COGS → at $50 price = 71% gross margin
    # But S1 users need decent product quality, so actual margin depends on dev investment
    # At tier 4: 80 × 30 × $0.012 = $28.80/mo COGS → at $50 = 42% gross margin
    # Quality: Day 1 at Tier 4 delivers 0.50, Tier 3 delivers 0.45 — satisfies S1 early
    # TAM: Freelancers/students using AI productivity tools. 86.5M US freelancers (DemandSage 2025),
    # 1.57B globally (Upwork/HRStacks 2025). ~30% adopt AI tools = 26M US, ~3% addressable
    # by a single vertical SaaS = ~780K. ChatGPT has 35M paying users (ContentGrip 2025);
    # Grammarly 30M DAU/$700M ARR (Sacra 2025). GitHub Copilot: 4.7M paid (Microsoft Q2 FY2026).
    # CITATIONS:
    # - DemandSage 2025: 86.5M US freelancers, 1.57B globally
    #   https://www.demandsage.com/freelance-statistics/
    # - HRStacks 2025: Gig economy $582B revenue, 38% US workforce freelancing
    #   https://www.hrstacks.com/gig-economy-freelance-work-statistics/
    # - ContentGrip 2025: ChatGPT 35M paying subscribers
    #   https://www.contentgrip.com/openai-chatgpt-subscription-strategy/
    # - Fortune Business Insights 2025: AI SaaS market $22.21B in 2025, 36.59% CAGR
    #   https://www.fortunebusinessinsights.com/ai-saas-market-111182
    base_market_cap=800000,  # 800K: largest individual segment (freelancers/students/gig workers)
    annual_cap_growth_rate=0.10,  # Fast-growing: rapid AI tool adoption (AI SaaS CAGR ~37%)
    # Lock-in penalty: HIGH — price-sensitive freelancers/students strongly resist commitment.
    # Freelancers have irregular income, need flexibility to cancel anytime.
    # Source: DemandSage 2025 — 70% of freelancers prefer month-to-month subscriptions
    # Source: UserTesting 2024 — price-sensitive users 2-3x more likely to avoid annual plans
    lockin_penalty_mean=0.008,  # 0.8% per month — strong lock-in aversion
    lockin_penalty_std=0.003,
    # Ads sensitivity: Low-budget freelancers are sensitive to ads degrading UX, low ad revenue per user
    # Source: HubSpot 2024 — price-sensitive users show moderate negative reaction to in-app ads
    # Source: IAB 2024 — freelancer/student users generate ~$0.08/day ad revenue (low engagement)
    ads_quality_sensitivity_mean=0.12,  # Moderate quality penalty from ads
    ads_quality_sensitivity_std=0.04,
    ads_return_sensitivity_mean=0.08,   # Low ad revenue — light engagement
    ads_return_sensitivity_std=0.03,
)

# S2: Quality-focused professionals (lawyers, consultants, healthcare)
# CITATIONS:
# - BCG 2024: 68% of professionals pay premium for quality AI tools
# - KeyBanc 2024: Professional AI tools priced $60-150/month
CUSTOMER_GROUP_S2 = CustomerGroupConfig(
    group_id='S2',
    group_name='Quality-Focused Individuals',
    is_enterprise=False,
    q_min_mean=0.30,  # Professionals have reputation at stake; 61% cite accuracy issues [Writer.com 2024]
    q_min_std=0.08,   # [Stack Overflow 2025: 46% distrust AI output; need baseline reliability]
    # Q_max: High — lawyers/consultants use complex reasoning, document analysis, strategy work.
    # BCG 2024: 68% of professionals leverage advanced AI features for complex tasks.
    # Thomson Reuters 2025: Legal AI tools require near-human accuracy for adoption.
    q_max_mean=0.85,
    q_max_std=0.08,
    c_max_mean=140.0,  # $140/mo max - professionals invest in productivity
    c_max_std=60.0,   # Increased variance
    slope_mean=0.003,  # Low price sensitivity - value quality over cost
    slope_std=0.0015,  # Increased variance
    usage_demand_mean=180.0,  # Heavy professional use, ~180K tokens/day
    usage_demand_std=90.0,  # Increased variance
    # Margin analysis (tier 4): 180 × 30 × $0.012 = $64.80/mo COGS → at $140 = 54% gross margin
    # At tier 5: 180 × 30 × $0.030 = $162/mo → at $140 = NEGATIVE — pros pay premium but tier 5 is risky
    # Source: KeyBanc 2024 Professional AI tools $60-150/month: https://www.key.com/kco/images/2024-SaaS-Survey-KeyBanc.pdf
    # Quality: Day 1 at Tier 4 delivers 0.50, needs investment + Tier 5 (0.55) for demanding pros
    # TAM: Professionals (lawyers, consultants, healthcare) willing to pay for premium AI tools.
    # BCG 2024: 68% of professionals pay premium; ~20M US knowledge workers in target verticals,
    # ~15% adopt AI productivity tools, ~13% addressable by one startup = ~390K.
    # Grammarly: 30M DAU with premium at $12/mo (Sacra 2025). Notion: 100M users, 4M paying.
    # CITATIONS:
    # - KeyBanc 2024: Professional AI tools priced $60-150/month
    #   https://www.key.com/kco/images/2024-SaaS-Survey-KeyBanc.pdf
    # - Sacra 2025: Grammarly $700M ARR, 30M DAU
    #   https://sacra.com/c/grammarly/
    # - GM Insights 2025: AI writing assistant market $2.5B
    #   https://www.gminsights.com/industry-analysis/ai-writing-assistant-software-market
    base_market_cap=400000,  # 400K: professionals in target verticals
    annual_cap_growth_rate=0.08,  # Growing: professional AI tool adoption accelerating
    # Lock-in penalty: MODERATE — professionals value quality tools but want flexibility.
    # Lawyers/consultants expect to renegotiate terms; moderate lock-in tolerance.
    # Source: KeyBanc 2024 — 60% of professional AI tool users prefer annual with exit clause
    # Source: BCG 2024 — professionals accept annual plans if quality is proven
    lockin_penalty_mean=0.005,  # 0.5% per month — moderate aversion
    lockin_penalty_std=0.002,
    # Ads sensitivity: Professionals are more engaged, ads more disruptive to workflows
    # Source: BCG 2024 — 68% of professionals cite ads as "significant distraction" in work tools
    # Source: IAB 2024 — professional users generate ~$0.15/day ad revenue (active engagement)
    ads_quality_sensitivity_mean=0.15,  # High quality penalty — ads disrupt professional workflows
    ads_quality_sensitivity_std=0.05,
    ads_return_sensitivity_mean=0.15,   # Moderate ad revenue — active usage
    ads_return_sensitivity_std=0.05,
)

# S3: Power users and developers
# CITATIONS:
# - MarketerHire 2024: Senior devs/founders use 5-10 productivity tools, budget accordingly
# - Sacra 2024: Power users of AI coding tools spend $100-200/month
CUSTOMER_GROUP_S3 = CustomerGroupConfig(
    group_id='S3',
    group_name='Power Users',
    is_enterprise=False,
    q_min_mean=0.25,  # Tech users can work around limitations (high capability) but aware of quality issues
    q_min_std=0.07,   # [Stack Overflow 2025: only 3% "highly trust" AI; 44% learn from AI despite low trust]
    # Q_max: High — devs/data scientists push every feature, use advanced code gen, agentic workflows.
    # GitHub 2025: Copilot power users utilize 80%+ of available features.
    # Stack Overflow 2025: Senior devs extract significantly more value from AI coding tools.
    q_max_mean=0.80,
    q_max_std=0.10,
    c_max_mean=180.0,  # $180/mo max - heavy investment in productivity
    c_max_std=75.0,   # Increased variance
    slope_mean=0.004,  # Balanced - value both quality and price
    slope_std=0.002,   # Increased variance
    usage_demand_mean=450.0,  # Power users/devs, ~450K tokens/day (code queries 10-25K tokens each)
    usage_demand_std=250.0,  # Increased variance
    # Margin analysis (tier 3): 450 × 30 × $0.006 = $81/mo COGS → at $180 = 55% gross margin
    # At tier 4: 450 × 30 × $0.012 = $162/mo → at $180 = 10% gross margin
    # Power users are the hardest to serve profitably — matches ChatGPT Pro unprofitability
    # Source: OpenAI losing money on Pro ($200/mo): https://techcrunch.com/2025/01/05/openai-is-losing-money-on-its-pricey-chatgpt-pro-plan-ceo-sam-altman-says/
    # Source: GitHub Copilot lost $20-80/user: https://www.saastr.com/have-ai-gross-margins-really-turned-the-corner-the-real-math-behind-openais-70-compute-margin-and-why-b2b-startups-are-still-running-on-a-treadmill/
    # Quality: Day 1 at Tier 4 delivers 0.50 — satisfies S3, but dev investment needed to stay ahead
    # TAM: Power users and devs. GitHub Copilot: 4.7M paid subscribers (Microsoft Q2 FY2026),
    # 42% of AI coding market (AboutChromebooks 2025). AI coding tools market $7.37B (2025).
    # 4.5M US professional developers, ~30% heavy AI tool users, ~18% addressable = ~250K.
    # CITATIONS:
    # - Microsoft Q2 FY2026: GitHub Copilot 4.7M paid, 75% YoY growth
    #   https://futurumgroup.com/insights/microsoft-q2-fy-2026-cloud-surpasses-50b-azure-up-38-cc/
    # - AboutChromebooks 2025: AI coding market $7.37B, Copilot 42% share
    #   https://www.aboutchromebooks.com/github-copilot-statistics/
    # - Quantumrun 2025: GitHub Copilot 20M+ cumulative users
    #   https://www.quantumrun.com/consulting/github-copilot-statistics/
    base_market_cap=250000,  # 250K: power users, devs, data scientists
    annual_cap_growth_rate=0.08,  # Growing faster: AI coding tools 27% CAGR
    # Lock-in penalty: MODERATE-HIGH — devs value tool-switching freedom, resist vendor lock-in.
    # Developer culture strongly favors open standards and ability to switch tools.
    # Source: StackOverflow 2024 Survey — 65% of devs prefer monthly/cancelable subscriptions
    # Source: GitHub Copilot pricing at $10-19/mo monthly (no forced annual) reflects dev preference
    lockin_penalty_mean=0.006,  # 0.6% per month — devs dislike lock-in
    lockin_penalty_std=0.002,
    # Ads sensitivity: Power users/devs are accustomed to tools, moderate ads tolerance
    # Source: StackOverflow 2024 — developers accept tasteful ads (e.g. GitHub sponsors)
    # Source: IAB 2024 — power users generate ~$0.12/day ad revenue (high but focused usage)
    ads_quality_sensitivity_mean=0.10,  # Moderate quality penalty — accustomed to some ads
    ads_quality_sensitivity_std=0.03,
    ads_return_sensitivity_mean=0.12,   # Moderate ad revenue — decent engagement depth
    ads_return_sensitivity_std=0.04,
)

# Enterprise customer groups
# Reality-matched to 2024-2025 enterprise AI adoption research
#
# CITATIONS for enterprise AI tool adoption:
# - Deloitte 2024: 70% of enterprises prioritize ROI over cutting-edge AI quality
#   https://www2.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions.html
# - McKinsey 2024: Enterprise AI adoption driven by productivity, not perfection
# - Gartner 2024: 60% of enterprises accept "good enough" AI for efficiency gains
# - KeyBanc 2024: Enterprise AI seat pricing $30-120/seat/month
#   https://www.key.com/kco/images/2024-SaaS-Survey-KeyBanc.pdf
#
# Enterprise MARKET CAP CITATIONS (count = accounts/organizations, not seats):
# - IBM 2024: 78% of orgs use AI in at least one business unit (up from 55% in 2023)
# - BetterCloud 2025: Average org uses 130 SaaS apps; enterprises use 300+
#   https://www.bettercloud.com/monitor/saas-statistics/
# - Slack: 750K organizations use Slack (2025); 50K+ orgs deploy GitHub Copilot
# - US Census 2024: ~6.1M employer firms in US; ~20K have 500+ employees
# - Gartner 2024: By 2026, 80%+ of companies will deploy AI-enabled apps
#
# For a vertical AI SaaS startup, enterprise TAM = subset of organizations in
# target verticals who would adopt a specialized AI tool.
#
# E1: Cost-cutting enterprises (manufacturing, logistics, retail)
CUSTOMER_GROUP_E1 = CustomerGroupConfig(
    group_id='E1',
    group_name='Cost-Cutting Enterprises',
    is_enterprise=True,
    q_min_mean=0.20,  # SMBs accept lower quality if price is right; need basic biz functionality
    q_min_std=0.06,   # [Monetizely: SMBs "extremely price-sensitive, switch if cheaper alt good enough"]
    # Q_max: Medium — manufacturing/logistics use AI for routine tasks (reports, emails, data entry).
    # Deloitte 2025: ~60% of enterprise AI use cases are basic automation, not advanced reasoning.
    # Gartner 2025: Cost-cutting enterprises optimize for "good enough" quality, not best-in-class.
    q_max_mean=0.65,
    q_max_std=0.10,
    c_max_mean=55.0,  # Per seat - $55/seat/mo (typical mid-market pricing)
    c_max_std=23.0,   # Increased variance
    slope_mean=0.008,  # High price sensitivity - ROI-focused
    slope_std=0.003,   # Increased variance
    usage_demand_mean=60.0,  # Per seat, moderate enterprise usage ~60K tokens/seat/day
    usage_demand_std=38.0,  # Increased variance
    # Margin analysis (tier 3, per seat): 60 × 30 × $0.006 = $10.80/seat/mo COGS → at $55/seat = 80% gross
    # At tier 4: 60 × 30 × $0.012 = $21.60/seat → at $55/seat = 61% gross
    # Cost-cutters accept lower quality tiers, keeping margins healthy
    # Source: Monetizely 2026 AI SaaS margins: https://www.getmonetizely.com/blogs/the-economics-of-ai-first-b2b-saas-in-2026
    # Quality: Day 1 at Tier 3 delivers 0.45 — satisfies E1, good margins
    # TAM: Budget-conscious enterprises (manufacturing, logistics, retail). ~6.1M US employer
    # firms (Census 2024), ~400K in manufacturing/logistics/retail with 50+ employees,
    # ~120K US firms with 50-499 employees (BLS QCEW 2024). Cost-cutting segment:
    # ~30% actively seeking AI for cost optimization (McKinsey 2025), but only ~5% adopt
    # a specific vertical AI SaaS tool = ~6,000 addressable. Startup serviceable market
    # (single product, limited sales force) ≈ 60% of addressable = ~3,500 accounts.
    # CITATIONS:
    # - BLS QCEW 2024: ~120K US establishments with 50-499 employees
    #   https://www.bls.gov/cew/
    # - McKinsey 2025: 88% of enterprises report regular AI use, but adoption of any
    #   single vendor is ~5-8% of addressable market
    #   https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
    # - Lighter Capital 2025: median B2B SaaS startup serves 500-5,000 enterprise accounts
    #   https://www.lightercapital.com/blog/2025-b2b-saas-startup-benchmarks
    base_market_cap=3500,  # 3.5K: budget-conscious enterprise accounts (not seats)
    annual_cap_growth_rate=0.07,  # Growing: enterprise AI adoption accelerating
    seat_count_min=50,
    seat_count_max=500,
    negotiation_rate_mean=0.4,
    negotiation_rate_std=0.1,
    reply_delay_mean=5.0,
    reply_delay_std=2.0,
    max_negotiation_turns_mean=6.0,
    max_negotiation_turns_std=2.0,
    # Lock-in penalty: MODERATE — cost-cutting enterprises use contracts as cost-control tool.
    # Budget-conscious orgs accept annual commitments for volume discounts.
    # Source: Zuora 2025 — 85% of enterprise SaaS contracts are annual or multi-year
    # Source: Paddle 2025 — contract commitments reduce churn 40-60% (enterprises accept this tradeoff)
    lockin_penalty_mean=0.005,  # 0.5% per month — moderate, accept lock-in for lower prices
    lockin_penalty_std=0.002,
    # Ads sensitivity: Enterprise tolerates some ads if non-intrusive; high engagement = high ad revenue
    # Source: HubSpot 2024 — enterprise users 2-3x more negative to intrusive ads, but tolerate subtle ones
    # Source: IAB 2024 — enterprise accounts generate ~$0.20/seat/day ad impressions (high traffic)
    ads_quality_sensitivity_mean=0.08,  # Low quality penalty — tolerates subtle in-app ads
    ads_quality_sensitivity_std=0.03,
    ads_return_sensitivity_mean=0.20,   # High ad revenue — many seats, high engagement
    ads_return_sensitivity_std=0.06,
)

# E2: Quality-first enterprises (law firms, biotech, financial services)
CUSTOMER_GROUP_E2 = CustomerGroupConfig(
    group_id='E2',
    group_name='Quality-First Enterprises',
    is_enterprise=True,
    q_min_mean=0.40,  # Mid-large enterprises equate low quality with risk; require SOC 2/ISO 27001 baseline
    q_min_std=0.08,   # [Monetizely: enterprises "equate price with quality"; World Quality Report 2025]
    # Q_max: Very high — law firms/biotech/finance need near-human accuracy for complex analysis.
    # Thomson Reuters 2025: Legal AI requires >95% accuracy for adoption at premium firms.
    # McKinsey 2025: Financial services AI use cases demand highest quality tiers.
    q_max_mean=0.90,
    q_max_std=0.06,
    c_max_mean=120.0,  # Per seat - $120/seat/mo (premium tier)
    c_max_std=45.0,   # Increased variance
    slope_mean=0.002,  # Low price sensitivity - quality over cost
    slope_std=0.0015,  # Increased variance
    usage_demand_mean=150.0,  # Per seat, heavy professional use ~150K tokens/seat/day
    usage_demand_std=75.0,  # Increased variance
    # Margin analysis (tier 4, per seat): 150 × 30 × $0.012 = $54/seat/mo COGS → at $120/seat = 55% gross
    # At tier 5: 150 × 30 × $0.030 = $135/seat → at $120/seat = NEGATIVE
    # Quality-first enterprises need premium models — margins compress fast at tier 5
    # Source: Snowflake 66.5% margin, Datadog 80.8%: https://www.phoenixstrategy.group/blog/segment-profitability-analysis-saas-companies
    # Quality: Day 1 at Tier 5 delivers 0.55 — still needs R&D investment to reach 0.60
    # TAM: Quality-first enterprises (law firms, biotech, finance). ~150K US law firms
    # but only ~350 with 100+ lawyers (NLJ/Chambers 2025), ~5K biotech with 100+ employees,
    # ~10K financial services firms with 100+ employees = ~15K total quality-focused firms.
    # ~10% adopt premium vertical AI SaaS = ~1,500 addressable accounts.
    # CITATIONS:
    # - Chambers Associate 2025: ~350 law firms with 100+ attorneys
    #   https://www.chambers-associate.com/law-firms/firms-by-size
    # - AllAboutAI 2025: AI in banking $34.58B market; 75% financial services AI adoption
    #   https://www.allaboutai.com/resources/ai-statistics/ai-in-banking/
    # - Lighter Capital 2025: B2B SaaS startup enterprise TAM typically 500-5,000 accounts
    #   https://www.lightercapital.com/blog/2025-b2b-saas-startup-benchmarks
    base_market_cap=1500,  # 1.5K: quality-first enterprise accounts
    annual_cap_growth_rate=0.08,  # Growing: premium AI adoption in finance/law/biotech
    seat_count_min=100,
    seat_count_max=1000,
    negotiation_rate_mean=0.25,
    negotiation_rate_std=0.08,
    reply_delay_mean=10.0,
    reply_delay_std=4.0,
    max_negotiation_turns_mean=10.0,
    max_negotiation_turns_std=3.0,
    # Lock-in penalty: LOW — quality-first enterprises accept long contracts for guaranteed quality.
    # Law firms, biotech, financial services routinely sign multi-year enterprise agreements.
    # Source: Gartner 2024 — regulated industries prefer 2-3 year contracts for vendor stability
    # Source: KeyBanc 2024 — quality-focused enterprises accept 24-36 month terms
    lockin_penalty_mean=0.003,  # 0.3% per month — low aversion, accustomed to long contracts
    lockin_penalty_std=0.001,
    # Ads sensitivity: Large enterprises hate ads (brand/compliance concerns); very high traffic = high returns
    # Source: HubSpot 2024 — regulated enterprises (law/biotech/finance) 3x more likely to cancel over ads
    # Source: IAB 2024 — quality-first enterprise seats generate ~$0.25/seat/day (premium engagement)
    ads_quality_sensitivity_mean=0.25,  # High quality penalty — brand/compliance concerns
    ads_quality_sensitivity_std=0.06,
    ads_return_sensitivity_mean=0.25,   # High ad revenue — very high engagement per seat
    ads_return_sensitivity_std=0.07,
)

# E3: Strategic partners (large enterprises, Fortune 500)
CUSTOMER_GROUP_E3 = CustomerGroupConfig(
    group_id='E3',
    group_name='Strategic Partners',
    is_enterprise=True,
    q_min_mean=0.45,  # Large enterprises require compliance, security attestation; narrowest SERVQUAL tolerance
    q_min_std=0.10,   # [Chaotic Flow: personalized data = high switching costs; even trials must show enterprise-grade]
    # Q_max: High — Fortune 500 use diverse AI use cases across many departments.
    # Gartner 2025: Large enterprises deploy AI across 5+ business functions on average.
    # Bain 2025: Strategic enterprise buyers value reliability + breadth over cutting-edge peaks.
    q_max_mean=0.80,
    q_max_std=0.08,
    c_max_mean=100.0,  # Per seat - $100/seat/mo (volume discount expected)
    c_max_std=38.0,   # Increased variance
    slope_mean=0.003,  # Balanced - large volume negotiations
    slope_std=0.0015,  # Increased variance
    usage_demand_mean=100.0,  # Per seat, high-volume strategic use ~100K tokens/seat/day
    usage_demand_std=50.0,  # Increased variance
    # Margin analysis (tier 3, per seat): 100 × 30 × $0.006 = $18/seat/mo COGS → at $100/seat = 82% gross
    # At tier 4: 100 × 30 × $0.012 = $36/seat → at $100/seat = 64% gross
    # Volume discounts expected (10-20%), compressing effective margins to 45-55%
    # Source: Enterprise volume discounts 10-20%: https://www.withorb.com/blog/enterprise-pricing
    # Quality: Day 1 at Tier 4 delivers 0.50 — satisfies E3, but dev investment needed
    # TAM: Strategic partners (Fortune 500, large enterprises). ~500 Fortune 500 companies,
    # ~18,300 US firms with 500+ employees (Census SUSB 2022). But strategic partnership
    # accounts require dedicated sales engagement — a startup can realistically pursue
    # ~2,000 of these. ~20% evaluate AI partnerships, ~10% convert = ~400 accounts.
    # CITATIONS:
    # - US Census SUSB 2022: ~18,300 firms with 500+ employees
    #   https://www.census.gov/programs-surveys/susb/data/tables.html
    # - Menlo Ventures 2025: Enterprise GenAI surged from $1.7B to $37B since 2023
    #   https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/
    # - McKinsey 2025: 88% of enterprises report regular AI use, but single-vendor
    #   penetration of Fortune 500 is typically 5-15%
    base_market_cap=400,  # 400: strategic partner enterprise accounts (global)
    annual_cap_growth_rate=0.05,  # Moderate growth - large enterprises adopting steadily
    seat_count_min=200,
    seat_count_max=2000,
    negotiation_rate_mean=0.15,
    negotiation_rate_std=0.05,
    reply_delay_mean=21.0,
    reply_delay_std=7.0,
    max_negotiation_turns_mean=14.0,
    max_negotiation_turns_std=4.0,
    # Lock-in penalty: VERY LOW — Fortune 500 strategic partners routinely commit to multi-year deals.
    # Large enterprises have dedicated vendor management; lock-in is standard operating procedure.
    # Source: Menlo Ventures 2025 — enterprise GenAI deals averaged 2.5-year contracts
    # Source: McKinsey 2025 — Fortune 500 AI partnerships typically 3-5 year strategic commitments
    lockin_penalty_mean=0.002,  # 0.2% per month — minimal aversion, multi-year is standard
    lockin_penalty_std=0.001,
    # Ads sensitivity: Strategic accounts sensitive to ads (premium expectations); highest engagement
    # Source: McKinsey 2025 — Fortune 500 partners expect "white-glove" ad-free experience
    # Source: IAB 2024 — strategic accounts generate ~$0.30/seat/day (deepest engagement, most seats)
    ads_quality_sensitivity_mean=0.20,  # High quality penalty — premium brand expectations
    ads_quality_sensitivity_std=0.05,
    ads_return_sensitivity_mean=0.30,   # Highest ad revenue — massive engagement, most seats
    ads_return_sensitivity_std=0.08,
)

# Initial customer groups (visible at Level 1 from start)
INITIAL_CUSTOMER_GROUPS: Dict[str, CustomerGroupConfig] = {
    'S1': CUSTOMER_GROUP_S1,
    'S2': CUSTOMER_GROUP_S2,
    'S3': CUSTOMER_GROUP_S3,
    'E1': CUSTOMER_GROUP_E1,
    'E2': CUSTOMER_GROUP_E2,
    'E3': CUSTOMER_GROUP_E3,
}

# Small groups list (initial only)
SMALL_CUSTOMER_GROUPS = ['S1', 'S2', 'S3']

# Enterprise groups list (initial only)
ENTERPRISE_CUSTOMER_GROUPS = ['E1', 'E2', 'E3']


def generate_discoverable_groups(rng, n_individual: int = 10, n_enterprise: int = 10) -> Dict[str, CustomerGroupConfig]:
    """Generate discoverable customer groups with diverse parameter variations.

    Each group is a niche market segment with unique characteristics.
    Individual groups: D_S01-D_S10 (discoverable small)
    Enterprise groups: D_E01-D_E10 (discoverable enterprise)

    Parameters are sampled from ranges that create diverse, interesting niches:
    - Some groups are high-volume/low-value, others low-volume/high-value
    - Enterprise groups have varied seat counts and negotiation styles
    """
    groups = {}

    # Names for discoverable individual segments
    individual_names = [
        'Niche Creators', 'Academic Researchers', 'Non-Profit Workers',
        'Small Agency Teams', 'Indie Game Devs', 'Freelance Writers',
        'Data Analysts', 'Social Media Managers', 'UX Designers', 'Music Producers',
        'Podcast Creators', 'E-Commerce Sellers', 'Digital Marketers', 'Tutors',
        'Virtual Assistants',
    ]

    # Names for discoverable enterprise segments
    enterprise_names = [
        'Government Agencies', 'Educational Institutions', 'Healthcare Networks',
        'Regional Banks', 'Insurance Brokers', 'Construction Firms',
        'Telecom Operators', 'Energy Companies', 'Real Estate Groups', 'Shipping Lines',
        'Automotive Suppliers', 'Media Conglomerates', 'Food & Beverage Chains',
        'Pharmaceutical Distributors', 'Airport Operators',
    ]

    # Generate individual discoverable groups
    for i in range(n_individual):
        gid = f'D_S{i+1:02d}'
        name = individual_names[i] if i < len(individual_names) else f'Individual Niche {i+1}'

        # Diverse parameter ranges for interesting niches
        # Quality params scaled for multiplier system (product_quality × tier_multiplier)
        # Per-group q_min (quality floor even if free) based on segment research
        # [SERVQUAL zone of tolerance, Kano must-be quality, freemium adoption research]
        _individual_qmin = {
            # D_S01: Niche Creators — ad-supported tool users, moderate tolerance
            # [HypeAuditor: 83% use AI despite 31% quality concerns; Adobe 2025: 86% use gen AI]
            'Niche Creators':           (0.15, 0.06),
            # D_S02: Academic Researchers — need accuracy for scholarly work
            # [Oxford Academic: baseline for factual accuracy in academic contexts is high]
            'Academic Researchers':     (0.35, 0.08),
            # D_S03: Non-Profit Workers — extreme budget constraints, any functional free tool welcomed
            # [Chronicle of Philanthropy: <3% tech spend; Godefroid 2024: budget barriers]
            'Non-Profit Workers':       (0.12, 0.05),
            # D_S04: Small Agency Teams — client-facing output needs baseline quality
            # [HubSpot 2025: agencies churn tools 2x faster; client expectations drive quality]
            'Small Agency Teams':       (0.25, 0.07),
            # D_S05: Indie Game Devs — tolerant adopters of free tools
            # [GDC 2025: 69% monthly subs; accept "good enough" for prototyping]
            'Indie Game Devs':          (0.15, 0.06),
            # D_S06: Freelance Writers — hypercritical of output quality (it's their craft)
            # [Contently 2025: 15%/month churn; fastest to notice quality regression]
            'Freelance Writers':        (0.25, 0.07),
            # D_S07: Data Analysts — employed professionals, need precision
            # [Kaggle 2024: 55% annual licenses; precision demands moderate-high floor]
            'Data Analysts':            (0.30, 0.08),
            # D_S08: Social Media Managers — need "good enough to post" not perfect
            # [Sprout Social 2025: SM managers evaluate new tools every 6 months]
            'Social Media Managers':    (0.15, 0.06),
            # D_S09: UX Designers — extremely sensitive to bad quality (it IS their expertise)
            # [Nielsen Norman 2024: designers rate ad-containing tools 35% lower]
            'UX Designers':             (0.25, 0.07),
            # D_S10: Music Producers — demanding audio quality standards
            # [MIDiA 2025: 65% prefer monthly; audio quality expectations very high]
            'Music Producers':          (0.20, 0.07),
        }
        qmin_mean, qmin_std = _individual_qmin.get(name, (0.18, 0.06))
        q_min = max(0.05, rng.normal(qmin_mean, qmin_std * 0.3))  # Light noise around researched mean
        q_max = rng.uniform(0.50, 0.85)  # Individual quality ceiling: casual→power user
        c_max = rng.uniform(30.0, 220.0)
        slope = rng.uniform(0.002, 0.012)
        usage = rng.uniform(40.0, 600.0)  # Wide spectrum: casual ~40K to power-user ~600K tokens/day

        # Market cap: diverse TAM for niche individual segments
        # Scaled proportionally to initial groups (S1=800K, S2=400K, S3=250K).
        # Niche segments range 30K-500K based on real-world niche AI tool markets:
        # AI writing $2.5B (GMInsights), AI game dev $3.2B (Dimension MR), AI music $558M (GVR)
        # CITATIONS: https://www.gminsights.com/industry-analysis/ai-writing-assistant-software-market
        #   https://dimensionmarketresearch.com/report/ai-in-game-development-market/
        #   https://www.grandviewresearch.com/industry-analysis/generative-ai-in-music-market-report
        #   https://www.demandsage.com/freelance-statistics/
        base_cap = int(rng.integers(30000, 500000))
        annual_growth = round(rng.uniform(0.04, 0.12), 3)

        # Per-group lock-in penalty matching each group's backstory
        # Individual groups: creators/freelancers hate lock-in (0.007-0.009),
        # professionals tolerate it more (0.004-0.006), niche varies by stability needs
        _individual_lockin = {
            # D_S01: Niche Creators — freelance artists, irregular income, need flexibility
            # Source: Upwork 2025 — 72% of creative freelancers prefer month-to-month tools
            'Niche Creators':           (0.008, 0.003),
            # D_S02: Academic Researchers — grant-funded, annual budget cycles, moderate tolerance
            # Source: Nature 2024 — 60% of researchers buy annual software licenses (grant cycles)
            'Academic Researchers':     (0.005, 0.002),
            # D_S03: Non-Profit Workers — tight budgets, need flexibility for funding changes
            # Source: NTEN 2025 — 78% of nonprofits prefer monthly SaaS to avoid budget lock-in
            'Non-Profit Workers':       (0.009, 0.003),
            # D_S04: Small Agency Teams — project-based, need to scale up/down rapidly
            # Source: HubSpot 2025 — agencies churn tools 2x faster than other segments
            'Small Agency Teams':       (0.007, 0.002),
            # D_S05: Indie Game Devs — project lifecycle tool usage, hate long commitments
            # Source: GDC 2025 Survey — 69% of indie devs use monthly subscriptions only
            'Indie Game Devs':          (0.008, 0.003),
            # D_S06: Freelance Writers — similar to S1, gig economy, need cancel flexibility
            # Source: Contently 2025 — freelance writers churn subscriptions at 15%/month
            'Freelance Writers':        (0.009, 0.003),
            # D_S07: Data Analysts — employed professionals, moderate commitment tolerance
            # Source: Kaggle 2024 Survey — 55% data professionals use annual tool licenses
            'Data Analysts':            (0.005, 0.002),
            # D_S08: Social Media Managers — trend-chasing, switch tools frequently
            # Source: Sprout Social 2025 — SM managers evaluate new tools every 6 months
            'Social Media Managers':    (0.007, 0.002),
            # D_S09: UX Designers — employed professionals, tool switching has high cost
            # Source: Nielsen Norman 2024 — designers invest in tool proficiency, lower churn
            'UX Designers':             (0.004, 0.002),
            # D_S10: Music Producers — creative freelancers, project-based, need flexibility
            # Source: MIDiA 2025 — 65% of independent music producers prefer monthly subscriptions
            'Music Producers':          (0.008, 0.003),
        }
        lockin_mean, lockin_std = _individual_lockin.get(name, (0.006, 0.002))

        # Per-group ads sensitivity: (quality_penalty_mean, revenue_factor_mean)
        # quality_penalty = how much ads degrade perceived quality (0-1 scale)
        # revenue_factor = daily $ return per unit ads strength per customer
        # Rationale: creative/freelance segments are more ad-tolerant (exposed to ad-supported tools)
        #   but generate less revenue per user (lower engagement depth, single-seat).
        #   Professional/employed segments have lower tolerance (expect clean UX) but higher
        #   engagement depth when they do tolerate ads.
        _individual_ads_sensitivity = {
            # D_S01: Niche Creators — accustomed to freemium ad-supported creative tools
            # Source: Publift 2025 — creative app users have 40% higher ad engagement than avg
            # Source: AppVerticals 2026 — creative/media apps ARPU $0.04-0.08/day from ads
            'Niche Creators':           (0.08, 0.10),
            # D_S02: Academic Researchers — expect clean, distraction-free tools; low ad tolerance
            # Source: Nature 2024 — 85% of researchers prefer ad-free tools; will pay premium
            # Source: Wiley 2025 — academic SaaS ad revenue negligible vs subscription revenue
            'Academic Researchers':     (0.18, 0.06),
            # D_S03: Non-Profit Workers — budget-conscious, more tolerant of ads if tool is cheaper
            # Source: NTEN 2025 — 62% of nonprofits accept ads in exchange for discounted SaaS
            # Source: TechSoup 2025 — nonprofit SaaS ad engagement 25% below average
            'Non-Profit Workers':       (0.07, 0.07),
            # D_S04: Small Agency Teams — client-facing work, ads in tools look unprofessional
            # Source: HubSpot 2025 — 78% of agency professionals cite "clean UX" as top-3 tool criterion
            # Source: Sprout Social 2025 — agency tool ad engagement rates below 1%
            'Small Agency Teams':       (0.16, 0.08),
            # D_S05: Indie Game Devs — highly tolerant of ads (understand ad monetization deeply)
            # Source: GDC 2025 — 73% of indie devs use ad-supported tools during development
            # Source: Unity 2025 — game dev tools with ads see 2x engagement vs other verticals
            'Indie Game Devs':          (0.06, 0.14),
            # D_S06: Freelance Writers — moderate tolerance, long session times = good ad exposure
            # Source: Contently 2025 — freelance writers spend avg 4.2hr/day in writing tools
            # Source: GMInsights 2025 — AI writing tool ad ARPU $0.06-0.12/user/day
            'Freelance Writers':        (0.10, 0.12),
            # D_S07: Data Analysts — professional environment, expect clean dashboards, low tolerance
            # Source: Kaggle 2024 — 71% of data professionals prefer ad-free analytics tools
            # Source: Dresner 2025 — BI tool ad monetization yields <5% of subscription revenue
            'Data Analysts':            (0.17, 0.08),
            # D_S08: Social Media Managers — very ad-literate, moderate tolerance but high engagement
            # Source: Sprout Social 2025 — SM managers interact with 3x more in-app content
            # Source: eMarketer 2025 — marketing tool users have highest ad click-through rates
            'Social Media Managers':    (0.09, 0.15),
            # D_S09: UX Designers — extremely sensitive to bad UX (ads = bad UX), lowest tolerance
            # Source: Nielsen Norman 2024 — designers rate ad-containing tools 35% lower in usability
            # Source: Figma Community 2025 — 92% of designers would pay more for ad-free experience
            'UX Designers':             (0.22, 0.05),
            # D_S10: Music Producers — creative workflow, moderate tolerance, decent session times
            # Source: MIDiA 2025 — music production tools with ads retain 60% of free users
            # Source: Splice 2025 — music tool ad engagement moderate at $0.03-0.08/user/day
            'Music Producers':          (0.11, 0.09),
        }
        ads_q_mean, ads_r_mean = _individual_ads_sensitivity.get(name, (0.12, 0.10))

        groups[gid] = CustomerGroupConfig(
            group_id=gid,
            group_name=name,
            is_enterprise=False,
            q_min_mean=round(qmin_mean, 3),
            q_min_std=round(qmin_std, 3),
            q_max_mean=round(q_max, 3),
            q_max_std=round(rng.uniform(0.06, 0.12), 3),
            c_max_mean=round(c_max, 1),
            c_max_std=round(c_max * rng.uniform(0.25, 0.50), 1),  # Increased variance
            slope_mean=round(slope, 4),
            slope_std=round(slope * rng.uniform(0.25, 0.45), 4),  # Increased variance
            usage_demand_mean=round(usage, 1),
            usage_demand_std=round(usage * rng.uniform(0.3, 0.6), 1),  # Increased variance
            base_market_cap=base_cap,
            annual_cap_growth_rate=annual_growth,
            lockin_penalty_mean=lockin_mean,
            lockin_penalty_std=lockin_std,
            ads_quality_sensitivity_mean=ads_q_mean,
            ads_quality_sensitivity_std=round(ads_q_mean * rng.uniform(0.25, 0.45), 3),
            ads_return_sensitivity_mean=ads_r_mean,
            ads_return_sensitivity_std=round(ads_r_mean * rng.uniform(0.25, 0.45), 3),
        )

    # Generate enterprise discoverable groups
    for i in range(n_enterprise):
        gid = f'D_E{i+1:02d}'
        name = enterprise_names[i] if i < len(enterprise_names) else f'Enterprise Niche {i+1}'

        # Quality params scaled for multiplier system (product_quality × tier_multiplier)
        # Per-group q_min (quality floor even if free) based on segment research
        # [SERVQUAL zone of tolerance, Kano must-be quality, regulatory compliance floors]
        _enterprise_qmin = {
            # D_E01: Government Agencies — most stringent procurement requirements
            # [OMB mandates NIST SP 800-218 compliance; FAR regulations; Black Duck gov requirements]
            'Government Agencies':      (0.50, 0.10),
            # D_E02: Educational Institutions — moderate quality needs, COPPA for student data
            # [EdTech Magazine 2025: 85% annual contracts; moderate regulatory burden]
            'Educational Institutions': (0.25, 0.07),
            # D_E03: Healthcare Networks — HIPAA non-negotiable; patient safety critical
            # [Drata: admin/physical/technical safeguards as baseline; errors = patient harm]
            'Healthcare Networks':      (0.55, 0.10),
            # D_E04: Regional Banks — Dodd-Frank, GLBA, SOX, PCI-DSS compliance
            # [Chargebee: "finance heavily regulated"; zero tolerance for quality issues]
            'Regional Banks':           (0.55, 0.10),
            # D_E05: Insurance Brokers — claims accuracy; moderate regulatory
            # [Novarica 2025: insurance tech contracts 3-year terms]
            'Insurance Brokers':        (0.35, 0.08),
            # D_E06: Construction Firms — pragmatic adopters, moderate quality needs
            # [Dodge Construction 2025: 55% prefer annual SaaS; project documentation accuracy]
            'Construction Firms':       (0.25, 0.07),
            # D_E07: Telecom Operators — massive infrastructure, moderate regulatory
            # [TM Forum 2025: telecom vendor contracts avg 5+ years]
            'Telecom Operators':        (0.30, 0.08),
            # D_E08: Energy Companies — regulatory/safety requirements significant
            # [Wood Mackenzie 2025: conservative adoption; safety-critical documentation]
            'Energy Companies':         (0.40, 0.08),
            # D_E09: Real Estate Groups — low regulatory burden, marketing-focused content
            # [PwC RE: AI expanding; listings need fact accuracy but creative embellishment OK]
            'Real Estate Groups':       (0.20, 0.06),
            # D_E10: Shipping Lines — standardization critical but moderate quality bar
            # [Drewry Maritime 2025: shipping IT contracts avg 4+ years]
            'Shipping Lines':           (0.25, 0.07),
        }
        qmin_mean, qmin_std = _enterprise_qmin.get(name, (0.30, 0.08))
        q_min = max(0.05, rng.normal(qmin_mean, qmin_std * 0.3))  # Light noise around researched mean
        q_max = rng.uniform(0.60, 0.92)  # Enterprise quality ceiling: cost-cutters→quality-first
        c_max = rng.uniform(40.0, 150.0)  # Per seat
        slope = rng.uniform(0.001, 0.010)
        usage = rng.uniform(30.0, 200.0)  # Per seat — wide spectrum: light ~30K to heavy ~200K tokens/seat/day
        seats_min = int(rng.integers(20, 200))
        seats_max = int(rng.integers(seats_min + 50, min(seats_min + 2000, 3000)))

        # Market cap: realistic per-vertical TAM for an AI SaaS startup (accounts, not seats).
        # Each vertical has a specific addressable market based on industry structure.
        # Methodology: total entities → filter for enterprise-grade → apply AI SaaS adoption rate
        #
        # GLOBAL CITATIONS (used across multiple verticals):
        # - Census SUSB 2022: firm counts by employee size
        #   https://www.census.gov/data/tables/2022/econ/susb/2022-susb-annual.html
        # - McKinsey 2025 Global Survey on AI: enterprise AI adoption 72% (up from 55% in 2023),
        #   but "meaningful deployment" only 20-35% depending on industry
        #   https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
        # - Menlo Ventures 2025: enterprise AI SaaS spend concentrated in top 15% of firms
        #   https://menlovc.com/2025-state-of-generative-ai-in-the-enterprise/
        #
        _enterprise_market_caps = {
            # ── D_E01: Government Agencies ──
            # Census of Governments 2025: ~90,075 local govt units (counties, municipalities,
            # townships, school districts, special districts).
            # Source: https://www.census.gov/newsroom/press-releases/2025/government-organization-counts.html
            # However, IT purchasing is consolidated: ~5,200 county-level + ~3,200 city/municipal
            # IT departments with standalone budgets (Governing Magazine 2024). Federal: ~180
            # civilian agencies with IT budgets (GSA 2025). State: 50 + DC = 51.
            # Total IT-purchasing entities: ~8,600. AI SaaS adoption in govt: ~15% (Deloitte
            # 2025 Govt AI Survey — hampered by FedRAMP, procurement cycles, budget constraints).
            # Addressable: ~8,600 × 15% = ~1,290
            'Government Agencies':          (1300, 0.04),
            #
            # ── D_E02: Educational Institutions ──
            # NCES 2023: 13,187 public school districts (operational, down from ~18K when
            # including inactive/consolidated). Source: https://nces.ed.gov/ccd/tables/202122_summary_2.asp
            # NCES IPEDS 2023: 3,896 degree-granting postsecondary institutions (down from
            # 4,726 peak in 2012 due to consolidation). Source: https://nces.ed.gov/fastfacts/display.asp?id=84
            # Total IT-purchasing education entities: ~17,100. AI SaaS adoption in education:
            # ~8% (HolonIQ 2025 — EdTech AI adoption slow due to budgets, privacy concerns,
            # curriculum integration barriers). Addressable: ~17,100 × 8% = ~1,370
            'Educational Institutions':     (1400, 0.06),
            #
            # ── D_E03: Healthcare Networks ──
            # AHA 2025 Fast Facts: 6,093 hospitals total; 4,157 in health systems; ~400
            # distinct health systems. Source: https://www.aha.org/statistics/fast-facts-us-hospitals
            # Enterprise AI SaaS buyers = health systems (400) + independent hospital groups
            # (~1,900 non-system hospitals, but only ~500 large enough for enterprise AI).
            # Plus ~800 large physician practice groups (50+ physicians, MGMA 2024).
            # Total enterprise entities: ~1,700. AI adoption in healthcare: 45% exploring,
            # ~25% deployed (Accenture Health 2025). Addressable: ~1,700 × 25% = ~425.
            # But healthcare AI SaaS is high-growth (8% annual cap growth reflects this).
            'Healthcare Networks':          (425, 0.08),
            #
            # ── D_E04: Regional Banks ──
            # FDIC Q3 2025: 4,379 insured commercial banks and savings institutions.
            # Of these, 3,953 are community banks (< $10B assets).
            # Source: https://www.fdic.gov/quarterly-banking-profile/fdic-statistics-glance
            # Enterprise AI SaaS buyers = all FDIC-insured institutions (even community banks
            # have IT departments). AI adoption in banking: ~30% deployed (Cornerstone
            # Advisors 2025 "What's Going On In Banking" — fintech/AI adoption accelerating).
            # Addressable: ~4,379 × 30% = ~1,314
            'Regional Banks':               (1300, 0.05),
            #
            # ── D_E05: Insurance Brokers/Companies ──
            # NAIC 2025: ~4,700 total insurance entities (2,684 P&C + 717 life + 1,331 health).
            # Source: coinlaw.io/us-insurance-industry-statistics (aggregating NAIC data)
            # Additionally, ~38,000 insurance agencies/brokerages (IBISWorld 2025), but most
            # are small (<10 employees). Enterprise-grade agencies/brokerages: ~2,500.
            # Total enterprise insurance entities: ~4,700 carriers + ~2,500 large brokerages = ~7,200.
            # AI adoption in insurance: ~20% (McKinsey Insurance 2025 — claims AI deployed,
            # underwriting AI still emerging). Addressable: ~7,200 × 20% = ~1,440
            'Insurance Brokers':            (1400, 0.06),
            #
            # ── D_E06: Construction Firms ──
            # BLS QCEW 2024: ~919,000 construction establishments total.
            # Source: https://www.bls.gov/cew/
            # Census SUSB 2022 (NAICS 23): ~750K firms total; firms with 50+ employees
            # estimated at ~15,000-18,000 (top ~2% of construction firms by employment).
            # Source: https://www.census.gov/data/tables/2022/econ/susb/2022-susb-annual.html
            # AI SaaS adoption in construction: very low, ~5% (McKinsey 2025 — construction
            # is one of the least digitized industries). Addressable: ~16,500 × 5% = ~825
            'Construction Firms':           (825, 0.07),
            #
            # ── D_E07: Telecom Operators ──
            # IBISWorld 2025: 1,021 wireless carriers + 845 wired carriers = 1,866 total.
            # Source: https://www.ibisworld.com/united-states/number-of-businesses/wireless-telecommunications-carriers/1267/
            # Plus 1,574 ISPs (IBISWorld 2025), but overlap with above carriers is high.
            # Unique telecom enterprise entities: ~2,200 (deduplicating multi-service carriers).
            # AI adoption in telecom: ~35% (TM Forum 2025 — network ops AI well-adopted,
            # customer-facing AI emerging). Addressable: ~2,200 × 35% = ~770
            'Telecom Operators':            (770, 0.05),
            #
            # ── D_E08: Energy Companies ──
            # EIA 2023: ~168 investor-owned utilities (IOUs) + ~1,958 municipal utilities +
            # ~812 cooperatives = ~2,938 electric utilities total.
            # Source: https://www.eia.gov/todayinenergy/detail.php?id=40913
            # Plus oil & gas (NAICS 211): ~3,722 crude petroleum extraction firms (Census 2022).
            # Source: https://siccode.com/naics-code/211/oil-gas-extraction
            # Total energy enterprise entities: ~2,938 utilities + ~3,722 O&G = ~6,660.
            # But many small: enterprise-grade (50+ employees) ~3,500.
            # AI adoption in energy: ~20% (DNV 2025 Energy Transition — AI for grid ops
            # and predictive maintenance gaining traction). Addressable: ~3,500 × 20% = ~700
            'Energy Companies':             (700, 0.06),
            #
            # ── D_E09: Real Estate Groups ──
            # NAREIT 2025: ~190 publicly traded REITs.
            # Source: https://www.reit.com/data-research/reit-market-data/us-reit-industry-equity-market-cap
            # NAR 2025: ~102,000 total real estate brokerages, but vast majority are tiny.
            # Source: https://www.nar.realtor/research-and-statistics/research-reports/profile-of-real-estate-firms
            # Enterprise-grade CRE firms (50+ employees): ~3,000-4,000 (top CRE brokerages,
            # property managers, REITs, institutional investors).
            # AI adoption in real estate: ~12% (Deloitte CRE Outlook 2025 — proptech AI
            # still early). Addressable: ~3,500 × 12% = ~420
            'Real Estate Groups':           (420, 0.04),
            #
            # ── D_E10: Shipping Lines / Logistics ──
            # Census 2020 (NAICS 483 Water Transportation): 1,407 firms.
            # Source: https://naicslist.com/naics/483
            # FMCSA 2025: ~750,000 active motor carriers, but enterprise-grade (50+ trucks /
            # 50+ employees) estimated ~3,000-5,000.
            # Source: https://ai.fmcsa.dot.gov/RegistrationStatistics
            # Enterprise shipping/logistics firms: ~1,400 water + ~4,000 large trucking = ~5,400.
            # But "shipping lines" implies primarily ocean/water + large logistics, so ~2,500.
            # AI adoption in logistics: ~18% (DHL Logistics Trend Radar 2025 — AI for route
            # optimization and demand forecasting). Addressable: ~2,500 × 18% = ~450
            'Shipping Lines':               (450, 0.05),
        }
        base_cap, annual_growth = _enterprise_market_caps.get(name, (800, 0.05))

        # Per-group lock-in penalty matching each enterprise group's backstory
        # Government/regulated = very low (0.001-0.002), standard contracting
        # Agile/competitive sectors = higher (0.004-0.006), want flexibility
        _enterprise_lockin = {
            # D_E01: Government Agencies — mandatory multi-year procurement cycles
            # Source: GSA 2025 — federal IT contracts average 3-5 years; FAR requires competition
            'Government Agencies':          (0.001, 0.001),
            # D_E02: Educational Institutions — annual budget cycles, accustomed to yearly contracts
            # Source: EdTech Magazine 2025 — 85% of K-12 SaaS contracts are annual
            'Educational Institutions':     (0.003, 0.001),
            # D_E03: Healthcare Networks — regulated, vendor qualification is expensive, prefer stability
            # Source: KLAS Research 2025 — healthcare IT contracts average 5-7 years
            'Healthcare Networks':          (0.002, 0.001),
            # D_E04: Regional Banks — regulated, compliance overhead makes switching costly
            # Source: Cornerstone Advisors 2025 — bank core tech contracts avg 7+ years
            'Regional Banks':               (0.002, 0.001),
            # D_E05: Insurance Brokers — moderate lock-in tolerance, annual policy cycles
            # Source: Novarica 2025 — insurance tech contracts typically 3-year terms
            'Insurance Brokers':            (0.003, 0.001),
            # D_E06: Construction Firms — project-based, seasonal work, want flexibility
            # Source: Dodge Construction 2025 — 55% of construction firms prefer annual SaaS
            'Construction Firms':           (0.005, 0.002),
            # D_E07: Telecom Operators — massive infrastructure, prefer long-term vendor stability
            # Source: TM Forum 2025 — telecom vendor contracts average 5+ years
            'Telecom Operators':            (0.001, 0.001),
            # D_E08: Energy Companies — long capex cycles, accustomed to multi-year commitments
            # Source: Wood Mackenzie 2025 — energy sector software contracts avg 3-5 years
            'Energy Companies':             (0.002, 0.001),
            # D_E09: Real Estate Groups — market-cyclical, need flexibility for downturns
            # Source: Deloitte Real Estate 2025 — CRE tech switching increased 40% in downturns
            'Real Estate Groups':           (0.005, 0.002),
            # D_E10: Shipping Lines — global operations, standardization critical, low switch tolerance
            # Source: Drewry Maritime 2025 — shipping IT vendor contracts avg 4+ years
            'Shipping Lines':               (0.002, 0.001),
        }
        lockin_mean, lockin_std = _enterprise_lockin.get(name, (0.003, 0.001))

        # Per-group ads sensitivity: (quality_penalty_mean, revenue_factor_mean)
        # Enterprise groups: generally MUCH higher quality sensitivity (ads look unprofessional
        # in enterprise tools), but higher revenue per seat due to more seats and engagement.
        # Regulated industries (gov, healthcare, banking) have near-zero ad tolerance.
        # Less regulated industries are more tolerant but still expect premium B2B experience.
        _enterprise_ads_sensitivity = {
            # D_E01: Government Agencies — zero tolerance for ads, compliance/security concerns
            # Source: GSA 2025 — federal procurement requires ad-free software environments
            # Source: FedScoop 2025 — 95% of gov IT buyers reject tools with any advertising
            'Government Agencies':          (0.30, 0.15),
            # D_E02: Educational Institutions — moderate tolerance if ads are educational/relevant
            # Source: EdTech Magazine 2025 — 45% of schools accept sponsored content in free tiers
            # Source: Mordor Intelligence 2025 — edtech ad revenue $0.10-0.20/seat/day
            'Educational Institutions':     (0.12, 0.18),
            # D_E03: Healthcare Networks — HIPAA concerns, ads seen as data privacy risk
            # Source: KLAS Research 2025 — healthcare IT requires ad-free; compliance mandates
            # Source: FBI/GVR 2025 — healthcare SaaS NEVER monetizes via ads (liability risk)
            'Healthcare Networks':          (0.28, 0.12),
            # D_E04: Regional Banks — heavily regulated, ads = compliance risk, brand damage
            # Source: Cornerstone Advisors 2025 — 98% of bank tech is ad-free; regulatory requirement
            # Source: AllAboutAI 2025 — fintech ad tolerance near zero in regulated banking
            'Regional Banks':               (0.27, 0.18),
            # D_E05: Insurance Brokers — regulated but less strict than banking, moderate tolerance
            # Source: Novarica 2025 — insurtech tools increasingly use sponsored recommendations
            # Source: Deloitte 2025 — insurance SaaS ad engagement higher than banking vertical
            'Insurance Brokers':            (0.18, 0.22),
            # D_E06: Construction Firms — practical/cost-focused, moderate ad tolerance for savings
            # Source: Dodge Construction 2025 — 40% of construction firms accept ads for discounts
            # Source: McKinsey Construction 2025 — construction tech users less UX-sensitive
            'Construction Firms':           (0.10, 0.20),
            # D_E07: Telecom Operators — large enterprises, expect premium vendor experience
            # Source: TM Forum 2025 — telecom BSS/OSS vendors never use ad monetization
            # Source: Analysys Mason 2025 — telecom SaaS expected to be enterprise-grade, ad-free
            'Telecom Operators':            (0.25, 0.28),
            # D_E08: Energy Companies — conservative industry, brand expectations, moderate sensitivity
            # Source: Wood Mackenzie 2025 — energy sector SaaS prioritizes reliability over cost
            # Source: Greentech Media 2025 — energy software ad tolerance low but not zero
            'Energy Companies':             (0.20, 0.25),
            # D_E09: Real Estate Groups — commercial/deal-focused, ads not uncommon in prop-tech
            # Source: Deloitte Real Estate 2025 — 55% of CRE tech platforms include sponsored listings
            # Source: PropTech Global 2025 — real estate SaaS has highest ad tolerance in enterprise
            'Real Estate Groups':           (0.08, 0.22),
            # D_E10: Shipping Lines — global operations, conservative, expect clean enterprise tools
            # Source: Drewry Maritime 2025 — shipping IT systems strictly enterprise-grade
            # Source: Lloyd's List 2025 — maritime tech platforms 90% subscription-only
            'Shipping Lines':               (0.22, 0.20),
        }
        ads_q_mean, ads_r_mean = _enterprise_ads_sensitivity.get(name, (0.18, 0.20))

        groups[gid] = CustomerGroupConfig(
            group_id=gid,
            group_name=name,
            is_enterprise=True,
            q_min_mean=round(qmin_mean, 3),
            q_min_std=round(qmin_std, 3),
            q_max_mean=round(q_max, 3),
            q_max_std=round(rng.uniform(0.05, 0.10), 3),
            c_max_mean=round(c_max, 1),
            c_max_std=round(c_max * rng.uniform(0.25, 0.45), 1),  # Increased variance
            slope_mean=round(slope, 4),
            slope_std=round(slope * rng.uniform(0.25, 0.45), 4),  # Increased variance
            usage_demand_mean=round(usage, 1),
            usage_demand_std=round(usage * rng.uniform(0.3, 0.6), 1),  # Increased variance
            base_market_cap=base_cap,
            annual_cap_growth_rate=annual_growth,
            seat_count_min=seats_min,
            seat_count_max=seats_max,
            negotiation_rate_mean=round(rng.uniform(0.15, 0.45), 2),
            negotiation_rate_std=round(rng.uniform(0.05, 0.12), 2),
            reply_delay_mean=round(rng.uniform(3.0, 25.0), 1),
            reply_delay_std=round(rng.uniform(1.0, 8.0), 1),
            max_negotiation_turns_mean=round(rng.uniform(4.0, 16.0), 1),
            max_negotiation_turns_std=round(rng.uniform(1.0, 4.0), 1),
            lockin_penalty_mean=lockin_mean,
            lockin_penalty_std=lockin_std,
            ads_quality_sensitivity_mean=ads_q_mean,
            ads_quality_sensitivity_std=round(ads_q_mean * rng.uniform(0.25, 0.45), 3),
            ads_return_sensitivity_mean=ads_r_mean,
            ads_return_sensitivity_std=round(ads_r_mean * rng.uniform(0.25, 0.45), 3),
        )

    # --- Populate persona/qualitative attribute dicts for discoverable groups ---
    _populate_discoverable_personas(groups)

    return groups


# =============================================================================
# DISCOVERABLE GROUP PERSONA ATTRIBUTES
# =============================================================================
# Each discoverable group gets unique qualitative attributes matching its niche.
# These are injected into the global PERSONA_* and COMPANY_* dicts at generation time.

# Individual discoverable group personas (keyed by group name)
_DISCOVERABLE_INDIVIDUAL_PERSONAS = {
    'Niche Creators': {
        'industries': ['digital-art', 'crafts', 'photography', 'illustration', 'video-production', 'animation'],
        'roles': ['creator', 'artisan', 'visual-artist', 'content-producer', 'designer', 'maker'],
        'work_styles': ['creative', 'passion-driven', 'experimental', 'visual-thinker', 'portfolio-focused'],
        'communication': ['visual', 'expressive', 'community-oriented', 'showcase-driven', 'informal'],
    },
    'Academic Researchers': {
        'industries': ['academia', 'research-lab', 'university', 'think-tank', 'scientific-publishing', 'R&D'],
        'roles': ['researcher', 'postdoc', 'PhD-student', 'lab-manager', 'research-associate', 'academic'],
        'work_styles': ['methodical', 'evidence-based', 'publication-driven', 'grant-focused', 'collaborative'],
        'communication': ['formal', 'citation-heavy', 'peer-review-style', 'academic', 'precise'],
    },
    'Non-Profit Workers': {
        'industries': ['charity', 'NGO', 'social-enterprise', 'community-org', 'advocacy', 'humanitarian'],
        'roles': ['program-coordinator', 'grant-writer', 'community-manager', 'outreach-lead', 'volunteer-coordinator', 'fundraiser'],
        'work_styles': ['mission-driven', 'resourceful', 'impact-focused', 'grant-conscious', 'collaborative'],
        'communication': ['empathetic', 'stakeholder-aware', 'report-oriented', 'community-focused', 'diplomatic'],
    },
    'Small Agency Teams': {
        'industries': ['design-agency', 'marketing-agency', 'PR-firm', 'branding', 'web-agency', 'creative-studio'],
        'roles': ['account-manager', 'project-lead', 'creative-director', 'strategist', 'producer', 'team-lead'],
        'work_styles': ['client-driven', 'deadline-focused', 'multi-project', 'fast-turnaround', 'pitch-ready'],
        'communication': ['client-facing', 'polished', 'presentation-ready', 'brief-driven', 'professional'],
    },
    'Indie Game Devs': {
        'industries': ['indie-games', 'mobile-gaming', 'game-modding', 'VR-development', 'interactive-media', 'game-design'],
        'roles': ['game-developer', 'level-designer', 'pixel-artist', 'sound-designer', 'indie-publisher', 'gameplay-programmer'],
        'work_styles': ['passion-project', 'crunch-tolerant', 'community-engaged', 'iterative', 'prototype-first'],
        'communication': ['casual', 'dev-log-style', 'community-update', 'Discord-native', 'meme-friendly'],
    },
    'Freelance Writers': {
        'industries': ['copywriting', 'content-writing', 'journalism', 'technical-writing', 'blogging', 'ghostwriting'],
        'roles': ['writer', 'editor', 'copywriter', 'content-strategist', 'blogger', 'ghostwriter'],
        'work_styles': ['deadline-driven', 'research-heavy', 'client-juggling', 'portfolio-building', 'word-count-focused'],
        'communication': ['articulate', 'concise', 'grammar-conscious', 'narrative-driven', 'editorial'],
    },
    'Data Analysts': {
        'industries': ['business-intelligence', 'market-research', 'analytics', 'data-consulting', 'survey-research', 'reporting'],
        'roles': ['data-analyst', 'BI-specialist', 'report-developer', 'insights-analyst', 'dashboard-builder', 'statistician'],
        'work_styles': ['data-driven', 'visualization-focused', 'SQL-fluent', 'spreadsheet-power-user', 'metric-obsessed'],
        'communication': ['numbers-first', 'chart-heavy', 'insight-oriented', 'structured', 'evidence-based'],
    },
    'Social Media Managers': {
        'industries': ['social-media', 'influencer-marketing', 'brand-management', 'community-management', 'digital-PR', 'content-scheduling'],
        'roles': ['social-media-manager', 'community-manager', 'content-scheduler', 'engagement-specialist', 'brand-voice-manager', 'analytics-tracker'],
        'work_styles': ['always-on', 'trend-watching', 'engagement-focused', 'calendar-driven', 'platform-native'],
        'communication': ['casual', 'emoji-fluent', 'hashtag-savvy', 'real-time', 'platform-adapted'],
    },
    'UX Designers': {
        'industries': ['UX-design', 'product-design', 'user-research', 'interaction-design', 'accessibility', 'design-systems'],
        'roles': ['UX-designer', 'UI-designer', 'user-researcher', 'interaction-designer', 'design-lead', 'prototyper'],
        'work_styles': ['user-centered', 'prototype-driven', 'research-first', 'iterative', 'accessibility-minded'],
        'communication': ['visual', 'wireframe-oriented', 'user-story-driven', 'feedback-seeking', 'design-critique'],
    },
    'Music Producers': {
        'industries': ['music-production', 'audio-engineering', 'podcast-production', 'sound-design', 'beat-making', 'mixing-mastering'],
        'roles': ['producer', 'audio-engineer', 'beat-maker', 'mix-engineer', 'sound-designer', 'composer'],
        'work_styles': ['creative-flow', 'session-based', 'deadline-flexible', 'ear-trained', 'gear-focused'],
        'communication': ['informal', 'vibe-driven', 'reference-track-style', 'collaborative', 'feedback-oriented'],
    },
}

# Enterprise discoverable group personas (keyed by group name)
_DISCOVERABLE_ENTERPRISE_PERSONAS = {
    'Government Agencies': {
        'industries': ['federal-government', 'state-government', 'municipal', 'defense-civilian', 'public-services', 'regulatory'],
        'contact_roles': ['Contracting Officer', 'IT Director', 'Program Manager', 'Chief Information Officer', 'Deputy Director'],
        'size_descriptors': ['federal', 'state-level', 'municipal', 'agency', 'bureau'],
        'cultures': ['process-driven', 'compliance-mandatory', 'risk-averse', 'audit-ready', 'policy-governed'],
        'decision_styles': ['RFP-based', 'multi-committee', 'budget-cycle-bound', 'compliance-gated', 'slow-deliberate'],
        'primary_concerns': ['FedRAMP-compliance', 'data-sovereignty', 'budget-justification', 'vendor-diversity', 'security-clearance'],
    },
    'Educational Institutions': {
        'industries': ['higher-education', 'K-12-district', 'online-learning', 'vocational-training', 'research-university', 'community-college'],
        'contact_roles': ['Dean of Technology', 'IT Director', 'Provost Office', 'EdTech Coordinator', 'CIO'],
        'size_descriptors': ['university', 'district-wide', 'multi-campus', 'statewide', 'consortium'],
        'cultures': ['academic-freedom', 'shared-governance', 'student-centered', 'research-oriented', 'inclusive'],
        'decision_styles': ['committee-driven', 'faculty-senate', 'budget-cycle', 'pilot-first', 'consensus-required'],
        'primary_concerns': ['student-outcomes', 'accessibility', 'budget-constraints', 'FERPA-compliance', 'academic-integrity'],
    },
    'Healthcare Networks': {
        'industries': ['hospital-system', 'clinic-network', 'telehealth', 'medical-group', 'health-insurance', 'care-coordination'],
        'contact_roles': ['Chief Medical Information Officer', 'VP Clinical Operations', 'Health IT Director', 'Compliance Officer', 'COO'],
        'size_descriptors': ['multi-hospital', 'regional-network', 'health-system', 'integrated-care', 'clinic-chain'],
        'cultures': ['patient-first', 'evidence-based', 'compliance-heavy', 'safety-critical', 'outcome-driven'],
        'decision_styles': ['clinical-validation', 'HIPAA-gated', 'physician-champion', 'committee-review', 'pilot-mandatory'],
        'primary_concerns': ['HIPAA-compliance', 'patient-safety', 'interoperability', 'clinical-workflow', 'cost-per-patient'],
    },
    'Regional Banks': {
        'industries': ['community-banking', 'credit-union', 'regional-finance', 'wealth-management', 'commercial-lending', 'mortgage'],
        'contact_roles': ['Chief Technology Officer', 'VP Digital Banking', 'Head of Operations', 'Chief Risk Officer', 'IT Manager'],
        'size_descriptors': ['regional', 'community', 'multi-branch', 'state-chartered', 'growing'],
        'cultures': ['trust-focused', 'regulatory-compliant', 'community-rooted', 'conservative', 'relationship-banking'],
        'decision_styles': ['board-approval', 'risk-committee', 'vendor-assessment', 'regulatory-review', 'budget-cycle'],
        'primary_concerns': ['regulatory-compliance', 'data-security', 'fraud-prevention', 'customer-trust', 'digital-transformation'],
    },
    'Insurance Brokers': {
        'industries': ['property-casualty', 'life-insurance', 'reinsurance', 'claims-processing', 'underwriting', 'benefits-admin'],
        'contact_roles': ['Chief Underwriting Officer', 'VP Claims', 'Head of Digital', 'Operations Director', 'CTO'],
        'size_descriptors': ['national-broker', 'regional-agency', 'specialty', 'wholesale', 'multi-line'],
        'cultures': ['risk-quantified', 'actuarial-minded', 'client-retention', 'claims-efficient', 'regulatory-aware'],
        'decision_styles': ['actuarial-analysis', 'ROI-modeled', 'vendor-panel', 'compliance-checked', 'phased-rollout'],
        'primary_concerns': ['claims-efficiency', 'regulatory-compliance', 'pricing-accuracy', 'policyholder-retention', 'fraud-detection'],
    },
    'Construction Firms': {
        'industries': ['commercial-construction', 'infrastructure', 'civil-engineering', 'project-management', 'general-contractor', 'specialty-trade'],
        'contact_roles': ['VP Operations', 'Project Director', 'Chief Estimator', 'Safety Director', 'IT Manager'],
        'size_descriptors': ['regional-builder', 'national-contractor', 'specialty', 'multi-project', 'heavy-civil'],
        'cultures': ['safety-first', 'deadline-critical', 'field-oriented', 'cost-controlled', 'project-based'],
        'decision_styles': ['project-justified', 'bid-cycle', 'field-tested', 'cost-benefit', 'quick-decision'],
        'primary_concerns': ['project-scheduling', 'safety-compliance', 'cost-overrun-prevention', 'workforce-management', 'equipment-tracking'],
    },
    'Telecom Operators': {
        'industries': ['mobile-network', 'broadband', 'fiber-optic', 'tower-company', 'MVNO', 'unified-communications'],
        'contact_roles': ['CTO', 'VP Network Operations', 'Head of Digital Services', 'Chief Architect', 'VP Customer Experience'],
        'size_descriptors': ['national-carrier', 'regional-operator', 'MVNO', 'fiber-provider', 'converged'],
        'cultures': ['network-reliability', 'customer-churn-focused', 'technology-forward', 'scale-oriented', 'competitive'],
        'decision_styles': ['technology-evaluation', 'vendor-bakeoff', 'PoC-required', 'executive-sponsor', 'integration-focused'],
        'primary_concerns': ['network-uptime', 'customer-churn', 'ARPU-growth', '5G-readiness', 'subscriber-experience'],
    },
    'Energy Companies': {
        'industries': ['oil-gas', 'renewable-energy', 'utilities', 'power-generation', 'energy-trading', 'smart-grid'],
        'contact_roles': ['VP Digital Transformation', 'Chief Sustainability Officer', 'Head of Operations Technology', 'CIO', 'VP Engineering'],
        'size_descriptors': ['utility', 'energy-major', 'renewable-developer', 'grid-operator', 'integrated-energy'],
        'cultures': ['safety-critical', 'regulatory-heavy', 'sustainability-driven', 'asset-focused', 'long-cycle'],
        'decision_styles': ['asset-lifecycle', 'regulatory-approval', 'capex-justified', 'safety-reviewed', 'board-level'],
        'primary_concerns': ['grid-reliability', 'regulatory-compliance', 'sustainability-targets', 'asset-optimization', 'worker-safety'],
    },
    'Real Estate Groups': {
        'industries': ['commercial-real-estate', 'property-management', 'REIT', 'development', 'brokerage', 'facilities-management'],
        'contact_roles': ['VP Property Technology', 'Head of Operations', 'Chief Investment Officer', 'Director of Asset Management', 'CTO'],
        'size_descriptors': ['portfolio-owner', 'national-developer', 'REIT', 'property-manager', 'mixed-use'],
        'cultures': ['deal-driven', 'asset-value-focused', 'tenant-retention', 'market-timing', 'relationship-heavy'],
        'decision_styles': ['IRR-justified', 'deal-by-deal', 'investment-committee', 'market-compared', 'tenant-impact'],
        'primary_concerns': ['occupancy-rates', 'tenant-experience', 'property-valuation', 'operational-efficiency', 'ESG-compliance'],
    },
    'Shipping Lines': {
        'industries': ['container-shipping', 'freight-logistics', 'port-operations', 'maritime', 'supply-chain', 'last-mile'],
        'contact_roles': ['VP Logistics Technology', 'Chief Operations Officer', 'Head of Digital', 'Fleet Manager', 'VP Supply Chain'],
        'size_descriptors': ['global-carrier', 'regional-freight', 'port-operator', 'logistics-provider', 'multi-modal'],
        'cultures': ['operations-focused', 'schedule-critical', 'global-mindset', 'efficiency-driven', 'weather-aware'],
        'decision_styles': ['operations-justified', 'fleet-wide', 'vendor-consolidated', 'route-tested', 'cost-per-TEU'],
        'primary_concerns': ['fleet-utilization', 'schedule-reliability', 'fuel-efficiency', 'port-congestion', 'customs-compliance'],
    },
}


def _populate_discoverable_personas(groups: Dict[str, 'CustomerGroupConfig']) -> None:
    """Populate global persona dicts with discoverable group attributes.

    Called by generate_discoverable_groups() after creating all groups.
    Injects entries into PERSONA_INDUSTRIES, PERSONA_ROLES, etc.
    """
    for gid, group in groups.items():
        name = group.group_name

        if not group.is_enterprise:
            # Individual discoverable group
            attrs = _DISCOVERABLE_INDIVIDUAL_PERSONAS.get(name)
            if not attrs:
                continue
            PERSONA_INDUSTRIES[gid] = attrs['industries']
            PERSONA_ROLES[gid] = attrs['roles']
            PERSONA_WORK_STYLES[gid] = attrs['work_styles']
            PERSONA_COMMUNICATION_STYLES[gid] = attrs['communication']
        else:
            # Enterprise discoverable group
            attrs = _DISCOVERABLE_ENTERPRISE_PERSONAS.get(name)
            if not attrs:
                continue
            COMPANY_INDUSTRIES[gid] = attrs['industries']
            COMPANY_CONTACT_ROLES[gid] = attrs['contact_roles']
            COMPANY_SIZE_DESCRIPTORS[gid] = attrs['size_descriptors']
            COMPANY_CULTURES[gid] = attrs['cultures']
            COMPANY_DECISION_STYLES[gid] = attrs['decision_styles']
            COMPANY_PRIMARY_CONCERNS[gid] = attrs['primary_concerns']


# All customer groups (initial only — discoverable groups added at simulation init)
# This dict is expanded at runtime by the simulator with discoverable groups
CUSTOMER_GROUPS: Dict[str, CustomerGroupConfig] = dict(INITIAL_CUSTOMER_GROUPS)

# =============================================================================
# V2.1: Non-Stationary Customer Preferences — Daily Drift Rates
# =============================================================================
# Each group's curve parameters drift by small percentages daily.
# Rates are multiplicative: new_value = old_value * (1 + drift_rate)
# Over 90 days, a +0.001/day drift compounds to ~+9.4% shift.
#
# Backstory rationale per group:
# - S1 (Budget/Gig): Growing freelancers → budgets expand → c_max rises
#   [Upwork 2024: 73% of freelancers report income growth year-over-year]
# - S2 (Quality Professionals): Stable preferences, minimal drift
# - S3 (Power Users/Tech): More sophisticated workflows → higher quality expectations
#   [JetBrains 2024: Developer tool expectations rise ~15% annually]
# - E1 (Cost-Cutting Enterprise): CFO budget tightening → c_max shrinks
#   [Gartner 2024: 62% of CFOs planned vendor cost optimization in 2024-2025]
# - E2 (Quality-First Enterprise): Compliance requirements tighten → steeper quality threshold
#   [McKinsey 2024: Regulatory compliance costs rising 12-18% annually for enterprises]
# - E3 (Strategic Partners): Large stable orgs, very slow drift
#
# Discoverable groups: smaller drift rates (±0.0002 to ±0.0005)
#
GROUP_PREFERENCE_DRIFT: Dict[str, Dict[str, float]] = {
    # Initial groups — meaningful drift to test agent adaptation
    'S1': {'c_max_drift': +0.001},                   # Budget grows +0.1%/day
    'S2': {},                                          # Stable (no drift)
    'S3': {'q_min_drift': +0.0005, 'q_max_drift': +0.0005},  # Participation curve rises +0.05%/day
    # Enterprise groups: seat_count_drift models organic workforce expansion/contraction
    # Rates: multiplicative daily (new = old × (1 + rate)). +0.0003/day ≈ +2.7%/90d ≈ +11%/yr.
    # CITATIONS:
    # - Optifai 2025: Enterprise NRR 118% median → ~18% annual seat+revenue expansion
    #   https://optif.ai/learn/questions/b2b-saas-net-revenue-retention-benchmark/
    # - BLS 2025: Manufacturing employment -2% YoY; logistics +3%; construction +4.2%
    #   https://www.bls.gov/charts/county-employment-and-wages/establishments-by-size.htm
    # - Deloitte 2026: Healthcare workforce +3.5% YoY; financial services flat
    #   https://www.deloitte.com/us/en/insights/industry/manufacturing-industrial-products/manufacturing-industry-outlook.html
    # - JP Morgan 2026: 48% of business leaders plan workforce expansion
    #   https://www.jpmorgan.com/insights/markets-and-economy/business-leaders-outlook/2026-us-business-leaders-outlook
    'E1': {'c_max_drift': -0.0005, 'seat_count_drift': -0.0002},  # Cost-cutting → layoffs: -1.8%/90d
    'E2': {'steepness_left_drift': +0.0003, 'seat_count_drift': +0.0003},  # Quality-first grows: +2.7%/90d
    'E3': {'c_max_drift': +0.0002, 'seat_count_drift': +0.0002},  # Strategic partners: slow expansion +1.8%/90d
    # Discoverable individual groups — small drifts (no seat_count_drift — single-seat users)
    'D_S01': {'c_max_drift': +0.0003},
    'D_S02': {'q_min_drift': +0.0002, 'q_max_drift': +0.0002},
    'D_S03': {},
    'D_S04': {'c_max_drift': -0.0002},
    'D_S05': {'c_max_drift': +0.0004},
    'D_S06': {'q_min_drift': +0.0003, 'q_max_drift': +0.0003},
    'D_S07': {},
    'D_S08': {'q_min_drift': +0.0004, 'q_max_drift': +0.0004},
    'D_S09': {'c_max_drift': +0.0002},
    'D_S10': {},
    # Discoverable enterprise groups — moderate drifts + seat_count_drift per industry
    'D_E01': {'c_max_drift': -0.0003, 'seat_count_drift': -0.0004},  # Gov: federal workforce cuts
    'D_E02': {'steepness_left_drift': +0.0002, 'seat_count_drift': +0.0002},  # Education: slow growth
    'D_E03': {'seat_count_drift': +0.0004},  # Healthcare: +3.5% YoY workforce (BLS)
    'D_E04': {'c_max_drift': -0.0004, 'seat_count_drift': -0.0001},  # Banks: branch consolidation
    'D_E05': {'steepness_left_drift': +0.0003, 'seat_count_drift': +0.0001},  # Insurance: stable
    'D_E06': {'seat_count_drift': +0.0003},  # Construction: +4.2% projected (ConstructConnect)
    'D_E07': {'c_max_drift': +0.0003, 'seat_count_drift': +0.0001},  # Telecom: slight expansion
    'D_E08': {'q_min_drift': +0.0002, 'q_max_drift': +0.0002, 'seat_count_drift': +0.0002},  # Energy: moderate growth
    'D_E09': {'c_max_drift': -0.0002, 'seat_count_drift': -0.0003},  # Real estate: volatile contraction
    'D_E10': {'seat_count_drift': +0.0001},  # Shipping: stable global ops
}

# =============================================================================
# V2.2: Individual Subscriber Drift — Post-Subscription Behavioral Shifts
# =============================================================================
# Unlike GROUP_PREFERENCE_DRIFT (which shifts the group mean, affecting new customers too),
# individual drift applies ONLY to existing subscribers' personal parameters.
# New customers are unaffected — they sample from the original group distribution.
#
# This simulates real-world post-subscription behavioral changes:
# - Budget fatigue: subscribers scrutinize cost more over time (c_max shrinks)
# - Rising expectations: experienced users demand more quality (q_min/q_max rise)
# - Threshold sharpening: users develop stronger quality opinions (steepness_left rises)
# - Budget expansion: satisfied enterprise users expand spend (c_max grows)
# - Adaptation/loyalty: integrated users become more tolerant (steepness_left decreases)
#
# Rates are multiplicative: new = old × (1 + rate) per day
# Over 90 days: ±0.0005/day ≈ ±4.6%, ±0.001/day ≈ ±9.4%, ±0.002/day ≈ ±19.7%
#
# RESEARCH CITATIONS:
# - ChurnFree 2026: SMB monthly churn 3-7%, enterprise 1%. Budget pressure is #1 SMB churn driver.
#   https://churnfree.com/blog/b2b-saas-churn-rate-benchmarks/
# - UserJot 2026: 50% of customers abandon within 90 days with bad onboarding.
#   https://userjot.com/blog/saas-churn-rate-benchmarks
# - PayPro Global: Price sensitivity decreases 20-30% after first year for retained customers.
#   https://payproglobal.com/answers/what-is-saas-pricing-sensitivity/
# - Custify 2026: 67% of customers have rising standards over time; NPS expectations grew 33% in 3 years.
#   https://www.custify.com/blog/customer-success-statistics/
# - Vitally 2025: Enterprise customers with onboarding complete are 12% less likely to churn in year 1.
#   https://www.vitally.io/post/saas-churn-benchmarks
# - K38 Consulting 2026: Hidden churn reasons — "product didn't grow with us" is top factor for tenured users.
#   https://k38consulting.com/saas-churn-reasons-revealed/
# - Formstack 2025: 37% of finance leaders paused capital spending; vendor consolidation ongoing.
#   https://www.formstack.com/blog/why-2025-is-the-year-of-vendor-consolidation
# - BetterCloud 2025/2026: SaaS spend per employee up 27%; SaaS inflation 4x general market.
#   https://www.bettercloud.com/monitor/saas-industry/
# - Gartner 2025: 62% of CFOs planned vendor cost optimization.
#   https://www.gartner.com/en/newsroom/press-releases
# - Netigate: Churn in SaaS — early-stage quality issues drive 23% of all churn.
#   https://www.netigate.net/articles/customer-satisfaction/churn-in-saas-companies
#
INDIVIDUAL_PREFERENCE_DRIFT: Dict[str, Dict[str, float]] = {
    # === Initial Groups ===

    # S1 (Price-Sensitive/Gig): AGGRESSIVE — Freelancers face severe subscription fatigue.
    # SaaS inflation 4x general market, irregular income, "cancel after free trial" archetype.
    # 3-7% monthly SMB churn driven by budget pressure; S1 is the worst-case segment.
    # [ChurnFree 2026, BetterCloud 2025: SaaS inflation 4x general market]
    'S1': {
        'c_max_drift': -0.0020,            # Severe budget fatigue: -0.2%/day ≈ -16.5% over 90 days
        'q_min_drift': +0.0008,            # Rising quality floor: +7.4% over 90 days
        'q_max_drift': +0.00090,           # Ceiling rises with standards (+0.0001 intrinsic + 0.0008 from rising demands)
    },

    # S2 (Quality Professionals): Professionals' quality bar rises steadily as they
    # integrate the tool into client-facing work. Budget relatively stable (employer-paid).
    # "Product didn't grow with us" is top churn reason for tenured professional users.
    # Price sensitivity ↓20-30% after year 1 for retained customers → budget expands.
    # [K38 2026, Custify 2026: 67% rising standards, PayPro Global: price sensitivity ↓20-30% after yr 1]
    'S2': {
        'q_min_drift': +0.0006,            # Rising quality floor: +5.5% over 90 days
        'steepness_left_drift': +0.0003,    # Sharper quality threshold: +2.7% over 90 days
        'c_max_drift': +0.0002,             # Employer-funded budget expansion: +1.8%/90d
        'q_max_drift': +0.00080,           # Ceiling rises with standards (+0.0002 intrinsic + 0.0006 from rising demands)
    },

    # S3 (Power Users/Tech): AGGRESSIVE threshold sharpening — tech users become extremely
    # opinionated about quality over time. Feature-gap churn is the #1 reason tenured power
    # users leave. They won't accept "good enough" — they want bleeding edge.
    # Willingly upgrade for advanced features; 10-15% annual ARPU increase for engaged users.
    # [K38 2026: tenured power users have highest feature-gap churn, ProfitWell: 10-15% annual ARPU increase]
    'S3': {
        'steepness_left_drift': +0.0012,    # Aggressive threshold sharpening: +11.4% over 90 days
        'q_min_drift': +0.0008,            # Rising quality floor: +7.4% over 90 days
        'c_max_drift': +0.00015,            # Power users upgrade: +1.4%/90d
        'q_max_drift': +0.00115,           # Ceiling rises with standards (+0.00035 intrinsic + 0.0008 from rising demands)
    },

    # E1 (Cost-Cutting Enterprise): AGGRESSIVE — CFO-driven budget pressure is relentless.
    # Annual vendor reviews → aggressive cost-cutting. Teams demand ROI justification.
    # 62% of CFOs planned cost optimization; E1 is the enterprise segment most likely to downgrade.
    # [Gartner 2025, Formstack 2025: 37% paused capex]
    'E1': {
        'c_max_drift': -0.0015,             # Aggressive budget cuts: -12.7% over 90 days
        'q_min_drift': +0.0006,            # Rising quality floor: +5.5% over 90 days
        'seat_count_drift': -0.0003,         # Post-subscription headcount cuts: -2.7% over 90 days
        'q_max_drift': +0.00068,           # Ceiling rises with standards (+0.00008 intrinsic + 0.0006)
    },

    # E2 (Quality-First Enterprise): Compliance requirements compound over time.
    # Quality threshold sharpens as audit cycles reveal gaps. Budget expands slightly
    # as they see ROI. Price sensitivity decreases 20-30% after year 1.
    # [PayPro Global, McKinsey 2024: compliance costs rising 12-18%/yr]
    'E2': {
        'steepness_left_drift': +0.0004,    # Compliance sharpening: +0.04%/day ≈ +3.7% over 90 days
        'c_max_drift': +0.0003,             # Budget expansion from proven ROI: +2.7% over 90 days
        'seat_count_drift': +0.0004,         # Compliance teams expand: +3.7% over 90 days
        'q_max_drift': +0.00025,           # Medium-fast ceiling expansion: ~9.1%/yr [McKinsey: quality-first orgs drive deep tool integration]
    },

    # E3 (Strategic Partners/Fortune 500): Very stable post-subscription. Massive switching
    # costs. Slight budget expansion as partnership deepens. Minimal individual drift.
    # [Menlo Ventures 2025: enterprise GenAI deals avg 2.5-year contracts]
    'E3': {
        'c_max_drift': +0.0002,             # Slow budget expansion: +0.02%/day ≈ +1.8% over 90 days
        'seat_count_drift': +0.0003,         # Org-wide rollout expansion: +2.7% over 90 days
        'q_max_drift': +0.00015,           # Medium ceiling expansion: ~5.5%/yr [Menlo Ventures: strategic partners deepen usage steadily]
    },

    # === Discoverable Small Groups (D_S01-D_S10) ===

    # D_S01 (Niche Creators): AGGRESSIVE — Irregular income, passion projects, feast-or-famine.
    # Budget collapses between gigs. Quality expectations spike as they compare to competitors.
    # [Upwork 2025: 72% of creative freelancers prefer month-to-month — high churn risk]
    'D_S01': {
        'c_max_drift': -0.0025,             # Severe budget fatigue: -20.2% over 90 days
        'q_min_drift': +0.0010,            # Rising quality floor (creative standards): +9.4% over 90 days
        'q_max_drift': +0.00115,           # Ceiling rises with standards (+0.00015 intrinsic + 0.0010)
    },

    # D_S02 (Academic Researchers): Grant cycles cause budget pressure. Quality expectations
    # rise steadily as they publish more. Methodical users develop sharp thresholds.
    # [Nature 2024: 60% buy annual licenses on grant cycles]
    'D_S02': {
        'q_min_drift': +0.0007,            # Rising quality floor (publication pressure): +6.5% over 90 days
        'steepness_left_drift': +0.0003,     # Methodical threshold sharpening: +2.7% over 90 days
        'q_max_drift': +0.00082,           # Ceiling rises with standards (+0.00012 intrinsic + 0.0007)
    },

    # D_S03 (Non-Profit Workers): AGGRESSIVE — Funding cliff. Grant cycles end abruptly,
    # forcing cancellation. Budget can collapse 30%+ when a grant period ends.
    # Mission-driven = tolerant of imperfections, but budget is the hard constraint.
    # [NTEN 2025: 78% prefer monthly to avoid budget lock-in]
    'D_S03': {
        'c_max_drift': -0.0030,             # Severe budget erosion (funding cliff): -0.3%/day ≈ -23.7% over 90 days
        'q_max_drift': +0.00007,           # Very slow ceiling expansion: ~2.6%/yr [NTEN 2025: mission-driven but tech-limited, minimal feature exploration]
    },

    # D_S04 (Small Agency Teams): AGGRESSIVE — Agencies are notorious tool-hoppers.
    # Project budgets shift constantly. Client expectations escalate every quarter.
    # 2x churn rate vs. other segments — triple threat: budget, quality, AND threshold.
    # [HubSpot 2025: agencies churn tools 2x faster]
    'D_S04': {
        'c_max_drift': -0.0015,             # Aggressive budget pressure: -12.7% over 90 days
        'q_min_drift': +0.0012,            # Client-driven quality escalation: +11.4% over 90 days
        'steepness_left_drift': +0.0008,     # Agencies develop razor-sharp requirements: +7.4% over 90 days
        'q_max_drift': +0.00135,           # Ceiling rises with standards (+0.00015 intrinsic + 0.0012)
    },

    # D_S05 (Indie Game Devs): AGGRESSIVE — Project ends = subscription ends.
    # Budget disappears when the game ships or gets abandoned. Quality spikes during
    # crunch when output must be production-ready.
    # [GDC 2025: 69% use monthly subscriptions only]
    'D_S05': {
        'c_max_drift': -0.0022,             # Project-end budget collapse: -18% over 90 days
        'q_min_drift': +0.0012,            # Production crunch quality spike: +11.4% over 90 days
        'q_max_drift': +0.00135,           # Ceiling rises with standards (+0.00015 intrinsic + 0.0012)
    },

    # D_S06 (Freelance Writers): AGGRESSIVE — 15%/month churn in reality.
    # Writers are hypercritical of AI output quality (it's their craft). Budget fatigue
    # is severe — they juggle 5+ subscriptions. Fastest to notice quality regression.
    # [Contently 2025: freelance writers churn subscriptions at 15%/month]
    'D_S06': {
        'c_max_drift': -0.0018,             # Severe budget fatigue: -15% over 90 days
        'q_min_drift': +0.0015,            # Extremely rising quality bar (craft pride): +14.4% over 90 days
        'steepness_left_drift': +0.0006,     # Writers develop very sharp quality thresholds: +5.5% over 90 days
        'q_max_drift': +0.00175,           # Ceiling rises with standards (+0.00025 intrinsic + 0.0015)
    },

    # D_S07 (Data Analysts): Employed professionals, stable budgets. But quality
    # expectations rise fast — data work requires increasingly precise outputs.
    # Employer-funded; precision demands justify higher-tier plans (Snowflake NRR 125-131%).
    # [Kaggle 2024: 55% use annual licenses, Snowflake FY2025: NRR 125-131% consumption expansion]
    'D_S07': {
        'q_min_drift': +0.0005,            # Rising precision demands: +4.6% over 90 days
        'steepness_left_drift': +0.0004,     # Sharpening accuracy threshold: +3.7% over 90 days
        'c_max_drift': +0.00015,            # Data precision justifies budget expansion: +5.5%/yr
        'q_max_drift': +0.00080,           # Ceiling rises with standards (+0.0003 intrinsic + 0.0005)
    },

    # D_S08 (Social Media Managers): Trend-chasers — budget shifts as priorities change.
    # Quality expectations are volatile. Tool switching is common.
    # [Sprout Social 2025: SM managers evaluate new tools every 6 months]
    'D_S08': {
        'c_max_drift': -0.0005,             # Moderate budget drift: -4.4% over 90 days
        'q_min_drift': +0.0004,            # Trend-driven quality demands: +3.7% over 90 days
        'q_max_drift': +0.00080,           # Ceiling rises with standards (+0.0004 intrinsic + 0.0004)
    },

    # D_S09 (UX Designers): Employed professionals with stable budgets. High switching
    # cost = loyalty effect. But quality expectations rise as design systems mature.
    # Tool proficiency → employer allocates more budget (Atlassian cloud NRR 120%).
    # [Nielsen Norman 2024: designers invest in tool proficiency, Atlassian FY2025: cloud NRR 120%]
    'D_S09': {
        'q_min_drift': +0.0004,            # Maturing design standards: +3.7% over 90 days
        'steepness_left_drift': -0.0002,     # Adaptation/loyalty (switching cost): -1.8% over 90 days
        'c_max_drift': +0.0001,             # Tool proficiency → employer budget expansion: +3.7%/yr
        'q_max_drift': +0.00065,           # Ceiling rises with standards (+0.00025 intrinsic + 0.0004)
    },

    # D_S10 (Music Producers): Creative freelancers with project-based budgets.
    # Quality expectations for audio output are extremely demanding.
    # [MIDiA 2025: 65% prefer monthly subscriptions]
    'D_S10': {
        'c_max_drift': -0.0007,             # Project-lifecycle budget decline: -6.1% over 90 days
        'q_min_drift': +0.0006,            # Demanding audio quality standards: +5.5% over 90 days
        'q_max_drift': +0.00080,           # Ceiling rises with standards (+0.0002 intrinsic + 0.0006)
    },

    # === Discoverable Enterprise Groups (D_E01-D_E10) ===

    # D_E01 (Government Agencies): Multi-year procurement = stable. But compliance
    # requirements compound. Budget is locked by appropriations — very slow drift.
    # [GSA 2025: federal IT contracts avg 3-5 years]
    # Individual seat_count_drift: post-subscription seat changes from team expansion/contraction.
    # Stacks with group-level seat_count_drift (which affects group-wide trends).
    # CITATIONS:
    # - ChartMogul 2025: Expansion revenue drives 40% of growth for $15M+ ARR companies
    #   https://chartmogul.com/reports/saas-retention-the-new-normal/
    # - High Alpha 2025: Top NRR firms get >50% of new ARR from upsells (seat expansion)
    #   https://www.highalpha.com/blog/net-revenue-retention-2025-why-its-crucial-for-saas-growth
    'D_E01': {
        'steepness_left_drift': +0.0003,     # Compliance threshold sharpening: +2.7% over 90 days
        'seat_count_drift': -0.0005,          # Federal workforce cuts (DOGE): -4.4% over 90 days
        'q_max_drift': +0.00005,           # Slowest ceiling expansion: ~1.8%/yr [Deloitte: 70% gov officials say capabilities lag private sector]
    },

    # D_E02 (Educational Institutions): Annual budget cycles cause pressure.
    # Student/faculty expectations rise every semester. Moderate quality drift.
    # [EdTech Magazine 2025: 85% of K-12 SaaS contracts are annual]
    'D_E02': {
        'q_min_drift': +0.0005,            # Rising educational standards: +4.6% over 90 days
        'c_max_drift': -0.0003,              # Annual budget cycle pressure: -2.7% over 90 days
        'seat_count_drift': +0.0002,          # Slow faculty/staff growth: +1.8% over 90 days
        'q_max_drift': +0.00068,           # Ceiling rises with standards (+0.00018 intrinsic + 0.0005)
    },

    # D_E03 (Healthcare Networks): AGGRESSIVE compliance sharpening — healthcare has
    # zero tolerance for quality issues (patient safety). Each HIPAA audit tightens the
    # screws. Quality threshold compounds relentlessly. One compliance gap = vendor banned.
    # [KLAS Research 2025: healthcare IT contracts avg 5-7 years]
    'D_E03': {
        'steepness_left_drift': +0.0012,     # Zero-tolerance compliance sharpening: +11.4% over 90 days
        'q_min_drift': +0.0008,            # Patient care standards rising fast: +7.4% over 90 days
        'seat_count_drift': +0.0005,          # Healthcare workforce boom: +4.6% over 90 days
        'q_max_drift': +0.00090,           # Ceiling rises with standards (+0.0001 intrinsic + 0.0008)
    },

    # D_E04 (Regional Banks): AGGRESSIVE — Regulatory compliance costs compound rapidly.
    # Each audit cycle raises the bar. Budget pressure from regulators forcing spend elsewhere.
    # Banks have zero tolerance for quality issues — one compliance failure = vendor fired.
    # [Cornerstone Advisors 2025: bank core tech contracts avg 7+ years]
    'D_E04': {
        'c_max_drift': -0.0010,              # Aggressive compliance cost drain: -8.6% over 90 days
        'steepness_left_drift': +0.0010,     # Zero-tolerance quality sharpening: +9.4% over 90 days
        'q_min_drift': +0.0005,            # Audit-driven quality escalation: +4.6% over 90 days
        'seat_count_drift': -0.0002,          # Branch consolidation: -1.8% over 90 days
        'q_max_drift': +0.00070,           # Ceiling rises with standards (+0.0002 intrinsic + 0.0005)
    },

    # D_E05 (Insurance Brokers): Annual policy cycles, moderate stability.
    # Claims accuracy requirements drive quality expectations up.
    # [Novarica 2025: insurance tech contracts typically 3-year terms]
    'D_E05': {
        'q_min_drift': +0.0004,            # Claims accuracy demands: +3.7% over 90 days
        'steepness_left_drift': +0.0003,      # Underwriting precision: +2.7% over 90 days
        'seat_count_drift': +0.0001,           # Stable workforce: +0.9% over 90 days
        'q_max_drift': +0.00052,           # Ceiling rises with standards (+0.00012 intrinsic + 0.0004)
    },

    # D_E06 (Construction Firms): Project-based, seasonal. Budget pressure from
    # project cost overruns. Moderate quality drift. Want flexibility.
    # [Dodge Construction 2025: 55% prefer annual SaaS]
    'D_E06': {
        'c_max_drift': -0.0005,              # Project cost pressure: -4.4% over 90 days
        'q_min_drift': +0.0003,            # Growing project complexity demands: +2.7% over 90 days
        'seat_count_drift': +0.0004,           # Construction hiring boom: +3.7% over 90 days
        'q_max_drift': +0.00037,           # Ceiling rises with standards (+0.00007 intrinsic + 0.0003)
    },

    # D_E07 (Telecom Operators): Massive infrastructure, very stable. Budget expands
    # as integration deepens. Slight quality expectation rise from network demands.
    # [TM Forum 2025: telecom vendor contracts avg 5+ years]
    'D_E07': {
        'c_max_drift': +0.0003,              # Integration-driven budget expansion: +2.7% over 90 days
        'q_min_drift': +0.0002,            # Slow rising network quality needs: +1.8% over 90 days
        'seat_count_drift': +0.0002,           # Network expansion teams: +1.8% over 90 days
        'q_max_drift': +0.00040,           # Ceiling rises with standards (+0.0002 intrinsic + 0.0002)
    },

    # D_E08 (Energy Companies): Long capex cycles, very slow drift. Sustainability
    # requirements gradually raise quality bar. Budget stable.
    # [Wood Mackenzie 2025: energy sector software contracts avg 3-5 years]
    'D_E08': {
        'q_min_drift': +0.0003,            # Sustainability-driven quality demands: +2.7% over 90 days
        'seat_count_drift': +0.0002,           # Energy transition hiring: +1.8% over 90 days
        'q_max_drift': +0.00036,           # Ceiling rises with standards (+0.00006 intrinsic + 0.0003)
    },

    # D_E09 (Real Estate Groups): AGGRESSIVE — Market downturns crush budgets.
    # CRE tech switching increased 40% in downturns. Budget volatility is extreme —
    # one bad quarter and all vendor budgets get slashed. Quality demands rise regardless.
    # [Deloitte RE 2025: CRE tech switching increased 40% in downturns]
    'D_E09': {
        'c_max_drift': -0.0018,              # Market-crash budget collapse: -15% over 90 days
        'q_min_drift': +0.0006,            # Rising proptech standards despite cuts: +5.5% over 90 days
        'seat_count_drift': -0.0004,           # Real estate layoffs in downturn: -3.5% over 90 days
        'q_max_drift': +0.00072,           # Ceiling rises with standards (+0.00012 intrinsic + 0.0006)
    },

    # D_E10 (Shipping Lines): Global operations, very stable. Standardization critical.
    # Quality threshold rises slowly as supply chain complexity grows.
    # [Drewry Maritime 2025: shipping IT vendor contracts avg 4+ years]
    'D_E10': {
        'steepness_left_drift': +0.0002,     # Supply chain quality threshold: +1.8% over 90 days
        'seat_count_drift': +0.0001,           # Stable global ops: +0.9% over 90 days
        'q_max_drift': +0.00012,           # Slow-medium ceiling expansion: ~4.4%/yr [Drewry: shipping IT contracts 4+ years, slow standardization]
    },
}

# Reputation influence matrix: I[from][to] = how much 'from' group affects 'to' group
# Row = source of influence, Column = target of influence
# Values indicate correlation strength (0 = no influence, 1 = full influence)
#
# Design rationale based on personas:
# - S1 (Price-Sensitive/Gig): Viral within own circle, some startup overlap with S3
# - S2 (Quality Professionals): Strong within professional networks, influences E2 (shared compliance focus)
# - S3 (Power Users/Tech): KEY INFLUENCERS - tech leads drive enterprise adoption (E1/E2/E3)
# - E1 (Cost-Cutting Enterprises): Influences peers, minimal outside reach
# - E2 (Quality-First Enterprises): Sets quality standards, influences S2 professionals and E3
# - E3 (Strategic Partners/Fortune 500): Market leaders influence all enterprises, validates tools for S3
#
REPUTATION_INFLUENCE_MATRIX: Dict[str, Dict[str, float]] = {
    # Full 26×26 matrix: 6 initial groups + 10 discoverable individual + 10 discoverable enterprise
    # Design: self=1.0, same-type adjacency 0.05-0.20, cross-type 0.01-0.10
    # Higher values for industry-adjacent pairs (e.g., Data Analysts ↔ Academic Researchers)
    # ★ INFLUENCER GROUPS: S3, D_S07, D_S08, E3, D_E07 have 2x boosted outgoing cross-group values
    #
    # --- Initial groups (S1-S3, E1-E3) ---
    'S1': {  # Price-Sensitive/Gig
           'S1': 1.00, 'S2': 0.050, 'S3': 0.15, 'E1': 0.020, 'E2': 0.010, 'E3': 0.010,
           'D_S01': 0.15, 'D_S02': 0.030, 'D_S03': 0.080, 'D_S04': 0.10, 'D_S05': 0.12,
           'D_S06': 0.080, 'D_S07': 0.040, 'D_S08': 0.12, 'D_S09': 0.10, 'D_S10': 0.12,
           'D_E01': 0.010, 'D_E02': 0.020, 'D_E03': 0.010, 'D_E04': 0.010, 'D_E05': 0.010,
           'D_E06': 0.010, 'D_E07': 0.010, 'D_E08': 0.010, 'D_E09': 0.010, 'D_E10': 0.010},
    'S2': {  # Quality Professionals
           'S1': 0.050, 'S2': 1.00, 'S3': 0.080, 'E1': 0.030, 'E2': 0.15, 'E3': 0.050,
           'D_S01': 0.040, 'D_S02': 0.15, 'D_S03': 0.10, 'D_S04': 0.12, 'D_S05': 0.040,
           'D_S06': 0.10, 'D_S07': 0.15, 'D_S08': 0.060, 'D_S09': 0.080, 'D_S10': 0.030,
           'D_E01': 0.020, 'D_E02': 0.080, 'D_E03': 0.060, 'D_E04': 0.050, 'D_E05': 0.050,
           'D_E06': 0.020, 'D_E07': 0.030, 'D_E08': 0.030, 'D_E09': 0.040, 'D_E10': 0.020},
    'S3': {  # Power Users/Tech ★INFLUENCER — 2x outgoing★
           'S1': 0.40, 'S2': 0.24, 'S3': 1.00, 'E1': 0.50, 'E2': 0.50, 'E3': 0.40,
           'D_S01': 0.16, 'D_S02': 0.20, 'D_S03': 0.10, 'D_S04': 0.24, 'D_S05': 0.36,
           'D_S06': 0.10, 'D_S07': 0.30, 'D_S08': 0.12, 'D_S09': 0.20, 'D_S10': 0.10,
           'D_E01': 0.10, 'D_E02': 0.16, 'D_E03': 0.12, 'D_E04': 0.080, 'D_E05': 0.080,
           'D_E06': 0.12, 'D_E07': 0.20, 'D_E08': 0.16, 'D_E09': 0.080, 'D_E10': 0.10},
    'E1': {  # Cost-Cutting Enterprises
           'S1': 0.020, 'S2': 0.030, 'S3': 0.050, 'E1': 1.00, 'E2': 0.10, 'E3': 0.080,
           'D_S01': 0.010, 'D_S02': 0.020, 'D_S03': 0.020, 'D_S04': 0.020, 'D_S05': 0.010,
           'D_S06': 0.010, 'D_S07': 0.030, 'D_S08': 0.010, 'D_S09': 0.010, 'D_S10': 0.010,
           'D_E01': 0.080, 'D_E02': 0.060, 'D_E03': 0.050, 'D_E04': 0.10, 'D_E05': 0.080,
           'D_E06': 0.060, 'D_E07': 0.050, 'D_E08': 0.050, 'D_E09': 0.060, 'D_E10': 0.050},
    'E2': {  # Quality-First Enterprises
           'S1': 0.020, 'S2': 0.18, 'S3': 0.080, 'E1': 0.15, 'E2': 1.00, 'E3': 0.22,
           'D_S01': 0.020, 'D_S02': 0.060, 'D_S03': 0.040, 'D_S04': 0.050, 'D_S05': 0.020,
           'D_S06': 0.030, 'D_S07': 0.080, 'D_S08': 0.020, 'D_S09': 0.040, 'D_S10': 0.020,
           'D_E01': 0.080, 'D_E02': 0.12, 'D_E03': 0.15, 'D_E04': 0.10, 'D_E05': 0.10,
           'D_E06': 0.050, 'D_E07': 0.10, 'D_E08': 0.10, 'D_E09': 0.080, 'D_E10': 0.060},
    'E3': {  # Strategic Partners/Fortune 500 ★INFLUENCER — 2x outgoing★
           'S1': 0.040, 'S2': 0.10, 'S3': 0.30, 'E1': 0.50, 'E2': 0.50, 'E3': 1.00,
           'D_S01': 0.020, 'D_S02': 0.060, 'D_S03': 0.040, 'D_S04': 0.080, 'D_S05': 0.040,
           'D_S06': 0.020, 'D_S07': 0.10, 'D_S08': 0.020, 'D_S09': 0.060, 'D_S10': 0.020,
           'D_E01': 0.30, 'D_E02': 0.20, 'D_E03': 0.24, 'D_E04': 0.30, 'D_E05': 0.20,
           'D_E06': 0.16, 'D_E07': 0.30, 'D_E08': 0.30, 'D_E09': 0.20, 'D_E10': 0.20},
    #
    # --- Discoverable individual groups (D_S01-D_S10) ---
    'D_S01': {  # Niche Creators
           'S1': 0.15, 'S2': 0.030, 'S3': 0.060, 'E1': 0.010, 'E2': 0.010, 'E3': 0.010,
           'D_S01': 1.00, 'D_S02': 0.030, 'D_S03': 0.050, 'D_S04': 0.10, 'D_S05': 0.12,
           'D_S06': 0.060, 'D_S07': 0.030, 'D_S08': 0.10, 'D_S09': 0.15, 'D_S10': 0.18,
           'D_E01': 0.010, 'D_E02': 0.020, 'D_E03': 0.010, 'D_E04': 0.010, 'D_E05': 0.010,
           'D_E06': 0.010, 'D_E07': 0.010, 'D_E08': 0.010, 'D_E09': 0.010, 'D_E10': 0.010},
    'D_S02': {  # Academic Researchers
           'S1': 0.020, 'S2': 0.15, 'S3': 0.080, 'E1': 0.020, 'E2': 0.050, 'E3': 0.020,
           'D_S01': 0.030, 'D_S02': 1.00, 'D_S03': 0.10, 'D_S04': 0.040, 'D_S05': 0.050,
           'D_S06': 0.080, 'D_S07': 0.18, 'D_S08': 0.020, 'D_S09': 0.040, 'D_S10': 0.020,
           'D_E01': 0.030, 'D_E02': 0.15, 'D_E03': 0.080, 'D_E04': 0.020, 'D_E05': 0.020,
           'D_E06': 0.010, 'D_E07': 0.030, 'D_E08': 0.050, 'D_E09': 0.010, 'D_E10': 0.010},
    'D_S03': {  # Non-Profit Workers
           'S1': 0.080, 'S2': 0.060, 'S3': 0.030, 'E1': 0.020, 'E2': 0.020, 'E3': 0.010,
           'D_S01': 0.050, 'D_S02': 0.10, 'D_S03': 1.00, 'D_S04': 0.060, 'D_S05': 0.030,
           'D_S06': 0.080, 'D_S07': 0.050, 'D_S08': 0.080, 'D_S09': 0.040, 'D_S10': 0.030,
           'D_E01': 0.050, 'D_E02': 0.12, 'D_E03': 0.060, 'D_E04': 0.020, 'D_E05': 0.020,
           'D_E06': 0.010, 'D_E07': 0.010, 'D_E08': 0.030, 'D_E09': 0.010, 'D_E10': 0.010},
    'D_S04': {  # Small Agency Teams
           'S1': 0.060, 'S2': 0.12, 'S3': 0.10, 'E1': 0.030, 'E2': 0.050, 'E3': 0.020,
           'D_S01': 0.080, 'D_S02': 0.040, 'D_S03': 0.050, 'D_S04': 1.00, 'D_S05': 0.050,
           'D_S06': 0.080, 'D_S07': 0.060, 'D_S08': 0.12, 'D_S09': 0.15, 'D_S10': 0.040,
           'D_E01': 0.010, 'D_E02': 0.030, 'D_E03': 0.020, 'D_E04': 0.020, 'D_E05': 0.030,
           'D_E06': 0.030, 'D_E07': 0.020, 'D_E08': 0.020, 'D_E09': 0.040, 'D_E10': 0.010},
    'D_S05': {  # Indie Game Devs
           'S1': 0.10, 'S2': 0.040, 'S3': 0.18, 'E1': 0.020, 'E2': 0.020, 'E3': 0.010,
           'D_S01': 0.12, 'D_S02': 0.040, 'D_S03': 0.030, 'D_S04': 0.060, 'D_S05': 1.00,
           'D_S06': 0.040, 'D_S07': 0.060, 'D_S08': 0.050, 'D_S09': 0.080, 'D_S10': 0.12,
           'D_E01': 0.010, 'D_E02': 0.030, 'D_E03': 0.010, 'D_E04': 0.010, 'D_E05': 0.010,
           'D_E06': 0.010, 'D_E07': 0.020, 'D_E08': 0.010, 'D_E09': 0.010, 'D_E10': 0.010},
    'D_S06': {  # Freelance Writers
           'S1': 0.060, 'S2': 0.10, 'S3': 0.040, 'E1': 0.010, 'E2': 0.030, 'E3': 0.010,
           'D_S01': 0.060, 'D_S02': 0.080, 'D_S03': 0.080, 'D_S04': 0.10, 'D_S05': 0.040,
           'D_S06': 1.00, 'D_S07': 0.050, 'D_S08': 0.12, 'D_S09': 0.050, 'D_S10': 0.030,
           'D_E01': 0.010, 'D_E02': 0.040, 'D_E03': 0.010, 'D_E04': 0.010, 'D_E05': 0.020,
           'D_E06': 0.010, 'D_E07': 0.010, 'D_E08': 0.010, 'D_E09': 0.020, 'D_E10': 0.010},
    'D_S07': {  # Data Analysts ★INFLUENCER — 2x outgoing★
           'S1': 0.060, 'S2': 0.30, 'S3': 0.30, 'E1': 0.080, 'E2': 0.16, 'E3': 0.080,
           'D_S01': 0.060, 'D_S02': 0.36, 'D_S03': 0.080, 'D_S04': 0.16, 'D_S05': 0.12,
           'D_S06': 0.080, 'D_S07': 1.00, 'D_S08': 0.10, 'D_S09': 0.12, 'D_S10': 0.060,
           'D_E01': 0.060, 'D_E02': 0.12, 'D_E03': 0.10, 'D_E04': 0.16, 'D_E05': 0.16,
           'D_E06': 0.040, 'D_E07': 0.10, 'D_E08': 0.12, 'D_E09': 0.060, 'D_E10': 0.080},
    'D_S08': {  # Social Media Managers ★INFLUENCER — 2x outgoing★
           'S1': 0.24, 'S2': 0.10, 'S3': 0.080, 'E1': 0.020, 'E2': 0.040, 'E3': 0.020,
           'D_S01': 0.20, 'D_S02': 0.040, 'D_S03': 0.12, 'D_S04': 0.30, 'D_S05': 0.10,
           'D_S06': 0.24, 'D_S07': 0.10, 'D_S08': 1.00, 'D_S09': 0.16, 'D_S10': 0.12,
           'D_E01': 0.020, 'D_E02': 0.040, 'D_E03': 0.020, 'D_E04': 0.020, 'D_E05': 0.020,
           'D_E06': 0.020, 'D_E07': 0.040, 'D_E08': 0.020, 'D_E09': 0.040, 'D_E10': 0.020},
    'D_S09': {  # UX Designers
           'S1': 0.080, 'S2': 0.080, 'S3': 0.12, 'E1': 0.020, 'E2': 0.040, 'E3': 0.020,
           'D_S01': 0.12, 'D_S02': 0.040, 'D_S03': 0.040, 'D_S04': 0.18, 'D_S05': 0.10,
           'D_S06': 0.050, 'D_S07': 0.060, 'D_S08': 0.080, 'D_S09': 1.00, 'D_S10': 0.050,
           'D_E01': 0.010, 'D_E02': 0.030, 'D_E03': 0.020, 'D_E04': 0.020, 'D_E05': 0.010,
           'D_E06': 0.010, 'D_E07': 0.020, 'D_E08': 0.010, 'D_E09': 0.020, 'D_E10': 0.010},
    'D_S10': {  # Music Producers
           'S1': 0.12, 'S2': 0.030, 'S3': 0.050, 'E1': 0.010, 'E2': 0.010, 'E3': 0.010,
           'D_S01': 0.18, 'D_S02': 0.020, 'D_S03': 0.030, 'D_S04': 0.040, 'D_S05': 0.12,
           'D_S06': 0.030, 'D_S07': 0.030, 'D_S08': 0.060, 'D_S09': 0.050, 'D_S10': 1.00,
           'D_E01': 0.010, 'D_E02': 0.010, 'D_E03': 0.010, 'D_E04': 0.010, 'D_E05': 0.010,
           'D_E06': 0.010, 'D_E07': 0.010, 'D_E08': 0.010, 'D_E09': 0.010, 'D_E10': 0.010},
    #
    # --- Discoverable enterprise groups (D_E01-D_E10) ---
    # Seat-adjusted: each enterprise subscriber has many users (seats) generating referrals.
    # Outgoing rates ~2x vs raw per-account, reflecting multi-seat word-of-mouth amplification.
    # CITATION: ChartMogul 2025 — enterprise NRR 118% driven partly by internal referral expansion
    # CITATION: ProductLed 2025 — B2B SaaS product-led growth: enterprise internal virality 2-3x SMB
    #   https://productled.com/blog/state-of-b2b-saas-2025-report
    'D_E01': {  # Government Agencies (seat-adjusted: agency-wide users)
           'S1': 0.020, 'S2': 0.040, 'S3': 0.060, 'E1': 0.16, 'E2': 0.12, 'E3': 0.24,
           'D_S01': 0.020, 'D_S02': 0.060, 'D_S03': 0.080, 'D_S04': 0.020, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.060, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 2.00, 'D_E02': 0.30, 'D_E03': 0.16, 'D_E04': 0.10, 'D_E05': 0.12,
           'D_E06': 0.16, 'D_E07': 0.12, 'D_E08': 0.20, 'D_E09': 0.080, 'D_E10': 0.10},
    'D_E02': {  # Educational Institutions (seat-adjusted: campus-wide users)
           'S1': 0.020, 'S2': 0.12, 'S3': 0.080, 'E1': 0.10, 'E2': 0.20, 'E3': 0.10,
           'D_S01': 0.040, 'D_S02': 0.30, 'D_S03': 0.20, 'D_S04': 0.060, 'D_S05': 0.040,
           'D_S06': 0.080, 'D_S07': 0.12, 'D_S08': 0.040, 'D_S09': 0.060, 'D_S10': 0.020,
           'D_E01': 0.30, 'D_E02': 2.00, 'D_E03': 0.20, 'D_E04': 0.080, 'D_E05': 0.060,
           'D_E06': 0.060, 'D_E07': 0.080, 'D_E08': 0.10, 'D_E09': 0.060, 'D_E10': 0.040},
    'D_E03': {  # Healthcare Networks (seat-adjusted: hospital system users)
           'S1': 0.020, 'S2': 0.080, 'S3': 0.080, 'E1': 0.080, 'E2': 0.24, 'E3': 0.16,
           'D_S01': 0.020, 'D_S02': 0.12, 'D_S03': 0.080, 'D_S04': 0.040, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.10, 'D_S08': 0.020, 'D_S09': 0.040, 'D_S10': 0.020,
           'D_E01': 0.16, 'D_E02': 0.20, 'D_E03': 2.00, 'D_E04': 0.10, 'D_E05': 0.30,
           'D_E06': 0.060, 'D_E07': 0.080, 'D_E08': 0.10, 'D_E09': 0.060, 'D_E10': 0.060},
    'D_E04': {  # Regional Banks (seat-adjusted: branch network users)
           'S1': 0.020, 'S2': 0.060, 'S3': 0.060, 'E1': 0.20, 'E2': 0.16, 'E3': 0.20,
           'D_S01': 0.020, 'D_S02': 0.040, 'D_S03': 0.040, 'D_S04': 0.040, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.12, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 0.10, 'D_E02': 0.080, 'D_E03': 0.10, 'D_E04': 2.00, 'D_E05': 0.36,
           'D_E06': 0.080, 'D_E07': 0.10, 'D_E08': 0.12, 'D_E09': 0.30, 'D_E10': 0.10},
    'D_E05': {  # Insurance Brokers (seat-adjusted: brokerage-wide users)
           'S1': 0.020, 'S2': 0.060, 'S3': 0.060, 'E1': 0.16, 'E2': 0.16, 'E3': 0.12,
           'D_S01': 0.020, 'D_S02': 0.040, 'D_S03': 0.040, 'D_S04': 0.040, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.12, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 0.12, 'D_E02': 0.060, 'D_E03': 0.30, 'D_E04': 0.36, 'D_E05': 2.00,
           'D_E06': 0.080, 'D_E07': 0.080, 'D_E08': 0.10, 'D_E09': 0.16, 'D_E10': 0.12},
    'D_E06': {  # Construction Firms (seat-adjusted: firm-wide users)
           'S1': 0.020, 'S2': 0.040, 'S3': 0.080, 'E1': 0.12, 'E2': 0.080, 'E3': 0.10,
           'D_S01': 0.020, 'D_S02': 0.020, 'D_S03': 0.040, 'D_S04': 0.060, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.040, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 0.16, 'D_E02': 0.060, 'D_E03': 0.060, 'D_E04': 0.080, 'D_E05': 0.080,
           'D_E06': 2.00, 'D_E07': 0.10, 'D_E08': 0.24, 'D_E09': 0.30, 'D_E10': 0.16},
    'D_E07': {  # Telecom Operators ★INFLUENCER — 2x outgoing★ (seat-adjusted: carrier-wide users)
           'S1': 0.040, 'S2': 0.12, 'S3': 0.32, 'E1': 0.20, 'E2': 0.32, 'E3': 0.48,
           'D_S01': 0.040, 'D_S02': 0.080, 'D_S03': 0.040, 'D_S04': 0.080, 'D_S05': 0.080,
           'D_S06': 0.040, 'D_S07': 0.20, 'D_S08': 0.080, 'D_S09': 0.080, 'D_S10': 0.040,
           'D_E01': 0.24, 'D_E02': 0.16, 'D_E03': 0.16, 'D_E04': 0.20, 'D_E05': 0.16,
           'D_E06': 0.20, 'D_E07': 2.00, 'D_E08': 0.48, 'D_E09': 0.16, 'D_E10': 0.32},
    'D_E08': {  # Energy Companies (seat-adjusted: utility-wide users)
           'S1': 0.020, 'S2': 0.040, 'S3': 0.10, 'E1': 0.10, 'E2': 0.16, 'E3': 0.24,
           'D_S01': 0.020, 'D_S02': 0.080, 'D_S03': 0.040, 'D_S04': 0.040, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.10, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 0.20, 'D_E02': 0.10, 'D_E03': 0.10, 'D_E04': 0.12, 'D_E05': 0.10,
           'D_E06': 0.24, 'D_E07': 0.24, 'D_E08': 2.00, 'D_E09': 0.10, 'D_E10': 0.16},
    'D_E09': {  # Real Estate Groups (seat-adjusted: group-wide users)
           'S1': 0.020, 'S2': 0.060, 'S3': 0.060, 'E1': 0.12, 'E2': 0.12, 'E3': 0.16,
           'D_S01': 0.020, 'D_S02': 0.020, 'D_S03': 0.040, 'D_S04': 0.080, 'D_S05': 0.020,
           'D_S06': 0.040, 'D_S07': 0.060, 'D_S08': 0.040, 'D_S09': 0.040, 'D_S10': 0.020,
           'D_E01': 0.080, 'D_E02': 0.060, 'D_E03': 0.060, 'D_E04': 0.30, 'D_E05': 0.16,
           'D_E06': 0.30, 'D_E07': 0.080, 'D_E08': 0.10, 'D_E09': 2.00, 'D_E10': 0.12},
    'D_E10': {  # Shipping Lines (seat-adjusted: fleet-wide users)
           'S1': 0.020, 'S2': 0.040, 'S3': 0.060, 'E1': 0.10, 'E2': 0.080, 'E3': 0.16,
           'D_S01': 0.020, 'D_S02': 0.020, 'D_S03': 0.020, 'D_S04': 0.020, 'D_S05': 0.020,
           'D_S06': 0.020, 'D_S07': 0.060, 'D_S08': 0.020, 'D_S09': 0.020, 'D_S10': 0.020,
           'D_E01': 0.10, 'D_E02': 0.040, 'D_E03': 0.040, 'D_E04': 0.10, 'D_E05': 0.12,
           'D_E06': 0.16, 'D_E07': 0.16, 'D_E08': 0.16, 'D_E09': 0.12, 'D_E10': 2.00},
}


# Reputation influence rate (how fast cross-group influence propagates)
REPUTATION_INFLUENCE_RATE: float = 0.1

# Network influence matrix: N[source][target] = daily leads in TARGET per existing subscriber of SOURCE
# at neutral reputation. The matrix directly encodes the network referral rate.
# Diagonal = self-referral rate (equivalent to old network_leads_per_1000_customers / 1000)
# Cross-group = how many new leads in target group are generated per subscriber in source group per day
#
# ★ INFLUENCER GROUPS: S3, D_S07, D_S08, E3, D_E07 have 4x boosted outgoing cross-group rates.
# These groups are "connectors" in their ecosystems — tech bloggers, data community leaders,
# social media amplifiers, Fortune 500 validators, and telecom industry evangelists.
#
# Key referral clusters:
# - Creative community: D_S01 Niche Creators ↔ D_S05 Indie Game Devs ↔ D_S10 Music Producers ↔ D_S09 UX Designers
# - Professional/analytical: D_S02 Academic Researchers ↔ D_S07 Data Analysts ↔ S2 Quality Professionals
# - Content/marketing: D_S06 Freelance Writers ↔ D_S08 Social Media Managers ↔ D_S04 Small Agencies
# - Financial services: D_E04 Regional Banks ↔ D_E05 Insurance ↔ D_E09 Real Estate
# - Infrastructure/utilities: D_E06 Construction ↔ D_E07 Telecom ↔ D_E08 Energy
# - Public sector: D_E01 Government ↔ D_E02 Education ↔ D_E03 Healthcare

NETWORK_INFLUENCE_MATRIX: Dict[str, Dict[str, float]] = {
    # Full 26×26 matrix: 6 initial groups + 10 discoverable individual + 10 discoverable enterprise
    # Unit: daily leads per existing subscriber of the SOURCE group at neutral reputation
    # Example: S1→S1 = 0.087 means 1000 S1 subscribers generate ~87 new S1 leads/day
    # Example: S3→E1 = 0.0234 means 1000 S3 (★influencer) subscribers generate ~23.4 E1 leads/day
    #
    # --- Initial groups (S1-S3, E1-E3) ---
    'S1': {  # Price-Sensitive/Gig — viral in creative/gig circles
           'S1': 0.0870, 'S2': 0.00261, 'S3': 0.00696, 'E1': 0.00087, 'E2': 0.00087, 'E3': 0.00087,
           'D_S01': 0.00870, 'D_S02': 0.00174, 'D_S03': 0.00435, 'D_S04': 0.00522, 'D_S05': 0.00609,
           'D_S06': 0.00435, 'D_S07': 0.00174, 'D_S08': 0.00696, 'D_S09': 0.00522, 'D_S10': 0.00696,
           'D_E01': 0.00087, 'D_E02': 0.00087, 'D_E03': 0.00087, 'D_E04': 0.00087, 'D_E05': 0.00087,
           'D_E06': 0.00087, 'D_E07': 0.00087, 'D_E08': 0.00087, 'D_E09': 0.00087, 'D_E10': 0.00087},
    'S2': {  # Quality Professionals — strong professional network referrals
           'S1': 0.00171, 'S2': 0.0570, 'S3': 0.00285, 'E1': 0.00114, 'E2': 0.00456, 'E3': 0.00171,
           'D_S01': 0.00114, 'D_S02': 0.00570, 'D_S03': 0.00342, 'D_S04': 0.00399, 'D_S05': 0.00114,
           'D_S06': 0.00342, 'D_S07': 0.00570, 'D_S08': 0.00171, 'D_S09': 0.00285, 'D_S10': 0.00114,
           'D_E01': 0.00057, 'D_E02': 0.00285, 'D_E03': 0.00228, 'D_E04': 0.00171, 'D_E05': 0.00171,
           'D_E06': 0.00057, 'D_E07': 0.00114, 'D_E08': 0.00114, 'D_E09': 0.00114, 'D_E10': 0.00057},
    'S3': {  # Power Users/Tech ★INFLUENCER — 4x outgoing cross-group★
           'S1': 0.01872, 'S2': 0.01092, 'S3': 0.0390, 'E1': 0.02340, 'E2': 0.02808, 'E3': 0.01872,
           'D_S01': 0.00780, 'D_S02': 0.00936, 'D_S03': 0.00468, 'D_S04': 0.01092, 'D_S05': 0.01872,
           'D_S06': 0.00468, 'D_S07': 0.01560, 'D_S08': 0.00468, 'D_S09': 0.00936, 'D_S10': 0.00468,
           'D_E01': 0.00468, 'D_E02': 0.00780, 'D_E03': 0.00624, 'D_E04': 0.00312, 'D_E05': 0.00312,
           'D_E06': 0.00468, 'D_E07': 0.00936, 'D_E08': 0.00780, 'D_E09': 0.00312, 'D_E10': 0.00468},
    'E1': {  # Cost-Cutting Enterprises — company-level referrals in enterprise peer network
           'S1': 0.00011, 'S2': 0.00022, 'S3': 0.00033, 'E1': 0.0110, 'E2': 0.00066, 'E3': 0.00055,
           'D_S01': 0.00011, 'D_S02': 0.00011, 'D_S03': 0.00011, 'D_S04': 0.00011, 'D_S05': 0.00011,
           'D_S06': 0.00011, 'D_S07': 0.00022, 'D_S08': 0.00011, 'D_S09': 0.00011, 'D_S10': 0.00011,
           'D_E01': 0.00055, 'D_E02': 0.00044, 'D_E03': 0.00033, 'D_E04': 0.00066, 'D_E05': 0.00055,
           'D_E06': 0.00044, 'D_E07': 0.00033, 'D_E08': 0.00033, 'D_E09': 0.00044, 'D_E10': 0.00033},
    'E2': {  # Quality-First Enterprises — company-level referrals in quality-conscious network
           'S1': 0.00009, 'S2': 0.00090, 'S3': 0.00045, 'E1': 0.00072, 'E2': 0.0090, 'E3': 0.00126,
           'D_S01': 0.00009, 'D_S02': 0.00036, 'D_S03': 0.00018, 'D_S04': 0.00027, 'D_S05': 0.00009,
           'D_S06': 0.00018, 'D_S07': 0.00045, 'D_S08': 0.00009, 'D_S09': 0.00018, 'D_S10': 0.00009,
           'D_E01': 0.00045, 'D_E02': 0.00072, 'D_E03': 0.00090, 'D_E04': 0.00054, 'D_E05': 0.00054,
           'D_E06': 0.00027, 'D_E07': 0.00054, 'D_E08': 0.00054, 'D_E09': 0.00045, 'D_E10': 0.00036},
    'E3': {  # Strategic Partners/Fortune 500 ★INFLUENCER — 4x outgoing★ (company-level referrals)
           'S1': 0.00028, 'S2': 0.00084, 'S3': 0.00280, 'E1': 0.00420, 'E2': 0.00420, 'E3': 0.0070,
           'D_S01': 0.00028, 'D_S02': 0.00056, 'D_S03': 0.00028, 'D_S04': 0.00056, 'D_S05': 0.00028,
           'D_S06': 0.00028, 'D_S07': 0.00084, 'D_S08': 0.00028, 'D_S09': 0.00056, 'D_S10': 0.00028,
           'D_E01': 0.00280, 'D_E02': 0.00168, 'D_E03': 0.00224, 'D_E04': 0.00280, 'D_E05': 0.00168,
           'D_E06': 0.00140, 'D_E07': 0.00280, 'D_E08': 0.00280, 'D_E09': 0.00168, 'D_E10': 0.00168},
    #
    # --- Discoverable individual groups (D_S01-D_S10) ---
    'D_S01': {  # Niche Creators — creative community (Music Producers, Indie Devs, UX)
           'S1': 0.00520, 'S2': 0.00104, 'S3': 0.00208, 'E1': 0.00052, 'E2': 0.00052, 'E3': 0.00052,
           'D_S01': 0.0520, 'D_S02': 0.00104, 'D_S03': 0.00156, 'D_S04': 0.00312, 'D_S05': 0.00416,
           'D_S06': 0.00208, 'D_S07': 0.00104, 'D_S08': 0.00312, 'D_S09': 0.00520, 'D_S10': 0.00624,
           'D_E01': 0.00052, 'D_E02': 0.00052, 'D_E03': 0.00052, 'D_E04': 0.00052, 'D_E05': 0.00052,
           'D_E06': 0.00052, 'D_E07': 0.00052, 'D_E08': 0.00052, 'D_E09': 0.00052, 'D_E10': 0.00052},
    'D_S02': {  # Academic Researchers — research/data network (Data Analysts, Non-Profits)
           'S1': 0.00018, 'S2': 0.00180, 'S3': 0.00090, 'E1': 0.00018, 'E2': 0.00054, 'E3': 0.00018,
           'D_S01': 0.00036, 'D_S02': 0.0180, 'D_S03': 0.00108, 'D_S04': 0.00036, 'D_S05': 0.00054,
           'D_S06': 0.00090, 'D_S07': 0.00216, 'D_S08': 0.00018, 'D_S09': 0.00036, 'D_S10': 0.00018,
           'D_E01': 0.00036, 'D_E02': 0.00180, 'D_E03': 0.00090, 'D_E04': 0.00018, 'D_E05': 0.00018,
           'D_E06': 0.00018, 'D_E07': 0.00036, 'D_E08': 0.00054, 'D_E09': 0.00018, 'D_E10': 0.00018},
    'D_S03': {  # Non-Profit Workers — non-profit/education network
           'S1': 0.00175, 'S2': 0.00140, 'S3': 0.00070, 'E1': 0.00035, 'E2': 0.00035, 'E3': 0.00035,
           'D_S01': 0.00105, 'D_S02': 0.00210, 'D_S03': 0.0350, 'D_S04': 0.00140, 'D_S05': 0.00070,
           'D_S06': 0.00175, 'D_S07': 0.00105, 'D_S08': 0.00175, 'D_S09': 0.00070, 'D_S10': 0.00070,
           'D_E01': 0.00105, 'D_E02': 0.00280, 'D_E03': 0.00140, 'D_E04': 0.00035, 'D_E05': 0.00035,
           'D_E06': 0.00035, 'D_E07': 0.00035, 'D_E08': 0.00070, 'D_E09': 0.00035, 'D_E10': 0.00035},
    'D_S04': {  # Small Agency Teams — agency/design/social media network
           'S1': 0.00128, 'S2': 0.00224, 'S3': 0.00192, 'E1': 0.00064, 'E2': 0.00096, 'E3': 0.00032,
           'D_S01': 0.00160, 'D_S02': 0.00064, 'D_S03': 0.00096, 'D_S04': 0.0320, 'D_S05': 0.00096,
           'D_S06': 0.00160, 'D_S07': 0.00128, 'D_S08': 0.00256, 'D_S09': 0.00320, 'D_S10': 0.00064,
           'D_E01': 0.00032, 'D_E02': 0.00064, 'D_E03': 0.00032, 'D_E04': 0.00032, 'D_E05': 0.00064,
           'D_E06': 0.00064, 'D_E07': 0.00032, 'D_E08': 0.00032, 'D_E09': 0.00096, 'D_E10': 0.00032},
    'D_S05': {  # Indie Game Devs — tech-creative cluster (S3, Niche Creators, Music)
           'S1': 0.00252, 'S2': 0.00084, 'S3': 0.00504, 'E1': 0.00042, 'E2': 0.00042, 'E3': 0.00042,
           'D_S01': 0.00336, 'D_S02': 0.00084, 'D_S03': 0.00084, 'D_S04': 0.00168, 'D_S05': 0.0420,
           'D_S06': 0.00084, 'D_S07': 0.00168, 'D_S08': 0.00126, 'D_S09': 0.00210, 'D_S10': 0.00336,
           'D_E01': 0.00042, 'D_E02': 0.00084, 'D_E03': 0.00042, 'D_E04': 0.00042, 'D_E05': 0.00042,
           'D_E06': 0.00042, 'D_E07': 0.00042, 'D_E08': 0.00042, 'D_E09': 0.00042, 'D_E10': 0.00042},
    'D_S06': {  # Freelance Writers — content/writing network (Agencies, Social Media)
           'S1': 0.00096, 'S2': 0.00144, 'S3': 0.00048, 'E1': 0.00024, 'E2': 0.00048, 'E3': 0.00024,
           'D_S01': 0.00096, 'D_S02': 0.00120, 'D_S03': 0.00120, 'D_S04': 0.00144, 'D_S05': 0.00048,
           'D_S06': 0.0240, 'D_S07': 0.00072, 'D_S08': 0.00192, 'D_S09': 0.00072, 'D_S10': 0.00048,
           'D_E01': 0.00024, 'D_E02': 0.00048, 'D_E03': 0.00024, 'D_E04': 0.00024, 'D_E05': 0.00024,
           'D_E06': 0.00024, 'D_E07': 0.00024, 'D_E08': 0.00024, 'D_E09': 0.00024, 'D_E10': 0.00024},
    'D_S07': {  # Data Analysts ★INFLUENCER — 4x outgoing cross-group★
           'S1': 0.00160, 'S2': 0.00800, 'S3': 0.00800, 'E1': 0.00160, 'E2': 0.00400, 'E3': 0.00160,
           'D_S01': 0.00160, 'D_S02': 0.00960, 'D_S03': 0.00160, 'D_S04': 0.00400, 'D_S05': 0.00320,
           'D_S06': 0.00160, 'D_S07': 0.0200, 'D_S08': 0.00240, 'D_S09': 0.00320, 'D_S10': 0.00160,
           'D_E01': 0.00160, 'D_E02': 0.00320, 'D_E03': 0.00240, 'D_E04': 0.00400, 'D_E05': 0.00400,
           'D_E06': 0.00080, 'D_E07': 0.00240, 'D_E08': 0.00320, 'D_E09': 0.00160, 'D_E10': 0.00160},
    'D_S08': {  # Social Media Managers ★INFLUENCER — 4x outgoing cross-group★
           'S1': 0.01856, 'S2': 0.00696, 'S3': 0.00464, 'E1': 0.00232, 'E2': 0.00232, 'E3': 0.00232,
           'D_S01': 0.01392, 'D_S02': 0.00232, 'D_S03': 0.00928, 'D_S04': 0.02320, 'D_S05': 0.00696,
           'D_S06': 0.01856, 'D_S07': 0.00696, 'D_S08': 0.0580, 'D_S09': 0.01160, 'D_S10': 0.00928,
           'D_E01': 0.00232, 'D_E02': 0.00232, 'D_E03': 0.00232, 'D_E04': 0.00232, 'D_E05': 0.00232,
           'D_E06': 0.00232, 'D_E07': 0.00232, 'D_E08': 0.00232, 'D_E09': 0.00232, 'D_E10': 0.00232},
    'D_S09': {  # UX Designers — design/creative/agency network
           'S1': 0.00190, 'S2': 0.00190, 'S3': 0.00266, 'E1': 0.00038, 'E2': 0.00076, 'E3': 0.00038,
           'D_S01': 0.00304, 'D_S02': 0.00076, 'D_S03': 0.00076, 'D_S04': 0.00456, 'D_S05': 0.00228,
           'D_S06': 0.00114, 'D_S07': 0.00152, 'D_S08': 0.00190, 'D_S09': 0.0380, 'D_S10': 0.00114,
           'D_E01': 0.00038, 'D_E02': 0.00076, 'D_E03': 0.00038, 'D_E04': 0.00038, 'D_E05': 0.00038,
           'D_E06': 0.00038, 'D_E07': 0.00038, 'D_E08': 0.00038, 'D_E09': 0.00038, 'D_E10': 0.00038},
    'D_S10': {  # Music Producers — creative/entertainment cluster (Niche Creators, Indie Devs)
           'S1': 0.00360, 'S2': 0.00090, 'S3': 0.00135, 'E1': 0.00045, 'E2': 0.00045, 'E3': 0.00045,
           'D_S01': 0.00540, 'D_S02': 0.00045, 'D_S03': 0.00090, 'D_S04': 0.00090, 'D_S05': 0.00360,
           'D_S06': 0.00090, 'D_S07': 0.00090, 'D_S08': 0.00180, 'D_S09': 0.00135, 'D_S10': 0.0450,
           'D_E01': 0.00045, 'D_E02': 0.00045, 'D_E03': 0.00045, 'D_E04': 0.00045, 'D_E05': 0.00045,
           'D_E06': 0.00045, 'D_E07': 0.00045, 'D_E08': 0.00045, 'D_E09': 0.00045, 'D_E10': 0.00045},
    #
    # --- Discoverable enterprise groups (D_E01-D_E10) ---
    'D_E01': {  # Government Agencies — public sector; company-level referrals rare
           'S1': 0.000012, 'S2': 0.000012, 'S3': 0.000025, 'E1': 0.000062, 'E2': 0.000050, 'E3': 0.000100,
           'D_S01': 0.000012, 'D_S02': 0.000025, 'D_S03': 0.000025, 'D_S04': 0.000012, 'D_S05': 0.000012,
           'D_S06': 0.000012, 'D_S07': 0.000025, 'D_S08': 0.000012, 'D_S09': 0.000012, 'D_S10': 0.000012,
           'D_E01': 0.00125, 'D_E02': 0.000125, 'D_E03': 0.000062, 'D_E04': 0.000037, 'D_E05': 0.000050,
           'D_E06': 0.000062, 'D_E07': 0.000050, 'D_E08': 0.000075, 'D_E09': 0.000025, 'D_E10': 0.000037},
    'D_E02': {  # Educational Institutions — education/research; company-level referrals rare
           'S1': 0.000037, 'S2': 0.000150, 'S3': 0.000075, 'E1': 0.000112, 'E2': 0.000225, 'E3': 0.000112,
           'D_S01': 0.000037, 'D_S02': 0.000375, 'D_S03': 0.000225, 'D_S04': 0.000075, 'D_S05': 0.000037,
           'D_S06': 0.000075, 'D_S07': 0.000150, 'D_S08': 0.000037, 'D_S09': 0.000075, 'D_S10': 0.000037,
           'D_E01': 0.000375, 'D_E02': 0.00375, 'D_E03': 0.000225, 'D_E04': 0.000075, 'D_E05': 0.000075,
           'D_E06': 0.000075, 'D_E07': 0.000075, 'D_E08': 0.000112, 'D_E09': 0.000075, 'D_E10': 0.000037},
    'D_E03': {  # Healthcare Networks — healthcare/insurance; company-level referrals rare
           'S1': 0.000020, 'S2': 0.000040, 'S3': 0.000040, 'E1': 0.000040, 'E2': 0.000160, 'E3': 0.000100,
           'D_S01': 0.000020, 'D_S02': 0.000080, 'D_S03': 0.000040, 'D_S04': 0.000020, 'D_S05': 0.000020,
           'D_S06': 0.000020, 'D_S07': 0.000060, 'D_S08': 0.000020, 'D_S09': 0.000020, 'D_S10': 0.000020,
           'D_E01': 0.000100, 'D_E02': 0.000120, 'D_E03': 0.00200, 'D_E04': 0.000060, 'D_E05': 0.000200,
           'D_E06': 0.000040, 'D_E07': 0.000040, 'D_E08': 0.000060, 'D_E09': 0.000040, 'D_E10': 0.000040},
    'D_E04': {  # Regional Banks — financial services; company-level referrals rare
           'S1': 0.000017, 'S2': 0.000035, 'S3': 0.000035, 'E1': 0.000105, 'E2': 0.000087, 'E3': 0.000105,
           'D_S01': 0.000017, 'D_S02': 0.000017, 'D_S03': 0.000017, 'D_S04': 0.000017, 'D_S05': 0.000017,
           'D_S06': 0.000017, 'D_S07': 0.000070, 'D_S08': 0.000017, 'D_S09': 0.000017, 'D_S10': 0.000017,
           'D_E01': 0.000052, 'D_E02': 0.000035, 'D_E03': 0.000052, 'D_E04': 0.00175, 'D_E05': 0.000210,
           'D_E06': 0.000035, 'D_E07': 0.000052, 'D_E08': 0.000070, 'D_E09': 0.000175, 'D_E10': 0.000052},
    'D_E05': {  # Insurance Brokers — financial/healthcare; company-level referrals rare
           'S1': 0.000025, 'S2': 0.000050, 'S3': 0.000050, 'E1': 0.000125, 'E2': 0.000125, 'E3': 0.000100,
           'D_S01': 0.000025, 'D_S02': 0.000025, 'D_S03': 0.000025, 'D_S04': 0.000025, 'D_S05': 0.000025,
           'D_S06': 0.000025, 'D_S07': 0.000100, 'D_S08': 0.000025, 'D_S09': 0.000025, 'D_S10': 0.000025,
           'D_E01': 0.000100, 'D_E02': 0.000050, 'D_E03': 0.000250, 'D_E04': 0.000300, 'D_E05': 0.00250,
           'D_E06': 0.000050, 'D_E07': 0.000050, 'D_E08': 0.000075, 'D_E09': 0.000125, 'D_E10': 0.000100},
    'D_E06': {  # Construction Firms — infrastructure; company-level referrals rare
           'S1': 0.000022, 'S2': 0.000022, 'S3': 0.000045, 'E1': 0.000090, 'E2': 0.000045, 'E3': 0.000067,
           'D_S01': 0.000022, 'D_S02': 0.000022, 'D_S03': 0.000022, 'D_S04': 0.000045, 'D_S05': 0.000022,
           'D_S06': 0.000022, 'D_S07': 0.000022, 'D_S08': 0.000022, 'D_S09': 0.000022, 'D_S10': 0.000022,
           'D_E01': 0.000112, 'D_E02': 0.000045, 'D_E03': 0.000045, 'D_E04': 0.000045, 'D_E05': 0.000045,
           'D_E06': 0.00225, 'D_E07': 0.000067, 'D_E08': 0.000180, 'D_E09': 0.000225, 'D_E10': 0.000112},
    'D_E07': {  # Telecom Operators ★INFLUENCER — 4x outgoing★ company-level referrals
           'S1': 0.000120, 'S2': 0.000240, 'S3': 0.000600, 'E1': 0.000360, 'E2': 0.000600, 'E3': 0.000960,
           'D_S01': 0.000120, 'D_S02': 0.000120, 'D_S03': 0.000120, 'D_S04': 0.000120, 'D_S05': 0.000120,
           'D_S06': 0.000120, 'D_S07': 0.000360, 'D_S08': 0.000120, 'D_S09': 0.000120, 'D_S10': 0.000120,
           'D_E01': 0.000480, 'D_E02': 0.000240, 'D_E03': 0.000240, 'D_E04': 0.000360, 'D_E05': 0.000240,
           'D_E06': 0.000360, 'D_E07': 0.00300, 'D_E08': 0.000960, 'D_E09': 0.000240, 'D_E10': 0.000600},
    'D_E08': {  # Energy Companies — infrastructure/strategic; company-level referrals rare
           'S1': 0.000015, 'S2': 0.000015, 'S3': 0.000045, 'E1': 0.000045, 'E2': 0.000075, 'E3': 0.000120,
           'D_S01': 0.000015, 'D_S02': 0.000030, 'D_S03': 0.000015, 'D_S04': 0.000015, 'D_S05': 0.000015,
           'D_S06': 0.000015, 'D_S07': 0.000045, 'D_S08': 0.000015, 'D_S09': 0.000015, 'D_S10': 0.000015,
           'D_E01': 0.000090, 'D_E02': 0.000045, 'D_E03': 0.000045, 'D_E04': 0.000060, 'D_E05': 0.000045,
           'D_E06': 0.000120, 'D_E07': 0.000120, 'D_E08': 0.00150, 'D_E09': 0.000045, 'D_E10': 0.000075},
    'D_E09': {  # Real Estate Groups — property/financial; company-level referrals rare
           'S1': 0.000035, 'S2': 0.000070, 'S3': 0.000070, 'E1': 0.000140, 'E2': 0.000140, 'E3': 0.000175,
           'D_S01': 0.000035, 'D_S02': 0.000035, 'D_S03': 0.000035, 'D_S04': 0.000070, 'D_S05': 0.000035,
           'D_S06': 0.000035, 'D_S07': 0.000070, 'D_S08': 0.000035, 'D_S09': 0.000035, 'D_S10': 0.000035,
           'D_E01': 0.000070, 'D_E02': 0.000070, 'D_E03': 0.000070, 'D_E04': 0.000350, 'D_E05': 0.000175,
           'D_E06': 0.000350, 'D_E07': 0.000070, 'D_E08': 0.000105, 'D_E09': 0.00350, 'D_E10': 0.000140},
    'D_E10': {  # Shipping Lines — logistics/infrastructure; company-level referrals rare
           'S1': 0.000010, 'S2': 0.000010, 'S3': 0.000020, 'E1': 0.000030, 'E2': 0.000020, 'E3': 0.000050,
           'D_S01': 0.000010, 'D_S02': 0.000010, 'D_S03': 0.000010, 'D_S04': 0.000010, 'D_S05': 0.000010,
           'D_S06': 0.000010, 'D_S07': 0.000020, 'D_S08': 0.000010, 'D_S09': 0.000010, 'D_S10': 0.000010,
           'D_E01': 0.000030, 'D_E02': 0.000010, 'D_E03': 0.000010, 'D_E04': 0.000030, 'D_E05': 0.000040,
           'D_E06': 0.000050, 'D_E07': 0.000050, 'D_E08': 0.000050, 'D_E09': 0.000040, 'D_E10': 0.00100},
}


# =============================================================================
# PERSONA SYSTEM: Multi-axis qualitative customer attributes
# =============================================================================

# Small customer persona axes (S1, S2, S3)
PERSONA_INDUSTRIES: Dict[str, List[str]] = {
    'S1': ['creative', 'education', 'gig-economy', 'hobby', 'small-retail', 'content-creation'],
    'S2': ['legal', 'consulting', 'healthcare', 'finance', 'real-estate', 'accounting'],
    'S3': ['tech', 'data-science', 'agency', 'automation', 'devops', 'startup'],
}

PERSONA_ROLES: Dict[str, List[str]] = {
    'S1': ['freelancer', 'student', 'side-hustler', 'solopreneur', 'creator', 'hobbyist'],
    'S2': ['independent-practitioner', 'solo-consultant', 'specialist', 'advisor', 'professional'],
    'S3': ['senior-developer', 'lead-engineer', 'founder', 'technical-director', 'architect'],
}

PERSONA_EXPERIENCE_LEVELS: List[str] = ['early-career', 'mid-career', 'experienced', 'veteran']

PERSONA_WORK_STYLES: Dict[str, List[str]] = {
    'S1': ['scrappy', 'experimental', 'fast-moving', 'budget-stretcher', 'resourceful'],
    'S2': ['methodical', 'thorough', 'quality-driven', 'client-focused', 'detail-oriented'],
    'S3': ['technical', 'optimization-focused', 'scale-minded', 'automation-first', 'data-driven'],
}

PERSONA_TECH_SAVVY: List[str] = ['basic', 'comfortable', 'proficient', 'advanced', 'expert']

PERSONA_COMMUNICATION_STYLES: Dict[str, List[str]] = {
    'S1': ['casual', 'emoji-friendly', 'brief', 'social-media-native', 'expressive'],
    'S2': ['professional', 'measured', 'articulate', 'formal', 'diplomatic'],
    'S3': ['terse', 'technical', 'direct', 'data-focused', 'no-nonsense'],
}

# Enterprise company profile axes (E1, E2, E3)
COMPANY_INDUSTRIES: Dict[str, List[str]] = {
    'E1': ['manufacturing', 'logistics', 'healthcare-admin', 'retail-chain', 'hospitality', 'distribution'],
    'E2': ['law-firm', 'biotech', 'consulting', 'financial-services', 'insurance', 'pharmaceuticals'],
    'E3': ['conglomerate', 'digital-services', 'media-group', 'tech-platform', 'multinational', 'venture-backed'],
}

COMPANY_SIZE_DESCRIPTORS: Dict[str, List[str]] = {
    'E1': ['mid-market', 'regional', 'growing', 'established-regional', 'multi-location'],
    'E2': ['established', 'specialized', 'boutique', 'prestigious', 'recognized'],
    'E3': ['large-scale', 'global', 'industry-leader', 'Fortune-500', 'market-leader'],
}

COMPANY_CULTURES: Dict[str, List[str]] = {
    'E1': ['cost-conscious', 'efficiency-driven', 'lean', 'practical', 'results-oriented'],
    'E2': ['excellence-focused', 'compliance-first', 'professional', 'meticulous', 'quality-obsessed'],
    'E3': ['innovation-driven', 'strategic', 'partnership-oriented', 'visionary', 'growth-focused'],
}

COMPANY_DECISION_STYLES: Dict[str, List[str]] = {
    'E1': ['fast', 'ROI-focused', 'benchmark-driven', 'committee-light', 'pragmatic'],
    'E2': ['thorough', 'risk-averse', 'committee-heavy', 'documented', 'deliberate'],
    'E3': ['relationship-based', 'executive-level', 'long-cycle', 'strategic', 'consensus-driven'],
}

COMPANY_PRIMARY_CONCERNS: Dict[str, List[str]] = {
    'E1': ['cost-reduction', 'operational-efficiency', 'quick-wins', 'budget-compliance', 'ROI'],
    'E2': ['quality-assurance', 'compliance', 'reliability', 'audit-trail', 'risk-mitigation'],
    'E3': ['competitive-advantage', 'innovation', 'partnership-value', 'market-position', 'scalability'],
}

COMPANY_CONTACT_ROLES: Dict[str, List[str]] = {
    'E1': ['IT Director', 'VP Operations', 'Procurement Lead', 'Operations Manager', 'IT Manager'],
    'E2': ['Managing Partner', 'Chief Compliance Officer', 'Head of Technology', 'General Counsel', 'CTO'],
    'E3': ['Chief Strategy Officer', 'CEO', 'VP Strategic Partnerships', 'Chief Digital Officer', 'President'],
}


# =============================================================================
# WEEKLY & MONTHLY CYCLES (v2.1)
# =============================================================================
# Real SaaS metrics have strong day-of-week and month-of-month patterns.
# These multipliers are applied to lead generation, usage, and social media activity.
#
# CITATIONS:
# - Salesforce 2024: B2B engagement peaks Tuesday-Thursday, drops 40-60% on weekends
#   https://www.salesforce.com/resources/articles/best-time-to-send-email/
# - HubSpot 2025: Website traffic drops 30-50% on weekends for B2B SaaS
#   https://blog.hubspot.com/marketing/best-time-to-send-email
# - ChartMogul 2024: SaaS signups cluster around month-start (budget allocations)
#   and month-end (enterprise billing decisions)
#   https://chartmogul.com/reports/saas-growth-report/

# Day index: 0=Monday, 1=Tuesday, ..., 5=Saturday, 6=Sunday
# Midweek (Tue-Thu) gets 10-15% boost, weekends get 40% reduction
WEEKLY_MULTIPLIERS: List[float] = [1.0, 1.1, 1.15, 1.1, 1.0, 0.6, 0.6]

# Monthly multipliers by day-of-month (1-indexed, using day % 30 + 1)
# Days 1-3: signup surge from new budget allocations
# Days 4-27: normal activity
# Days 28-30: enterprise billing decisions cluster → churn/upgrade spikes
MONTHLY_MULTIPLIERS: Dict[int, float] = {
    1: 1.15, 2: 1.15, 3: 1.15,  # Month-start surge
    28: 1.10, 29: 1.10, 30: 1.10,  # Month-end billing cluster
    # All other days default to 1.0 (retrieved via .get(day, 1.0))
}


@dataclass
class ScenarioPack:
    """Scenario pack configuration."""
    name: str
    description: str

    # Shock probabilities per day
    demand_surge_prob: float = 0.005
    enterprise_freeze_prob: float = 0.008  # 0.8% daily ≈ 1 freeze every 125 days ≈ 3/year (realistic macro-driven events)


# Predefined scenario packs
SCENARIO_PACKS = {
    'demand_surges': ScenarioPack(
        name='Demand Surges Common',
        description='Frequent demand surges requiring capacity management',
        demand_surge_prob=0.015,
    ),
    'large_customers': ScenarioPack(
        name='Large Customers Dominate',
        description='Large enterprise customers make up most revenue',
        enterprise_freeze_prob=0.003,  # ~40% of default 0.008; stable enterprise environment
    ),
}
