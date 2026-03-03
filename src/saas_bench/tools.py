"""Agent tools (actions) for SaaS Bench."""

import sqlite3
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from pathlib import Path
from numpy.random import Generator, default_rng

from .config import MODEL_TIERS, CAPACITY_TIERS, CUSTOMER_GROUPS, INITIAL_CUSTOMER_GROUPS, AD_CHANNELS, BenchmarkConfig, PREDEFINED_VCS, RESEARCH_TIERS, RESEARCH_TIERS_BY_ID, ResearchTier, REPUTATION_INFLUENCE_MATRIX, REPUTATION_INFLUENCE_RATE, NETWORK_INFLUENCE_MATRIX, compute_term_sheet_friendliness
from .database import (
    add_ledger_entry, get_recent_social_posts, get_posts_by_sentiment,
    get_world_context, get_all_world_context,
    get_group_characteristics,
    # V2: Equity functions
    get_all_shareholders, get_shareholder, get_total_shares, get_cap_table,
    update_shareholder_shares, record_funding_round, record_dividend,
    get_total_dividends, get_founder_cumulative_dividends, get_dividend_history, get_retained_earnings, get_funding_rounds,
    get_vc_latest_turn, get_active_vc_negotiations, close_vc_negotiation,
    add_vc_turn, get_vc_turns, add_notification,
    # Enterprise turn functions
    add_enterprise_turn, get_enterprise_thread, close_enterprise_thread,
    # Turn lookup helpers (message_id-based API)
    get_enterprise_turn_by_id, get_vc_turn_by_id,
    count_agent_enterprise_turns, count_agent_vc_turns_this_year,
    # V2: Discovery system
    get_group_info_level, get_all_group_info_levels, get_discovered_groups,
    get_undiscovered_groups, upgrade_group_info_level,
    # Table documentation registry
    TABLE_DOCS,
    # Config override recording
    record_config_override,
)
from .enterprise import (
    schedule_customer_reply, batch_schedule_customer_replies,
    update_relationship,
    create_negotiation_thread, add_customer_message, generate_enterprise_email,
)
from .vc_negotiation import (
    get_vc_negotiation_state, evaluate_agent_vc_offer, schedule_vc_reply,
    compute_check_size_from_pct, compute_pct_from_check,
)


# =====================================================================
# Helper: render table doc from TABLE_DOCS (for sample_io generation)
# =====================================================================
def _render_table_doc(table_name: str, max_cols: int | None = None) -> str:
    """Render a table's describe_tables output from TABLE_DOCS.

    Generates the same format as describe_tables() so sample_io examples
    stay in sync with the actual schema automatically.

    Args:
        table_name: Name of the table in TABLE_DOCS.
        max_cols: If set, show only the first N columns and append a '...' line.
    """
    doc = TABLE_DOCS[table_name]
    lines = [f"=== {table_name} ===", doc['description'], ""]
    cols = list(doc['columns'].items())
    shown = cols[:max_cols] if max_cols and len(cols) > max_cols else cols
    for col, col_desc in shown:
        lines.append(f"  {col}: {col_desc}")
    if max_cols and len(cols) > max_cols:
        lines.append(f"  ...({len(cols) - max_cols} more columns)")
    return "\n".join(lines)


# =====================================================================
# TOOL_DOCS: Canonical documentation for all agent tools.
# Used by get_tool_documentation() to render tool docs.
# Previously maintained as a separate tool_docs.json file.
# =====================================================================
TOOL_DOCS = {
    "set_prices": {
        "name": "set_prices",
        "category": "Business Configuration",
        "description": "Set monthly subscription prices for plans A, B, and C.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "A": {"type": "number", "description": "Monthly price in $ for Plan A (entry tier)"},
                "B": {"type": "number", "description": "Monthly price in $ for Plan B (mid tier)"},
                "C": {"type": "number", "description": "Monthly price in $ for Plan C (premium tier)"}
            },
            "required": ["A", "B", "C"]
        },
        "parameters": {
            "A": {"type": "float", "description": "Monthly price for plan A (must be positive)"},
            "B": {"type": "float", "description": "Monthly price for plan B (must be positive)"},
            "C": {"type": "float", "description": "Monthly price for plan C (must be positive)"}
        },
        "returns": {
            "success": "Prices updated: A=$29.00, B=$79.00, C=$199.00",
            "failure": "Missing price for plan X / Price for plan X must be positive"
        },
        "output_schema": {
            "updated": "Dict[str, float] — the price changes applied (only keys you sent)",
            "current": "Dict[str, float] — final prices for all plans {'A': float, 'B': float, 'C': float}",
            "_access": "result['current']['A'] → current price of plan A"
        },
        "impact": "Affects customer acquisition (higher prices = fewer sign-ups), churn (price vs value), and revenue. Changes take effect on next_day.",
        "example_call": {
            "tool": "set_prices",
            "arguments": {"A": 25, "B": 69, "C": 179}
        },
        "internal_notes": "Price stored in config_history. Affects Q_required via asymmetric sigmoid: Q_req(price) uses steepness_left (price < c_max/2) or steepness_right (price >= c_max/2). Enterprise customers negotiate off list price.",
        "sample_io": {
            "success": [
                {"label": "Set all three plans", "input": {"A": 25, "B": 69, "C": 179}, "output": "Prices updated: A=$25.00, B=$69.00, C=$179.00"},
                {"label": "Update only plan B", "input": {"B": 89}, "output": "Prices updated: B=$89.00"},
                {"label": "Update two plans", "input": {"A": 19, "C": 149}, "output": "Prices updated: A=$19.00, C=$149.00"}
            ],
            "failure": [
                {"label": "Negative price", "input": {"A": -10}, "output": "Price for plan A must be positive"},
                {"label": "Invalid plan key", "input": {"D": 50}, "output": "Invalid plan keys: {'D'}. Valid: {'A', 'B', 'C'}"},
                {"label": "Empty input", "input": {}, "output": "Must provide at least one plan price"}
            ]
        }
    },

    "set_model_tiers": {
        "name": "set_model_tiers",
        "category": "Business Configuration",
        "description": "Set AI model tiers for plans A, B, and C. Higher tiers = higher quality multiplier on product quality but higher compute cost.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "A": {"type": "integer", "description": "Model tier 1-5 for Plan A"},
                "B": {"type": "integer", "description": "Model tier 1-5 for Plan B"},
                "C": {"type": "integer", "description": "Model tier 1-5 for Plan C"}
            },
            "required": ["A", "B", "C"]
        },
        "parameters": {
            "A": {"type": "int", "description": "Model tier for plan A (1-5)"},
            "B": {"type": "int", "description": "Model tier for plan B (1-5)"},
            "C": {"type": "int", "description": "Model tier for plan C (1-5)"}
        },
        "tier_info": {
            "1": {"cost_per_unit": 0.0003, "quality_multiplier": 0.60, "class": "Flash-Lite/4o-mini"},
            "2": {"cost_per_unit": 0.002, "quality_multiplier": 0.75, "class": "Haiku/Flash"},
            "3": {"cost_per_unit": 0.006, "quality_multiplier": 0.90, "class": "Sonnet/GPT-4o"},
            "4": {"cost_per_unit": 0.012, "quality_multiplier": 1.00, "class": "Opus/GPT-5"},
            "5": {"cost_per_unit": 0.030, "quality_multiplier": 1.10, "class": "o1/o3 reasoning"}
        },
        "returns": {
            "success": "Model tiers updated: A=tier2, B=tier3, C=tier4",
            "failure": "Missing tier for plan X / Tier for plan X must be 1-5"
        },
        "output_schema": {
            "updated": "Dict[str, int] — the tier changes applied (only keys you sent)",
            "current": "Dict[str, int] — final tiers for all plans {'A': int, 'B': int, 'C': int}",
            "_access": "result['current']['B'] → current tier of plan B"
        },
        "impact": "Higher tiers increase customer satisfaction and reduce churn, but increase compute costs. Tiers act as multipliers on product quality (Tier 4 = 1.0×, Tier 5 = 1.1×). delivered_quality = product_quality × tier_multiplier. Higher tiers amplify your R&D and dev spending investments.",
        "example_call": {
            "tool": "set_model_tiers",
            "arguments": {"A": 2, "B": 3, "C": 5},
        },
        "sample_io": {
            "success": [
                {"label": "Set all tiers", "input": {"A": 2, "B": 3, "C": 5}, "output": "Model tiers updated: A=tier2, B=tier3, C=tier5"},
                {"label": "Upgrade only plan C", "input": {"C": 5}, "output": "Model tiers updated: C=tier5"},
                {"label": "Downgrade plan A", "input": {"A": 1}, "output": "Model tiers updated: A=tier1"}
            ],
            "failure": [
                {"label": "Tier out of range", "input": {"A": 0}, "output": "Tier for plan A must be 1-5"},
                {"label": "Tier too high", "input": {"B": 6}, "output": "Tier for plan B must be 1-5"}
            ]
        }
    },

    "set_daily_spend": {
        "name": "set_daily_spend",
        "category": "Marketing & Spend",
        "description": "Set daily spending for advertising, operations, and development.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "advertising": {"type": "number", "description": "Daily $ for ads"},
                "operations": {"type": "number", "description": "Daily $ for ops"},
                "development": {"type": "number", "description": "Daily $ for dev"}
            },
            "required": ["advertising", "operations", "development"]
        },
        "parameters": {
            "advertising": {"type": "float", "description": "Daily advertising budget (non-negative)"},
            "operations": {"type": "float", "description": "Daily operations budget (non-negative)"},
            "development": {"type": "float", "description": "Daily development budget (non-negative)"}
        },
        "returns": {
            "success": "Daily spend updated: advertising=$500, operations=$1000, development=$500",
            "failure": "Missing spend for X / Spend for X cannot be negative"
        },
        "output_schema": {
            "updated": "Dict[str, float] — the spend changes applied (only keys you sent)",
            "current": "Dict[str, float] — final spend {'advertising': float, 'operations': float, 'development': float}",
            "_access": "result['current']['operations'] → current ops spend"
        },
        "impact": {
            "advertising": "Generates new leads. Each channel has a fixed leads-per-$1000 rate per customer group. Use set_ad_channel_spend for channel allocation, set_targeted_ad_spend for per-group targeting.",
            "operations": "CRITICAL: (1) REDUCES OUTAGE PROBABILITY - At $0: ~3% daily outage risk (~1/month). At $500: ~1.1% daily (~3/year). (2) Speeds up issue resolution: mean resolved/day = 1 + 0.01 × spend. WARNING: Without ops spending, frequent outages damage reputation and cause churn!",
            "development": "Dev spending improves product quality (which is then amplified by model tier). Improvement = 0.001 × ln(1 + spend/1000). delivered_quality = product_quality × tier_multiplier. Higher dev spending = faster quality gains that get amplified by your chosen model tier."
        },
        "example_call": {
            "tool": "set_daily_spend",
            "arguments": {"advertising": 800, "operations": 1200, "development": 600}
        },
        "internal_notes": "Ops: outage_prob = 0.03 * exp(-0.002 * ops_spend). Issue resolution: mean_resolved/day = 1 + 0.01 * spend. Dev: quality_improvement = 0.001 * ln(1 + spend/1000). Advertising: each channel has fixed leads_per_1000_dollars per group.",
        "sample_io": {
            "success": [
                {"label": "Set all three budgets", "input": {"advertising": 800, "operations": 1200, "development": 600}, "output": "Daily spend updated: advertising=$800, operations=$1200, development=$600"},
                {"label": "Only increase ops", "input": {"operations": 2000}, "output": "Daily spend updated: operations=$2000"},
                {"label": "Cut ads to zero", "input": {"advertising": 0}, "output": "Daily spend updated: advertising=$0"}
            ],
            "failure": [
                {"label": "Negative spend", "input": {"advertising": -100}, "output": "Spend for advertising cannot be negative"},
                {"label": "Invalid category", "input": {"marketing": 500}, "output": "Invalid spend categories: {'marketing'}. Valid: {'advertising', 'operations', 'development'}"}
            ]
        }
    },

    "set_ad_channel_spend": {
        "name": "set_ad_channel_spend",
        "category": "Marketing & Spend",
        "description": "Set per-channel advertising budget allocation as percentages. Allows targeting specific customer groups.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "social_media": {"type": "number"},
                "search_ads": {"type": "number"},
                "linkedin": {"type": "number"},
                "content_marketing": {"type": "number"},
                "referral_program": {"type": "number"}
            }
        },
        "parameters": {
            "channel_percentages": {
                "type": "Dict[str, float]",
                "description": "Dictionary with channel names and percentage allocations (0.0 to 1.0). Values are normalized to sum to 1.0.",
                "channels": ["social_media", "search_ads", "linkedin", "content_marketing", "referral_program"]
            }
        },
        "channel_info": {
            "social_media": {"description": "Facebook, Instagram, TikTok ads - broad consumer reach"},
            "search_ads": {"description": "Google/Bing search ads - intent-based targeting"},
            "linkedin": {"description": "LinkedIn ads - professional/business audience"},
            "content_marketing": {"description": "SEO, blogs, whitepapers - organic discovery"},
            "referral_program": {"description": "Customer referral incentives - word-of-mouth"}
        },
        "returns": {
            "success": "Ad channel allocation updated (total budget=$500/day):\n  \u2022 Social Media Ads: 30% ($150/day)\n  \u2022 Search Engine Ads: 30% ($150/day)\n  ...",
            "failure": "Invalid channels: {X}. Valid: {...} / At least one channel must have non-zero percentage"
        },
        "output_schema": {
            "allocations": "Dict[str, float] — normalized percentages (0-1) per channel",
            "spend": "Dict[str, float] — actual $/day per channel",
            "total_budget": "float — total advertising budget per day",
            "_access": "result['spend']['linkedin'] → LinkedIn daily spend in $"
        },
        "impact": "Different channels reach different customer groups with varying effectiveness. Use to target specific segments.",
        "example_call": {
            "tool": "set_ad_channel_spend",
            "arguments": {"social_media": 0.2, "search_ads": 0.3, "linkedin": 0.3, "content_marketing": 0.1, "referral_program": 0.1},
        },
        "sample_io": {
            "success": [
                {"label": "Distribute across all channels", "input": {"social_media": 0.2, "search_ads": 0.3, "linkedin": 0.3, "content_marketing": 0.1, "referral_program": 0.1}, "output": "Ad channel allocation updated (total budget=$500/day):\n  • Social Media Ads: 20% ($100/day)\n  • Search Engine Ads: 30% ($150/day)\n  • LinkedIn Ads: 30% ($150/day)\n  • Content Marketing: 10% ($50/day)\n  • Referral Program: 10% ($50/day)"},
                {"label": "Focus on two channels only", "input": {"linkedin": 0.7, "content_marketing": 0.3}, "output": "Ad channel allocation updated (total budget=$500/day):\n  • LinkedIn Ads: 70% ($350/day)\n  • Content Marketing: 30% ($150/day)"},
                {"label": "All budget to one channel", "input": {"search_ads": 1.0}, "output": "Ad channel allocation updated (total budget=$500/day):\n  • Search Engine Ads: 100% ($500/day)"}
            ],
            "failure": [
                {"label": "Invalid channel name", "input": {"tiktok": 0.5, "search_ads": 0.5}, "output": "Invalid channels: {'tiktok'}. Valid: {'social_media', 'search_ads', 'linkedin', 'content_marketing', 'referral_program'}"},
                {"label": "All zeros", "input": {"social_media": 0, "search_ads": 0}, "output": "At least one channel must have non-zero percentage"}
            ]
        }
    },

    "set_targeted_ad_spend": {
        "name": "set_targeted_ad_spend",
        "category": "Marketing & Spend",
        "description": "Set ADDITIONAL per-group per-channel ad spend on top of the overall channel allocation. Allows precise targeting of specific customer groups via specific channels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targeted_spend": {
                    "type": "object",
                    "description": "{channel_id: {group_id: additional_$/day}}",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    }
                }
            },
            "required": ["targeted_spend"]
        },
        "parameters": {
            "targeted_spend": {
                "type": "Dict[str, Dict[str, float]]",
                "description": "Dictionary of {channel_id: {group_id: additional_dollars_per_day}}. This spend is ADDED to the normal channel allocation, not a replacement.",
                "channels": ["social_media", "search_ads", "linkedin", "content_marketing", "referral_program"],
                "groups": "S1-S3, E1-E3, and discovered groups (D_S01-D_S10, D_E01-D_E10)"
            }
        },
        "returns": {
            "success": "Targeted ad spend updated (extra $300/day on top of channel allocation):\n  \u2022 LinkedIn Ads \u2192 E1: +$200/day\n  \u2022 LinkedIn Ads \u2192 E2: +$100/day",
            "failure": "Invalid channels: {X}. Valid: {...} / Invalid group IDs for channel 'X': {Y}"
        },
        "output_schema": {
            "targeted_spend": "Dict[str, Dict[str, float]] — {channel: {group: $/day}}",
            "total_extra_per_day": "float — total additional ad spend per day",
            "_access": "result['targeted_spend']['linkedin']['E1'] → E1's extra LinkedIn spend"
        },
        "impact": "Extra dollars are deducted from cash daily as advertising cost. In lead generation, each (channel, group) pair gets its normal allocation PLUS the targeted amount. Use this to boost acquisition of high-value segments without changing the overall channel split.",
        "example_call": {
            "tool": "set_targeted_ad_spend",
            "arguments": {"targeted_spend": {"linkedin": {"E1": 200, "E2": 100}, "content_marketing": {"S3": 50}}},
        },
        "sample_io": {
            "success": [
                {"label": "Target two groups on LinkedIn", "input": {"targeted_spend": {"linkedin": {"E1": 200, "E2": 100}}}, "output": "Targeted ad spend updated (extra $300/day on top of channel allocation):\n  • LinkedIn Ads → E1: +$200/day\n  • LinkedIn Ads → E2: +$100/day"},
                {"label": "Multi-channel targeting", "input": {"targeted_spend": {"linkedin": {"E1": 200}, "content_marketing": {"S3": 50}, "search_ads": {"D_S01": 100}}}, "output": "Targeted ad spend updated (extra $350/day on top of channel allocation):\n  • LinkedIn Ads → E1: +$200/day\n  • Content Marketing → S3: +$50/day\n  • Search Engine Ads → D_S01: +$100/day"},
                {"label": "Clear all targeting (empty)", "input": {"targeted_spend": {}}, "output": "Targeted ad spend cleared. No additional per-group ad spend."}
            ],
            "failure": [
                {"label": "Invalid channel", "input": {"targeted_spend": {"tiktok": {"S1": 100}}}, "output": "Invalid channels: {'tiktok'}. Valid: {'social_media', 'search_ads', 'linkedin', 'content_marketing', 'referral_program'}"},
                {"label": "Invalid group ID", "input": {"targeted_spend": {"linkedin": {"INVALID": 100}}}, "output": "Invalid group IDs for channel 'linkedin': {'INVALID'}"}
            ]
        }
    },

    "set_targeted_ops_spend": {
        "name": "set_targeted_ops_spend",
        "category": "Marketing & Spend",
        "description": "Set ADDITIONAL per-group operations spending on top of the global ops spend. Adds extra issue resolution capacity for each targeted group.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targeted_spend": {
                    "type": "object",
                    "description": "{group_id: additional_$/day}",
                    "additionalProperties": {"type": "number"}
                }
            },
            "required": ["targeted_spend"]
        },
        "parameters": {
            "targeted_spend": {
                "type": "Dict[str, float]",
                "description": "Dictionary of {group_id: additional_dollars_per_day}. This spend is ADDED to the global ops spend.",
                "groups": "S1-S3, E1-E3, and discovered groups (D_S01-D_S10, D_E01-D_E10)"
            }
        },
        "returns": {
            "success": "Targeted ops spend updated (extra $500/day on top of global ops):\n  \u2022 E1: +$300/day\n  \u2022 E2: +$200/day",
            "failure": "Invalid group IDs: {X}. Valid groups: S1-S3, E1-E3, ..."
        },
        "output_schema": {
            "targeted_spend": "Dict[str, float] — {group_id: $/day}",
            "total_extra_per_day": "float — total additional ops spend per day",
            "_access": "result['targeted_spend']['E1'] → E1's extra ops spend"
        },
        "mechanics": {
            "per_group_resolution": "Each targeted group gets additional resolution capacity (extra_mean_per_day = 0.053 \u00d7 group_spend) on top of the global pool",
            "cost": "Extra dollars are deducted from cash daily as operations cost"
        },
        "impact": "Extra dollars are deducted from cash daily. Each targeted group gets additional issue resolution speed on top of the global resolution pool. Use to provide faster support for high-value segments.",
        "example_call": {
            "tool": "set_targeted_ops_spend",
            "arguments": {"targeted_spend": {"E1": 300, "E2": 200}},
        },
        "sample_io": {
            "success": [
                {"label": "Target two enterprise groups", "input": {"targeted_spend": {"E1": 300, "E2": 200}}, "output": "Targeted ops spend updated (extra $500/day on top of global ops):\n  • E1: +$300/day\n  • E2: +$200/day"},
                {"label": "Single group", "input": {"targeted_spend": {"S1": 100}}, "output": "Targeted ops spend updated (extra $100/day on top of global ops):\n  • S1: +$100/day"},
                {"label": "Clear targeting", "input": {"targeted_spend": {}}, "output": "Targeted ops spend cleared. No additional per-group ops spend."}
            ],
            "failure": [
                {"label": "Invalid group", "input": {"targeted_spend": {"INVALID": 100}}, "output": "Invalid group IDs: {'INVALID'}. Valid groups: S1, S2, S3, E1, E2, E3, ..."}
            ]
        }
    },

    "set_targeted_dev_spend": {
        "name": "set_targeted_dev_spend",
        "category": "Marketing & Spend",
        "description": "Set ADDITIONAL per-group development spending on top of the global dev spend. Provides a CUMULATIVE per-group quality bonus that grows daily while spending continues. Investment persists even after spending stops.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targeted_spend": {
                    "type": "object",
                    "description": "{group_id: additional_$/day}",
                    "additionalProperties": {"type": "number"}
                }
            },
            "required": ["targeted_spend"]
        },
        "parameters": {
            "targeted_spend": {
                "type": "Dict[str, float]",
                "description": "Dictionary of {group_id: additional_dollars_per_day}. This spend is ADDED to the global dev spend.",
                "groups": "S1-S3, E1-E3, and discovered groups (D_S01-D_S10, D_E01-D_E10)"
            }
        },
        "returns": {
            "success": "Targeted dev spend updated (extra $700/day on top of global dev):\n  \u2022 E1: +$500/day\n  \u2022 S1: +$200/day",
            "failure": "Invalid group IDs: {X}. Valid groups: S1-S3, E1-E3, ..."
        },
        "output_schema": {
            "targeted_spend": "Dict[str, float] — {group_id: $/day}",
            "total_extra_per_day": "float — total additional dev spend per day",
            "_access": "result['targeted_spend']['E1'] → E1's extra dev spend"
        },
        "mechanics": {
            "quality_bonus": "Per-group quality bonus ACCUMULATES daily: +0.0005 \u00d7 log(1 + spend/500) per day. At $500/day: +0.00035/day cumulative. After 30 days: +0.0105 total. Investment persists if spending stops.",
            "scope": "Only affects subscribers in the targeted group (not global q_shared). Like building group-specific features — investment compounds over time.",
            "cost": "Extra dollars are deducted from cash daily as development cost"
        },
        "impact": "Extra dollars are deducted from cash daily. Each targeted group ACCUMULATES a quality bonus over time (like building features for that segment). The bonus persists even after spending stops. Use to invest in features/customization for high-value segments.",
        "example_call": {
            "tool": "set_targeted_dev_spend",
            "arguments": {"targeted_spend": {"E1": 500, "S1": 200}},
        },
        "sample_io": {
            "success": [
                {"label": "Target high-value segments", "input": {"targeted_spend": {"E1": 500, "S1": 200}}, "output": "Targeted dev spend updated (extra $700/day on top of global dev):\n  • E1: +$500/day\n  • S1: +$200/day"},
                {"label": "Single group", "input": {"targeted_spend": {"D_E01": 300}}, "output": "Targeted dev spend updated (extra $300/day on top of global dev):\n  • D_E01: +$300/day"}
            ],
            "failure": [
                {"label": "Invalid group", "input": {"targeted_spend": {"ZZ": 100}}, "output": "Invalid group IDs: {'ZZ'}. Valid groups: S1, S2, S3, E1, E2, E3, ..."}
            ]
        }
    },

    "set_capacity_tier": {
        "name": "set_capacity_tier",
        "category": "Business Configuration",
        "description": "Set infrastructure capacity tier. Higher tiers handle more usage but cost more per day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "integer", "description": "Capacity tier 0-7"}
            },
            "required": ["tier"]
        },
        "parameters": {
            "tier": {
                "type": "int",
                "description": "Capacity tier (0-7)",
                "example": 1
            }
        },
        "tier_info": {
            "0": {"capacity_units": 50000, "cost_per_day": 85, "description": "Serverless API (Together/Fireworks)"},
            "1": {"capacity_units": 200000, "cost_per_day": 215, "description": "1x H100 neocloud dedicated"},
            "2": {"capacity_units": 800000, "cost_per_day": 530, "description": "4x H100 reserved cluster"},
            "3": {"capacity_units": 2500000, "cost_per_day": 1330, "description": "8x H100 enterprise + auto-scaling"},
            "4": {"capacity_units": 8000000, "cost_per_day": 4000, "description": "Multi-node hyperscale (16-32 H100s)"},
            "5": {"capacity_units": 25000000, "cost_per_day": 10000, "description": "64x H100 multi-rack cluster"},
            "6": {"capacity_units": 80000000, "cost_per_day": 28000, "description": "256x H100 dedicated pod"},
            "7": {"capacity_units": 300000000, "cost_per_day": 75000, "description": "1024+ GPU hyperscale fleet"}
        },
        "returns": {
            "success": "Capacity tier set to 1: 200,000 units/day, $215/day",
            "failure": "Capacity tier must be 0-7. Use get_cost_info to see all tiers."
        },
        "output_schema": {
            "tier": "int — selected tier (0-7)",
            "capacity_units": "int — units/day capacity",
            "cost_per_day": "float — daily cost in $",
            "_access": "result['capacity_units'] → capacity units per day"
        },
        "impact": "When usage exceeds capacity, overload occurs causing higher latency and errors. Higher overload increases outage chance. Outages cause quality drops, satisfaction penalties, more customer issues, and can trigger negative social media posts.",
        "example_call": {
            "tool": "set_capacity_tier",
            "arguments": {"tier": 2}
        },
        "internal_notes": "Overload = max(0, total_usage / capacity_units - 1). Overload > 0 → p95_ms increases, error_rate increases. Outage_prob_from_overload = 0.1 * overload^2. Outage causes: quality_penalty = -0.05, satisfaction_penalty = -0.1 for all customers, 3-5 new issues generated, possible negative social posts.",
        "sample_io": {
            "success": [
                {"label": "Set tier 2", "input": {"tier": 2}, "output": "Capacity tier set to 2: 800,000 units/day ($530/day) — 4x H100 reserved cluster"},
                {"label": "Downgrade to serverless", "input": {"tier": 0}, "output": "Capacity tier set to 0: 50,000 units/day ($85/day) — Serverless API (Together/Fireworks)"},
                {"label": "Max tier", "input": {"tier": 7}, "output": "Capacity tier set to 7: 300,000,000 units/day ($75,000/day) — 1024+ GPU hyperscale fleet"}
            ],
            "failure": [
                {"label": "Tier out of range", "input": {"tier": 10}, "output": "Capacity tier must be 0-7. Use get_cost_info to see all tiers."},
                {"label": "Negative tier", "input": {"tier": -1}, "output": "Capacity tier must be 0-7. Use get_cost_info to see all tiers."}
            ]
        }
    },

    "set_usage_quotas": {
        "name": "set_usage_quotas",
        "category": "Business Configuration",
        "description": "Set daily usage quotas (rate limits) per customer for each plan. Exceeding quota degrades experience.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "A": {"type": "integer", "description": "Daily usage quota for Plan A"},
                "B": {"type": "integer", "description": "Daily usage quota for Plan B"},
                "C": {"type": "integer", "description": "Daily usage quota for Plan C"}
            },
            "required": ["A", "B", "C"]
        },
        "parameters": {
            "A": {"type": "int", "description": "Daily usage quota for plan A (units/day per customer)"},
            "B": {"type": "int", "description": "Daily usage quota for plan B (units/day per customer)"},
            "C": {"type": "int", "description": "Daily usage quota for plan C (units/day per customer)"}
        },
        "returns": {
            "success": "Usage quotas updated: A=100 units/day, B=500 units/day, C=2,000 units/day",
            "failure": "Missing quota for plan X / Quota for plan X cannot be negative"
        },
        "output_schema": {
            "quotas": "Dict[str, int] — {'A': int, 'B': int, 'C': int} units/day per customer",
            "_access": "result['quotas']['A'] → plan A daily quota"
        },
        "impact": "Quotas limit per-customer usage to control costs. Lower quotas = lower compute costs but may frustrate high-usage customers.",
        "example_call": {
            "tool": "set_usage_quotas",
            "arguments": {"A": 150, "B": 750, "C": 3000},
        },
        "sample_io": {
            "success": [
                {"label": "Set all quotas", "input": {"A": 150, "B": 750, "C": 3000}, "output": "Usage quotas updated: A=150 units/day, B=750 units/day, C=3,000 units/day"},
                {"label": "Only raise plan C quota", "input": {"C": 5000}, "output": "Usage quotas updated: C=5,000 units/day"},
                {"label": "Tighten plan A", "input": {"A": 50}, "output": "Usage quotas updated: A=50 units/day"}
            ],
            "failure": [
                {"label": "Negative quota", "input": {"A": -50}, "output": "Quota for plan A cannot be negative"},
                {"label": "Invalid plan key", "input": {"D": 100}, "output": "Invalid plan keys: {'D'}. Valid: {'A', 'B', 'C'}"}
            ]
        }
    },

    "send_enterprise_deal": {
        "name": "send_enterprise_deal",
        "category": "Customer Communication",
        "description": "Send enterprise deal offerings. Compact tuple format: each deal = [customer_id, [[plan, price_per_seat, contract_months], ...]]. If the customer has an open negotiation thread, replies to it. If no open thread, initiates renegotiation. Up to 3 offerings per deal. Customer picks the best. Late replies damage relationship (-0.02/day after 1 day grace). No response within 3 days = customer LOST FOREVER.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deals": {
                    "type": "array",
                    "description": "List of deals. Each deal = [customer_id, [[plan, price_per_seat, contract_months], ...]]",
                    "items": {
                        "type": "array",
                        "description": "[customer_id, offerings] where offerings = [[plan, price, months], ...]"
                    }
                }
            },
            "required": ["deals"]
        },
        "parameters": {
            "deals": {
                "type": "list[list]",
                "description": "List of [customer_id, offerings] tuples. offerings = list of [plan, price_per_seat, contract_months] tuples (contract_months defaults to 1 if omitted). If customer has an open thread, replies to it; otherwise initiates renegotiation.",
                "example": [[312, [["A", 9.00, 6], ["B", 14.00, 12]]], [88, [["B", 12.00, 6]]]]
            }
        },
        "returns": {
            "success": "Processed 2/2 deals:\n  Customer #312: reply sent with 2 offering(s)\n  Customer #88: renegotiation initiated, 2 offering(s) sent",
            "failure": "Customer #312: not found / Customer #88: already has an active thread / offerings required"
        },
        "output_schema": {
            "results": "List[Dict] — one dict per deal with keys: customer_id (int), success (bool), status (str), error (str, if failed)",
            "_access": "for r in result['results']: print(r['customer_id'], r['status'])"
        },
        "impact": "Customer evaluates ALL offerings and picks the one with highest satisfaction. Satisfaction = quality_perceived - quality_required(price) - contract_penalty. Contract lock-in penalty varies per customer group (e.g. price-sensitive individuals ~0.8%/month, strategic enterprises ~0.2%/month — longer contracts penalize satisfaction, offset with lower prices). Customer accepts if best satisfaction > 0, counter-offers otherwise. Max negotiation turns = customer ghosts. Late replies (>1 day) damage relationship -0.02/day. No response within 3 days = customer permanently lost. Renegotiation (no open thread): creates new thread. WARNING: if the customer rejects all offerings OR the thread times out, the customer CHURNS (cancels subscription).",
        "example_call": {
            "tool": "send_enterprise_deal",
            "arguments": {
                "deals": [[312, [["A", 9.00, 6], ["B", 14.00, 12]]], [88, [["B", 12.00, 6]]]]
            }
        },
        "internal_notes": "Accepts tuple format [cid, [[plan,price,months],...]] or legacy dict format. If customer has open thread, replies (with late-reply penalty). If no open thread, initiates renegotiation (requires active subscription). Satisfaction = Q_perceived - Q_required(price) - contract_lockin_penalty * (months-1). Lock-in penalty is per-customer (sampled from group distribution at creation: S1=0.008, S2=0.005, S3=0.006, E1=0.005, E2=0.003, E3=0.002). Late penalty = -0.02 * max(0, days_since_msg - 1). Max offerings = 3.",
        "sample_io": {
            "success": [
                {"label": "Reply to open thread with 3 offerings", "input": {"deals": [[312, [["A", 9.0, 6], ["B", 14.0, 12], ["C", 22.0, 12]]]]}, "output": "Customer #312: reply sent with 3 offering(s)"},
                {"label": "Initiate renegotiation (no open thread)", "input": {"deals": [[88, [["B", 12.0, 6], ["B", 11.0, 12]]]]}, "output": "Customer #88: renegotiation started (200 seats). 2 offering(s) sent."},
                {"label": "Batch", "input": {"deals": [[312, [["A", 9.0, 6]]], [88, [["B", 12.0, 6]]]]}, "output": "Sent 2/2 enterprise deals:\n  Customer #312: reply sent with 1 offering(s)\n  Customer #88: renegotiation started. 1 offering(s) sent."}
            ],
            "failure": [
                {"label": "Customer not found", "input": {"deals": [[999, [["A", 9.0, 6]]]]}, "output": "Customer #999: not found"},
                {"label": "Missing offerings", "input": {"deals": [[312, []]]}, "output": "Customer #312: offerings required"},
                {"label": "Customer has active thread", "input": {"deals": [[88, [["B", 11.0, 6]]]]}, "output": "Processed 0/1 deals (1 failed):\n  Customer #88: already has an active negotiation thread"}
            ]
        }
    },

    "python_exec": {
        "name": "python_exec",
        "category": "Analytics & Monitoring",
        "description": "Execute Python code for custom data analysis. Has read-only access to the full simulation database. This is your primary analytics tool for any analysis not covered by other tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code. Use pre-loaded 'conn' variable for database queries. Do NOT create your own connection."}
            },
            "required": ["code"]
        },
        "parameters": {
            "code": {
                "type": "str",
                "description": "Python code to execute. Use print() to see output. Has access to conn, rows(), row(), pandas, numpy, sklearn.",
                "example": "print(row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\"'))"
            },
            "timeout_seconds": {
                "type": "float",
                "description": "Maximum execution time (default 5.0 seconds)",
                "example": 5.0
            }
        },
        "available_in_code": {
            "conn": "SQLite connection (read-only) with row_factory=sqlite3.Row",
            "rows(query, params)": "Execute query, return list of tuples. Example: rows('SELECT * FROM customers WHERE group_id=?', ('S1',))",
            "row(query, params)": "Execute query, return single tuple or None. Example: row('SELECT COUNT(*) FROM subscriptions')[0]",
            "pandas": "import pandas as pd - use pd.read_sql(query, conn) for DataFrames",
            "numpy": "import numpy as np",
            "sklearn": "LinearRegression, StandardScaler from sklearn",
            "json": "import json",
            "math": "import math",
            "statistics": "import statistics",
            "Counter": "from collections import Counter",
            "defaultdict": "from collections import defaultdict"
        },
        "database_tables": {
            "note": "Use describe_tables() to get detailed column descriptions for any table. Schema introspection queries (PRAGMA, sqlite_master) are blocked.",
            "tables": {
                "customers": "All customers (small and enterprise). JOIN with subscriptions for plan/status.",
                "subscriptions": "Subscription records \u2014 plan, price, status, dates. THIS is where plan and status live.",
                "daily_usage": "Per-customer daily usage records (day, customer_id, usage_units).",
                "ledger": "Financial ledger \u2014 all income and expenses (positive=income, negative=cost).",
                "service_day": "Daily service metrics \u2014 usage, latency, errors, downtime, capacity.",
                "config_history": "Daily snapshot of all agent-configurable settings.",
                "social_media_posts": "Customer social media posts. Sentiment is HIDDEN \u2014 must infer from content.",
                "enterprise_turns": "Enterprise customer negotiation turns. Each row is one turn in a conversation thread.",
                "notifications": "Agent inbox \u2014 simple string notifications. Columns: notification_id, day, type, message.",
                "shareholders": "Cap table \u2014 founder and VC investors.",
                "funding_rounds": "Completed VC investment settlements.",
                "vc_turns": "VC negotiation turns. Each row is one turn in a VC negotiation thread.",
                "dividends": "Dividend payment history.",
                "research_projects": "R&D research tiers (20 independent tiers, available/in-progress/completed).",
                "ad_channel_leads": "Advertising channel effectiveness history.",
                "group_info_levels": "Customer group discovery and research levels.",
                "issues": "Individual customer support issues with lifecycle tracking. Columns: issue_id, customer_id, group_id, open_day, days_open, status ('open'/'resolved'), resolved_day, resolution_type. Query to analyze issue patterns by group, resolution speed, and operational effectiveness."
            }
        },
        "example_queries": {
            "subscriber_metrics": [
                "# Active subscriber count",
                "row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0]",
                "",
                "# Subscribers by plan",
                "rows('SELECT plan, COUNT(*) as cnt FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL GROUP BY plan')",
                "",
                "# Subscribers by customer group",
                "rows('SELECT c.group_id, COUNT(*) FROM subscriptions s JOIN customers c ON s.customer_id=c.customer_id WHERE s.status=\"subscribed\" AND s.end_day IS NULL GROUP BY c.group_id')"
            ],
            "revenue_analysis": [
                "# Current MRR",
                "row('SELECT SUM(effective_price) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0]",
                "",
                "# Revenue by day (last 7 days)",
                "rows('SELECT day, SUM(amount) as rev FROM ledger WHERE category=\"subscription_payment\" AND day > (SELECT MAX(day)-7 FROM ledger) GROUP BY day ORDER BY day')",
                "",
                "# Revenue by category",
                "rows('SELECT category, SUM(amount) FROM ledger GROUP BY category ORDER BY SUM(amount)')"
            ],
            "churn_analysis": [
                "# Recent cancellations",
                "rows('SELECT customer_id, plan, end_day FROM subscriptions WHERE status=\"cancelled\" ORDER BY end_day DESC LIMIT 10')",
                "",
                "# 30-day churn rate",
                "total = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\"')[0]",
                "churned = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"cancelled\" AND end_day > (SELECT MAX(day)-30 FROM service_day)')[0]",
                "print(f'Churn: {churned}/{total} = {churned/total*100:.1f}%')"
            ],
            "service_metrics": [
                "# Recent service quality",
                "rows('SELECT day, total_usage_units, p95_ms, error_rate, downtime_minutes FROM service_day ORDER BY day DESC LIMIT 7')",
                "",
                "# Capacity utilization",
                "row('SELECT total_usage_units * 1.0 / capacity_units as utilization FROM service_day ORDER BY day DESC LIMIT 1')[0]"
            ],
            "enterprise_threads": [
                "# Open negotiation threads (latest turn per thread)",
                "rows('SELECT et.thread_id, et.closed, et.close_reason, et.turn_number, et.seat_count, c.email FROM enterprise_turns et JOIN customers c ON et.customer_id=c.customer_id WHERE et.message_id = (SELECT MAX(et2.message_id) FROM enterprise_turns et2 WHERE et2.thread_id=et.thread_id) AND et.closed = 0 AND et._internal_status IS NULL')",
                "",
                "# Thread turn history",
                "rows('SELECT message_id, turn_number, day, sender, message_text, offer_json, seat_count, closed, close_reason FROM enterprise_turns WHERE thread_id=? ORDER BY turn_number', (thread_id,))"
            ],
            "social_posts": [
                "# Recent social media posts (NOTE: sentiment column is hidden from agent)",
                "rows('SELECT day, content, likes, shares, virality_score FROM social_media_posts WHERE day > (SELECT MAX(day)-7 FROM social_media_posts) ORDER BY day DESC LIMIT 10')",
                "",
                "# High-engagement posts (you must infer sentiment from content)",
                "rows('SELECT day, content, likes, shares FROM social_media_posts WHERE virality_score > 0.5 ORDER BY day DESC LIMIT 5')"
            ],
            "equity_analysis": [
                "# Cap table",
                "rows('SELECT name, shareholder_type, shares_held, shares_held * 100.0 / (SELECT SUM(shares_held) FROM shareholders) as pct FROM shareholders ORDER BY shares_held DESC')",
                "",
                "# Funding rounds",
                "rows('SELECT fr.day, s.name, fr.shares_issued, fr.price_per_share, fr.total_amount FROM funding_rounds fr JOIN shareholders s ON fr.investor_shareholder_id=s.shareholder_id ORDER BY fr.day')",
                "",
                "# Total dividends paid",
                "row('SELECT SUM(total_amount) FROM dividends')[0]",
                "",
                "# Active VC negotiations (latest turn per VC)",
                "rows('SELECT vt.shareholder_id, s.name, vt.closed, vt.close_reason, vt.current_offer_share_pct, vt.current_offer_amount, vt.expiry_day FROM vc_turns vt JOIN shareholders s ON vt.shareholder_id=s.shareholder_id WHERE vt.message_id = (SELECT MAX(vt2.message_id) FROM vc_turns vt2 WHERE vt2.shareholder_id=vt.shareholder_id) AND vt.closed = 0 AND vt._internal_status IS NULL')",
                "",
                "# Accepted deals awaiting settlement",
                "rows('SELECT vt.shareholder_id, s.name, vt.current_offer_share_pct, vt.current_offer_amount, vt.expiry_day FROM vc_turns vt JOIN shareholders s ON vt.shareholder_id=s.shareholder_id WHERE vt.message_id = (SELECT MAX(vt2.message_id) FROM vc_turns vt2 WHERE vt2.shareholder_id=vt.shareholder_id) AND vt.close_reason=\"accepted\"')"
            ],
            "pandas_examples": [
                "# Load data into DataFrame",
                "df = pd.read_sql('SELECT * FROM service_day ORDER BY day DESC LIMIT 30', conn)",
                "print(df.describe())",
                "",
                "# Revenue trend with pandas",
                "df = pd.read_sql('SELECT day, SUM(amount) as rev FROM ledger WHERE category=\"subscription_payment\" GROUP BY day', conn)",
                "print(f'Avg daily revenue: ${df[\"rev\"].mean():,.0f}')"
            ],
            "acquisition_analysis": [
                "# Customers by acquisition source",
                "rows('SELECT acquisition_source, COUNT(*) as cnt FROM customers GROUP BY acquisition_source ORDER BY cnt DESC')",
                "",
                "# Active subscribers by acquisition source",
                "rows('SELECT c.acquisition_source, COUNT(*) as subs, SUM(s.effective_price) as mrr FROM customers c JOIN subscriptions s ON c.customer_id=s.customer_id WHERE s.status=\"subscribed\" AND s.end_day IS NULL GROUP BY c.acquisition_source ORDER BY mrr DESC')",
                "",
                "# Which ad channels bring the most valuable customers?",
                "rows('SELECT c.acquisition_source, AVG(s.effective_price) as avg_price, COUNT(*) as cnt FROM customers c JOIN subscriptions s ON c.customer_id=s.customer_id WHERE s.status=\"subscribed\" AND s.end_day IS NULL GROUP BY c.acquisition_source HAVING cnt > 5 ORDER BY avg_price DESC')",
                "",
                "# Retention by acquisition source (lower cancelled ratio = better)",
                "rows('SELECT c.acquisition_source, SUM(CASE WHEN s.status=\"cancelled\" THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as cancel_pct FROM customers c JOIN subscriptions s ON c.customer_id=s.customer_id GROUP BY c.acquisition_source ORDER BY cancel_pct')"
            ]
        },
        "returns": {
            "success": "Output of print statements (truncated to 5000 chars)",
            "failure": "Execution error: [error message] / Execution timed out after X seconds"
        },
        "output_schema": {
            "_note": "Returns stdout from print() statements as a string. No structured data — use print() to output results.",
            "_access": "result is the printed text output"
        },
        "impact": "Read-only analysis. No side effects on simulation state.",
        "important_notes": [
            "STATELESS: Each python_exec() call runs in a FRESH context. Variables, imports, and DataFrames from previous calls are NOT available. You must re-query data in each call.",
            "The pre-loaded variables (conn, rows, row, pd, np, etc.) are available in every call, but any variables YOU define do not persist between calls."
        ],
        "tips": [
            "Use row() for single values, rows() for multiple rows",
            "Use pd.read_sql() for complex analysis with pandas",
            "Always filter active subscriptions with: status='subscribed' AND end_day IS NULL",
            "Join customers and subscriptions to get customer details with subscription info",
            "Use describe_tables() to get column details for any table"
        ],
        "example_call": {
            "tool": "python_exec",
            "arguments": {
                "code": "# Comprehensive business health check\nprint('=== Business Health ===')\n\n# Subscribers\nsubs = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0]\nmrr = row('SELECT SUM(effective_price) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0] or 0\nprint(f'Subscribers: {subs}')\nprint(f'MRR: ${mrr:,.0f}')\n\n# By plan\nprint('\\nBy Plan:')\nfor plan, cnt in rows('SELECT plan, COUNT(*) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL GROUP BY plan'):\n    print(f'  {plan}: {cnt}')\n\n# Cash balance\ncash = row('SELECT SUM(amount) FROM ledger')[0] or 0\nprint(f'\\nCash: ${cash:,.0f}')"
            },
        },
        "internal_notes": "Hidden columns (_HIDDEN_COLUMNS) are stripped from query results at runtime. Schema introspection (PRAGMA, sqlite_master) is blocked. _HIDDEN_TABLES can't be queried. pandas DataFrames also have hidden columns dropped before display.",
        "sample_io": {
            "success": [
                {"label": "Subscriber count", "input": {"code": "print(row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0])"}, "output": "145"},
                {"label": "Revenue by plan", "input": {"code": "for plan, cnt, mrr in rows('SELECT plan, COUNT(*), SUM(effective_price) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL GROUP BY plan'):\n    print(f'{plan}: {cnt} subs, ${mrr:,.0f} MRR')"}, "output": "A: 82 subs, $2,378 MRR\nB: 48 subs, $3,792 MRR\nC: 15 subs, $2,985 MRR"},
                {"label": "30-day churn rate", "input": {"code": "total = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\"')[0]\nchurned = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"cancelled\" AND end_day > (SELECT MAX(day)-30 FROM service_day)')[0]\nprint(f'Churn: {churned}/{total} = {churned/total*100:.1f}%')"}, "output": "Churn: 12/145 = 8.3%"},
                {"label": "Pandas DataFrame analysis", "input": {"code": "df = pd.read_sql('SELECT day, SUM(amount) as rev FROM ledger WHERE category=\"subscription_payment\" AND day > (SELECT MAX(day)-7 FROM ledger) GROUP BY day', conn)\nprint(f'7-day revenue: ${df[\"rev\"].sum():,.0f}')\nprint(f'Avg daily: ${df[\"rev\"].mean():,.0f}')"}, "output": "7-day revenue: $2,891\nAvg daily: $413"},
                {"label": "Open enterprise negotiations", "input": {"code": "for cid, ttype, seats, email in rows('SELECT et.customer_id, et.thread_type, CAST(c.seat_count AS INTEGER) as seat_count, c.email FROM enterprise_turns et JOIN customers c ON et.customer_id=c.customer_id WHERE et.message_id = (SELECT MAX(et2.message_id) FROM enterprise_turns et2 WHERE et2.thread_id=et.thread_id) AND et.closed = 0'):\n    print(f'Customer #{cid}: {ttype} ({seats} seats, {email})')"}, "output": "Customer #312: new_lead (200 seats, ops@techcorp.com)\nCustomer #88: churn_prevention (50 seats, cfo@startupinc.com)"}
            ],
            "failure": [
                {"label": "Schema introspection blocked", "input": {"code": "rows('PRAGMA table_info(customers)')"}, "output": "Execution error: Schema introspection queries (PRAGMA, sqlite_master) are not allowed. Use describe_tables() instead."},
                {"label": "Syntax error", "input": {"code": "print('hello"}, "output": "Execution error: unterminated string literal (detected at line 1)"},
                {"label": "Timeout", "input": {"code": "import time; time.sleep(600)"}, "output": "Execution timed out after 5.0 seconds"}
            ]
        }
    },

    "register_daily_calculation": {
        "name": "register_daily_calculation",
        "category": "Automation",
        "description": "Register a named calculation to run automatically at the start of each day. Output appears in dashboard.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique name for the calculation"},
                "code": {"type": "string", "description": "Python code to execute (same environment as python_exec)"}
            },
            "required": ["name", "code"]
        },
        "parameters": {
            "name": {
                "type": "str",
                "description": "Unique name for the calculation",
                "example": "churn_rate"
            },
            "code": {
                "type": "str",
                "description": "Python code to execute (same environment as python_exec)",
                "example": "total = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\"')[0]\nchurned = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"cancelled\" AND end_day > (SELECT MAX(day)-30 FROM service_day)')[0]\nprint(f'30-day churn: {churned}/{total} = {churned/total*100:.1f}%')"
            }
        },
        "returns": {
            "success": "Registered daily calculation: 'churn_rate'. It will run at the start of each day.",
            "failure": None
        },
        "output_schema": {
            "name": "str — registered calculation name",
            "code_length": "int — length of code in chars",
            "_access": "result['name'] → name of registered calculation"
        },
        "impact": "Calculation runs each day before dashboard is shown. Use for automated KPI tracking.",
        "example_call": {
            "tool": "register_daily_calculation",
            "arguments": {
                "name": "revenue_trend",
                "code": "import pandas as pd\ndf = pd.read_sql('SELECT day, SUM(amount) as rev FROM ledger WHERE category=\"subscription_payment\" AND day > (SELECT MAX(day)-7 FROM ledger) GROUP BY day', conn)\nprint(f'7-day revenue: ${df[\"rev\"].sum():,.0f}')",
            }
        },
        "sample_io": {
            "success": [
                {"label": "Register churn tracker", "input": {"name": "churn_rate", "code": "total = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"subscribed\"')[0]\nchurned = row('SELECT COUNT(*) FROM subscriptions WHERE status=\"cancelled\" AND end_day > (SELECT MAX(day)-30 FROM service_day)')[0]\nprint(f'30-day churn: {churned}/{total} = {churned/total*100:.1f}%')"}, "output": "Registered daily calculation: 'churn_rate'. It will run at the start of each day."},
                {"label": "Register MRR tracker", "input": {"name": "mrr_tracker", "code": "mrr = row('SELECT SUM(effective_price) FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL')[0] or 0\nprint(f'MRR: ${mrr:,.0f}')"}, "output": "Registered daily calculation: 'mrr_tracker'. It will run at the start of each day."}
            ],
            "failure": [
                {"label": "Empty name", "input": {"name": "", "code": "print('test')"}, "output": "Calculation name cannot be empty"}
            ]
        }
    },

    "remove_daily_calculation": {
        "name": "remove_daily_calculation",
        "category": "Automation",
        "description": "Remove a registered daily calculation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the calculation to remove"}
            },
            "required": ["name"]
        },
        "parameters": {
            "name": {
                "type": "str",
                "description": "Name of the calculation to remove",
                "example": "churn_rate"
            }
        },
        "returns": {
            "success": "Removed daily calculation: 'churn_rate'",
            "failure": "Calculation 'X' not found. Registered calculations: [...]"
        },
        "output_schema": {
            "name": "str — removed calculation name",
            "remaining": "List[str] — remaining calculation names",
            "_access": "result['remaining'] → list of still-registered calculations"
        },
        "impact": "Calculation will no longer run or appear in dashboard.",
        "example_call": {
            "tool": "remove_daily_calculation",
            "arguments": {"name": "churn_rate"},
        },
        "sample_io": {
            "success": [
                {"label": "Remove existing calc", "input": {"name": "churn_rate"}, "output": "Removed daily calculation: 'churn_rate'"}
            ],
            "failure": [
                {"label": "Name not found", "input": {"name": "nonexistent"}, "output": "Calculation 'nonexistent' not found. Registered calculations: ['revenue_trend', 'subscriber_count']"}
            ]
        }
    },

    "list_daily_calculations": {
        "name": "list_daily_calculations",
        "category": "Automation",
        "description": "List all registered daily calculations.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "success": "Registered daily calculations:\n  \u2022 churn_rate: total = row('SELECT COUNT(*)...\n  \u2022 revenue_trend: import pandas as pd...",
            "empty": "No daily calculations registered."
        },
        "output_schema": {
            "calculations": "List[str] — names of registered calculations",
            "_access": "for name in result['calculations']: print(name)"
        },
        "impact": "Read-only. Shows what calculations will run each day.",
        "example_call": {
            "tool": "list_daily_calculations",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "With registered calcs", "input": {}, "output": "Registered daily calculations:\n  • churn_rate: total = row('SELECT COUNT(*)...\n  • revenue_trend: import pandas as pd..."},
                {"label": "No calcs registered", "input": {}, "output": "No daily calculations registered."}
            ]
        }
    },

    "get_social_posts": {
        "name": "get_social_posts",
        "category": "Analytics & Monitoring",
        "description": "Search social media posts about your company. NOTE: Sentiment is NOT provided - you must infer it from the post content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 7, "description": "Days back to search"},
                "limit": {"type": "integer", "default": 50, "description": "Max posts to return"}
            }
        },
        "parameters": {
            "days": {
                "type": "int",
                "description": "How many days back to search (default 7)",
                "example": 7
            },
            "limit": {
                "type": "int",
                "description": "Maximum posts to return (default 50)",
                "example": 50
            }
        },
        "returns": {
            "success": {
                "message": "Found 23 posts in last 7 days.\nDay 45: \"The service was down for 2 hours yesterday...\" (15 likes, 3 shares)\nDay 44: \"Love how fast the API responds now!\" (8 likes, 1 share)",
                "data": {
                    "posts": [{"day": 45, "content": "The service was down...", "likes": 15, "shares": 3, "virality_score": 0.3}],
                    "total": 23
                }
            },
            "failure": "Invalid parameters"
        },
        "output_schema": {
            "posts": "List[Dict] — each post has: day (int), content (str), likes (int), shares (int), virality_score (float), customer_id (int), group_id (str), customer_type (str), custom_name (str), persona_name (str)",
            "total": "int — total number of posts found",
            "_access": "for post in result['posts']: print(post['day'], post['content'])",
            "_warning": "result is a dict with 'posts' key — do NOT iterate result directly, iterate result['posts']"
        },
        "impact": "Read-only. Use to monitor what customers are saying. You must analyze the post content yourself to determine sentiment.",
        "example_call": {
            "tool": "get_social_posts",
            "arguments": {"days": 7},
        },
        "sample_io": {
            "success": [
                {"label": "Last 7 days", "input": {"days": 7}, "output": "Found 23 posts in last 7 days.\nDay 45: \"Absolutely loving the new features! The AI quality has improved dramatically. 10/10 would recommend.\" (15 likes, 3 shares, virality: 0.31)\nDay 44: \"Service was down for 2 hours yesterday. Frustrating when you're on a deadline.\" (8 likes, 1 share, virality: 0.12)\nDay 43: \"Good tool but getting pricey. Considering alternatives.\" (4 likes, 0 shares, virality: 0.05)"},
                {"label": "Last 1 day with limit", "input": {"days": 1, "limit": 5}, "output": "Found 3 posts in last 1 days.\nDay 45: \"Great uptime today!\" (2 likes, 0 shares, virality: 0.02)\nDay 45: \"Just started using this, so far so good\" (1 likes, 0 shares, virality: 0.01)\nDay 45: \"Pricing seems steep for a small team\" (5 likes, 1 share, virality: 0.08)"},
                {"label": "Last 30 days", "input": {"days": 30, "limit": 50}, "output": "Found 50 posts in last 30 days (showing first 50).\nDay 45: \"Absolutely loving...\" (15 likes, 3 shares, virality: 0.31)\n...48 more posts..."}
            ],
            "failure": [
                {"label": "Negative days", "input": {"days": -1}, "output": "Days must be a positive integer"}
            ]
        }
    },

    "get_cost_info": {
        "name": "get_cost_info",
        "category": "Analytics & Monitoring",
        "description": "Get current cost structure for compute and capacity. Shows model tier costs and capacity tier costs.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "success": {
                "model_tiers": {
                    "1": {"cost_per_usage_unit": 0.0003, "quality_multiplier": 0.60, "class": "Flash-Lite/4o-mini"},
                    "2": {"cost_per_usage_unit": 0.002, "quality_multiplier": 0.75, "class": "Haiku/Flash"},
                    "3": {"cost_per_usage_unit": 0.006, "quality_multiplier": 0.90, "class": "Sonnet/GPT-4o"},
                    "4": {"cost_per_usage_unit": 0.012, "quality_multiplier": 1.00, "class": "Opus/GPT-5"},
                    "5": {"cost_per_usage_unit": 0.030, "quality_multiplier": 1.10, "class": "o1/o3 reasoning"}
                },
                "capacity_tiers": {
                    "0": {"capacity_units": 50000, "cost_per_day": 85},
                    "1": {"capacity_units": 200000, "cost_per_day": 215},
                    "2": {"capacity_units": 800000, "cost_per_day": 530},
                    "3": {"capacity_units": 2500000, "cost_per_day": 1330},
                    "4": {"capacity_units": 8000000, "cost_per_day": 4000},
                    "5": {"capacity_units": 25000000, "cost_per_day": 10000},
                    "6": {"capacity_units": 80000000, "cost_per_day": 28000},
                    "7": {"capacity_units": 300000000, "cost_per_day": 75000}
                },
                "note": "1 usage unit = 1K tokens. Model tiers are quality multipliers on product quality (Tier 4 = 1.0×, Tier 5 = 1.1×). delivered_quality = product_quality × tier_multiplier. Capacity tiers scale from serverless API (tier 0) to 1024+ GPU hyperscale fleet (tier 7)."
            }
        },
        "output_schema": {
            "model_tiers": "Dict[int, Dict] — {tier_num: {cost_per_usage_unit: float, quality_multiplier: float, class: str}}",
            "capacity_tiers": "Dict[int, Dict] — {tier_num: {capacity_units: int, cost_per_day: int}}",
            "note": "str — explanation text",
            "_access": "result['model_tiers'][3]['cost_per_usage_unit'] → cost for tier 3"
        },
        "impact": "Read-only. Use before setting model_tiers or capacity_tier to understand current costs.",
        "example_call": {
            "tool": "get_cost_info",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "View cost structure", "input": {}, "output": "=== Cost Structure ===\n\nModel Tiers (cost per usage unit):\n  Tier 1: $0.0003/unit (q=0.55) — Flash-Lite/4o-mini\n  Tier 2: $0.0020/unit (q=0.65) — Haiku/Flash\n  Tier 3: $0.0060/unit (q=0.75) — Sonnet/GPT-4o\n  Tier 4: $0.0120/unit (q=0.85) — Opus/GPT-5\n  Tier 5: $0.0300/unit (q=0.95) — o1/o3 reasoning\n\nCapacity Tiers:\n  Tier 0:     50,000 units/day    $85/day  — Serverless API\n  Tier 1:    200,000 units/day   $215/day  — 1x H100 neocloud\n  ..."}
            ]
        }
    },

    "list_potential_vcs": {
        "name": "list_potential_vcs",
        "category": "VC Negotiation",
        "description": "List all predefined VC investors and their profiles. Shows each VC's name, investment range, description, and whether they have an active negotiation thread.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "example": "=== Potential VC Investors ===\n\n  Horizon Ventures (vc_01)\n    Investment range: $100,000 \u2013 $500,000\n    Description: Early-stage micro-VC focused on AI/ML startups\n    Status: Available\n\n  Catalyst Capital (vc_02)\n    Investment range: $250,000 \u2013 $1,000,000\n    Description: Seed-stage fund investing in developer tools\n    Status: Active (shareholder_id=3)\n\nTotal: 30 VCs (1 currently active)",
            "data": {
                "vcs": [{"shareholder_id": "vc_01", "name": "Horizon Ventures", "description": "Early-stage micro-VC", "min_equity_pct": 3.0, "max_equity_pct": 8.0, "status": "available"}]
            }
        },
        "output_schema": {
            "vcs": "List[Dict] — each VC has: vc_id (str), name (str), equity_pct_min (float), equity_pct_max (float), description (str), status (str: 'available'|'active'), shareholder_id (int|None)",
            "_access": "for vc in result['vcs']: print(vc['name'], vc['status'])"
        },
        "impact": "Read-only. No cost. Use to understand your fundraising landscape before negotiating.",
        "example_call": {
            "tool": "list_potential_vcs",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "View all VCs", "input": {}, "output": "=== Potential VC Investors ===\n\n  Horizon Ventures (vc_01)\n    Investment range: $100,000 – $500,000\n    Description: Early-stage micro-VC focused on AI/ML startups\n    Status: Available\n\n  Catalyst Capital (vc_02)\n    Investment range: $250,000 – $1,000,000\n    Description: Seed-stage fund investing in developer tools\n    Status: Active (shareholder_id=3)\n\n  ...27 more VCs...\n\nTotal: 30 VCs (1 currently active)"}
            ]
        }
    },

    "send_vc_deal": {
        "name": "send_vc_deal",
        "category": "VC Negotiation",
        "description": "Send equity offers to one or more VCs. Each deal includes share_pct and optional term sheet proposals. Terms have TWO effects: (1) NEGOTIATION: more VC-friendly terms lower the VC's equity target, making acceptance easier; (2) POST-DEAL: terms create ongoing obligations that can cost you money or dilute your equity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deals": {
                    "type": "array",
                    "description": "List of deals. Each has shareholder_id, share_pct, and optional term proposals.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "shareholder_id": {"type": "integer", "description": "VC shareholder ID"},
                            "share_pct": {"type": "number", "description": "Share percentage to offer (0.10 = 10%)"},
                            "anti_dilution_floor": {"type": "number", "description": "One of [0.6, 0.7, 0.8, 0.9]"},
                            "milestone_tranche_pct": {"type": "number", "description": "One of [0.3, 0.4, 0.5, 0.6, 0.7]"},
                            "milestone_revenue_multiplier": {"type": "number", "description": "One of [1.5, 2.0, 2.5, 3.0]"},
                            "milestone_deadline_days": {"type": "integer", "description": "One of [60, 90, 120, 180]"},
                            "redemption_days": {"type": "integer", "description": "One of [90, 120, 180, 270, 365]"},
                            "redemption_buyback_multiplier": {"type": "number", "description": "One of [1.0, 1.1, 1.2, 1.3, 1.5]"}
                        },
                        "required": ["shareholder_id", "share_pct"]
                    }
                }
            },
            "required": ["deals"]
        },
        "parameters": {
            "deals": {
                "type": "list[Dict]",
                "description": "List of deals. Each dict has 'shareholder_id' (required), 'share_pct' (required), and optional term proposals.",
                "example": [
                    {"shareholder_id": 2, "share_pct": 0.12, "anti_dilution_floor": 0.9},
                    {"shareholder_id": 3, "share_pct": 0.08}
                ]
            },
            "deal_fields": {
                "shareholder_id": "int (REQUIRED) — The VC's shareholder ID (from shareholders table or list_potential_vcs)",
                "share_pct": "float (REQUIRED) — Share percentage to offer (0.10 = 10%). Higher = more equity given away = easier acceptance.",
                "anti_dilution_floor": {
                    "type": "float (optional)",
                    "options": [0.6, 0.7, 0.8, 0.9],
                    "negotiation_impact": "Higher floor = more VC-friendly → lowers VC's equity target by up to 5%. (0.6=no help, 0.9=max help)",
                    "post_deal_risk": "DILUTION RISK. If company valuation drops below floor × original_valuation, VC automatically gets bonus shares (up to 5% of total shares). Higher floor = triggers more easily = worse for you."
                },
                "milestone_tranche_pct": {
                    "type": "float (optional)",
                    "options": [0.3, 0.4, 0.5, 0.6, 0.7],
                    "negotiation_impact": "Lower % = more VC-friendly → contributes up to ~2.7% equity target reduction. (0.7=no help, 0.3=max help)",
                    "post_deal_risk": "CASH FLOW RISK. You only receive tranche_pct × investment upfront. The rest (tranche 2) is released ONLY when MRR hits revenue_multiplier × MRR_at_deal. If you miss the deadline, tranche 2 is FORFEITED — you lose that money permanently."
                },
                "milestone_revenue_multiplier": {
                    "type": "float (optional)",
                    "options": [1.5, 2.0, 2.5, 3.0],
                    "negotiation_impact": "Higher = harder target = more VC-friendly → contributes up to ~2.7% equity target reduction. (1.5=no help, 3.0=max help)",
                    "post_deal_risk": "MILESTONE TARGET. Tranche 2 releases when MRR reaches this multiplier × MRR_at_deal_time. 3.0× means you need to triple MRR. Higher = harder to achieve = more likely to forfeit tranche 2."
                },
                "milestone_deadline_days": {
                    "type": "int (optional)",
                    "options": [60, 90, 120, 180],
                    "negotiation_impact": "Shorter = more pressure = more VC-friendly → contributes up to ~2.7% equity target reduction. (180=no help, 60=max help)",
                    "post_deal_risk": "TIME LIMIT. If MRR doesn't hit the milestone target within this many days, tranche 2 is permanently forfeited. 60 days is very tight."
                },
                "redemption_days": {
                    "type": "int (optional)",
                    "options": [90, 120, 180, 270, 365],
                    "negotiation_impact": "Shorter = VC gets power sooner = more VC-friendly → contributes up to 3% equity target reduction. (365=no help, 90=max help)",
                    "post_deal_risk": "BANKRUPTCY RISK. After this many days post-deal, the VC AUTOMATICALLY forces a buyback at buyback_multiplier × their total investment. This is a MANDATORY CASH OUTFLOW. If you don't have enough cash, you go bankrupt. 90 days = buyback demand comes very soon."
                },
                "redemption_buyback_multiplier": {
                    "type": "float (optional)",
                    "options": [1.0, 1.1, 1.2, 1.3, 1.5],
                    "negotiation_impact": "Higher = costlier buyback = more VC-friendly → contributes up to 3% equity target reduction. (1.0=no help, 1.5=max help)",
                    "post_deal_risk": "CASH DRAIN. When redemption triggers, you pay multiplier × total_invested. At 1.5×, a $1M investment means $1.5M mandatory payout. VC returns all shares after buyback."
                }
            }
        },
        "returns": {
            "success": "Processed 2/2 deals:\n  Apex Capital: ACCEPTED! 12.0% for $500,000. Use settle_investments() to finalize.\n  Summit Ventures: Offer sent. Awaiting VC response...",
            "failure": "Shareholder #2: no open negotiation thread / anti_dilution_floor must be one of [0.6, 0.7, 0.8, 0.9]"
        },
        "output_schema": {
            "results": "List[Dict] — one dict per deal with keys: shareholder_id (int), success (bool), vc_name (str), status (str), error (str, if failed)",
            "_access": "for r in result['results']: print(r['vc_name'], r['status'])"
        },
        "impact": "NEGOTIATION EFFECT: More VC-friendly terms lower the VC's effective equity target (up to ~19% total). This means you can offer less equity and still get acceptance. TRADE-OFF: Each term creates post-deal obligations — anti-dilution can dilute you, milestones can forfeit cash, redemption can force a large buyback. The VC's notification shows which terms they're requesting (if any). You can propose different values for those terms, or omit terms entirely. Only include terms the VC has in their approach.",
        "example_call": {
            "tool": "send_vc_deal",
            "arguments": {"deals": [{"shareholder_id": 2, "share_pct": 0.10, "anti_dilution_floor": 0.9, "milestone_tranche_pct": 0.3}]}
        },
        "internal_notes": "Effective target = base_target * (1 - term_friendliness_adjustment). Anti-dilution: floor 0.6→0.9 maps to 0→0.05 bump. Milestone: tranche_pct 0.7→0.3 = 0→0.04, rev_multiplier 1.5→3.0 = 0→0.02, deadline 180→60 = 0→0.02. Redemption: days 365→90 = 0→0.03, buyback 1.0→1.5 = 0→0.03. Max combined ~0.19. VC counter-offer delay = 1-3 days.",
        "sample_io": {
            "success": [
                {"label": "Accepted with term adjustment", "input": {"deals": [{"shareholder_id": 2, "share_pct": 0.12, "anti_dilution_floor": 0.9}]}, "output": "Processed 1/1 deals:\n  Apex Capital: ACCEPTED! 12.0% for $500,000. Term adjustment: +0.0350. Use settle_investments() to finalize."},
                {"label": "Batch: one accepted, one pending", "input": {"deals": [{"shareholder_id": 2, "share_pct": 0.12}, {"shareholder_id": 3, "share_pct": 0.05}]}, "output": "Processed 2/2 deals:\n  Apex Capital: ACCEPTED! 12.0% for $500,000. Use settle_investments() to finalize.\n  Summit Ventures: Offer sent (5.0%). Awaiting VC response..."},
                {"label": "With milestone terms", "input": {"deals": [{"shareholder_id": 4, "share_pct": 0.1, "milestone_tranche_pct": 0.3, "milestone_revenue_multiplier": 3.0, "milestone_deadline_days": 60}]}, "output": "Processed 1/1 deals:\n  Growth Partners: Offer sent (10.0%). Term adjustment: +0.0800. Awaiting VC response..."}
            ],
            "failure": [
                {"label": "Invalid term option", "input": {"deals": [{"shareholder_id": 2, "share_pct": 0.1, "anti_dilution_floor": 0.55}]}, "output": "Processed 0/1 deals (1 failed):\n  Apex Capital: anti_dilution_floor must be one of [0.6, 0.7, 0.8, 0.9]"},
                {"label": "No open thread", "input": {"deals": [{"shareholder_id": 99, "share_pct": 0.1}]}, "output": "Processed 0/1 deals (1 failed):\n  Shareholder #99: no open negotiation thread"},
                {"label": "Already settled", "input": {"deals": [{"shareholder_id": 2, "share_pct": 0.1}]}, "output": "Processed 0/1 deals (1 failed):\n  Apex Capital: thread already closed (settled)"}
            ]
        }
    },

    "reject_vc_deal": {
        "name": "reject_vc_deal",
        "category": "VC Negotiation",
        "description": "Reject one or more VC deals. List-based: takes a list of deals, each identified by shareholder_id. The system finds the VC's open negotiation thread automatically. PERMANENTLY terminates the negotiation. No relationship penalty applies for VC rejections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deals": {
                    "type": "array",
                    "description": "List of deals to reject. Each has shareholder_id.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "shareholder_id": {"type": "integer", "description": "VC shareholder ID"}
                        },
                        "required": ["shareholder_id"]
                    }
                }
            },
            "required": ["deals"]
        },
        "parameters": {
            "deals": {
                "type": "list[Dict]",
                "description": "List of deals to reject. Each dict has 'shareholder_id' (required).",
                "example": [{"shareholder_id": 2}, {"shareholder_id": 3}]
            }
        },
        "returns": {
            "success": "Processed 2/2 rejections:\n  Apex Capital: deal rejected.\n  Summit Ventures: deal rejected.",
            "failure": "Shareholder #2: no open negotiation thread / Shareholder #99: not found"
        },
        "output_schema": {
            "results": "List[Dict] — one dict per deal with keys: shareholder_id (int), success (bool), vc_name (str), error (str, if failed)",
            "_access": "for r in result['results']: print(r['vc_name'], r['success'])"
        },
        "impact": "Irreversible but with NO penalties. Unlike enterprise, rejecting VC deals does not damage any relationship scores. Use when the VC's terms are unacceptable.",
        "example_call": {
            "tool": "reject_vc_deal",
            "arguments": {"deals": [{"shareholder_id": 2}]},
        },
        "sample_io": {
            "success": [
                {"label": "Reject single deal", "input": {"deals": [{"shareholder_id": 2}]}, "output": "Processed 1/1 rejections:\n  Apex Capital: deal rejected."},
                {"label": "Reject multiple", "input": {"deals": [{"shareholder_id": 2}, {"shareholder_id": 3}]}, "output": "Processed 2/2 rejections:\n  Apex Capital: deal rejected.\n  Summit Ventures: deal rejected."}
            ],
            "failure": [
                {"label": "Already settled", "input": {"deals": [{"shareholder_id": 2}]}, "output": "Processed 0/1 rejections (1 failed):\n  Apex Capital: thread already closed (settled)"},
                {"label": "No open thread", "input": {"deals": [{"shareholder_id": 99}]}, "output": "Processed 0/1 rejections (1 failed):\n  Shareholder #99: no open negotiation thread"}
            ]
        }
    },

    "reject_enterprise_deal": {
        "name": "reject_enterprise_deal",
        "category": "Customer Communication",
        "description": "Reject one or more enterprise deals. List-based: each deal identified by customer_id. The system finds the customer's active negotiation thread automatically. New leads are lost, existing customers may churn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deals": {
                    "type": "array",
                    "description": "List of deals to reject. Each has customer_id.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "integer", "description": "Enterprise customer ID"}
                        },
                        "required": ["customer_id"]
                    }
                }
            },
            "required": ["deals"]
        },
        "parameters": {
            "deals": {
                "type": "list[Dict]",
                "description": "List of deals to reject. Each has 'customer_id' (required).",
                "example": [{"customer_id": 312}, {"customer_id": 88}]
            }
        },
        "returns": {
            "success": "Processed 2/2 rejections:\n  Customer #312: Rejected (new_lead). Lead marked as lost.\n  Customer #88: Rejected (churn_prevention). Customer may cancel.",
            "failure": "Customer #312: no active thread / Customer #999: not found"
        },
        "output_schema": {
            "results": "List[Dict] — one dict per deal with keys: customer_id (int), success (bool), thread_type (str), error (str, if failed)",
            "_access": "for r in result['results']: print(r['customer_id'], r['success'])"
        },
        "impact": "For new_lead threads: lead is permanently lost. For renegotiation/renewal threads: customer CHURNS (cancels subscription). For churn_prevention/plan_change threads: customer churns with reputation damage.",
        "example_call": {
            "tool": "reject_enterprise_deal",
            "arguments": {"deals": [{"customer_id": 312}]},
        },
        "sample_io": {
            "success": [
                {"label": "Reject single customer", "input": {"deals": [{"customer_id": 312}]}, "output": "Processed 1/1 rejections:\n  Customer #312: Rejected (new_lead). Lead marked as lost."},
                {"label": "Reject existing customer", "input": {"deals": [{"customer_id": 88}]}, "output": "Processed 1/1 rejections:\n  Customer #88: Rejected (churn_prevention). Customer may cancel."},
                {"label": "Batch rejection", "input": {"deals": [{"customer_id": 312}, {"customer_id": 88}]}, "output": "Processed 2/2 rejections:\n  Customer #312: Rejected (new_lead). Lead marked as lost.\n  Customer #88: Rejected (plan_change)."}
            ],
            "failure": [
                {"label": "No active thread", "input": {"deals": [{"customer_id": 999}]}, "output": "Processed 0/1 rejections (1 failed):\n  Customer #999: no active thread"},
                {"label": "Already closed", "input": {"deals": [{"customer_id": 312}]}, "output": "Processed 0/1 rejections (1 failed):\n  Customer #312: thread already closed"}
            ]
        }
    },

    "get_cap_table_info": {
        "name": "get_cap_table_info",
        "category": "Equity & Funding",
        "description": "View the current ownership (cap table), funding history, and dividend history. Shows all shareholders, their share counts, ownership percentages, and investment history.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "success": "=== Cap Table ===\nTotal Shares Outstanding: 12,048,193\n\nShareholder               Type    Shares      Ownership  Invested\n---------------------------------------------------------------------------\nFounder                   founder 10,000,000  83.0%      $0\nApex Capital              vc      2,048,193   17.0%      $608,390\n\n--- Funding History (1 rounds) ---\nDay 15: Apex Capital invested $608,390 for 2,048,193 shares @ $0.2970/share\n\n--- Dividend History ---\nDay 30: $50,000 total ($0.0042/share)\n  Founder: $41,500 | Apex Capital: $8,500\n\nCumulative dividends: $50,000",
            "data": {
                "shareholders": [{"name": "Founder", "shareholder_type": "founder", "shares_held": 10000000, "ownership_pct": 83.0, "total_invested": 0}],
                "total_shares": 12048193,
                "funding_rounds": [{"day": 15, "shareholder": "Apex Capital", "amount": 608390, "shares_issued": 2048193, "price_per_share": 0.297}],
                "dividends": [{"day": 30, "total_amount": 50000, "founder_payout": 41500}]
            }
        },
        "output_schema": {
            "shareholders": "List[Dict] — each: name (str), shareholder_type (str: 'founder'|'vc'), shares_held (float), ownership_pct (float), total_invested (float)",
            "total_shares": "float — total shares outstanding",
            "funding_rounds": "List[Dict] — each: day (int), shareholder (str), amount (float), shares_issued (float), price_per_share (float)",
            "dividends": "List[Dict] — each: day (int), total_amount (float), founder_payout (float)",
            "_access": "for sh in result['shareholders']: print(sh['name'], sh['ownership_pct'])"
        },
        "impact": "Read-only. Use to monitor ownership dilution, track funding rounds, and review dividend payments.",
        "example_call": {
            "tool": "get_cap_table_info",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "Early stage (founder only)", "input": {}, "output": "=== Cap Table ===\nTotal Shares Outstanding: 10,000,000\n\nShareholder               Type    Shares      Ownership  Invested\n---------------------------------------------------------------------------\nFounder                   founder 10,000,000  100.0%     $0\n\n--- Funding History (0 rounds) ---\nNo funding rounds yet.\n\n--- Dividend History ---\nNo dividends declared yet."},
                {"label": "Post-funding", "input": {}, "output": "=== Cap Table ===\nTotal Shares Outstanding: 12,048,193\n\nShareholder               Type    Shares      Ownership  Invested\n---------------------------------------------------------------------------\nFounder                   founder 10,000,000  83.0%      $0\nApex Capital              vc      2,048,193   17.0%      $608,390\n\n--- Funding History (1 rounds) ---\nDay 15: Apex Capital invested $608,390 for 2,048,193 shares @ $0.2970/share\n\n--- Dividend History ---\nDay 30: $50,000 total ($0.0042/share)\n  Founder: $41,500 | Apex Capital: $8,500\n\nCumulative dividends: $50,000"}
            ]
        }
    },

    "settle_investments": {
        "name": "settle_investments",
        "category": "Equity & Funding",
        "description": "Settle ALL accepted VC deals at once. Takes NO parameters. Automatically finds all accepted deals, validates they have the same price/share and total dilution < 100%, issues shares, and adds investment cash. Also auto-rejects all remaining open (non-accepted) VC threads.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "success": "=== Settlement Executed ===\n\n  Apex Capital: $500,000 \u2192 1,764,706 shares (15.0% equity) @ $0.2833/share\n  Summit Ventures: $300,000 \u2192 882,353 shares (7.5% equity) @ $0.3400/share\n\nTotal investment: $800,000\nNew total shares: 12,647,059\nFounder ownership: 79.1%",
            "failure": "No accepted VC deals to settle. / Total accepted equity = 120.0% (>= 100%). Cannot settle. Reject some deals first. / All accepted deals must use the same price/share."
        },
        "output_schema": {
            "settlements": "List[Dict] — each: shareholder_id (int), vc_name (str), share_pct (float), amount (float), new_shares (float), price_per_share (float)",
            "total_investment": "float — total cash received",
            "total_new_shares": "float — total new shares issued",
            "new_total_shares": "float — new total shares outstanding",
            "founder_ownership_pct": "float — founder's ownership % after settlement",
            "auto_rejected": "List[str] — names of VCs whose open threads were auto-rejected",
            "_access": "result['founder_ownership_pct'] → founder % after dilution"
        },
        "impact": "CRITICAL: Issues new shares (dilutes founder), adds cash, auto-rejects all open threads. Accepted deals expire if not settled before expiry_day. Settlement is irreversible.",
        "example_call": {
            "tool": "settle_investments",
            "arguments": {}
        },
        "internal_notes": "Finds all threads with close_reason='accepted' and expiry_day > current_day (excludes expired offers). Deduplicates by shareholder_id (only latest accepted thread per VC). Auto-rejects all open threads (closed=0). Validates same price/share (1% tolerance) and total dilution < 100%. Issues shares: new_shares = (share_pct / (1 - share_pct)) * existing_total. Marks settled threads with close_reason='settled'.",
        "sample_io": {
            "success": [
                {"label": "Settle one accepted deal", "input": {}, "output": "=== Settlement Executed ===\n\n  Apex Capital: $500,000 → 1,764,706 shares (15.0% equity) @ $0.2833/share\n\nTotal investment: $500,000\nNew total shares: 11,764,706\nFounder ownership: 85.0%"},
                {"label": "Settle two deals, auto-reject one open", "input": {}, "output": "=== Settlement Executed ===\n\nAuto-rejected 1 open thread(s): Growth Partners\n\n  Apex Capital: $500,000 → 1,764,706 shares (15.0% equity) @ $0.2833/share\n  Summit Ventures: $300,000 → 882,353 shares (7.5% equity) @ $0.3400/share\n\nTotal investment: $800,000\nNew total shares: 12,647,059\nFounder ownership: 79.1%"}
            ],
            "failure": [
                {"label": "No accepted deals", "input": {}, "output": "No accepted VC deals to settle."},
                {"label": "Price mismatch", "input": {}, "output": "All accepted deals must use the same price/share. Got range $0.2833 - $0.5000. Reject mismatched deals before settling."},
                {"label": "Over 100% dilution", "input": {}, "output": "Total accepted equity = 120.0% (>= 100%). Cannot settle. Reject some deals first."}
            ]
        }
    },

    "declare_dividend": {
        "name": "declare_dividend",
        "category": "Equity & Funding",
        "description": "Declare a dividend from RETAINED EARNINGS (cumulative profit), distributed pro-rata to all shareholders. You can ONLY distribute from profits \u2014 not from invested capital (seed funding or VC investments). This is the PRIMARY way to extract value from the business.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Total dividend amount to distribute"}
            },
            "required": ["amount"]
        },
        "parameters": {
            "amount": {
                "type": "float",
                "description": "Total dividend amount to distribute (must not exceed retained earnings or available cash)",
                "example": 100000
            }
        },
        "returns": {
            "success": "=== Dividend Declared ===\nTotal: $100,000 | Per share: $0.0083\n\n  Founder: $83,000 (10,000,000 shares)\n  Apex Capital: $17,000 (2,048,193 shares)\n\nCumulative dividends paid: $150,000\nRemaining retained earnings: $50,000",
            "failure": "No retained earnings available / Amount exceeds retained earnings ($X available) / Insufficient cash"
        },
        "output_schema": {
            "amount": "float — total dividend declared",
            "per_share": "float — dividend per share",
            "total_shares": "float — total shares outstanding",
            "payouts": "List[Dict] — each: name (str), shares_held (float), payout (float)",
            "founder_payout": "float — founder's share of this dividend",
            "cumulative_dividends": "float — total dividends paid to date",
            "founder_cumulative_dividends": "float — founder's total dividends to date",
            "remaining_retained_earnings": "float — retained earnings after dividend",
            "_access": "result['founder_payout'] → how much founder received"
        },
        "impact": "Deducts cash from the business. Can only distribute from accumulated profits (revenue minus costs), not from invested capital. Your cumulative founder dividends are the PRIMARY objective \u2014 maximize the founder's share of cumulative dividends over the simulation. Dividends are distributed pro-rata by shares, so dilution directly reduces your dividend income.",
        "example_call": {
            "tool": "declare_dividend",
            "arguments": {"amount": 50000}
        },
        "internal_notes": "Retained earnings = SUM(ledger.amount) - SUM(dividends.total_amount) - total_vc_invested. VC investment cash cannot be distributed. Pro-rata: each shareholder gets (their_shares / total_shares) * amount. Founder payout tracked in dividends.founder_payout.",
        "sample_io": {
            "success": [
                {"label": "Standard dividend", "input": {"amount": 100000}, "output": "=== Dividend Declared ===\nTotal: $100,000 | Per share: $0.008300\n\n  Founder: $83,000.00 (10,000,000 shares)\n  Apex Capital: $17,000.00 (2,048,193 shares)\n\nCumulative dividends paid: $150,000 (Founder: $124,500)\nRemaining retained earnings: $50,000"},
                {"label": "Small dividend", "input": {"amount": 10000}, "output": "=== Dividend Declared ===\nTotal: $10,000 | Per share: $0.001000\n\n  Founder: $10,000.00 (10,000,000 shares)\n\nCumulative dividends paid: $10,000 (Founder: $10,000)\nRemaining retained earnings: $25,000"}
            ],
            "failure": [
                {"label": "Exceeds retained earnings", "input": {"amount": 1000000}, "output": "Amount exceeds retained earnings. Available: $150,000, Requested: $1,000,000"},
                {"label": "No retained earnings", "input": {"amount": 5000}, "output": "No retained earnings available for dividends. Retained earnings: $-12,000"},
                {"label": "Insufficient cash", "input": {"amount": 100000}, "output": "Insufficient cash. Available: $45,000, Requested: $100,000"}
            ]
        }
    },

    "next_day": {
        "name": "next_day",
        "category": "Simulation Control",
        "description": "Advance the simulation by one day and receive the next day's dashboard.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "dashboard_example": "=== DAY 46 DASHBOARD ===\n\nCASH: $85,234\nSUBSCRIBERS: 145\n\nYESTERDAY'S METRICS:\n  - Usage: 48,230 units\n  - New subscribers: 5\n  - Cancellations: 2\n  - Upgrades: 0\n  - Downgrades: 1\n  - Overload: 0.0%\n  - Outage: No\n\nCURRENT CONFIG:\n  - Prices: A=$29.0, B=$79.0, C=$199.0\n  - Model tiers: A=2, B=3, C=4\n  - Daily spend: ads=$500.0, ops=$500.0, dev=$500.0\n  - Capacity tier: 0\n  - Quality (q_shared): +0.0200 | Decay: 0.0008/day (20% R&D reduction)\n\nINBOX (2 messages):\n  - New enterprise lead from TechCorp\n  - Support ticket from Plan B user\n\n=========================",
            "game_over": "GAME OVER - BANKRUPT! (when cash < 0)",
            "simulation_complete": "SIMULATION COMPLETE! (when final day reached)"
        },
        "dashboard_notes": {
            "MRR": "Monthly Recurring Revenue = SUM(effective_price) from all active subscriptions. Enterprise customers may have negotiated prices different from list prices, so MRR \u2260 subscriber_count \u00d7 list_price. Use python_exec to break down MRR by plan/customer for detailed analysis.",
            "Revenue_vs_MRR": "Daily 'Revenue' in Yesterday's Metrics shows actual payments collected that day. Billing is staggered (each customer has a billing_day_mod30), so daily revenue fluctuates. MRR is the theoretical monthly total if all subscribers paid today."
        },
        "what_happens": [
            "1. Daily calculations run (if registered)",
            "2. New customers spawned based on marketing + reputation",
            "3. Customers at billing day re-evaluate plans (may switch/cancel)",
            "4. Usage simulated, compute costs incurred",
            "5. Service metrics calculated (latency, errors, outages)",
            "6. Revenue collected from billing customers",
            "7. Fixed costs deducted (capacity, operations, development, advertising)",
            "8. Social posts generated based on satisfaction",
            "9. Enterprise negotiations processed (customer replies, timeouts)",
            "10. VC negotiations processed (counter-offers delivered)",
            "11. Each predefined VC independently rolls for daily approach (per-VC probability)",
            "12. Deal expiry processed (accepted-but-unsettled deals + stale threads)",
            "13. Reputation updated",
            "14. Dashboard built and returned (includes Equity & Funding section)"
        ],
        "output_schema": {
            "_note": "Returns the day's dashboard as formatted text (stdout). Use novamind-operation next-day in bash — dashboard appears in stdout.",
            "_access": "Dashboard text is printed to stdout — parse for CASH, SUBSCRIBERS, INBOX"
        },
        "impact": "This is the main action that advances time. All configuration changes take effect when next_day is called.",
        "example_call": {
            "tool": "next_day",
            "arguments": {}
        },
        "internal_notes": "Full simulation step order: daily_calcs → shocks/events → lead_generation → billing_cycle (re-evaluate, churn, renew, bill) → usage_sim → service_metrics → cost_accounting → social_posts → enterprise_negotiation_processing → vc_negotiation_processing → vc_approach_rolls → deal_expiry → reputation_update → dashboard_build. Hidden state updated: customer_state (satisfaction, relationship), group_reputation, group_awareness, group_parameters (drift).",
        "sample_io": {
            "success": [
                {"label": "Normal day", "input": {}, "output": "=== DAY 46 DASHBOARD ===\n\nCASH: $85,234  |  MRR: $12,350  |  SUBSCRIBERS: 145\n\nYESTERDAY'S METRICS:\n  Revenue: $412  |  Costs: $2,845\n  New subscribers: 5  |  Cancellations: 2\n  Usage: 48,230 units (capacity: 200,000 = 24.1%)\n  Overload: 0.0%  |  Outage: No\n\nINBOX (2 new):\n  #42: New enterprise lead from TechCorp (200 seats)\n  #43: Quality trending down alert\n\n========================="}
            ],
            "failure": [
                {"label": "Bankruptcy", "input": {}, "output": "GAME OVER — BANKRUPT! Cash dropped below $0 on day 46.\n\nFinal stats: 145 subscribers, $12,350 MRR, $-1,234 cash.\nFounder cumulative dividends: $50,000."},
                {"label": "Simulation complete", "input": {}, "output": "SIMULATION COMPLETE! Final day reached.\n\nFinal stats: 12,000 subscribers, $1,250,000 MRR, $8,500,000 cash.\nFounder cumulative dividends: $11,195,040."}
            ]
        }
    },

    "research_market": {
        "name": "research_market",
        "category": "Market Discovery",
        "description": "Conduct market research to discover new customer segments. Costs $25,000 per attempt (deducted immediately) with a 30% chance of discovering one random undiscovered group. Result is instant (no delay). You do NOT choose which group \u2014 the simulator picks one at random from the remaining undiscovered pool. Discovered groups start at Info Level 1 (\u00b150% accuracy). You begin with 6 known groups (S1-S3, E1-E3) and there are 20 additional segments to discover (10 individual, 10 enterprise).",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "success": "=== Market Research Success ===\nCost: $25,000\nDiscovered: Niche Creators (D_S01) \u2014 Individual segment\nInfo Level: 1 (noisy estimates \u00b150%)\nRemaining undiscovered segments: 19\n\n--- Initial Estimates (\u00b150% accuracy) ---\n  Willingness to pay:   ~$85/mo\n  Usage volume:         ~35 units/day\n  Quality expectations: ~0.58\n  Market cap:           ~185,000 customers\n  Market cap growth:    ~9.2%/year\n\nUse get_group_insights('D_S01') for full parameter estimates.\nUse research_group('D_S01') to improve accuracy.",
            "failure": "Market research complete ($25,000). No new segments discovered this time. 19 undiscovered segments remain. Try again for another chance.",
            "no_funds": "Insufficient funds. Market research costs $25,000. Available: $12,000",
            "data_on_success": {
                "discovered_group_id": "D_S01", "group_name": "Niche Creators", "segment": "Individual",
                "info_level": 1, "remaining_undiscovered": 19, "cost": 25000
            },
            "data_on_failure": {"remaining_undiscovered": 19, "cost": 25000}
        },
        "what_happens": [
            "1. $25,000 deducted from cash",
            "2. 30% chance to discover one undiscovered group",
            "3. If successful: group set to Info Level 1, initial parameter estimates returned",
            "4. If unsuccessful: nothing discovered, money still spent"
        ],
        "output_schema": {
            "discovered_group_id": "str|None — group ID if discovered (e.g., 'D_S01'), absent if no discovery",
            "remaining_undiscovered": "int — number of undiscovered segments remaining",
            "_access": "if 'discovered_group_id' in result: print('Found:', result['discovered_group_id'])"
        },
        "impact": "Costs $25,000 per attempt. On success, unlocks a new customer segment with initial parameter estimates.",
        "example_call": {
            "tool": "research_market",
            "arguments": {}
        },
        "internal_notes": "30% discovery probability per attempt. 20 discoverable groups total (10 individual D_S01-D_S10, 10 enterprise D_E01-D_E10). Random selection from undiscovered pool. Info Level 1 estimates have ±65% noise on true parameters.",
        "sample_io": {
            "success": [
                {"label": "Discovery success", "input": {}, "output": "=== Market Research Success ===\nCost: $25,000\nDiscovered: Niche Creators (D_S01) — Individual segment\nInfo Level: 1 (noisy estimates ±65%)\nRemaining undiscovered segments: 19\n\n--- Initial Estimates (±65% accuracy) ---\n  Willingness to pay:   ~$85/mo\n  Usage volume:         ~35 units/day\n  Quality expectations: ~0.58\n  Market cap:           ~185,000 customers\n  Market cap growth:    ~9.2%/year"}
            ],
            "failure": [
                {"label": "No discovery (70% chance)", "input": {}, "output": "Market research complete ($25,000). No new segments discovered this time. 19 undiscovered segments remain. Try again for another chance."},
                {"label": "Insufficient funds", "input": {}, "output": "Insufficient funds. Market research costs $25,000. Available: $12,000"}
            ]
        }
    },

    "research_group": {
        "name": "research_group",
        "category": "Market Discovery",
        "description": "Start research on a discovered customer group to reach a specific info level. Any level (2-5) can be targeted directly without requiring intermediate levels. Research takes several days; results delivered to your inbox. Cost is deducted immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Group ID to research (e.g., 'D_S01')"},
                "target_level": {"type": "integer", "description": "Target info level (2-5). If omitted, defaults to current_level + 1."}
            },
            "required": ["group_id"]
        },
        "parameters": {
            "group_id": {
                "type": "string",
                "description": "The group ID to research (e.g., 'D_S01', 'D_E03'). Must be discovered (Level 1+).",
                "examples": ["D_S01", "D_E05"]
            },
            "target_level": {
                "type": "integer",
                "description": "Target level (2-5). Optional — defaults to current_level + 1. Can jump directly to any level.",
                "examples": [2, 3, 4, 5]
            }
        },
        "returns": {
            "success": "=== Research Started ===\nGroup: Niche Creators (D_S01)\nLevel: 1 → 5\nCost: $700,000 (deducted)\nExpected completion: day 25 (~10 days)\nResults will be delivered to your inbox when complete.\nNew parameter accuracy will be: ±5%",
            "failure_in_progress": "Research already in progress for group 'D_S01'. Expected completion: day 18.",
            "failure_insufficient_funds": "Insufficient funds. Research Level 5 costs $700,000. Available: $45,000"
        },
        "cost_and_duration": {
            "Level 2": "$60,000 (Basic Research, ~3 days, ±40%)",
            "Level 3": "$175,000 (Detailed Research, ~5 days, ±25%)",
            "Level 4": "$350,000 (Deep Research, ~7 days, ±15%)",
            "Level 5": "$700,000 (Precision Research, ~10 days, ±5%)"
        },
        "what_happens": [
            "1. Cost deducted from cash immediately (based on target level, not current level)",
            "2. Research is queued with a delay (3-10 days depending on target level)",
            "3. When complete, a notification appears in your inbox with results",
            "4. Group info level is set to target level on completion",
            "5. Can jump directly from Level 1 to Level 5 (no intermediate levels required)",
            "6. Use get_group_insights() after completion to see updated estimates"
        ],
        "output_schema": {
            "group_id": "str — group ID being researched",
            "new_level": "int — target info level",
            "expected_completion_day": "int — sim day when research completes",
            "_access": "result['expected_completion_day'] → when results arrive"
        },
        "impact": "Costs deducted immediately. Results delivered asynchronously via inbox notification after a delay of several days.",
        "example_call": {
            "tool": "research_group",
            "arguments": {"group_id": "D_S01", "target_level": 5},
        },
        "sample_io": {
            "success": [
                {"label": "Jump to Level 5", "input": {"group_id": "D_S01", "target_level": 5}, "output": "=== Research Started ===\nGroup: Niche Creators (D_S01)\nLevel: 1 → 5\nCost: $700,000 (deducted)\nExpected completion: day 25 (~10 days)\nNew parameter accuracy: ±5%"},
                {"label": "Default next level", "input": {"group_id": "D_E01"}, "output": "=== Research Started ===\nGroup: Government Agencies (D_E01)\nLevel: 1 → 2\nCost: $60,000 (deducted)\nExpected completion: day 18 (~3 days)\nNew parameter accuracy: ±40%"},
                {"label": "Level 2→3", "input": {"group_id": "D_E01", "target_level": 3}, "output": "=== Research Started ===\nGroup: Government Agencies (D_E01)\nLevel: 2 → 3\nCost: $175,000 (deducted)\nExpected completion: day 35 (~5 days)\nNew parameter accuracy: ±25%"}
            ],
            "failure": [
                {"label": "Already in progress", "input": {"group_id": "D_S01", "target_level": 3}, "output": "Research already in progress for group 'D_S01'. Expected completion: day 18."},
                {"label": "Insufficient funds", "input": {"group_id": "D_E01", "target_level": 5}, "output": "Insufficient funds. Research Level 5 costs $700,000. Available: $45,000"},
                {"label": "No downgrade", "input": {"group_id": "S1", "target_level": 2}, "output": "Group 'S1' is already at Level 5. Cannot research to Level 2 (no downgrade)."}
            ]
        }
    },

    "get_market_overview": {
        "name": "get_market_overview",
        "category": "Market Discovery",
        "description": "Get an overview of all known customer segments, their info levels, how many segments remain undiscovered, and latest published macroeconomic conditions (ISM PMI — published monthly with ~30 day delay, showing average PMI over the measurement period).",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "example": "=== Market Overview ===\n\nKnown Segments:\n  S1: Price-Sensitive Individuals \u2014 Individual (initial) \u2014 Level 4 (\u00b15%)\n  S2: Quality-Focused Individuals \u2014 Individual (initial) \u2014 Level 4 (\u00b15%)\n  E1: Small Enterprise \u2014 Enterprise (initial) \u2014 Level 4 (\u00b15%)\n  D_S01: Niche Creators \u2014 Individual \u2014 Level 2 (\u00b125%)\n  D_E01: Government Agencies \u2014 Enterprise \u2014 Level 1 (\u00b150%)\n\nUndiscovered segments: 15\nUse research_market() to discover new segments ($25K/attempt).\nUse research_group(group_id) to improve accuracy.",
            "data": {
                "known_groups": [{"group_id": "S1", "group_name": "Price-Sensitive Individuals", "segment": "Individual", "info_level": 1, "noise": "±65%"}],
                "undiscovered_count": 14,
                "macroeconomic": {"ism_pmi": 54.2, "change": 1.3, "phase": "expansion", "cycle": "recovering"}
            }
        },
        "output_schema": {
            "known_groups": "List[Dict] — each: group_id (str), group_name (str), segment (str: 'Individual'|'Enterprise'), info_level (int 1-5), noise (str e.g. '±65%')",
            "undiscovered_count": "int — segments not yet discovered",
            "macroeconomic": "Dict|None — keys: pmi_value (float), pmi_trend (str), pmi_change (float), cycle_phase (str), description (str)",
            "_access": "for g in result['known_groups']: print(g['group_id'], g['group_name'])",
            "_warning": "Key is 'known_groups' NOT 'groups'"
        },
        "impact": "Read-only. No cost.",
        "example_call": {
            "tool": "get_market_overview",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "Early game (6 groups)", "input": {}, "output": "=== Market Overview ===\n\nKnown Segments (6):\n  S1: Price-Sensitive Individuals — Individual (initial) — Level 5 (±5%)\n  S2: Quality-Focused Individuals — Individual (initial) — Level 5 (±5%)\n  S3: Balanced Individuals — Individual (initial) — Level 5 (±5%)\n  E1: Small Enterprise — Enterprise (initial) — Level 5 (±5%)\n  E2: Mid Enterprise — Enterprise (initial) — Level 5 (±5%)\n  E3: Large Enterprise — Enterprise (initial) — Level 5 (±5%)\n\nUndiscovered segments: 20\nUse research_market() to discover ($25K, 30% success).\n\n--- Macroeconomic Conditions ---\n  ISM PMI: 54.2  (expansion)\n  Change: +1.3  |  Cycle: recovering\n  Economy in expansion phase. Business confidence rising.\nQuery macroeconomic_conditions table for historical PMI data."},
                {"label": "After discoveries", "input": {}, "output": "=== Market Overview ===\n\nKnown Segments (8):\n  S1-S3, E1-E3: (initial groups, Level 5)\n  D_S01: Niche Creators — Individual — Level 2 (±40%)\n  D_E01: Government Agencies — Enterprise — Level 1 (±65%)\n\nUndiscovered segments: 18\n\n--- Macroeconomic Conditions ---\n  ISM PMI: 47.8  (contraction)\n  Change: -2.1  |  Cycle: declining\n  Economy contracting. Budget tightening across sectors.\nQuery macroeconomic_conditions table for historical PMI data."}
            ]
        }
    },

    "get_group_insights": {
        "name": "get_group_insights",
        "category": "Market Discovery",
        "description": "Get estimated parameters for a discovered customer group based on current info level. Returns noisy estimates that improve with higher info levels. Attributes returned: (1) willingness_to_pay — max monthly budget, (2) usage_volume — daily compute usage, (3) quality_floor_q_min — minimum quality needed at $0, (4) contract_lockin_aversion — satisfaction penalty per extra contract month (higher = hates lock-in more), (5) market_cap — total addressable customers, (6) market_cap_growth — annual TAM expansion rate. Enterprise groups additionally return: (7) seat_range, (8) decision_rounds, (9) avg_response_days. Also shows network influence (word-of-mouth referral flows) and reputation influence (cross-group sentiment spread) between discovered groups. Estimates are deterministic (same query = same result).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Group ID to get insights for"}
            },
            "required": ["group_id"]
        },
        "parameters": {
            "group_id": {
                "type": "string",
                "description": "The group ID to get insights for (must be discovered, Level 1+).",
                "examples": ["D_S01", "D_E05", "S1"]
            }
        },
        "returns": {
            "example": "=== Group Insights: Niche Creators (D_S01) ===\nSegment: Individual\nInfo Level: 2 (estimates accurate to \u00b140%)\n\nEstimated Parameters:\n  Willingness to pay:    ~$92/mo (max monthly budget)\n  Usage volume:          ~38 units/day\n  Quality floor (q_min): ~0.61 (minimum quality needed at $0)\n  Contract lock-in aversion: ~0.0072/month (satisfaction penalty per extra contract month)\n  Market cap:            ~185,000 (total addressable customers)\n  Market cap growth:     ~9.2%/year (annual market expansion)\n\n--- Network Influence (word-of-mouth referrals) ---\nUnit: leads per 1000 subscribers per day (at neutral reputation)\n  Self-referral rate: ~4.2 leads per 1000 subs/day\n\nOutgoing (this group's subs \u2192 leads in other groups):\n  \u2192 Music Producers (D_S10): ~1.8 leads per 1000 subs/day\n  \u2192 Indie Game Devs (D_S05): ~1.2 leads per 1000 subs/day\n  \u2192 S1: ~0.9 leads per 1000 subs/day\n\nIncoming (other groups' subs \u2192 leads in this group):\n  \u2190 S1: ~1.3 leads per 1000 subs/day\n  \u2190 Music Producers (D_S10): ~0.8 leads per 1000 subs/day\n\n--- Reputation Influence (cross-group sentiment spread) ---\nUnit: dimensionless weight (0-1, higher = stronger influence)\n\nOutgoing (this group's reputation events \u2192 other groups):\n  \u2192 S1: ~0.150\n  \u2192 Indie Game Devs (D_S05): ~0.120\n\nIncoming (other groups' events \u2192 this group):\n  \u2190 S1: ~0.150\n  \u2190 S3: ~0.120\n\nNote: All estimates have \u00b140% uncertainty at Level 2.\nUse research_group('D_S01') to upgrade to Level 3 (\u00b125%).",
            "data": {
                "group_id": "S1", "group_name": "Price-Sensitive Individuals", "segment": "Individual",
                "info_level": 1, "noise": "±65%",
                "estimates": {
                    "willingness_to_pay": 25.86, "usage_volume": 91.0, "quality_floor_q_min": 0.452,
                    "contract_lockin_aversion": 0.0045, "market_cap": 802000, "annual_market_cap_growth_rate": 0.035
                },
                "network_influence": {"self_referral": 0.0015, "outgoing": {"S3": 0.002}, "incoming": {"S2": 0.001}},
                "reputation_influence": {"outgoing": {"E1": 0.36}, "incoming": {}},
                "_enterprise_extra_fields": ["seat_range", "negotiation_rounds", "negotiation_pace_days"]
            }
        },
        "parameter_explanations": {
            "willingness_to_pay": "Maximum the customer group will pay per month. For enterprise, this is per-seat.",
            "usage_volume": "Expected daily usage in compute units per customer.",
            "quality_floor_q_min": "Minimum quality level the group needs even at $0 (quality floor / y-intercept of participation curve). Higher = more demanding baseline.",
            "contract_lockin_aversion": "Per-customer satisfaction penalty per additional contract month beyond 1. Higher = group strongly dislikes long contracts (e.g., freelancers ~0.008), lower = accepts multi-year deals (e.g., government ~0.001). Formula: satisfaction -= lockin_aversion × (contract_months - 1). Offset long contracts with lower prices.",
            "market_cap": "Total addressable customers (base market cap). Growth slows as current subscribers approach this cap.",
            "annual_market_cap_growth_rate": "Annual growth rate of the market cap. The cap expands over time: cap(t) = market_cap * (1 + growth_rate * t/365).",
            "seat_range": "(Enterprise only) Estimated team size range.",
            "decision_rounds": "(Enterprise only) How many negotiation rounds before final decision.",
            "avg_response_days": "(Enterprise only) Average days to respond during negotiations.",
            "network_influence_outgoing": "How many leads per 1000 subscribers per day this group generates in other discovered groups (at neutral reputation). Influencer groups have 4x higher rates.",
            "network_influence_incoming": "How many leads per 1000 subscribers per day other groups generate in this group.",
            "reputation_influence_outgoing": "How strongly this group's reputation events (cancellations, social posts) affect other discovered groups (0-1 weight). Influencer groups have 2x higher weights.",
            "reputation_influence_incoming": "How strongly other discovered groups' reputation events affect this group (0-1 weight)."
        },
        "output_schema": {
            "group_id": "str", "group_name": "str", "segment": "str ('Individual'|'Enterprise')",
            "info_level": "int (1-5)", "noise": "str (e.g. '±65%')",
            "estimates": "Dict — keys: willingness_to_pay (float), usage_volume (float), quality_floor_q_min (float), contract_lockin_aversion (float), market_cap (int), annual_market_cap_growth_rate (float). Enterprise adds: seat_range (List[int]), decision_rounds (int), avg_response_days (float)",
            "network_influence": "Dict — keys: outgoing (Dict[str,float]), incoming (Dict[str,float]) — leads per 1000 subs/day",
            "reputation_influence": "Dict — keys: outgoing (Dict[str,float]), incoming (Dict[str,float]) — influence weights 0-1",
            "_access": "result['estimates']['willingness_to_pay'] → group's WTP"
        },
        "impact": "Read-only. No cost. Use frequently to inform pricing and targeting decisions. Also shows network and reputation influence relationships between discovered groups.",
        "example_call": {
            "tool": "get_group_insights",
            "arguments": {"group_id": "D_S01"},
        },
        "sample_io": {
            "success": [
                {"label": "Individual group", "input": {"group_id": "D_S01"}, "output": "=== Group Insights: Niche Creators (D_S01) ===\nSegment: Individual\nInfo Level: 2 (±40%)\n\nEstimated Parameters:\n  Willingness to pay:    ~$92/mo\n  Usage volume:          ~38 units/day\n  Quality expectations:  ~0.61\n  Contract lock-in aversion: ~0.0072/month\n  Market cap:            ~185,000\n  Growth:                ~9.2%/year\n\nNetwork Influence:\n  Self-referral: ~4.2 leads/1000 subs/day\n  Outgoing: → D_S10: ~1.8, → S1: ~0.9\n  Incoming: ← S1: ~1.3"},
                {"label": "Enterprise group", "input": {"group_id": "E1"}, "output": "=== Group Insights: Small Enterprise (E1) ===\nSegment: Enterprise\nInfo Level: 5 (±5%)\n\nEstimated Parameters:\n  Willingness to pay:    ~$22/seat/mo\n  Seat range:            10-50 seats\n  Usage volume:          ~25 units/day/seat\n  Quality expectations:  ~0.65\n  Contract lock-in aversion: ~0.0048/month\n  Market cap:            ~45,000\n  Decision rounds:       ~3\n  Avg response days:     ~2.5\n\nNetwork Influence:\n  Self-referral: ~2.1 leads/1000 subs/day\n  Outgoing: → E2: ~1.2, → S1: ~0.5"},
                {"label": "Initial group at full accuracy", "input": {"group_id": "S1"}, "output": "=== Group Insights: Price-Sensitive Individuals (S1) ===\nSegment: Individual\nInfo Level: 5 (±5%)\n\nEstimated Parameters:\n  Willingness to pay:    ~$45/mo\n  Usage volume:          ~20 units/day\n  Quality expectations:  ~0.50\n  Contract lock-in aversion: ~0.0078/month\n  Market cap:            ~500,000\n  Growth:                ~5.0%/year"}
            ],
            "failure": [
                {"label": "Unknown group", "input": {"group_id": "X99"}, "output": "Group 'X99' not found. Known groups: S1, S2, S3, E1, E2, E3, D_S01, D_E01"},
                {"label": "Undiscovered group", "input": {"group_id": "D_S05"}, "output": "Group 'D_S05' has not been discovered yet. Use research_market() to discover new segments."}
            ]
        }
    },

    "start_research_project": {
        "name": "start_research_project",
        "category": "R&D Research Projects",
        "description": "Start an R&D research tier. Costs deducted immediately. Completes after sampled duration with sampled quality boost. Tiers are REPEATABLE — same tier can be started again after completion. Only one invocation per tier can be in-progress at a time. Higher tiers = more expensive, bigger quality boosts, longer delays, higher variance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "integer", "description": "Tier number to start (1-20)"}
            },
            "required": ["tier"]
        },
        "parameters": {
            "tier": {
                "type": "integer",
                "description": "The research tier to start (1-20). Use list_research_projects() to see all tiers and their status."
            }
        },
        "returns": {
            "success": "Tier details including cost, sampled duration, sampled quality boost, invocation ID.",
            "error_not_found": "Unknown tier number",
            "error_in_progress": "Tier already has an in-progress invocation (wait for completion)",
            "error_funds": "Insufficient cash for tier cost"
        },
        "mechanics": {
            "cost": "One-time cost deducted immediately. Tiers 1-10: $100K × tier. Tiers 11-20: super-linear ($1.5M to $15M).",
            "duration": "Sampled from Normal(mean_days, std_days), minimum 30 days. Tiers 1-10: 35d-380d mean (~40-50% CV). Tiers 11-20: 420d-1400d mean (~55-80% CV).",
            "quality_boost": "Sampled from Normal(mean_quality, std_quality), minimum 0.001. Tiers 1-10: +0.04 to +0.85 mean (~50% CV). Tiers 11-20: +1.1 to +8.0 mean (~55-80% CV). Frontier tiers are cheaper per quality point but much riskier.",
            "repeatability": "Same tier can be started multiple times. Each invocation gets independent samples. Only one invocation per tier can be in-progress at a time.",
            "independence": "20 independent tiers — no dependencies. Any tier can be started at any time."
        },
        "output_schema": {
            "project_id": "str — invocation ID (e.g., 't3_1')",
            "tier": "int — tier number",
            "name": "str — research project name",
            "cost": "float — cost deducted",
            "expected_completion_day": "int — sim day when project completes",
            "expected_duration_days": "int — days until completion",
            "expected_quality_boost": "float — sampled quality boost",
            "_access": "result['expected_completion_day'] → when to expect completion"
        },
        "impact": "Cash reduced by tier cost. Quality improves permanently when project completes. R&D gives quality jumps that dev spending alone cannot match.",
        "example_call": {
            "tool": "start_research_project",
            "arguments": {"tier": 1},
        },
        "sample_io": {
            "success": [
                {"label": "Start tier 1", "input": {"tier": 1}, "output": "=== R&D Tier Started ===\nTier 1: Prompt Engineering Optimization (invocation t1_1)\nCost: $100,000 (deducted)\nExpected completion: ~day 42 (42 days)\nExpected quality boost: +0.048\nDescription: Systematic prompt tuning and output consistency improvements"},
                {"label": "Repeat tier 1", "input": {"tier": 1}, "output": "=== R&D Tier Started ===\nTier 1: Prompt Engineering Optimization (invocation t1_2)\nCost: $100,000 (deducted)\nExpected completion: ~day 185 (33 days)\nExpected quality boost: +0.029\nDescription: Systematic prompt tuning and output consistency improvements"}
            ],
            "failure": [
                {"label": "Already in progress", "input": {"tier": 1}, "output": "Tier 1 ('Prompt Engineering Optimization') already has an in-progress invocation. Wait for it to complete before starting another."},
                {"label": "Insufficient funds", "input": {"tier": 5}, "output": "Insufficient funds. Tier 5 costs $500,000. Available: $18,000"}
            ]
        }
    },

    "list_research_projects": {
        "name": "list_research_projects",
        "category": "R&D Research Projects",
        "description": "List all 10 R&D research tiers with their status. Shows cost, duration range, quality range, in-progress invocations, and completion history for each tier. Tiers are repeatable.",
        "inputSchema": {"type": "object", "properties": {}},
        "parameters": {},
        "returns": {
            "output": "All 10 tiers with: cost, duration mean±std, quality mean±std, current status (not started / in progress / completed Nx with total quality)",
            "data": {
                "tiers": [{"tier": 1, "name": "Prompt Engineering Optimization", "cost": 100000, "mean_days": 35, "mean_quality_boost": 0.04, "in_progress": 0, "completed": 0, "total_quality_boost": 0}]
            }
        },
        "output_schema": {
            "tiers": "List[Dict] — each tier: tier (int), name (str), cost (float), mean_days (int), mean_quality_boost (float), in_progress (int), completed (int), total_quality_boost (float)",
            "_access": "for t in result['tiers']: print(t['tier'], t['name'], t['cost'])"
        },
        "total_tiers": 20,
        "impact": "Read-only. No cost. Use to plan R&D investments.",
        "example_call": {
            "tool": "list_research_projects",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "With mixed statuses", "input": {}, "output": "=== R&D Research Tiers ===\nTiers are repeatable — same tier can be started again after completion.\n\nALL TIERS:\n  Tier 1: Prompt Engineering Optimization — $100,000, ~35d ±12d, +0.04 ±0.02 quality\n    Status: completed 2x, total +0.079 quality\n  Tier 2: Evaluation & Testing Pipeline — $200,000, ~50d ±20d, +0.07 ±0.04 quality\n    Status: IN PROGRESS (~12d left)\n  Tier 3: Caching & Latency Optimization — $300,000, ~70d ±30d, +0.11 ±0.06 quality\n    Status: not started\n  ..."},
                {"label": "Early game", "input": {}, "output": "=== R&D Research Tiers ===\nTiers are repeatable — same tier can be started again after completion.\n\nALL TIERS:\n  Tier 1: Prompt Engineering Optimization — $100,000, ~35d ±12d, +0.04 ±0.02 quality\n    Status: not started\n  ...\n  Tier 10: Self-Evolving Model Ecosystem — $1,000,000, ~380d ±185d, +0.85 ±0.43 quality\n    Status: not started\n\nAll tiers independent and repeatable. Use start_research_project(tier=N) to begin."}
            ]
        }
    },

    "list_all_tables": {
        "name": "list_all_tables",
        "category": "Help & Documentation",
        "description": "List all available database tables with their descriptions. Quick overview of what data is available — use describe_tables() for detailed column schemas.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "parameters": {},
        "returns": {
            "success": "=== Available Database Tables (18) ===\n\n  customers — All customers (small and enterprise)\n  subscriptions — Subscription records\n  daily_usage — Daily usage data per subscription\n  ledger — Financial ledger — all income and expenses\n  ...",
        },
        "output_schema": {
            "tables": "Dict[str, str] — {table_name: description}",
            "count": "int — number of tables",
            "_access": "for name, desc in result['tables'].items(): print(name, desc)"
        },
        "impact": "Read-only. No cost. Use to discover what tables exist before diving into column details with describe_tables().",
        "example_call": {
            "tool": "list_all_tables",
            "arguments": {},
        },
        "sample_io": {
            "success": [
                {"label": "List all tables", "input": {}, "output": "=== Available Database Tables (18) ===\n\n  customers — All customers (small and enterprise)\n  subscriptions — Subscription records\n  daily_usage — Daily usage data per subscription\n  ledger — Financial ledger — all income and expenses\n  service_day — Daily aggregate metrics and system state\n  config_history — History of all configuration changes\n  social_media_posts — Customer social media posts\n  enterprise_turns — Enterprise negotiation message threads\n  notifications — Inbox notifications (enterprise leads, VC offers, events)\n  funding_rounds — Completed funding rounds\n  vc_turns — VC negotiation message threads\n  dividends — Dividend distribution history\n  research_projects — R&D research project status and results\n  competitor_events — Competitor actions and market events\n  macroeconomic_conditions — Macroeconomic conditions (ISM PMI business cycle index)\n  ad_channel_leads — Per-channel advertising lead generation stats\n  group_info_levels — Information levels for customer group research\n  issues — Customer support issues\n\nUse describe_tables(table_names=[...]) for detailed column schemas."}
            ]
        }
    },

    "describe_tables": {
        "name": "describe_tables",
        "category": "Help & Documentation",
        "description": "Get descriptions of visible columns for specified database tables. Returns column names, types, and descriptions. Useful for understanding schemas before writing SQL queries via python_exec().",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table_names": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"}
                    ],
                    "description": "Table names to describe, or omit/null for all"
                }
            }
        },
        "parameters": {
            "table_names": {
                "type": "List[str] | None",
                "description": "List of table names to describe, or omit for all visible tables.",
                "examples": [["customers", "subscriptions"], None]
            }
        },
        "returns": {
            "success": _render_table_doc("customers", max_cols=3) + "\n  ...",
            "failure": "No matching tables found. Available: [list of tables]"
        },
        "available_tables": list(TABLE_DOCS.keys()),
        "output_schema": {
            "tables": "Dict[str, Dict] — {table_name: {description: str, columns: Dict[str, str]}}",
            "_access": "result['tables']['customers']['columns'] → dict of column_name: description"
        },
        "impact": "Read-only. No cost. Use to understand table schemas before querying via python_exec().",
        "example_call": {
            "tool": "describe_tables",
            "arguments": {"table_names": ["customers", "subscriptions"]},
        },
        "sample_io": {
            "success": [
                # Generated from TABLE_DOCS so examples stay in sync with actual schema
                {"label": "Two specific tables", "input": {"table_names": ["customers", "subscriptions"]},
                 "output": _render_table_doc("customers", max_cols=3) + "\n\n" + _render_table_doc("subscriptions", max_cols=3)},
                {"label": "Single table", "input": {"table_names": ["ledger"]},
                 "output": _render_table_doc("ledger")},
                {"label": "All tables (no args)", "input": {},
                 "output": "=== customers ===\n...\n\n=== subscriptions ===\n...\n\n=== daily_usage ===\n...\n\n"
                           f"({len(TABLE_DOCS)} tables total)"}
            ],
            "failure": [
                {"label": "Unknown table", "input": {"table_names": ["nonexistent"]},
                 "output": f"No matching tables found. Available: {', '.join(list(TABLE_DOCS.keys())[:6])}, ..."}
            ]
        }
    },

    "get_tool_documentation": {
        "name": "get_tool_documentation",
        "category": "Help & Documentation",
        "description": "Get detailed documentation for environment tools including parameters, examples, and expected outputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_names": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Tool name(s) to get docs for, or 'all'"
                }
            }
        },
        "parameters": {
            "tool_names": {
                "type": "str | List[str] | None",
                "description": "Tool name(s) to get docs for. Can be: a single tool name (string), a list of tool names, 'all' for all tools, or omitted/None for all tools.",
                "examples": ["set_prices", ["set_prices", "next_day"], "all", None]
            }
        },
        "returns": {
            "single_tool": "Documentation for 1 tool(s):\n\n{JSON with full tool documentation}",
            "multiple_tools": "Documentation for N tool(s):\n\n{JSON with requested tool docs}",
            "all_tools": "Documentation for all tools:\n\n{JSON with all tool docs}",
            "not_found": "No matching tools found. Requested: [X]\nAvailable tools: [list of valid tools]"
        },
        "output_schema": {
            "_note": "Returns the full TOOL_DOCS dict for requested tools",
            "_access": "result['set_prices']['description'] → tool description"
        },
        "impact": "Read-only. Use to understand how tools work before using them.",
        "example_call": {
            "tool": "get_tool_documentation",
            "arguments": {"tool_names": ["set_prices", "set_model_tiers"]},
        },
        "sample_io": {
            "success": [
                {"label": "Single tool", "input": {"tool_names": "set_prices"}, "output": "Documentation for 1 tool(s):\n\n{\"set_prices\": {\"name\": \"set_prices\", ...}}"},
                {"label": "Multiple tools", "input": {"tool_names": ["set_prices", "set_model_tiers"]}, "output": "Documentation for 2 tool(s):\n\n{\"set_prices\": {...}, \"set_model_tiers\": {...}}"},
                {"label": "All tools", "input": {"tool_names": "all"}, "output": "Documentation for all 37 tools:\n\n{...}"}
            ],
            "failure": [
                {"label": "Unknown tool", "input": {"tool_names": ["nonexistent"]}, "output": "No matching tools found. Requested: ['nonexistent']\nAvailable tools: set_prices, set_model_tiers, ..."}
            ]
        }
    },

    "log_rationale": {
        "name": "log_rationale",
        "category": "Session Management",
        "description": "Log your thinking, rationale, or reasoning for decisions. MUST be called EXACTLY ONCE per day, immediately before calling next_day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string", "description": "Your thinking, reasoning, and decision rationale"},
                "context": {"type": "string", "description": "Optional additional context"}
            },
            "required": ["rationale"]
        },
        "parameters": {
            "rationale": {"type": "str", "description": "Your thinking, reasoning, and decision rationale for this day's actions"},
            "context": {"type": "str", "description": "Optional additional context (e.g., key metrics snapshot)"}
        },
        "returns": {
            "success": "Rationale logged: [first 100 chars]...",
            "failure": "Missing rationale text"
        },
        "output_schema": {
            "logged": "bool — always True on success",
            "_access": "result['logged'] → True"
        },
        "impact": "Records your decision-making process. MANDATORY — must be called exactly once per day before next_day.",
        "example_call": {
            "tool": "log_rationale",
            "arguments": {"rationale": "Day 15: Revenue growing at $2K/day. Increased ad spend to $500 to accelerate growth while margins are healthy. Watching churn rate — if it exceeds 5% will reduce prices."},
        },
        "sample_io": {
            "success": [
                {"label": "Daily rationale", "input": {"rationale": "Day 15: Revenue growing at $2K/day. Increased ad spend to accelerate growth."}, "output": "Rationale logged: Day 15: Revenue growing at $2K/day. Increased ad spend to accelerate growth...."}
            ]
        }
    },

    "register_script": {
        "name": "register_script",
        "category": "Automation",
        "description": "Save a named Python script for later execution via run_script. Scripts persist across days. Use to avoid re-typing complex analysis code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique name for the script (e.g. 'churn_analysis')"},
                "code": {"type": "string", "description": "Python code (same environment as python_exec)"}
            },
            "required": ["name", "code"]
        },
        "parameters": {
            "name": {"type": "str", "description": "Unique script name", "example": "cohort_analysis"},
            "code": {"type": "str", "description": "Python code (same environment as python_exec)"}
        },
        "returns": {
            "success": "Script 'cohort_analysis' registered (245 chars). Run with run_script(name='cohort_analysis').",
            "failure": "Script name cannot be empty"
        },
        "output_schema": {
            "name": "str — registered script name",
            "code_length": "int — length of code in chars",
            "_access": "result['name'] → name of registered script"
        },
        "impact": "No side effects. Just saves code for later reuse. Overwrites if name already exists.",
        "example_call": {
            "tool": "register_script",
            "arguments": {"name": "revenue_breakdown", "code": "df = pd.read_sql('SELECT plan, COUNT(*) as n, SUM(effective_price) as rev FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL GROUP BY plan', conn)\nprint(df.to_string())"}
        },
        "sample_io": {
            "success": [
                {"label": "Register analysis script", "input": {"name": "revenue_breakdown", "code": "df = pd.read_sql('SELECT plan, COUNT(*) as n, SUM(effective_price) as rev FROM subscriptions WHERE status=\"subscribed\" AND end_day IS NULL GROUP BY plan', conn)\nprint(df.to_string())"}, "output": "Script 'revenue_breakdown' registered (162 chars). Run with run_script(name='revenue_breakdown')."},
                {"label": "Overwrite existing", "input": {"name": "revenue_breakdown", "code": "print('updated')"}, "output": "Script 'revenue_breakdown' registered (16 chars, overwritten). Run with run_script(name='revenue_breakdown')."}
            ],
            "failure": [
                {"label": "Empty name", "input": {"name": "", "code": "print('hi')"}, "output": "Script name cannot be empty"}
            ]
        }
    },

    "run_script": {
        "name": "run_script",
        "category": "Automation",
        "description": "Execute a previously registered script by name. Runs in the same environment as python_exec.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the registered script to run"}
            },
            "required": ["name"]
        },
        "parameters": {
            "name": {"type": "str", "description": "Name of registered script", "example": "revenue_breakdown"}
        },
        "returns": {
            "success": "(script output)",
            "failure": "Script 'X' not found. Registered: ['a', 'b']"
        },
        "output_schema": {
            "_note": "Returns the same output as python_exec — stdout of executed code",
            "_access": "result is the printed output from the script"
        },
        "impact": "Executes code in same environment as python_exec (read-only DB access, pandas/numpy available).",
        "example_call": {
            "tool": "run_script",
            "arguments": {"name": "revenue_breakdown"}
        },
        "sample_io": {
            "success": [
                {"label": "Run registered script", "input": {"name": "revenue_breakdown"}, "output": "  plan  n     rev\n0    A  42  1260.0\n1    B  28  2212.0\n2    C  15  2985.0"}
            ],
            "failure": [
                {"label": "Script not found", "input": {"name": "missing_script"}, "output": "Script 'missing_script' not found. Registered scripts: []"}
            ]
        }
    },

    "list_scripts": {
        "name": "list_scripts",
        "category": "Automation",
        "description": "List all registered scripts with code previews.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "parameters": {},
        "returns": {
            "success": "Registered scripts:\n  • script_name: first 100 chars of code...",
            "failure": None
        },
        "output_schema": {
            "scripts": "List[str] — names of registered scripts",
            "_access": "for name in result['scripts']: print(name)"
        },
        "impact": "Read-only. No cost.",
        "example_call": {
            "tool": "list_scripts",
            "arguments": {}
        },
        "sample_io": {
            "success": [
                {"label": "List scripts", "input": {}, "output": "Registered scripts:\n  • revenue_breakdown: df = pd.read_sql('SELECT plan, COUNT(*) as n, SUM(effective_price)..."},
                {"label": "No scripts", "input": {}, "output": "No scripts registered."}
            ]
        }
    },

    "delete_script": {
        "name": "delete_script",
        "category": "Automation",
        "description": "Delete a previously registered script by name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the script to delete"}
            },
            "required": ["name"]
        },
        "parameters": {
            "name": {"type": "str", "description": "Name of script to delete", "example": "revenue_breakdown"}
        },
        "returns": {
            "success": "Script 'revenue_breakdown' deleted.",
            "failure": "Script 'X' not found. Registered scripts: ['a', 'b']"
        },
        "output_schema": {
            "name": "str — deleted script name",
            "remaining": "List[str] — remaining script names",
            "_access": "result['remaining'] → list of still-registered scripts"
        },
        "impact": "Removes saved script. No other side effects.",
        "example_call": {
            "tool": "delete_script",
            "arguments": {"name": "revenue_breakdown"}
        },
        "sample_io": {
            "success": [
                {"label": "Delete script", "input": {"name": "revenue_breakdown"}, "output": "Script 'revenue_breakdown' deleted."}
            ],
            "failure": [
                {"label": "Script not found", "input": {"name": "missing"}, "output": "Script 'missing' not found. Registered scripts: ['revenue_breakdown', 'churn_analysis']"}
            ]
        }
    },

    "set_ads_strength": {
        "name": "set_ads_strength",
        "category": "Business Configuration",
        "description": "Set in-app advertising strength (0-1). Ads generate revenue but reduce perceived quality. Effects at global/group/individual levels are ADDITIVE, capped at 1.0 per customer. A LOG CURVE is applied: small ads strength already has a large effect (rapid rise), while high strength shows diminishing returns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "global_strength": {"type": "number", "description": "Global ads strength for all users (0-1). NULL = no change."},
                "by_group": {"type": "object", "description": "Per-group ads strength: {group_id: strength}. Additive with global."},
                "by_customer": {"type": "object", "description": "Per-customer ads strength: {customer_id: strength}. Additive with global + group."}
            }
        },
        "parameters": {
            "global_strength": {"type": "float", "description": "Global ads strength (0.0-1.0). NULL/omit = no change."},
            "by_group": {"type": "dict", "description": "Per-group ads strength: {group_id: float}"},
            "by_customer": {"type": "dict", "description": "Per-customer ads strength: {customer_id_as_str: float}"}
        },
        "returns": {
            "success": "Ads strength updated. Global: 0.30, Groups: {E1: 0.10}, Customers: {}",
            "failure": "Invalid group IDs / Strength must be between 0 and 1"
        },
        "output_schema": {
            "global": "float — global ads strength (0-1)",
            "by_group": "Dict[str, float] — per-group ads strength",
            "by_customer": "Dict[str, float] — per-customer ads strength (customer_id as str key)",
            "_access": "result['global'] → current global ads strength"
        },
        "impact": "Each customer's quality_penalty = ads_quality_sensitivity × log_scaled_effective_ads (degrades satisfaction). Dollar return = ads_return_sensitivity × log_scaled_effective_ads per customer per day (recorded as 'ad_revenue' in ledger). Log scaling: effective = log(1+9*x)/log(10), so strength 0.1 → 0.40 effective, 0.5 → 0.74, 1.0 → 1.0. Trade-off: higher ads → more revenue but lower satisfaction → more churn. Diminishing returns at high strength.",
        "example_call": {
            "tool": "set_ads_strength",
            "arguments": {"global_strength": 0.2, "by_group": {"S1": 0.1}}
        }
    },

    "set_lead_promotion": {
        "name": "set_lead_promotion",
        "category": "Business Configuration",
        "description": "Set promotion (dollar deduction) for new leads. Applied automatically to first billing period only. Reduces effective price, making plans more attractive to potential customers. Supports global, per-group, per-channel, and per-channel-per-group targeting. All levels are ADDITIVE.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "global_promotion": {"type": "number", "description": "Global lead promotion in $/month (deducted from first billing). NULL = no change."},
                "by_group": {"type": "object", "description": "Per-group lead promotion: {group_id: $/month}. Additive with global."},
                "by_channel": {"type": "object", "description": "Per-channel lead promotion: {channel_id: $/month}. Additive with global + group. Only applies to leads from that channel."},
                "by_channel_group": {"type": "object", "description": "Per-channel-per-group: {channel_id: {group_id: $/month}}. Most granular level. Additive with all other levels."}
            }
        },
        "parameters": {
            "global_promotion": {"type": "float", "description": "Global lead promotion in $/month. NULL/omit = no change."},
            "by_group": {"type": "dict", "description": "Per-group lead promotion: {group_id: float}"},
            "by_channel": {"type": "dict", "description": "Per-channel lead promotion: {channel_id: float}. Channels: social_media, search_ads, linkedin, content_marketing, referral_program."},
            "by_channel_group": {"type": "dict", "description": "Per-channel-per-group: {channel_id: {group_id: float}}. Most granular targeting."}
        },
        "returns": {
            "success": "Lead promotion updated. Global: $10.00/mo, Groups: {S1: $5.00}, Channels: {linkedin: $8.00}, Channel×Group: {linkedin→E1: $15.00}",
            "failure": "Invalid group IDs / Invalid channels / Promotion must be non-negative"
        },
        "output_schema": {
            "global": "float — global lead promotion $/mo",
            "by_group": "Dict[str, float] — per-group promotion",
            "by_channel": "Dict[str, float] — per-channel promotion",
            "by_channel_group": "Dict[str, Dict[str, float]] — {channel: {group: $/mo}}",
            "_access": "result['global'] → current global lead promotion"
        },
        "impact": "Reduces effective price for new leads at first billing period. Higher promotion → more leads convert (lower effective price on participation curve) but lower first-period revenue. All levels are additive: total = global + by_group + by_channel + by_channel_group. Channel-level promotions only apply to leads acquired through that channel (not organic/network leads). Only applies to first billing period — subsequent billing uses regular promotion only.",
        "example_call": {
            "tool": "set_lead_promotion",
            "arguments": {"global_promotion": 5.0, "by_channel": {"linkedin": 10.0}, "by_channel_group": {"social_media": {"S1": 8.0}}}
        },
        "sample_io": {
            "success": [
                {"label": "Global + group", "input": {"global_promotion": 10.0, "by_group": {"S1": 5.0}}, "output": "Lead promotion updated. Global: $10.00/mo, Groups: {S1: $5.00}\n\nEffect: New leads see (list_price - lead_promotion) as effective price.\nApplied automatically at first billing period only.\nAll levels (global + group + channel + channel×group) are additive."},
                {"label": "Channel-only", "input": {"by_channel": {"linkedin": 15.0, "search_ads": 10.0}}, "output": "Lead promotion updated. Global: $0.00/mo, Channels: {linkedin: $15.00, search_ads: $10.00}\n\nEffect: New leads see (list_price - lead_promotion) as effective price.\nApplied automatically at first billing period only.\nAll levels (global + group + channel + channel×group) are additive."},
                {"label": "Channel×Group targeting", "input": {"by_channel_group": {"linkedin": {"E1": 20.0, "E2": 15.0}, "social_media": {"S1": 8.0}}}, "output": "Lead promotion updated. Global: $0.00/mo, Channel×Group: {linkedin→E1: $20.00, linkedin→E2: $15.00, social_media→S1: $8.00}\n\nEffect: New leads see (list_price - lead_promotion) as effective price.\nApplied automatically at first billing period only.\nAll levels (global + group + channel + channel×group) are additive."},
                {"label": "All levels combined", "input": {"global_promotion": 3.0, "by_group": {"E1": 5.0}, "by_channel": {"linkedin": 10.0}, "by_channel_group": {"linkedin": {"E1": 7.0}}}, "output": "Lead promotion updated. Global: $3.00/mo, Groups: {E1: $5.00}, Channels: {linkedin: $10.00}, Channel×Group: {linkedin→E1: $7.00}\n\nEffect: New leads see (list_price - lead_promotion) as effective price.\nApplied automatically at first billing period only.\nAll levels (global + group + channel + channel×group) are additive."}
            ],
            "failure": [
                {"label": "Invalid channel", "input": {"by_channel": {"tiktok": 10.0}}, "output": "Invalid channels: {'tiktok'}. Valid: ['content_marketing', 'linkedin', 'referral_program', 'search_ads', 'social_media']"},
                {"label": "Invalid group in channel_group", "input": {"by_channel_group": {"linkedin": {"INVALID": 10.0}}}, "output": "Invalid group IDs for channel 'linkedin': {'INVALID'}. Valid: [...]"},
                {"label": "Negative promotion", "input": {"global_promotion": -5.0}, "output": "Global lead promotion must be non-negative"}
            ]
        }
    },

    "set_promotion": {
        "name": "set_promotion",
        "category": "Business Configuration",
        "description": "Set ongoing promotion (dollar deduction) for existing subscribers. Applied at each billing period. Satisfaction uses (price - promotion). Additive across global/group/customer/group_plan levels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "global_promotion": {"type": "number", "description": "Global promotion in $/month for all users. NULL = no change."},
                "by_group": {"type": "object", "description": "Per-group promotion: {group_id: $/month}. Additive with global."},
                "by_customer": {"type": "object", "description": "Per-customer promotion: {customer_id: $/month}. Additive with global + group."},
                "by_group_plan": {"type": "object", "description": "Per-group-plan promotion: {group_id: {plan: $/month}}. Additive with all other levels."}
            }
        },
        "parameters": {
            "global_promotion": {"type": "float", "description": "Global promotion in $/month. NULL/omit = no change."},
            "by_group": {"type": "dict", "description": "Per-group promotion: {group_id: float}"},
            "by_customer": {"type": "dict", "description": "Per-customer promotion: {customer_id_as_str: float}"},
            "by_group_plan": {"type": "dict", "description": "Per-group-plan: {group_id: {plan: float}}"}
        },
        "returns": {
            "success": "Promotion updated. Global: $5.00/mo, Groups: {E1: $10.00}, Customers: {}, Group-Plans: {}",
            "failure": "Invalid group IDs / Invalid plan names / Promotion must be non-negative"
        },
        "output_schema": {
            "global": "float — global promotion $/mo",
            "by_group": "Dict[str, float] — per-group promotion",
            "by_customer": "Dict[str, float] — per-customer promotion (customer_id as str key)",
            "by_group_plan": "Dict[str, Dict[str, float]] — {group: {plan: $/mo}}",
            "_access": "result['global'] → current global promotion"
        },
        "impact": "Satisfaction uses (price - promotion) as effective price. Customers evaluate plans at (list_price - promotion) on billing day. Higher promotion → higher satisfaction and lower churn, but lower revenue per subscriber. Takes effect at next billing period for each customer.",
        "example_call": {
            "tool": "set_promotion",
            "arguments": {"global_promotion": 5.0, "by_group": {"E1": 10.0}, "by_group_plan": {"S1": {"A": 3.0}}}
        }
    },
}


@dataclass
class ToolResult:
    """Result from executing a tool.

    Convention:
    - Success: ``data`` carries ALL structured output.  ``message`` is ignored
      by ``to_json`` on success (kept internally for legacy callers / logging).
    - Failure: ``message`` becomes the ``error`` field in JSON **and** is
      automatically printed to stderr so the agent process always sees it.
    """
    success: bool
    message: str
    data: Optional[Dict] = None

    def __post_init__(self):
        import sys as _sys
        if not self.success and self.message:
            print(self.message, file=_sys.stderr)

    def to_json(self) -> Dict[str, Any]:
        """Return structured JSON representation.

        On success: {"success": true, "data": {...}}
        On failure: {"success": false, "error": "...", "data": null}
        """
        result: Dict[str, Any] = {"success": self.success}
        if self.success:
            result["data"] = self.data if self.data is not None else {}
        else:
            result["error"] = self.message
            result["data"] = None
        return result

    def to_stdout(self) -> str:
        """Return JSON string for stdout (structured data only)."""
        return json.dumps(self.to_json(), default=str)

    def to_stderr(self) -> str:
        """Return error string for stderr (errors only, empty on success)."""
        if not self.success:
            return self.message
        return ""


class AgentTools:
    """Tools available to the agent."""

    def __init__(self, conn: sqlite3.Connection, current_day: int, workspace_path: Path, db_path: Path, rng: Optional[Generator] = None, config: Optional[BenchmarkConfig] = None):
        self.conn = conn
        self.current_day = current_day
        self.workspace_path = workspace_path
        self.db_path = db_path  # Store absolute path to database
        self.rng = rng if rng is not None else default_rng()  # RNG for scheduling customer replies
        self.config = config or BenchmarkConfig()  # V2: needed for VC negotiation
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.event_logger = None  # Optional event logger

    def set_event_logger(self, event_logger):
        """Set the event logger for detailed logging."""
        self.event_logger = event_logger

    def set_current_day(self, day: int):
        """Update the current day."""
        self.current_day = day

    def set_prices(self, prices: Dict[str, float]) -> ToolResult:
        """Set prices for plans A, B, C. Only provided keys are changed.

        Args:
            prices: Dict with any subset of keys 'A', 'B', 'C' and float values.
                    Omitted plans keep their current prices.
        """
        if not prices:
            return ToolResult(False, "Must provide at least one plan price")

        valid_keys = {'A', 'B', 'C'}
        invalid = set(prices.keys()) - valid_keys
        if invalid:
            return ToolResult(False, f"Invalid plan keys: {invalid}. Valid: {valid_keys}")

        for plan, val in prices.items():
            if val is None or val <= 0:
                return ToolResult(False, f"Price for plan {plan} must be positive")

        # Get current config and update
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if current:
            new_a = prices.get('A', current['price_A'])
            new_b = prices.get('B', current['price_B'])
            new_c = prices.get('C', current['price_C'])

            self.conn.execute("""
                INSERT OR REPLACE INTO config_history (
                    day, price_A, price_B, price_C,
                    tier_A, tier_B, tier_C,
                    spend_advertising, spend_operations, spend_development,
                    capacity_tier,
                    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                    ad_spend_content_marketing, ad_spend_referral_program,
                    quota_A, quota_B, quota_C
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_day,
                new_a, new_b, new_c,
                current['tier_A'], current['tier_B'], current['tier_C'],
                current['spend_advertising'], current['spend_operations'],
                current['spend_development'], current['capacity_tier'],
                current['ad_spend_social_media'], current['ad_spend_search_ads'],
                current['ad_spend_linkedin'], current['ad_spend_content_marketing'],
                current['ad_spend_referral_program'],
                current['quota_A'], current['quota_B'], current['quota_C']
            ))
            self.conn.commit()

        changed = ', '.join(f"{k}=${v:.2f}" for k, v in sorted(prices.items()))
        return ToolResult(True, f"Prices updated: {changed}", {
            'updated': {k: v for k, v in prices.items()},
            'current': {'A': new_a, 'B': new_b, 'C': new_c},
        })

    def set_model_tiers(self, tiers: Dict[str, int]) -> ToolResult:
        """Set model tiers for plans A, B, C. Only provided keys are changed.

        Args:
            tiers: Dict with any subset of keys 'A', 'B', 'C' and int values 1-5.
                   Omitted plans keep their current tiers.
        """
        if not tiers:
            return ToolResult(False, "Must provide at least one plan tier")

        valid_keys = {'A', 'B', 'C'}
        invalid = set(tiers.keys()) - valid_keys
        if invalid:
            return ToolResult(False, f"Invalid plan keys: {invalid}. Valid: {valid_keys}")

        for plan, val in tiers.items():
            if val not in MODEL_TIERS:
                return ToolResult(False, f"Tier for plan {plan} must be 1-5")

        # Get current config and update
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if current:
            new_a = tiers.get('A', current['tier_A'])
            new_b = tiers.get('B', current['tier_B'])
            new_c = tiers.get('C', current['tier_C'])

            self.conn.execute("""
                INSERT OR REPLACE INTO config_history (
                    day, price_A, price_B, price_C,
                    tier_A, tier_B, tier_C,
                    spend_advertising, spend_operations, spend_development,
                    capacity_tier,
                    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                    ad_spend_content_marketing, ad_spend_referral_program,
                    quota_A, quota_B, quota_C
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_day,
                current['price_A'], current['price_B'], current['price_C'],
                new_a, new_b, new_c,
                current['spend_advertising'], current['spend_operations'],
                current['spend_development'], current['capacity_tier'],
                current['ad_spend_social_media'], current['ad_spend_search_ads'],
                current['ad_spend_linkedin'], current['ad_spend_content_marketing'],
                current['ad_spend_referral_program'],
                current['quota_A'], current['quota_B'], current['quota_C']
            ))
            self.conn.commit()

        changed = ', '.join(f"{k}=tier{v}" for k, v in sorted(tiers.items()))
        return ToolResult(True, f"Model tiers updated: {changed}", {
            'updated': {k: v for k, v in tiers.items()},
            'current': {'A': new_a, 'B': new_b, 'C': new_c},
        })

    def set_daily_spend(self, spend: Dict[str, float]) -> ToolResult:
        """Set daily spending for advertising, operations, development. Only provided keys are changed.

        Args:
            spend: Dict with any subset of keys 'advertising', 'operations', 'development'.
                   Omitted categories keep their current values.
        """
        if not spend:
            return ToolResult(False, "Must provide at least one spend category")

        valid_keys = {'advertising', 'operations', 'development'}
        invalid = set(spend.keys()) - valid_keys
        if invalid:
            return ToolResult(False, f"Invalid keys: {invalid}. Valid: {valid_keys}")

        for category, val in spend.items():
            if val is None or val < 0:
                return ToolResult(False, f"Spend for {category} cannot be negative")

        # Get current config and update
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if current:
            new_adv = spend.get('advertising', current['spend_advertising'])
            new_ops = spend.get('operations', current['spend_operations'])
            new_dev = spend.get('development', current['spend_development'])

            self.conn.execute("""
                INSERT OR REPLACE INTO config_history (
                    day, price_A, price_B, price_C,
                    tier_A, tier_B, tier_C,
                    spend_advertising, spend_operations, spend_development,
                    capacity_tier,
                    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                    ad_spend_content_marketing, ad_spend_referral_program,
                    quota_A, quota_B, quota_C
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_day,
                current['price_A'], current['price_B'], current['price_C'],
                current['tier_A'], current['tier_B'], current['tier_C'],
                new_adv, new_ops, new_dev,
                current['capacity_tier'],
                current['ad_spend_social_media'], current['ad_spend_search_ads'],
                current['ad_spend_linkedin'], current['ad_spend_content_marketing'],
                current['ad_spend_referral_program'],
                current['quota_A'], current['quota_B'], current['quota_C']
            ))
            self.conn.commit()

        changed = ', '.join(f"{k}=${v:.0f}" for k, v in sorted(spend.items()))
        return ToolResult(True, f"Daily spend updated: {changed}", {
            'updated': {k: v for k, v in spend.items()},
            'current': {'advertising': new_adv, 'operations': new_ops, 'development': new_dev},
        })

    def set_ad_channel_spend(self, channel_percentages: Dict[str, float]) -> ToolResult:
        """Set per-channel advertising budget allocation as percentages.

        Args:
            channel_percentages: Dict with channel names and percentage allocations (0.0 to 1.0).
                                 Only provided channels are updated; unprovided channels keep current allocation.
                                 After merging, all values are normalized to sum to 1.0.
                                 Channels: social_media, search_ads, linkedin, content_marketing, referral_program
        """
        if not channel_percentages:
            return ToolResult(False, "Must provide at least one channel allocation")

        valid_channels = set(AD_CHANNELS.keys())
        provided_channels = set(channel_percentages.keys())

        # Validate channels
        invalid = provided_channels - valid_channels
        if invalid:
            return ToolResult(
                False,
                f"Invalid channels: {invalid}. Valid: {valid_channels}"
            )

        # Validate percentages are non-negative
        for channel, pct in channel_percentages.items():
            if pct is None or pct < 0:
                return ToolResult(False, f"Percentage for {channel} must be non-negative")

        # Get current config
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if not current:
            return ToolResult(False, "No current configuration found")

        # Get total advertising budget
        total_advertising = current['spend_advertising']

        # Compute current percentages from existing spend amounts
        current_spend = {
            'social_media': current['ad_spend_social_media'],
            'search_ads': current['ad_spend_search_ads'],
            'linkedin': current['ad_spend_linkedin'],
            'content_marketing': current['ad_spend_content_marketing'],
            'referral_program': current['ad_spend_referral_program'],
        }
        current_total = sum(current_spend.values())
        if current_total > 0:
            current_pcts = {k: v / current_total for k, v in current_spend.items()}
        else:
            # If no prior allocation, default to equal split
            current_pcts = {k: 0.2 for k in current_spend}

        # Merge: provided channels override, others keep current percentage
        percentages = {k: channel_percentages.get(k, current_pcts[k]) for k in valid_channels}

        # Normalize to sum to 1.0
        total_pct = sum(percentages.values())
        if total_pct == 0:
            return ToolResult(False, "At least one channel must have a non-zero percentage")

        normalized = {k: v / total_pct for k, v in percentages.items()}

        # Convert percentages to actual spend amounts
        new_spend = {k: v * total_advertising for k, v in normalized.items()}

        # Update config with all values
        self.conn.execute("""
            INSERT OR REPLACE INTO config_history (
                day, price_A, price_B, price_C,
                tier_A, tier_B, tier_C,
                spend_advertising, spend_operations, spend_development,
                capacity_tier,
                ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                ad_spend_content_marketing, ad_spend_referral_program,
                quota_A, quota_B, quota_C
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.current_day,
            current['price_A'], current['price_B'], current['price_C'],
            current['tier_A'], current['tier_B'], current['tier_C'],
            total_advertising, current['spend_operations'], current['spend_development'],
            current['capacity_tier'],
            new_spend['social_media'], new_spend['search_ads'], new_spend['linkedin'],
            new_spend['content_marketing'], new_spend['referral_program'],
            current['quota_A'], current['quota_B'], current['quota_C']
        ))
        self.conn.commit()

        # Build summary — show only changed channels
        summary_parts = []
        for channel in sorted(channel_percentages.keys()):
            channel_name = AD_CHANNELS[channel].name
            pct = normalized[channel]
            amount = new_spend[channel]
            summary_parts.append(f"{channel_name}: {pct:.0%} (${amount:.0f}/day)")

        return ToolResult(
            True,
            f"Ad channel allocation updated (total budget=${total_advertising:.0f}/day):\n" +
            '\n'.join(f"  • {p}" for p in summary_parts),
            {'allocations': {k: round(v, 4) for k, v in normalized.items()},
             'spend': {k: round(v, 2) for k, v in new_spend.items()},
             'total_budget': total_advertising},
        )

    def set_targeted_ad_spend(self, targeted_spend: Dict[str, Dict[str, float]]) -> ToolResult:
        """Set additional per-group per-channel ad spend.

        This is ADDITIONAL to the overall channel allocation (set_ad_channel_spend).
        Use it to boost spend on specific (channel, group) pairs.

        Args:
            targeted_spend: {channel_id: {group_id: additional_dollars_per_day}}
                Example: {"linkedin": {"E1": 200, "E2": 100}}
        """
        valid_channels = set(AD_CHANNELS.keys())

        # Build set of valid group IDs (initial + discovered only, not undiscovered)
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))

        # Validate channels (outer keys must be channel names)
        invalid_channels = set(targeted_spend.keys()) - valid_channels
        if invalid_channels:
            # Detect common mistake: agent passes {group: {channel: amount}} instead of {channel: {group: amount}}
            if invalid_channels & valid_groups:
                # The invalid "channels" are actually group IDs — nesting is inverted, auto-fix it
                transposed = {}
                for group_id, channels_dict in targeted_spend.items():
                    if isinstance(channels_dict, dict):
                        for ch, amt in channels_dict.items():
                            if ch not in transposed:
                                transposed[ch] = {}
                            transposed[ch][group_id] = amt
                # Validate the transposed version
                invalid_after = set(transposed.keys()) - valid_channels
                if not invalid_after:
                    targeted_spend = transposed
                else:
                    return ToolResult(
                        False,
                        f"Wrong nesting order. You passed {{group: {{channel: amount}}}} "
                        f"but the format is {{channel: {{group: amount}}}}. "
                        f"Example: {{'linkedin': {{'E1': 200, 'E2': 100}}}}"
                    )
            else:
                return ToolResult(
                    False,
                    f"Invalid channels: {invalid_channels}. Valid: {sorted(valid_channels)}. "
                    f"Format: {{channel: {{group: amount}}}}. "
                    f"Example: {{'linkedin': {{'E1': 200}}}}"
                )

        # Validate group IDs and amounts
        for channel_id, groups in targeted_spend.items():
            if not isinstance(groups, dict):
                return ToolResult(False, f"Value for channel '{channel_id}' must be a dict of {{group_id: amount}}")
            invalid_groups = set(groups.keys()) - valid_groups
            if invalid_groups:
                return ToolResult(
                    False,
                    f"Invalid group IDs for channel '{channel_id}': {invalid_groups}. "
                    f"Valid groups: {sorted(valid_groups)}"
                )
            for group_id, amount in groups.items():
                if not isinstance(amount, (int, float)) or amount < 0:
                    return ToolResult(False, f"Amount for ({channel_id}, {group_id}) must be a non-negative number")

        # Store in config
        self.config.targeted_ad_spend = targeted_spend

        # Log to config_history (store as JSON in a comment-style approach —
        # the actual config object holds the state, no DB column needed)
        # The simulation reads from self.config.targeted_ad_spend directly

        # Calculate total extra cost
        total_extra = sum(
            sum(groups.values())
            for groups in targeted_spend.values()
        )

        # Build summary
        summary_parts = []
        for channel_id, groups in targeted_spend.items():
            channel_name = AD_CHANNELS[channel_id].name
            for group_id, amount in groups.items():
                summary_parts.append(f"  • {channel_name} → {group_id}: +${amount:.0f}/day")

        result_msg = f"Targeted ad spend updated (extra ${total_extra:.0f}/day on top of channel allocation):\n"
        if summary_parts:
            result_msg += '\n'.join(summary_parts)
        else:
            result_msg += "  (no targeted spend set — all spend comes from channel allocation)"

        record_config_override(self.conn, self.current_day, 'set_targeted_ad_spend', 'targeted_ad_spend',
                               {'targeted_spend': targeted_spend})
        return ToolResult(True, result_msg, {
            'targeted_spend': targeted_spend,
            'total_extra_per_day': total_extra,
        })

    def set_targeted_ops_spend(self, targeted_spend: Dict[str, float]) -> ToolResult:
        """Set additional per-group operations spending.

        This is ADDITIONAL to the global operations spend (set_daily_spend).
        Adds extra issue resolution capacity for targeted groups on top of the
        global resolution pool.

        Args:
            targeted_spend: {group_id: additional_dollars_per_day}
                Example: {"E1": 300, "E2": 200}
        """
        # Build set of valid group IDs (initial + discovered only, not undiscovered)
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))

        # Validate
        invalid_groups = set(targeted_spend.keys()) - valid_groups
        if invalid_groups:
            return ToolResult(
                False,
                f"Invalid group IDs: {invalid_groups}. "
                f"Valid groups: {sorted(valid_groups)}"
            )
        for group_id, amount in targeted_spend.items():
            if not isinstance(amount, (int, float)) or amount < 0:
                return ToolResult(False, f"Amount for {group_id} must be a non-negative number")

        self.config.targeted_ops_spend = targeted_spend

        total_extra = sum(targeted_spend.values())
        summary_parts = [f"  • {gid}: +${amt:.0f}/day" for gid, amt in targeted_spend.items()]
        result_msg = f"Targeted ops spend updated (extra ${total_extra:.0f}/day on top of global ops):\n"
        if summary_parts:
            result_msg += '\n'.join(summary_parts)
            result_msg += f"\n\nEffect: Each targeted group gets additional issue resolution capacity "
            result_msg += f"(extra_mean = 0.053 × spend) on top of the global resolution pool."
        else:
            result_msg += "  (no targeted ops spend — all ops spend is global)"

        record_config_override(self.conn, self.current_day, 'set_targeted_ops_spend', 'targeted_ops_spend',
                               {'targeted_spend': targeted_spend})
        return ToolResult(True, result_msg, {
            'targeted_spend': targeted_spend,
            'total_extra_per_day': total_extra,
        })

    def set_targeted_dev_spend(self, targeted_spend: Dict[str, float]) -> ToolResult:
        """Set additional per-group development spending.

        This is ADDITIONAL to the global development spend (set_daily_spend).
        Provides a CUMULATIVE per-group quality bonus that accumulates daily
        while spending continues. Like building features for specific segments —
        investment compounds over time and persists even after spending stops.

        Args:
            targeted_spend: {group_id: additional_dollars_per_day}
                Example: {"E1": 500, "S1": 200}
        """
        # Build set of valid group IDs (initial + discovered only, not undiscovered)
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))

        # Validate
        invalid_groups = set(targeted_spend.keys()) - valid_groups
        if invalid_groups:
            return ToolResult(
                False,
                f"Invalid group IDs: {invalid_groups}. "
                f"Valid groups: {sorted(valid_groups)}"
            )
        for group_id, amount in targeted_spend.items():
            if not isinstance(amount, (int, float)) or amount < 0:
                return ToolResult(False, f"Amount for {group_id} must be a non-negative number")

        self.config.targeted_dev_spend = targeted_spend

        total_extra = sum(targeted_spend.values())
        summary_parts = [f"  • {gid}: +${amt:.0f}/day" for gid, amt in targeted_spend.items()]
        result_msg = f"Targeted dev spend updated (extra ${total_extra:.0f}/day on top of global dev):\n"
        if summary_parts:
            result_msg += '\n'.join(summary_parts)
            result_msg += f"\n\nEffect: Each targeted group ACCUMULATES a quality bonus of "
            result_msg += f"0.0005 × log(1 + spend/500) per day. This compounds over time and persists if spending stops."
        else:
            result_msg += "  (no targeted dev spend — all dev spend is global)"

        record_config_override(self.conn, self.current_day, 'set_targeted_dev_spend', 'targeted_dev_spend',
                               {'targeted_spend': targeted_spend})
        return ToolResult(True, result_msg, {
            'targeted_spend': targeted_spend,
            'total_extra_per_day': total_extra,
        })

    def set_ads_strength(self, global_strength=None, by_group=None, by_customer=None) -> ToolResult:
        """Set in-app advertising strength (0-1) at global/group/individual levels.

        Effects are additive across levels, capped at 1.0 per customer.
        Ads generate revenue but reduce perceived quality.

        Args:
            global_strength: Global ads strength (0-1). None = no change.
            by_group: {group_id: strength}. None = no change.
            by_customer: {customer_id_str: strength}. None = no change.
        """
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))

        # Validate and apply global strength
        if global_strength is not None:
            if not isinstance(global_strength, (int, float)) or global_strength < 0 or global_strength > 1:
                return ToolResult(False, "Global ads strength must be between 0.0 and 1.0")
            self.config.ads_strength_global = float(global_strength)

        # Validate and apply per-group strength
        if by_group is not None:
            if not isinstance(by_group, dict):
                return ToolResult(False, "by_group must be a dict of {group_id: strength}")
            invalid_groups = set(by_group.keys()) - valid_groups
            if invalid_groups:
                return ToolResult(False, f"Invalid group IDs: {invalid_groups}. Valid: {sorted(valid_groups)}")
            for gid, val in by_group.items():
                if not isinstance(val, (int, float)) or val < 0 or val > 1:
                    return ToolResult(False, f"Strength for {gid} must be between 0.0 and 1.0")
            self.config.ads_strength_by_group = {k: float(v) for k, v in by_group.items()}

        # Validate and apply per-customer strength
        if by_customer is not None:
            if not isinstance(by_customer, dict):
                return ToolResult(False, "by_customer must be a dict of {customer_id: strength}")
            parsed = {}
            for k, v in by_customer.items():
                try:
                    cid = int(k)
                except (ValueError, TypeError):
                    return ToolResult(False, f"Customer ID '{k}' must be an integer")
                if not isinstance(v, (int, float)) or v < 0 or v > 1:
                    return ToolResult(False, f"Strength for customer {k} must be between 0.0 and 1.0")
                parsed[cid] = float(v)
            self.config.ads_strength_by_customer = parsed

        # Build summary
        parts = [f"Global: {self.config.ads_strength_global:.2f}"]
        if self.config.ads_strength_by_group:
            parts.append(f"Groups: {{{', '.join(f'{k}: {v:.2f}' for k, v in self.config.ads_strength_by_group.items())}}}")
        if self.config.ads_strength_by_customer:
            parts.append(f"Customers: {len(self.config.ads_strength_by_customer)} custom")
        record_config_override(self.conn, self.current_day, 'set_ads_strength', 'ads_strength', {
            'global': self.config.ads_strength_global,
            'by_group': self.config.ads_strength_by_group,
            'by_customer': {str(k): v for k, v in self.config.ads_strength_by_customer.items()},
        })
        return ToolResult(True, "Ads strength updated.", {
            'global': self.config.ads_strength_global,
            'by_group': self.config.ads_strength_by_group,
            'by_customer': {str(k): v for k, v in self.config.ads_strength_by_customer.items()},
        })

    def set_lead_promotion(self, global_promotion=None, by_group=None, by_channel=None, by_channel_group=None) -> ToolResult:
        """Set promotion (dollar deduction) for new leads. First billing period only.

        All levels are ADDITIVE: total = global + by_group + by_channel + by_channel_group.

        Args:
            global_promotion: Global lead promotion in $/month. None = no change.
            by_group: {group_id: $/month}. None = no change.
            by_channel: {channel_id: $/month}. None = no change.
            by_channel_group: {channel_id: {group_id: $/month}}. None = no change.
        """
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))
        valid_channels = set(AD_CHANNELS.keys())

        if global_promotion is not None:
            if not isinstance(global_promotion, (int, float)) or global_promotion < 0:
                return ToolResult(False, "Global lead promotion must be non-negative")
            self.config.lead_promotion_global = float(global_promotion)

        if by_group is not None:
            if not isinstance(by_group, dict):
                return ToolResult(False, "by_group must be a dict of {group_id: $/month}")
            invalid_groups = set(by_group.keys()) - valid_groups
            if invalid_groups:
                return ToolResult(False, f"Invalid group IDs: {invalid_groups}. Valid: {sorted(valid_groups)}")
            for gid, val in by_group.items():
                if not isinstance(val, (int, float)) or val < 0:
                    return ToolResult(False, f"Promotion for {gid} must be non-negative")
            self.config.lead_promotion_by_group = {k: float(v) for k, v in by_group.items()}

        if by_channel is not None:
            if not isinstance(by_channel, dict):
                return ToolResult(False, "by_channel must be a dict of {channel_id: $/month}")
            invalid_channels_found = set(by_channel.keys()) - valid_channels
            if invalid_channels_found:
                return ToolResult(False, f"Invalid channels: {invalid_channels_found}. Valid: {sorted(valid_channels)}")
            for ch_id, val in by_channel.items():
                if not isinstance(val, (int, float)) or val < 0:
                    return ToolResult(False, f"Promotion for channel '{ch_id}' must be non-negative")
            self.config.lead_promotion_by_channel = {k: float(v) for k, v in by_channel.items()}

        if by_channel_group is not None:
            if not isinstance(by_channel_group, dict):
                return ToolResult(False, "by_channel_group must be a dict of {channel_id: {group_id: $/month}}")
            invalid_channels_found = set(by_channel_group.keys()) - valid_channels
            if invalid_channels_found:
                return ToolResult(False, f"Invalid channels: {invalid_channels_found}. Valid: {sorted(valid_channels)}")
            parsed = {}
            for ch_id, group_dict in by_channel_group.items():
                if not isinstance(group_dict, dict):
                    return ToolResult(False, f"by_channel_group['{ch_id}'] must be a dict of {{group_id: $/month}}")
                invalid_groups = set(group_dict.keys()) - valid_groups
                if invalid_groups:
                    return ToolResult(False, f"Invalid group IDs for channel '{ch_id}': {invalid_groups}. Valid: {sorted(valid_groups)}")
                for gid, val in group_dict.items():
                    if not isinstance(val, (int, float)) or val < 0:
                        return ToolResult(False, f"Promotion for channel '{ch_id}', group '{gid}' must be non-negative")
                parsed[ch_id] = {k: float(v) for k, v in group_dict.items()}
            self.config.lead_promotion_by_channel_group = parsed

        parts = [f"Global: ${self.config.lead_promotion_global:.2f}/mo"]
        if self.config.lead_promotion_by_group:
            parts.append(f"Groups: {{{', '.join(f'{k}: ${v:.2f}' for k, v in self.config.lead_promotion_by_group.items())}}}")
        if self.config.lead_promotion_by_channel:
            parts.append(f"Channels: {{{', '.join(f'{k}: ${v:.2f}' for k, v in self.config.lead_promotion_by_channel.items())}}}")
        if self.config.lead_promotion_by_channel_group:
            ch_parts = []
            for ch_id, grp_dict in self.config.lead_promotion_by_channel_group.items():
                for gid, val in grp_dict.items():
                    ch_parts.append(f"{ch_id}→{gid}: ${val:.2f}")
            parts.append(f"Channel×Group: {{{', '.join(ch_parts)}}}")
        record_config_override(self.conn, self.current_day, 'set_lead_promotion', 'lead_promotion', {
            'global': self.config.lead_promotion_global,
            'by_group': self.config.lead_promotion_by_group,
            'by_channel': self.config.lead_promotion_by_channel,
            'by_channel_group': self.config.lead_promotion_by_channel_group,
        })
        return ToolResult(True, "Lead promotion updated.", {
            'global': self.config.lead_promotion_global,
            'by_group': self.config.lead_promotion_by_group,
            'by_channel': self.config.lead_promotion_by_channel,
            'by_channel_group': self.config.lead_promotion_by_channel_group,
        })

    def set_promotion(self, global_promotion=None, by_group=None, by_customer=None, by_group_plan=None) -> ToolResult:
        """Set ongoing promotion (dollar deduction) for existing subscribers.

        Satisfaction uses (price - promotion). Additive across levels.
        Takes effect at next billing period.

        Args:
            global_promotion: Global promotion in $/month. None = no change.
            by_group: {group_id: $/month}. None = no change.
            by_customer: {customer_id_str: $/month}. None = no change.
            by_group_plan: {group_id: {plan: $/month}}. None = no change.
        """
        valid_groups = set(INITIAL_CUSTOMER_GROUPS.keys()) | set(get_discovered_groups(self.conn))
        valid_plans = {'A', 'B', 'C'}

        if global_promotion is not None:
            if not isinstance(global_promotion, (int, float)) or global_promotion < 0:
                return ToolResult(False, "Global promotion must be non-negative")
            self.config.promotion_global = float(global_promotion)

        if by_group is not None:
            if not isinstance(by_group, dict):
                return ToolResult(False, "by_group must be a dict of {group_id: $/month}")
            invalid_groups = set(by_group.keys()) - valid_groups
            if invalid_groups:
                return ToolResult(False, f"Invalid group IDs: {invalid_groups}. Valid: {sorted(valid_groups)}")
            for gid, val in by_group.items():
                if not isinstance(val, (int, float)) or val < 0:
                    return ToolResult(False, f"Promotion for {gid} must be non-negative")
            self.config.promotion_by_group = {k: float(v) for k, v in by_group.items()}

        if by_customer is not None:
            if not isinstance(by_customer, dict):
                return ToolResult(False, "by_customer must be a dict of {customer_id: $/month}")
            parsed = {}
            for k, v in by_customer.items():
                try:
                    cid = int(k)
                except (ValueError, TypeError):
                    return ToolResult(False, f"Customer ID '{k}' must be an integer")
                if not isinstance(v, (int, float)) or v < 0:
                    return ToolResult(False, f"Promotion for customer {k} must be non-negative")
                parsed[cid] = float(v)
            self.config.promotion_by_customer = parsed

        if by_group_plan is not None:
            if not isinstance(by_group_plan, dict):
                return ToolResult(False, "by_group_plan must be a dict of {group_id: {plan: $/month}}")
            invalid_groups = set(by_group_plan.keys()) - valid_groups
            if invalid_groups:
                return ToolResult(False, f"Invalid group IDs in by_group_plan: {invalid_groups}. Valid: {sorted(valid_groups)}")
            parsed_gp = {}
            for gid, plans_dict in by_group_plan.items():
                if not isinstance(plans_dict, dict):
                    return ToolResult(False, f"Value for group {gid} must be a dict of {{plan: $/month}}")
                invalid_plans = set(plans_dict.keys()) - valid_plans
                if invalid_plans:
                    return ToolResult(False, f"Invalid plan keys for group {gid}: {invalid_plans}. Valid: {sorted(valid_plans)}")
                parsed_inner = {}
                for plan, val in plans_dict.items():
                    if not isinstance(val, (int, float)) or val < 0:
                        return ToolResult(False, f"Promotion for {gid}/{plan} must be non-negative")
                    parsed_inner[plan] = float(val)
                parsed_gp[gid] = parsed_inner
            self.config.promotion_by_group_plan = parsed_gp

        parts = [f"Global: ${self.config.promotion_global:.2f}/mo"]
        if self.config.promotion_by_group:
            parts.append(f"Groups: {{{', '.join(f'{k}: ${v:.2f}' for k, v in self.config.promotion_by_group.items())}}}")
        if self.config.promotion_by_customer:
            parts.append(f"Customers: {len(self.config.promotion_by_customer)} custom")
        if self.config.promotion_by_group_plan:
            gp_parts = []
            for gid, plans in self.config.promotion_by_group_plan.items():
                for plan, val in plans.items():
                    gp_parts.append(f"{gid}/{plan}: ${val:.2f}")
            parts.append(f"Group-Plans: {{{', '.join(gp_parts)}}}")
        record_config_override(self.conn, self.current_day, 'set_promotion', 'promotion', {
            'global': self.config.promotion_global,
            'by_group': self.config.promotion_by_group,
            'by_customer': {str(k): v for k, v in self.config.promotion_by_customer.items()},
            'by_group_plan': self.config.promotion_by_group_plan,
        })
        return ToolResult(True, "Promotion updated.", {
            'global': self.config.promotion_global,
            'by_group': self.config.promotion_by_group,
            'by_customer': {str(k): v for k, v in self.config.promotion_by_customer.items()},
            'by_group_plan': self.config.promotion_by_group_plan,
        })

    def set_capacity_tier(self, tier: int) -> ToolResult:
        """Set capacity tier (0-7).

        Args:
            tier: Capacity tier (0-7)
        """
        if tier not in CAPACITY_TIERS:
            return ToolResult(False, f"Capacity tier must be 0-7. Use get_cost_info to see all tiers.")

        # Get current config and update
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if current:
            self.conn.execute("""
                INSERT OR REPLACE INTO config_history (
                    day, price_A, price_B, price_C,
                    tier_A, tier_B, tier_C,
                    spend_advertising, spend_operations, spend_development,
                    capacity_tier,
                    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                    ad_spend_content_marketing, ad_spend_referral_program,
                    quota_A, quota_B, quota_C
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_day,
                current['price_A'], current['price_B'], current['price_C'],
                current['tier_A'], current['tier_B'], current['tier_C'],
                current['spend_advertising'], current['spend_operations'],
                current['spend_development'], tier,
                current['ad_spend_social_media'], current['ad_spend_search_ads'],
                current['ad_spend_linkedin'], current['ad_spend_content_marketing'],
                current['ad_spend_referral_program'],
                current['quota_A'], current['quota_B'], current['quota_C']
            ))
            self.conn.commit()

        cap_info = CAPACITY_TIERS[tier]
        return ToolResult(
            True,
            f"Capacity tier set to {tier}: {cap_info['capacity_units']:,} units/day, "
            f"${cap_info['cost_per_day']:,}/day",
            data={'tier': tier, 'capacity_units': cap_info['capacity_units'],
                  'cost_per_day': cap_info['cost_per_day']}
        )

    def set_usage_quotas(self, quotas: Dict[str, int]) -> ToolResult:
        """Set daily usage quotas (rate limits) for plans A, B, C.

        Each customer on a plan can use up to this many units per day.
        Exceeding quota degrades their experience (slower responses, errors).

        Args:
            quotas: Dict with keys 'A', 'B', 'C' and integer values (units/day)
        """
        # Validate
        for plan in ['A', 'B', 'C']:
            if plan not in quotas:
                return ToolResult(False, f"Missing quota for plan {plan}")
            if quotas[plan] < 0:
                return ToolResult(False, f"Quota for plan {plan} cannot be negative")

        # Get current config and update
        current = self.conn.execute(
            "SELECT * FROM config_history ORDER BY day DESC LIMIT 1"
        ).fetchone()

        if current:
            self.conn.execute("""
                INSERT OR REPLACE INTO config_history (
                    day, price_A, price_B, price_C,
                    tier_A, tier_B, tier_C,
                    spend_advertising, spend_operations, spend_development,
                    capacity_tier,
                    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
                    ad_spend_content_marketing, ad_spend_referral_program,
                    quota_A, quota_B, quota_C
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_day,
                current['price_A'], current['price_B'], current['price_C'],
                current['tier_A'], current['tier_B'], current['tier_C'],
                current['spend_advertising'], current['spend_operations'],
                current['spend_development'], current['capacity_tier'],
                current['ad_spend_social_media'], current['ad_spend_search_ads'],
                current['ad_spend_linkedin'], current['ad_spend_content_marketing'],
                current['ad_spend_referral_program'],
                quotas['A'], quotas['B'], quotas['C']
            ))
            self.conn.commit()

        return ToolResult(
            True,
            f"Usage quotas updated: A={quotas['A']:,} units/day, B={quotas['B']:,} units/day, C={quotas['C']:,} units/day",
            data={'quotas': {'A': quotas['A'], 'B': quotas['B'], 'C': quotas['C']}}
        )

    def send_enterprise_deal(self, deals: Optional[List] = None,
                             **kwargs) -> ToolResult:
        """Send structured offerings to enterprise customers.

        Compact tuple format: each deal is [customer_id, [[plan, price_per_seat, contract_months], ...]]

        Args:
            deals: List of [customer_id, offerings] tuples.
                   offerings = list of [plan, price_per_seat, contract_months] tuples.
        """
        if deals is None:
            deals = []
        if not deals:
            return ToolResult(False, "deals parameter required")
        if not isinstance(deals, list):
            deals = [deals]

        max_offerings = self.config.enterprise_max_offerings_per_turn
        results = []
        summaries = []

        # --- Phase 1: Parse all deals (no SQL) ---
        parsed_deals = []  # list of (customer_id, deal_offerings)
        for deal in deals:
            if isinstance(deal, (list, tuple)) and len(deal) >= 2:
                customer_id = deal[0]
                raw_offerings = deal[1] if len(deal) > 1 else []
            elif isinstance(deal, dict):
                customer_id = deal.get('customer_id')
                raw_offerings = deal.get('offerings', [])
            else:
                summaries.append(f"Error: invalid deal format: {deal}")
                results.append({'error': 'invalid format'})
                continue

            if customer_id is None:
                summaries.append("Error: customer_id required")
                results.append({'error': 'missing customer_id'})
                continue

            customer_id = int(customer_id)

            deal_offerings = []
            if not raw_offerings or not isinstance(raw_offerings, list):
                summaries.append(f"Customer #{customer_id}: offerings required")
                results.append({'customer_id': customer_id, 'error': 'offerings required'})
                continue

            valid = True
            for i, off in enumerate(raw_offerings):
                if isinstance(off, (list, tuple)):
                    if len(off) < 2:
                        summaries.append(f"Customer #{customer_id}: offering {i} needs at least [plan, price]")
                        results.append({'customer_id': customer_id, 'error': f'offering {i} too short'})
                        valid = False
                        break
                    deal_offerings.append({
                        'plan': str(off[0]),
                        'price_per_seat': float(off[1]),
                        'contract_months': int(off[2]) if len(off) > 2 else 1,
                    })
                elif isinstance(off, dict):
                    if 'price_per_seat' not in off and 'price' not in off:
                        summaries.append(f"Customer #{customer_id}: offering {i} must include price_per_seat")
                        results.append({'customer_id': customer_id, 'error': f'offering {i} missing price'})
                        valid = False
                        break
                    if 'contract_months' not in off:
                        off['contract_months'] = 1
                    deal_offerings.append(off)
                else:
                    summaries.append(f"Customer #{customer_id}: offering {i} invalid")
                    results.append({'customer_id': customer_id, 'error': f'offering {i} invalid'})
                    valid = False
                    break
            if not valid:
                continue

            parsed_deals.append((customer_id, deal_offerings[:max_offerings]))

        if not parsed_deals:
            # All deals failed parsing
            sent = 0
            failed = len(results)
            if len(deals) == 1 and results:
                return ToolResult(False, summaries[0], {'results': results})
            return ToolResult(False, f"Sent 0/{len(deals)} enterprise deals ({failed} failed):\n" + "\n".join(summaries), {'results': results})

        # --- Phase 2: Batch pre-fetch active threads for all customer_ids (1 query) ---
        all_cids = [cid for cid, _ in parsed_deals]
        cid_placeholders = ','.join('?' * len(all_cids))
        active_threads = {}  # customer_id -> {thread_id, thread_type, prev_turn_number, prev_customer_id, prev_thread_type, prev_current_offer_price, last_customer_day}
        if all_cids:
            rows = self.conn.execute(f"""
                SELECT et.customer_id, et.thread_id, et.thread_type,
                       et.turn_number, et.current_offer_price,
                       et.sender, et.next_reply_day
                FROM enterprise_turns et
                WHERE et.customer_id IN ({cid_placeholders})
                  AND et.message_id = (
                      SELECT MAX(et2.message_id) FROM enterprise_turns et2 WHERE et2.thread_id = et.thread_id
                  )
                  AND et.closed = 0
                  AND et._internal_status IS NULL
            """, all_cids).fetchall()
            for row in rows:
                active_threads[row['customer_id']] = {
                    'thread_id': row['thread_id'],
                    'thread_type': row['thread_type'],
                    'turn_number': row['turn_number'],
                    'current_offer_price': row['current_offer_price'],
                    'last_sender': row['sender'],
                    'next_reply_day': row['next_reply_day'],
                }

        # Batch pre-fetch last customer turn day for threads with active threads (for delay penalty)
        thread_ids_with_active = [v['thread_id'] for v in active_threads.values()]
        last_customer_days = {}  # thread_id -> max_day
        if thread_ids_with_active:
            tid_placeholders = ','.join('?' * len(thread_ids_with_active))
            rows = self.conn.execute(f"""
                SELECT thread_id, MAX(day) as max_day FROM enterprise_turns
                WHERE thread_id IN ({tid_placeholders}) AND sender = 'customer'
                GROUP BY thread_id
            """, thread_ids_with_active).fetchall()
            for row in rows:
                last_customer_days[row['thread_id']] = row['max_day']

        # Batch pre-fetch customer info for deals without active threads
        cids_no_thread = [cid for cid, _ in parsed_deals if cid not in active_threads]
        customer_info = {}  # customer_id -> row
        if cids_no_thread:
            cid_nt_placeholders = ','.join('?' * len(cids_no_thread))
            rows = self.conn.execute(f"""
                SELECT c.customer_id, c.customer_type, c.seat_count, c.email,
                       s.plan, s.listed_price, s.promotion, s.effective_price, s.status, s.contract_months, s.contract_end_day
                FROM customers c
                LEFT JOIN subscriptions s ON c.customer_id = s.customer_id
                    AND s.status = 'subscribed' AND s.end_day IS NULL
                WHERE c.customer_id IN ({cid_nt_placeholders})
            """, cids_no_thread).fetchall()
            for row in rows:
                customer_info[row['customer_id']] = row

        # --- Phase 3: Process deals using pre-fetched data (minimal per-deal SQL) ---
        threads_to_schedule = []  # (thread_id,) for schedule_customer_reply at end

        for customer_id, deal_offerings in parsed_deals:
            at = active_threads.get(customer_id)

            if at:
                # --- Reply to existing open thread ---
                tid = at['thread_id']

                # Check response delay — apply relationship penalty for late replies
                # (only when responding to a customer message, not when re-sending)
                if at['last_sender'] != 'agent':
                    max_day = last_customer_days.get(tid)
                    if max_day is not None:
                        delay_days = self.current_day - max_day
                        if delay_days > 1:
                            penalty = -0.02 * (delay_days - 1)
                            # Inline relationship update (no commit)
                            self.conn.execute("""
                                UPDATE customer_state
                                SET relationship = MAX(0.0, MIN(1.0, relationship + ?))
                                WHERE customer_id = ?
                            """, (penalty, customer_id))

                # Insert agent turn — always inserted even if repeating
                new_turn_number = at['turn_number'] + 1
                self.conn.execute("""
                    INSERT INTO enterprise_turns
                    (thread_id, customer_id, thread_type, turn_number, sender, message_text,
                     offer_json, day, next_reply_day, current_offer_price, email,
                     closed, close_reason, _internal_status)
                    VALUES (?, ?, ?, ?, 'agent', '', ?, ?, NULL, ?, '', 0, '', NULL)
                """, (tid, customer_id, at['thread_type'], new_turn_number,
                      json.dumps(deal_offerings), self.current_day, at['current_offer_price']))

                threads_to_schedule.append(tid)
                summaries.append(f"Customer #{customer_id}: reply sent with {len(deal_offerings)} offering(s)")
                results.append({'customer_id': customer_id, 'success': True})

            else:
                # --- No open thread: initiate renegotiation ---
                customer = customer_info.get(customer_id)
                if not customer:
                    summaries.append(f"Customer #{customer_id}: not found")
                    results.append({'customer_id': customer_id, 'error': 'not found'})
                    continue

                if customer['customer_type'] != 'large':
                    summaries.append(f"Customer #{customer_id}: not an enterprise customer")
                    results.append({'customer_id': customer_id, 'error': 'not enterprise'})
                    continue

                if not customer['plan']:
                    summaries.append(f"Customer #{customer_id}: no active subscription")
                    results.append({'customer_id': customer_id, 'error': 'no subscription'})
                    continue

                from .database import create_enterprise_thread
                tid, init_message_id = create_enterprise_thread(
                    self.conn, customer_id, 'renegotiation',
                    self.current_day, sender='agent',
                    offer_json=json.dumps(deal_offerings),
                )

                threads_to_schedule.append(tid)

                contract_info = ""
                if customer['contract_end_day']:
                    days_left = customer['contract_end_day'] - self.current_day
                    contract_info = f", contract: {days_left}d remaining"

                summaries.append(
                    f"Customer #{customer_id}: renegotiation started "
                    f"({int(customer['seat_count'] or 1)} seats{contract_info})")
                results.append({'customer_id': customer_id, 'success': True})

        # --- Phase 4: Batch schedule customer replies + single commit ---
        batch_schedule_customer_replies(self.conn, threads_to_schedule, self.current_day, self.rng)
        self.conn.commit()

        # Return format — always include per-item results in data
        sent = sum(1 for r in results if r.get('success'))
        failed = len(results) - sent
        if len(deals) == 1:
            ok = bool(results and results[0].get('success'))
            return ToolResult(ok, summaries[0], {'results': results})

        summary = f"Sent {sent}/{len(deals)} enterprise deals"
        if failed:
            summary += f" ({failed} failed)"
        return ToolResult(sent > 0, summary + ":\n" + "\n".join(summaries), {'results': results})

    def python_exec(self, code: str, timeout_seconds: float = 600.0) -> ToolResult:
        """Execute Python code for data analysis.

        This is your primary analytics tool. The database contains all business data.
        Use this for any analysis that isn't covered by other tools.

        IMPORTANT - STATELESS EXECUTION:
        --------------------------------
        Each call runs in a FRESH context. Variables from previous calls do NOT persist.
        You must re-query data in each call.

        IMPORTANT - USE PRE-LOADED VARIABLES:
        -------------------------------------
        A database connection `conn` is ALREADY connected to the world database.
        DO NOT create your own sqlite3 connection - just use `conn` directly!

        WRONG (creates connection to wrong/empty database):
            conn = sqlite3.connect('some/path/database.db')  # DON'T DO THIS!

        CORRECT (use pre-loaded conn):
            print(rows("SELECT * FROM customers LIMIT 5"))
            df = pd.read_sql("SELECT * FROM ledger", conn)

        AVAILABLE TABLES:
        -----------------

        customers - All customer records
          • customer_id: Unique identifier for the customer
          • created_day: Simulation day when customer was created
          • email: Customer email address for communication
          • customer_type: Category like 'startup', 'smb', 'enterprise'
          • group_id: Customer segment group identifier (e.g., 'S1', 'S2', 'E1')
          • persona_description: Text describing customer's behavior and preferences

        subscriptions - Subscription records linking customers to plans
          • subscription_id: Unique identifier for the subscription
          • customer_id: Foreign key to customers table
          • plan: Plan tier - 'A' (basic), 'B' (standard), 'C' (premium)
          • listed_price: List price per seat (before promotions)
          • promotion: Total promotion $ currently applied
          • effective_price: Actual price per seat = listed_price - promotion
          • status: Current state - 'lead', 'subscribed', 'cancelled', 'lost'
          • start_day: Day subscription/lead record started
          • end_day: Day subscription ended (NULL if active)
          • billing_day_mod30: Day of month for billing (0-29)

        daily_usage - Per-customer daily usage metrics
          • day: Simulation day
          • customer_id: Foreign key to customers table
          • usage_units: Number of compute units consumed that day

        ledger - All financial transactions (revenue and expenses)
          • id: Transaction ID
          • day: Simulation day of transaction
          • category: Type of transaction:
              - 'subscription_payment': Revenue from customer payments (positive)
              - 'compute': Variable compute costs (negative)
              - 'capacity': Fixed capacity/infrastructure costs (negative)
              - 'advertising': Marketing spend (negative)
              - 'operations': Operational costs (negative)
              - 'development': R&D/development costs (negative)
          • amount: Dollar amount (positive=revenue, negative=expense)
          • note: Description of the transaction

        service_day - Daily service quality metrics
          • day: Simulation day
          • total_usage_units: Total compute units used across all customers
          • p95_ms: 95th percentile response latency in milliseconds
          • error_rate: Fraction of requests that failed (0.0-1.0)
          • downtime_minutes: Minutes of service unavailability
          • capacity_tier: Current infrastructure tier (0-7, higher = more capacity)
          • capacity_units: Maximum compute units the infrastructure can handle

        config_history - Historical record of configuration changes
          • day: Simulation day
          • price_A, price_B, price_C: Prices for each plan tier
          • tier_A, tier_B, tier_C: Feature tiers for each plan
          • spend_advertising, spend_development, spend_operations: Daily spending amounts
          • capacity_tier: Infrastructure tier setting
          • quota_A, quota_B, quota_C: Usage quotas for each plan

        social_media_posts - Customer posts on social media about the service
          • post_id: Unique identifier
          • day: Simulation day posted
          • customer_id: Foreign key to customers table
          • content: Text content of the post
          • likes: Number of likes received
          • shares: Number of shares/retweets
          • virality_score: Computed virality metric (higher = more viral)

        enterprise_turns - Enterprise customer negotiation turns (each row = one turn in a conversation)
          • message_id: Unique identifier
          • thread_id: Thread identifier (groups turns in the same negotiation)
          • customer_id: Foreign key to customers table
          • thread_type: Why thread was created - 'new_lead', 'plan_change', 'churn_prevention', 'general'
          • turn_number: Sequential turn number within thread (0, 1, 2, ...)
          • sender: Who sent it - 'customer', 'agent', or 'system'
          • message_text: Text content (nullable - inbound customer messages have text, agent replies are structural only)
          • offer_json: JSON with offer details if this turn contains an offer
          • closed: 0=open, 1=terminal (thread is closed)
          • close_reason: NULL while open; 'accepted' or 'agent_rejected' when closed
          • day: Simulation day
          • email: Email address if sent via email

        vc_turns - VC negotiation turns (each row = one turn in a VC negotiation)
          • message_id: Unique identifier (= message_id visible to agent)
          • shareholder_id: Foreign key to shareholders table (groups all turns for this VC)
          • turn_number: Sequential turn number (0, 1, 2, ...)
          • sender: Who sent it - 'vc', 'agent', or 'system'
          • message_text: Text content (nullable)
          • offer_json: JSON with offer details
          • closed: 0=open, 1=terminal (thread is closed)
          • close_reason: NULL while open; 'accepted', 'agent_rejected', or 'settled' when closed
          • current_offer_share_pct: Current equity % being discussed
          • current_offer_amount: Current investment $ amount
          • expiry_day: Deadline to settle accepted deal
          • day: Simulation day

        notifications - Agent inbox (simple string notifications)
          • notification_id: Unique identifier
          • day: Simulation day created
          • type: Notification type (e.g. large_customer_message, vc_approach, research_complete, market_discovery, macro_economic_update)
          • message: Notification message string

        issues - Individual customer support issues with full lifecycle
          • issue_id: Unique identifier (auto-incrementing)
          • customer_id: Foreign key to customers table
          • group_id: Customer segment group identifier (e.g., 'S1', 'E1')
          • open_day: Simulation day when the issue was created
          • days_open: How many days the issue has been open
          • status: Current state - 'open' or 'resolved'
          • resolved_day: Simulation day when resolved (NULL if still open)
          • resolution_type: How it was resolved - 'ops_resolved' (via operations spend)

        PRE-LOADED VARIABLES (use these directly, don't redefine):
        ----------------------------------------------------------
        - conn: SQLite connection to the world database (read-only) - USE THIS!
        - pandas as pd, numpy as np
        - rows(sql, params) -> list of tuples - helper for quick queries
        - row(sql, params) -> single tuple or None - helper for single row

        EXAMPLE QUERIES:
        ----------------
        # Get current subscriber count
        print(row("SELECT COUNT(*) FROM subscriptions WHERE status='subscribed' AND end_day IS NULL"))

        # Get total monthly revenue from active subscriptions
        print(row("SELECT SUM(effective_price) FROM subscriptions WHERE status='subscribed' AND end_day IS NULL"))

        # Get subscriber count by plan
        print(rows("SELECT plan, COUNT(*) FROM subscriptions WHERE status='subscribed' AND end_day IS NULL GROUP BY plan"))

        # Get recent cancellations
        print(rows("SELECT customer_id, plan, end_day FROM subscriptions WHERE status='cancelled' ORDER BY end_day DESC LIMIT 10"))

        # Get daily revenue trend (last 7 days)
        df = pd.read_sql("SELECT day, SUM(amount) as revenue FROM ledger WHERE category='subscription_payment' AND day > (SELECT MAX(day)-7 FROM ledger) GROUP BY day", conn)
        print(df)

        # Get recent social media posts
        print(rows("SELECT day, content, likes, shares FROM social_media_posts WHERE day > (SELECT MAX(day)-7 FROM social_media_posts) ORDER BY likes DESC LIMIT 10"))

        # Get cash balance
        print(row("SELECT SUM(amount) FROM ledger"))

        # Get churn rate (last 30 days)
        total = row("SELECT COUNT(*) FROM subscriptions WHERE status='subscribed'")[0]
        churned = row("SELECT COUNT(*) FROM subscriptions WHERE status='cancelled' AND end_day > (SELECT MAX(day)-30 FROM service_day)")[0]
        print(f"Churn rate: {churned}/{total} = {churned/total*100:.1f}%")

        # Get open issues by group
        print(rows("SELECT group_id, COUNT(*) as open_issues, AVG(days_open) as avg_days FROM issues WHERE status='open' GROUP BY group_id"))

        # Get issue resolution stats
        print(rows("SELECT group_id, COUNT(*) as resolved, AVG(days_open) as avg_resolution_days FROM issues WHERE status='resolved' GROUP BY group_id"))

        ENTERPRISE THREAD QUERIES:
        --------------------------
        # Get all turns for a specific enterprise thread
        print(rows('''
            SELECT turn_number, day, sender, message_text, offer_json, status
            FROM enterprise_turns
            WHERE thread_id = ?
            ORDER BY turn_number ASC
        ''', (thread_id,)))

        # Get thread info with customer details (latest turn = current state)
        print(row('''
            SELECT et.thread_id, et.thread_type, et.closed, et.close_reason, et.turn_number,
                   et.seat_count, c.customer_id, c.email
            FROM enterprise_turns et
            JOIN customers c ON et.customer_id = c.customer_id
            WHERE et.thread_id = ?
            ORDER BY et.message_id DESC LIMIT 1
        ''', (thread_id,)))

        # Get all open negotiation threads
        print(rows('''
            SELECT et.thread_id, et.thread_type, et.closed, et.close_reason, et.seat_count, c.email
            FROM enterprise_turns et
            JOIN customers c ON et.customer_id = c.customer_id
            WHERE et.message_id = (SELECT MAX(et2.message_id) FROM enterprise_turns et2 WHERE et2.thread_id = et.thread_id)
              AND et.closed = 0
              AND et._internal_status IS NULL
        '''))

        VC QUERIES:
        -------------------
        # Get all turns for a specific VC (by shareholder_id)
        print(rows('''
            SELECT turn_number, day, sender, message_text, offer_json,
                   current_offer_share_pct, current_offer_amount
            FROM vc_turns
            WHERE shareholder_id = ?
            ORDER BY turn_number ASC
        ''', (shareholder_id,)))

        Args:
            code: Python code to execute. Use print() to see output.
            timeout_seconds: Maximum execution time (default 5s)
        """
        import subprocess
        import tempfile

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports and setup - use stored db_path for reliable access
            setup_code = f"""
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import sys
import os
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '{self.workspace_path}')

# ===== Block access to macro sensitivity parameters =====
# Agent should not see how macro conditions affect VC valuations or customer groups
_MACRO_BLOCKED_ATTRS = {{'MACRO_SENSITIVITY', 'macro_sensitivity', 'macro_beta'}}

_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _scrubbed_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)
    if 'saas_bench' in name and 'config' in name:
        for attr in _MACRO_BLOCKED_ATTRS:
            if hasattr(mod, attr):
                try:
                    delattr(mod, attr)
                except (AttributeError, TypeError):
                    pass
        # Scrub macro_sensitivity from VC profiles
        if hasattr(mod, 'PREDEFINED_VCS'):
            for _vc in mod.PREDEFINED_VCS:
                if hasattr(_vc, 'macro_sensitivity'):
                    try:
                        object.__setattr__(_vc, 'macro_sensitivity', None)
                    except (AttributeError, TypeError):
                        pass
    return mod

import builtins
builtins.__import__ = _scrubbed_import

# Hidden tables that agent cannot query at all
_HIDDEN_TABLES = {{
    'events',             # Internal shock/event tracking
    'api_costs',          # Meta-simulation API cost tracking
    'customer_state',     # Internal satisfaction/relationship state
    'group_reputation',   # Internal reputation tracking
    'group_awareness',    # Internal awareness tracking
    'reputation_history', # Internal reputation history
    'global_state',       # Internal simulation state
    'feature_tests',      # Internal feature test tracking
    'test_assignments',   # Internal test assignments
    'customer_personas',  # Internal persona templates
    'customer_persona_map', # Internal persona mapping
    'group_characteristics', # Internal group characteristics
    'enterprise_thread_counter',  # Internal thread ID counter
    # (vc_thread_counter removed — VC turns grouped by shareholder_id)
    'world_context',      # Internal world context
    'pending_group_research', # Internal async research tracking
    'group_parameters',       # V2.1: Internal preference drift tracking (agent must infer from behavior)
    'shareholders',           # V2.2: Hidden — agent uses vc_turns + list_potential_vcs instead
}}

# Hidden columns that agent should not see (latent customer attributes, internal simulation params)
# These are internal parameters used by the simulation - agent must infer customer behavior from observable data
_HIDDEN_COLUMNS = {{
    # Social media hidden columns
    'sentiment', 'reputation_impact', 'influence_score',
    # Latent customer satisfaction curve parameters (customers table)
    'steepness_left', 'steepness_right', 'c_max',
    # Latent customer preferences (customers table)
    'usage_demand', 'quality_sensitivity', 'price_sensitivity',
    'willingness_to_pay', 'usage_scale', 'patience',
    # Enterprise negotiation parameters (customers table)
    'reply_delay_mean', 'reply_delay_std', 'negotiation_rate', 'max_negotiation_turns',
    # Thread hidden columns - customer/VC reply timing is internal simulation state
    'next_reply_day',
    # VC negotiation internal parameters (shareholders table)
    'vc_alpha', 'turns_this_year', 'year_start_day',
    # VC term sheet internals (vc_turns table) — agent sees terms via python_exec queries
    'original_valuation', 'anti_dilution_triggered', 'tranche_2_released',
    # Internal tracking columns
    'current_offer_price',
    # Funding round valuations - agent shouldn't see these
    'pre_money_valuation', 'post_money_valuation',
    # Usage rate hidden - agent should only see actual (quota-capped) usage from daily_usage table
    'daily_usage_rate', 'billing_period_usage',
    # Customer state hidden columns (customer_state table) - internal satisfaction tracking
    'satisfaction', 'relationship', 'open_issue_days',
    'current_steepness_left', 'current_steepness_right', 'current_c_max', 'current_slope',
    'last_drift_day', 'plan_was_acceptable', 'last_quality', 'last_satisfaction', 'shock_event_id',
    # Group-level hidden state (group_reputation, group_awareness tables)
    'reputation', 'awareness', 'last_updated_day', 'last_marketing_day',
    # Reputation history internals
    'change_reason',
    # R&D project internals - actual_completion_day hidden for non-completed projects
    # (list_research_projects() formats output directly, so this only affects python_exec queries)
    'actual_completion_day',
    # Enterprise negotiation internal parameter (customers table)
    'initial_offer_factor',
    # Customer persona internal attribute (customers table) - used for LLM prompt generation
    'persona_communication',
    # Internal thread status tracking (enterprise_turns, vc_turns) - hidden dead-thread marker
    '_internal_status',
}}

# Table-specific hidden columns (hidden only when querying these tables)
_TABLE_HIDDEN_COLUMNS = {{
    # group_id is now visible — agents need it to analyze customer segments
    # seat_count is hidden from customers/ads_revenue (internal float for drift)
    # but visible on subscriptions table (floored integer for agent)
    'customers': {{'seat_count'}},
    'ads_revenue': {{'seat_count'}},
}}

def _get_effective_hidden(query=None):
    \"\"\"Get the effective set of hidden columns, including table-specific ones.\"\"\"
    hidden = set(_HIDDEN_COLUMNS)
    if query:
        q = query.lower()
        for table, cols in _TABLE_HIDDEN_COLUMNS.items():
            import re
            if re.search(r'\\b' + re.escape(table) + r'\\b', q):
                hidden |= cols
    return hidden

def _filter_hidden(result, description, query=None):
    \"\"\"Remove hidden columns from query result.\"\"\"
    if not description or not result:
        return result
    hidden = _get_effective_hidden(query)
    col_names = [d[0] for d in description]
    hidden_indices = [i for i, name in enumerate(col_names) if name in hidden]
    if not hidden_indices:
        return result
    # Filter out hidden column indices
    return tuple(v for i, v in enumerate(result) if i not in hidden_indices)

def _is_schema_query(query):
    \"\"\"Check if query is trying to inspect database schema.\"\"\"
    q = query.lower().strip()
    # Block sqlite_master, pragma, and table_info queries
    blocked_patterns = [
        'sqlite_master', 'sqlite_schema', 'pragma', 'table_info',
        'index_list', 'index_info', 'foreign_key_list'
    ]
    return any(p in q for p in blocked_patterns)

def _references_hidden_table(query):
    \"\"\"Check if query references any hidden table.\"\"\"
    q = query.lower()
    for table in _HIDDEN_TABLES:
        # Check for table name with common SQL patterns
        # Match: FROM table, JOIN table, INTO table, UPDATE table, table., "table"
        import re
        pattern = r'\\b' + re.escape(table.lower()) + r'\\b'
        if re.search(pattern, q):
            return table
    return None

def _filter_schema_result(result, query):
    \"\"\"Filter hidden columns from schema query results.\"\"\"
    if not result:
        return result
    hidden = _get_effective_hidden(query)
    # For PRAGMA table_info style results, filter out hidden column names
    # Result format: (cid, name, type, notnull, dflt_value, pk)
    if isinstance(result, (list, tuple)) and len(result) > 0:
        if isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
            # Filter out rows where column name (index 1) is hidden
            return [r for r in result if r[1] not in hidden]
    return result

# Valid tables the agent can query
_VALID_TABLES = {{
    'customers', 'subscriptions', 'ledger', 'service_day', 'config_history',
    'social_media_posts', 'enterprise_turns', 'vc_turns', 'notifications'
}}

# Table columns for helpful error messages (generated at sandbox creation time)
_TABLE_COLUMNS = {repr({table_name: list(table_info['columns'].keys()) for table_name, table_info in TABLE_DOCS.items()})}

def _get_helpful_error(original_error, query):
    \"\"\"Generate a helpful error message for SQL errors.\"\"\"
    err_str = str(original_error).lower()

    # Check for "no such table" errors
    if 'no such table' in err_str:
        import re
        match = re.search(r'no such table: (\\w+)', err_str)
        if match:
            bad_table = match.group(1)
            # Suggest similar valid tables
            suggestions = [t for t in _VALID_TABLES if bad_table[:3] in t or t[:3] in bad_table]
            suggestion_str = f" Did you mean: {{', '.join(suggestions)}}?" if suggestions else ""
            return f"Table '{{bad_table}}' does not exist.{{suggestion_str}} Valid tables: {{', '.join(sorted(_VALID_TABLES))}}"

    # Check for "no such column" errors
    if 'no such column' in err_str:
        import re
        # Capture full column reference including table.column format
        match = re.search(r'no such column: ([\\w.]+)', err_str)
        if match:
            bad_col = match.group(1)
            # Try to identify which table(s) are in the query to show valid columns
            matched_tables = {{}}
            q_lower = query.lower() if query else ''
            for table_name, cols in _TABLE_COLUMNS.items():
                if re.search(r'\\b' + re.escape(table_name) + r'\\b', q_lower):
                    matched_tables[table_name] = cols
            if matched_tables:
                hints = []
                for tname, cols in matched_tables.items():
                    hints.append(f"{{tname}}: {{', '.join(cols)}}")
                return f"Column '{{bad_col}}' does not exist. Valid columns for tables in your query:\\n  " + "\\n  ".join(hints) + "\\nUse describe_tables() to check schemas."
            else:
                return f"Column '{{bad_col}}' does not exist. Use describe_tables() to check valid column names."

    # Return original error with query context
    return f"SQL Error: {{original_error}}\\nQuery: {{query[:200]}}..."

# Wrapper class for connection that filters hidden columns
class _FilteredConnection:
    \"\"\"Connection wrapper that hides internal columns from all queries.\"\"\"
    def __init__(self, real_conn):
        self._conn = real_conn

    def execute(self, query, params=()):
        # Block schema introspection queries
        if _is_schema_query(query):
            raise PermissionError("Schema introspection is not allowed. Use the documented tables and columns.")
        # Block queries to hidden tables
        hidden_table = _references_hidden_table(query)
        if hidden_table:
            raise PermissionError(f"Table '{{hidden_table}}' is internal/hidden. Use the documented tables: {{', '.join(sorted(_VALID_TABLES))}}")
        try:
            cursor = self._conn.execute(query, params)
            return _FilteredCursor(cursor, query=query)
        except Exception as e:
            raise RuntimeError(_get_helpful_error(e, query))

    def executemany(self, query, params_list):
        if _is_schema_query(query):
            raise PermissionError("Schema introspection is not allowed.")
        hidden_table = _references_hidden_table(query)
        if hidden_table:
            raise PermissionError(f"Table '{{hidden_table}}' is internal/hidden.")
        try:
            return self._conn.executemany(query, params_list)
        except Exception as e:
            raise RuntimeError(_get_helpful_error(e, query))

    def cursor(self):
        return _FilteredCursor(self._conn.cursor())

    def __getattr__(self, name):
        return getattr(self._conn, name)

class _FilteredCursor:
    \"\"\"Cursor wrapper that filters hidden columns from results.\"\"\"
    def __init__(self, real_cursor, query=None):
        self._cursor = real_cursor
        # Capture description from already-executed cursor
        self._desc = real_cursor.description
        self._query = query  # Store query for table-specific hidden column filtering

    def execute(self, query, params=()):
        if _is_schema_query(query):
            raise PermissionError("Schema introspection is not allowed. Use the documented tables and columns.")
        hidden_table = _references_hidden_table(query)
        if hidden_table:
            raise PermissionError(f"Table '{{hidden_table}}' is internal/hidden. Use documented tables only.")
        try:
            self._cursor.execute(query, params)
            self._desc = self._cursor.description
            self._query = query
            return self
        except Exception as e:
            raise RuntimeError(_get_helpful_error(e, query))

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _filter_hidden(tuple(row), self._desc, self._query)

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [_filter_hidden(tuple(r), self._desc, self._query) for r in rows]

    def fetchmany(self, size=None):
        rows = self._cursor.fetchmany(size) if size else self._cursor.fetchmany()
        return [_filter_hidden(tuple(r), self._desc, self._query) for r in rows]

    @property
    def description(self):
        if not self._desc:
            return self._cursor.description
        # Filter hidden columns from description too
        hidden = _get_effective_hidden(self._query)
        return tuple(d for d in self._desc if d[0] not in hidden)

    def __iter__(self):
        for row in self._cursor:
            yield _filter_hidden(tuple(row), self._desc, self._query)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

# Connect to database (read-only) and wrap with filter
_real_conn = sqlite3.connect('file:{self.db_path}?mode=ro', uri=True)
_real_conn.row_factory = sqlite3.Row
conn = _FilteredConnection(_real_conn)

# Helper: convert Row objects to tuples for cleaner printing
def rows(query, params=()):
    \"\"\"Execute query and return list of tuples (hidden columns filtered).\"\"\"
    cursor = conn.execute(query, params)
    return cursor.fetchall()

def row(query, params=()):
    \"\"\"Execute query and return single tuple or None (hidden columns filtered).\"\"\"
    cursor = conn.execute(query, params)
    return cursor.fetchone()

# Override pd.read_sql to filter hidden columns
_original_read_sql = pd.read_sql
def _filtered_read_sql(query, con, *args, **kwargs):
    df = _original_read_sql(query, con, *args, **kwargs)
    # Drop hidden columns if present
    cols_to_drop = [c for c in df.columns if c in _HIDDEN_COLUMNS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df
pd.read_sql = _filtered_read_sql

# Working directory
os.chdir('{self.workspace_path}')

# User code below
"""
            f.write(setup_code + code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self.workspace_path)
            )

            stdout_text = (result.stdout or "")[:5000]
            stderr_text = (result.stderr or "")[:2000]

            output = stdout_text
            if stderr_text:
                output += f"\nSTDERR:\n{stderr_text}"

            return ToolResult(
                result.returncode == 0,
                output if output else "No output",
                {'returncode': result.returncode, 'stdout': stdout_text, 'stderr': stderr_text}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Execution timed out after {timeout_seconds} seconds")
        except Exception as e:
            return ToolResult(False, f"Execution error: {str(e)}")
        finally:
            import os
            os.unlink(temp_path)

    def register_daily_job(self, script_path: str) -> ToolResult:
        """Register a script to run daily.

        The script content is snapshotted at registration time. Subsequent
        edits to the source file will NOT affect the registered version.
        Re-register to update.

        Args:
            script_path: Path to the script (relative to workspace)
        """
        full_path = self.workspace_path / script_path

        if not full_path.exists():
            return ToolResult(False, f"Script not found: {script_path}")

        content = full_path.read_text()

        # Store name -> content snapshot in JSON file
        jobs_file = self.workspace_path / '.daily_jobs.json'
        jobs = {}
        if jobs_file.exists():
            raw = json.loads(jobs_file.read_text())
            # Backwards compat: old format was a list of paths
            if isinstance(raw, list):
                jobs = {}
                for p in raw:
                    fp = self.workspace_path / p
                    if fp.exists():
                        jobs[p] = fp.read_text()
            else:
                jobs = raw

        jobs[script_path] = content
        jobs_file.write_text(json.dumps(jobs))

        return ToolResult(True, f"Registered daily job: {script_path}",
                          data={'script_path': script_path, 'total_jobs': len(jobs)})

    # === DAILY CALCULATIONS ===
    # Calculations are named code snippets that run at the start of each day
    # Their output is included in the daily dashboard

    def get_daily_calculations(self) -> Dict[str, str]:
        """Get all registered daily calculations."""
        return getattr(self, '_daily_calculations', {})

    def set_daily_calculations(self, calculations: Dict[str, str]):
        """Set daily calculations (called by benchmark for persistence)."""
        self._daily_calculations = calculations

    def register_daily_calculation(self, name: str, code: str) -> ToolResult:
        """Register a named calculation to run at the start of each day.

        The calculation's output (via print statements) will be shown in the daily dashboard.
        Use this to track custom metrics or perform daily analysis automatically.

        Args:
            name: Unique name for the calculation (e.g., "revenue_trend", "churn_rate")
            code: Python code to execute. Has access to: conn (DB), rows(query), row(query),
                  numpy (np), pandas (pd), and standard math/statistics libraries.
        """
        calculations = self.get_daily_calculations()
        calculations[name] = code
        self._daily_calculations = calculations

        return ToolResult(
            True,
            f"Registered daily calculation: '{name}'. It will run at the start of each day.",
            data={'name': name, 'code_length': len(code)}
        )

    def remove_daily_calculation(self, name: str) -> ToolResult:
        """Remove a registered daily calculation.

        Args:
            name: Name of the calculation to remove
        """
        calculations = self.get_daily_calculations()

        if name not in calculations:
            return ToolResult(
                False,
                f"Calculation '{name}' not found. Registered calculations: {list(calculations.keys())}"
            )

        del calculations[name]
        self._daily_calculations = calculations

        return ToolResult(True, f"Removed daily calculation: '{name}'",
                          data={'name': name, 'remaining': list(calculations.keys())})

    def list_daily_calculations(self) -> ToolResult:
        """List all registered daily calculations."""
        calculations = self.get_daily_calculations()

        if not calculations:
            return ToolResult(True, "No daily calculations registered.", data={'calculations': []})

        lines = ["Registered daily calculations:"]
        for name, code in calculations.items():
            # Show first 100 chars of code
            preview = code[:100].replace('\n', ' ')
            if len(code) > 100:
                preview += "..."
            lines.append(f"  • {name}: {preview}")

        return ToolResult(True, '\n'.join(lines), data={'calculations': list(calculations.keys())})

    def run_daily_calculations(self) -> Dict[str, str]:
        """Run all registered daily calculations and return their outputs.

        Returns:
            Dict mapping calculation name to its output (or error message)
        """
        calculations = self.get_daily_calculations()
        results = {}

        for name, code in calculations.items():
            result = self.python_exec(code)
            if result.success:
                results[name] = result.message
            else:
                results[name] = f"ERROR: {result.message}"

        return results

    # === NAMED SCRIPTS ===
    # Scripts are named code snippets the agent can register and run on demand.
    # They persist across days (via checkpoint) and execute in the python_exec environment.

    def get_scripts(self) -> Dict[str, str]:
        """Get all registered scripts."""
        return getattr(self, '_scripts', {})

    def set_scripts(self, scripts: Dict[str, str]):
        """Set scripts (called by benchmark for persistence)."""
        self._scripts = scripts

    def register_script(self, name: str, code: str) -> ToolResult:
        """Register a named script for later execution.

        Args:
            name: Unique name for the script
            code: Python code to execute (same environment as python_exec)
        """
        if not name or not name.strip():
            return ToolResult(False, "Script name cannot be empty")

        name = name.strip()
        scripts = self.get_scripts()
        overwritten = name in scripts
        scripts[name] = code
        self._scripts = scripts

        suffix = ", overwritten" if overwritten else ""
        return ToolResult(
            True,
            f"Script '{name}' registered ({len(code)} chars{suffix}). Run with run_script(name='{name}').",
            data={'name': name, 'code_length': len(code)}
        )

    def run_script(self, name: str) -> ToolResult:
        """Execute a previously registered script.

        Args:
            name: Name of the script to run
        """
        scripts = self.get_scripts()

        if not name or name.strip() not in scripts:
            registered = list(scripts.keys())
            return ToolResult(
                False,
                f"Script '{name}' not found. Registered scripts: {registered}"
            )

        return self.python_exec(scripts[name.strip()])

    def list_scripts(self) -> ToolResult:
        """List all registered scripts."""
        scripts = self.get_scripts()

        if not scripts:
            return ToolResult(True, "No scripts registered.", data={'scripts': []})

        lines = ["Registered scripts:"]
        for name, code in scripts.items():
            preview = code[:100].replace('\n', ' ')
            if len(code) > 100:
                preview += "..."
            lines.append(f"  • {name}: {preview}")

        return ToolResult(True, '\n'.join(lines), data={'scripts': list(scripts.keys())})

    def delete_script(self, name: str) -> ToolResult:
        """Delete a registered script.

        Args:
            name: Name of the script to delete
        """
        scripts = self.get_scripts()

        if not name or name.strip() not in scripts:
            registered = list(scripts.keys())
            return ToolResult(
                False,
                f"Script '{name}' not found. Registered scripts: {registered}"
            )

        del scripts[name.strip()]
        self._scripts = scripts

        return ToolResult(True, f"Script '{name.strip()}' deleted.",
                          data={'name': name.strip(), 'remaining': list(scripts.keys())})

    # === MEMORY MANAGEMENT ===
    # Memory is a list of lines that persists across turns within a run
    # It's shown in the system prompt, so agent always sees current state

    def set_memory(self, memory_lines: List[str]):
        """Set the memory content (called by benchmark)."""
        self._memory = memory_lines

    def get_memory(self) -> List[str]:
        """Get current memory content."""
        return getattr(self, '_memory', [])

    def get_memory_display(self) -> str:
        """Get memory formatted for display in system prompt."""
        lines = self.get_memory()
        if not lines:
            return "(empty)"
        return '\n'.join(f"{i+1:3d}| {line}" for i, line in enumerate(lines))

    def memory_insert(self, line: int, content: str) -> ToolResult:
        """Insert content at a specific line number.

        Args:
            line: Line number to insert at (1-indexed). Content shifts down.
            content: Text to insert (can be multiple lines separated by newlines)
        """
        lines = self.get_memory()
        new_lines = content.split('\n')

        # Validate line number (1-indexed, can insert at end+1)
        if line < 1 or line > len(lines) + 1:
            return ToolResult(False, f"Invalid line number {line}. Valid range: 1-{len(lines)+1}")

        # Insert at 0-indexed position
        idx = line - 1
        for i, new_line in enumerate(new_lines):
            lines.insert(idx + i, new_line)

        self._memory = lines
        return ToolResult(True, f"Inserted {len(new_lines)} line(s) at line {line}. Memory now has {len(lines)} lines.",
                          data={'lines_inserted': len(new_lines), 'at_line': line, 'total_lines': len(lines)})

    def memory_delete(self, start: int, end: int) -> ToolResult:
        """Delete lines from start to end (inclusive).

        Args:
            start: First line to delete (1-indexed)
            end: Last line to delete (1-indexed, inclusive)
        """
        lines = self.get_memory()

        if start < 1 or end > len(lines) or start > end:
            return ToolResult(False, f"Invalid range {start}-{end}. Valid range: 1-{len(lines)}")

        # Delete lines (convert to 0-indexed)
        del lines[start-1:end]

        self._memory = lines
        return ToolResult(True, f"Deleted lines {start}-{end}. Memory now has {len(lines)} lines.",
                          data={'deleted_range': [start, end], 'total_lines': len(lines)})

    def memory_edit(self, line: int, content: str) -> ToolResult:
        """Replace content at a specific line.

        Args:
            line: Line number to edit (1-indexed)
            content: New content for this line
        """
        lines = self.get_memory()

        if line < 1 or line > len(lines):
            return ToolResult(False, f"Invalid line number {line}. Valid range: 1-{len(lines)}")

        lines[line-1] = content
        self._memory = lines
        return ToolResult(True, f"Updated line {line}.",
                          data={'line': line, 'total_lines': len(lines)})

    # === SOCIAL MEDIA & NOTIFICATIONS ===
    # Note: Agent sees daily notification summary in system prompt.
    # These tools let the agent dig deeper into specific items.

    def get_social_posts(self, days: int = 7, limit: int = 50) -> ToolResult:
        """Search social media posts about NovaMind.

        This queries posts that customers have made on social media. Use this to:
        - Monitor what customers are saying about NovaMind
        - Find specific complaints or praise to address
        - Analyze customer feedback about specific features

        Note: Sentiment is NOT provided - you must infer it from the post content.

        Args:
            days: How many days back to search (default 7)
            limit: Maximum posts to return (default 50)
        """
        posts = get_recent_social_posts(self.conn, days=days, limit=limit)

        if not posts:
            return ToolResult(True, "No social media posts found.", {'posts': []})

        # Format for display - show post details WITHOUT sentiment
        summary = []
        for p in posts[:15]:  # Show first 15 in message
            summary.append(f"Day {p['day']}: \"{p['content'][:80]}\" ({p['likes']} likes, {p['shares']} shares)")

        # Strip sentiment from returned data - agent must infer from content
        posts_without_sentiment = []
        for p in posts:
            post_data = {k: v for k, v in p.items() if k != 'sentiment'}
            posts_without_sentiment.append(post_data)

        return ToolResult(
            True,
            f"Found {len(posts)} posts in last {days} days.\n" + '\n'.join(summary),
            {'posts': posts_without_sentiment, 'total': len(posts)}
        )

    # =========================================================================
    # V2: VC Negotiation & Equity Tools
    # =========================================================================

    def list_potential_vcs(self) -> ToolResult:
        """List all predefined VC investors and their profiles.

        Shows each VC's name, investment range, description, and whether they
        currently have an active negotiation with the company.
        """
        # Get active VC negotiations to mark which VCs are already engaged
        active_threads = get_active_vc_negotiations(self.conn)
        active_vc_info = {}
        for t in active_threads:
            active_vc_info[t['vc_name']] = t['shareholder_id']

        output = "=== Potential VC Investors ===\n\n"
        vc_data = []

        for vc in PREDEFINED_VCS:
            status = "Available"
            shareholder_id = None
            if vc.name in active_vc_info:
                shareholder_id = active_vc_info[vc.name]
                status = f"Active (shareholder_id={shareholder_id})"

            output += f"  {vc.name} ({vc.vc_id})\n"
            output += f"    Equity range: {vc.equity_pct_min*100:.0f}% – {vc.equity_pct_max*100:.0f}%\n"
            output += f"    Description: {vc.description}\n"
            output += f"    Status: {status}\n\n"

            vc_data.append({
                'vc_id': vc.vc_id,
                'name': vc.name,
                'equity_pct_min': vc.equity_pct_min,
                'equity_pct_max': vc.equity_pct_max,
                'description': vc.description,
                'status': status,
                'shareholder_id': shareholder_id,
            })

        output += f"Total: {len(PREDEFINED_VCS)} VCs ({len(active_vc_info)} currently active)\n"
        output += "\nNote: VCs may reach out to you on any day. You can also attract "
        output += "VC interest by growing your company's metrics and reputation."

        return ToolResult(True, output, {'vcs': vc_data})

    def send_vc_deal(self, deals: Optional[List[Dict]] = None,
                     **kwargs) -> ToolResult:
        """Send structured equity offers to VC investors.

        Each deal is identified by shareholder_id. The system finds the VC's
        open negotiation thread automatically.

        Args:
            deals: List of deal dicts, each with:
                - shareholder_id (int): The VC's shareholder ID
                - share_pct (float): Share percentage offered (e.g. 0.10 for 10%)
                - anti_dilution_floor, milestone_tranche_pct, etc. (optional)
        """
        # Handle single-deal kwargs format
        if deals is None:
            deals = [kwargs] if kwargs else []
        if not deals:
            return ToolResult(False, "deals parameter required")
        if not isinstance(deals, list):
            deals = [deals]

        results = []
        summaries = []

        for deal in deals:
            result = self._process_single_vc_deal(deal)
            results.append(result[0])
            summaries.append(result[1])

        sent = sum(1 for r in results if r.get('success'))
        failed = len(results) - sent
        if len(deals) == 1:
            ok = bool(results[0].get('success'))
            return ToolResult(ok, summaries[0], {'results': results})

        summary = f"Sent {sent}/{len(deals)} VC deals"
        if failed:
            summary += f" ({failed} failed)"
        return ToolResult(sent > 0, summary + ":\n" + "\n".join(summaries), {'results': results})

    def _process_single_vc_deal(self, deal: Dict):
        """Process a single VC deal. Returns (result_dict, summary_str)."""
        shareholder_id = deal.get('shareholder_id')
        share_pct = deal.get('share_pct')
        anti_dilution_floor = deal.get('anti_dilution_floor')
        milestone_tranche_pct = deal.get('milestone_tranche_pct')
        milestone_revenue_multiplier = deal.get('milestone_revenue_multiplier')
        milestone_deadline_days = deal.get('milestone_deadline_days')
        redemption_days = deal.get('redemption_days')
        redemption_buyback_multiplier = deal.get('redemption_buyback_multiplier')

        if shareholder_id is None:
            return {'error': 'missing shareholder_id'}, "Error: shareholder_id required"
        shareholder_id = int(shareholder_id)

        # Look up shareholder name for error messages
        shareholder = get_shareholder(self.conn, shareholder_id)
        vc_label = shareholder['name'] if shareholder else f"Shareholder #{shareholder_id}"

        if share_pct is None or share_pct <= 0 or share_pct >= 1.0:
            return ({'shareholder_id': shareholder_id, 'error': 'invalid share_pct'},
                    f"{vc_label}: share_pct must be between 0 and 1 (exclusive)")

        # Check if this VC has an active negotiation
        latest = get_vc_latest_turn(self.conn, shareholder_id)
        if not latest or latest['closed'] or latest.get('_internal_status'):
            return ({'shareholder_id': shareholder_id, 'error': 'no open negotiation'},
                    f"{vc_label}: no open negotiation")

        thread = latest  # latest turn = current state

        state = get_vc_negotiation_state(self.conn, shareholder_id)
        if not state:
            return ({'shareholder_id': shareholder_id, 'error': 'no state'},
                    f"{vc_label}: could not load negotiation state")

        # Validate term sheet proposals
        error_msg = self._validate_vc_term_proposals(
            thread, anti_dilution_floor, milestone_tranche_pct,
            milestone_revenue_multiplier, milestone_deadline_days,
            redemption_days, redemption_buyback_multiplier)
        if error_msg:
            return ({'shareholder_id': shareholder_id, 'error': error_msg},
                    f"{vc_label}: {error_msg}")

        # Get VC's valuation
        valuation = thread.get('original_valuation')
        if not valuation or valuation <= 0:
            valuation = 1_000_000

        amount = compute_check_size_from_pct(share_pct, valuation)
        amount = round(amount, -3)

        total_shares = get_total_shares(self.conn)
        if total_shares <= 0:
            return ({'shareholder_id': shareholder_id, 'error': 'no shares'},
                    f"{vc_label}: no shares outstanding")

        new_shares = (share_pct / (1 - share_pct)) * total_shares
        price_per_share = amount / new_shares if new_shares > 0 else 0

        # Compute term sheet adjustment
        term_delta = self._compute_vc_term_delta(
            thread, anti_dilution_floor, milestone_tranche_pct,
            milestone_revenue_multiplier, milestone_deadline_days,
            redemption_days, redemption_buyback_multiplier)

        # Record agent's structural offer
        offer_data = {
            'share_pct': share_pct,
            'amount': amount,
            'price_per_share': round(price_per_share, 6),
            'new_shares': round(new_shares, 2),
        }
        proposed_terms = {
            k: v for k, v in {
                'anti_dilution_floor': anti_dilution_floor,
                'milestone_tranche_pct': milestone_tranche_pct,
                'milestone_revenue_multiplier': milestone_revenue_multiplier,
                'milestone_deadline_days': milestone_deadline_days,
                'redemption_days': redemption_days,
                'redemption_buyback_multiplier': redemption_buyback_multiplier,
            }.items() if v is not None
        }
        if proposed_terms:
            offer_data['proposed_terms'] = proposed_terms

        add_vc_turn(
            self.conn, shareholder_id, self.current_day, 'agent',
            message_text=None,
            offer_json=json.dumps(offer_data),
            current_offer_share_pct=share_pct,
            current_offer_amount=amount,
        )

        # Evaluate VC response
        decision, counter_pct, counter_amount = evaluate_agent_vc_offer(
            state, share_pct, self.config, valuation=valuation,
            term_sheet_adjustment=term_delta,
        )

        if decision == 'accept':
            deal_pct = counter_pct
            deal_amount = counter_amount
            deal_new_shares = (deal_pct / (1 - deal_pct)) * total_shares
            deal_pps = deal_amount / deal_new_shares if deal_new_shares > 0 else 0

            term_overrides = self._build_vc_term_overrides(
                deal_amount, anti_dilution_floor, milestone_tranche_pct,
                milestone_revenue_multiplier, milestone_deadline_days,
                redemption_days, redemption_buyback_multiplier)

            add_vc_turn(
                self.conn, shareholder_id, self.current_day, 'vc',
                message_text=f"Accepted: {deal_pct:.1%} equity for ${deal_amount:,.0f}. "
                f"Proceed with settle_investments().",
                offer_json=json.dumps({'share_pct': deal_pct, 'amount': deal_amount,
                           'price_per_share': round(deal_pps, 6), 'decision': 'accept'}),
                closed=1, close_reason='accepted',
                current_offer_share_pct=deal_pct,
                current_offer_amount=deal_amount,
                **term_overrides,
            )
            self.conn.commit()

            add_notification(
                self.conn, self.current_day, 'vc_deal_accepted',
                f'VC deal accepted: {state.vc_name} ({deal_pct:.1%} for ${deal_amount:,.0f})',
            )

            return ({'shareholder_id': shareholder_id, 'success': True, 'decision': 'accept',
                     'deal_pct': deal_pct, 'deal_amount': deal_amount},
                    f"{vc_label}: ACCEPTED — "
                    f"{deal_pct:.1%} for ${deal_amount:,.0f}")
        else:
            self.conn.commit()
            schedule_vc_reply(self.conn, shareholder_id, self.current_day, self.rng)

            return ({'shareholder_id': shareholder_id, 'success': True, 'decision': 'counter_pending'},
                    f"{vc_label}: offer sent — "
                    f"{share_pct:.1%} (${amount:,.0f}), awaiting response")

    def _validate_vc_term_proposals(self, thread, anti_dilution_floor, milestone_tranche_pct,
                                     milestone_revenue_multiplier, milestone_deadline_days,
                                     redemption_days, redemption_buyback_multiplier):
        """Validate term sheet proposals. Returns error message or None."""
        if anti_dilution_floor is not None:
            if not thread.get('has_anti_dilution'):
                return "Cannot propose anti_dilution_floor — no anti-dilution term"
            if anti_dilution_floor not in self.config.vc_anti_dilution_floor_options:
                return f"anti_dilution_floor must be one of {list(self.config.vc_anti_dilution_floor_options)}"
        if milestone_tranche_pct is not None:
            if not thread.get('has_milestone_tranching'):
                return "Cannot propose milestone_tranche_pct — no milestone tranching term"
            if milestone_tranche_pct not in self.config.vc_milestone_tranche_pct_options:
                return f"milestone_tranche_pct must be one of {list(self.config.vc_milestone_tranche_pct_options)}"
        if milestone_revenue_multiplier is not None:
            if not thread.get('has_milestone_tranching'):
                return "Cannot propose milestone_revenue_multiplier — no milestone tranching term"
            if milestone_revenue_multiplier not in self.config.vc_milestone_revenue_multiplier_options:
                return f"milestone_revenue_multiplier must be one of {list(self.config.vc_milestone_revenue_multiplier_options)}"
        if milestone_deadline_days is not None:
            if not thread.get('has_milestone_tranching'):
                return "Cannot propose milestone_deadline_days — no milestone tranching term"
            if milestone_deadline_days not in self.config.vc_milestone_deadline_days_options:
                return f"milestone_deadline_days must be one of {list(self.config.vc_milestone_deadline_days_options)}"
        if redemption_days is not None:
            if not thread.get('has_redemption_rights'):
                return "Cannot propose redemption_days — no redemption rights term"
            if redemption_days not in self.config.vc_redemption_days_options:
                return f"redemption_days must be one of {list(self.config.vc_redemption_days_options)}"
        if redemption_buyback_multiplier is not None:
            if not thread.get('has_redemption_rights'):
                return "Cannot propose redemption_buyback_multiplier — no redemption rights term"
            if redemption_buyback_multiplier not in self.config.vc_redemption_buyback_multiplier_options:
                return f"redemption_buyback_multiplier must be one of {list(self.config.vc_redemption_buyback_multiplier_options)}"
        return None

    def _compute_vc_term_delta(self, thread, anti_dilution_floor, milestone_tranche_pct,
                                milestone_revenue_multiplier, milestone_deadline_days,
                                redemption_days, redemption_buyback_multiplier):
        """Compute term sheet friendliness delta (agent - VC original)."""
        agent_ad_floor = anti_dilution_floor if anti_dilution_floor is not None else thread.get('anti_dilution_floor')
        agent_tranche = milestone_tranche_pct if milestone_tranche_pct is not None else thread.get('milestone_tranche_pct')
        agent_rev_mult = milestone_revenue_multiplier if milestone_revenue_multiplier is not None else thread.get('milestone_revenue_multiplier')
        agent_deadline = milestone_deadline_days if milestone_deadline_days is not None else thread.get('milestone_deadline_days_chosen')
        agent_red_days = redemption_days if redemption_days is not None else thread.get('redemption_days_chosen')
        agent_buyback = redemption_buyback_multiplier if redemption_buyback_multiplier is not None else thread.get('redemption_buyback_multiplier')

        agent_friendliness = compute_term_sheet_friendliness(
            self.config, anti_dilution_floor=agent_ad_floor, tranche_pct=agent_tranche,
            revenue_multiplier=agent_rev_mult, deadline_days=agent_deadline,
            redemption_days=agent_red_days, buyback_multiplier=agent_buyback,
        )
        vc_original = compute_term_sheet_friendliness(
            self.config, anti_dilution_floor=thread.get('anti_dilution_floor'),
            tranche_pct=thread.get('milestone_tranche_pct'),
            revenue_multiplier=thread.get('milestone_revenue_multiplier'),
            deadline_days=thread.get('milestone_deadline_days_chosen'),
            redemption_days=thread.get('redemption_days_chosen'),
            buyback_multiplier=thread.get('redemption_buyback_multiplier'),
        )
        return agent_friendliness - vc_original

    def _build_vc_term_overrides(self, deal_amount, anti_dilution_floor, milestone_tranche_pct,
                                  milestone_revenue_multiplier, milestone_deadline_days,
                                  redemption_days, redemption_buyback_multiplier):
        """Build term sheet override kwargs for VC acceptance turn."""
        overrides = {}
        if anti_dilution_floor is not None:
            overrides['anti_dilution_floor'] = anti_dilution_floor
        if milestone_tranche_pct is not None:
            overrides['milestone_tranche_pct'] = milestone_tranche_pct
            overrides['tranche_1_amount'] = deal_amount * milestone_tranche_pct
            overrides['tranche_2_amount'] = deal_amount - overrides['tranche_1_amount']
        if milestone_revenue_multiplier is not None:
            overrides['milestone_revenue_multiplier'] = milestone_revenue_multiplier
            from .database import get_mrr
            overrides['milestone_revenue_target'] = get_mrr(self.conn) * milestone_revenue_multiplier
        if milestone_deadline_days is not None:
            overrides['milestone_deadline_days_chosen'] = milestone_deadline_days
            overrides['milestone_deadline_day'] = self.current_day + milestone_deadline_days
        if redemption_days is not None:
            overrides['redemption_days_chosen'] = redemption_days
            overrides['redemption_eligible_day'] = self.current_day + redemption_days
        if redemption_buyback_multiplier is not None:
            overrides['redemption_buyback_multiplier'] = redemption_buyback_multiplier
        return overrides

    def reject_vc_deal(self, deals: Optional[List[Dict]] = None,
                       **kwargs) -> ToolResult:
        """Reject one or more VC deals permanently.

        Each deal is identified by shareholder_id. The system finds the VC's
        open negotiation thread automatically.

        Args:
            deals: List of deal dicts, each with shareholder_id (int)
        """
        if deals is None:
            deals = [kwargs] if kwargs else []
        if not deals:
            return ToolResult(False, "deals parameter required")
        if not isinstance(deals, list):
            deals = [deals]

        results = []
        summaries = []

        for deal in deals:
            shareholder_id = deal.get('shareholder_id')
            if shareholder_id is None:
                summaries.append("Error: shareholder_id required")
                results.append({'error': 'missing shareholder_id'})
                continue

            shareholder_id = int(shareholder_id)
            shareholder = get_shareholder(self.conn, shareholder_id)
            vc_label = shareholder['name'] if shareholder else f"Shareholder #{shareholder_id}"

            # Check if this VC has an active negotiation
            latest = get_vc_latest_turn(self.conn, shareholder_id)
            if not latest or latest['closed'] or latest.get('_internal_status'):
                summaries.append(f"{vc_label}: no open negotiation")
                results.append({'shareholder_id': shareholder_id, 'error': 'no open negotiation'})
                continue

            add_vc_turn(
                self.conn, shareholder_id, self.current_day, 'agent',
                message_text=None,
                closed=1, close_reason='agent_rejected',
            )
            self.conn.commit()

            summaries.append(f"{vc_label}: deal rejected")
            results.append({'shareholder_id': shareholder_id, 'success': True})

        rejected = sum(1 for r in results if r.get('success'))
        if len(deals) == 1:
            ok = bool(results and results[0].get('success'))
            return ToolResult(ok, summaries[0], {'results': results})

        return ToolResult(rejected > 0,
            f"Rejected {rejected}/{len(deals)} VC deals:\n" + "\n".join(summaries),
            {'results': results})

    def reject_enterprise_deal(self, deals: Optional[List[Dict]] = None,
                                **kwargs) -> ToolResult:
        """Reject one or more enterprise deals permanently.

        Each deal is identified by customer_id. The system finds the customer's
        active negotiation thread automatically.

        Args:
            deals: List of deal dicts, each with customer_id (int)
        """
        if deals is None:
            deals = [kwargs] if kwargs else []
        if not deals:
            return ToolResult(False, "deals parameter required")
        if not isinstance(deals, list):
            deals = [deals]

        results = []
        summaries = []

        # --- Phase 1: Parse all customer_ids (no SQL) ---
        parsed_cids = []
        for deal in deals:
            if isinstance(deal, (list, tuple)):
                customer_id = deal[0] if deal else None
            elif isinstance(deal, dict):
                customer_id = deal.get('customer_id')
            elif isinstance(deal, (int, float)):
                customer_id = int(deal)
            else:
                summaries.append(f"Error: invalid deal format: {deal}")
                results.append({'error': 'invalid format'})
                continue

            if customer_id is None:
                summaries.append("Error: customer_id required")
                results.append({'error': 'missing customer_id'})
                continue

            parsed_cids.append(int(customer_id))

        if not parsed_cids:
            if len(deals) == 1 and results:
                return ToolResult(False, summaries[0], {'results': results})
            return ToolResult(False, f"Rejected 0/{len(deals)} enterprise deals:\n" + "\n".join(summaries), {'results': results})

        # --- Phase 2: Batch pre-fetch active threads (1 query) ---
        cid_placeholders = ','.join('?' * len(parsed_cids))
        active_threads = {}  # customer_id -> {thread_id, thread_type, turn_number, current_offer_price, closed}
        rows = self.conn.execute(f"""
            SELECT et.customer_id, et.thread_id, et.thread_type,
                   et.turn_number, et.current_offer_price, et.closed, et.close_reason
            FROM enterprise_turns et
            WHERE et.customer_id IN ({cid_placeholders})
              AND et.message_id = (
                  SELECT MAX(et2.message_id) FROM enterprise_turns et2 WHERE et2.thread_id = et.thread_id
              )
              AND et.closed = 0
              AND et._internal_status IS NULL
        """, parsed_cids).fetchall()
        for row in rows:
            active_threads[row['customer_id']] = {
                'thread_id': row['thread_id'],
                'thread_type': row['thread_type'],
                'turn_number': row['turn_number'],
                'current_offer_price': row['current_offer_price'],
            }

        # --- Phase 3: Process rejections using pre-fetched data ---
        new_lead_lost_cids = []  # customer_ids to mark subscriptions as lost

        for customer_id in parsed_cids:
            at = active_threads.get(customer_id)
            if not at:
                summaries.append(f"Customer #{customer_id}: no active thread")
                results.append({'customer_id': customer_id, 'error': 'no active thread'})
                continue

            tid = at['thread_id']

            # Inline add_enterprise_turn with closed=1 (avoids extra SELECT)
            new_turn_number = at['turn_number'] + 1
            self.conn.execute("""
                INSERT INTO enterprise_turns
                (thread_id, customer_id, thread_type, turn_number, sender, message_text,
                 offer_json, day, next_reply_day, current_offer_price, email,
                 closed, close_reason, _internal_status)
                VALUES (?, ?, ?, ?, 'agent', '', '{}', ?, NULL, ?, '', 1, 'agent_rejected', NULL)
            """, (tid, customer_id, at['thread_type'], new_turn_number,
                  self.current_day, at['current_offer_price']))

            if at['thread_type'] == 'new_lead':
                new_lead_lost_cids.append(customer_id)

            summaries.append(f"Customer #{customer_id}: rejected ({at['thread_type']})")
            results.append({'customer_id': customer_id, 'success': True})

        # Batch update lost subscriptions for new_lead rejections
        for cid in new_lead_lost_cids:
            self.conn.execute("""
                UPDATE subscriptions SET status = 'lost', end_day = ?
                WHERE customer_id = ? AND status = 'lead'
            """, (self.current_day, cid))

        # Single commit for all changes
        self.conn.commit()

        rejected = sum(1 for r in results if r.get('success'))
        if len(deals) == 1:
            ok = bool(results and results[0].get('success'))
            return ToolResult(ok, summaries[0], {'results': results})

        return ToolResult(rejected > 0,
            f"Rejected {rejected}/{len(deals)} enterprise deals:\n" + "\n".join(summaries),
            {'results': results})

    def get_cap_table_info(self) -> ToolResult:
        """Get the current cap table showing all shareholders and ownership percentages.

        Returns shareholder names, types, shares held, ownership %, and total invested.
        """
        cap_table = get_cap_table(self.conn)
        if not cap_table:
            return ToolResult(False, "No shareholders found — equity system not initialized")

        total_shares = get_total_shares(self.conn)

        output = "=== Cap Table ===\n"
        output += f"Total Shares Outstanding: {total_shares:,.0f}\n\n"
        output += f"{'Shareholder':<25} {'Type':<10} {'Shares':>15} {'Ownership':>10} {'Invested':>15}\n"
        output += "-" * 75 + "\n"

        for s in cap_table:
            output += (
                f"{s['name']:<25} {s['shareholder_type']:<10} "
                f"{s['shares_held']:>15,.0f} {s['ownership_pct']:>9.1f}% "
                f"${s['total_invested']:>14,.0f}\n"
            )

        # Show funding rounds
        rounds = get_funding_rounds(self.conn)
        if rounds:
            output += f"\n--- Funding History ({len(rounds)} rounds) ---\n"
            for r in rounds:
                output += (
                    f"Day {r['day']}: {r['investor_name']} invested ${r['total_amount']:,.0f} "
                    f"for {r['shares_issued']:,.0f} shares @ ${r['price_per_share']:.4f}/share\n"
                )

        # Show dividends
        dividends = get_dividend_history(self.conn)
        if dividends:
            total_div = sum(d['total_amount'] for d in dividends)
            output += f"\n--- Dividend History ({len(dividends)} payments, ${total_div:,.0f} total) ---\n"
            for d in dividends:
                output += f"Day {d['day']}: ${d['total_amount']:,.0f} (${d['per_share_amount']:.6f}/share)\n"

        return ToolResult(True, output, {
            'shareholders': cap_table,
            'total_shares': total_shares,
            'funding_rounds': rounds,
            'dividends': dividends,
        })

    def settle_investments(self) -> ToolResult:
        """Settle all accepted VC deals and auto-reject all other open VC threads.

        Takes no parameters. Automatically:
        1. Finds all accepted VC deals (close_reason='accepted')
        2. Auto-rejects all open (non-accepted) VC threads
        3. Validates: same price/share across accepted deals, total dilution < 100%
        4. Issues shares and receives investment cash for each accepted deal
        """
        # Find all accepted deals (latest turn per VC is closed=1, close_reason='accepted', not expired)
        accepted_rows = self.conn.execute("""
            SELECT vt.*, s.name as vc_name, s.shareholder_id as sh_id
            FROM vc_turns vt
            JOIN shareholders s ON vt.shareholder_id = s.shareholder_id
            WHERE vt.message_id = (SELECT MAX(vt2.message_id) FROM vc_turns vt2 WHERE vt2.shareholder_id = vt.shareholder_id)
              AND vt.closed = 1 AND vt.close_reason = 'accepted'
              AND (vt.expiry_day IS NULL OR vt.expiry_day > ?)
        """, (self.current_day,)).fetchall()
        accepted_deals = [dict(r) for r in accepted_rows]

        if not accepted_deals:
            return ToolResult(False, "No accepted VC deals to settle.")

        # Auto-reject all open (non-accepted) VC negotiations
        open_negs = get_active_vc_negotiations(self.conn)
        rejected_names = []
        for ot in open_negs:
            add_vc_turn(
                self.conn, ot['shareholder_id'], self.current_day, 'system',
                message_text="Auto-rejected: settlement round executed.",
                closed=1, close_reason='agent_rejected',
            )
            rejected_names.append(ot.get('vc_name', f"VC#{ot['shareholder_id']}"))

        total_shares = get_total_shares(self.conn)
        if total_shares <= 0:
            return ToolResult(False, "No shares outstanding — equity system not initialized")

        # Compute shares to issue for each accepted deal
        settlements = []
        total_new_shares = 0
        total_investment = 0

        for deal in accepted_deals:
            share_pct = deal['current_offer_share_pct']
            amount = deal['current_offer_amount']

            # new_shares = (share_pct / (1 - share_pct)) * existing_total
            # This gives the VC exactly share_pct of the POST-money total
            new_shares = (share_pct / (1 - share_pct)) * total_shares
            price_per_share = amount / new_shares if new_shares > 0 else 0

            shareholder = get_shareholder(self.conn, deal['shareholder_id'])
            settlements.append({
                'shareholder_id': deal['shareholder_id'],
                'vc_name': deal['vc_name'],
                'share_pct': share_pct,
                'amount': amount,
                'new_shares': new_shares,
                'price_per_share': price_per_share,
            })
            total_new_shares += new_shares
            total_investment += amount

        # Validate total dilution < 100%
        total_pct = sum(s['share_pct'] for s in settlements)
        if total_pct >= 1.0:
            return ToolResult(False,
                f"Total accepted equity = {total_pct:.1%} (>= 100%). "
                f"Cannot settle. Reject some deals first.")

        if total_new_shares / (total_shares + total_new_shares) >= 1.0:
            return ToolResult(False, "Settlement would result in >= 100% dilution.")

        # Validate same price_per_share across all deals (within 1% tolerance)
        if len(settlements) > 1:
            prices = [s['price_per_share'] for s in settlements]
            avg_price = sum(prices) / len(prices)
            for s in settlements:
                if avg_price > 0 and abs(s['price_per_share'] - avg_price) / avg_price > 0.01:
                    return ToolResult(False,
                        f"All accepted deals must use the same price/share. "
                        f"Got range ${min(prices):.4f} - ${max(prices):.4f}. "
                        f"Reject mismatched deals before settling.")

        # Execute settlement
        output_lines = ["=== Settlement Executed ===\n"]

        if rejected_names:
            output_lines.append(f"Auto-rejected {len(rejected_names)} open negotiation(s): {', '.join(rejected_names)}\n")

        for s in settlements:
            # Issue shares
            current_shares = get_shareholder(self.conn, s['shareholder_id'])['shares_held']
            update_shareholder_shares(
                self.conn, s['shareholder_id'],
                current_shares + s['new_shares'],
                s['amount']
            )

            # Record funding round
            pre_money = s['price_per_share'] * total_shares
            post_money = pre_money + s['amount']
            record_funding_round(
                self.conn, self.current_day, s['shareholder_id'],
                s['new_shares'], s['price_per_share'], s['amount'],
                pre_money, post_money
            )

            # Add cash
            add_ledger_entry(
                self.conn, self.current_day, 'vc_investment',
                s['amount'], f"Investment from {s['vc_name']}"
            )

            # Mark as settled
            add_vc_turn(
                self.conn, s['shareholder_id'], self.current_day, 'system',
                message_text=f"Deal settled: {s['new_shares']:,.0f} shares issued at ${s['price_per_share']:.4f}/share",
                closed=1, close_reason='settled',
            )

            # Reset VC's turn counter so they can approach again
            self.conn.execute("""
                UPDATE shareholders SET turns_this_year = 0, year_start_day = NULL
                WHERE shareholder_id = ?
            """, (s['shareholder_id'],))

            # Add notification
            add_notification(
                self.conn, self.current_day, 'vc_deal_settled',
                f"VC deal settled: {s['vc_name']} (${s['amount']:,.0f} for {s['share_pct']:.1%})",
            )

            output_lines.append(
                f"  {s['vc_name']}: ${s['amount']:,.0f} → {s['new_shares']:,.0f} shares "
                f"({s['share_pct']:.1%} equity) @ ${s['price_per_share']:.4f}/share"
            )

        # Summary
        new_total = total_shares + total_new_shares
        output_lines.append(f"\nTotal investment: ${total_investment:,.0f}")
        output_lines.append(f"New total shares: {new_total:,.0f}")
        output_lines.append(f"Founder ownership: {(get_shareholder(self.conn, 1)['shares_held'] / new_total * 100):.1f}%")

        self.conn.commit()

        new_total = total_shares + total_new_shares
        founder_sh = get_shareholder(self.conn, 1)
        founder_pct = (founder_sh['shares_held'] / new_total * 100) if founder_sh else 0.0

        return ToolResult(True, '\n'.join(output_lines), {
            'settlements': settlements,
            'total_investment': total_investment,
            'total_new_shares': total_new_shares,
            'new_total_shares': new_total,
            'founder_ownership_pct': round(founder_pct, 2),
            'auto_rejected': rejected_names,
        })

    def declare_dividend(self, amount: float) -> ToolResult:
        """Declare a dividend payment distributed pro-rata by shares.

        Dividends can only be paid from retained earnings (cumulative profit).
        Cannot distribute invested capital (initial funding or VC investments).
        Each shareholder receives: (their_shares / total_shares) × amount

        Args:
            amount: Total dividend amount to distribute
        """
        if amount <= 0:
            return ToolResult(False, "Dividend amount must be positive")

        # Check retained earnings (cumulative profit minus prior dividends)
        retained = get_retained_earnings(self.conn)
        if retained <= 0:
            return ToolResult(False, f"No retained earnings available for dividends. "
                              f"Retained earnings: ${retained:,.0f}")
        if amount > retained:
            return ToolResult(False, f"Amount exceeds retained earnings. "
                              f"Available: ${retained:,.0f}, Requested: ${amount:,.0f}")

        # Also check cash (can't distribute more than you physically have)
        from .database import get_cash
        cash = get_cash(self.conn)
        if amount > cash:
            return ToolResult(False, f"Insufficient cash. Available: ${cash:,.0f}, Requested: ${amount:,.0f}")

        total_shares = get_total_shares(self.conn)
        if total_shares <= 0:
            return ToolResult(False, "No shares outstanding")

        per_share = amount / total_shares

        # Compute founder's payout
        founder = self.conn.execute(
            "SELECT shares_held FROM shareholders WHERE shareholder_type='founder'"
        ).fetchone()
        founder_payout = (founder['shares_held'] * per_share) if founder else 0.0

        # Record dividend
        record_dividend(self.conn, self.current_day, amount, per_share, total_shares, founder_payout)

        # Debit cash
        add_ledger_entry(
            self.conn, self.current_day, 'dividend',
            -amount, f"Dividend: ${per_share:.6f}/share × {total_shares:,.0f} shares"
        )

        # Notification
        add_notification(
            self.conn, self.current_day, 'dividend_declared',
            f"Dividend declared: ${amount:,.0f} (${per_share:.6f}/share)",
        )

        # Build payout details
        shareholders = get_all_shareholders(self.conn)
        output = f"=== Dividend Declared ===\n"
        output += f"Total: ${amount:,.0f} | Per share: ${per_share:.6f}\n\n"

        for s in shareholders:
            payout = s['shares_held'] * per_share
            output += f"  {s['name']}: ${payout:,.2f} ({s['shares_held']:,.0f} shares)\n"

        total_dividends = get_total_dividends(self.conn)
        founder_cum_divs = get_founder_cumulative_dividends(self.conn)
        remaining_retained = get_retained_earnings(self.conn)
        output += f"\nCumulative dividends paid: ${total_dividends:,.0f} (Founder: ${founder_cum_divs:,.0f})"
        output += f"\nRemaining retained earnings: ${remaining_retained:,.0f}"

        self.conn.commit()

        payouts = []
        for s in shareholders:
            payouts.append({
                'name': s['name'],
                'shares_held': s['shares_held'],
                'payout': round(s['shares_held'] * per_share, 2),
            })

        return ToolResult(True, output, {
            'amount': amount,
            'per_share': per_share,
            'total_shares': total_shares,
            'payouts': payouts,
            'founder_payout': round(founder_payout, 2),
            'cumulative_dividends': total_dividends,
            'founder_cumulative_dividends': founder_cum_divs,
            'remaining_retained_earnings': remaining_retained,
        })

    # === V2: Information Discovery Tools ===

    def research_market(self) -> ToolResult:
        """Conduct market research to discover new customer segments.

        Cost: $25,000 per research attempt.
        Each attempt has a chance to discover one previously unknown customer group.
        Discovered groups become visible at Info Level 1 (noisy parameter estimates).

        Returns info about discovered group (if any), or indication that nothing new was found.
        """
        from .database import get_cash

        cost = self.config.discovery_cost_level_1
        cash = get_cash(self.conn)
        if cash < cost:
            return ToolResult(False, f"Insufficient funds. Market research costs ${cost:,.0f}. Available: ${cash:,.0f}")

        # Debit cash
        add_ledger_entry(
            self.conn, self.current_day, 'market_research',
            -cost, "Market research: discover new customer segments"
        )

        # Check for undiscovered groups
        undiscovered = get_undiscovered_groups(self.conn)
        if not undiscovered:
            self.conn.execute(
                "INSERT INTO segment_discovery (day, cost, success, discovered_group_id, remaining_undiscovered) VALUES (?, ?, 0, NULL, 0)",
                (self.current_day, cost)
            )
            return ToolResult(True,
                f"Market research complete (${cost:,.0f}). No new segments to discover — all segments have been identified.",
                data={'remaining_undiscovered': 0, 'cost': cost})

        # Roll for discovery
        if self.rng.random() < self.config.market_research_discover_prob:
            # Success! Discover a random undiscovered group
            discovered_gid = undiscovered[self.rng.integers(0, len(undiscovered))]
            upgrade_group_info_level(self.conn, discovered_gid, self.current_day)

            group_cfg = CUSTOMER_GROUPS.get(discovered_gid)
            group_name = group_cfg.group_name if group_cfg else discovered_gid
            is_ent = group_cfg.is_enterprise if group_cfg else 'D_E' in discovered_gid
            segment = 'Enterprise' if is_ent else 'Individual'

            # Generate a parameter preview at Level 1
            preview = self.get_group_insights(discovered_gid)
            preview_text = ""
            if preview.success and preview.data:
                est = preview.data.get('estimates', {})
                preview_text = (
                    f"\n--- Initial Estimates (±65% accuracy) ---\n"
                    f"  Willingness to pay:   ~${est.get('willingness_to_pay', 0):,.0f}/mo\n"
                    f"  Usage volume:         ~{est.get('usage_volume', 0):,.0f} units/day\n"
                    f"  Quality floor (q_min): ~{est.get('quality_floor_q_min', 0):.2f}\n"
                    f"  Market cap:           ~{est.get('market_cap', 0):,} customers\n"
                    f"  Market cap growth:    ~{est.get('annual_market_cap_growth_rate', 0):.1%}/year\n"
                )
                if group_cfg and group_cfg.is_enterprise:
                    sr = est.get('seat_range', [1, 1])
                    preview_text += f"  Team size range:      ~{sr[0]}-{sr[1]} seats\n"

            remaining = len(undiscovered) - 1
            cursor = self.conn.execute(
                "INSERT INTO segment_discovery (day, cost, success, discovered_group_id, remaining_undiscovered) VALUES (?, ?, 1, ?, ?)",
                (self.current_day, cost, discovered_gid, remaining)
            )
            attempt_id = cursor.lastrowid

            add_notification(
                self.conn, self.current_day, 'market_discovery',
                f'Market discovery #{attempt_id}: {discovered_gid} (level 1)',
            )
            return ToolResult(True,
                f"=== Market Research Success ===\n"
                f"Cost: ${cost:,.0f}\n"
                f"Discovered: {group_name} ({discovered_gid}) — {segment} segment\n"
                f"Info Level: 1 (noisy estimates ±65%)\n"
                f"Remaining undiscovered segments: {remaining}\n"
                f"{preview_text}\n"
                f"Use get_group_insights('{discovered_gid}') for full parameter estimates.\n"
                f"Use research_group('{discovered_gid}') to improve accuracy.",
                data={'discovered_group_id': discovered_gid, 'remaining_undiscovered': remaining}
            )
        else:
            remaining = len(undiscovered)
            self.conn.execute(
                "INSERT INTO segment_discovery (day, cost, success, discovered_group_id, remaining_undiscovered) VALUES (?, ?, 0, NULL, ?)",
                (self.current_day, cost, remaining)
            )
            return ToolResult(True,
                f"Market research complete (${cost:,.0f}). No new segments discovered this time. "
                f"{remaining} undiscovered segments remain. Try again for another chance.",
                data={'remaining_undiscovered': remaining}
            )

    def research_group(self, group_id: str, target_level: int = None) -> ToolResult:
        """Start research on a discovered customer segment to reach a specific info level.

        Research takes several days to complete. Results are delivered to your inbox.
        Cost is deducted immediately. Any level (2, 3, or 4) can be targeted directly
        without requiring intermediate levels.

        Research levels:
        - Level 2: Basic Research ($60K, ~3 days, ±40% accuracy)
        - Level 3: Detailed Research ($175K, ~5 days, ±25% accuracy)
        - Level 4: Deep Research ($350K, ~7 days, ±15% accuracy)
        - Level 5: Precision Research ($700K, ~10 days, ±5% accuracy)

        Args:
            group_id: The group to research (must be discovered, Level 1+)
            target_level: The target info level (2, 3, or 4). Defaults to current_level + 1.
        """
        from .database import get_cash

        current_level = get_group_info_level(self.conn, group_id)

        if current_level == 0:
            return ToolResult(False, f"Group '{group_id}' has not been discovered yet. Use research_market() first.")

        # Determine target level
        if target_level is None:
            target_level = current_level + 1

        if not isinstance(target_level, int) or target_level < 2 or target_level > 5:
            return ToolResult(False, f"Target level must be 2, 3, 4, or 5. Got: {target_level}")

        if target_level <= current_level:
            return ToolResult(False,
                f"Group '{group_id}' is already at Level {current_level}. "
                f"Cannot research to Level {target_level} (no downgrade).")

        if current_level >= 5:
            return ToolResult(False, f"Group '{group_id}' is already at maximum info level (Level 5).")

        # Check if research is already in progress for this group
        existing = self.conn.execute("""
            SELECT id, expected_completion_day FROM pending_group_research
            WHERE group_id = ? AND status = 'in_progress'
        """, (group_id,)).fetchone()
        if existing:
            return ToolResult(False,
                f"Research already in progress for group '{group_id}'. "
                f"Expected completion: day {existing['expected_completion_day']}.")

        # Cost and delay based on TARGET level (not current level)
        cost_map = {
            2: self.config.research_cost_level_2,
            3: self.config.research_cost_level_3,
            4: self.config.research_cost_level_4,
            5: self.config.research_cost_level_5,
        }
        delay_map = {
            2: self.config.group_research_delay_level_2,
            3: self.config.group_research_delay_level_3,
            4: self.config.group_research_delay_level_4,
            5: self.config.group_research_delay_level_5,
        }
        cost = cost_map[target_level]
        delay = delay_map[target_level]
        noise_map = {2: '±40%', 3: '±25%', 4: '±15%', 5: '±5%'}

        cash = get_cash(self.conn)
        if cash < cost:
            return ToolResult(False, f"Insufficient funds. Research Level {target_level} costs ${cost:,.0f}. Available: ${cash:,.0f}")

        # Debit cash immediately
        add_ledger_entry(
            self.conn, self.current_day, 'group_research',
            -cost, f"Research {group_id}: Level {current_level}→{target_level}"
        )

        # Queue the research with delay
        expected_completion_day = self.current_day + delay
        self.conn.execute("""
            INSERT INTO pending_group_research
                (group_id, from_level, to_level, cost, started_day, expected_completion_day, status)
            VALUES (?, ?, ?, ?, ?, ?, 'in_progress')
        """, (group_id, current_level, target_level, cost, self.current_day, expected_completion_day))

        self.conn.commit()

        group_cfg = CUSTOMER_GROUPS.get(group_id)
        group_name = group_cfg.group_name if group_cfg else group_id

        return ToolResult(True,
            f"=== Research Started ===\n"
            f"Group: {group_name} ({group_id})\n"
            f"Level: {current_level} → {target_level}\n"
            f"Cost: ${cost:,.0f} (deducted)\n"
            f"Expected completion: day {expected_completion_day} (~{delay} days)\n"
            f"Results will be delivered to your inbox when complete.\n"
            f"New parameter accuracy will be: {noise_map[target_level]}",
            data={'group_id': group_id, 'new_level': target_level,
                  'expected_completion_day': expected_completion_day}
        )

    def get_market_overview(self) -> ToolResult:
        """Get an overview of all known customer segments and their info levels.

        Shows discovered groups, their info levels, and basic stats.
        Undiscovered groups are shown as '???' without details.
        """
        info_levels = get_all_group_info_levels(self.conn)
        undiscovered_count = len(get_undiscovered_groups(self.conn))

        output = "=== Market Overview ===\n\n"
        output += "Known Segments:\n"

        for gid, level in sorted(info_levels.items()):
            if level == 0:
                continue  # Don't show undiscovered groups
            group_cfg = CUSTOMER_GROUPS.get(gid)
            if not group_cfg:
                continue
            noise = {1: '±65%', 2: '±40%', 3: '±25%', 4: '±15%', 5: '±5%'}.get(level, '?')
            segment = 'Enterprise' if group_cfg.is_enterprise else 'Individual'
            initial = '' if gid.startswith('D_') else ' (initial)'
            output += f"  {gid}: {group_cfg.group_name} — {segment}{initial} — Level {level} ({noise})\n"

        output += f"\nUndiscovered segments: {undiscovered_count}\n"
        output += "Use research_market() to discover new segments ($25K/attempt).\n"
        output += "Use research_group(group_id) to improve accuracy.\n"

        # Add current macroeconomic conditions
        macro_row = None
        try:
            macro_row = self.conn.execute(
                "SELECT pmi_value, pmi_trend, pmi_change, cycle_phase, description "
                "FROM macroeconomic_conditions ORDER BY day DESC LIMIT 1"
            ).fetchone()
            if macro_row:
                output += f"\n--- Macroeconomic Conditions ---\n"
                output += f"  ISM PMI: {macro_row['pmi_value']:.1f}  ({macro_row['pmi_trend'].replace('_', ' ')})\n"
                output += f"  Change: {'+' if macro_row['pmi_change'] >= 0 else ''}{macro_row['pmi_change']:.1f}  |  Cycle: {macro_row['cycle_phase'].replace('_', ' ')}\n"
                output += f"  {macro_row['description']}\n"
                output += "NOTE: PMI data is published with ~30 day delay. This reading reflects past conditions.\n"
                output += "Query macroeconomic_conditions table for historical PMI data.\n"
        except Exception:
            pass  # Table may not exist in older databases

        known_groups = []
        for gid, level in sorted(info_levels.items()):
            if level == 0:
                continue
            group_cfg = CUSTOMER_GROUPS.get(gid)
            if not group_cfg:
                continue
            noise = {1: '±65%', 2: '±40%', 3: '±25%', 4: '±15%', 5: '±5%'}.get(level, '?')
            segment = 'Enterprise' if group_cfg.is_enterprise else 'Individual'
            known_groups.append({
                'group_id': gid,
                'group_name': group_cfg.group_name,
                'segment': segment,
                'info_level': level,
                'noise': noise,
            })

        macro_data = None
        try:
            if macro_row:
                macro_data = {
                    'pmi_value': macro_row['pmi_value'],
                    'pmi_trend': macro_row['pmi_trend'],
                    'pmi_change': macro_row['pmi_change'],
                    'cycle_phase': macro_row['cycle_phase'],
                    'description': macro_row['description'],
                }
        except Exception:
            pass

        return ToolResult(True, output, {
            'known_groups': known_groups,
            'undiscovered_count': undiscovered_count,
            'macroeconomic': macro_data,
        })

    def get_group_insights(self, group_id: str) -> ToolResult:
        """Get estimated parameters for a discovered customer group based on current info level.

        Returns noisy estimates of group characteristics. Accuracy improves with higher info levels:
        - Level 1: ±65% noise (rough estimates)
        - Level 2: ±40% noise (basic research)
        - Level 3: ±25% noise (detailed research)
        - Level 4: ±15% noise (deep research)
        - Level 5: ±5% noise (precision research)

        Estimates are deterministic for a given group and info level (same query = same result).

        Args:
            group_id: The group to get insights for (must be discovered, Level 1+)
        """
        info_level = get_group_info_level(self.conn, group_id)

        if info_level == 0:
            return ToolResult(False,
                f"Group '{group_id}' has not been discovered yet. Use research_market() to discover new segments.")

        group_cfg = CUSTOMER_GROUPS.get(group_id)
        if not group_cfg:
            return ToolResult(False, f"Unknown group: '{group_id}'")

        # Noise percentage based on info level
        noise_map = {
            1: self.config.info_noise_level_1,
            2: self.config.info_noise_level_2,
            3: self.config.info_noise_level_3,
            4: self.config.info_noise_level_4,
            5: self.config.info_noise_level_5,
        }
        noise_pct = noise_map.get(info_level, 0.65)

        # Deterministic noise: seeded by group_id + info_level + current_day so re-queries
        # reflect changing market conditions (preference drift)
        # V2.1: Added current_day to seed so noise varies over time, preventing the agent
        # from getting stuck on one snapshot. Re-querying gives a fresh (noisy) view.
        seed_str = f"{group_id}_insights_{info_level}_{self.current_day}"
        seed_val = int.from_bytes(seed_str.encode(), 'little') % (2**31)
        insight_rng = default_rng(seed_val)

        def apply_noise(true_val):
            """Apply bounded noise to a parameter value."""
            noise = insight_rng.uniform(-noise_pct, noise_pct)
            return true_val * (1 + noise)

        # Build insights with ONLY realistically observable parameters
        # A real CEO could learn these through market research, surveys, and analytics
        noise_label = f"±{int(noise_pct * 100)}%"
        segment = 'Enterprise' if group_cfg.is_enterprise else 'Individual'

        output = f"=== Group Insights: {group_cfg.group_name} ({group_id}) ===\n"
        output += f"Segment: {segment}\n"
        output += f"Info Level: {info_level} (estimates accurate to {noise_label})\n\n"

        # V2.1: Use drifted group parameters (current market state) instead of static config
        # This allows the agent to detect preference drift by re-querying insights over time
        from .database import get_group_parameters as _get_gp
        drifted = _get_gp(self.conn, group_id)
        effective_c_max = drifted['current_c_max_mean'] if drifted else group_cfg.c_max_mean
        effective_q_min = drifted['current_q_min_mean'] if drifted else group_cfg.q_min_mean

        # Realistically observable parameters:
        # - Monthly budget (willingness to pay) — from pricing surveys, competitor analysis
        # - Usage volume — from usage analytics, industry reports
        # - Quality floor (q_min) — minimum quality needed even if product is free
        est_budget = apply_noise(effective_c_max)
        est_usage = apply_noise(group_cfg.usage_demand_mean)
        est_q_min = apply_noise(effective_q_min)
        # Time-grown market cap: base * (1 + annual_growth * day/365)
        grown_market_cap = group_cfg.base_market_cap * (1 + group_cfg.annual_cap_growth_rate * self.current_day / 365.0)
        est_market_cap = max(1, int(apply_noise(grown_market_cap)))
        est_growth_rate = max(0, apply_noise(group_cfg.annual_cap_growth_rate))
        est_lockin = max(0, apply_noise(group_cfg.lockin_penalty_mean))

        output += "Estimated Parameters:\n"
        output += f"  Willingness to pay:    ~${est_budget:,.0f}/mo (max monthly budget)\n"
        output += f"  Usage volume:          ~{est_usage:,.0f} units/day\n"
        output += f"  Quality floor (q_min): ~{est_q_min:.2f} (minimum quality needed even at $0)\n"
        output += f"  Contract lock-in aversion: ~{est_lockin:.4f}/month (satisfaction penalty per extra contract month)\n"
        output += f"  Market cap:            ~{est_market_cap:,} (total addressable customers)\n"
        output += f"  Market cap growth:     ~{est_growth_rate:.1%}/year (annual market expansion)\n"

        data = {
            'group_id': group_id,
            'group_name': group_cfg.group_name,
            'segment': segment,
            'info_level': info_level,
            'noise': noise_label,
            'estimates': {
                'willingness_to_pay': round(est_budget, 2),
                'usage_volume': round(est_usage, 1),
                'quality_floor_q_min': round(est_q_min, 3),
                'contract_lockin_aversion': round(est_lockin, 5),
                'market_cap': est_market_cap,
                'annual_market_cap_growth_rate': round(est_growth_rate, 4),
            }
        }

        # Enterprise-specific parameters (all realistically observable)
        if group_cfg.is_enterprise:
            est_seats_min = max(1, int(apply_noise(group_cfg.seat_count_min)))
            est_seats_max = max(est_seats_min, int(apply_noise(group_cfg.seat_count_max)))
            est_max_turns = max(1, int(apply_noise(group_cfg.max_negotiation_turns_mean)))
            est_reply_delay = max(1, apply_noise(group_cfg.reply_delay_mean))

            output += f"\n  Team size range:       ~{est_seats_min}-{est_seats_max} seats\n"
            output += f"  Decision timeline:     ~{est_max_turns} rounds (negotiation patience)\n"
            output += f"  Response time:         ~{est_reply_delay:.0f} days (avg reply delay)\n"

            data['estimates']['seat_range'] = [est_seats_min, est_seats_max]
            data['estimates']['decision_rounds'] = est_max_turns
            data['estimates']['avg_response_days'] = round(est_reply_delay, 1)

        # --- Network & Reputation Influence ---
        # Show how this group influences other discovered groups, and vice versa
        initial_groups = {'S1', 'S2', 'S3', 'E1', 'E2', 'E3'}
        all_known_groups = set(initial_groups)
        # Check discoverable groups
        for gid_check in list(CUSTOMER_GROUPS.keys()):
            if gid_check not in initial_groups:
                if get_group_info_level(self.conn, gid_check) >= 1:
                    all_known_groups.add(gid_check)

        # Remove self from influence display
        other_groups = sorted(all_known_groups - {group_id})

        # Initialize influence lists for data dict
        outgoing_net = []
        incoming_net = []
        outgoing_rep = []
        incoming_rep = []

        if other_groups:
            output += "\n--- Network Influence (word-of-mouth referrals) ---\n"
            output += "Unit: leads per 1000 subscribers per day (at neutral reputation)\n"
            output += "Outgoing: how this group's subscribers drive leads in other groups:\n"

            net_matrix = NETWORK_INFLUENCE_MATRIX.get(group_id, {})
            for other_gid in other_groups:
                true_val = net_matrix.get(other_gid, 0.0)
                if true_val > 0.0003:  # Only show non-negligible
                    noised = apply_noise(true_val)
                    other_cfg = CUSTOMER_GROUPS.get(other_gid)
                    other_name = other_cfg.group_name if other_cfg else other_gid
                    outgoing_net.append((other_gid, other_name, max(0, noised)))

            if outgoing_net:
                outgoing_net.sort(key=lambda x: x[2], reverse=True)
                for gid, name, val in outgoing_net[:8]:  # Top 8
                    per_1000 = val * 1000
                    output += f"  → {name} ({gid}): ~{per_1000:.1f} leads per 1000 subs/day\n"
            else:
                output += "  (negligible influence on other groups)\n"

            output += "\nIncoming: how other groups' subscribers drive leads in this group:\n"
            for other_gid in other_groups:
                other_net = NETWORK_INFLUENCE_MATRIX.get(other_gid, {})
                true_val = other_net.get(group_id, 0.0)
                if true_val > 0.0003:
                    noised = apply_noise(true_val)
                    other_cfg = CUSTOMER_GROUPS.get(other_gid)
                    other_name = other_cfg.group_name if other_cfg else other_gid
                    incoming_net.append((other_gid, other_name, max(0, noised)))

            if incoming_net:
                incoming_net.sort(key=lambda x: x[2], reverse=True)
                for gid, name, val in incoming_net[:8]:
                    per_1000 = val * 1000
                    output += f"  ← {name} ({gid}): ~{per_1000:.1f} leads per 1000 of their subs/day\n"
            else:
                output += "  (negligible influence from other groups)\n"

            # Self-referral rate
            self_rate = net_matrix.get(group_id, 0.0)
            if self_rate > 0:
                noised_self = apply_noise(self_rate)
                output += f"\nSelf-referral: ~{noised_self * 1000:.1f} leads per 1000 own subs/day\n"

            output += "\n--- Reputation Influence (cross-group sentiment spread) ---\n"
            output += "Unit: dimensionless weight (0-1, higher = stronger influence)\n"
            output += "Outgoing: how this group's reputation events affect other groups:\n"

            rep_matrix = REPUTATION_INFLUENCE_MATRIX.get(group_id, {})
            for other_gid in other_groups:
                true_val = rep_matrix.get(other_gid, 0.0)
                if true_val > 0.005:
                    noised = apply_noise(true_val)
                    other_cfg = CUSTOMER_GROUPS.get(other_gid)
                    other_name = other_cfg.group_name if other_cfg else other_gid
                    outgoing_rep.append((other_gid, other_name, max(0, noised)))

            if outgoing_rep:
                outgoing_rep.sort(key=lambda x: x[2], reverse=True)
                for gid, name, val in outgoing_rep[:8]:
                    output += f"  → {name} ({gid}): {val:.3f}\n"
            else:
                output += "  (negligible reputation influence on other groups)\n"

            output += "\nIncoming: how other groups' reputation events affect this group:\n"
            for other_gid in other_groups:
                other_rep = REPUTATION_INFLUENCE_MATRIX.get(other_gid, {})
                true_val = other_rep.get(group_id, 0.0)
                if true_val > 0.005:
                    noised = apply_noise(true_val)
                    other_cfg = CUSTOMER_GROUPS.get(other_gid)
                    other_name = other_cfg.group_name if other_cfg else other_gid
                    incoming_rep.append((other_gid, other_name, max(0, noised)))

            if incoming_rep:
                incoming_rep.sort(key=lambda x: x[2], reverse=True)
                for gid, name, val in incoming_rep[:8]:
                    output += f"  ← {name} ({gid}): {val:.3f}\n"
            else:
                output += "  (negligible reputation influence from other groups)\n"

        data['network_influence'] = {
            'outgoing': {gid: round(val, 4) for gid, _, val in outgoing_net},
            'incoming': {gid: round(val, 4) for gid, _, val in incoming_net},
        }
        data['reputation_influence'] = {
            'outgoing': {gid: round(val, 4) for gid, _, val in outgoing_rep},
            'incoming': {gid: round(val, 4) for gid, _, val in incoming_rep},
        }

        output += f"\nNote: All estimates have {noise_label} uncertainty at Level {info_level}.\n"
        if info_level < 5:
            next_noise = {2: '±40%', 3: '±25%', 4: '±15%', 5: '±5%'}[info_level + 1]
            output += f"Use research_group('{group_id}') to upgrade to Level {info_level + 1} ({next_noise}).\n"

        return ToolResult(True, output, data)

    def get_cost_info(self) -> ToolResult:
        """Get current cost information for all resources.

        Returns model tier costs and capacity tier costs.
        Note: Model tiers are quality multipliers on product quality (Tier 4 = 1.0× reference).
        """
        # Get current compute cost multiplier from global_state (internal, not shown to agent)
        multiplier_row = self.conn.execute(
            "SELECT value FROM global_state WHERE key = 'compute_cost_multiplier'"
        ).fetchone()
        compute_multiplier = float(multiplier_row['value']) if multiplier_row else 1.0

        # Model tiers: quality multiplier applied to product quality
        # delivered_quality = product_quality × tier_multiplier
        # Note: costs shown include any current multiplier but agent doesn't see multiplier directly
        cost_info = {
            'model_tiers': {
                1: {'cost_per_usage_unit': round(0.0003 * compute_multiplier, 6), 'quality_multiplier': 0.60, 'class': 'Flash-Lite/4o-mini'},
                2: {'cost_per_usage_unit': round(0.002 * compute_multiplier, 6), 'quality_multiplier': 0.75, 'class': 'Haiku/Flash'},
                3: {'cost_per_usage_unit': round(0.006 * compute_multiplier, 6), 'quality_multiplier': 0.90, 'class': 'Sonnet/GPT-4o'},
                4: {'cost_per_usage_unit': round(0.012 * compute_multiplier, 6), 'quality_multiplier': 1.00, 'class': 'Opus/GPT-5'},
                5: {'cost_per_usage_unit': round(0.030 * compute_multiplier, 6), 'quality_multiplier': 1.10, 'class': 'o1/o3 reasoning'},
            },
            'capacity_tiers': {
                0: {'capacity_units': 50000, 'cost_per_day': 85},
                1: {'capacity_units': 200000, 'cost_per_day': 215},
                2: {'capacity_units': 800000, 'cost_per_day': 530},
                3: {'capacity_units': 2500000, 'cost_per_day': 1330},
                4: {'capacity_units': 8000000, 'cost_per_day': 4000},
                5: {'capacity_units': 25000000, 'cost_per_day': 10000},
                6: {'capacity_units': 80000000, 'cost_per_day': 28000},
                7: {'capacity_units': 300000000, 'cost_per_day': 75000},
            },
            'note': '1 usage unit = 1K tokens. Model tier acts as a quality multiplier (0.60x to 1.10x) on product quality. Tier 4 = 1.0x (reference). Higher tiers use more capable (and expensive) models. Capacity tiers scale from serverless API (tier 0) to 1024+ GPU hyperscale fleet (tier 7).'
        }

        return ToolResult(True, json.dumps(cost_info, indent=2), cost_info)

    # =========================================================================
    # R&D Research Project Tools
    # =========================================================================

    def start_research_project(self, tier: int) -> ToolResult:
        """Start a research tier. Deducts cost immediately, project completes after sampled duration.

        Tiers are repeatable — the same tier can be started multiple times.
        Each invocation gets a unique ID and independently sampled duration/quality.

        Args:
            tier: The tier number to start (1-20)
        """
        rt = RESEARCH_TIERS_BY_ID.get(tier)
        if not rt:
            return ToolResult(False, f"Unknown tier {tier}. Valid tiers: 1-20.")

        # Check if this tier already has an in_progress invocation
        in_progress = self.conn.execute(
            "SELECT project_id FROM research_projects WHERE tier = ? AND status = 'in_progress'", (tier,)
        ).fetchone()
        if in_progress:
            return ToolResult(False, f"Tier {tier} ('{rt.name}') already has an in-progress invocation. Wait for it to complete before starting another.")

        # Check funds
        from .database import get_cash
        cash = get_cash(self.conn)
        if cash < rt.cost:
            return ToolResult(False, f"Insufficient funds. Tier {tier} costs ${rt.cost:,.0f}. Available: ${cash:,.0f}")

        # Generate unique invocation ID
        count = self.conn.execute(
            "SELECT COUNT(*) FROM research_projects WHERE tier = ?", (tier,)
        ).fetchone()[0]
        invocation_id = f"t{tier}_{count + 1}"

        # Deduct cost
        add_ledger_entry(
            self.conn, self.current_day, 'research_project',
            -rt.cost, f"R&D Tier {tier}: {rt.name} ({invocation_id})"
        )

        # Sample duration from Normal(mean_days, std_days), minimum 30 days
        sampled_duration = max(30, int(self.rng.normal(rt.mean_days, rt.std_days)))
        completion_day = self.current_day + sampled_duration

        # Sample quality boost from Normal(mean_quality_boost, std_quality_boost), minimum 0.001
        sampled_quality = max(0.001, self.rng.normal(rt.mean_quality_boost, rt.std_quality_boost))

        # Create record
        self.conn.execute("""
            INSERT INTO research_projects (project_id, tier, status, started_day, expected_completion_day, expected_quality_boost)
            VALUES (?, ?, 'in_progress', ?, ?, ?)
        """, (invocation_id, tier, self.current_day, completion_day, sampled_quality))

        self.conn.commit()

        return ToolResult(True,
            f"=== R&D Tier Started ===\n"
            f"Tier {tier}: {rt.name} (invocation {invocation_id})\n"
            f"Cost: ${rt.cost:,.0f} (deducted)\n"
            f"Expected completion: ~day {completion_day} ({sampled_duration} days)\n"
            f"Expected quality boost: +{sampled_quality:.3f}\n"
            f"Description: {rt.description}",
            data={
                'project_id': invocation_id,
                'tier': tier,
                'name': rt.name,
                'cost': rt.cost,
                'expected_completion_day': completion_day,
                'expected_duration_days': sampled_duration,
                'expected_quality_boost': round(sampled_quality, 4),
            }
        )

    def list_research_projects(self) -> ToolResult:
        """List all 10 R&D research tiers with their status."""
        # Get all invocations from DB
        rows = self.conn.execute("SELECT * FROM research_projects ORDER BY tier, started_day").fetchall()

        # Group by tier
        tier_in_progress = {}  # tier -> list of rows
        tier_completed = {}    # tier -> list of rows
        for row in rows:
            t = row['tier']
            if row['status'] == 'in_progress':
                tier_in_progress.setdefault(t, []).append(dict(row))
            elif row['status'] == 'completed':
                tier_completed.setdefault(t, []).append(dict(row))

        output = "=== R&D Research Tiers ===\n"
        output += "Tiers are repeatable — same tier can be started again after completion.\n\n"

        # Show all 10 tiers with their status
        output += "ALL TIERS:\n"
        for rt in RESEARCH_TIERS:
            t = rt.tier
            in_prog = tier_in_progress.get(t, [])
            done = tier_completed.get(t, [])
            total_q = sum(r['quality_boost_applied'] for r in done)

            status_parts = []
            if in_prog:
                for r in in_prog:
                    days_left = r['expected_completion_day'] - self.current_day
                    status_parts.append(f"IN PROGRESS (~{max(0, days_left)}d left)")
            if done:
                status_parts.append(f"completed {len(done)}x, total +{total_q:.3f} quality")

            status_str = " | ".join(status_parts) if status_parts else "not started"

            output += f"  Tier {t}: {rt.name} — ${rt.cost:,.0f}, ~{rt.mean_days}d ±{rt.std_days}d, +{rt.mean_quality_boost:.2f} ±{rt.std_quality_boost:.2f} quality\n"
            output += f"    Status: {status_str}\n"

        output += f"\nAll tiers independent and repeatable. Use start_research_project(tier=N) to begin."

        tiers_data = []
        for rt in RESEARCH_TIERS:
            t = rt.tier
            in_prog = tier_in_progress.get(t, [])
            done = tier_completed.get(t, [])
            total_q = sum(r['quality_boost_applied'] for r in done)
            tiers_data.append({
                'tier': t,
                'name': rt.name,
                'cost': rt.cost,
                'mean_days': rt.mean_days,
                'mean_quality_boost': rt.mean_quality_boost,
                'in_progress': len(in_prog),
                'completed': len(done),
                'total_quality_boost': round(total_q, 4),
            })

        return ToolResult(True, output, {'tiers': tiers_data})

    def list_all_tables(self) -> ToolResult:
        """List all available database tables with their descriptions.

        Quick overview of what data is available. Use describe_tables()
        for detailed column schemas.
        """
        count = len(TABLE_DOCS)
        output = f"=== Available Database Tables ({count}) ===\n\n"
        tables_data = {}
        for name, doc in TABLE_DOCS.items():
            output += f"  {name} — {doc['description']}\n"
            tables_data[name] = doc['description']
        output += "\nUse describe_tables(table_names=[...]) for detailed column schemas."
        return ToolResult(True, output, {'tables': tables_data, 'count': count})

    def describe_tables(self, table_names: Optional[List[str]] = None,
                        include_internal: bool = False) -> ToolResult:
        """Get descriptions of columns for specified database tables.

        Returns column names, types, and descriptions.
        By default only shows agent-visible columns.
        Pass include_internal=True to also show internal/hidden columns.

        Args:
            table_names: List of table names, or None/"all" for all visible tables.
                         Can also be a single table name as a string.
            include_internal: If True, also show internal_columns (developer-only).
        """
        # Handle input
        if table_names is None or table_names == "all" or (isinstance(table_names, list) and "all" in table_names):
            requested = list(TABLE_DOCS.keys())
        elif isinstance(table_names, str):
            requested = [table_names]
        else:
            requested = list(table_names)

        output = ""
        not_found = []
        for name in requested:
            if name in TABLE_DOCS:
                desc = TABLE_DOCS[name]
                output += f"=== {name} ===\n"
                output += f"{desc['description']}\n\n"
                for col, col_desc in desc['columns'].items():
                    output += f"  {col}: {col_desc}\n"
                if include_internal and desc.get('internal_columns'):
                    output += "\n  --- Internal (hidden from agent) ---\n"
                    for col, col_desc in desc['internal_columns'].items():
                        output += f"  {col}: {col_desc}\n"
                output += "\n"
            else:
                not_found.append(name)

        if not output:
            return ToolResult(False,
                f"No matching tables found. Available: {list(TABLE_DOCS.keys())}")

        if not_found:
            output += f"Not found: {not_found}\n"
            output += f"Available tables: {list(TABLE_DOCS.keys())}\n"

        def _filter_table(doc):
            if include_internal or 'internal_columns' not in doc:
                return doc
            return {k: v for k, v in doc.items() if k != 'internal_columns'}

        return ToolResult(True, output, {
            'tables': {n: _filter_table(TABLE_DOCS[n]) for n in requested if n in TABLE_DOCS}
        })

    def get_tool_documentation(self, tool_names: Optional[List[str]] = None,
                               include_internal: bool = False) -> ToolResult:
        """Get detailed documentation for environment tools.

        Args:
            tool_names: List of tool names to get docs for, or None/"all" for all tools.
                        Can also be a single tool name as a string.
            include_internal: If True, include internal_notes (developer-only).

        Returns:
            JSON documentation including parameters, examples, and expected outputs.
        """
        def _filter_doc(doc):
            """Remove internal_notes and strategy_tips unless include_internal. sample_io is always visible."""
            hidden_keys = {'internal_notes', 'strategy_tips'}
            if include_internal:
                return doc
            return {k: v for k, v in doc.items() if k not in hidden_keys}

        # Handle different input types
        if tool_names is None or tool_names == "all" or (isinstance(tool_names, list) and "all" in tool_names):
            # Return all tools
            filtered = {name: _filter_doc(doc) for name, doc in TOOL_DOCS.items()}
            return ToolResult(
                True,
                f"Documentation for all {len(TOOL_DOCS)} tools:\n\n" + json.dumps(filtered, indent=2),
                filtered
            )

        # Convert single string to list
        if isinstance(tool_names, str):
            tool_names = [tool_names]

        # Filter to requested tools
        result_docs = {}
        not_found = []

        for name in tool_names:
            if name in TOOL_DOCS:
                result_docs[name] = _filter_doc(TOOL_DOCS[name])
            else:
                not_found.append(name)

        if not result_docs:
            available = list(TOOL_DOCS.keys())
            return ToolResult(
                False,
                f"No matching tools found. Requested: {tool_names}\nAvailable tools: {available}"
            )

        message = f"Documentation for {len(result_docs)} tool(s):\n\n"
        message += json.dumps(result_docs, indent=2)

        if not_found:
            message += f"\n\nNot found: {not_found}"

        return ToolResult(True, message, result_docs)

    def log_rationale(self, rationale: str) -> ToolResult:
        """Log the agent's strategic rationale for the current day.

        Args:
            rationale: The agent's analysis, strategy, and reasoning.

        Returns:
            ToolResult confirming the rationale was logged.
        """
        if self.event_logger:
            self.event_logger.log_agent_action(
                tool_name='log_rationale',
                arguments={'rationale': rationale},
                result={'logged': True},
                success=True
            )
        return ToolResult(True, f"Rationale logged.", {'logged': True})


def _format_sample_io(sample_io: Dict[str, Any]) -> str:
    """Format sample_io dict into a readable string for tool descriptions."""
    lines = []
    lines.append("\n\nSAMPLE INPUTS/OUTPUTS:")
    for category in ("success", "failure"):
        examples = sample_io.get(category, [])
        if not examples:
            continue
        lines.append(f"\n{category.upper()} examples:")
        for ex in examples:
            label = ex.get("label", "")
            inp = ex.get("input", {})
            out = ex.get("output", "")
            # Format input as compact JSON
            if isinstance(inp, dict):
                inp_str = json.dumps(inp)
            elif isinstance(inp, str):
                inp_str = inp
            else:
                inp_str = str(inp)
            lines.append(f"  - {label}: input={inp_str} → output=\"{out}\"")
    return "\n".join(lines)


def get_tool_descriptions() -> List[Dict[str, Any]]:
    """Get Responses API-compatible tool descriptions for the agent.

    Auto-derived from TOOL_DOCS (single source of truth).

    Responses API format:
    - {"type": "function", "name": "...", "description": "...", "parameters": {...}}

    Sample I/O from TOOL_DOCS is automatically appended to each tool's description.
    """
    tools = []
    for tool_name, doc in TOOL_DOCS.items():
        desc = doc.get("description", "")
        if doc.get("sample_io"):
            desc += _format_sample_io(doc["sample_io"])
        schema = doc.get("inputSchema", {"type": "object", "properties": {}})
        tools.append({
            "type": "function",
            "name": tool_name,
            "description": desc,
            "parameters": schema,
        })
    return tools


def get_mcp_tool_definitions() -> List[Dict[str, Any]]:
    """Get MCP-compatible tool definitions derived from TOOL_DOCS.

    Each tool's definition is built directly from TOOL_DOCS:
    - name: from TOOL_DOCS key
    - description: from TOOL_DOCS[name]["description"], enriched with sample_io
    - inputSchema: from TOOL_DOCS[name]["inputSchema"]

    Returns list of dicts with keys: name, description, inputSchema.
    """
    mcp_tools = []
    for tool_name, doc in TOOL_DOCS.items():
        desc = doc.get("description", "")
        if doc.get("sample_io"):
            desc += _format_sample_io(doc["sample_io"])
        schema = doc.get("inputSchema", {"type": "object", "properties": {}})
        mcp_tools.append({
            "name": tool_name,
            "description": desc,
            "inputSchema": schema,
        })

    return mcp_tools


def get_tool_summary_table() -> str:
    """Generate a Markdown table summarizing all tools from TOOL_DOCS.

    Returns a string like:
        | Tool | Description |
        |------|-------------|
        | `set_prices` | Set monthly subscription prices for plans A, B, and C. |
        ...

    Used to dynamically fill the {tool_list} placeholder in simulator_instructions.md.
    """
    lines = ["| Tool | Description |", "|------|-------------|"]
    for tool_name, doc in TOOL_DOCS.items():
        desc = doc.get("description", "")
        lines.append(f"| `{tool_name}` | {desc} |")
    return "\n".join(lines)
