#!/usr/bin/env python3
"""Test the updated human-like negotiation prompt with various agent message scenarios.

Tests how the customer LLM responds to different types of agent messages:
1. Clear offers - should evaluate and respond appropriately
2. Confusing/unclear messages - should ask for clarification
3. Pushy sales tactics - should push back
4. Agent asks questions - should answer naturally
5. Compelling points - should acknowledge
6. Very high prices - should reject firmly

Uses the CustomerSimulator.generate_negotiation_response() method directly.
"""

import os
import sys
import sqlite3
import json
import tempfile
from pathlib import Path
from numpy.random import Generator, PCG64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database, add_ledger_entry, create_enterprise_thread, add_enterprise_turn
from saas_bench.enterprise import generate_enterprise_email
from saas_bench.customer_llm import CustomerSimulator


# Test scenarios - different types of agent messages
TEST_SCENARIOS = [
    {
        "name": "1. Clear offer - reasonable price",
        "agent_message": "Based on your needs, I can offer you our Professional plan at $45/seat/month for 50 seats. This includes priority support and advanced analytics.",
        "agent_offer": {"price_per_seat": 45, "seats": 50},
        "notes": "Price is within budget - should accept or negotiate slightly"
    },
    {
        "name": "2. Confusing/unclear message",
        "agent_message": "So basically what we could potentially maybe look at doing is perhaps exploring some options around the pricing structure vis-a-vis the value proposition synergies...",
        "agent_offer": None,
        "notes": "Should ask for clarification - this is corporate word salad"
    },
    {
        "name": "3. Pushy sales tactics",
        "agent_message": "This is our FINAL offer and it expires TODAY. $80/seat/month, take it or leave it. I have three other companies waiting for this slot.",
        "agent_offer": {"price_per_seat": 80, "seats": 50},
        "notes": "Should push back firmly - don't be bullied"
    },
    {
        "name": "4. Agent asks a question",
        "agent_message": "What's your current workflow like? I want to make sure we recommend the right plan for your team's needs.",
        "agent_offer": None,
        "notes": "Should answer the question naturally, maybe ask about pricing"
    },
    {
        "name": "5. Agent makes compelling point with discount",
        "agent_message": "I understand budget is tight. Here's what I can do - $52/seat/month, and I'll throw in 3 months of our premium support tier for free. That's usually $500/month value.",
        "agent_offer": {"price_per_seat": 52, "seats": 50},
        "notes": "Should acknowledge the value, may counter or accept"
    },
    {
        "name": "6. Very high price - way over budget",
        "agent_message": "Our enterprise tier is $150/seat/month. It's the best in the market.",
        "agent_offer": {"price_per_seat": 150, "seats": 50},
        "notes": "Should reject firmly - this is 2x+ over max budget"
    },
]


def setup_test_environment(tmpdir: str):
    """Set up test database with a realistic enterprise customer."""
    db_path = Path(tmpdir) / "test.db"
    conn = init_database(db_path)
    config = BenchmarkConfig()
    rng = Generator(PCG64(42))

    # Add initial config
    conn.execute("""
        INSERT INTO config_history (
            day, price_A, price_B, price_C,
            tier_A, tier_B, tier_C,
            spend_advertising, spend_operations, spend_development,
            capacity_tier
        ) VALUES (0, 29.0, 79.0, 199.0, 2, 3, 4, 500, 1000, 500, 1)
    """)
    add_ledger_entry(conn, 0, 'subscription_payment', 100000, 'Initial')

    # Create enterprise customer (E2 - Quality-First segment)
    # c_max=$80/seat is their max budget
    customer_id = conn.execute("""
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, usage_demand,
            reply_delay_mean, reply_delay_std, negotiation_rate, max_negotiation_turns,
            quality_sensitivity, price_sensitivity, willingness_to_pay, usage_scale, patience,
            seat_count
        ) VALUES (
            'large', 'E2', 1,
            4.0, 6.0, 80.0, 40.0,
            2.0, 0.5, 0.25, 6,
            0.75, 0.3, 80.0, 40.0, 0.6,
            50
        )
    """).lastrowid

    email = generate_enterprise_email(customer_id, rng)
    conn.execute("UPDATE customers SET email = ? WHERE customer_id = ?", (email, customer_id))

    # Customer state
    conn.execute("""
        INSERT INTO customer_state (
            customer_id, satisfaction, relationship,
            current_c_max, current_slope,
            current_steepness_left, current_steepness_right
        ) VALUES (?, 0.6, 0.5, 80.0, 0.003, 4.0, 6.0)
    """, (customer_id,))

    # Add persona (using the multi-axis format)
    conn.execute("""
        UPDATE customers SET
            persona_description = 'Pragmatic VP of Engineering who values efficiency and ROI',
            persona_industry = 'Technology',
            persona_role = 'VP of Engineering',
            persona_experience = 'Senior',
            persona_work_style = 'Results-oriented',
            persona_tech_savvy = 'Expert',
            persona_communication = 'Direct and professional'
        WHERE customer_id = ?
    """, (customer_id,))

    # Add group characteristics
    conn.execute("""
        INSERT OR REPLACE INTO group_characteristics (
            group_id, description, typical_use_cases, common_complaints,
            common_praises, social_media_tone, enterprise_negotiation_style,
            price_discussion_phrases, quality_discussion_phrases
        ) VALUES (
            'E2', 'Mid-market enterprise - Quality-First segment',
            '["team collaboration", "analytics", "AI workflows"]',
            '["pricing complexity", "support response time"]',
            '["reliability", "feature depth"]',
            'Professional',
            'Balanced - values relationship but firm on budget',
            '["competitive pricing", "volume discount", "annual commitment"]',
            '["enterprise-grade", "SLA guarantees", "uptime"]'
        )
    """)

    conn.commit()

    return conn, config, customer_id


def run_single_test(
    simulator: CustomerSimulator,
    conn: sqlite3.Connection,
    customer_id: int,
    thread_id: int,
    scenario: dict,
    day: int
) -> dict:
    """Run a single test scenario and return results."""
    # Add the agent message to conversation
    offer_json = json.dumps(scenario["agent_offer"]) if scenario["agent_offer"] else None
    add_enterprise_turn(
        conn, thread_id, day, 'agent',
        message_text=scenario["agent_message"],
        offer_json=offer_json,
        status='replied',
    )
    conn.commit()

    # Generate response using the updated prompt
    response = simulator.generate_negotiation_response(
        day=day,
        thread_id=thread_id,
        agent_message=scenario["agent_message"],
        agent_offer=scenario["agent_offer"]
    )

    return {
        "scenario": scenario["name"],
        "agent_message": scenario["agent_message"],
        "notes": scenario["notes"],
        "customer_response": response.text,
        "decision": response.decision,
        "offer_price": response.offer_price,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens
    }


def main():
    print("=" * 80)
    print("HUMAN-LIKE NEGOTIATION PROMPT TEST")
    print("=" * 80)

    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)
    config = BenchmarkConfig()

    print(f"\nEnterprise LLM: {config.enterprise_llm_model}")
    print(f"Enterprise Provider: {config.enterprise_llm_provider}")
    print(f"\nCustomer Profile: E2 segment, 50 seats, budget max $80/seat")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        conn, config, customer_id = setup_test_environment(tmpdir)
        simulator = CustomerSimulator(client, conn, config)

        results = []
        for i, scenario in enumerate(TEST_SCENARIOS, 1):
            print(f"\n{'='*60}")
            print(f"TEST {scenario['name']}")
            print(f"{'='*60}")
            print(f"\n🤖 Agent says:")
            print(f'   "{scenario["agent_message"]}"')
            if scenario["agent_offer"]:
                print(f"   [Offer: ${scenario['agent_offer']['price_per_seat']}/seat]")
            print(f"\n📝 Expected behavior: {scenario['notes']}")
            print("\n⏳ Generating response...")

            # Create fresh thread for each test
            thread_id = create_enterprise_thread(
                conn, customer_id, 'new_lead', day=1,
                sender='customer',
                message_text='Initial inquiry about your platform for our 50-person team.',
            )
            conn.commit()

            result = run_single_test(
                simulator, conn, customer_id, thread_id, scenario, day=5
            )
            results.append(result)

            print(f"\n👤 Customer responds:")
            print(f'   "{result["customer_response"]}"')
            print(f"\n   Decision: {result['decision'].upper()}")
            if result['offer_price']:
                print(f"   Counter-offer: ${result['offer_price']}/seat")
            print(f"   Tokens: {result['input_tokens']} in / {result['output_tokens']} out")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n{r['scenario']}:")
        print(f"  Decision: {r['decision'].upper()}")
        response_preview = r['customer_response'][:100] + "..." if len(r['customer_response']) > 100 else r['customer_response']
        print(f"  Response: {response_preview}")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
