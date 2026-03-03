#!/usr/bin/env python3
"""Test negotiation prompt and show RAW LLM responses."""

import os
from openai import OpenAI

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "1. Clear offer - reasonable price",
        "agent_message": "Based on your needs, I can offer you our Professional plan at $45/seat/month for 50 seats. This includes priority support and advanced analytics.",
        "agent_offer": {"price_per_seat": 45, "seats": 50},
    },
    {
        "name": "2. Confusing/unclear message",
        "agent_message": "So basically what we could potentially maybe look at doing is perhaps exploring some options around the pricing structure vis-a-vis the value proposition synergies...",
        "agent_offer": None,
    },
    {
        "name": "3. Pushy sales tactics",
        "agent_message": "This is our FINAL offer and it expires TODAY. $80/seat/month, take it or leave it. I have three other companies waiting for this slot.",
        "agent_offer": {"price_per_seat": 80, "seats": 50},
    },
    {
        "name": "4. Agent asks a question",
        "agent_message": "What's your current workflow like? I want to make sure we recommend the right plan for your team's needs.",
        "agent_offer": None,
    },
    {
        "name": "5. Agent makes compelling point with discount",
        "agent_message": "I understand budget is tight. Here's what I can do - $52/seat/month, and I'll throw in 3 months of our premium support tier for free. That's usually $500/month value.",
        "agent_offer": {"price_per_seat": 52, "seats": 50},
    },
    {
        "name": "6. Very high price - way over budget",
        "agent_message": "Our enterprise tier is $150/seat/month. It's the best in the market.",
        "agent_offer": {"price_per_seat": 150, "seats": 50},
    },
]

# Customer profile (simulating E2 enterprise customer)
PERSONA_CONTEXT = """
Customer Profile:
- Description: Pragmatic VP of Engineering who values efficiency and ROI
- Industry: Technology
- Role: VP of Engineering
- Experience: Senior
- Communication Style: Direct and professional

Company Profile:
- Size: Mid-size startup (200 employees)
- Culture: Fast-paced and innovative
- Decision Style: Data-driven
- Primary Concern: ROI and scalability
- Negotiation Style: Balanced - values relationship but firm on budget
"""


def build_system_prompt(agent_offer, decision, final_offer_price):
    """Build the system prompt (same as in customer_llm.py)"""

    # Mock negotiation state values
    seat_count = 50
    max_accepting_price = 80.0  # c_max
    customer_offer_price = 50.44  # target price
    relationship = 0.5
    thread_type = "new_lead"
    current_plan = None
    current_price = 0

    conversation_history = "(No prior messages)"

    system_prompt = f"""You ARE this enterprise customer. React and respond like a real person would in a business negotiation.

{PERSONA_CONTEXT}

=== YOUR INTERNAL KNOWLEDGE (reference only when relevant) ===
- You need {seat_count} seats
- Your budget ceiling: ${max_accepting_price:.2f}/seat/month (don't reveal this)
- Your target price: ${customer_offer_price:.2f}/seat/month
- Current subscription: {current_plan or 'None'} at ${current_price}/month
- Relationship with this vendor: {relationship:.0%} (affects trust level)
- Thread context: {thread_type}
=== END INTERNAL KNOWLEDGE ===

Recent Conversation:
{conversation_history}

HOW TO RESPOND:
1. Read the agent's message carefully. What are they actually saying?
2. React naturally as a human would:
   - If their message is unclear or confusing → ask for clarification
   - If they're being pushy → push back or express hesitation
   - If they make a compelling point → acknowledge it
   - If they ask a question → answer it naturally
   - If they make an offer → evaluate it against your budget

3. Your current position on pricing: {decision.upper()}
   - ACCEPT: Their offer (${agent_offer.get('price_per_seat', 0) if agent_offer else 0:.2f}/seat) works for you
   - COUNTER: Propose ${final_offer_price:.2f}/seat/month
   - REJECT: Price is too high or deal doesn't work

4. Keep it natural:
   - Don't robotically state your decision
   - Respond to what they said, THEN weave in your position
   - Show appropriate emotion (enthusiasm, frustration, caution)
   - 2-4 sentences, like a real email/chat response

Output JSON:
{{
    "response": "Your natural response as this person",
    "decision": "{decision}",
    "offer_price": {final_offer_price:.2f}
}}"""

    return system_prompt


def determine_decision(agent_offer):
    """Determine decision based on offer price (simplified chassis logic)"""
    max_accepting_price = 80.0
    target_price = 50.44

    if agent_offer is None:
        return "counter", target_price

    price = agent_offer.get("price_per_seat", 0)

    if price <= target_price:
        return "accept", price
    elif price <= max_accepting_price:
        return "counter", target_price
    else:
        return "counter", target_price  # Still counter even if over budget


def main():
    print("=" * 80)
    print("NEGOTIATION PROMPT TEST - RAW LLM RESPONSES")
    print("=" * 80)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    print("\nModel: gpt-5.2")
    print("Reasoning: low")
    print("Customer: E2 segment, 50 seats, budget max $80/seat, target $50.44/seat")
    print("-" * 80)

    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*80}")
        print(f"TEST: {scenario['name']}")
        print(f"{'='*80}")

        print(f"\n🤖 AGENT MESSAGE:")
        print(f'"{scenario["agent_message"]}"')
        if scenario["agent_offer"]:
            print(f"[Offer: ${scenario['agent_offer']['price_per_seat']}/seat]")

        # Determine decision
        decision, final_offer_price = determine_decision(scenario["agent_offer"])
        print(f"\n📊 CHASSIS DECISION: {decision.upper()}, offer=${final_offer_price:.2f}/seat")

        # Build prompt
        system_prompt = build_system_prompt(scenario["agent_offer"], decision, final_offer_price)
        user_prompt = f'Agent says: "{scenario["agent_message"]}"\n\nRespond as the enterprise customer.'

        print(f"\n📝 SYSTEM PROMPT (abbreviated):")
        print("-" * 40)
        # Show just the HOW TO RESPOND section
        how_to_respond = system_prompt.split("HOW TO RESPOND:")[1].split("Output JSON:")[0]
        print(f"HOW TO RESPOND:{how_to_respond}")
        print("-" * 40)

        print(f"\n⏳ Calling LLM...")

        response = client.responses.create(
            model="gpt-5.2",
            reasoning={"effort": "low"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=300
        )

        raw_output = response.output_text

        print(f"\n📤 RAW LLM RESPONSE:")
        print("=" * 40)
        print(raw_output)
        print("=" * 40)
        print(f"\nTokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

        print("\n" + "-" * 80)

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
