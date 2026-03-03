"""Startup backstory, customer personas, and LLM-generated content for SaaS Bench.

This module provides:
1. Detailed startup backstory and world context
2. Pre-generated customer persona templates for each group
3. Group-level characteristics for consistent LLM generation
4. Functions for generating social media posts based on satisfaction
"""

import json
import sqlite3
from typing import Optional, List, Dict
from numpy.random import Generator

from .database import (
    set_world_context, get_world_context,
    add_customer_persona, get_personas_for_group, assign_persona_to_customer,
    get_customer_persona, set_group_characteristics, get_group_characteristics,
    add_social_media_post, get_group_reputation
)
from .config import CUSTOMER_GROUPS


# =============================================================================
# Startup Backstory
# =============================================================================

STARTUP_BACKSTORY = """
# NovaMind AI - Company Backstory

## The Founding Story

NovaMind AI was founded in 2023 by three former Google Brain researchers: Dr. Sarah Chen (CEO),
Marcus Rodriguez (CTO), and Dr. Aisha Patel (Chief Scientist). After years of working on
large language models, they saw an opportunity to democratize AI-powered productivity tools.

The company started in a small office in San Francisco's SOMA district, funded by a
$2M seed round from Sequoia. Their vision: make enterprise-grade AI accessible to
everyone from solo freelancers to Fortune 500 companies.

## The Product: NovaMind Assistant

NovaMind Assistant is an AI-powered productivity platform that helps users with:
- Document analysis and summarization
- Email drafting and communication assistance
- Data analysis and visualization
- Code review and debugging
- Creative writing and brainstorming

The platform uses a proprietary AI model trained on domain-specific data, offering
superior performance in business contexts compared to general-purpose chatbots.

## The Market Position

- **Plan A (Starter)**: $29/month - For individual users and small teams
- **Plan B (Professional)**: $79/month - For growing businesses
- **Plan C (Enterprise)**: $199/month - For large organizations with advanced needs

## Current Situation (Day 1 of Simulation)

After 18 months of development, NovaMind has just launched publicly. The company has:
- $500,000 in runway
- A small but dedicated early user base
- Growing word-of-mouth buzz in tech circles
- Competition from established players like Notion AI and Jasper

The founding team has brought you on as the COO to manage day-to-day operations while
they focus on product development. Your mission: grow the company to profitability
within a year while maintaining service quality and customer satisfaction.

## Key Challenges

1. **Unit Economics**: AI compute costs are high. Finding the right price-quality balance is crucial.
2. **Capacity Planning**: Over-provisioning burns cash; under-provisioning causes outages.
3. **Customer Segmentation**: Different customers have vastly different needs and price sensitivities.
4. **Enterprise Sales**: Large deals are lucrative but require relationship management.
5. **Reputation**: In the AI space, one viral negative review can undo months of good work.

## The Team

- **Engineering**: 8 developers working on the AI model and platform
- **Operations**: You (with budget for contractors)
- **Sales**: 2 account executives focused on enterprise deals
- **Support**: 1 customer success manager handling tickets

## Competitors

- **Notion AI**: Well-funded, established brand, but expensive
- **Jasper**: Strong in marketing content, weak in technical use cases
- **ChatGPT**: Generic, not specialized, but massive brand awareness
- **Various startups**: Fragmented market with many small players

## The Path Forward

Success means maximizing founder dividends while building a sustainable business.
Failure means running out of cash or destroying the brand through poor service.

The choice is yours. Good luck.
"""


# =============================================================================
# Group Characteristics (for consistent LLM generation)
# =============================================================================

GROUP_CHARACTERISTICS = {
    'S1': {
        'group_id': 'S1',
        'description': 'Price-sensitive individual users, often freelancers, students, or hobbyists. '
                      'They carefully evaluate cost vs value and are quick to churn if prices increase. '
                      'Most active on social media platforms (Instagram, TikTok, Twitter/X) where they '
                      'discover new tools through influencer recommendations and viral posts.',
        'typical_use_cases': json.dumps([
            'Personal productivity and task management',
            'Occasional document summarization',
            'Learning and experimentation with AI tools',
            'Side project assistance',
            'Email drafting for personal use'
        ]),
        'common_complaints': json.dumps([
            'Too expensive for occasional use',
            'Wish there was a cheaper tier',
            'Usage limits are too restrictive',
            'Not worth the price compared to free alternatives',
            'Billing is confusing'
        ]),
        'common_praises': json.dumps([
            'Great value for the price!',
            'Saves me so much time',
            'Love the simple interface',
            'Perfect for my needs',
            'Finally an AI tool I can afford'
        ]),
        'social_media_tone': 'Casual, price-focused, often compares to free alternatives. '
                            'Uses hashtags and emojis. Appreciates deals and discounts. '
                            'Most active on Instagram, TikTok, and Twitter/X. Discovers tools through '
                            'social media ads and influencer content.',
        'enterprise_negotiation_style': None,
        'price_discussion_phrases': json.dumps([
            'Is there a discount for yearly billing?',
            'This is a bit steep for my budget',
            'Can I get a student discount?',
            'The free tier of X does almost the same thing',
            'I might downgrade if prices go up'
        ]),
        'quality_discussion_phrases': json.dumps([
            'It gets the job done',
            'Not perfect but good enough',
            'Works well for basic tasks',
            'Sometimes the AI misunderstands me',
            'Decent quality for the price'
        ])
    },
    'S2': {
        'group_id': 'S2',
        'description': 'Quality-focused professional individuals who use AI tools daily for work. '
                      'They value reliability and output quality over price. '
                      'Research tools thoroughly via Google searches, read reviews and comparison articles. '
                      'Often found on professional forums and consume content marketing (blog posts, webinars).',
        'typical_use_cases': json.dumps([
            'Professional document preparation',
            'Client communication drafting',
            'Research and analysis',
            'Presentation creation',
            'Complex writing projects'
        ]),
        'common_complaints': json.dumps([
            'Output quality inconsistent',
            'Outages during critical deadlines',
            'Need better formatting options',
            'AI sometimes produces errors in specialized content',
            'Support response time too slow'
        ]),
        'common_praises': json.dumps([
            'Excellent quality output',
            'Essential for my workflow',
            'Reliability is outstanding',
            'Premium features justify the cost',
            'Customer support is responsive'
        ]),
        'social_media_tone': 'Professional, detailed reviews, focuses on specific features and quality. '
                            'Often shares examples of work done with the tool. '
                            'Discovers tools through search engines, blog reviews, and content marketing. '
                            'Active on LinkedIn and professional communities.',
        'enterprise_negotiation_style': None,
        'price_discussion_phrases': json.dumps([
            'Quality matters more than price to me',
            'I need this for my business, so cost is secondary',
            'Happy to pay more for premium features',
            'ROI is excellent for my use case',
            'Price is fair for what you get'
        ]),
        'quality_discussion_phrases': json.dumps([
            'The AI understands context remarkably well',
            'Output quality rivals human writers',
            'Accuracy is critical for my work',
            'I need consistent, professional results',
            'Quality has improved significantly'
        ])
    },
    'S3': {
        'group_id': 'S3',
        'description': 'Power users who push the platform to its limits. Heavy usage, often developers '
                      'or content creators who integrate AI into their core workflow. '
                      'Deep in tech communities - Hacker News, Reddit, Discord, and dev Twitter. '
                      'Strong trust in peer recommendations and technical content (blog posts, tutorials).',
        'typical_use_cases': json.dumps([
            'Bulk content generation',
            'API integration for automation',
            'Code review and debugging at scale',
            'Continuous document processing',
            'Multi-project management'
        ]),
        'common_complaints': json.dumps([
            'Usage limits too restrictive for my workflow',
            'Rate limiting interrupts my automation',
            'Need higher API quotas',
            'Performance degrades under heavy load',
            'Need better bulk processing features'
        ]),
        'common_praises': json.dumps([
            'Incredible time savings for high-volume work',
            'API is well-designed',
            'Scales well with my needs',
            'Power features are game-changing',
            'Best tool for heavy users'
        ]),
        'social_media_tone': 'Technical, detailed, often shares metrics and benchmarks. '
                            'Active in tech communities (Hacker News, Reddit, dev Twitter). '
                            'Discovers tools through technical content marketing, SEO, and referrals from peers. '
                            'Provides tutorials and tips to their community.',
        'enterprise_negotiation_style': None,
        'price_discussion_phrases': json.dumps([
            'I need unlimited or higher tier options',
            'Would pay more for no rate limits',
            'Usage-based pricing would be better for me',
            'Current limits dont scale with my needs',
            'Need enterprise features at individual pricing'
        ]),
        'quality_discussion_phrases': json.dumps([
            'Performance under load matters most',
            'Need consistent response times',
            'API reliability is crucial',
            'Quality at scale is the challenge',
            'Bulk processing quality varies'
        ])
    },
    'E1': {
        'group_id': 'E1',
        'description': 'Cost-cutting enterprises focused on reducing operational costs. '
                      'They evaluate ROI carefully and have strict budget constraints. '
                      'Decision makers are active on LinkedIn, attend virtual webinars, and '
                      'respond to targeted B2B advertising. Procurement teams compare vendor pricing.',
        'typical_use_cases': json.dumps([
            'Automating repetitive documentation tasks',
            'Reducing customer service costs',
            'Streamlining internal communications',
            'Cost-effective content production',
            'Replacing expensive consultants'
        ]),
        'common_complaints': json.dumps([
            'TCO is higher than promised',
            'Need more seats at lower cost',
            'Hidden costs in implementation',
            'ROI not meeting projections',
            'Competitors offer better volume discounts'
        ]),
        'common_praises': json.dumps([
            'Significant cost savings achieved',
            'Great ROI on investment',
            'Reduced headcount costs',
            'Efficient for our use case',
            'Volume pricing is competitive'
        ]),
        'social_media_tone': 'Professional, ROI-focused, often cites specific cost savings. '
                            'Primarily active on LinkedIn. Responds to targeted B2B advertising '
                            'and LinkedIn sponsored content. Attends vendor webinars.',
        'enterprise_negotiation_style': 'Aggressive on price, wants volume discounts, '
                                       'references competitor pricing, focused on per-seat cost.',
        'price_discussion_phrases': json.dumps([
            'Our budget is firm at X per seat',
            'Competitor Y offered us Z% less',
            'We need at least 20% discount for this volume',
            'Can we do a pilot at reduced rates?',
            'What volume discount can you offer?'
        ]),
        'quality_discussion_phrases': json.dumps([
            'Quality is acceptable for our needs',
            'Good enough is good enough',
            'We need reliability more than cutting-edge',
            'Consistency matters more than perfection',
            'Enterprise SLA requirements must be met'
        ])
    },
    'E2': {
        'group_id': 'E2',
        'description': 'Quality-first enterprises where output quality directly impacts revenue. '
                      'Often in consulting, legal, or finance where accuracy is paramount. '
                      'Executives evaluate vendors through LinkedIn thought leadership, case studies, '
                      'and content marketing (whitepapers, industry reports). Trust peer recommendations.',
        'typical_use_cases': json.dumps([
            'Client deliverable preparation',
            'Legal document analysis',
            'Financial report generation',
            'High-stakes communication drafting',
            'Compliance documentation'
        ]),
        'common_complaints': json.dumps([
            'Accuracy issues in specialized domains',
            'Need better audit trails',
            'Compliance features lacking',
            'Output not meeting quality bar',
            'Need human review integration'
        ]),
        'common_praises': json.dumps([
            'Quality meets our high standards',
            'Excellent for professional services',
            'Clients cant tell AI-assisted from human',
            'Compliance features are robust',
            'Premium quality justifies premium price'
        ]),
        'social_media_tone': 'Formal, case-study oriented, focuses on specific outcomes. '
                            'Highly active on LinkedIn. Evaluates vendors through thought leadership '
                            'content, whitepapers, and industry case studies. Values content marketing.',
        'enterprise_negotiation_style': 'Willing to pay premium for quality guarantees, '
                                       'wants SLAs with teeth, focused on compliance and security.',
        'price_discussion_phrases': json.dumps([
            'Price is secondary to quality guarantees',
            'We need SLA with financial penalties',
            'Happy to pay more for compliance features',
            'What security certifications do you have?',
            'Premium tier with dedicated support?'
        ]),
        'quality_discussion_phrases': json.dumps([
            'Accuracy must be 99%+ for our use case',
            'Output quality directly impacts our revenue',
            'We need audit trails for compliance',
            'Domain expertise in AI model is crucial',
            'Quality regression is unacceptable'
        ])
    },
    'E3': {
        'group_id': 'E3',
        'description': 'Strategic partner enterprises looking for long-term AI integration. '
                      'They want deep partnerships and co-development opportunities. '
                      'C-level executives network through LinkedIn, industry conferences, and '
                      'executive referral networks. Prefer relationship-based vendor selection.',
        'typical_use_cases': json.dumps([
            'Company-wide AI transformation',
            'Custom model training',
            'API integration into products',
            'White-label solutions',
            'Strategic AI initiatives'
        ]),
        'common_complaints': json.dumps([
            'Need more customization options',
            'Roadmap doesnt align with our needs',
            'Want more input on feature development',
            'Integration complexity too high',
            'Need dedicated account management'
        ]),
        'common_praises': json.dumps([
            'Excellent strategic partner',
            'Responsive to our specific needs',
            'Co-development has been valuable',
            'Long-term vision aligns with ours',
            'True partnership mentality'
        ]),
        'social_media_tone': 'Strategic, partnership-focused, often announces joint initiatives. '
                            'C-level engagement on LinkedIn, press release style. '
                            'Discovers vendors through executive referrals, LinkedIn networking, '
                            'and industry conferences. Referral programs highly effective.',
        'enterprise_negotiation_style': 'Relationship-focused, wants partnership terms, '
                                       'interested in equity/co-development arrangements, long deal cycles.',
        'price_discussion_phrases': json.dumps([
            'Were thinking partnership, not just vendor',
            'Can we discuss revenue sharing?',
            'What does a strategic partnership look like?',
            'Were open to equity conversations',
            'What partnership pricing can you offer?'
        ]),
        'quality_discussion_phrases': json.dumps([
            'Quality for our specific domain matters',
            'We want to help improve the model',
            'Custom training on our data?',
            'Enterprise features roadmap alignment',
            'Strategic quality improvements together'
        ])
    }
}


# =============================================================================
# Persona Templates (Pre-generated for each group)
# =============================================================================

PERSONA_TEMPLATES = {
    'S1': [
        {
            'name': 'Alex Chen',
            'job_title': 'Freelance Graphic Designer',
            'industry': 'Creative Services',
            'personality_traits': json.dumps(['budget-conscious', 'practical', 'comparison-shopper']),
            'communication_style': 'Casual and direct, uses emojis, appreciates quick responses',
            'pain_points': json.dumps(['tight budgets', 'unpredictable income', 'time constraints']),
            'goals': json.dumps(['grow freelance business', 'find affordable tools', 'save time']),
            'writing_style': 'Short, punchy posts with emojis. Often compares prices.',
            'backstory': 'Left corporate job to pursue freelance design. Carefully evaluates every subscription.'
        },
        {
            'name': 'Jordan Taylor',
            'job_title': 'Graduate Student',
            'industry': 'Academia',
            'personality_traits': json.dumps(['curious', 'budget-limited', 'tech-savvy']),
            'communication_style': 'Informal, uses internet slang, quick to share opinions',
            'pain_points': json.dumps(['student budget', 'heavy workload', 'research pressure']),
            'goals': json.dumps(['finish thesis', 'find research tools', 'stay under budget']),
            'writing_style': 'Casual with academic undertones. Appreciates student discounts.',
            'backstory': 'PhD candidate using AI tools to help with literature reviews and writing.'
        },
        {
            'name': 'Sam Rivera',
            'job_title': 'Side Hustle Entrepreneur',
            'industry': 'E-commerce',
            'personality_traits': json.dumps(['hustler', 'deal-seeker', 'resourceful']),
            'communication_style': 'Energetic, ROI-focused even for small amounts',
            'pain_points': json.dumps(['bootstrap budget', 'time split with day job', 'competition']),
            'goals': json.dumps(['scale side business', 'automate tasks', 'minimize costs']),
            'writing_style': 'Enthusiastic about deals, shares money-saving tips.',
            'backstory': 'Building an Etsy store while working full-time. Every dollar counts.'
        },
    ],
    'S2': [
        {
            'name': 'Dr. Michael Foster',
            'job_title': 'Management Consultant',
            'industry': 'Consulting',
            'personality_traits': json.dumps(['quality-focused', 'professional', 'detail-oriented']),
            'communication_style': 'Formal, articulate, provides structured feedback',
            'pain_points': json.dumps(['client deliverable quality', 'tight deadlines', 'consistency']),
            'goals': json.dumps(['impress clients', 'efficient workflows', 'premium outputs']),
            'writing_style': 'Professional reviews with specific examples and metrics.',
            'backstory': 'Senior consultant at boutique firm. Quality of output directly impacts reputation.'
        },
        {
            'name': 'Rachel Kim',
            'job_title': 'Technical Writer',
            'industry': 'Technology',
            'personality_traits': json.dumps(['meticulous', 'quality-obsessed', 'deadline-driven']),
            'communication_style': 'Clear, precise, appreciates accuracy',
            'pain_points': json.dumps(['documentation accuracy', 'version control', 'technical accuracy']),
            'goals': json.dumps(['produce excellent docs', 'meet deadlines', 'maintain quality']),
            'writing_style': 'Detailed, technical, focuses on accuracy and reliability.',
            'backstory': 'Lead technical writer for a SaaS company. AI assists with first drafts.'
        },
        {
            'name': 'David Okonkwo',
            'job_title': 'Independent Attorney',
            'industry': 'Legal',
            'personality_traits': json.dumps(['cautious', 'thorough', 'quality-demanding']),
            'communication_style': 'Measured, professional, expects precision',
            'pain_points': json.dumps(['legal accuracy', 'research time', 'client communication']),
            'goals': json.dumps(['efficient legal research', 'quality briefs', 'client satisfaction']),
            'writing_style': 'Formal, detailed reviews focused on accuracy and reliability.',
            'backstory': 'Solo practitioner using AI to compete with larger firms.'
        },
    ],
    'S3': [
        {
            'name': 'Nina Petrov',
            'job_title': 'Full-Stack Developer',
            'industry': 'Technology',
            'personality_traits': json.dumps(['technical', 'efficiency-focused', 'automation-lover']),
            'communication_style': 'Technical, direct, appreciates API documentation',
            'pain_points': json.dumps(['rate limits', 'API reliability', 'integration complexity']),
            'goals': json.dumps(['automate workflows', 'build integrations', 'maximize efficiency']),
            'writing_style': 'Technical posts with code snippets and benchmarks.',
            'backstory': 'Senior developer who has integrated AI into multiple applications.'
        },
        {
            'name': 'Marcus Zhang',
            'job_title': 'Content Agency Owner',
            'industry': 'Marketing',
            'personality_traits': json.dumps(['scale-focused', 'metrics-driven', 'efficiency-obsessed']),
            'communication_style': 'Business-oriented, talks in numbers and scale',
            'pain_points': json.dumps(['content volume demands', 'quality at scale', 'client deadlines']),
            'goals': json.dumps(['scale content production', 'maintain quality', 'reduce costs']),
            'writing_style': 'Metrics-heavy, shares volume and efficiency stats.',
            'backstory': 'Runs a 5-person content agency, AI is core to the business model.'
        },
        {
            'name': 'Sophie Anderson',
            'job_title': 'Data Scientist',
            'industry': 'Finance',
            'personality_traits': json.dumps(['analytical', 'heavy-user', 'benchmark-focused']),
            'communication_style': 'Data-driven, shares performance metrics',
            'pain_points': json.dumps(['processing limits', 'accuracy for analysis', 'API throughput']),
            'goals': json.dumps(['automate analysis', 'process large datasets', 'reliable outputs']),
            'writing_style': 'Analytical, shares benchmarks and performance data.',
            'backstory': 'Uses AI for financial analysis automation, pushes system limits daily.'
        },
    ],
    'E1': [
        {
            'name': 'Jennifer Walsh',
            'job_title': 'VP of Operations',
            'company_name': 'MidWest Manufacturing Co.',
            'industry': 'Manufacturing',
            'personality_traits': json.dumps(['cost-conscious', 'ROI-focused', 'practical']),
            'communication_style': 'Professional, always ties back to cost savings',
            'pain_points': json.dumps(['operational costs', 'efficiency gaps', 'budget constraints']),
            'goals': json.dumps(['reduce costs', 'improve efficiency', 'justify AI investment']),
            'writing_style': 'ROI-focused, shares cost savings achievements.',
            'backstory': 'Tasked with digital transformation on a tight budget.'
        },
        {
            'name': 'Robert Chen',
            'job_title': 'Director of IT',
            'company_name': 'Regional Healthcare Network',
            'industry': 'Healthcare',
            'personality_traits': json.dumps(['budget-constrained', 'compliance-aware', 'cautious']),
            'communication_style': 'Formal, risk-aware, budget-conscious',
            'pain_points': json.dumps(['tight IT budgets', 'compliance requirements', 'legacy systems']),
            'goals': json.dumps(['modernize affordably', 'maintain compliance', 'reduce manual work']),
            'writing_style': 'Professional, focuses on practical cost-benefit.',
            'backstory': 'Managing IT for a healthcare network with strict budget limits.'
        },
    ],
    'E2': [
        {
            'name': 'Victoria Sterling',
            'job_title': 'Partner',
            'company_name': 'Sterling & Associates Law Firm',
            'industry': 'Legal',
            'personality_traits': json.dumps(['quality-demanding', 'detail-oriented', 'reputation-conscious']),
            'communication_style': 'Formal, expects excellence, detailed feedback',
            'pain_points': json.dumps(['accuracy requirements', 'client expectations', 'liability concerns']),
            'goals': json.dumps(['maintain reputation', 'efficient research', 'quality outputs']),
            'writing_style': 'Formal, detailed, focuses on quality and reliability.',
            'backstory': 'Senior partner at a prestigious law firm, quality is non-negotiable.'
        },
        {
            'name': 'Dr. James Nakamura',
            'job_title': 'Chief Medical Officer',
            'company_name': 'BioTech Innovations Inc.',
            'industry': 'Biotechnology',
            'personality_traits': json.dumps(['precision-focused', 'compliance-driven', 'scientifically rigorous']),
            'communication_style': 'Scientific, precise, expects accuracy',
            'pain_points': json.dumps(['regulatory compliance', 'documentation accuracy', 'audit trails']),
            'goals': json.dumps(['maintain compliance', 'accurate documentation', 'efficient workflows']),
            'writing_style': 'Scientific, precise, compliance-focused.',
            'backstory': 'Oversees medical documentation for clinical trials, accuracy is critical.'
        },
    ],
    'E3': [
        {
            'name': 'Catherine Dubois',
            'job_title': 'Chief Strategy Officer',
            'company_name': 'Global Dynamics Corp.',
            'industry': 'Conglomerate',
            'personality_traits': json.dumps(['strategic', 'partnership-oriented', 'long-term thinker']),
            'communication_style': 'Strategic, big-picture focused, relationship-building',
            'pain_points': json.dumps(['AI strategy alignment', 'vendor relationships', 'transformation pace']),
            'goals': json.dumps(['strategic AI integration', 'partnership development', 'competitive advantage']),
            'writing_style': 'Strategic, partnership-focused, executive level.',
            'backstory': 'Leading company-wide AI transformation initiative.'
        },
        {
            'name': 'Thomas Brennan',
            'job_title': 'CEO',
            'company_name': 'Nexus Digital Solutions',
            'industry': 'Digital Services',
            'personality_traits': json.dumps(['visionary', 'partnership-seeking', 'growth-focused']),
            'communication_style': 'Visionary, talks about long-term potential',
            'pain_points': json.dumps(['scaling challenges', 'technology partnerships', 'market positioning']),
            'goals': json.dumps(['strategic growth', 'technology leadership', 'strong partnerships']),
            'writing_style': 'Visionary, announces partnerships and strategic moves.',
            'backstory': 'Building a digital services company with AI at the core.'
        },
    ],
}


# =============================================================================
# Social Media Post Generation
# =============================================================================

# Template posts for different sentiments and groups (used when no LLM available)
# V2.2: Expanded from ~4 templates per bucket to ~8 for much lower repeat rates
SOCIAL_POST_TEMPLATES = {
    'positive': {
        'S1': [
            "Just discovered {product}! Finally an AI tool that doesn't break the bank. ",
            "Loving {product} so far. Gets the job done without costing a fortune. Highly recommend for freelancers!",
            "{product} has been a game-changer for my side hustle. Worth every penny!",
            "Student discount on {product}?? Say less! This is exactly what I needed for my research.",
            "Saved 3 hours on a client gig today thanks to {product}. Paying for itself already.",
            "My friend asked what AI tool I use and I couldn't stop talking about {product} lol",
            "Used {product} to draft a proposal and the client signed same day. Just saying.",
            "{product} + coffee = unstoppable freelance morning. Best productivity stack.",
        ],
        'S2': [
            "{product} has elevated my client deliverables to a new level. The quality is outstanding.",
            "Just finished a major project with {product}. Client was impressed with the professionalism. Worth the investment.",
            "The reliability of {product} during crunch time is what keeps me coming back. Quality matters.",
            "Six months with {product} and my workflow has never been smoother. Excellent tool for professionals.",
            "Renewed my {product} subscription without hesitation. Consistent quality quarter after quarter.",
            "Used {product} for a compliance-sensitive deliverable. Output accuracy was spot on.",
            "Client feedback since adopting {product}: 'Your turnaround time is incredible.' Enough said.",
            "The attention to detail in {product} outputs saves me at least an hour of editing per project.",
        ],
        'S3': [
            "Benchmark results are in: {product} API is handling 10k requests/day flawlessly. Color me impressed.",
            "Just automated my entire content pipeline with {product}. The API documentation is actually good!",
            "Power user review: {product} scales beautifully. Finally a tool that keeps up with my workflow.",
            "Built a custom integration with {product} this weekend. Developer experience is top-notch.",
            "Pushed 50k API calls through {product} last night. Zero errors. Zero throttling. Chef's kiss.",
            "{product} latency p99 is under 200ms for my use case. Benchmarked it myself.",
            "Migrated my pipeline from competitor to {product}. 40% cost reduction, same output quality.",
            "The webhook support in {product} just simplified my entire event-driven architecture.",
        ],
        'E1': [
            "Our team's productivity increased 40% after implementing {product}. The ROI speaks for itself.",
            "Successfully deployed {product} across 50 seats. Cost savings are significant. #DigitalTransformation",
            "{product} helped us reduce operational costs by $X this quarter. Solid investment.",
            "After thorough evaluation, {product} delivered the best value for our enterprise needs.",
            "Quarterly review: {product} adoption at 92% across departments. Users love it.",
            "Our procurement team was skeptical but the {product} numbers speak for themselves.",
            "Consolidated three vendor contracts into just {product}. Simpler and cheaper.",
            "Training costs for {product} were minimal — team was productive within a week.",
        ],
        'E2': [
            "{product} meets our strict quality standards for client work. Impressive accuracy and reliability.",
            "Our compliance team approved {product} for sensitive document work. Security and quality aligned.",
            "Enterprise review: {product} delivers consistent quality at scale. Worth the premium.",
            "The audit trail features in {product} are exactly what our legal team needed.",
            "Passed our annual security audit with {product} in the stack. No findings.",
            "{product} output accuracy exceeds our 99.2% threshold consistently. Data doesn't lie.",
            "Our regulated clients trust work produced with {product}. That's the highest bar.",
            "Just renewed our {product} enterprise contract. Quality hasn't wavered in 12 months.",
        ],
        'E3': [
            "Excited to announce our strategic partnership with {product} for company-wide AI transformation.",
            "{product} team has been an excellent partner in our digital journey. True collaboration.",
            "Our AI strategy is coming together thanks to {product}. Looking forward to deeper integration.",
            "Strategic alignment with {product} is opening new possibilities for our business.",
            "Joint roadmap session with {product} was one of the most productive meetings this quarter.",
            "The {product} executive team truly understands our long-term vision. Rare in vendors.",
            "Expanding our {product} deployment to three new business units. Strong internal demand.",
            "Our board highlighted the {product} partnership as a key strategic asset this quarter.",
        ],
    },
    'negative': {
        'S1': [
            "Had to cancel {product} subscription. Just too expensive for what I actually use. ",
            "Another outage on {product}? This is the third time this month. Considering alternatives...",
            "{product} raised prices again?! Time to look for cheaper options.",
            "Disappointed with {product}. The free tier of alternatives does almost the same thing.",
            "Can't justify paying for {product} when half the features don't work properly.",
            "{product} billing is confusing. Got charged for overages I didn't even understand.",
            "My outputs from {product} have been noticeably worse this week. What changed?",
            "Asked {product} support a simple question two weeks ago. Still waiting.",
        ],
        'S2': [
            "Missed a client deadline because {product} was down. This is unacceptable for professional use.",
            "Quality has declined at {product}. My last three outputs needed significant editing. Concerning.",
            "Support response time at {product} is terrible. Been waiting 48 hours for a critical issue.",
            "{product} made errors in a legal document. Had to apologize to client. Not happy.",
            "I used to recommend {product} to colleagues. Not anymore. Quality is inconsistent.",
            "Had to redo a client presentation because {product} output was below standard. Lost half a day.",
            "The {product} UI update broke my saved workflows. Zero warning, zero migration path.",
            "Invoicing error from {product} took three emails to resolve. Unprofessional.",
        ],
        'S3': [
            "Hit rate limits again on {product}. Automation completely broke. Need higher quotas or I'm switching.",
            "{product} API degraded for 4 hours yesterday. Cost me significant productivity. Do better.",
            "Bulk processing on {product} has become unreliable. Considering self-hosting alternatives.",
            "The power user experience at {product} is getting worse. Rate limits are killing my workflow.",
            "{product} broke their API backwards compatibility. My integration is down. No changelog, nothing.",
            "Latency spikes on {product} are making my real-time pipeline unusable during peak hours.",
            "Opened a P1 ticket with {product} about data loss. Got a canned response. Unreal.",
            "Downgrading my {product} plan. The premium features don't justify 3x the cost anymore.",
        ],
        'E1': [
            "ROI projections for {product} are not meeting expectations. Reviewing our contract.",
            "Total cost of ownership for {product} is higher than pitched. Hidden costs everywhere.",
            "{product} implementation costs blew past budget. Not impressed with vendor transparency.",
            "Competitor offered 30% less than {product} for same features. Negotiation time.",
            "Our finance team flagged {product} as the most over-budget SaaS tool this quarter.",
            "User adoption of {product} dropped to 45%. Employees say it's not intuitive enough.",
            "{product} renewal quote came in 25% higher with no new features. Hard pass.",
            "We're running a formal RFP to replace {product}. Market has caught up.",
        ],
        'E2': [
            "Accuracy issues in {product} caused a compliance incident. Escalating to leadership.",
            "{product} quality regression is impacting our client relationships. Unacceptable.",
            "Failed audit because {product} logging wasn't comprehensive enough. Major gap.",
            "Our quality team flagged serious concerns about {product} output consistency.",
            "{product} data handling practices don't meet our updated regulatory requirements.",
            "Third quality incident with {product} this quarter. Documenting for executive review.",
            "Our clients noticed the decline in output quality since {product}'s last update.",
            "Requested SOC 2 Type II report from {product} three months ago. Still waiting.",
        ],
        'E3': [
            "Partnership discussions with {product} have stalled. Roadmap doesn't align with our needs.",
            "Strategic concerns about {product} long-term viability. Evaluating alternatives.",
            "{product} team turnover is affecting our integration timeline. Frustrating.",
            "Expected more from {product} as a strategic partner. Communication has been poor.",
            "Our CTO is questioning the {product} partnership. Deliverables are consistently late.",
            "{product} promised dedicated support. Instead we get the same queue as everyone else.",
            "Joint initiative with {product} is six months behind schedule. Losing confidence.",
            "Board is asking tough questions about our {product} dependency. Fair questions.",
        ],
    },
    'neutral': {
        'S1': [
            "Trying out {product}. Jury is still out. Will report back after testing it more.",
            "{product} is okay. Does what it says. Nothing special, nothing terrible.",
            "Switched to {product} from another tool. About the same honestly.",
            "Used {product} for a quick task today. It worked. That's about it.",
            "Free trial of {product} ends next week. Haven't decided if I'll pay yet.",
            "Anyone else using {product}? Curious what others think before I commit.",
            "{product} does 80% of what I need. The other 20% I still do manually.",
        ],
        'S2': [
            "Evaluating {product} for professional use. Mixed results so far.",
            "{product} works fine for basic tasks. Still testing for complex work.",
            "Three months with {product}. It's adequate but not exceptional.",
            "Showed {product} to a colleague. They said 'interesting' which means they're on the fence too.",
            "{product} handles routine work well. For edge cases, I still double-check everything.",
            "Comparing {product} against two competitors this month. No clear winner yet.",
            "The {product} learning curve is steeper than I expected but manageable.",
        ],
        'S3': [
            "Running benchmarks on {product}. Results are within expected range.",
            "{product} API is stable but nothing groundbreaking. Gets the job done.",
            "Tested {product} for bulk processing. Acceptable performance.",
            "Profiled {product} API response times. Median is fine, but p99 needs work.",
            "Wrote a wrapper around {product} to handle the edge cases. Works now.",
            "{product} documentation is decent but some endpoints are under-documented.",
            "Stress testing {product} at 2x our normal load. Holding up so far.",
        ],
        'E1': [
            "Piloting {product} across two departments. Collecting data.",
            "{product} evaluation ongoing. ROI unclear at this stage.",
            "Mixed feedback from team on {product}. Need more time to assess.",
            "Running a 90-day trial of {product}. Month one data looks flat.",
            "Our IT team has no major complaints about {product}. No major praise either.",
            "Brought {product} to the vendor review meeting. Committee wants more data.",
            "{product} onboarding for our team went smoothly. Usage metrics will tell the real story.",
        ],
        'E2': [
            "Compliance review of {product} is progressing. Some concerns, some positives.",
            "{product} quality is inconsistent. Some outputs excellent, others need work.",
            "Enterprise evaluation of {product} continues. Decision pending.",
            "Our legal team reviewed {product} terms. A few clauses need renegotiation.",
            "Quality spot-check on {product} outputs: 87% meet our threshold. Not bad, not great.",
            "Waiting on {product} to provide their updated security questionnaire responses.",
            "{product} works for standard use cases. Still evaluating for our specialized workflows.",
        ],
        'E3': [
            "Exploring partnership options with {product}. Early discussions.",
            "Strategic assessment of {product} ongoing. Potential is there.",
            "Engaging with {product} on integration possibilities. TBD.",
            "Had an introductory call with {product} leadership. Aligned on vision, details TBD.",
            "Our innovation team is prototyping with {product}. Too early to judge.",
            "Mapping {product} capabilities against our three-year technology roadmap.",
            "Due diligence on {product} partnership is underway. Report expected next month.",
        ],
    },
}


# =============================================================================
# Initialization Functions
# =============================================================================

def initialize_world_context(conn: sqlite3.Connection):
    """Initialize the startup backstory and world context."""
    set_world_context(conn, 'startup_backstory', STARTUP_BACKSTORY)
    set_world_context(conn, 'company_name', 'NovaMind AI')
    set_world_context(conn, 'product_name', 'NovaMind Assistant')
    set_world_context(conn, 'founded_year', '2023')
    set_world_context(conn, 'founders', 'Dr. Sarah Chen, Marcus Rodriguez, Dr. Aisha Patel')
    set_world_context(conn, 'mission', 'Democratize enterprise-grade AI productivity tools')
    conn.commit()


def initialize_group_characteristics(conn: sqlite3.Connection):
    """Initialize characteristics for all customer groups."""
    for group_id, chars in GROUP_CHARACTERISTICS.items():
        set_group_characteristics(conn, **chars)
    conn.commit()


def initialize_persona_templates(conn: sqlite3.Connection):
    """Initialize persona templates for all customer groups."""
    for group_id, personas in PERSONA_TEMPLATES.items():
        for persona in personas:
            add_customer_persona(
                conn,
                group_id=group_id,
                name=persona['name'],
                job_title=persona.get('job_title'),
                company_name=persona.get('company_name'),
                industry=persona.get('industry'),
                personality_traits=persona['personality_traits'],
                communication_style=persona['communication_style'],
                pain_points=persona['pain_points'],
                goals=persona['goals'],
                writing_style=persona.get('writing_style'),
                backstory=persona.get('backstory')
            )
    conn.commit()


def initialize_all_personas(conn: sqlite3.Connection):
    """Initialize all persona-related data."""
    initialize_world_context(conn)
    initialize_group_characteristics(conn)
    initialize_persona_templates(conn)


# =============================================================================
# Multi-Axis Persona Generation
# =============================================================================

def generate_customer_persona(group_id: str, rng: Generator,
                              usage_demand: float = 50.0,
                              c_max: float = 100.0,
                              seat_count: int = None) -> dict:
    """Generate a multi-axis persona for a customer based on their group and attributes.

    Args:
        group_id: Customer group (S1, S2, S3, E1, E2, E3)
        rng: Random number generator
        usage_demand: Customer's usage demand (affects usage pattern description)
        c_max: Customer's budget constraint (affects price sensitivity description)
        seat_count: Enterprise seat count (for company size context)

    Returns:
        Dictionary with all persona axes and generated description
    """
    from .config import (
        PERSONA_INDUSTRIES, PERSONA_ROLES, PERSONA_EXPERIENCE_LEVELS,
        PERSONA_WORK_STYLES, PERSONA_TECH_SAVVY, PERSONA_COMMUNICATION_STYLES,
        COMPANY_INDUSTRIES, COMPANY_SIZE_DESCRIPTORS, COMPANY_CULTURES,
        COMPANY_DECISION_STYLES, COMPANY_PRIMARY_CONCERNS, COMPANY_CONTACT_ROLES,
        CUSTOMER_GROUPS
    )

    group = CUSTOMER_GROUPS.get(group_id)
    is_enterprise = group.is_enterprise if group else (group_id.startswith('E') or group_id.startswith('D_E'))

    persona = {'group_id': group_id}

    if not is_enterprise:
        # Small customer persona (S1, S2, S3)
        persona['persona_industry'] = rng.choice(PERSONA_INDUSTRIES.get(group_id, ['general']))
        persona['persona_role'] = rng.choice(PERSONA_ROLES.get(group_id, ['professional']))
        persona['persona_experience'] = rng.choice(PERSONA_EXPERIENCE_LEVELS)
        persona['persona_work_style'] = rng.choice(PERSONA_WORK_STYLES.get(group_id, ['balanced']))
        persona['persona_tech_savvy'] = rng.choice(PERSONA_TECH_SAVVY)
        persona['persona_communication'] = rng.choice(PERSONA_COMMUNICATION_STYLES.get(group_id, ['professional']))

        # Derive usage pattern from usage_demand
        if usage_demand < 30:
            usage_pattern = 'light user'
        elif usage_demand < 70:
            usage_pattern = 'regular user'
        elif usage_demand < 120:
            usage_pattern = 'heavy user'
        else:
            usage_pattern = 'power user'

        # Derive price sensitivity from c_max relative to group
        group_c_max_mean = group.c_max_mean if group else 100.0
        if c_max < group_c_max_mean * 0.7:
            price_sensitivity = 'budget-conscious'
        elif c_max < group_c_max_mean * 1.0:
            price_sensitivity = 'value-oriented'
        else:
            price_sensitivity = 'willing to invest'

        # Derive quality expectation description from group's quality floor
        q_min_mean = group.q_min_mean if group else 0.25
        if q_min_mean < 0.20:
            quality_desc = 'pragmatic about quality'
        elif q_min_mean < 0.30:
            quality_desc = 'expects solid quality'
        elif q_min_mean < 0.40:
            quality_desc = 'high standards'
        else:
            quality_desc = 'demands excellence'

        # Generate description
        persona['persona_description'] = (
            f"{persona['persona_experience'].replace('-', ' ').title()} "
            f"{persona['persona_industry']} {persona['persona_role'].replace('-', ' ')}. "
            f"{usage_pattern.title()}, {price_sensitivity}. "
            f"{quality_desc.capitalize()}. "
            f"{persona['persona_work_style'].replace('-', ' ').capitalize()} work style, "
            f"{persona['persona_communication'].replace('-', ' ')} communicator."
        )

    else:
        # Enterprise customer persona (E1, E2, E3)
        persona['persona_industry'] = rng.choice(COMPANY_INDUSTRIES.get(group_id, ['enterprise']))
        persona['persona_role'] = rng.choice(COMPANY_CONTACT_ROLES.get(group_id, ['Director']))
        persona['persona_experience'] = 'executive'  # Enterprise contacts are senior
        persona['persona_tech_savvy'] = rng.choice(['proficient', 'advanced', 'expert'])
        persona['persona_communication'] = 'professional'  # Enterprise is always professional

        # Enterprise-specific fields
        persona['company_size_descriptor'] = rng.choice(COMPANY_SIZE_DESCRIPTORS.get(group_id, ['established']))
        persona['company_culture'] = rng.choice(COMPANY_CULTURES.get(group_id, ['professional']))
        persona['company_decision_style'] = rng.choice(COMPANY_DECISION_STYLES.get(group_id, ['thorough']))
        persona['company_primary_concern'] = rng.choice(COMPANY_PRIMARY_CONCERNS.get(group_id, ['value']))
        persona['persona_work_style'] = persona['company_culture']  # Align with company

        # Size description based on seat count
        if seat_count:
            if seat_count < 100:
                size_desc = f"({seat_count} employees)"
            elif seat_count < 500:
                size_desc = f"({seat_count} employees)"
            else:
                size_desc = f"({seat_count}+ employees)"
        else:
            size_desc = ""

        # Derive quality expectation description from group's quality floor
        q_min_mean = group.q_min_mean if group else 0.30
        if q_min_mean < 0.25:
            quality_desc = 'pragmatic about quality'
        elif q_min_mean < 0.35:
            quality_desc = 'expects reliable quality'
        elif q_min_mean < 0.45:
            quality_desc = 'demands high quality'
        else:
            quality_desc = 'requires excellence'

        # Generate company profile description
        persona['persona_description'] = (
            f"{persona['company_size_descriptor'].replace('-', ' ').title()} "
            f"{persona['persona_industry'].replace('-', ' ')} company {size_desc}. "
            f"{persona['company_culture'].replace('-', ' ').title()} culture, "
            f"{persona['company_decision_style'].replace('-', ' ')} decision-making. "
            f"Primary focus: {persona['company_primary_concern'].replace('-', ' ')}. "
            f"{quality_desc.capitalize()}. "
            f"Contact: {persona['persona_role']}."
        )

    return persona


def get_persona_for_llm(persona: dict, group_id: str) -> dict:
    """Format persona data for LLM prompt generation.

    Args:
        persona: The customer's persona dictionary
        group_id: Customer group ID

    Returns:
        Dictionary formatted for LLM context
    """
    is_enterprise = group_id.startswith('E') or group_id.startswith('D_E')

    llm_context = {
        'description': persona.get('persona_description', ''),
        'industry': persona.get('persona_industry', ''),
        'role': persona.get('persona_role', ''),
        'experience': persona.get('persona_experience', ''),
        'work_style': persona.get('persona_work_style', ''),
        'tech_savvy': persona.get('persona_tech_savvy', ''),
        'communication_style': persona.get('persona_communication', ''),
    }

    if is_enterprise:
        llm_context.update({
            'company_size': persona.get('company_size_descriptor', ''),
            'company_culture': persona.get('company_culture', ''),
            'decision_style': persona.get('company_decision_style', ''),
            'primary_concern': persona.get('company_primary_concern', ''),
        })

    return llm_context


# =============================================================================
# Social Media Post Generation
# =============================================================================

def determine_post_sentiment(satisfaction: float, rng: Generator) -> Optional[str]:
    """Determine post sentiment based on satisfaction level.

    Satisfaction is unbounded quality surplus: 0 = neutral, positive = happy, negative = unhappy.
    Typical range: -0.5 to +0.5, but can exceed.
    """
    if satisfaction >= 0.2:
        probs = {'positive': 0.7, 'neutral': 0.25, 'negative': 0.05}
    elif satisfaction >= 0.05:
        probs = {'positive': 0.4, 'neutral': 0.5, 'negative': 0.1}
    elif satisfaction >= -0.05:
        probs = {'positive': 0.15, 'neutral': 0.55, 'negative': 0.3}
    elif satisfaction >= -0.2:
        probs = {'positive': 0.05, 'neutral': 0.35, 'negative': 0.6}
    else:
        probs = {'positive': 0.02, 'neutral': 0.18, 'negative': 0.8}

    roll = rng.random()
    cumulative = 0.0
    for sentiment, prob in probs.items():
        cumulative += prob
        if roll < cumulative:
            return sentiment
    return 'neutral'


def generate_template_post(group_id: str, sentiment: str, rng: Generator) -> str:
    """Generate a post from templates (fallback when no LLM)."""
    templates = SOCIAL_POST_TEMPLATES.get(sentiment, {}).get(group_id, [])
    if not templates:
        # Fallback to generic
        templates = SOCIAL_POST_TEMPLATES.get(sentiment, {}).get('S1', ['Using the product.'])

    template = rng.choice(templates)
    return template.format(product='NovaMind')


def calculate_virality(sentiment: str, group_id: str, rng: Generator) -> tuple:
    """Calculate likes, shares, and virality score for a post."""
    # Base engagement varies by group type
    is_enterprise = group_id.startswith('E')

    if is_enterprise:
        base_likes = rng.integers(5, 50)
        base_shares = rng.integers(0, 10)
    else:
        base_likes = rng.integers(1, 100)
        base_shares = rng.integers(0, 30)

    # Negative posts tend to go more viral
    if sentiment == 'negative':
        virality_multiplier = 1.5 + rng.random()
    elif sentiment == 'positive':
        virality_multiplier = 1.0 + 0.5 * rng.random()
    else:
        virality_multiplier = 0.5 + 0.5 * rng.random()

    likes = int(base_likes * virality_multiplier)
    shares = int(base_shares * virality_multiplier)

    # Virality score (0-1, with rare spikes)
    virality_score = min(1.0, (likes + shares * 3) / 200 * virality_multiplier)

    # Rare viral spike (1% chance)
    if rng.random() < 0.01:
        virality_score = min(1.0, virality_score * (3 + 2 * rng.random()))
        likes *= int(5 + 5 * rng.random())
        shares *= int(3 + 3 * rng.random())

    return likes, shares, virality_score


def calculate_reputation_impact(sentiment: str, virality_score: float,
                                 group_id: str, rng: Generator,
                                 satisfaction: float = 0.0) -> float:
    """Calculate reputation impact from a post.

    Uses asymmetric impact: negative satisfaction produces disproportionately
    large negative reputation impact (negativity bias — bad news travels farther).

    Args:
        satisfaction: Unbounded quality surplus (0=neutral). Used to scale impact.
    """
    if sentiment == 'positive':
        # Positive impact scales linearly with satisfaction
        base_impact = 0.005 + 0.01 * rng.random()
    elif sentiment == 'negative':
        # Negative impact scales QUADRATICALLY — disproportionate damage
        # sat=-0.1 → 1.5x base, sat=-0.3 → 3.7x base, sat=-0.5 → 6x base
        neg_scale = 1.0 + 20.0 * satisfaction * satisfaction  # sat is negative, sat² is positive
        base_impact = -(0.01 + 0.02 * rng.random()) * neg_scale
    else:
        base_impact = (-0.002 + 0.004 * rng.random())  # Neutral is slightly random

    # Virality amplifies impact
    impact = base_impact * (1 + 2 * virality_score)

    # Enterprise posts have more weight
    if group_id.startswith('E'):
        impact *= 1.5

    return impact


def should_customer_post(satisfaction: float, days_subscribed: int,
                         rng: Generator) -> bool:
    """Determine if a customer should post on social media today.

    Satisfaction is unbounded quality surplus: 0 = neutral.
    Extreme values (far from 0) = more likely to post.
    """
    abs_sat = abs(satisfaction)
    if abs_sat >= 0.2:
        base_prob = 0.02  # 2% daily chance for extreme satisfaction
    elif abs_sat >= 0.1:
        base_prob = 0.01  # 1% for moderately extreme
    else:
        base_prob = 0.003  # 0.3% for neutral

    # New customers more likely to post (first 30 days)
    if days_subscribed <= 30:
        base_prob *= 2

    return rng.random() < base_prob


def generate_social_post(conn: sqlite3.Connection, day: int, customer_id: int,
                         satisfaction: float, group_id: str, rng: Generator,
                         llm_generate_func=None,
                         influence_score: float = 0.0) -> Optional[int]:
    """Generate and store a social media post.

    Args:
        conn: Database connection
        day: Current simulation day
        customer_id: Customer who is posting
        satisfaction: Customer's satisfaction level
        group_id: Customer's group (S1, S2, etc.)
        rng: Random number generator
        llm_generate_func: Optional function to generate post with LLM
                          Signature: (persona: dict, sentiment: str, context: dict) -> str
        influence_score: V2.1 - Group influence weight (hidden from agent)

    Returns:
        post_id if post was created, None otherwise
    """
    # Determine sentiment
    sentiment = determine_post_sentiment(satisfaction, rng)

    # Generate content
    if llm_generate_func:
        persona = get_customer_persona(conn, customer_id)
        context = {
            'satisfaction': satisfaction,
            'day': day,
            'product_name': get_world_context(conn, 'product_name') or 'NovaMind',
        }
        content = llm_generate_func(persona, sentiment, context)
    else:
        content = generate_template_post(group_id, sentiment, rng)

    # Calculate engagement
    likes, shares, virality = calculate_virality(sentiment, group_id, rng)

    # Store post (V2.1: includes influence_score)
    # Posts no longer directly impact reputation — reputation is driven solely by
    # System 1 (daily satisfaction weighted rep delta + direct event-based damage)
    post_id = add_social_media_post(
        conn, day, customer_id, sentiment, content,
        likes, shares, virality, 0.0,
        influence_score=influence_score
    )

    # Create notification for agent
    details = json.dumps({
        'post_id': post_id,
        'customer_id': customer_id,
        'group_id': group_id,
        'sentiment': sentiment,
        'likes': likes,
        'shares': shares,
        'virality_score': virality,
    })

    return post_id


def process_social_media_reputation(conn: sqlite3.Connection, day: int):
    """Apply reputation impacts from today's social media posts to group reputations."""
    from .config import REPUTATION_INFLUENCE_MATRIX

    # Get today's posts
    posts = conn.execute("""
        SELECT p.post_id, p.reputation_impact, c.group_id
        FROM social_media_posts p
        JOIN customers c ON p.customer_id = c.customer_id
        WHERE p.day = ?
    """, (day,)).fetchall()

    if not posts:
        return

    # Aggregate impacts by group
    group_impacts = {}
    for post in posts:
        group_id = post['group_id']
        impact = post['reputation_impact']
        group_impacts[group_id] = group_impacts.get(group_id, 0) + impact

    # Apply to reputations with cross-group influence
    from .database import set_group_reputation, get_discovered_groups

    # Only discovered groups can be source and target of cross-group reputation influence
    discovered_group_ids = set(get_discovered_groups(conn))

    for source_group, impact in group_impacts.items():
        # Source must be discovered to have cross-group influence
        if source_group not in discovered_group_ids:
            continue

        # Apply to source group directly
        current_rep = get_group_reputation(conn, source_group)
        new_rep = max(0.0, min(1.0, current_rep + impact))
        set_group_reputation(conn, source_group, new_rep, day, 'social_media')

        # Apply cross-group influence (only to other discovered groups)
        influence_row = REPUTATION_INFLUENCE_MATRIX.get(source_group, {})
        for target_group, influence in influence_row.items():
            if target_group != source_group and influence > 0 and target_group in discovered_group_ids:
                cross_impact = impact * influence * 0.3  # Dampened cross-influence
                target_rep = get_group_reputation(conn, target_group)
                new_target_rep = max(0.0, min(1.0, target_rep + cross_impact))
                if abs(cross_impact) > 0.001:  # Only log significant changes
                    set_group_reputation(
                        conn, target_group, new_target_rep, day,
                        f'cross_influence_from_{source_group}'
                    )
