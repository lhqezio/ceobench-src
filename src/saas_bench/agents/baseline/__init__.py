"""Baseline LLM agent for SaaS Bench.

This agent uses an LLM to make decisions. Supports OpenAI-compatible APIs
(OpenAI, xAI) and Anthropic APIs (direct, Bedrock).
It maintains a conversation context and refreshes it after each day.
"""

from .agent import BaselineAgent

__all__ = ['BaselineAgent']
