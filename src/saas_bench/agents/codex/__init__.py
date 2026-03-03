"""Codex agent implementation for SaaS Bench.

This module provides a Codex-based agent that can run the SaaS Bench simulation.
Codex is OpenAI's coding agent CLI that supports MCP (Model Context Protocol).

Usage (simple test):
    from saas_bench.agents.codex import run_test

    result = run_test(
        total_days=10,
        seed=42,
        scenario="default",
        model="gpt-5-codex"
    )

Usage (full runner):
    from saas_bench.agents.codex import CodexRunner, AgentConfig

    config = AgentConfig(
        model="gpt-5-codex",
        seed=42,
        scenario="default",
        total_days=3650,
    )
    runner = CodexRunner(config)
    result = runner.run(verbose=True)

Key differences from Claude Code agent:
- Uses AGENTS.md instead of CLAUDE.md for system prompt
- Uses .codex/config.toml for MCP configuration
- Uses `codex exec` with --json --full-auto for headless execution
"""

from .run_test import run_test
from .runner import CodexRunner, AgentConfig, RunResult, run_codex_agent

__all__ = [
    'run_test',
    'CodexRunner',
    'AgentConfig',
    'RunResult',
    'run_codex_agent',
]
