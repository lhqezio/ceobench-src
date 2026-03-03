"""OpenCode Agent for SaaS Bench.

This agent uses OpenCode (opencode-ai), an open-source AI coding agent CLI,
to interact with the SaaS Bench simulation.

OpenCode supports:
- AGENTS.md for custom instructions (equivalent to CLAUDE.md)
- opencode.json for model/provider configuration
- MCP (Model Context Protocol) for tool integration
- Non-interactive execution via `opencode run`
"""

from .runner import OpenCodeRunner, run_opencode_agent, AgentConfig, RunResult

__all__ = ['OpenCodeRunner', 'run_opencode_agent', 'AgentConfig', 'RunResult']
