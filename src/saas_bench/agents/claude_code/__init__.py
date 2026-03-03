"""Claude Code Agent for SaaS Bench.

This module implements a Claude Code-based agent that uses MCP (Model Context Protocol)
to interact with the SaaS Bench simulation environment.
"""

from .mcp_server import MCPServer
from .runner import ClaudeCodeRunner

__all__ = ['MCPServer', 'ClaudeCodeRunner']
