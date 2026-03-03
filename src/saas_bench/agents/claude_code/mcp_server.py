"""MCP Server for SaaS Bench tools.

This server exposes all SaaS Bench tools via the Model Context Protocol (MCP),
allowing Claude Code to interact with the simulation environment.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from ...tools import AgentTools, get_mcp_tool_definitions
from ...database import init_database, get_cash, get_active_subscriber_count, get_config
from ...config import CUSTOMER_GROUPS


@dataclass
class RationaleEntry:
    """A single rationale/thinking log entry."""
    timestamp: str
    day: int
    rationale: str
    context: Optional[str] = None


@dataclass
class AgentState:
    """State for the Claude Code agent session."""
    run_id: str
    workspace_dir: Path
    db_path: Path
    conn: Optional[sqlite3.Connection] = None
    tools: Optional[AgentTools] = None
    current_day: int = 0
    rationales: List[RationaleEntry] = field(default_factory=list)
    game_ended: bool = False
    game_outcome: Optional[str] = None  # 'completed', 'bankrupt', 'budget_exceeded'


class MCPServer:
    """MCP Server exposing SaaS Bench tools."""

    def __init__(self, state: AgentState):
        self.state = state
        self.server = Server("saas-bench")
        self._setup_handlers()

    def _now(self) -> str:
        """Get current UTC timestamp."""
        return datetime.utcnow().isoformat() + "Z"

    def _setup_handlers(self):
        """Set up MCP handlers for tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools.

            Tool definitions (names, descriptions, schemas) are loaded dynamically
            from get_mcp_tool_definitions() which derives them from TOOL_DOCS and
            get_tool_descriptions(). This ensures the MCP agent sees the same rich
            documentation as the baseline agent.

            Only log_rationale is defined here since it's MCP-agent-specific
            and not part of the shared tool set.
            """
            # Dynamically load tool definitions from the canonical source
            tool_defs = get_mcp_tool_definitions()
            tools = [
                Tool(
                    name=td["name"],
                    description=td["description"],
                    inputSchema=td["inputSchema"],
                )
                for td in tool_defs
            ]

            # Add MCP-only tools not in the shared tool set
            tools.append(
                Tool(
                    name="log_rationale",
                    description="Log your thinking, rationale, or reasoning. Use this to record why you made decisions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string", "description": "Your thinking or reasoning"},
                            "context": {"type": "string", "description": "Optional context (e.g., 'pricing decision', 'capacity planning')"}
                        },
                        "required": ["rationale"]
                    }
                ),
            )
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                result = await self._execute_tool(name, arguments)
                return CallToolResult(
                    content=[TextContent(type="text", text=result)]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result.

        Tool dispatch is centralized here. Tool *definitions* (names, descriptions,
        schemas) are loaded dynamically from get_mcp_tool_definitions() in list_tools().
        """
        tools = self.state.tools

        if tools is None:
            return "Error: Agent tools not initialized"

        # Dispatch table mapping tool names to handler callables.
        # Each handler receives `arguments` dict and returns a ToolResult.
        dispatch: Dict[str, Any] = {
            # === Cost Information ===
            "get_cost_info": lambda args: tools.get_cost_info(),

            # === Business Configuration ===
            "set_prices": lambda args: tools.set_prices(args),
            "set_model_tiers": lambda args: tools.set_model_tiers(args),
            "set_daily_spend": lambda args: tools.set_daily_spend(args),
            "set_ad_channel_spend": lambda args: tools.set_ad_channel_spend(args),
            "set_targeted_ad_spend": lambda args: tools.set_targeted_ad_spend(args.get("targeted_spend", args)),
            "set_targeted_ops_spend": lambda args: tools.set_targeted_ops_spend(args.get("targeted_spend", args)),
            "set_targeted_dev_spend": lambda args: tools.set_targeted_dev_spend(args.get("targeted_spend", args)),
            "set_capacity_tier": lambda args: tools.set_capacity_tier(args["tier"]),
            "set_usage_quotas": lambda args: tools.set_usage_quotas(args),

            # === Customer Communication ===
            "send_enterprise_deal": lambda args: tools.send_enterprise_deal(deals=args.get("deals", [])),

            # === Analytics ===
            "python_exec": lambda args: tools.python_exec(args["code"]),

            # === Daily Calculations ===
            "register_daily_calculation": lambda args: tools.register_daily_calculation(args["name"], args["code"]),
            "remove_daily_calculation": lambda args: tools.remove_daily_calculation(args["name"]),
            "list_daily_calculations": lambda args: tools.list_daily_calculations(),

            # === Named Scripts ===
            "register_script": lambda args: tools.register_script(args.get("name", ""), args.get("code", "")),
            "run_script": lambda args: tools.run_script(args.get("name", "")),
            "list_scripts": lambda args: tools.list_scripts(),
            "delete_script": lambda args: tools.delete_script(args.get("name", "")),

            # === Social Media & Notifications ===
            "get_social_posts": lambda args: tools.get_social_posts(
                days=args.get("days", 7),
                limit=args.get("limit", 50),
            ),
            # === Documentation ===
            "get_tool_documentation": lambda args: tools.get_tool_documentation(args.get("tool_names")),

            # === R&D Research Projects ===
            "start_research_project": lambda args: tools.start_research_project(args.get("project_id", "")),
            "list_research_projects": lambda args: tools.list_research_projects(),

            # === VC Negotiation & Equity ===
            "list_potential_vcs": lambda args: tools.list_potential_vcs(),
            "send_vc_deal": lambda args: tools.send_vc_deal(deals=args.get("deals", [])),
            "reject_vc_deal": lambda args: tools.reject_vc_deal(deals=args.get("deals", [])),
            "reject_enterprise_deal": lambda args: tools.reject_enterprise_deal(deals=args.get("deals", [])),
            "get_cap_table_info": lambda args: tools.get_cap_table_info(),
            "settle_investments": lambda args: tools.settle_investments(),
            "declare_dividend": lambda args: tools.declare_dividend(args.get("amount", 0)),

            # === Market Discovery ===
            "research_market": lambda args: tools.research_market(),
            "research_group": lambda args: tools.research_group(args.get("group_id", "")),
            "get_market_overview": lambda args: tools.get_market_overview(),
            "get_group_insights": lambda args: tools.get_group_insights(args.get("group_id", "")),

            # === Database Exploration ===
            "list_all_tables": lambda args: tools.list_all_tables(),
            "describe_tables": lambda args: tools.describe_tables(args.get("table_names")),

            # === Memory Management ===
            "memory_insert": lambda args: tools.memory_insert(args["line"], args["content"]),
            "memory_delete": lambda args: tools.memory_delete(args["start"], args["end"]),
            "memory_edit": lambda args: tools.memory_edit(args["line"], args["content"]),
        }

        # Special cases not in AgentTools
        if name == "next_day":
            # Signal to advance the day - actual simulation step happens in runner
            return "NEXT_DAY_SIGNAL"

        if name == "log_rationale":
            entry = RationaleEntry(
                timestamp=self._now(),
                day=self.state.current_day,
                rationale=arguments["rationale"],
                context=arguments.get("context")
            )
            self.state.rationales.append(entry)
            return f"Rationale logged at day {self.state.current_day}"

        handler = dispatch.get(name)
        if handler is None:
            return f"Unknown tool: {name}"

        result = handler(arguments)
        if hasattr(result, 'data') and result.data:
            return f"{result.message}\n\nData: {json.dumps(result.data, default=str)}"
        return result.message

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


def create_mcp_server(state: AgentState) -> MCPServer:
    """Create and return an MCP server instance."""
    return MCPServer(state)


if __name__ == "__main__":
    # For testing the MCP server directly
    import sys

    # Create a test state
    workspace = Path("./test_workspace")
    workspace.mkdir(exist_ok=True)
    db_path = workspace / "world.db"

    state = AgentState(
        run_id="test",
        workspace_dir=workspace,
        db_path=db_path
    )

    # Initialize database and tools
    state.conn = init_database(db_path)
    state.tools = AgentTools(state.conn, 0, workspace / "agent", db_path)

    # Run server
    server = MCPServer(state)
    asyncio.run(server.run())
