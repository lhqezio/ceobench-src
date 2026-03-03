"""HTTP JSON-RPC server for NovaMind API.

Bridges the novamind_api Python library (running in a subprocess) to the
AgentTools instance (running in the main runner process). Communication
is via HTTP on localhost with a random OS-assigned port.
"""

import json
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set

from .tools import AgentTools, ToolResult
from .database import TABLE_DOCS
from .environment import build_daily_dashboard


# ---- Hidden columns / tables (same policy as python_exec sandbox) ----

_HIDDEN_TABLES: Set[str] = {
    'events',             # Internal shock/event tracking
    'api_costs',          # Meta-simulation API cost tracking
    'customer_state',     # Internal satisfaction/relationship state
    'shareholders',       # Internal equity table (use VC tools instead)
}

_HIDDEN_COLUMNS: Set[str] = {
    # Social media hidden columns
    'sentiment', 'reputation_impact', 'influence_score',
    # Latent customer satisfaction curve parameters (customers table)
    'c_max', 'alpha', 'beta', 'budget_limit',
    'steepness_left', 'steepness_right', 'quality_indifference_point',
    'usage_scale', 'trial_period_days', 'ads_return_sensitivity',
    # Subscription internals
    'daily_usage_rate', 'billing_period_usage', 'churn_reason', 'first_billing_done',
    # Research internals
    'actual_completion_day',
    # Ads revenue internals
    'sensitivity',
}

# Table-specific hidden columns (hidden only when querying these tables)
_TABLE_HIDDEN_COLUMNS: Dict[str, Set[str]] = {
    # seat_count hidden from customers/ads_revenue (internal float for drift)
    # but visible on subscriptions table (floored integer for agent)
    'customers': {'seat_count'},
    'ads_revenue': {'seat_count'},
}


def _is_schema_query(query: str) -> bool:
    """Check if query is trying to inspect database schema."""
    q = query.lower().strip()
    if q.startswith('pragma'):
        return True
    if 'sqlite_master' in q or 'sqlite_schema' in q:
        return True
    return False


def _references_hidden_table(query: str) -> Optional[str]:
    """Check if query references a hidden table. Returns table name or None."""
    q = query.lower()
    for table in _HIDDEN_TABLES:
        if re.search(r'\b' + re.escape(table) + r'\b', q):
            return table
    return None


def _get_effective_hidden(sql: str = None) -> Set[str]:
    """Get the effective set of hidden columns, including table-specific ones."""
    hidden = set(_HIDDEN_COLUMNS)
    if sql:
        q = sql.lower()
        for table, cols in _TABLE_HIDDEN_COLUMNS.items():
            if re.search(r'\b' + re.escape(table) + r'\b', q):
                hidden |= cols
    return hidden


def _strip_hidden_columns(rows: List[Dict], columns: List[str], sql: str = None) -> List[Dict]:
    """Remove hidden columns from result rows."""
    hidden = _get_effective_hidden(sql)
    visible = [c for c in columns if c not in hidden]
    return [{k: row[k] for k in visible if k in row} for row in rows]


# Build table→columns mapping for helpful error messages (exclude hidden columns)
_TABLE_COLUMNS: Dict[str, List[str]] = {
    table_name: [
        c for c in table_info['columns'].keys()
        if c not in _HIDDEN_COLUMNS and c not in _TABLE_HIDDEN_COLUMNS.get(table_name, set())
    ]
    for table_name, table_info in TABLE_DOCS.items()
}

# Build column→valid_values mapping for enum hint messages.
# Parses TABLE_DOCS column descriptions for patterns like "'val1', 'val2', 'val3'"
_COLUMN_ENUM_VALUES: Dict[str, Dict[str, List[str]]] = {}  # table -> {col -> [values]}
for _tname, _tinfo in TABLE_DOCS.items():
    for _col, _desc in _tinfo.get('columns', {}).items():
        # Skip descriptions with "e.g." — those are examples, not exhaustive enums
        if 'e.g.' in _desc.lower():
            continue
        # Extract quoted enum values from descriptions like "TEXT — 'lead', 'subscribed', 'cancelled', 'lost'"
        _vals = re.findall(r"'([^']+)'", _desc)
        if len(_vals) >= 2:  # Only treat as enum if 2+ values found
            _COLUMN_ENUM_VALUES.setdefault(_tname, {})[_col] = _vals


def _get_enum_hint_for_query(sql: str, rows: List[Dict]) -> Optional[str]:
    """If a query returned 0 rows and uses string comparisons on enum columns,
    return a hint about valid values. Returns None if no hint is applicable."""
    if rows:  # Only hint on empty results
        return None

    sql_lower = sql.lower()

    # Find table aliases: "FROM tablename alias" or "JOIN tablename alias" or "tablename AS alias"
    alias_map: Dict[str, str] = {}  # alias -> table_name
    for table_name in _COLUMN_ENUM_VALUES:
        # Match: tablename alias (no AS), tablename AS alias
        for m in re.finditer(
            r'\b' + re.escape(table_name) + r'\s+(?:as\s+)?(\w+)',
            sql_lower
        ):
            alias = m.group(1)
            # Skip SQL keywords that might follow table name
            if alias not in ('on', 'where', 'set', 'join', 'inner', 'left', 'right',
                             'outer', 'cross', 'group', 'order', 'having', 'limit',
                             'union', 'except', 'intersect', 'and', 'or', 'not',
                             'select', 'from', 'as', 'natural', 'using'):
                alias_map[alias] = table_name
        # Also match bare table name (no alias)
        if re.search(r'\b' + re.escape(table_name) + r'\b', sql_lower):
            alias_map[table_name] = table_name

    if not alias_map:
        return None

    # Find string comparisons: col = "val", col = 'val', alias.col = "val", alias.col = 'val'
    hints = []
    for m in re.finditer(r"(\w+)\.(\w+)\s*=\s*[\"']([^\"']+)[\"']", sql_lower):
        prefix, col, val = m.group(1), m.group(2), m.group(3)
        table = alias_map.get(prefix)
        if table and table in _COLUMN_ENUM_VALUES:
            enum_vals = _COLUMN_ENUM_VALUES[table].get(col)
            if enum_vals and val not in enum_vals:
                hints.append(
                    f"'{val}' is not a valid value for {table}.{col}. "
                    f"Valid values: {', '.join(repr(v) for v in enum_vals)}"
                )

    # Also match unqualified: col = "val"
    for m in re.finditer(r"(?<!\.)(\w+)\s*=\s*[\"']([^\"']+)[\"']", sql_lower):
        col, val = m.group(1), m.group(2)
        # Check if this is a prefix.col pattern (already handled above)
        start = m.start()
        if start > 0 and sql_lower[start - 1] == '.':
            continue
        # Find which tables in the query have this column with enum values
        for alias, table in alias_map.items():
            if table in _COLUMN_ENUM_VALUES:
                enum_vals = _COLUMN_ENUM_VALUES[table].get(col)
                if enum_vals and val not in enum_vals:
                    hints.append(
                        f"'{val}' is not a valid value for {table}.{col}. "
                        f"Valid values: {', '.join(repr(v) for v in enum_vals)}"
                    )

    # Deduplicate
    seen = set()
    unique_hints = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            unique_hints.append(h)

    if unique_hints:
        return "Note: " + "; ".join(unique_hints)
    return None


def _get_helpful_query_error(error: Exception, sql: str) -> str:
    """Generate a helpful error message for SQL errors, including column hints."""
    err_str = str(error).lower()

    if 'no such column' in err_str:
        match = re.search(r'no such column: ([\w.]+)', str(error))
        if match:
            bad_col = match.group(1)
            # Find tables referenced in the query
            sql_lower = sql.lower()
            matched_tables = {}
            for table_name, cols in _TABLE_COLUMNS.items():
                if re.search(r'\b' + re.escape(table_name) + r'\b', sql_lower):
                    matched_tables[table_name] = cols
            if matched_tables:
                hints = []
                for tname, cols in matched_tables.items():
                    hints.append(f"  {tname}: {', '.join(cols)}")
                return (
                    f"no such column: {bad_col}. "
                    f"Valid columns for tables in your query:\n"
                    + "\n".join(hints)
                )
            return f"no such column: {bad_col}. Use describe_tables() or read docs/tables/ to check column names."

    if 'no such table' in err_str:
        match = re.search(r'no such table: (\w+)', str(error))
        if match:
            bad_table = match.group(1)
            valid = sorted(_TABLE_COLUMNS.keys())
            return f"no such table: {bad_table}. Valid tables: {', '.join(valid)}"

    if 'ambiguous column name' in err_str:
        match = re.search(r'ambiguous column name: (\w+)', str(error))
        if match:
            col = match.group(1)
            # Find which tables have this column
            tables_with_col = [t for t, cols in _TABLE_COLUMNS.items() if col in cols]
            return (
                f"ambiguous column name: {col}. "
                f"This column exists in: {', '.join(tables_with_col)}. "
                f"Use table aliases (e.g. t.{col}) to disambiguate."
            )

    return str(error)


class _APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the NovaMind API server."""

    # Suppress default logging to stderr
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == '/call':
            self._handle_call()
        elif self.path == '/next-day':
            self._handle_next_day()
        elif self.path == '/query':
            self._handle_query()
        elif self.path == '/daily-scripts':
            self._handle_daily_scripts_post()
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_GET(self):
        if self.path == '/vars':
            self._handle_vars()
        elif self.path == '/health':
            self._send_json({"status": "ok"})
        elif self.path == '/daily-scripts':
            self._handle_daily_scripts_get()
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_DELETE(self):
        if self.path == '/daily-scripts':
            self._handle_daily_scripts_delete()
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def _read_body(self) -> Dict:
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _send_json(self, data: Dict, status: int = 200):
        response = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _handle_call(self):
        """Handle a tool call: POST /call {"tool": "...", "args": {...}}."""
        try:
            body = self._read_body()
            tool_name = body.get('tool', '')
            args = body.get('args', {})

            server: NovaMindAPIServer = self.server._api_server
            result = server.execute_tool(tool_name, args)

            if isinstance(result, ToolResult):
                self._send_json(result.to_json())
            else:
                # Fallback for non-ToolResult returns
                self._send_json({"success": True, "data": {"output": str(result)}, "message": str(result)})
        except Exception as e:
            self._send_json({"success": False, "error": str(e), "data": None}, 500)

    def _handle_next_day(self):
        """Handle next-day advancement: POST /next-day."""
        try:
            server: NovaMindAPIServer = self.server._api_server
            result = server.advance_day()
            self._send_json(result)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._send_json({"success": False, "error": f"{e}\n{tb}", "data": None}, 500)

    def _handle_query(self):
        """Handle SQL queries: POST /query {"sql": "SELECT ..."}

        Applies hidden column/table filtering so the agent cannot
        access internal simulation state.
        """
        try:
            body = self._read_body()
            sql = body.get('sql', '').strip()
            if not sql:
                self._send_json({"success": False, "error": "No SQL query provided"}, 400)
                return

            # Block schema introspection
            if _is_schema_query(sql):
                self._send_json({
                    "success": False,
                    "error": "Schema introspection queries (PRAGMA, sqlite_master) are not allowed. Read docs/tables/ for table schemas.",
                }, 403)
                return

            # Block hidden tables
            hidden_table = _references_hidden_table(sql)
            if hidden_table:
                self._send_json({
                    "success": False,
                    "error": f"Table '{hidden_table}' is not accessible.",
                }, 403)
                return

            # Block writes
            sql_lower = sql.lower().lstrip()
            if sql_lower.startswith(('insert', 'update', 'delete', 'drop', 'alter', 'create')):
                self._send_json({
                    "success": False,
                    "error": "Write queries are not allowed. Use the novamind_api for all actions.",
                }, 403)
                return

            # Enforce a row limit to prevent 60MB+ JSON responses from
            # killing the agent's bash command timeout.  If the user's SQL
            # already contains a LIMIT we respect it; otherwise we cap at
            # _QUERY_ROW_LIMIT and tell the agent to narrow its query.
            _QUERY_ROW_LIMIT = 5000

            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                cursor = server.conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                # Fetch up to limit+1 rows to detect overflow
                rows_raw = cursor.fetchmany(_QUERY_ROW_LIMIT + 1)
                truncated = len(rows_raw) > _QUERY_ROW_LIMIT
                if truncated:
                    rows_raw = rows_raw[:_QUERY_ROW_LIMIT]
                rows = [dict(row) for row in rows_raw]

            # Strip hidden columns from results
            hidden = _get_effective_hidden(sql)
            if rows and columns:
                rows = _strip_hidden_columns(rows, columns, sql)

            response = {
                "success": True,
                "columns": [c for c in columns if c not in hidden],
                "rows": rows,
                "row_count": len(rows),
            }

            if truncated:
                response["truncated"] = True
                response["warning"] = (
                    f"Result exceeded {_QUERY_ROW_LIMIT} rows and was truncated. "
                    f"Add a LIMIT clause to your query, or use COUNT/GROUP BY to "
                    f"aggregate results instead of fetching all rows."
                )

            # Add enum value hints if query returned 0 rows with wrong enum values
            enum_hint = _get_enum_hint_for_query(sql, rows)
            if enum_hint:
                response["hint"] = enum_hint

            self._send_json(response)

        except Exception as e:
            self._send_json({"success": False, "error": _get_helpful_query_error(e, sql)}, 500)

    def _handle_daily_scripts_post(self):
        """Register a daily script snapshot: POST /daily-scripts {"name": "x.py", "content": "..."}."""
        try:
            body = self._read_body()
            name = body.get('name', '')
            content = body.get('content', '')
            if not name:
                self._send_json({"success": False, "error": "name required"}, 400)
                return
            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                server._daily_scripts[name] = content
            self._send_json({"success": True, "data": {"name": name, "registered": True}})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, 500)

    def _handle_daily_scripts_get(self):
        """List registered daily scripts: GET /daily-scripts."""
        server: NovaMindAPIServer = self.server._api_server
        with server._lock:
            scripts = [{"name": n, "size": len(c)} for n, c in server._daily_scripts.items()]
        self._send_json({"success": True, "data": {"scripts": scripts}})

    def _handle_daily_scripts_delete(self):
        """Remove a daily script: DELETE /daily-scripts {"name": "x.py"}."""
        try:
            body = self._read_body()
            name = body.get('name', '')
            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                if name not in server._daily_scripts:
                    self._send_json({"success": False, "error": f"Script not found: {name}"}, 404)
                    return
                del server._daily_scripts[name]
            self._send_json({"success": True, "data": {"removed": name}})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, 500)

    def _handle_vars(self):
        """Handle variable queries: GET /vars."""
        server: NovaMindAPIServer = self.server._api_server
        self._send_json({
            "current_day": server.tools.current_day,
        })


# Map tool names to AgentTools methods + argument extraction
_TOOL_DISPATCH = {
    'set_prices': lambda tools, args: tools.set_prices({k: v for k, v in args.items() if v is not None}),
    'set_model_tiers': lambda tools, args: tools.set_model_tiers({k: v for k, v in args.items() if v is not None}),
    'set_daily_spend': lambda tools, args: tools.set_daily_spend({k: v for k, v in args.items() if v is not None}),
    'set_ad_channel_spend': lambda tools, args: tools.set_ad_channel_spend({k: v for k, v in args.items() if v is not None}),
    'set_targeted_ad_spend': lambda tools, args: tools.set_targeted_ad_spend(args.get('targeted_spend', args)),
    'set_capacity_tier': lambda tools, args: tools.set_capacity_tier(args.get('tier', args.get('capacity_tier', 0))),
    'set_usage_quotas': lambda tools, args: tools.set_usage_quotas(args),
    'send_enterprise_deal': lambda tools, args: tools.send_enterprise_deal(deals=args.get('deals', [])),
    'reject_enterprise_deal': lambda tools, args: tools.reject_enterprise_deal(deals=args.get('deals', [])),
    'get_social_posts': lambda tools, args: tools.get_social_posts(args.get('days', 7), args.get('limit', 50)),
    'get_cost_info': lambda tools, args: tools.get_cost_info(),
    'log_rationale': lambda tools, args: tools.log_rationale(args.get('rationale', args.get('text', ''))),
    'start_research_project': lambda tools, args: tools.start_research_project(args.get('tier', args.get('project_id', ''))),
    'list_research_projects': lambda tools, args: tools.list_research_projects(),
    'list_potential_vcs': lambda tools, args: tools.list_potential_vcs(),
    'send_vc_deal': lambda tools, args: tools.send_vc_deal(deals=args.get('deals', [])),
    'reject_vc_deal': lambda tools, args: tools.reject_vc_deal(deals=args.get('deals', [])),
    'get_cap_table_info': lambda tools, args: tools.get_cap_table_info(),
    'settle_investments': lambda tools, args: tools.settle_investments(),
    'declare_dividend': lambda tools, args: tools.declare_dividend(args.get('amount', 0)),
    'research_market': lambda tools, args: tools.research_market(),
    'research_group': lambda tools, args: tools.research_group(args.get('group_id', ''), args.get('target_level')),
    'get_market_overview': lambda tools, args: tools.get_market_overview(),
    'get_group_insights': lambda tools, args: tools.get_group_insights(args.get('group_id', '')),
    'set_targeted_ops_spend': lambda tools, args: tools.set_targeted_ops_spend(args.get('targeted_spend', args)),
    'set_targeted_dev_spend': lambda tools, args: tools.set_targeted_dev_spend(args.get('targeted_spend', args)),
    'set_ads_strength': lambda tools, args: tools.set_ads_strength(
        global_strength=args.get('global_strength'),
        by_group=args.get('by_group'),
        by_customer=args.get('by_customer'),
    ),
    'set_lead_promotion': lambda tools, args: tools.set_lead_promotion(
        global_promotion=args.get('global_promotion'),
        by_group=args.get('by_group'),
        by_channel=args.get('by_channel'),
        by_channel_group=args.get('by_channel_group'),
    ),
    'set_promotion': lambda tools, args: tools.set_promotion(
        global_promotion=args.get('global_promotion'),
        by_group=args.get('by_group'),
        by_customer=args.get('by_customer'),
        by_group_plan=args.get('by_group_plan'),
    ),
}


class NovaMindAPIServer:
    """HTTP API server wrapping AgentTools for subprocess communication.

    Usage:
        server = NovaMindAPIServer(tools, simulator, conn)
        server.start()  # Starts in background thread
        port = server.port  # OS-assigned port
        ...
        server.stop()
    """

    def __init__(self, tools: AgentTools, simulator=None, conn=None,
                 day_callback=None, dashboard_callback=None):
        """Initialize the API server.

        Args:
            tools: AgentTools instance to dispatch calls to
            simulator: Simulator instance for next-day advancement
            conn: Database connection for dashboard building
            day_callback: Optional callback(day, dashboard) called after advancing a day
            dashboard_callback: Optional callback(day) -> dashboard string
        """
        self.tools = tools
        self.simulator = simulator
        self.conn = conn
        self.day_callback = day_callback
        self.dashboard_callback = dashboard_callback
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0
        self._lock = threading.RLock()
        self._last_dashboard: str = ""
        self._last_day_result = None
        self._daily_scripts: Dict[str, str] = {}  # name -> content snapshot
        self._step_day_timed_out: bool = False  # Set when step_day exceeds timeout

    def start(self):
        """Start the HTTP server in a background thread."""
        self._httpd = ThreadingHTTPServer(('127.0.0.1', 0), _APIHandler)
        self._httpd._api_server = self
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool call with thread safety."""
        with self._lock:
            dispatch_fn = _TOOL_DISPATCH.get(tool_name)
            if dispatch_fn is None:
                return ToolResult(False, f"Unknown tool: {tool_name}")
            return dispatch_fn(self.tools, args)

    # Maximum allowed time for step_day before auto-quit (seconds)
    STEP_DAY_TIMEOUT = 1200

    def advance_day(self) -> Dict[str, Any]:
        """Advance the simulator by one day and return the dashboard.

        Enforces a hard timeout (STEP_DAY_TIMEOUT seconds) on step_day().
        If exceeded, returns an error so the runner can save checkpoint and exit.
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        with self._lock:
            if self.simulator is None:
                return {"success": False, "error": "No simulator configured"}

        # Run step_day in a worker thread so we can enforce a timeout.
        # We release _lock during step_day so the long computation doesn't
        # block /query or /vars endpoints.
        _step_start = _time.monotonic()

        def _do_step():
            return self.simulator.step_day()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_step)
        try:
            day_result = future.result(timeout=self.STEP_DAY_TIMEOUT)
        except FuturesTimeoutError:
            elapsed = _time.monotonic() - _step_start
            self._last_step_elapsed = elapsed
            self._step_day_timed_out = True
            # Don't wait for the orphaned thread — let it die with the process
            executor.shutdown(wait=False, cancel_futures=True)
            return {
                "success": False,
                "error": "step_day_timeout",
                "elapsed": elapsed,
                "message": f"step_day exceeded {self.STEP_DAY_TIMEOUT}s timeout ({elapsed:.1f}s elapsed). Save checkpoint and exit.",
            }
        executor.shutdown(wait=False)

        self._last_step_elapsed = _time.monotonic() - _step_start

        with self._lock:
            self._last_day_result = day_result
            new_day = self.tools.current_day + 1
            self.tools.set_current_day(new_day)

        # Build dashboard OUTSIDE the lock so daily scripts can call back
        # to the API server (e.g., nm.query()) without deadlocking.
        if self.dashboard_callback:
            dashboard = self.dashboard_callback(new_day, day_result)
        elif self.conn:
            dashboard = build_daily_dashboard(self.conn, new_day, day_result)
        else:
            dashboard = f"=== Day {new_day} Dashboard ===\n(No dashboard data available)"

        with self._lock:
            self._last_dashboard = dashboard

        if self.day_callback:
            self.day_callback(new_day, dashboard)

        return {
            "success": True,
            "day": new_day,
            "dashboard": dashboard,
        }

    @property
    def last_dashboard(self) -> str:
        return self._last_dashboard

    def get_daily_scripts(self) -> Dict[str, str]:
        """Get all registered daily script snapshots (name -> content)."""
        with self._lock:
            return dict(self._daily_scripts)

    def set_daily_scripts(self, scripts: Dict[str, str]):
        """Restore daily scripts from checkpoint."""
        with self._lock:
            self._daily_scripts = dict(scripts)
