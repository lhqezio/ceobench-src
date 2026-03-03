"""HTTP client for communicating with the NovaMind API server."""

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


class NovaMindAPIError(Exception):
    """Raised when an API call fails."""
    pass


class _Vars:
    """Namespace for simulator variables (e.g., current_day)."""

    @property
    def current_day(self) -> int:
        """Get the current simulation day."""
        data = get_vars()
        return data.get('current_day', 0)


def _get_port() -> int:
    """Get the API server port from environment."""
    port_str = os.environ.get('NOVAMIND_API_PORT', '')
    if not port_str:
        raise NovaMindAPIError(
            "NOVAMIND_API_PORT not set. Are you running inside a novamind-operation environment?"
        )
    return int(port_str)


def _base_url() -> str:
    return f"http://127.0.0.1:{_get_port()}"


def call(tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call a tool on the API server and return the result.

    Args:
        tool_name: Name of the tool to call
        args: Tool arguments as a dict

    Returns:
        Dict with the structured result data

    Raises:
        NovaMindAPIError: On failure (also prints to stderr)
    """
    url = f"{_base_url()}/call"
    payload = json.dumps({"tool": tool_name, "args": args or {}}).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"Failed to connect to API server: {e}")
    except json.JSONDecodeError as e:
        raise NovaMindAPIError(f"Invalid JSON response: {e}")

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    return result.get('data', {})


def next_day() -> Dict[str, Any]:
    """Advance the simulator to the next day.

    Returns:
        Dict with 'day' and 'dashboard' keys
    """
    url = f"{_base_url()}/next-day"
    req = urllib.request.Request(
        url,
        data=b'{}',
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            result = json.loads(body)
            error_msg = result.get('error', f'HTTP {e.code}')
        except Exception:
            error_msg = f"HTTP {e.code}: {body.decode('utf-8', errors='replace')[:500]}"
        raise NovaMindAPIError(error_msg)
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"Failed to connect to API server: {e}")

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    return result


def query(sql: str) -> Dict[str, Any]:
    """Execute a read-only SQL query against the simulator database.

    Hidden columns and internal tables are automatically filtered.
    Write queries are blocked — use the novamind_api functions instead.

    Args:
        sql: SQL SELECT query string.

    Returns:
        Dict with 'columns' (list of column names), 'rows' (list of dicts),
        and 'row_count' (int).

    Raises:
        NovaMindAPIError: On failure (blocked query, syntax error, etc.)
    """
    url = f"{_base_url()}/query"
    payload = json.dumps({"sql": sql}).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            result = json.loads(body)
        except Exception:
            raise NovaMindAPIError(f"Query failed: HTTP {e.code}")
        error_msg = result.get('error', f'HTTP {e.code}')
        raise NovaMindAPIError(error_msg)
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"Failed to connect to API server: {e}")

    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        raise NovaMindAPIError(error_msg)

    # Print enum value hints to stderr so agent sees them
    hint = result.get('hint')
    if hint:
        print(f"\n⚠️  {hint}", file=sys.stderr)

    # Print truncation warning to stderr so agent knows to narrow the query
    warning = result.get('warning')
    if warning:
        print(f"\n⚠️  {warning}", file=sys.stderr)

    return result


def get_vars() -> Dict[str, Any]:
    """Get simulator variables."""
    url = f"{_base_url()}/vars"
    req = urllib.request.Request(url, method='GET')

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"Failed to connect to API server: {e}")


def _post(path: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic POST to the API server."""
    url = f"{_base_url()}{path}"
    payload = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"POST {path} failed: {e}")


def _get(path: str) -> Dict[str, Any]:
    """Generic GET from the API server."""
    url = f"{_base_url()}{path}"
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"GET {path} failed: {e}")


def _delete(path: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generic DELETE to the API server."""
    url = f"{_base_url()}{path}"
    payload = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method='DELETE')
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise NovaMindAPIError(f"DELETE {path} failed: {e}")


# Singleton vars instance
vars = _Vars()
