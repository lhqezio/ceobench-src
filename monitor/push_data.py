"""Push run data to Modal volume for the monitoring dashboard.

Runs locally on the cluster. Dumps all run stats + recent actions to JSON,
then uploads to a Modal volume that the dashboard app reads from.

Usage:
    # One-shot push
    python push_data.py

    # Continuous push every N seconds
    python push_data.py --loop 30
"""

import json
import sqlite3
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import db_protection
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

RUNS_DIR = _PROJECT_ROOT / "bash_agent_runs"
OUTPUT_FILE = Path(__file__).parent / "data.json"
MODAL_VOLUME = "bossbench-monitor-data"


def _open_run_db(run_dir: Path) -> sqlite3.Connection | None:
    """Open the run's obfuscated .nmdb database into an in-memory connection.

    Returns an open sqlite3.Connection or None if not found.
    """
    nmdb_path = run_dir / "world.nmdb"
    if not nmdb_path.exists():
        # --workspace creates nested subdirs; search recursively
        candidates = list(run_dir.rglob("world.nmdb"))
        if candidates:
            nmdb_path = candidates[0]
        else:
            return None
    try:
        from saas_bench.db_protection import load_session_db
        return load_session_db(nmdb_path)
    except Exception:
        return None

# Run registry
RUN_REGISTRY = {
    "d759b619": {"label": "GPT-5.4 xhigh v3.3d", "model": "gpt-5.4", "seed": 42, "days": 500},
    "d66cfedf": {"label": "GLM-5.1-FP8 v3.3d", "model": "GLM-5.1-FP8", "seed": 42, "days": 500},
}


def get_run_ids():
    if not RUNS_DIR.exists():
        return []
    dirs = sorted(RUNS_DIR.iterdir())
    ids = [d.name.replace("run_", "") for d in dirs if d.is_dir() and d.name.startswith("run_")]
    registry_order = list(RUN_REGISTRY.keys())
    known = [r for r in registry_order if r in ids]
    unknown = [r for r in ids if r not in registry_order]
    return known + unknown


def get_founder_dividends_from_db(run_dir: Path) -> float:
    """Quick SQLite query for cumulative founder dividends. Returns 0 if DB locked."""
    conn = _open_run_db(run_dir)
    if not conn:
        return 0
    try:
        row = conn.execute("SELECT COALESCE(SUM(founder_payout), 0) FROM dividends").fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def get_dividend_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Cumulative founder dividends by day. Returns list of {day, dividends}."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT day, founder_payout FROM dividends ORDER BY day"
        ).fetchall()
        conn.close()
        if not rows:
            return []
        # Build cumulative series
        series = []
        cumulative = 0.0
        for day, payout in rows:
            cumulative += payout
            series.append({"day": day, "dividends": round(cumulative, 2)})
        # Downsample if too many points
        if len(series) > max_points:
            step = len(series) // max_points
            series = [s for i, s in enumerate(series) if i % step == 0 or i == len(series) - 1]
        return series
    except Exception:
        return []


def get_profit_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Monthly (30-day rolling) profit series. Returns list of {day, revenue, costs, profit}."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        # Get all distinct days from ledger
        max_day_row = conn.execute("SELECT MAX(day) FROM ledger").fetchone()
        if not max_day_row or max_day_row[0] is None:
            conn.close()
            return []
        max_day = max_day_row[0]
        series = []
        # Rolling 30-day windows sampled every 7 days for smoother chart
        day = 29  # first full 30-day window ends at day 29 (days 0-29)
        while day <= max_day:
            start_day = day - 29
            row = conn.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN amount > 0 AND category != 'initial_funding' THEN amount ELSE 0 END), 0) as revenue,
                    COALESCE(SUM(CASE WHEN amount < 0 THEN amount ELSE 0 END), 0) as costs
                FROM ledger WHERE day BETWEEN ? AND ?
            """, (start_day, day)).fetchone()
            revenue = round(row[0], 2)
            costs = round(row[1], 2)
            series.append({"day": day, "revenue": revenue, "costs": costs, "profit": round(revenue + costs, 2)})
            day += 7
        conn.close()
        if len(series) > max_points:
            step = len(series) // max_points
            series = [s for i, s in enumerate(series) if i % step == 0 or i == len(series) - 1]
        return series
    except Exception:
        return []


def get_ads_revenue_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Daily ads revenue per active customer group. Returns list of {day, group_id, revenue}."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT day, group_id, SUM(revenue) as total_revenue "
            "FROM ads_revenue GROUP BY day, group_id ORDER BY day, group_id"
        ).fetchall()
        conn.close()
        if not rows:
            return []
        series = [{"day": r[0], "group_id": r[1], "revenue": round(r[2], 2)} for r in rows]
        unique_days = sorted(set(r[0] for r in rows))
        if len(unique_days) > max_points:
            step = len(unique_days) // max_points
            keep_days = set(d for i, d in enumerate(unique_days) if i % step == 0 or i == len(unique_days) - 1)
            series = [s for s in series if s["day"] in keep_days]
        return series
    except Exception:
        return []


def get_reputation_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Daily reputation per group from hidden snapshot table. Returns list of {day, group_id, reputation}."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT day, group_id, reputation FROM _hidden_group_params_history ORDER BY day, group_id"
        ).fetchall()
        conn.close()
        if not rows:
            return []
        series = [{"day": r[0], "group_id": r[1], "reputation": round(r[2], 6)} for r in rows]
        # Downsample if too many points (per-group, so total rows = days × groups)
        unique_days = sorted(set(r[0] for r in rows))
        if len(unique_days) > max_points:
            step = len(unique_days) // max_points
            keep_days = set(d for i, d in enumerate(unique_days) if i % step == 0 or i == len(unique_days) - 1)
            series = [s for s in series if s["day"] in keep_days]
        return series
    except Exception:
        return []


def get_quality_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Quality per group × plan over time from _hidden_quality_snapshot."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT day, group_id, plan, delivered_quality FROM _hidden_quality_snapshot ORDER BY day, group_id, plan"
        ).fetchall()
        conn.close()
        if not rows:
            return []
        series = [{"day": r[0], "group_id": r[1], "plan": r[2], "quality": round(r[3], 4)} for r in rows]
        unique_days = sorted(set(r[0] for r in rows))
        if len(unique_days) > max_points:
            step = len(unique_days) // max_points
            keep_days = set(d for i, d in enumerate(unique_days) if i % step == 0 or i == len(unique_days) - 1)
            series = [s for s in series if s["day"] in keep_days]
        return series
    except Exception:
        return []


def get_qmin_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Effective q_min per group over time from _hidden_group_params_history.

    Supports both old schema (current_q_min_mean column) and new accumulator
    schema (drift_q_bias_total + global_q_bias_total applied to static base).
    """
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        # Detect schema: check which columns exist
        cols = {r[1] for r in conn.execute("PRAGMA table_info(_hidden_group_params_history)").fetchall()}
        use_accumulators = 'drift_q_bias_total' in cols

        if use_accumulators:
            from saas_bench.config import CUSTOMER_GROUPS
            base_qmin = {gid: g.q_min_mean for gid, g in CUSTOMER_GROUPS.items()}
            rows = conn.execute(
                "SELECT day, group_id, drift_q_bias_total, global_q_bias_total "
                "FROM _hidden_group_params_history ORDER BY day, group_id"
            ).fetchall()
            conn.close()
            if not rows:
                return []
            series = []
            for r in rows:
                day, group_id, drift_q_bias, global_q_bias = r
                base = base_qmin.get(group_id, 0.5)
                effective_qmin = base + global_q_bias + drift_q_bias
                series.append({"day": day, "group_id": group_id, "q_min": round(effective_qmin, 4)})
        else:
            # Old schema: current_q_min_mean is already the effective value
            rows = conn.execute(
                "SELECT day, group_id, current_q_min_mean "
                "FROM _hidden_group_params_history ORDER BY day, group_id"
            ).fetchall()
            conn.close()
            if not rows:
                return []
            series = [{"day": r[0], "group_id": r[1], "q_min": round(r[2], 4)} for r in rows]

        unique_days = sorted(set(s["day"] for s in series))
        if len(unique_days) > max_points:
            step = len(unique_days) // max_points
            keep_days = set(d for i, d in enumerate(unique_days) if i % step == 0 or i == len(unique_days) - 1)
            series = [s for s in series if s["day"] in keep_days]
        return series
    except Exception:
        return []


def get_discovered_group_ids(run_dir: Path) -> set:
    """Return set of discovered group_ids (info_level >= 1)."""
    conn = _open_run_db(run_dir)
    if not conn:
        return set()
    try:
        rows = conn.execute(
            "SELECT group_id FROM group_info_levels WHERE info_level >= 1"
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def get_seat_series_from_db(run_dir: Path, max_points: int = 200) -> list:
    """Individual subs + enterprise seats per day from _hidden_group_params_history days."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        # Get all days from group_params_history as reference
        days = conn.execute(
            "SELECT DISTINCT day FROM _hidden_group_params_history ORDER BY day"
        ).fetchall()
        if not days:
            conn.close()
            return []
        series = []
        for (day,) in days:
            row = conn.execute(
                """SELECT
                    COALESCE(SUM(CASE WHEN seat_count = 1 THEN 1 ELSE 0 END), 0) as individual,
                    COALESCE(SUM(CASE WHEN seat_count > 1 THEN seat_count ELSE 0 END), 0) as enterprise_seats
                FROM subscriptions
                WHERE status IN ('subscribed', 'cancelled') AND start_day <= ? AND (end_day IS NULL OR end_day > ?)""",
                (day, day)
            ).fetchone()
            series.append({"day": day, "individual": row[0], "enterprise_seats": row[1]})
        conn.close()
        if len(series) > max_points:
            step = len(series) // max_points
            series = [s for i, s in enumerate(series) if i % step == 0 or i == len(series) - 1]
        return series
    except Exception:
        return []


def get_group_discovery_from_db(run_dir: Path) -> list:
    """Group discovery status from group_info_levels."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT group_id, info_level, is_discoverable, discovered_day FROM group_info_levels ORDER BY group_id"
        ).fetchall()
        conn.close()
        return [{"group_id": r[0], "info_level": r[1], "is_discoverable": r[2], "discovered_day": r[3]} for r in rows]
    except Exception:
        return []


def get_customer_social_posts_from_db(run_dir: Path, limit: int = 50) -> list:
    """Last N customer social media posts."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        # Check if source_group_id column exists (added in v3.2k)
        cols = {c[1] for c in conn.execute("PRAGMA table_info(social_media_posts)").fetchall()}
        has_source_gid = 'source_group_id' in cols
        if has_source_gid:
            group_expr = "COALESCE(p.source_group_id, c.group_id)"
        else:
            group_expr = "c.group_id"
        rows = conn.execute(
            f"""SELECT p.post_id, p.day, p.customer_id,
                      {group_expr} AS group_id,
                      p.sentiment, p.content,
                      p.likes, p.shares, p.reply_to_agent_post_id
               FROM social_media_posts p
               LEFT JOIN customers c ON p.customer_id = c.customer_id
               ORDER BY p.post_id DESC LIMIT ?""", (limit,)
        ).fetchall()
        conn.close()
        def _to_int(v):
            """Convert bytes/numpy types to plain int for JSON serialization."""
            if isinstance(v, bytes):
                return int.from_bytes(v, 'little') if v else 0
            if v is None:
                return 0
            return int(v)
        return [{"post_id": r[0], "day": r[1], "customer_id": r[2], "group_id": r[3],
                 "sentiment": r[4], "content": r[5], "likes": _to_int(r[6]), "shares": _to_int(r[7]),
                 "reply_to_agent_post_id": r[8]} for r in rows]
    except Exception:
        return []


def get_agent_social_posts_from_db(run_dir: Path, limit: int = 50) -> list:
    """Last N agent social media posts with scores, views, and customer replies."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        # Check column availability
        col_names = [c[1] for c in conn.execute("PRAGMA table_info(agent_social_media_posts)").fetchall()]
        has_reasoning = 'reasoning_by_group' in col_names
        smp_cols = {c[1] for c in conn.execute("PRAGMA table_info(social_media_posts)").fetchall()}
        reply_group_expr = "COALESCE(s.source_group_id, c.group_id)" if 'source_group_id' in smp_cols else "c.group_id"
        if has_reasoning:
            posts = conn.execute(
                """SELECT agent_post_id, day, content, reply_to_post_id,
                          effect_by_group, views, views_by_group, reasoning_by_group
                   FROM agent_social_media_posts ORDER BY agent_post_id DESC LIMIT ?""", (limit,)
            ).fetchall()
        else:
            posts = conn.execute(
                """SELECT agent_post_id, day, content, reply_to_post_id,
                          effect_by_group, views, views_by_group
                   FROM agent_social_media_posts ORDER BY agent_post_id DESC LIMIT ?""", (limit,)
            ).fetchall()
        result = []
        for p in posts:
            post_id = p[0]
            effects = {}
            views_by_group = {}
            reasoning = {}
            try:
                effects = json.loads(p[4]) if p[4] else {}
            except Exception:
                pass
            try:
                views_by_group = json.loads(p[6]) if p[6] else {}
            except Exception:
                pass
            if has_reasoning:
                try:
                    reasoning = json.loads(p[7]) if p[7] else {}
                except Exception:
                    pass
            # Get customer replies to this agent post
            replies = conn.execute(
                f"""SELECT s.post_id, s.day, s.customer_id,
                          {reply_group_expr} AS group_id,
                          s.sentiment, s.content
                   FROM social_media_posts s
                   LEFT JOIN customers c ON s.customer_id = c.customer_id
                   WHERE s.reply_to_agent_post_id = ?
                   ORDER BY s.post_id""", (post_id,)
            ).fetchall()
            reply_list = [{"post_id": r[0], "day": r[1], "customer_id": r[2], "group_id": r[3],
                           "sentiment": r[4], "content": r[5]} for r in replies]
            mults = {}  # Per-post multiplier removed; overall next-day multiplier is at run level
            result.append({
                "agent_post_id": post_id, "day": p[1], "content": p[2],
                "reply_to_post_id": p[3], "effect_by_group": effects,
                "views": p[5], "views_by_group": views_by_group,
                "reasoning_by_group": reasoning,
                "replies": reply_list, "multipliers": mults,
            })
        conn.close()
        return result
    except Exception:
        return []


def _brief_args(args):
    """Short preview of tool arguments."""
    if not args:
        return ""
    if isinstance(args, str):
        return args[:80]
    if isinstance(args, dict):
        if "command" in args:
            return str(args["command"])[:80]
        if "path" in args:
            return str(args["path"])[:80]
        if "code" in args:
            return str(args["code"])[:80]
    try:
        s = json.dumps(args)
        return s[:80]
    except Exception:
        return ""


def _resolve_inner_run_dir(run_dir: Path) -> Path:
    """Resolve the actual inner run directory created by --workspace.

    --workspace creates: run_<outer>/run_<inner>/config.json, logs/, agent_workspace/...
    Returns the inner dir if it exists, otherwise the original run_dir.
    """
    # Look for config.json recursively — it's always at the inner run root
    candidates = list(run_dir.glob("run_*/config.json"))
    if candidates:
        return candidates[0].parent
    return run_dir


def get_run_data(run_id: str) -> dict:
    run_dir = RUNS_DIR / f"run_{run_id}"
    inner_dir = _resolve_inner_run_dir(run_dir)
    reg = RUN_REGISTRY.get(run_id, {})
    data = {
        "run_id": run_id,
        "label": reg.get("label", f"run_{run_id}"),
        "model": reg.get("model", "unknown"),
        "seed": reg.get("seed"),
        "total_days": reg.get("days"),
    }

    # Last heartbeat: newest file mtime in the run directory
    try:
        newest_mtime = max(
            f.stat().st_mtime
            for f in run_dir.rglob("*")
            if f.is_file()
        )
        data["last_heartbeat"] = datetime.fromtimestamp(
            newest_mtime, tz=__import__('datetime').timezone.utc
        ).isoformat()
    except (ValueError, OSError):
        data["last_heartbeat"] = None

    # Config
    config_path = inner_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            data["model"] = cfg.get("model", data["model"])
            data["seed"] = cfg.get("seed", data["seed"])
            if data["total_days"] is None:
                data["total_days"] = cfg.get("total_days")

    # Checkpoint
    cp_path = inner_dir / "checkpoint.json"
    if cp_path.exists():
        try:
            with open(cp_path) as f:
                cp = json.load(f)
                data["current_day"] = cp.get("day", cp.get("current_day"))
                data["agent_turns"] = cp.get("agent_total_turns")
                data["total_input_tokens"] = cp.get("total_input_tokens", 0)
                data["total_output_tokens"] = cp.get("total_output_tokens", 0)
                data["total_cached_tokens"] = cp.get("total_cached_tokens", 0)
                data["total_reasoning_tokens"] = cp.get("total_reasoning_tokens", 0)
        except (json.JSONDecodeError, ValueError):
            data["current_day"] = None
            data["agent_turns"] = None
    else:
        data["current_day"] = None
        data["agent_turns"] = None

    # Stats: try JSONL run log first, fall back to DB
    # Search for run_*.jsonl recursively (--workspace nests it deeply)
    run_jsonl = inner_dir / "logs" / f"run_{run_id}.jsonl"
    if not run_jsonl.exists():
        jsonl_candidates = list(run_dir.rglob("run_*.jsonl"))
        if jsonl_candidates:
            run_jsonl = jsonl_candidates[0]
    got_stats_from_jsonl = False
    if run_jsonl.exists():
        try:
            snapshots = []
            with open(run_jsonl) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("category") == "daily_snapshot":
                            d = entry.get("details", {})
                            d["day"] = entry.get("day")
                            snapshots.append(d)
                    except json.JSONDecodeError:
                        continue

            if snapshots:
                got_stats_from_jsonl = True
                latest = snapshots[-1]
                data["cash"] = latest.get("cash", 0)
                data["subscribers"] = latest.get("subscribers", 0)
                data["mrr"] = latest.get("mrr", 0)

                # Derive current_day from latest snapshot if checkpoint missing
                if data.get("current_day") is None and latest.get("day") is not None:
                    data["current_day"] = latest["day"]

                step = max(1, len(snapshots) // 200)
                data["cash_series"] = [
                    {"day": s["day"], "cash": round(s.get("cash", 0), 2)}
                    for i, s in enumerate(snapshots)
                    if i % step == 0 or i == len(snapshots) - 1
                ]
                data["sub_series"] = [
                    {"day": s["day"], "subscribers": s.get("subscribers", 0)}
                    for i, s in enumerate(snapshots)
                    if i % step == 0 or i == len(snapshots) - 1
                ]
        except Exception as e:
            data["db_error"] = str(e)

    # Fallback: get cash/subs/MRR from DB when JSONL not available
    if not got_stats_from_jsonl:
        conn = _open_run_db(run_dir)
        if conn:
            # Each query in its own try/except so one failure doesn't block others
            try:
                row = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM ledger").fetchone()
                data["cash"] = round(row[0], 2) if row else 0
            except Exception as e:
                data.setdefault("db_errors", []).append(f"cash: {e}")

            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM subscriptions WHERE status='subscribed' AND end_day IS NULL"
                ).fetchone()
                data["subscribers"] = row[0] if row else 0
            except Exception as e:
                data.setdefault("db_errors", []).append(f"subscribers: {e}")

            try:
                row = conn.execute("""
                    SELECT COALESCE(SUM(s.effective_price * s.seat_count), 0)
                    FROM subscriptions s
                    WHERE s.status='subscribed' AND s.end_day IS NULL
                """).fetchone()
                data["mrr"] = round(row[0], 2) if row else 0
            except Exception as e:
                data.setdefault("db_errors", []).append(f"mrr: {e}")

            try:
                rows = conn.execute("""
                    SELECT day, SUM(amount) as daily_total
                    FROM ledger GROUP BY day ORDER BY day
                """).fetchall()
                if rows:
                    cash_series = []
                    cumulative = 0.0
                    for day, daily_total in rows:
                        cumulative += daily_total
                        cash_series.append({"day": day, "cash": round(cumulative, 2)})
                    if len(cash_series) > 200:
                        step = len(cash_series) // 200
                        cash_series = [s for i, s in enumerate(cash_series) if i % step == 0 or i == len(cash_series) - 1]
                    data["cash_series"] = cash_series
            except Exception as e:
                data.setdefault("db_errors", []).append(f"cash_series: {e}")

            try:
                hist_days = conn.execute(
                    "SELECT DISTINCT day FROM _hidden_group_params_history ORDER BY day"
                ).fetchall()
                sub_series = []
                for (day,) in hist_days:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM subscriptions WHERE status='subscribed' AND start_day <= ? AND (end_day IS NULL OR end_day > ?)",
                        (day, day)
                    ).fetchone()
                    sub_series.append({"day": day, "subscribers": row[0]})
                if len(sub_series) > 200:
                    step = len(sub_series) // 200
                    sub_series = [s for i, s in enumerate(sub_series) if i % step == 0 or i == len(sub_series) - 1]
                data["sub_series"] = sub_series
            except Exception as e:
                data.setdefault("db_errors", []).append(f"sub_series: {e}")

            # Derive current_day from DB if still missing
            if data.get("current_day") is None:
                try:
                    row = conn.execute("SELECT MAX(day) FROM _hidden_group_params_history").fetchone()
                    if row and row[0] is not None:
                        data["current_day"] = row[0]
                except Exception:
                    pass

            conn.close()

    # Always derive current_day from the actual simulation DB (most accurate).
    # checkpoint.json "day" may be in week-units if using step_week(), while
    # total_days and series data are in actual simulation days.
    db_conn = _open_run_db(run_dir)
    if db_conn:
        try:
            row = db_conn.execute("SELECT MAX(day) FROM _hidden_group_params_history").fetchone()
            if row and row[0] is not None:
                data["current_day"] = row[0]
        except Exception:
            pass
        try:
            db_conn.close()
        except Exception:
            pass

    # Founder dividends from SQLite DB (small table, quick query)
    data["founder_dividends"] = get_founder_dividends_from_db(run_dir)
    data["dividend_series"] = get_dividend_series_from_db(run_dir)

    # Monthly profit series (30-day rolling windows)
    profit_series = get_profit_series_from_db(run_dir)
    data["profit_series"] = profit_series
    if profit_series:
        data["monthly_profit"] = profit_series[-1]["profit"]
    else:
        data["monthly_profit"] = None

    # Discovered groups — used to filter charts
    discovered = get_discovered_group_ids(run_dir)

    # Per-group reputation timeseries (discovered only)
    data["reputation_series"] = [s for s in get_reputation_series_from_db(run_dir) if s["group_id"] in discovered]

    # Quality per group/plan over time (discovered only)
    data["quality_series"] = [s for s in get_quality_series_from_db(run_dir) if s["group_id"] in discovered]

    # Q_min per group over time (discovered only)
    data["qmin_series"] = [s for s in get_qmin_series_from_db(run_dir) if s["group_id"] in discovered]

    # Ads revenue per group over time (discovered only)
    data["ads_revenue_series"] = [s for s in get_ads_revenue_series_from_db(run_dir) if s["group_id"] in discovered]

    # Seat series (individual + enterprise)
    data["seat_series"] = get_seat_series_from_db(run_dir)

    # Group discovery status
    data["group_discovery"] = get_group_discovery_from_db(run_dir)

    # Customer social media posts (last 50)
    data["customer_social_posts"] = get_customer_social_posts_from_db(run_dir)

    # Agent social media posts with scores, views, multipliers, replies (last 50)
    data["agent_social_posts"] = get_agent_social_posts_from_db(run_dir)

    # Next-day overall lead multiplier per group (from social media effects)
    try:
        from saas_bench.database import compute_social_media_multiplier
        sm_conn = _open_run_db(run_dir)
        if sm_conn and data.get("current_day"):
            next_day = data["current_day"] + 1
            next_day_mults = {}
            for gid in discovered:
                next_day_mults[gid] = round(compute_social_media_multiplier(sm_conn, next_day, gid), 4)
            data["next_day_social_multiplier"] = next_day_mults
            sm_conn.close()
    except Exception:
        pass

    # Recent actions (last 100)
    inner_id = inner_dir.name.replace("run_", "") if inner_dir != run_dir else run_id
    tr_path = inner_dir / "logs" / f"tool_results_{inner_id}.jsonl"
    actions = []
    if tr_path.exists():
        with open(tr_path) as f:
            for line in f:
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        data["tool_calls_count"] = len(actions)
        # Keep last 100
        actions = actions[-100:]
        actions.reverse()
    data["recent_actions"] = actions

    # Daily rationales (extract from tool_results — log_rationale calls)
    import re
    rationales = []
    if tr_path.exists():
        # First pass: collect write_file entries for script-based rationales
        script_contents = {}  # filename -> content
        entries = []
        with open(tr_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    if entry.get("tool") == "write_file":
                        path = entry.get("arguments", {}).get("path", "")
                        if "log_rationale" in path and path.endswith(".py"):
                            fname = path.rsplit("/", 1)[-1] if "/" in path else path
                            script_contents[fname] = entry.get("arguments", {}).get("content", "")
                except (json.JSONDecodeError, Exception):
                    continue
        # Second pass: extract rationales
        for entry in entries:
            try:
                tool = entry.get("tool", "")
                cmd = entry.get("arguments", {}).get("command", "")
                result = str(entry.get("result", ""))
                if tool != "bash" or "log_rationale" not in cmd:
                    continue
                text = ""
                # Case 1: inline log_rationale('''...''') or similar
                for pattern in [r"log_rationale\('''(.*?)'''\)", r'log_rationale\("""(.*?)"""\)', r"log_rationale\('(.*?)'\)", r'log_rationale\("(.*?)"\)']:
                    m = re.search(pattern, cmd, re.DOTALL)
                    if m:
                        text = m.group(1)
                        break
                # Case 2: log_rationale(\n    rationale='...' or keyword arg
                if not text:
                    m = re.search(r"log_rationale\(\s*(?:rationale=)?['\"](.+?)['\"]", cmd, re.DOTALL)
                    if m:
                        text = m.group(1)
                # Case 3: script-based (e.g. ./novamind-operation python log_rationale_d46.py)
                if not text:
                    m = re.search(r"(log_rationale\w*\.py)", cmd)
                    if m:
                        fname = m.group(1)
                        script = script_contents.get(fname, "")
                        if script:
                            # Extract from script content
                            for sp in [r"log_rationale\('''(.*?)'''\)", r'log_rationale\("""(.*?)"""\)', r"log_rationale\('(.*?)'\)", r'log_rationale\("(.*?)"\)']:
                                sm = re.search(sp, script, re.DOTALL)
                                if sm:
                                    text = sm.group(1)
                                    break
                            if not text:
                                sm = re.search(r"log_rationale\(\s*(?:rationale=)?['\"](.+?)['\"]", script, re.DOTALL)
                                if sm:
                                    text = sm.group(1)
                # Only include if we got real text (not raw commands)
                if text and not text.startswith("./novamind"):
                    rationales.append({
                        "day": entry.get("day"),
                        "turn": entry.get("turn"),
                        "timestamp": entry.get("timestamp"),
                        "text": text[:3000],
                    })
            except Exception:
                continue
    data["daily_rationales"] = rationales

    # Recent raw responses (last 30)
    rr_path = inner_dir / "logs" / f"raw_responses_{inner_id}.jsonl"
    responses = []
    if rr_path.exists():
        with open(rr_path) as f:
            for line in f:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        responses = responses[-30:]
        responses.reverse()
    data["recent_responses"] = responses

    # Timing data (from timing_<run_id>.jsonl)
    timing_path = inner_dir / "logs" / f"timing_{inner_id}.jsonl"
    recent_turns = []
    if timing_path.exists():
        day_summaries = []
        try:
            with open(timing_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("event") == "day_summary":
                            day_summaries.append(entry)
                        elif entry.get("event") in ("llm_call", "tool_exec"):
                            recent_turns.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        # All day summaries for charts
        data["timing_day_summaries"] = day_summaries
        # Recent turns (last 50) for the timing log
        data["timing_recent_turns"] = recent_turns[-50:][::-1]
        # Cumulative timing stats
        if day_summaries:
            data["timing_total_llm"] = sum(d.get("llm_total_s", 0) for d in day_summaries)
            data["timing_total_step"] = sum(d.get("step_day_s", 0) for d in day_summaries)
            data["timing_total_tool"] = sum(d.get("tool_total_s", 0) for d in day_summaries)
            data["timing_avg_day"] = round(
                sum(d.get("elapsed_s", 0) for d in day_summaries) / len(day_summaries), 1
            )

    # Build unified recent_activity: merge tool_results + timing llm_calls
    # This ensures LLM thinking turns show up in the dashboard too
    activity = []
    for a in (actions or []):
        activity.append({
            "type": "tool",
            "tool": a.get("tool", "?"),
            "day": a.get("day"),
            "turn": a.get("turn"),
            "timestamp": a.get("timestamp"),
            "preview": _brief_args(a.get("arguments")),
        })
    for t in recent_turns[-100:]:
        if t.get("event") == "llm_call":
            activity.append({
                "type": "llm",
                "tool": t.get("tool", ""),
                "day": t.get("day"),
                "turn": t.get("turn"),
                "timestamp": t.get("timestamp"),
                "elapsed_s": t.get("elapsed_s"),
                "preview": (t.get("tool_preview") or "")[:80],
            })
    # Sort by timestamp descending, keep last 10
    activity.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    data["recent_activity"] = activity[:10]

    return data


def push_data():
    """Collect all run data and write to JSON file."""
    run_ids = get_run_ids()
    all_data = {
        "timestamp": datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
        "runs": [get_run_data(rid) for rid in run_ids],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f)
    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pushed {len(run_ids)} runs ({size_mb:.1f} MB) to {OUTPUT_FILE}")

    # Upload to Modal volume
    try:
        result = subprocess.run(
            ["modal", "volume", "put", MODAL_VOLUME, str(OUTPUT_FILE), "/data.json", "--force"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  → Uploaded to Modal volume {MODAL_VOLUME}")
        else:
            # Volume might not exist yet, create it
            if "not found" in result.stderr.lower():
                subprocess.run(["modal", "volume", "create", MODAL_VOLUME], capture_output=True, text=True)
                subprocess.run(
                    ["modal", "volume", "put", MODAL_VOLUME, str(OUTPUT_FILE), "/data.json", "--force"],
                    capture_output=True, text=True, timeout=30,
                )
                print(f"  → Created volume and uploaded")
            else:
                print(f"  → Upload failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"  → Upload error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0 = one-shot)")
    args = parser.parse_args()

    if args.loop > 0:
        print(f"Pushing data every {args.loop}s. Ctrl+C to stop.")
        while True:
            try:
                push_data()
                time.sleep(args.loop)
            except KeyboardInterrupt:
                break
    else:
        push_data()


if __name__ == "__main__":
    main()
