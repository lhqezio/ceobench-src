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
        return None
    try:
        from saas_bench.db_protection import load_session_db
        return load_session_db(nmdb_path)
    except Exception:
        return None

# Run registry
RUN_REGISTRY = {
    "5464038b": {"label": "GLM-5 v3.2 run 1", "model": "GLM-5-FP8", "seed": 42, "days": 1095},
    "5f35a596": {"label": "GLM-5 v3.2 run 2", "model": "GLM-5-FP8", "seed": 42, "days": 1095},
    "83e43a30": {"label": "GLM-5 v3.2 run 3", "model": "GLM-5-FP8", "seed": 42, "days": 1095},
    "9f984e69": {"label": "GLM-5 v3.2 run 4", "model": "GLM-5-FP8", "seed": 42, "days": 1095},
    "eea79b7a": {"label": "GLM-5 v3.2 run 5", "model": "GLM-5-FP8", "seed": 42, "days": 1095},
    "b2a6fd1f": {"label": "Sonnet 4.6 Bedrock run 1", "model": "claude-sonnet-4-6", "seed": 42, "days": 1095},
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
    """Q_min per group over time from _hidden_group_params_history."""
    conn = _open_run_db(run_dir)
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT day, group_id, current_q_min_mean FROM _hidden_group_params_history ORDER BY day, group_id"
        ).fetchall()
        conn.close()
        if not rows:
            return []
        series = [{"day": r[0], "group_id": r[1], "q_min": round(r[2], 4)} for r in rows]
        unique_days = sorted(set(r[0] for r in rows))
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
                WHERE status = 'subscribed' AND start_day <= ? AND (end_day IS NULL OR end_day > ?)""",
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
        rows = conn.execute(
            """SELECT p.post_id, p.day, p.customer_id, c.group_id, p.sentiment, p.content,
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
        # Check if reasoning_by_group column exists
        col_names = [c[1] for c in conn.execute("PRAGMA table_info(agent_social_media_posts)").fetchall()]
        has_reasoning = 'reasoning_by_group' in col_names
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
                """SELECT s.post_id, s.day, s.customer_id, c.group_id, s.sentiment, s.content
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


def get_run_data(run_id: str) -> dict:
    run_dir = RUNS_DIR / f"run_{run_id}"
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
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            data["model"] = cfg.get("model", data["model"])
            data["seed"] = cfg.get("seed", data["seed"])
            if data["total_days"] is None:
                data["total_days"] = cfg.get("total_days")

    # Checkpoint
    cp_path = run_dir / "checkpoint.json"
    if cp_path.exists():
        try:
            with open(cp_path) as f:
                cp = json.load(f)
                data["current_day"] = cp.get("day", cp.get("current_day"))
                data["agent_turns"] = cp.get("agent_total_turns")
        except (json.JSONDecodeError, ValueError):
            data["current_day"] = None
            data["agent_turns"] = None
    else:
        data["current_day"] = None
        data["agent_turns"] = None

    # Stats: try JSONL run log first, fall back to DB
    run_jsonl = run_dir / "logs" / f"run_{run_id}.jsonl"
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

            conn.close()

    # Founder dividends from SQLite DB (small table, quick query)
    data["founder_dividends"] = get_founder_dividends_from_db(run_dir)
    data["dividend_series"] = get_dividend_series_from_db(run_dir)

    # Discovered groups — used to filter charts
    discovered = get_discovered_group_ids(run_dir)

    # Per-group reputation timeseries (discovered only)
    data["reputation_series"] = [s for s in get_reputation_series_from_db(run_dir) if s["group_id"] in discovered]

    # Quality per group/plan over time (discovered only)
    data["quality_series"] = [s for s in get_quality_series_from_db(run_dir) if s["group_id"] in discovered]

    # Q_min per group over time (discovered only)
    data["qmin_series"] = [s for s in get_qmin_series_from_db(run_dir) if s["group_id"] in discovered]

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
    tr_path = run_dir / "logs" / f"tool_results_{run_id}.jsonl"
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

    # Recent raw responses (last 30)
    rr_path = run_dir / "logs" / f"raw_responses_{run_id}.jsonl"
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
    timing_path = run_dir / "logs" / f"timing_{run_id}.jsonl"
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
