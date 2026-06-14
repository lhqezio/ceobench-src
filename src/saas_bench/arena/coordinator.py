"""Runtime coordination for CEOBench Arena weekly barriers."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping


PredictionBody = Mapping[str, Mapping[str, float]]
AdvanceCallback = Callable[[Mapping[str, "ArenaNextWeekSubmission"]], Mapping[str, dict]]


@dataclass(frozen=True)
class ArenaNextWeekSubmission:
    """One company's arena next-week submission."""

    company_id: str
    api_port: int
    day: int
    rationale: str
    predictions: PredictionBody

    @property
    def next_week_body(self) -> dict:
        return {
            "rationale": self.rationale,
            "predictions": self.predictions,
        }


class ArenaNextWeekCoordinator:
    """Synchronize arena companies at the weekly next-week barrier."""

    def __init__(
        self,
        company_ids: list[str],
        advance_callback: AdvanceCallback,
        *,
        wait_timeout_s: float = 7200.0,
    ):
        if not company_ids:
            raise ValueError("Arena coordinator requires at least one company")
        if len(set(company_ids)) != len(company_ids):
            raise ValueError("Arena coordinator company_ids must be unique")

        self.company_ids = tuple(company_ids)
        self._company_id_set = set(company_ids)
        self._advance_callback = advance_callback
        self._wait_timeout_s = wait_timeout_s
        self._condition = threading.Condition()
        self._submissions_by_day: dict[int, dict[str, ArenaNextWeekSubmission]] = {}
        self._results_by_day: dict[int, dict[str, dict]] = {}
        self._advancing_days: set[int] = set()

    def submit(self, submission: ArenaNextWeekSubmission) -> dict:
        """Submit one company's week and block until the shared week advances."""

        if submission.company_id not in self._company_id_set:
            return {
                "success": False,
                "error": f"Unknown arena company_id: {submission.company_id}",
            }

        deadline = time.monotonic() + self._wait_timeout_s
        with self._condition:
            day_submissions = self._submissions_by_day.setdefault(submission.day, {})
            day_submissions[submission.company_id] = submission

            if self._all_submitted_locked(submission.day):
                self._advance_locked(submission.day)

            while submission.day not in self._results_by_day:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "success": False,
                        "error": "arena_next_week_timeout",
                        "message": "Timed out waiting for other arena companies to submit next-week.",
                    }
                self._condition.wait(timeout=remaining)

            return self._results_by_day[submission.day].get(
                submission.company_id,
                {
                    "success": False,
                    "error": "arena_missing_company_result",
                },
            )

    def _all_submitted_locked(self, day: int) -> bool:
        return set(self._submissions_by_day.get(day, {})) == self._company_id_set

    def _advance_locked(self, day: int) -> None:
        if day in self._results_by_day or day in self._advancing_days:
            return

        self._advancing_days.add(day)
        submissions = dict(self._submissions_by_day[day])
        self._condition.release()
        try:
            try:
                results = dict(self._advance_callback(submissions))
            except Exception as exc:
                results = {
                    company_id: {
                        "success": False,
                        "error": "arena_advance_failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                    for company_id in self.company_ids
                }
        finally:
            self._condition.acquire()

        self._results_by_day[day] = results
        self._advancing_days.discard(day)
        self._condition.notify_all()


class ArenaCoordinatorHTTPServer:
    """Small localhost HTTP server used by arena operation wrappers."""

    def __init__(
        self,
        coordinator: ArenaNextWeekCoordinator,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.coordinator = coordinator
        self.host = host
        self._server = _ArenaHTTPServer((host, port), _ArenaRequestHandler, coordinator)
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ceobench-arena-coordinator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class _ArenaHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, coordinator: ArenaNextWeekCoordinator):
        super().__init__(server_address, handler_class)
        self.coordinator = coordinator


class _ArenaRequestHandler(BaseHTTPRequestHandler):
    server: _ArenaHTTPServer

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"success": True})
            return
        self._send_json({"success": False, "error": "not_found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/next-week":
            self._send_json({"success": False, "error": "not_found"}, status=404)
            return

        body = self._read_json()
        try:
            submission = ArenaNextWeekSubmission(
                company_id=str(body["company_id"]),
                api_port=int(body["api_port"]),
                day=int(body["day"]),
                rationale=str(body["rationale"]),
                predictions=body["predictions"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(
                {
                    "success": False,
                    "error": "invalid_arena_submission",
                    "message": str(exc),
                },
                status=400,
            )
            return

        result = self.server.coordinator.submit(submission)
        self._send_json(result, status=200 if result.get("success") else 500)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def http_get_json(port: int, path: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post_json(
    port: int,
    path: str,
    body: Mapping,
    *,
    timeout: float = 7200.0,
) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except Exception:
            return {"success": False, "error": f"HTTP {exc.code}"}
