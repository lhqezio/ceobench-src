"""Regression tests for CEOBench Arena runtime coordination."""

from __future__ import annotations

import json
import threading
import urllib.request

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.arena_runner import parse_arena_model_specs
from saas_bench.arena.coordinator import (
    ArenaCoordinatorHTTPServer,
    ArenaNextWeekCoordinator,
)


def test_weekly_dashboard_header_marks_bash_agent_day_advanced():
    agent = BashAgent(
        tool_descriptions=[],
        client=object(),
        model="test-model",
    )

    assert agent.check_day_advanced("=== Week 3 Dashboard (Day 21) ===\nCash: $1")
    assert agent.day_advanced
    assert agent.new_dashboard.startswith("=== Week 3 Dashboard (Day 21) ===")


def test_arena_model_specs_parse_provider_model_pairs():
    specs = parse_arena_model_specs(
        "anthropic:claude-sonnet-4-6,gpt-5",
        count=3,
        default_provider="openai",
        default_model="gpt-5-mini",
    )

    assert [(spec.provider, spec.model) for spec in specs] == [
        ("anthropic", "claude-sonnet-4-6"),
        ("openai", "gpt-5"),
        ("openai", "gpt-5-mini"),
    ]


def test_next_week_coordinator_blocks_until_all_companies_submit():
    callback_calls = []

    def advance(submissions):
        callback_calls.append(set(submissions))
        return {
            company_id: {
                "success": True,
                "day": submission.day + 7,
                "dashboard": f"=== Week 1 Dashboard (Day {submission.day + 7}) ===\n{company_id}",
            }
            for company_id, submission in submissions.items()
        }

    coordinator = ArenaNextWeekCoordinator(
        ["company_0", "company_1"],
        advance,
        wait_timeout_s=5,
    )
    server = ArenaCoordinatorHTTPServer(coordinator)
    server.start()
    try:
        results = {}

        def submit(company_id):
            payload = {
                "company_id": company_id,
                "api_port": 12345,
                "day": 0,
                "rationale": "test",
                "predictions": {
                    "cash_1wk": {"point": 1, "lower": 0, "upper": 2},
                    "cash_4wk": {"point": 1, "lower": 0, "upper": 2},
                    "cash_12wk": {"point": 1, "lower": 0, "upper": 2},
                    "cash_26wk": {"point": 1, "lower": 0, "upper": 2},
                },
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/next-week",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                results[company_id] = json.loads(resp.read())

        threads = [
            threading.Thread(target=submit, args=("company_0",)),
            threading.Thread(target=submit, args=("company_1",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert callback_calls == [{"company_0", "company_1"}]
        assert results["company_0"]["success"]
        assert results["company_1"]["success"]
        assert "company_0" in results["company_0"]["dashboard"]
        assert "company_1" in results["company_1"]["dashboard"]
    finally:
        server.stop()
