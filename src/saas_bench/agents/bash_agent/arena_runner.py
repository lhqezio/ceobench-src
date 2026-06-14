"""Arena runner for multiple bash agents sharing a weekly barrier."""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from saas_bench.arena import (
    ArenaCoordinatorHTTPServer,
    ArenaNextWeekCoordinator,
    ArenaNextWeekSubmission,
    make_company_specs,
)
from saas_bench.arena.coordinator import http_post_json
from saas_bench.config import BenchmarkConfig

from .run_test import BashAgentRunner


@dataclass(frozen=True)
class ArenaModelSpec:
    provider: str | None
    model: str | None


def parse_arena_model_specs(
    raw: str | None,
    *,
    count: int,
    default_provider: str | None,
    default_model: str | None,
) -> list[ArenaModelSpec]:
    """Parse provider:model arena model specs, repeating defaults as needed."""

    entries = [item.strip() for item in (raw or "").split(",") if item.strip()]
    specs: list[ArenaModelSpec] = []
    for entry in entries:
        if ":" in entry:
            provider, model = entry.split(":", 1)
            specs.append(ArenaModelSpec(provider.strip() or None, model.strip() or None))
        else:
            specs.append(ArenaModelSpec(default_provider, entry))

    while len(specs) < count:
        specs.append(ArenaModelSpec(default_provider, default_model))

    if len(specs) > count:
        raise ValueError("--arena-models cannot list more models than --arena-companies")

    return specs


class ArenaBashAgentRunner:
    """Run multiple ordinary bash-agent CEOBench companies in arena mode."""

    def __init__(
        self,
        *,
        company_count: int,
        arena_models: str | None = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: int = 42,
        scenario: str = "default",
        total_days: int = 3650,
        initial_cash: float = 1_000_000.0,
        workspace_base: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
        label: Optional[str] = None,
    ):
        if company_count < 2:
            raise ValueError("ArenaBashAgentRunner requires at least two companies")

        default_config = BenchmarkConfig()
        self.company_count = company_count
        self.default_model = model or default_config.agent_llm_model
        self.default_provider = provider or default_config.agent_llm_provider
        self.base_url = base_url
        self.api_key = api_key
        self.seed = seed
        self.scenario = scenario
        self.total_days = (total_days // 7) * 7
        self.initial_cash = initial_cash
        self.reasoning_effort = reasoning_effort or default_config.agent_llm_reasoning_effort
        self.label = label

        self.run_id = str(uuid.uuid4())[:8]
        base = (workspace_base or Path("./bash_agent_runs")).resolve()
        self.workspace_dir = base / f"arena_{self.run_id}"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._runners: dict[str, BashAgentRunner] = {}

        self.model_specs = parse_arena_model_specs(
            arena_models,
            count=company_count,
            default_provider=self.default_provider,
            default_model=self.default_model,
        )
        self.company_specs = make_company_specs(
            company_count,
            agent_models=[spec.model for spec in self.model_specs],
            starting_cash=initial_cash,
        )

    def run(self, verbose: bool = True) -> dict[str, Any]:
        coordinator = ArenaNextWeekCoordinator(
            [spec.company_id for spec in self.company_specs],
            self._advance_submitted_week,
        )
        server = ArenaCoordinatorHTTPServer(coordinator)
        server.start()

        if verbose:
            print(f"\n{'='*60}")
            print("Starting CEOBench Arena")
            print(f"Arena Run ID: {self.run_id}")
            print(f"Companies: {self.company_count}")
            print(f"Coordinator Port: {server.port}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        try:
            self._write_config(server.port)
            self._create_company_runners(server.port)
            results = self._run_company_threads(verbose=verbose)
        finally:
            server.stop()

        outcomes = {company_id: result.get("outcome") for company_id, result in results.items()}
        return {
            "run_id": self.run_id,
            "arena": True,
            "companies": results,
            "outcomes": outcomes,
            "workspace_dir": str(self.workspace_dir),
        }

    def _create_company_runners(self, coordinator_port: int) -> None:
        companies_root = self.workspace_dir / "companies"
        companies_root.mkdir(exist_ok=True)

        for index, spec in enumerate(self.company_specs):
            model_spec = self.model_specs[index]
            runner = BashAgentRunner(
                model=model_spec.model,
                provider=model_spec.provider,
                base_url=self.base_url,
                api_key=self.api_key,
                seed=self.seed + index,
                scenario=self.scenario,
                total_days=self.total_days,
                initial_cash=spec.starting_cash or self.initial_cash,
                workspace_base=companies_root / spec.company_id,
                reasoning_effort=self.reasoning_effort,
                label=self.label or f"arena:{spec.display_name}",
                arena_company_id=spec.company_id,
                arena_display_name=spec.display_name,
                arena_coordinator_port=coordinator_port,
                arena_company_count=self.company_count,
            )
            self._runners[spec.company_id] = runner

    def _run_company_threads(self, *, verbose: bool) -> dict[str, dict]:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.company_count) as executor:
            future_by_company = {
                executor.submit(runner.run, verbose=verbose): company_id
                for company_id, runner in self._runners.items()
            }
            for future in as_completed(future_by_company):
                company_id = future_by_company[future]
                results[company_id] = future.result()
        return results

    def _advance_submitted_week(
        self,
        submissions: dict[str, ArenaNextWeekSubmission],
    ) -> dict[str, dict]:
        """Advance all submitted companies through their ordinary CEOBench servers."""

        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=len(submissions)) as executor:
            future_by_company = {
                executor.submit(
                    http_post_json,
                    submission.api_port,
                    "/next-week",
                    submission.next_week_body,
                ): company_id
                for company_id, submission in submissions.items()
            }
            for future in as_completed(future_by_company):
                company_id = future_by_company[future]
                result = future.result()
                if result.get("success"):
                    results[company_id] = result
                else:
                    results[company_id] = {
                        "success": False,
                        "error": result.get("error", "arena_company_advance_failed"),
                        "message": result.get("message", ""),
                    }
        return results

    def _write_config(self, coordinator_port: int) -> None:
        config = {
            "run_id": self.run_id,
            "arena": True,
            "company_count": self.company_count,
            "companies": [
                {
                    "company_id": spec.company_id,
                    "display_name": spec.display_name,
                    "provider": self.model_specs[index].provider,
                    "model": self.model_specs[index].model,
                }
                for index, spec in enumerate(self.company_specs)
            ],
            "seed": self.seed,
            "scenario": self.scenario,
            "total_days": self.total_days,
            "initial_cash": self.initial_cash,
            "coordinator_port": coordinator_port,
            "public_dir_override": os.environ.get("NOVAMIND_PUBLIC_DIR") or None,
        }
        (self.workspace_dir / "config.json").write_text(json.dumps(config, indent=2))
