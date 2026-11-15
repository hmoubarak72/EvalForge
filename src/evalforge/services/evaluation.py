"""Stage 5: evaluation runner.

Runs the active dataset against a model endpoint with bounded concurrency,
captures raw outputs, latency, token usage, errors and retries, scores every
response and persists an immutable run with a structured report.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from evalforge import __version__
from evalforge.adapters.llm.base import ChatClient, ChatResult, LLMError
from evalforge.config import EvaluationSettings
from evalforge.db.base import utcnow
from evalforge.db.models import EvalTestCase, EvaluationResult, EvaluationRun
from evalforge.domain.enums import RunStatus
from evalforge.domain.errors import InvalidRequestError
from evalforge.domain.schemas import EndpointConfig
from evalforge.repositories import AuditRepository, DatasetRepository
from evalforge.services.dataset import labels_from_row
from evalforge.services.regression import RegressionService
from evalforge.services.scoring import Scorer, ScoreResult
from evalforge.telemetry import bind_correlation_id, get_logger

log = get_logger(__name__)

MAX_FAILED_CASES_IN_REPORT = 200


def code_version() -> str:
    """Code version recorded on every run (override with EVALFORGE_CODE_VERSION)."""
    return os.environ.get("EVALFORGE_CODE_VERSION", __version__)


@dataclass(frozen=True)
class ModelOutput:
    """Raw outcome of calling the endpoint for one test case."""

    text: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    error: str | None = None


def _group_summary(rows: list[tuple[bool, float]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for ok, _ in rows if ok)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_quality_score": round(sum(s for _, s in rows) / total, 3) if total else 0.0,
    }


def build_report(
    scored: list[tuple[EvalTestCase, ModelOutput, ScoreResult]],
    settings: EvaluationSettings,
    runtime_s: float,
) -> dict[str, Any]:
    """Assemble the structured evaluation report."""
    by_category: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    by_difficulty: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    failed_cases: list[dict[str, Any]] = []
    for case, output, result in scored:
        row = (result.passed, result.quality_score)
        by_category[case.category].append(row)
        by_difficulty[case.difficulty].append(row)
        for name, value in result.metric_scores.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                metric_values[name].append(float(value))
        if not result.passed:
            failed_cases.append(
                {
                    "test_case_id": case.id,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "prompt": case.user_prompt[:300],
                    "response": output.text[:500],
                    "quality_score": result.quality_score,
                    "failure_reasons": result.failure_reasons,
                    "error": output.error,
                }
            )
    input_tokens = sum(o.input_tokens for _, o, _ in scored)
    output_tokens = sum(o.output_tokens for _, o, _ in scored)
    latencies = sorted(o.latency_ms for _, o, _ in scored if not o.error)
    cost = (
        input_tokens / 1000.0 * settings.cost_per_1k_input_tokens
        + output_tokens / 1000.0 * settings.cost_per_1k_output_tokens
    )
    # Metrics on a 1-5 scale are reported as averages; 0-1 metrics as pass rates.
    by_metric = {}
    for name, values in sorted(metric_values.items()):
        mean = sum(values) / len(values)
        if max(values) > 1.0:
            by_metric[name] = {"average": round(mean, 3), "scale": "1-5", "cases": len(values)}
        else:
            by_metric[name] = {
                "average": round(mean, 4),
                "pass_rate": round(sum(1 for v in values if v >= 1.0) / len(values), 4),
                "scale": "0-1",
                "cases": len(values),
            }
    return {
        "overall": {
            **_group_summary([(r.passed, r.quality_score) for _, _, r in scored]),
            "errors": sum(1 for _, o, _ in scored if o.error),
        },
        "by_category": {k: _group_summary(v) for k, v in sorted(by_category.items())},
        "by_difficulty": {k: _group_summary(v) for k, v in sorted(by_difficulty.items())},
        "by_metric": by_metric,
        "failed_cases": failed_cases[:MAX_FAILED_CASES_IN_REPORT],
        "failed_cases_truncated": len(failed_cases) > MAX_FAILED_CASES_IN_REPORT,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "runtime_s": round(runtime_s, 3),
            "average_latency_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0,
            "retries": sum(o.retries for _, o, _ in scored),
        },
    }


class EvaluationService:
    """Runs evaluations and persists results; triggers regression comparison."""

    def __init__(
        self,
        settings: EvaluationSettings,
        scorer_factory: Callable[[str], Scorer],
        client_factory: Callable[[EndpointConfig], ChatClient],
        regression: RegressionService,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings = settings
        self._scorer_factory = scorer_factory
        self._client_factory = client_factory
        self._regression = regression
        self._clock = clock

    def default_endpoint(self) -> EndpointConfig:
        """Endpoint configuration derived from the application settings."""
        s = self._settings
        return EndpointConfig(
            provider=s.provider,
            base_url=s.base_url,
            model=s.model,
            api_key_env=s.api_key_env,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            timeout_s=s.timeout_s,
            max_retries=s.max_retries,
            retry_backoff_s=s.retry_backoff_s,
            concurrency=s.concurrency,
        )

    def run(
        self,
        session: Session,
        endpoint: EndpointConfig | None = None,
        *,
        name: str = "",
        grading_mode: str | None = None,
        comparison_group: str | None = None,
        baseline_run_id: str | None = None,
    ) -> EvaluationRun:
        """Blocking entry point (CLI, scheduler, sync API handlers)."""
        return asyncio.run(
            self.arun(
                session,
                endpoint,
                name=name,
                grading_mode=grading_mode,
                comparison_group=comparison_group,
                baseline_run_id=baseline_run_id,
            )
        )

    async def arun(
        self,
        session: Session,
        endpoint: EndpointConfig | None = None,
        *,
        name: str = "",
        grading_mode: str | None = None,
        comparison_group: str | None = None,
        baseline_run_id: str | None = None,
    ) -> EvaluationRun:
        """Evaluate the active dataset against an endpoint."""
        endpoint = endpoint or self.default_endpoint()
        mode = grading_mode or self._settings.grading_mode
        dataset = DatasetRepository(session)
        cases = [c for c in dataset.active_test_cases() if c.current_label is not None]
        if not cases:
            raise InvalidRequestError("the evaluation dataset has no active test cases")
        scorer = self._scorer_factory(mode)
        client = self._client_factory(endpoint)

        with bind_correlation_id(stage="evaluation") as correlation_id:
            run = EvaluationRun(
                name=name or f"{endpoint.model} @ v{dataset.current_version()}",
                status=RunStatus.RUNNING.value,
                comparison_group=comparison_group or self._settings.comparison_group,
                endpoint_config=endpoint.public_dict(),
                model_name=endpoint.model,
                grading_mode=mode,
                dataset_version=dataset.current_version(),
                dataset_size=len(cases),
                code_version=code_version(),
                correlation_id=correlation_id,
                started_at=self._clock(),
            )
            session.add(run)
            session.flush()
            log.info("evaluation.started", run_id=run.id, model=endpoint.model, cases=len(cases))
            started = time.perf_counter()
            try:
                outputs = await self._collect(client, endpoint, cases)
                scored: list[tuple[EvalTestCase, ModelOutput, ScoreResult]] = []
                for case, output in zip(cases, outputs, strict=True):
                    label = case.current_label
                    assert label is not None  # filtered above
                    labels = labels_from_row(label, case.category, case.difficulty)
                    result = await asyncio.to_thread(
                        scorer.score, case.user_prompt, output.text, labels, output.error
                    )
                    scored.append((case, output, result))
                    session.add(
                        EvaluationResult(
                            run_id=run.id,
                            test_case_id=case.id,
                            label_id=label.id,
                            category=case.category,
                            difficulty=case.difficulty,
                            raw_response=output.text,
                            latency_ms=output.latency_ms,
                            input_tokens=output.input_tokens,
                            output_tokens=output.output_tokens,
                            retries=output.retries,
                            error=output.error,
                            passed=result.passed,
                            quality_score=result.quality_score,
                            detected_behavior=result.detected_behavior,
                            metric_scores=result.metric_scores,
                            failure_reasons=result.failure_reasons,
                            created_at=self._clock(),
                        )
                    )
            except Exception as exc:
                run.status = RunStatus.FAILED.value
                run.error = f"{type(exc).__name__}: {exc}"
                run.finished_at = self._clock()
                session.flush()
                log.error("evaluation.failed", run_id=run.id, error=run.error)
                raise

            runtime = time.perf_counter() - started
            report = build_report(scored, self._settings, runtime)
            report["run"] = {
                "id": run.id,
                "model": endpoint.model,
                "provider": endpoint.provider,
                "dataset_version": run.dataset_version,
                "dataset_size": run.dataset_size,
                "code_version": run.code_version,
                "grading_mode": mode,
            }
            run.report = report
            run.pass_rate = report["overall"]["pass_rate"]
            run.average_quality_score = report["overall"]["average_quality_score"]
            run.total_input_tokens = report["usage"]["input_tokens"]
            run.total_output_tokens = report["usage"]["output_tokens"]
            run.estimated_cost_usd = report["usage"]["estimated_cost_usd"]
            run.runtime_s = report["usage"]["runtime_s"]
            run.status = RunStatus.COMPLETED.value
            run.finished_at = self._clock()
            AuditRepository(session).record(
                "evaluation.completed",
                "evaluation_run",
                run.id,
                payload={"model": endpoint.model, "pass_rate": run.pass_rate},
            )
            session.flush()
            self._regression.compare_run(session, run, baseline_run_id)
            log.info(
                "evaluation.finished",
                run_id=run.id,
                pass_rate=run.pass_rate,
                average_quality_score=run.average_quality_score,
                runtime_s=run.runtime_s,
            )
            return run

    async def _collect(
        self, client: ChatClient, endpoint: EndpointConfig, cases: list[EvalTestCase]
    ) -> list[ModelOutput]:
        semaphore = asyncio.Semaphore(endpoint.concurrency)

        async def call(case: EvalTestCase) -> ModelOutput:
            messages = []
            if case.system_prompt:
                messages.append({"role": "system", "content": case.system_prompt})
            messages.append({"role": "user", "content": case.user_prompt})
            async with semaphore:
                started = time.perf_counter()
                try:
                    result: ChatResult = await client.acomplete(
                        messages, temperature=endpoint.temperature, max_tokens=endpoint.max_tokens
                    )
                except LLMError as exc:
                    return ModelOutput(
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        retries=endpoint.max_retries,
                        error=str(exc),
                    )
            return ModelOutput(
                text=result.text,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                retries=result.retries,
            )

        return list(await asyncio.gather(*(call(case) for case in cases)))
