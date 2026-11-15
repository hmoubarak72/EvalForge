"""Stage 6: regression tracking.

Compares an evaluation run with the previous comparable run (same comparison
group) on the test cases both runs share. Reports are stored once per run and
never overwritten.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from evalforge.config import RegressionSettings
from evalforge.db.models import EvaluationResult, EvaluationRun, RegressionReport
from evalforge.domain.errors import InvalidRequestError, NotFoundError
from evalforge.repositories import AuditRepository, RunRepository
from evalforge.telemetry import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CaseOutcome:
    """Minimal per-case result used for run comparison."""

    test_case_id: str
    passed: bool
    quality_score: float
    category: str
    difficulty: str

    @classmethod
    def from_result(cls, result: EvaluationResult) -> CaseOutcome:
        return cls(
            result.test_case_id,
            result.passed,
            result.quality_score,
            result.category,
            result.difficulty,
        )


def _group_shifts(
    current: Sequence[CaseOutcome],
    baseline: dict[str, CaseOutcome],
    attribute: str,
    threshold: float,
    min_group_size: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[CaseOutcome, CaseOutcome]]] = defaultdict(list)
    for outcome in current:
        groups[getattr(outcome, attribute)].append((baseline[outcome.test_case_id], outcome))
    shifts = []
    for name in sorted(groups):
        pairs = groups[name]
        if len(pairs) < min_group_size:
            continue
        before = sum(1 for old, _ in pairs if old.passed) / len(pairs)
        after = sum(1 for _, new in pairs if new.passed) / len(pairs)
        delta = after - before
        if abs(delta) >= threshold:
            shifts.append(
                {
                    attribute: name,
                    "cases": len(pairs),
                    "baseline_pass_rate": round(before, 4),
                    "current_pass_rate": round(after, 4),
                    "delta": round(delta, 4),
                    "direction": "regressed" if delta < 0 else "improved",
                }
            )
    return shifts


def compare_outcomes(
    current: Sequence[CaseOutcome],
    baseline: Sequence[CaseOutcome],
    settings: RegressionSettings,
) -> dict[str, Any]:
    """Pure comparison of two result sets; returns the regression report body."""
    baseline_by_id = {o.test_case_id: o for o in baseline}
    current_by_id = {o.test_case_id: o for o in current}
    shared = [o for o in current if o.test_case_id in baseline_by_id]

    new_failures, new_passes, score_changes = [], [], []
    for outcome in shared:
        old = baseline_by_id[outcome.test_case_id]
        entry = {
            "test_case_id": outcome.test_case_id,
            "category": outcome.category,
            "difficulty": outcome.difficulty,
            "baseline_score": old.quality_score,
            "current_score": outcome.quality_score,
        }
        if old.passed and not outcome.passed:
            new_failures.append(entry)
        elif not old.passed and outcome.passed:
            new_passes.append(entry)
        delta = outcome.quality_score - old.quality_score
        if abs(delta) >= settings.score_change_threshold:
            score_changes.append({**entry, "delta": round(delta, 3)})

    def pass_rate(outcomes: Sequence[CaseOutcome]) -> float:
        return sum(1 for o in outcomes if o.passed) / len(outcomes) if outcomes else 0.0

    def mean_score(outcomes: Sequence[CaseOutcome]) -> float:
        return sum(o.quality_score for o in outcomes) / len(outcomes) if outcomes else 0.0

    shared_baseline = [baseline_by_id[o.test_case_id] for o in shared]
    overall_delta = pass_rate(shared) - pass_rate(shared_baseline)
    category_shifts = _group_shifts(
        shared,
        baseline_by_id,
        "category",
        settings.category_shift_threshold,
        settings.min_group_size,
    )
    difficulty_shifts = _group_shifts(
        shared,
        baseline_by_id,
        "difficulty",
        settings.difficulty_shift_threshold,
        settings.min_group_size,
    )
    regressed_groups = [
        s for s in [*category_shifts, *difficulty_shifts] if s["direction"] == "regressed"
    ]
    major = overall_delta <= -settings.overall_pass_rate_drop_threshold or bool(regressed_groups)
    has_regression = bool(new_failures) or major
    severity = "major" if major else "minor" if has_regression else "none"

    summary_parts = [
        f"{len(shared)} shared cases",
        f"pass rate {pass_rate(shared_baseline):.1%} -> {pass_rate(shared):.1%} "
        f"({overall_delta:+.1%})",
        f"{len(new_failures)} new failure(s)",
        f"{len(new_passes)} newly passing",
    ]
    if regressed_groups:
        names = ", ".join(str(s.get("category") or s.get("difficulty")) for s in regressed_groups)
        summary_parts.append(f"regressed groups: {names}")

    return {
        "severity": severity,
        "has_regression": has_regression,
        "summary": "; ".join(summary_parts),
        "shared_cases": len(shared),
        "added_cases": len([o for o in current if o.test_case_id not in baseline_by_id]),
        "removed_cases": len([o for o in baseline if o.test_case_id not in current_by_id]),
        "overall": {
            "baseline_pass_rate": round(pass_rate(shared_baseline), 4),
            "current_pass_rate": round(pass_rate(shared), 4),
            "pass_rate_delta": round(overall_delta, 4),
            "baseline_average_score": round(mean_score(shared_baseline), 3),
            "current_average_score": round(mean_score(shared), 3),
        },
        "new_failures": new_failures,
        "new_passes": new_passes,
        "score_changes": sorted(score_changes, key=lambda e: e["delta"]),
        "category_shifts": category_shifts,
        "difficulty_shifts": difficulty_shifts,
        "thresholds": settings.model_dump(),
    }


class RegressionService:
    """Creates and retrieves regression reports."""

    def __init__(self, settings: RegressionSettings) -> None:
        self._settings = settings

    def compare_run(
        self, session: Session, run: EvaluationRun, baseline_run_id: str | None = None
    ) -> RegressionReport | None:
        """Compare a run with its baseline; returns None when no baseline exists.

        Idempotent: an existing report for the run is returned unchanged.
        """
        runs = RunRepository(session)
        existing = runs.regression_report_for(run.id)
        if existing is not None:
            return existing
        if baseline_run_id:
            baseline = runs.get_evaluation_run(baseline_run_id)
            if baseline is None:
                raise NotFoundError("evaluation_run", baseline_run_id)
            if baseline.id == run.id:
                raise InvalidRequestError("a run cannot be its own baseline")
        else:
            baseline = runs.previous_comparable_run(run)
        if baseline is None:
            return None
        body = compare_outcomes(
            [CaseOutcome.from_result(r) for r in runs.results_for_run(run.id)],
            [CaseOutcome.from_result(r) for r in runs.results_for_run(baseline.id)],
            self._settings,
        )
        body["run"] = {
            "id": run.id,
            "model": run.model_name,
            "dataset_version": run.dataset_version,
        }
        body["baseline"] = {
            "id": baseline.id,
            "model": baseline.model_name,
            "dataset_version": baseline.dataset_version,
        }
        report = RegressionReport(
            run_id=run.id,
            baseline_run_id=baseline.id,
            has_regression=body["has_regression"],
            new_failures=len(body["new_failures"]),
            new_passes=len(body["new_passes"]),
            report=body,
            created_at=run.finished_at or run.started_at,
        )
        session.add(report)
        AuditRepository(session).record(
            "regression.report_created",
            "evaluation_run",
            run.id,
            payload={"baseline_run_id": baseline.id, "severity": body["severity"]},
        )
        session.flush()
        log.info(
            "regression.compared",
            run_id=run.id,
            baseline_run_id=baseline.id,
            severity=body["severity"],
            new_failures=report.new_failures,
        )
        return report
