"""Pipeline orchestration.

Runs the stages in order, each in its own database transaction, and records
per-stage results on the ``pipeline_runs`` row. Because every stage is
idempotent and results are committed stage by stage, a failed run can be
resumed with :meth:`PipelineOrchestrator.resume` without repeating work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from evalforge.db.base import utcnow
from evalforge.db.models import PipelineRun
from evalforge.db.session import session_scope
from evalforge.domain.enums import RunStatus
from evalforge.domain.errors import InvalidRequestError, NotFoundError
from evalforge.repositories import AuditRepository, InteractionRepository, RunRepository
from evalforge.services.classification import ClassificationService
from evalforge.services.dedup import DedupService
from evalforge.services.ingestion import IngestionService
from evalforge.services.labeling import LabelingService
from evalforge.services.metrics import DatasetMetricsService
from evalforge.services.sampling import SamplingService
from evalforge.telemetry import bind_correlation_id, get_logger

log = get_logger(__name__)

STAGES: tuple[str, ...] = ("ingestion", "sampling", "classification", "labeling", "deduplication")


@dataclass
class PipelineResult:
    """Detached summary of a pipeline run (safe to use after sessions close)."""

    run_id: str
    status: str
    stages: list[str]
    stage_results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class PipelineOrchestrator:
    """Coordinates the ingestion -> dataset pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        ingestion: IngestionService,
        sampling: SamplingService,
        classification: ClassificationService,
        labeling: LabelingService,
        dedup: DedupService,
        metrics: DatasetMetricsService,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._session_factory = session_factory
        self._ingestion = ingestion
        self._sampling = sampling
        self._classification = classification
        self._labeling = labeling
        self._dedup = dedup
        self._metrics = metrics
        self._clock = clock

    def run(
        self,
        stages: Sequence[str] | None = None,
        *,
        trigger: str = "manual",
        options: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run the given stages (all by default) as a new pipeline run.

        Options: ``strategy``, ``sample_size``, ``seed`` (sampling overrides)
        and ``source_ids`` (restrict ingestion to specific sources).
        """
        selected = self._validate_stages(stages)
        options = dict(options or {})
        with bind_correlation_id(trigger=trigger) as correlation_id:
            with session_scope(self._session_factory) as session:
                run = PipelineRun(
                    status=RunStatus.RUNNING.value,
                    trigger=trigger,
                    stages=selected,
                    config=options,
                    stage_results={},
                    correlation_id=correlation_id,
                    started_at=self._clock(),
                )
                session.add(run)
                session.flush()
                run_id = run.id
            log.info("pipeline.started", pipeline_run_id=run_id, stages=selected)
            return self._execute(run_id, selected, options)

    def resume(self, run_id: str) -> PipelineResult:
        """Re-run the stages of a failed run that have not completed yet."""
        with session_scope(self._session_factory) as session:
            run = RunRepository(session).get_pipeline_run(run_id)
            if run is None:
                raise NotFoundError("pipeline_run", run_id)
            if run.status == RunStatus.COMPLETED.value:
                raise InvalidRequestError("pipeline run already completed")
            remaining = [s for s in run.stages if s not in run.stage_results]
            options = dict(run.config)
            correlation_id = run.correlation_id
            run.status = RunStatus.RUNNING.value
            run.error = None
        with bind_correlation_id(correlation_id, trigger="resume"):
            log.info("pipeline.resumed", pipeline_run_id=run_id, stages=remaining)
            return self._execute(run_id, remaining, options)

    @staticmethod
    def _validate_stages(stages: Sequence[str] | None) -> list[str]:
        if not stages:
            return list(STAGES)
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise InvalidRequestError(
                f"unknown stage(s): {', '.join(unknown)}; valid stages: {', '.join(STAGES)}"
            )
        return [s for s in STAGES if s in stages]  # canonical order, no duplicates

    def _execute(
        self, run_id: str, stages: Sequence[str], options: dict[str, Any]
    ) -> PipelineResult:
        error: str | None = None
        for stage in stages:
            try:
                with session_scope(self._session_factory) as session:
                    result = self._run_stage(session, stage, run_id, options)
                    run = session.get(PipelineRun, run_id)
                    assert run is not None
                    run.stage_results = {**run.stage_results, stage: result}
                log.info("pipeline.stage_completed", pipeline_run_id=run_id, stage=stage)
            except Exception as exc:
                error = f"stage '{stage}' failed: {type(exc).__name__}: {exc}"
                log.error(
                    "pipeline.stage_failed",
                    pipeline_run_id=run_id,
                    stage=stage,
                    error=error,
                    exc_info=True,
                )
                break
        with session_scope(self._session_factory) as session:
            run = session.get(PipelineRun, run_id)
            assert run is not None
            run.status = RunStatus.FAILED.value if error else RunStatus.COMPLETED.value
            run.error = error
            run.finished_at = self._clock()
            AuditRepository(session).record(
                f"pipeline.{run.status}",
                "pipeline_run",
                run_id,
                payload={"stages": list(run.stages), "trigger": run.trigger},
            )
            summary = PipelineResult(
                run_id, run.status, list(run.stages), dict(run.stage_results), error
            )
        log.info("pipeline.finished", pipeline_run_id=run_id, status=summary.status)
        return summary

    def _run_stage(
        self, session: Session, stage: str, run_id: str, options: dict[str, Any]
    ) -> dict[str, Any]:
        if stage == "ingestion":
            return self._ingest(session, run_id, options.get("source_ids"))
        if stage == "sampling":
            samples = self._sampling.sample(
                session,
                run_id,
                strategy=options.get("strategy"),
                sample_size=options.get("sample_size"),
                seed=options.get("seed"),
                underrepresented_features=self._metrics.underrepresented_features(session),
            )
            reasons: dict[str, int] = {}
            for sample in samples:
                for reason in sample.reasons:
                    key = str(reason).split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
            return {
                "sampled": len(samples),
                "strategy": samples[0].strategy if samples else options.get("strategy"),
                "seed": samples[0].seed if samples else options.get("seed"),
                "reason_counts": reasons,
            }
        if stage == "classification":
            repo = InteractionRepository(session)
            ids = [s.interaction_id for s in repo.samples_for_run(run_id)]
            if not ids:  # stage triggered on its own: classify the backlog
                ids = repo.unclassified_sampled_ids()
            return self._classification.classify(session, ids, run_id)
        if stage == "labeling":
            return self._labeling.run(session, run_id)
        if stage == "deduplication":
            return self._dedup.run(session, run_id)
        raise InvalidRequestError(f"unknown stage: {stage}")

    def _ingest(
        self, session: Session, run_id: str, source_ids: Sequence[str] | None
    ) -> dict[str, Any]:
        sources = InteractionRepository(session).list_sources(enabled_only=True)
        if source_ids:
            sources = [s for s in sources if s.id in set(source_ids)]
        totals = {
            "records_received": 0,
            "records_accepted": 0,
            "records_rejected": 0,
            "duplicates": 0,
            "redactions": 0,
            "failures": 0,
        }
        per_source: dict[str, Any] = {}
        for source in sources:
            run = self._ingestion.ingest_source(session, source, pipeline_run_id=run_id)
            metrics = self._ingestion.metrics(run)
            per_source[source.name] = metrics
            for key in totals:
                totals[key] += int(metrics[key])
        return {"sources": len(sources), **totals, "per_source": per_source}
