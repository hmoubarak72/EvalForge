"""Stage 1: ingestion and normalization.

Reads raw logs through a source adapter, validates them against the canonical
schema, redacts PII, deduplicates on a stable fingerprint and persists the
result. Metrics for every pass are stored in ``ingestion_runs``.

Idempotency: re-ingesting the same records is a no-op because the fingerprint
column is unique and checked before insert.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from evalforge.adapters.sources import RawRecord, SourceError, build_adapter
from evalforge.db.base import utcnow
from evalforge.db.models import IngestionRun, IngestionSource, InteractionLog
from evalforge.domain.enums import RunStatus
from evalforge.domain.schemas import LLMInteraction
from evalforge.repositories import AuditRepository, InteractionRepository
from evalforge.services.redaction import Redactor
from evalforge.telemetry import current_correlation_id, get_logger

log = get_logger(__name__)

MAX_ERROR_DETAILS = 50
# Re-read a small window before the cursor so late-arriving records are not
# missed; fingerprint deduplication makes the overlap harmless.
CURSOR_OVERLAP = timedelta(minutes=30)


def fingerprint(interaction: LLMInteraction) -> str:
    """Stable content + metadata fingerprint of a (redacted) interaction.

    The upstream ID is deliberately excluded so the same call delivered by two
    sources (e.g. app logs and traces) deduplicates.
    """
    parts = [
        interaction.timestamp.isoformat(timespec="milliseconds"),
        interaction.application_feature,
        interaction.model_name,
        interaction.system_prompt,
        interaction.user_prompt,
        interaction.raw_response,
        str(interaction.metadata.get("session_id", "")),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class IngestionService:
    """Ingests one configured source into the central log store."""

    def __init__(
        self,
        redactor: Redactor,
        clock: Callable[[], datetime] = utcnow,
        batch_size: int = 500,
    ) -> None:
        self._redactor = redactor
        self._clock = clock
        self._batch_size = batch_size

    def ingest_source(
        self,
        session: Session,
        source: IngestionSource,
        pipeline_run_id: str | None = None,
    ) -> IngestionRun:
        """Run one ingestion pass. Never raises for source or record errors;
        they are counted, logged and stored on the returned run."""
        now = self._clock()
        run = IngestionRun(
            source_id=source.id,
            pipeline_run_id=pipeline_run_id,
            status=RunStatus.RUNNING.value,
            started_at=now,
            correlation_id=current_correlation_id(),
            error_details=[],
        )
        session.add(run)
        session.flush()
        since = source.cursor_timestamp - CURSOR_OVERLAP if source.cursor_timestamp else None
        errors: list[dict[str, str]] = []
        newest: datetime | None = source.cursor_timestamp
        batch: list[tuple[LLMInteraction, str]] = []
        seen: set[str] = set()

        try:
            adapter = build_adapter(source.kind, source.name, dict(source.config))
            for record in adapter.read(since=since, until=now):
                run.records_received += 1
                prepared = self._prepare(record, run, errors)
                if prepared is None:
                    continue
                interaction, digest = prepared
                if digest in seen:
                    run.duplicates += 1
                    continue
                seen.add(digest)
                batch.append((interaction, digest))
                if newest is None or interaction.timestamp > newest:
                    newest = interaction.timestamp
                if len(batch) >= self._batch_size:
                    self._flush(session, source, run, batch, errors)
                    batch = []
            self._flush(session, source, run, batch, errors)
            run.status = RunStatus.COMPLETED.value
            source.cursor_timestamp = newest
        except SourceError as exc:
            run.status = RunStatus.FAILED.value
            run.failures += 1
            errors.append({"stage": "source", "error": str(exc)})
            log.error("ingestion.source_failed", source=source.name, error=str(exc))

        run.error_details = errors[:MAX_ERROR_DETAILS]
        run.finished_at = self._clock()
        AuditRepository(session).record(
            "ingestion.completed"
            if run.status == RunStatus.COMPLETED.value
            else "ingestion.failed",
            "ingestion_run",
            run.id,
            payload=self.metrics(run),
        )
        session.flush()
        log.info("ingestion.finished", source=source.name, **self.metrics(run))
        return run

    def _prepare(
        self, record: RawRecord, run: IngestionRun, errors: list[dict[str, str]]
    ) -> tuple[LLMInteraction, str] | None:
        if record.interaction is None:
            run.records_rejected += 1
            errors.append({"stage": "validation", "error": record.error or "invalid record"})
            return None
        result = self._redactor.redact_interaction(record.interaction)
        run.redactions += result.redaction_count
        return result.interaction, fingerprint(result.interaction)

    def _flush(
        self,
        session: Session,
        source: IngestionSource,
        run: IngestionRun,
        batch: list[tuple[LLMInteraction, str]],
        errors: list[dict[str, str]],
    ) -> None:
        if not batch:
            return
        existing = InteractionRepository(session).existing_fingerprints(d for _, d in batch)
        for interaction, digest in batch:
            if digest in existing:
                run.duplicates += 1
                continue
            try:
                with session.begin_nested():
                    session.add(self._to_row(interaction, digest, source.id, run.id))
                run.records_accepted += 1
            except Exception as exc:  # a single bad row must not abort the batch
                run.failures += 1
                errors.append({"stage": "persist", "error": type(exc).__name__})
                log.error("ingestion.persist_failed", error=type(exc).__name__)
        session.flush()

    def _to_row(
        self, interaction: LLMInteraction, digest: str, source_id: str, run_id: str
    ) -> InteractionLog:
        return InteractionLog(
            fingerprint=digest,
            source_id=source_id,
            ingestion_run_id=run_id,
            external_id=interaction.id,
            timestamp=interaction.timestamp,
            application_feature=interaction.application_feature,
            model_name=interaction.model_name,
            system_prompt=interaction.system_prompt,
            user_prompt=interaction.user_prompt,
            raw_response=interaction.raw_response,
            latency_ms=interaction.latency_ms,
            input_tokens=interaction.input_tokens,
            output_tokens=interaction.output_tokens,
            error_status=interaction.error_status,
            user_feedback=interaction.user_feedback.value if interaction.user_feedback else None,
            retry_count=interaction.retry_count,
            response_length=interaction.response_length,
            content_filter_triggered=interaction.content_filter_triggered,
            confidence_score=interaction.confidence_score,
            metadata_=interaction.metadata,
            redaction_status=interaction.redaction_status.value,
            ingested_at=self._clock(),
        )

    @staticmethod
    def metrics(run: IngestionRun) -> dict[str, int | str]:
        """Ingestion metrics as a plain dictionary."""
        return {
            "status": run.status,
            "records_received": run.records_received,
            "records_accepted": run.records_accepted,
            "records_rejected": run.records_rejected,
            "duplicates": run.duplicates,
            "redactions": run.redactions,
            "failures": run.failures,
        }
