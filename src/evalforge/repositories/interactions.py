"""Repository for sources, interaction logs, samples, clusters, classifications."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.db.models import (
    Cluster,
    IngestionRun,
    IngestionSource,
    InteractionClassification,
    InteractionLog,
    Sample,
)
from evalforge.repositories.base import Page, paginate

_CHUNK = 500


class InteractionRepository:
    """Queries over the log store and the classifier outputs."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- sources ---------------------------------------------------------
    def get_source(self, source_id: str) -> IngestionSource | None:
        return self._session.get(IngestionSource, source_id)

    def get_source_by_name(self, name: str) -> IngestionSource | None:
        return self._session.scalar(select(IngestionSource).where(IngestionSource.name == name))

    def list_sources(self, enabled_only: bool = False) -> list[IngestionSource]:
        statement = select(IngestionSource).order_by(IngestionSource.name)
        if enabled_only:
            statement = statement.where(IngestionSource.enabled.is_(True))
        return list(self._session.scalars(statement))

    def list_ingestion_runs(self, limit: int = 50, offset: int = 0) -> Page[IngestionRun]:
        statement = select(IngestionRun).order_by(IngestionRun.started_at.desc())
        return paginate(self._session, statement, limit, offset)

    # -- interactions ----------------------------------------------------
    def existing_fingerprints(self, fingerprints: Iterable[str]) -> set[str]:
        """Return the subset of fingerprints already present in the log store."""
        wanted = list(fingerprints)
        found: set[str] = set()
        for start in range(0, len(wanted), _CHUNK):
            chunk = wanted[start : start + _CHUNK]
            found.update(
                self._session.scalars(
                    select(InteractionLog.fingerprint).where(InteractionLog.fingerprint.in_(chunk))
                )
            )
        return found

    def get(self, interaction_id: str) -> InteractionLog | None:
        return self._session.get(InteractionLog, interaction_id)

    def get_many(self, interaction_ids: Sequence[str]) -> list[InteractionLog]:
        rows: list[InteractionLog] = []
        for start in range(0, len(interaction_ids), _CHUNK):
            chunk = interaction_ids[start : start + _CHUNK]
            rows.extend(
                self._session.scalars(select(InteractionLog).where(InteractionLog.id.in_(chunk)))
            )
        return rows

    def list_interactions(
        self,
        *,
        feature: str | None = None,
        model: str | None = None,
        feedback: str | None = None,
        has_error: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[InteractionLog]:
        statement = select(InteractionLog).order_by(InteractionLog.timestamp.desc())
        if feature:
            statement = statement.where(InteractionLog.application_feature == feature)
        if model:
            statement = statement.where(InteractionLog.model_name == model)
        if feedback:
            statement = statement.where(InteractionLog.user_feedback == feedback)
        if has_error is True:
            statement = statement.where(InteractionLog.error_status.is_not(None))
        elif has_error is False:
            statement = statement.where(InteractionLog.error_status.is_(None))
        if since:
            statement = statement.where(InteractionLog.timestamp >= since)
        if until:
            statement = statement.where(InteractionLog.timestamp <= until)
        return paginate(self._session, statement, limit, offset)

    def count(self) -> int:
        return int(self._session.scalar(select(func.count(InteractionLog.id))) or 0)

    def unsampled(
        self,
        *,
        exclude_previously_sampled: bool,
        since: datetime | None,
        until: datetime | None,
    ) -> list[InteractionLog]:
        """Interactions eligible for sampling in a pipeline run."""
        statement = select(InteractionLog).order_by(InteractionLog.timestamp, InteractionLog.id)
        if exclude_previously_sampled:
            statement = statement.where(
                InteractionLog.id.not_in(select(Sample.interaction_id).distinct())
            )
        if since:
            statement = statement.where(InteractionLog.timestamp >= since)
        if until:
            statement = statement.where(InteractionLog.timestamp <= until)
        return list(self._session.scalars(statement))

    def session_neighbors(
        self, interaction: InteractionLog, window_start: datetime, window_end: datetime
    ) -> list[InteractionLog]:
        """Other interactions of the same feature inside a time window."""
        statement = select(InteractionLog).where(
            InteractionLog.application_feature == interaction.application_feature,
            InteractionLog.timestamp >= window_start,
            InteractionLog.timestamp <= window_end,
            InteractionLog.id != interaction.id,
        )
        return list(self._session.scalars(statement))

    # -- samples -----------------------------------------------------------
    def samples_for_run(self, pipeline_run_id: str) -> list[Sample]:
        return list(
            self._session.scalars(
                select(Sample).where(Sample.pipeline_run_id == pipeline_run_id).order_by(Sample.id)
            )
        )

    def unclassified_sampled_ids(self) -> list[str]:
        """Sampled interactions (any run) that have no classification yet."""
        statement = (
            select(Sample.interaction_id)
            .where(Sample.interaction_id.not_in(select(InteractionClassification.interaction_id)))
            .distinct()
        )
        return list(self._session.scalars(statement))

    def list_samples(
        self, pipeline_run_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> Page[Sample]:
        statement = select(Sample).order_by(Sample.created_at.desc(), Sample.id)
        if pipeline_run_id:
            statement = statement.where(Sample.pipeline_run_id == pipeline_run_id)
        return paginate(self._session, statement, limit, offset)

    # -- clusters ----------------------------------------------------------
    def list_clusters(self) -> list[Cluster]:
        return list(self._session.scalars(select(Cluster).order_by(Cluster.size.desc())))

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        return self._session.get(Cluster, cluster_id)

    def cluster_name_exists(self, name: str) -> bool:
        return self._session.scalar(select(Cluster.id).where(Cluster.name == name)) is not None

    # -- classifications ---------------------------------------------------
    def classification_for(self, interaction_id: str) -> InteractionClassification | None:
        return self._session.scalar(
            select(InteractionClassification).where(
                InteractionClassification.interaction_id == interaction_id
            )
        )

    def classified_ids(self, interaction_ids: Sequence[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(interaction_ids), _CHUNK):
            chunk = interaction_ids[start : start + _CHUNK]
            found.update(
                self._session.scalars(
                    select(InteractionClassification.interaction_id).where(
                        InteractionClassification.interaction_id.in_(chunk)
                    )
                )
            )
        return found

    def recent_classifications(self, limit: int) -> list[InteractionClassification]:
        """Most recent classifications, used as clustering context."""
        if limit <= 0:
            return []
        statement = (
            select(InteractionClassification)
            .order_by(InteractionClassification.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def list_classifications(
        self,
        *,
        cluster_id: str | None = None,
        difficulty: str | None = None,
        outliers_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[InteractionClassification]:
        statement = select(InteractionClassification).order_by(
            InteractionClassification.created_at.desc(), InteractionClassification.id
        )
        if cluster_id:
            statement = statement.where(InteractionClassification.cluster_id == cluster_id)
        if difficulty:
            statement = statement.where(InteractionClassification.difficulty == difficulty)
        if outliers_only:
            statement = statement.where(InteractionClassification.is_outlier.is_(True))
        return paginate(self._session, statement, limit, offset)
