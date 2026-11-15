"""Repository for candidates, test cases, labels, versions and the review queue."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from evalforge.db.models import (
    DatasetVersion,
    EvalCandidate,
    EvalTestCase,
    InteractionClassification,
    InteractionLog,
    ReviewDecision,
    ReviewQueueItem,
    TestCaseLabel,
)
from evalforge.domain.enums import CandidateStatus, ReviewStatus, TestCaseStatus
from evalforge.repositories.base import Page, paginate


class DatasetRepository:
    """Queries over the evolving evaluation dataset."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- candidates --------------------------------------------------------
    def unlabeled_classifications(self) -> list[InteractionClassification]:
        """Classifications that have not been turned into a candidate yet."""
        statement = (
            select(InteractionClassification)
            .join(InteractionLog, InteractionLog.id == InteractionClassification.interaction_id)
            .where(
                InteractionClassification.interaction_id.not_in(
                    select(EvalCandidate.interaction_id)
                )
            )
            .options(selectinload(InteractionClassification.interaction))
            .order_by(InteractionLog.timestamp, InteractionClassification.id)
        )
        return list(self._session.scalars(statement))

    def get_candidate(self, candidate_id: str) -> EvalCandidate | None:
        return self._session.get(EvalCandidate, candidate_id)

    def candidates_with_status(self, status: CandidateStatus) -> list[EvalCandidate]:
        statement = (
            select(EvalCandidate)
            .where(EvalCandidate.status == status.value)
            .order_by(EvalCandidate.priority.desc(), EvalCandidate.id)
        )
        return list(self._session.scalars(statement))

    def list_candidates(
        self,
        *,
        status: str | None = None,
        cluster_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[EvalCandidate]:
        statement = select(EvalCandidate).order_by(
            EvalCandidate.created_at.desc(), EvalCandidate.id
        )
        if status:
            statement = statement.where(EvalCandidate.status == status)
        if cluster_id:
            statement = statement.where(EvalCandidate.cluster_id == cluster_id)
        return paginate(self._session, statement, limit, offset)

    def candidate_status_counts(self) -> dict[str, int]:
        rows = self._session.execute(
            select(EvalCandidate.status, func.count()).group_by(EvalCandidate.status)
        )
        return {status: int(count) for status, count in rows}

    # -- test cases --------------------------------------------------------
    def get_test_case(self, test_case_id: str) -> EvalTestCase | None:
        return self._session.get(EvalTestCase, test_case_id)

    def test_case_by_hash(self, content_hash: str) -> EvalTestCase | None:
        return self._session.scalar(
            select(EvalTestCase).where(EvalTestCase.content_hash == content_hash)
        )

    def active_test_cases(self) -> list[EvalTestCase]:
        statement = (
            select(EvalTestCase)
            .where(EvalTestCase.status == TestCaseStatus.ACTIVE.value)
            .options(selectinload(EvalTestCase.labels))
            .order_by(EvalTestCase.created_at, EvalTestCase.id)
        )
        return list(self._session.scalars(statement))

    def all_test_cases(self) -> list[EvalTestCase]:
        statement = (
            select(EvalTestCase)
            .options(selectinload(EvalTestCase.labels))
            .order_by(EvalTestCase.created_at, EvalTestCase.id)
        )
        return list(self._session.scalars(statement))

    def list_test_cases(
        self,
        *,
        category: str | None = None,
        difficulty: str | None = None,
        source: str | None = None,
        status: str | None = None,
        case_type: str | None = None,
        min_quality: int | None = None,
        max_quality: int | None = None,
        created_since: datetime | None = None,
        created_until: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[EvalTestCase]:
        statement = (
            select(EvalTestCase)
            .options(selectinload(EvalTestCase.labels))
            .order_by(EvalTestCase.created_at.desc(), EvalTestCase.id)
        )
        if category:
            statement = statement.where(EvalTestCase.category == category)
        if difficulty:
            statement = statement.where(EvalTestCase.difficulty == difficulty)
        if source:
            statement = statement.where(EvalTestCase.source == source)
        if status:
            statement = statement.where(EvalTestCase.status == status)
        if case_type:
            statement = statement.where(EvalTestCase.case_type == case_type)
        if created_since:
            statement = statement.where(EvalTestCase.created_at >= created_since)
        if created_until:
            statement = statement.where(EvalTestCase.created_at <= created_until)
        if search:
            statement = statement.where(EvalTestCase.user_prompt.ilike(f"%{search}%"))
        if min_quality is not None or max_quality is not None:
            statement = statement.join(
                TestCaseLabel,
                (TestCaseLabel.test_case_id == EvalTestCase.id)
                & TestCaseLabel.is_current.is_(True),
            )
            if min_quality is not None:
                statement = statement.where(TestCaseLabel.expected_quality_score >= min_quality)
            if max_quality is not None:
                statement = statement.where(TestCaseLabel.expected_quality_score <= max_quality)
        return paginate(self._session, statement, limit, offset)

    # -- dataset versions ----------------------------------------------------
    def current_version(self) -> int:
        return int(self._session.scalar(select(func.max(DatasetVersion.version))) or 0)

    def bump_version(
        self, reason: str, now: datetime, version_number: int | None = None
    ) -> DatasetVersion:
        """Record a new dataset version after the active dataset changed."""
        self._session.flush()
        if version_number is None:
            version_number = self.current_version() + 1
        case_count = int(
            self._session.scalar(
                select(func.count(EvalTestCase.id)).where(
                    EvalTestCase.status == TestCaseStatus.ACTIVE.value
                )
            )
            or 0
        )
        version = DatasetVersion(
            version=version_number, reason=reason[:255], case_count=case_count, created_at=now
        )
        self._session.add(version)
        self._session.flush()
        return version

    def list_versions(self) -> list[DatasetVersion]:
        return list(self._session.scalars(select(DatasetVersion).order_by(DatasetVersion.version)))

    # -- review queue --------------------------------------------------------
    def get_review_item(self, item_id: str) -> ReviewQueueItem | None:
        return self._session.get(ReviewQueueItem, item_id)

    def pending_review_items(self) -> list[ReviewQueueItem]:
        statement = (
            select(ReviewQueueItem)
            .where(ReviewQueueItem.status == ReviewStatus.PENDING.value)
            .order_by(ReviewQueueItem.priority.desc(), ReviewQueueItem.created_at)
        )
        return list(self._session.scalars(statement))

    def list_review_items(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> Page[ReviewQueueItem]:
        statement = select(ReviewQueueItem).order_by(
            ReviewQueueItem.priority.desc(), ReviewQueueItem.created_at, ReviewQueueItem.id
        )
        if status:
            statement = statement.where(ReviewQueueItem.status == status)
        return paginate(self._session, statement, limit, offset)

    def all_review_decisions(self) -> list[ReviewDecision]:
        return list(
            self._session.scalars(select(ReviewDecision).order_by(ReviewDecision.created_at))
        )
