"""Human-in-the-loop curation: review queue actions and test-case edits.

Every action is recorded as an append-only ``review_decisions`` row and an
audit event. The first decisive action (approve / edit / reject) resolves a
queue item; later decisions by other reviewers are stored as second opinions
(``applied = False``) so inter-annotator agreement can be measured without
changing the dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from evalforge.db.base import utcnow
from evalforge.db.models import EvalCandidate, EvalTestCase, ReviewDecision, ReviewQueueItem
from evalforge.domain.enums import (
    CandidateStatus,
    ReviewAction,
    ReviewStatus,
    TestCaseSource,
    TestCaseStatus,
)
from evalforge.domain.errors import ConflictError, NotFoundError
from evalforge.domain.schemas import LabelEdit, LabelSet
from evalforge.domain.text_signals import content_hash
from evalforge.repositories import AuditRepository, DatasetRepository
from evalforge.services.dataset import (
    SimilarityIndex,
    label_row,
    labels_from_row,
    promote_candidate,
)


@dataclass(frozen=True)
class ReviewOutcome:
    """Result of a reviewer action."""

    item: ReviewQueueItem
    decision: ReviewDecision
    applied: bool
    test_case: EvalTestCase | None = None


class ReviewService:
    """Applies reviewer decisions to the queue and the dataset."""

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock

    # -- queue actions ---------------------------------------------------------
    def approve(
        self, session: Session, item_id: str, reviewer: str, notes: str = ""
    ) -> ReviewOutcome:
        """Approve the auto-generated labels as-is."""
        return self._decide(session, item_id, reviewer, ReviewAction.APPROVE, None, notes)

    def edit(
        self, session: Session, item_id: str, reviewer: str, edit: LabelEdit, notes: str = ""
    ) -> ReviewOutcome:
        """Approve the candidate with edited labels."""
        return self._decide(session, item_id, reviewer, ReviewAction.EDIT, edit, notes)

    def reject(
        self, session: Session, item_id: str, reviewer: str, notes: str = ""
    ) -> ReviewOutcome:
        """Reject the candidate; it never enters the dataset."""
        return self._decide(session, item_id, reviewer, ReviewAction.REJECT, None, notes)

    def annotate(self, session: Session, item_id: str, reviewer: str, notes: str) -> ReviewOutcome:
        """Attach a note without resolving the item."""
        item = self._get_item(session, item_id)
        decision = self._record(session, item, reviewer, ReviewAction.NOTE, None, notes, True)
        return ReviewOutcome(item, decision, applied=True)

    def _decide(
        self,
        session: Session,
        item_id: str,
        reviewer: str,
        action: ReviewAction,
        edit: LabelEdit | None,
        notes: str,
    ) -> ReviewOutcome:
        item = self._get_item(session, item_id)
        candidate = item.candidate
        if item.status != ReviewStatus.PENDING.value:
            # Second opinion: recorded for agreement metrics, dataset untouched.
            decision = self._record(session, item, reviewer, action, edit, notes, False)
            return ReviewOutcome(item, decision, applied=False)

        now = self._clock()
        dataset = DatasetRepository(session)
        test_case: EvalTestCase | None = None
        if action is ReviewAction.REJECT:
            item.status = ReviewStatus.REJECTED.value
            candidate.status = CandidateStatus.REJECTED.value
        else:
            labels = LabelSet.model_validate(candidate.proposed_labels)
            if edit is not None:
                labels = edit.apply_to(labels)
            labels = labels.model_copy(update={"confidence": 1.0})
            digest = content_hash(
                candidate.interaction.system_prompt, candidate.interaction.user_prompt
            )
            if dataset.test_case_by_hash(digest) is not None:
                raise ConflictError(
                    "an identical test case (same system and user prompt) already exists; "
                    "reject this candidate instead"
                )
            version = dataset.current_version() + 1
            test_case = promote_candidate(
                session,
                candidate,
                labels,
                source=TestCaseSource.HUMAN_REVIEWED,
                created_by=reviewer,
                dataset_version=version,
                labeling_model=f"human:{reviewer}",
                now=now,
                notes=notes,
            )
            dataset.bump_version(f"review: case approved by {reviewer}", now, version)
            item.status = (
                ReviewStatus.EDITED.value if edit is not None else ReviewStatus.APPROVED.value
            )
            candidate.status = CandidateStatus.APPROVED.value
        item.resolved_at = now
        decision = self._record(session, item, reviewer, action, edit, notes, True)
        session.flush()
        return ReviewOutcome(item, decision, applied=True, test_case=test_case)

    def _record(
        self,
        session: Session,
        item: ReviewQueueItem,
        reviewer: str,
        action: ReviewAction,
        edit: LabelEdit | None,
        notes: str,
        applied: bool,
    ) -> ReviewDecision:
        decision = ReviewDecision(
            review_item_id=item.id,
            reviewer=reviewer,
            action=action.value,
            edited_labels=edit.model_dump(mode="json", exclude_none=True) if edit else None,
            notes=notes,
            applied=applied,
            created_at=self._clock(),
        )
        session.add(decision)
        AuditRepository(session).record(
            f"review.{action.value}",
            "review_queue_item",
            item.id,
            actor=reviewer,
            payload={"candidate_id": item.candidate_id, "applied": applied},
        )
        session.flush()
        return decision

    @staticmethod
    def _get_item(session: Session, item_id: str) -> ReviewQueueItem:
        item = DatasetRepository(session).get_review_item(item_id)
        if item is None:
            raise NotFoundError("review_queue_item", item_id)
        return item

    # -- reviewer context --------------------------------------------------------
    def similar_cases(
        self, session: Session, candidate: EvalCandidate, k: int = 5
    ) -> list[tuple[EvalTestCase, float]]:
        """Existing test cases most similar to a candidate (reviewer context)."""
        dataset = DatasetRepository(session)
        cases = {c.id: c for c in dataset.active_test_cases()}
        index = SimilarityIndex()
        for case in cases.values():
            index.add("test_case", case.id, case.embedding)
        return [
            (cases[entity_id], round(similarity, 4))
            for _, entity_id, similarity in index.nearest(candidate.classification.embedding, k)
        ]

    # -- test-case curation ------------------------------------------------------
    def edit_test_case(
        self,
        session: Session,
        test_case_id: str,
        reviewer: str,
        edit: LabelEdit,
        notes: str = "",
    ) -> EvalTestCase:
        """Create a new label version for an existing test case."""
        dataset = DatasetRepository(session)
        test_case = dataset.get_test_case(test_case_id)
        if test_case is None:
            raise NotFoundError("test_case", test_case_id)
        current = test_case.current_label
        if current is None:
            raise ConflictError("test case has no current label version")
        now = self._clock()
        labels = edit.apply_to(labels_from_row(current, test_case.category, test_case.difficulty))
        labels = labels.model_copy(update={"confidence": 1.0})
        current.is_current = False
        test_case.labels.append(
            label_row(
                labels,
                version=current.version + 1,
                agreement=None,
                labeling_model=f"human:{reviewer}",
                labeling_passes=1,
                created_by=reviewer,
                now=now,
            )
        )
        test_case.category = labels.category
        test_case.difficulty = labels.difficulty.value
        test_case.source = TestCaseSource.HUMAN_REVIEWED.value
        test_case.updated_at = now
        if notes:
            test_case.notes = f"{test_case.notes}\n{notes}".strip()
        dataset.bump_version(f"review: labels of a case edited by {reviewer}", now)
        AuditRepository(session).record(
            "test_case.labels_edited",
            "test_case",
            test_case.id,
            actor=reviewer,
            payload={"label_version": current.version + 1},
        )
        session.flush()
        return test_case

    def add_note(
        self, session: Session, test_case_id: str, reviewer: str, note: str
    ) -> EvalTestCase:
        """Append a reviewer note to a test case."""
        test_case = DatasetRepository(session).get_test_case(test_case_id)
        if test_case is None:
            raise NotFoundError("test_case", test_case_id)
        stamp = self._clock().date().isoformat()
        test_case.notes = f"{test_case.notes}\n[{stamp} {reviewer}] {note}".strip()
        AuditRepository(session).record(
            "test_case.note_added", "test_case", test_case.id, actor=reviewer
        )
        session.flush()
        return test_case

    def retire_test_case(
        self, session: Session, test_case_id: str, reviewer: str, reason: str = ""
    ) -> EvalTestCase:
        """Remove a test case from the active dataset (history is preserved)."""
        dataset = DatasetRepository(session)
        test_case = dataset.get_test_case(test_case_id)
        if test_case is None:
            raise NotFoundError("test_case", test_case_id)
        if test_case.status == TestCaseStatus.RETIRED.value:
            raise ConflictError("test case is already retired")
        now = self._clock()
        test_case.status = TestCaseStatus.RETIRED.value
        test_case.updated_at = now
        if reason:
            test_case.notes = f"{test_case.notes}\n[retired by {reviewer}] {reason}".strip()
        dataset.bump_version(f"review: case retired by {reviewer}", now)
        AuditRepository(session).record(
            "test_case.retired",
            "test_case",
            test_case.id,
            actor=reviewer,
            payload={"reason": reason},
        )
        session.flush()
        return test_case
