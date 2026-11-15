"""Stage 4b: deduplication and confidence-based routing.

Processes labeled candidates in priority order:

1. Near-duplicates of an existing test case, or of a candidate already waiting
   for review, are skipped (cosine similarity above the threshold, 0.92 by
   default).
2. High-confidence, conflict-free candidates are promoted into the dataset.
3. Everything else is routed to the human review queue.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from evalforge.config import DedupSettings, LabelingSettings
from evalforge.db.base import utcnow
from evalforge.db.models import EvalTestCase, ReviewQueueItem
from evalforge.domain.enums import CandidateStatus, ReviewStatus, TestCaseSource
from evalforge.domain.schemas import LabelSet
from evalforge.domain.text_signals import content_hash
from evalforge.repositories import AuditRepository, DatasetRepository
from evalforge.services.dataset import SimilarityIndex, promote_candidate
from evalforge.telemetry import get_logger

log = get_logger(__name__)


class DedupService:
    """Deduplicates labeled candidates and routes them to dataset or review."""

    def __init__(
        self,
        dedup: DedupSettings,
        labeling: LabelingSettings,
        labeling_model: str,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._dedup = dedup
        self._labeling = labeling
        self._labeling_model = labeling_model
        self._clock = clock

    def build_index(self, session: Session) -> SimilarityIndex:
        """Index of active test cases and candidates awaiting review."""
        dataset = DatasetRepository(session)
        index = SimilarityIndex()
        for case in dataset.active_test_cases():
            index.add("test_case", case.id, case.embedding)
        for item in dataset.pending_review_items():
            index.add("candidate", item.candidate_id, item.candidate.classification.embedding)
        return index

    def run(self, session: Session, pipeline_run_id: str | None = None) -> dict[str, Any]:
        """Route every labeled, still-pending candidate. Idempotent."""
        dataset = DatasetRepository(session)
        audit = AuditRepository(session)
        pending = [
            c for c in dataset.candidates_with_status(CandidateStatus.PENDING) if c.proposed_labels
        ]
        stats: dict[str, Any] = {
            "processed": len(pending),
            "auto_accepted": 0,
            "sent_to_review": 0,
            "duplicates": 0,
            "similarity_threshold": self._dedup.similarity_threshold,
            "auto_accept_confidence": self._labeling.auto_accept_confidence,
        }
        if not pending:
            return stats
        index = self.build_index(session)
        now = self._clock()
        next_version = dataset.current_version() + 1
        promoted: list[EvalTestCase] = []

        for candidate in pending:
            vector = candidate.classification.embedding
            nearest = index.nearest(vector, k=1)
            if nearest:
                kind, entity_id, similarity = nearest[0]
                candidate.max_similarity = round(similarity, 4)
                if similarity > self._dedup.similarity_threshold:
                    candidate.status = CandidateStatus.DUPLICATE.value
                    if kind == "test_case":
                        candidate.duplicate_of_test_case_id = entity_id
                    else:
                        candidate.conflicts = [
                            *candidate.conflicts,
                            f"duplicate_of_pending_candidate:{entity_id}",
                        ]
                    stats["duplicates"] += 1
                    continue
            # Exact repeats of a retired case are not in the similarity index.
            retired = dataset.test_case_by_hash(
                content_hash(candidate.interaction.system_prompt, candidate.interaction.user_prompt)
            )
            if retired is not None:
                candidate.status = CandidateStatus.DUPLICATE.value
                candidate.duplicate_of_test_case_id = retired.id
                candidate.max_similarity = 1.0
                stats["duplicates"] += 1
                continue
            labels = LabelSet.model_validate(candidate.proposed_labels)
            confidence = candidate.confidence or 0.0
            if confidence >= self._labeling.auto_accept_confidence and not candidate.conflicts:
                test_case = promote_candidate(
                    session,
                    candidate,
                    labels,
                    source=TestCaseSource.AUTO_LABELED,
                    created_by="auto",
                    dataset_version=next_version,
                    labeling_model=self._labeling_model,
                    now=now,
                )
                candidate.status = CandidateStatus.AUTO_ACCEPTED.value
                index.add("test_case", test_case.id, vector)
                promoted.append(test_case)
                stats["auto_accepted"] += 1
            else:
                reasons = list(candidate.conflicts)
                if confidence < self._labeling.auto_accept_confidence:
                    reasons.insert(
                        0,
                        f"low_confidence ({confidence:.2f} < "
                        f"{self._labeling.auto_accept_confidence:.2f})",
                    )
                session.add(
                    ReviewQueueItem(
                        candidate_id=candidate.id,
                        status=ReviewStatus.PENDING.value,
                        reason="; ".join(reasons)[:255],
                        priority=candidate.priority,
                        created_at=now,
                    )
                )
                candidate.status = CandidateStatus.NEEDS_REVIEW.value
                index.add("candidate", candidate.id, vector)
                stats["sent_to_review"] += 1

        if promoted:
            version = dataset.bump_version(
                f"pipeline: {len(promoted)} auto-labeled case(s) added", now, next_version
            )
            stats["dataset_version"] = version.version
            audit.record(
                "dataset.cases_added",
                "dataset_version",
                str(version.version),
                payload={"count": len(promoted), "pipeline_run_id": pipeline_run_id},
            )
        session.flush()
        log.info("dedup.finished", **stats)
        return stats
