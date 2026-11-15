"""Stage 4a: candidate selection and auto-labeling.

Selects the most valuable classified interactions (outliers, clear failures,
strong exemplars, under-represented clusters), runs the configured number of
labeling passes and stores the consensus labels with agreement statistics.

Candidates that are near-duplicates of an existing test case are detected
*before* labeling so no model calls are wasted on them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from evalforge.config import DedupSettings, LabelingSettings
from evalforge.db.base import utcnow
from evalforge.db.models import EvalCandidate, InteractionClassification
from evalforge.domain.enums import CandidateStatus, CaseType, Difficulty
from evalforge.repositories import DatasetRepository, InteractionRepository
from evalforge.services.dataset import SimilarityIndex
from evalforge.services.labelers import Labeler, LabelingContext, aggregate_passes
from evalforge.services.metrics import UNCATEGORIZED, DatasetMetricsService
from evalforge.telemetry import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CandidateDecision:
    """Why (and how urgently) a classified interaction becomes a candidate."""

    priority: float
    reasons: list[str]
    case_type: CaseType


def select_candidate(
    classification: InteractionClassification, cluster_deficit: float
) -> CandidateDecision | None:
    """Decide whether a classified interaction is worth turning into a test case.

    Returns ``None`` for infrastructure failures (no model behaviour to test)
    and for unremarkable mid-quality, non-outlier interactions.
    """
    interaction = classification.interaction
    if interaction.error_status or not interaction.raw_response.strip():
        return None
    quality = classification.quality_score
    reasons = list(classification.outlier_reasons)
    priority = 0.4 * min(len(reasons), 3)
    if quality <= 2.5:
        reasons.append("low_quality_response")
        priority += 1.0
    elif quality >= 4.0:
        reasons.append("high_quality_exemplar")
        priority += 0.5
    elif not classification.is_outlier:
        return None
    if classification.difficulty == Difficulty.ADVERSARIAL.value:
        reasons.append("adversarial_input")
        priority += 0.5
    if cluster_deficit >= 0.5:
        reasons.append("underrepresented_cluster")
    priority += cluster_deficit
    case_type = CaseType.POSITIVE if quality >= 4.0 else CaseType.NEGATIVE
    return CandidateDecision(round(priority, 4), reasons, case_type)


class LabelingService:
    """Creates and labels evaluation candidates."""

    def __init__(
        self,
        settings: LabelingSettings,
        dedup: DedupSettings,
        labeler: Labeler,
        metrics: DatasetMetricsService,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings = settings
        self._dedup = dedup
        self._labeler = labeler
        self._metrics = metrics
        self._clock = clock

    def run(self, session: Session, pipeline_run_id: str | None = None) -> dict[str, Any]:
        """Select and label new candidates. Idempotent: an interaction can only
        ever have one candidate, so re-running never duplicates work."""
        dataset = DatasetRepository(session)
        clusters = {c.id: c for c in InteractionRepository(session).list_clusters()}
        deficits = self._metrics.cluster_deficits(session)

        ranked: list[tuple[CandidateDecision, InteractionClassification]] = []
        considered = 0
        for classification in dataset.unlabeled_classifications():
            considered += 1
            deficit = deficits.get(classification.cluster_id or "", 0.0)
            decision = select_candidate(classification, deficit)
            if decision is not None:
                ranked.append((decision, classification))
        ranked.sort(key=lambda pair: (-pair[0].priority, pair[1].interaction_id))
        selected = ranked[: self._settings.max_candidates_per_run]

        index = SimilarityIndex()
        for case in dataset.active_test_cases():
            index.add("test_case", case.id, case.embedding)

        now = self._clock()
        stats: dict[str, Any] = {
            "considered": considered,
            "eligible": len(ranked),
            "labeled": 0,
            "duplicates_skipped": 0,
            "labeling_passes": self._settings.passes,
            "labeling_model": self._labeler.model_name,
        }
        for decision, classification in selected:
            cluster = clusters.get(classification.cluster_id or "")
            candidate = EvalCandidate(
                interaction_id=classification.interaction_id,
                classification_id=classification.id,
                pipeline_run_id=pipeline_run_id,
                cluster_id=classification.cluster_id,
                status=CandidateStatus.PENDING.value,
                case_type=decision.case_type.value,
                selection_reasons=decision.reasons,
                priority=decision.priority,
                proposed_labels={},
                label_passes=[],
                conflicts=[],
                created_at=now,
                updated_at=now,
            )
            session.add(candidate)
            nearest = index.nearest(classification.embedding, k=1)
            if nearest and nearest[0][2] > self._dedup.similarity_threshold:
                candidate.status = CandidateStatus.DUPLICATE.value
                candidate.max_similarity = round(nearest[0][2], 4)
                candidate.duplicate_of_test_case_id = nearest[0][1]
                stats["duplicates_skipped"] += 1
                continue
            context = LabelingContext(
                interaction=classification.interaction,
                classification=classification,
                category=cluster.name if cluster else UNCATEGORIZED,
                case_type=decision.case_type,
            )
            passes = [
                self._labeler.label(context, pass_index)
                for pass_index in range(self._settings.passes)
            ]
            aggregated = aggregate_passes(passes)
            candidate.proposed_labels = aggregated.labels.model_dump(mode="json")
            candidate.label_passes = [p.model_dump(mode="json") for p in passes]
            candidate.confidence = aggregated.confidence
            candidate.agreement = aggregated.agreement
            candidate.conflicts = aggregated.conflicts
            stats["labeled"] += 1
        session.flush()
        log.info("labeling.finished", **stats)
        return stats
