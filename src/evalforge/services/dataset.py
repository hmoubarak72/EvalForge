"""Shared dataset operations: similarity index and candidate promotion."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from evalforge.db.models import EvalCandidate, EvalTestCase, TestCaseLabel
from evalforge.domain.enums import TestCaseSource, TestCaseStatus
from evalforge.domain.schemas import LabelSet
from evalforge.domain.text_signals import content_hash
from evalforge.services.clustering import normalize_rows


class SimilarityIndex:
    """In-memory cosine-similarity index over embeddings.

    Brute force is adequate up to tens of thousands of cases; swap in pgvector
    or FAISS behind the same interface for larger datasets.
    """

    def __init__(self) -> None:
        self._keys: list[tuple[str, str]] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, kind: str, entity_id: str, vector: Sequence[float]) -> None:
        """Add one normalized vector under a ``(kind, id)`` key."""
        if not len(vector):
            return
        self._keys.append((kind, entity_id))
        self._rows.append(normalize_rows(np.array([vector], dtype=float))[0])
        self._matrix = None

    def nearest(self, vector: Sequence[float], k: int = 1) -> list[tuple[str, str, float]]:
        """Return up to ``k`` ``(kind, id, similarity)`` tuples, best first."""
        if not self._keys or not len(vector):
            return []
        if self._matrix is None:
            self._matrix = np.vstack(self._rows)
        query = normalize_rows(np.array([vector], dtype=float))[0]
        similarities = self._matrix @ query
        order = np.argsort(-similarities)[:k]
        return [(*self._keys[int(i)], float(similarities[int(i)])) for i in order]


def labels_from_row(label: TestCaseLabel, category: str, difficulty: str) -> LabelSet:
    """Rebuild a :class:`LabelSet` from a stored label version."""
    return LabelSet.model_validate(
        {
            "question_type": label.question_type,
            "expected_quality_score": label.expected_quality_score,
            "expected_behavior": label.expected_behavior,
            "required_assertions": label.required_assertions,
            "forbidden_assertions": label.forbidden_assertions,
            "reference_answer": label.reference_answer,
            "reference_verified": label.reference_verified,
            "rubric": label.rubric,
            "category": category,
            "difficulty": difficulty,
            "confidence": label.confidence,
        }
    )


def label_row(
    labels: LabelSet,
    *,
    version: int,
    agreement: float | None,
    labeling_model: str,
    labeling_passes: int,
    created_by: str,
    now: datetime,
) -> TestCaseLabel:
    """Build a label-version row from a label set."""
    dumped = labels.model_dump(mode="json")
    return TestCaseLabel(
        version=version,
        is_current=True,
        question_type=dumped["question_type"],
        expected_quality_score=dumped["expected_quality_score"],
        expected_behavior=dumped["expected_behavior"],
        required_assertions=dumped["required_assertions"],
        forbidden_assertions=dumped["forbidden_assertions"],
        reference_answer=dumped["reference_answer"],
        reference_verified=dumped["reference_verified"],
        rubric=dumped["rubric"],
        confidence=dumped["confidence"],
        agreement=agreement,
        labeling_model=labeling_model,
        labeling_passes=labeling_passes,
        created_by=created_by,
        created_at=now,
    )


def promote_candidate(
    session: Session,
    candidate: EvalCandidate,
    labels: LabelSet,
    *,
    source: TestCaseSource,
    created_by: str,
    dataset_version: int,
    labeling_model: str,
    now: datetime,
    notes: str = "",
) -> EvalTestCase:
    """Create a test case (with its first label version) from a candidate."""
    interaction = candidate.interaction
    test_case = EvalTestCase(
        candidate_id=candidate.id,
        source_interaction_id=interaction.id,
        cluster_id=candidate.cluster_id,
        content_hash=content_hash(interaction.system_prompt, interaction.user_prompt),
        added_in_version=dataset_version,
        application_feature=interaction.application_feature,
        system_prompt=interaction.system_prompt,
        user_prompt=interaction.user_prompt,
        production_response=interaction.raw_response,
        category=labels.category,
        difficulty=labels.difficulty.value,
        case_type=candidate.case_type,
        source=source.value,
        status=TestCaseStatus.ACTIVE.value,
        embedding=list(candidate.classification.embedding),
        notes=notes,
        source_timestamp=interaction.timestamp,
        created_at=now,
        updated_at=now,
    )
    test_case.labels.append(
        label_row(
            labels,
            version=1,
            agreement=candidate.agreement,
            labeling_model=labeling_model,
            labeling_passes=max(1, len(candidate.label_passes)),
            created_by=created_by,
            now=now,
        )
    )
    session.add(test_case)
    session.flush()
    return test_case
