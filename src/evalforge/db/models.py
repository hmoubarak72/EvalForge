"""SQLAlchemy 2.0 ORM models.

Design notes
------------
* Primary keys are string UUIDs so the schema is portable between PostgreSQL
  and the SQLite development mode.
* Flexible payloads use JSONB on PostgreSQL (``JSONType``).
* ``interaction_logs`` is append-only and only referenced by ID from the rest
  of the schema, so it can later be moved to ClickHouse with the application
  tables keeping an ``interaction_id`` string reference.
* Historical results (evaluation runs/results, regression reports, label
  versions, review decisions, audit events) are never updated in place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalforge.db.base import Base, JSONType, UTCDateTime, new_id, utcnow


def _pk() -> Mapped[str]:
    return mapped_column(String(36), primary_key=True, default=new_id)


def _created() -> Mapped[datetime]:
    return mapped_column(UTCDateTime(), default=utcnow, nullable=False)


class IngestionSource(Base):
    """A configured log source (JSON logs, OTel export, or database read)."""

    __tablename__ = "ingestion_sources"

    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Incremental cursor: newest interaction timestamp ingested so far.
    cursor_timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = _created()


class PipelineRun(Base):
    """One execution of the (full or partial) pipeline."""

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = _pk()
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    trigger: Mapped[str] = mapped_column(String(32), default="manual", nullable=False)
    stages: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    stage_results: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class IngestionRun(Base):
    """Metrics for one ingestion pass over one source."""

    __tablename__ = "ingestion_runs"

    id: Mapped[str] = _pk()
    source_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    records_received: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_accepted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicates: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    redactions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_details: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    source: Mapped[IngestionSource] = relationship()


class InteractionLog(Base):
    """Normalized, redacted production LLM interaction (canonical log record)."""

    __tablename__ = "interaction_logs"
    __table_args__ = (
        Index("ix_interaction_logs_feature_timestamp", "application_feature", "timestamp"),
        Index("ix_interaction_logs_model_timestamp", "model_name", "timestamp"),
    )

    id: Mapped[str] = _pk()
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_sources.id", ondelete="SET NULL"), nullable=True, index=True
    )
    ingestion_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"), nullable=True
    )
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    application_feature: Mapped[str] = mapped_column(String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str] = mapped_column(Text, default="", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_status: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_feedback: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    response_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content_filter_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONType, default=dict, nullable=False
    )
    redaction_status: Mapped[str] = mapped_column(String(16), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class Sample(Base):
    """Record of an interaction selected by a sampling run, with the reasons."""

    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "interaction_id", name="uq_samples_run_interaction"),
    )

    id: Mapped[str] = _pk()
    pipeline_run_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interaction_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    stratum: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reasons: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    created_at: Mapped[datetime] = _created()

    interaction: Mapped[InteractionLog] = relationship()


class Cluster(Base):
    """A persistent semantic category discovered by HDBSCAN."""

    __tablename__ = "clusters"

    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    keywords: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    naming_method: Mapped[str] = mapped_column(String(32), default="deterministic", nullable=False)
    centroid: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    representative_interaction_ids: Mapped[list[Any]] = mapped_column(
        JSONType, default=list, nullable=False
    )
    first_pipeline_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )


class InteractionClassification(Base):
    """Auditable classifier outputs for one sampled interaction."""

    __tablename__ = "interaction_classifications"
    __table_args__ = (
        UniqueConstraint("interaction_id", name="uq_interaction_classifications_interaction_id"),
        Index("ix_interaction_classifications_cluster_difficulty", "cluster_id", "difficulty"),
    )

    id: Mapped[str] = _pk()
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interaction_logs.id", ondelete="CASCADE"), nullable=False
    )
    pipeline_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cluster_id: Mapped[str | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True
    )
    cluster_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_outlier: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    outlier_reasons: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    difficulty_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    difficulty_signals: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    helpfulness: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[int] = mapped_column(Integer, nullable=False)
    completeness: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    judge_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    judge_rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Prompt embedding (used for clustering and dedup) and response embedding.
    embedding: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    response_embedding: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(120), nullable=False)
    classifier_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _created()

    interaction: Mapped[InteractionLog] = relationship()
    cluster: Mapped[Cluster | None] = relationship()


class EvalCandidate(Base):
    """An interaction proposed as an evaluation case, with proposed labels."""

    __tablename__ = "eval_candidates"
    __table_args__ = (UniqueConstraint("interaction_id", name="uq_eval_candidates_interaction_id"),)

    id: Mapped[str] = _pk()
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interaction_logs.id", ondelete="CASCADE"), nullable=False
    )
    classification_id: Mapped[str] = mapped_column(
        ForeignKey("interaction_classifications.id", ondelete="CASCADE"), nullable=False
    )
    pipeline_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cluster_id: Mapped[str | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    case_type: Mapped[str] = mapped_column(String(16), nullable=False)
    selection_reasons: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    priority: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    proposed_labels: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    label_passes: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    conflicts: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    max_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    duplicate_of_test_case_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )

    interaction: Mapped[InteractionLog] = relationship()
    classification: Mapped[InteractionClassification] = relationship()
    cluster: Mapped[Cluster | None] = relationship()


class DatasetVersion(Base):
    """Monotonic dataset version, bumped whenever the active dataset changes."""

    __tablename__ = "dataset_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _created()


class EvalTestCase(Base):
    """A curated evaluation case in the dataset."""

    __tablename__ = "eval_test_cases"
    __table_args__ = (
        Index("ix_eval_test_cases_category_difficulty", "category", "difficulty"),
        Index("ix_eval_test_cases_status_created", "status", "created_at"),
    )

    id: Mapped[str] = _pk()
    candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("eval_candidates.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    source_interaction_id: Mapped[str | None] = mapped_column(
        ForeignKey("interaction_logs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cluster_id: Mapped[str | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    added_in_version: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    application_feature: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    production_response: Mapped[str] = mapped_column(Text, default="", nullable=False)
    category: Mapped[str] = mapped_column(String(160), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    case_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    embedding: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )

    labels: Mapped[list[TestCaseLabel]] = relationship(
        back_populates="test_case", order_by="TestCaseLabel.version", cascade="all, delete-orphan"
    )

    @property
    def current_label(self) -> TestCaseLabel | None:
        """Return the label version currently in force."""
        for label in reversed(self.labels):
            if label.is_current:
                return label
        return None


class TestCaseLabel(Base):
    """Versioned labels of a test case; edits create a new version."""

    __test__ = False  # not a pytest class
    __tablename__ = "test_case_labels"
    __table_args__ = (
        UniqueConstraint("test_case_id", "version", name="uq_test_case_labels_case_version"),
    )

    id: Mapped[str] = _pk()
    test_case_id: Mapped[str] = mapped_column(
        ForeignKey("eval_test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    question_type: Mapped[str] = mapped_column(String(16), nullable=False)
    expected_quality_score: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_behavior: Mapped[str] = mapped_column(String(32), nullable=False)
    required_assertions: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    forbidden_assertions: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rubric: Mapped[list[Any] | None] = mapped_column(JSONType, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    labeling_model: Mapped[str] = mapped_column(String(120), nullable=False)
    labeling_passes: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_by: Mapped[str] = mapped_column(String(120), default="auto", nullable=False)
    created_at: Mapped[datetime] = _created()

    test_case: Mapped[EvalTestCase] = relationship(back_populates="labels")


class ReviewQueueItem(Base):
    """A low-confidence or conflicting candidate awaiting human review."""

    __tablename__ = "review_queue_items"
    __table_args__ = (Index("ix_review_queue_items_status_priority", "status", "priority"),)

    id: Mapped[str] = _pk()
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("eval_candidates.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    reason: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    priority: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = _created()
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    candidate: Mapped[EvalCandidate] = relationship()
    decisions: Mapped[list[ReviewDecision]] = relationship(
        back_populates="item", order_by="ReviewDecision.created_at", cascade="all, delete-orphan"
    )


class ReviewDecision(Base):
    """One reviewer action on a review-queue item (append-only)."""

    __tablename__ = "review_decisions"
    __table_args__ = (Index("ix_review_decisions_reviewer_created", "reviewer", "created_at"),)

    id: Mapped[str] = _pk()
    review_item_id: Mapped[str] = mapped_column(
        ForeignKey("review_queue_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer: Mapped[str] = mapped_column(String(120), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    edited_labels: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # False for second opinions recorded after the item was already resolved.
    applied: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _created()

    item: Mapped[ReviewQueueItem] = relationship(back_populates="decisions")


class EvaluationRun(Base):
    """One evaluation of a model endpoint against a dataset version."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (Index("ix_evaluation_runs_group_started", "comparison_group", "started_at"),)

    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    comparison_group: Mapped[str] = mapped_column(String(120), default="default", nullable=False)
    endpoint_config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    grading_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    code_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    runtime_s: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    """Raw output and scores for one test case in one evaluation run."""

    __tablename__ = "evaluation_results"
    __table_args__ = (
        UniqueConstraint("run_id", "test_case_id", name="uq_evaluation_results_run_case"),
        Index("ix_evaluation_results_run_passed", "run_id", "passed"),
    )

    id: Mapped[str] = _pk()
    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    test_case_id: Mapped[str] = mapped_column(
        ForeignKey("eval_test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label_id: Mapped[str | None] = mapped_column(
        ForeignKey("test_case_labels.id", ondelete="SET NULL"), nullable=True
    )
    category: Mapped[str] = mapped_column(String(160), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_response: Mapped[str] = mapped_column(Text, default="", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    detected_behavior: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    metric_scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    failure_reasons: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    created_at: Mapped[datetime] = _created()

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
    test_case: Mapped[EvalTestCase] = relationship()


class RegressionReport(Base):
    """Comparison of an evaluation run with its previous comparable run."""

    __tablename__ = "regression_reports"

    id: Mapped[str] = _pk()
    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    baseline_run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    has_regression: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    new_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    new_passes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = _created()


class AuditEvent(Base):
    """Append-only audit trail of state-changing actions."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[str] = _pk()
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
