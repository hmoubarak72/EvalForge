"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from evalforge.domain.schemas import EndpointConfig, LabelEdit

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for response models built from ORM rows."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class PageOut(BaseModel, Generic[T]):
    """Paginated list response."""

    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorBody(BaseModel):
    """Structured error payload."""

    code: str
    message: str
    correlation_id: str | None = None
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Envelope for every non-2xx response."""

    error: ErrorBody


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessOut(BaseModel):
    status: Literal["ready", "not_ready"]
    database: bool
    offline_mode: bool
    scheduler_enabled: bool


# -- sources / ingestion -----------------------------------------------------
class SourceCreate(BaseModel):
    """Create (or update by name) a log source."""

    name: str = Field(min_length=1, max_length=120)
    kind: Literal["json_logs", "otel_traces", "database"]
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SourceOut(ORMModel):
    id: str
    name: str
    kind: str
    config: dict[str, Any]
    enabled: bool
    cursor_timestamp: datetime | None
    created_at: datetime


class IngestionRunOut(ORMModel):
    id: str
    source_id: str
    pipeline_run_id: str | None
    status: str
    records_received: int
    records_accepted: int
    records_rejected: int
    duplicates: int
    redactions: int
    failures: int
    error_details: list[Any]
    started_at: datetime
    finished_at: datetime | None


# -- pipeline ------------------------------------------------------------------
StageName = Literal["ingestion", "sampling", "classification", "labeling", "deduplication"]


class PipelineRunCreate(BaseModel):
    """Trigger a pipeline run; omit ``stages`` to run the full pipeline."""

    stages: list[StageName] | None = None
    strategy: Literal["uniform", "stratified", "signal_boosted"] | None = None
    sample_size: int | None = Field(default=None, ge=1, le=100000)
    seed: int | None = None
    source_ids: list[str] | None = None

    def options(self) -> dict[str, Any]:
        return self.model_dump(exclude={"stages"}, exclude_none=True)


class StageTrigger(BaseModel):
    """Options when triggering a single stage."""

    strategy: Literal["uniform", "stratified", "signal_boosted"] | None = None
    sample_size: int | None = Field(default=None, ge=1, le=100000)
    seed: int | None = None
    source_ids: list[str] | None = None


class PipelineRunOut(ORMModel):
    id: str
    status: str
    trigger: str
    stages: list[Any]
    config: dict[str, Any]
    stage_results: dict[str, Any]
    error: str | None
    correlation_id: str | None
    started_at: datetime
    finished_at: datetime | None


# -- interactions / clusters -----------------------------------------------------
class ClassificationOut(ORMModel):
    id: str
    interaction_id: str
    pipeline_run_id: str | None
    cluster_id: str | None
    cluster_probability: float | None
    is_outlier: bool
    outlier_reasons: list[Any]
    difficulty: str
    difficulty_confidence: float
    difficulty_signals: list[Any]
    helpfulness: int
    accuracy: int
    completeness: int
    quality_score: float
    judge_confidence: float
    judge_rationale: str
    embedding_model: str
    judge_model: str
    classifier_version: str
    created_at: datetime


class InteractionOut(ORMModel):
    id: str
    external_id: str | None
    timestamp: datetime
    application_feature: str
    model_name: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    error_status: str | None
    user_feedback: str | None
    retry_count: int
    response_length: int
    content_filter_triggered: bool
    confidence_score: float | None
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    redaction_status: str
    ingested_at: datetime


class InteractionDetailOut(InteractionOut):
    classification: ClassificationOut | None = None


class ClusterOut(ORMModel):
    id: str
    name: str
    description: str
    keywords: list[Any]
    naming_method: str
    size: int
    representative_interaction_ids: list[Any]
    created_at: datetime
    updated_at: datetime


# -- candidates / test cases -------------------------------------------------------
class CandidateOut(ORMModel):
    id: str
    interaction_id: str
    classification_id: str
    pipeline_run_id: str | None
    cluster_id: str | None
    status: str
    case_type: str
    selection_reasons: list[Any]
    priority: float
    proposed_labels: dict[str, Any]
    confidence: float | None
    agreement: float | None
    conflicts: list[Any]
    max_similarity: float | None
    duplicate_of_test_case_id: str | None
    created_at: datetime
    updated_at: datetime


class CandidateDetailOut(CandidateOut):
    label_passes: list[Any]
    interaction: InteractionOut
    classification: ClassificationOut


class LabelOut(ORMModel):
    id: str
    version: int
    is_current: bool
    question_type: str
    expected_quality_score: int
    expected_behavior: str
    required_assertions: list[Any]
    forbidden_assertions: list[Any]
    reference_answer: str | None
    reference_verified: bool | None
    rubric: list[Any] | None
    confidence: float
    agreement: float | None
    labeling_model: str
    labeling_passes: int
    created_by: str
    created_at: datetime


class TestCaseOut(ORMModel):
    __test__ = False

    id: str
    candidate_id: str | None
    source_interaction_id: str | None
    cluster_id: str | None
    added_in_version: int
    application_feature: str
    system_prompt: str
    user_prompt: str
    production_response: str
    category: str
    difficulty: str
    case_type: str
    source: str
    status: str
    notes: str
    source_timestamp: datetime | None
    created_at: datetime
    updated_at: datetime
    current_label: LabelOut | None


class TestCaseDetailOut(TestCaseOut):
    __test__ = False

    labels: list[LabelOut]


class LabelEditRequest(BaseModel):
    """Edit labels (review queue item or existing test case)."""

    labels: LabelEdit
    notes: str = ""


class NoteRequest(BaseModel):
    notes: str = Field(min_length=1, max_length=4000)


class DecisionRequest(BaseModel):
    notes: str = Field(default="", max_length=4000)


class RetireRequest(BaseModel):
    reason: str = Field(default="", max_length=4000)


# -- review queue --------------------------------------------------------------------
class ReviewDecisionOut(ORMModel):
    id: str
    review_item_id: str
    reviewer: str
    action: str
    edited_labels: dict[str, Any] | None
    notes: str
    applied: bool
    created_at: datetime


class ReviewItemOut(ORMModel):
    id: str
    candidate_id: str
    status: str
    reason: str
    priority: float
    created_at: datetime
    resolved_at: datetime | None
    candidate: CandidateOut


class SimilarCaseOut(BaseModel):
    similarity: float
    test_case: TestCaseOut


class ReviewItemDetailOut(ReviewItemOut):
    candidate: CandidateDetailOut
    decisions: list[ReviewDecisionOut]
    similar_cases: list[SimilarCaseOut] = Field(default_factory=list)


class ReviewOutcomeOut(BaseModel):
    """Result of a reviewer action."""

    applied: bool = Field(
        description="False when the item was already resolved and the decision was stored "
        "as a second opinion only."
    )
    item: ReviewItemOut
    decision: ReviewDecisionOut
    test_case: TestCaseOut | None = None


# -- evaluation -----------------------------------------------------------------------
class EvaluationCreate(BaseModel):
    """Run the current dataset against a model endpoint."""

    endpoint: EndpointConfig | None = Field(
        default=None, description="Defaults to the configured EVALFORGE_EVALUATION__* endpoint."
    )
    name: str = Field(default="", max_length=160)
    grading_mode: Literal["deterministic", "llm", "hybrid"] | None = None
    comparison_group: str | None = Field(default=None, max_length=120)
    baseline_run_id: str | None = None


class EvaluationRunOut(ORMModel):
    id: str
    name: str
    status: str
    comparison_group: str
    endpoint_config: dict[str, Any]
    model_name: str
    grading_mode: str
    dataset_version: int
    dataset_size: int
    code_version: str
    pass_rate: float | None
    average_quality_score: float | None
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    runtime_s: float
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class EvaluationResultOut(ORMModel):
    id: str
    run_id: str
    test_case_id: str
    category: str
    difficulty: str
    raw_response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    retries: int
    error: str | None
    passed: bool
    quality_score: float
    detected_behavior: str
    metric_scores: dict[str, Any]
    failure_reasons: list[Any]


class EvaluationReportOut(BaseModel):
    run: EvaluationRunOut
    report: dict[str, Any]


class RegressionReportOut(ORMModel):
    id: str
    run_id: str
    baseline_run_id: str
    has_regression: bool
    new_failures: int
    new_passes: int
    report: dict[str, Any]
    created_at: datetime
