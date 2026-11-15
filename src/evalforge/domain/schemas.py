"""Canonical domain schemas (Pydantic v2).

These models are the contract between adapters, services and persistence. They
perform validation of all external input before anything reaches the database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalforge.domain.enums import (
    Difficulty,
    ExpectedBehavior,
    QuestionType,
    RedactionStatus,
    UserFeedback,
)


class LLMInteraction(BaseModel):
    """Canonical, normalized record of one production LLM call."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    id: str | None = Field(default=None, description="ID assigned by the upstream system, if any")
    timestamp: datetime
    application_feature: str = Field(min_length=1, max_length=120)
    model_name: str = Field(min_length=1, max_length=120)
    system_prompt: str = ""
    user_prompt: str = Field(min_length=1)
    raw_response: str = ""
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    error_status: str | None = None
    user_feedback: UserFeedback | None = None
    retry_count: int = Field(default=0, ge=0)
    response_length: int = Field(default=0, ge=0)
    content_filter_triggered: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    redaction_status: RedactionStatus = RedactionStatus.SKIPPED
    ingested_at: datetime | None = None

    @field_validator("timestamp", "ingested_at")
    @classmethod
    def _ensure_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("error_status", mode="before")
    @classmethod
    def _blank_error_is_none(cls, value: Any) -> Any:
        if value in ("", "ok", "OK", "none", "None"):
            return None
        return value

    @model_validator(mode="after")
    def _derive_response_length(self) -> LLMInteraction:
        if self.response_length == 0 and self.raw_response:
            self.response_length = len(self.raw_response)
        return self


AssertionType = Literal[
    "contains",  # response contains the value (case-insensitive)
    "contains_any",  # response contains at least one of the values
    "regex",  # response matches the regular expression
    "min_length",  # response has at least N characters
    "max_length",  # response has at most N characters
    "concept",  # semantic concept; LLM-graded, keyword fallback offline
]


class Assertion(BaseModel):
    """A checkable statement about a model response."""

    model_config = ConfigDict(extra="ignore")

    type: AssertionType = "contains"
    value: str | int | list[str]
    description: str = ""

    def key(self) -> str:
        """Stable identity used when comparing labels between passes."""
        value = self.value
        if isinstance(value, list):
            value = "|".join(sorted(str(v).lower() for v in value))
        return f"{self.type}:{str(value).lower()}"


class RubricCriterion(BaseModel):
    """One criterion of an evaluation rubric."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    weight: float = Field(default=1.0, gt=0)
    keywords: list[str] = Field(default_factory=list)


class LabelSet(BaseModel):
    """The full set of labels attached to a candidate or test case."""

    model_config = ConfigDict(extra="ignore")

    question_type: QuestionType = QuestionType.FACTUAL
    expected_quality_score: int = Field(default=4, ge=1, le=5)
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    required_assertions: list[Assertion] = Field(default_factory=list)
    forbidden_assertions: list[Assertion] = Field(default_factory=list)
    reference_answer: str | None = None
    reference_verified: bool | None = None
    rubric: list[RubricCriterion] | None = None
    category: str = "uncategorized"
    difficulty: Difficulty = Difficulty.MODERATE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LabelEdit(BaseModel):
    """Partial label update submitted by a human reviewer."""

    model_config = ConfigDict(extra="forbid")

    question_type: QuestionType | None = None
    expected_quality_score: int | None = Field(default=None, ge=1, le=5)
    expected_behavior: ExpectedBehavior | None = None
    required_assertions: list[Assertion] | None = None
    forbidden_assertions: list[Assertion] | None = None
    reference_answer: str | None = None
    rubric: list[RubricCriterion] | None = None
    category: str | None = None
    difficulty: Difficulty | None = None

    def apply_to(self, labels: LabelSet) -> LabelSet:
        """Return a copy of ``labels`` with the edited fields replaced."""
        updates = self.model_dump(exclude_none=True)
        merged = labels.model_dump()
        merged.update(updates)
        return LabelSet.model_validate(merged)


class EndpointConfig(BaseModel):
    """Model endpoint under evaluation."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["mock", "openai_compatible"] = "mock"
    base_url: str = "https://api.openai.com/v1"
    model: str = "mock-model-v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=0.5, ge=0)
    concurrency: int = Field(default=8, ge=1, le=64)

    def public_dict(self) -> dict[str, Any]:
        """Serializable configuration; contains the env-var *name*, never a key."""
        return self.model_dump()
