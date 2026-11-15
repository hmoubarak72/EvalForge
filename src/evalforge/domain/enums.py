"""Enumerations shared across the domain, persistence and API layers."""

from __future__ import annotations

from enum import StrEnum


class UserFeedback(StrEnum):
    """Explicit or implicit user feedback on a production response."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    EDIT = "edit"
    RETRY = "retry"


NEGATIVE_FEEDBACK = frozenset({UserFeedback.THUMBS_DOWN, UserFeedback.EDIT, UserFeedback.RETRY})


class RedactionStatus(StrEnum):
    """Outcome of PII redaction for a stored interaction."""

    CLEAN = "clean"  # redaction ran, nothing found
    REDACTED = "redacted"  # redaction ran and replaced at least one span
    SKIPPED = "skipped"  # redaction explicitly disabled by configuration


class SourceKind(StrEnum):
    """Supported log-source adapter kinds."""

    JSON_LOGS = "json_logs"
    OTEL_TRACES = "otel_traces"
    DATABASE = "database"


class RunStatus(StrEnum):
    """Lifecycle of ingestion, pipeline and evaluation runs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SamplingStrategy(StrEnum):
    """Available sampling strategies."""

    UNIFORM = "uniform"
    STRATIFIED = "stratified"
    SIGNAL_BOOSTED = "signal_boosted"


class Difficulty(StrEnum):
    """Difficulty label of an interaction / test case."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class ExpectedBehavior(StrEnum):
    """What a correct model should do for a test case."""

    ANSWER = "answer"
    REFUSE = "refuse"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"


class QuestionType(StrEnum):
    """Whether a case has a single golden answer or needs a rubric."""

    FACTUAL = "factual"
    SUBJECTIVE = "subjective"


class CaseType(StrEnum):
    """Positive cases guard good behaviour; negative cases guard known failures."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class CandidateStatus(StrEnum):
    """Lifecycle of an evaluation candidate."""

    PENDING = "pending"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_REVIEW = "needs_review"
    DUPLICATE = "duplicate"
    APPROVED = "approved"
    REJECTED = "rejected"


class TestCaseSource(StrEnum):
    """How a test case's labels were validated."""

    __test__ = False  # not a pytest class

    AUTO_LABELED = "auto_labeled"
    HUMAN_REVIEWED = "human_reviewed"


class TestCaseStatus(StrEnum):
    """Whether a test case is part of the active dataset."""

    __test__ = False

    ACTIVE = "active"
    RETIRED = "retired"


class ReviewStatus(StrEnum):
    """State of a review-queue item."""

    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class ReviewAction(StrEnum):
    """Actions a reviewer can record."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    NOTE = "note"


class OutlierReason(StrEnum):
    """Why an interaction was flagged as a valuable eval candidate."""

    NOVEL_PROMPT = "novel_prompt"
    LOW_CONFIDENCE = "low_confidence"
    IMMEDIATE_RETRY = "immediate_retry"
    REPHRASED_PROMPT = "rephrased_prompt"
    LONG_RESPONSE = "unusually_long_response"
    SHORT_RESPONSE = "unusually_short_response"
    CONTENT_FILTER = "content_filter_triggered"
    NEGATIVE_FEEDBACK = "negative_feedback"
