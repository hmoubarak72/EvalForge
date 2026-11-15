"""Application configuration.

All settings are loaded from environment variables (prefix ``EVALFORGE_``) or a
``.env`` file via Pydantic Settings. Nested settings use a double underscore,
for example ``EVALFORGE_SAMPLING__SAMPLE_SIZE=80``.

Secrets are never stored in settings objects: model providers reference the
*name* of an environment variable (``api_key_env``) that is resolved only when a
client is constructed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SamplingStrategyName = Literal["uniform", "stratified", "signal_boosted"]
ProviderName = Literal["mock", "openai_compatible"]
GradingMode = Literal["deterministic", "llm", "hybrid"]
AuthMode = Literal["local_dev", "token"]


class SourceConfig(BaseModel):
    """Declarative definition of one log source."""

    name: str
    kind: Literal["json_logs", "otel_traces", "database"]
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RedactionRule(BaseModel):
    """A single regex redaction rule."""

    name: str
    pattern: str
    replacement: str


class RedactionSettings(BaseModel):
    """PII redaction configuration."""

    enabled: bool = True
    # Names of built-in rules to apply; see evalforge.services.redaction.BUILTIN_RULES.
    builtin_rules: list[str] = Field(
        default_factory=lambda: [
            "email",
            "credit_card",
            "ssn",
            "phone",
            "ip_address",
            "api_key",
            "iban",
        ]
    )
    custom_rules: list[RedactionRule] = Field(default_factory=list)
    # Optional lightweight named-entity detection (heuristic, offline).
    ner_enabled: bool = True
    # Metadata keys whose values are always dropped before persistence.
    drop_metadata_keys: list[str] = Field(
        default_factory=lambda: ["user_email", "user_name", "ip", "phone", "address"]
    )


class SamplingSettings(BaseModel):
    """Default sampling configuration; can be overridden per pipeline run."""

    strategy: SamplingStrategyName = "signal_boosted"
    sample_size: int = Field(default=60, ge=1)
    seed: int = 42
    exclude_previously_sampled: bool = True
    lookback_days: int | None = Field(default=None, ge=1)
    stratify_by: list[str] = Field(
        default_factory=lambda: [
            "application_feature",
            "model_name",
            "error_status",
            "user_feedback",
        ]
    )
    # Signal-boost weights (added to a base weight of 1.0).
    boost_negative_feedback: float = 4.0
    boost_retry: float = 3.0
    boost_high_latency: float = 2.0
    boost_low_confidence: float = 3.0
    boost_content_filter: float = 4.0
    boost_unusual_length: float = 2.0
    boost_error: float = 3.0
    boost_underrepresented_cluster: float = 2.0
    high_latency_ms: int = 8000
    low_confidence_threshold: float = 0.5


class ModelProviderSettings(BaseModel):
    """Connection settings for one OpenAI-compatible (or mock) model role."""

    provider: ProviderName = "mock"
    base_url: str = "https://api.openai.com/v1"
    model: str = "mock-model"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0
    max_retries: int = 2

    def resolve_api_key(self) -> str | None:
        """Read the API key from the configured environment variable."""
        return os.environ.get(self.api_key_env)


class EmbeddingSettings(ModelProviderSettings):
    """Embedding model settings."""

    model: str = "mock-embedding-256"
    dimensions: int = 256
    batch_size: int = 64


class ClusteringSettings(BaseModel):
    """HDBSCAN and cluster-matching parameters."""

    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int | None = Field(default=2, ge=1)
    cluster_selection_epsilon: float = 0.0
    # "leaf" yields small homogeneous topics; "eom" favours fewer, larger clusters.
    cluster_selection_method: Literal["leaf", "eom"] = "leaf"
    # Reduce embeddings with PCA before clustering (0 disables reduction).
    reduce_dimensions: int = Field(default=15, ge=0)
    # How many previously embedded interactions to cluster alongside new ones.
    context_window: int = Field(default=2000, ge=0)
    # A new cluster is merged into an existing one above this centroid similarity.
    match_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    # A discovered group only becomes a new cluster when its members are, on
    # average, at least this similar to the group centroid (rejects groups that
    # merge unrelated topics). Tune per embedding model.
    min_cohesion: float = Field(default=0.55, ge=0.0, le=1.0)
    # A point joins an existing cluster only above this similarity to its centroid.
    member_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # A noise point is attached to an existing cluster above this similarity.
    assign_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    seed: int = 42


class ClassificationSettings(BaseModel):
    """Outlier and quality-judge settings."""

    low_confidence_threshold: float = 0.5
    length_zscore_threshold: float = 3.0
    short_response_chars: int = 25
    long_response_chars: int = 4000
    rephrase_window_minutes: int = 10
    rephrase_similarity: float = 0.55
    classifier_version: str = "rules-v1"


class LabelingSettings(ModelProviderSettings):
    """Auto-labeling settings."""

    model: str = "mock-labeler"
    passes: int = Field(default=3, ge=1, le=9)
    pass_temperature: float = 0.4
    auto_accept_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    max_candidates_per_run: int = Field(default=40, ge=1)
    seed: int = 42


class DedupSettings(BaseModel):
    """Near-duplicate detection against the existing dataset."""

    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)


class EvaluationSettings(ModelProviderSettings):
    """Default evaluation endpoint and grading configuration."""

    model: str = "mock-model-v1"
    concurrency: int = Field(default=8, ge=1)
    retry_backoff_s: float = 0.5
    grading_mode: GradingMode = "deterministic"
    # A case passes when score >= expected_quality_score - tolerance.
    pass_score_tolerance: float = 1.0
    # USD per 1K tokens, used for cost estimates only.
    cost_per_1k_input_tokens: float = 0.0005
    cost_per_1k_output_tokens: float = 0.0015
    comparison_group: str = "default"


class RegressionSettings(BaseModel):
    """Thresholds for regression detection."""

    score_change_threshold: float = 1.0
    category_shift_threshold: float = 0.10
    difficulty_shift_threshold: float = 0.10
    overall_pass_rate_drop_threshold: float = 0.05
    min_group_size: int = 3


class SchedulerSettings(BaseModel):
    """Nightly pipeline schedule (APScheduler cron trigger)."""

    enabled: bool = False
    cron: str = "0 2 * * *"
    timezone: str = "UTC"
    run_evaluation: bool = True


class AuthSettings(BaseModel):
    """Write-protection settings shared by the API and the dashboard."""

    mode: AuthMode = "local_dev"
    # Name of the env var holding the shared token when mode == "token".
    token_env: str = "EVALFORGE_AUTH_TOKEN"

    def resolve_token(self) -> str | None:
        """Read the shared write token from the environment."""
        return os.environ.get(self.token_env)


class Settings(BaseSettings):
    """Top-level settings object, constructed once and injected everywhere."""

    model_config = SettingsConfigDict(
        env_prefix="EVALFORGE_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = "development"
    database_url: str = "sqlite:///./evalforge.db"
    # Feature flags
    offline_mode: bool = True
    demo_mode: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    sources: list[SourceConfig] = Field(default_factory=list)
    redaction: RedactionSettings = Field(default_factory=RedactionSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    judge: ModelProviderSettings = Field(
        default_factory=lambda: ModelProviderSettings(model="mock-judge")
    )
    labeling: LabelingSettings = Field(default_factory=LabelingSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    regression: RegressionSettings = Field(default_factory=RegressionSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    def provider_for(self, role: ModelProviderSettings) -> ProviderName:
        """Return the effective provider for a role, honouring offline mode."""
        return "mock" if self.offline_mode else role.provider


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached, read-only)."""
    return Settings()
