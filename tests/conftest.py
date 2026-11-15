"""Shared fixtures. Nothing here (or in any test) calls an external API."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from evalforge.adapters.llm.base import ChatResult, Message
from evalforge.config import Settings
from evalforge.container import Container, build_container, register_source
from evalforge.db.models import InteractionLog
from evalforge.db.session import create_all, make_engine, session_scope
from evalforge.demo.generator import (
    generate_interactions,
    write_jsonl,
    write_otel,
    write_sqlite_table,
)

FIXED_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


class ScriptedChatClient:
    """Test double for an LLM: returns queued (or a default) canned response."""

    def __init__(self, responses: Sequence[str] | None = None, default: str = "{}") -> None:
        self.model_name = "scripted-model"
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], **_: Any) -> ChatResult:
        self.calls.append(list(messages))
        text = self._responses.pop(0) if self._responses else self._default
        return ChatResult(text=text, input_tokens=10, output_tokens=5, latency_ms=1)

    async def acomplete(self, messages: Sequence[Message], **kwargs: Any) -> ChatResult:
        return self.complete(messages, **kwargs)


def make_settings(**overrides: Any) -> Settings:
    """Offline settings on an in-memory SQLite database."""
    base: dict[str, Any] = {
        "database_url": "sqlite://",
        "offline_mode": True,
        "log_level": "ERROR",
        # Label every eligible candidate so one run yields a sizeable dataset.
        "labeling": {"max_candidates_per_run": 500},
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def make_container(**kwargs: Any) -> Container:
    settings = kwargs.pop("settings", None) or make_settings()
    engine = make_engine(settings.database_url)
    create_all(engine)
    return build_container(
        settings, engine=engine, clock=kwargs.pop("clock", lambda: FIXED_NOW), **kwargs
    )


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def container() -> Container:
    return make_container()


@pytest.fixture
def session(container: Container) -> Iterator[Session]:
    with session_scope(container.session_factory) as db:
        yield db


@pytest.fixture(scope="session")
def demo_records() -> list[dict[str, Any]]:
    """Realistic synthetic production logs (normal, ambiguous, adversarial,
    retries, low-confidence, PII, errors, content-filter events)."""
    return generate_interactions(count=320, seed=11, end=FIXED_NOW, days=14)


def write_demo_sources(records: list[dict[str, Any]], directory: Path) -> dict[str, Path]:
    paths = {
        "json": directory / "app.jsonl",
        "otel": directory / "traces.json",
        "db": directory / "legacy.db",
    }
    write_jsonl(records[:200], paths["json"])
    write_otel(records[200:270], paths["otel"])
    write_sqlite_table(records[270:], paths["db"])
    return paths


DB_COLUMN_MAP = {
    "id": "request_id",
    "timestamp": "created_at",
    "raw_response": "completion",
    "latency_ms": "duration_ms",
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "error_status": "error_code",
    "user_feedback": "feedback",
    "retry_count": "retries",
    "content_filter_triggered": "filtered",
    "confidence_score": "confidence",
    "metadata": "meta",
    "application_feature": "feature",
    "model_name": "model",
    "user_prompt": "prompt",
}


def register_demo_sources(container: Container, paths: dict[str, Path]) -> None:
    with session_scope(container.session_factory) as db:
        register_source(db, "json", "json_logs", {"path": paths["json"].as_posix()})
        register_source(db, "otel", "otel_traces", {"path": paths["otel"].as_posix()})
        register_source(
            db,
            "db",
            "database",
            {
                "url": f"sqlite:///{paths['db'].as_posix()}",
                "table": "llm_calls",
                "column_map": DB_COLUMN_MAP,
            },
        )


@pytest.fixture(scope="module")
def seeded_container(
    tmp_path_factory: pytest.TempPathFactory, demo_records: list[dict[str, Any]]
) -> Container:
    """A container whose database went through one full pipeline run."""
    container = make_container()
    paths = write_demo_sources(demo_records, tmp_path_factory.mktemp("sources"))
    register_demo_sources(container, paths)
    result = container.orchestrator.run(options={"sample_size": 200, "seed": 3})
    assert result.status == "completed", result.error
    return container


def make_interaction(**overrides: Any) -> InteractionLog:
    """Build an unsaved interaction row with sensible defaults."""
    values: dict[str, Any] = {
        "id": "i-1",
        "fingerprint": "fp-1",
        "timestamp": FIXED_NOW,
        "application_feature": "customer_support",
        "model_name": "gpt-4o-mini",
        "system_prompt": "You are Aria, the support assistant for Northwind Outfitters.",
        "user_prompt": "What is your refund policy for hiking boots?",
        "raw_response": "You can return hiking boots within 30 days of delivery for a full refund.",
        "latency_ms": 900,
        "input_tokens": 40,
        "output_tokens": 30,
        "error_status": None,
        "user_feedback": None,
        "retry_count": 0,
        "content_filter_triggered": False,
        "confidence_score": 0.9,
        "metadata_": {},
        "redaction_status": "clean",
    }
    values.update(overrides)
    values.setdefault("response_length", len(values["raw_response"]))
    return InteractionLog(**values)
