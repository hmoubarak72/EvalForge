"""Normalization adapters, redaction and the ingestion service."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from evalforge.adapters.sources import SourceError, build_adapter
from evalforge.adapters.sources.base import parse_timestamp
from evalforge.adapters.sources.json_logs import normalize_mapping
from evalforge.config import RedactionRule, RedactionSettings
from evalforge.container import Container, register_source
from evalforge.db.models import InteractionLog
from evalforge.db.session import session_scope
from evalforge.demo.generator import write_jsonl, write_otel, write_sqlite_table
from evalforge.domain.enums import RedactionStatus
from evalforge.domain.schemas import LLMInteraction
from evalforge.services.ingestion import fingerprint
from evalforge.services.redaction import Redactor
from tests.conftest import DB_COLUMN_MAP, FIXED_NOW

RAW = {
    "request_id": 42,
    "ts": "2026-06-01T10:00:00Z",
    "feature": "customer_support",
    "model": "gpt-4o-mini",
    "prompt": "Where is my order?",
    "response": "It ships tomorrow.",
    "duration_ms": 850,
    "prompt_tokens": 12,
    "completion_tokens": 6,
    "feedback": "thumbs_down",
    "session_id": "sess-1",
}


# -- normalization ---------------------------------------------------------------
def test_normalize_mapping_maps_aliases_and_keeps_unknown_keys() -> None:
    interaction = normalize_mapping(RAW)
    assert interaction.id == "42"
    assert interaction.application_feature == "customer_support"
    assert interaction.user_prompt == "Where is my order?"
    assert interaction.latency_ms == 850
    assert interaction.user_feedback == "thumbs_down"
    assert interaction.response_length == len("It ships tomorrow.")
    assert interaction.metadata["session_id"] == "sess-1"
    assert interaction.timestamp.tzinfo is not None


@pytest.mark.parametrize(
    "broken",
    [
        {**RAW, "ts": None},
        {**RAW, "prompt": ""},
        {**RAW, "duration_ms": -1},
        {**RAW, "feedback": "five_stars"},
        {**RAW, "confidence": 3.0},
    ],
)
def test_normalize_mapping_rejects_invalid_records(broken: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_mapping(broken)


def test_parse_timestamp_handles_epoch_units_and_iso() -> None:
    expected = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    seconds = expected.timestamp()
    assert parse_timestamp(seconds) == expected
    assert parse_timestamp(int(seconds * 1000)) == expected
    assert parse_timestamp(str(int(seconds * 1e9))) == expected
    assert parse_timestamp("2026-06-01T10:00:00Z") == expected
    assert parse_timestamp("2026-06-01T10:00:00").tzinfo is UTC


def test_json_adapter_reports_bad_lines_without_aborting(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    path.write_text(
        "\n".join([json.dumps(RAW), "{not json}", json.dumps({"ts": "x"}), json.dumps([1])]),
        encoding="utf-8",
    )
    records = list(build_adapter("json_logs", "s", {"path": str(path)}).read())
    assert [r.ok for r in records] == [True, False, False, False]
    assert all(r.error for r in records[1:])


def test_json_adapter_filters_by_time_window(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    rows = [{**RAW, "ts": f"2026-06-0{day}T10:00:00Z"} for day in (1, 2, 3)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    adapter = build_adapter("json_logs", "s", {"path": str(path)})
    window = list(
        adapter.read(
            since=datetime(2026, 6, 1, 12, tzinfo=UTC), until=datetime(2026, 6, 2, 12, tzinfo=UTC)
        )
    )
    assert len(window) == 1


def test_missing_source_raises_source_error(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        list(build_adapter("json_logs", "s", {"path": str(tmp_path / "nope.jsonl")}).read())
    with pytest.raises(SourceError):
        build_adapter("kafka", "s", {})
    with pytest.raises(SourceError):
        list(build_adapter("database", "s", {"url": "sqlite://", "table": "x; DROP"}).read())


def test_otel_adapter_maps_genai_spans_and_skips_other_spans(
    tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    path = tmp_path / "traces.json"
    subset = demo_records[:25]
    write_otel(subset, path)
    records = list(build_adapter("otel_traces", "otel", {"path": str(path)}).read())
    assert len(records) == len(subset)  # the db.query spans are ignored
    assert all(r.ok for r in records)
    mapped = {
        (r.interaction.timestamp, r.interaction.user_prompt): r.interaction
        for r in records
        if r.interaction
    }
    for original in subset:
        key = (datetime.fromisoformat(original["timestamp"]), original["user_prompt"])
        interaction = mapped[key]
        assert interaction.model_name == original["model_name"]
        assert interaction.application_feature == original["application_feature"]
        assert interaction.latency_ms == original["latency_ms"]
        assert interaction.user_feedback == original["user_feedback"]
        assert interaction.error_status == original["error_status"]
        assert interaction.content_filter_triggered == original["content_filter_triggered"]
        assert "trace_id" in interaction.metadata


def test_database_adapter_reads_rows_with_column_map(
    tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    path = tmp_path / "legacy.db"
    write_sqlite_table(demo_records[:15], path)
    adapter = build_adapter(
        "database",
        "db",
        {"url": f"sqlite:///{path.as_posix()}", "table": "llm_calls", "column_map": DB_COLUMN_MAP},
    )
    records = list(adapter.read())
    assert len(records) == 15 and all(r.ok for r in records)
    first = records[0].interaction
    assert first is not None and isinstance(first.metadata, dict)
    assert first.metadata.get("session_id")


# -- redaction --------------------------------------------------------------------
def test_redactor_replaces_pii_but_keeps_business_numbers() -> None:
    redactor = Redactor(RedactionSettings())
    text = (
        "My name is Priya Raman, email priya.raman7@example.com, phone +1 (555) 234-9876. "
        "Card 4111 1111 1111 1111, SSN 123-45-6789, key sk-abcdefghijklmnop1234, ip 10.2.3.4. "
        "Order NW-48213 cost $250,000 for 1,240 employees in 30 days."
    )
    redacted, count = redactor.redact_text(text)
    for secret in ("Priya", "example.com", "234-9876", "4111", "123-45-6789", "sk-abc", "10.2.3.4"):
        assert secret not in redacted
    for placeholder in (
        "[NAME]",
        "[EMAIL]",
        "[PHONE]",
        "[CREDIT_CARD]",
        "[SSN]",
        "[API_KEY]",
        "[IP_ADDRESS]",
    ):
        assert placeholder in redacted
    for kept in ("NW-48213", "$250,000", "1,240 employees", "30 days"):
        assert kept in redacted
    assert count == 7


def test_redactor_ignores_non_luhn_digit_runs() -> None:
    redacted, count = Redactor(RedactionSettings()).redact_text("tracking 1234 5678 9012 3456")
    assert "[CREDIT_CARD]" not in redacted and count == 0


def test_redactor_custom_rules_metadata_and_status() -> None:
    settings = RedactionSettings(
        custom_rules=[RedactionRule(name="order", pattern=r"NW-\d+", replacement="[ORDER]")]
    )
    interaction = LLMInteraction(
        timestamp=FIXED_NOW,
        application_feature="f",
        model_name="m",
        user_prompt="Order NW-1 for a@b.co",
        raw_response="ok",
        metadata={"user_email": "a@b.co", "nested": {"note": "call 555-234-9876"}, "n": 3},
    )
    result = Redactor(settings).redact_interaction(interaction)
    assert result.interaction.user_prompt == "Order [ORDER] for [EMAIL]"
    assert "user_email" not in result.interaction.metadata
    assert result.interaction.metadata["nested"]["note"] == "call [PHONE]"
    assert result.interaction.metadata["n"] == 3
    assert result.interaction.redaction_status is RedactionStatus.REDACTED
    assert result.redaction_count == 4

    clean = Redactor(settings).redact_interaction(
        interaction.model_copy(update={"user_prompt": "hello", "metadata": {}})
    )
    assert clean.interaction.redaction_status is RedactionStatus.CLEAN
    skipped = Redactor(RedactionSettings(enabled=False)).redact_interaction(interaction)
    assert skipped.interaction.redaction_status is RedactionStatus.SKIPPED
    with pytest.raises(ValueError):
        Redactor(RedactionSettings(builtin_rules=["nope"]))


# -- ingestion service ----------------------------------------------------------------
def test_fingerprint_is_stable_and_ignores_upstream_id() -> None:
    a = normalize_mapping(RAW)
    b = normalize_mapping({**RAW, "request_id": 99})
    c = normalize_mapping({**RAW, "prompt": "Different"})
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)


def test_ingestion_metrics_dedup_redaction_and_idempotency(
    container: Container, tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    subset = demo_records[:60]
    path = tmp_path / "app.jsonl"
    write_jsonl(subset + subset[:4], path)  # 4 in-file duplicates
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{broken\n")
        handle.write(json.dumps({"timestamp": subset[0]["timestamp"], "feature": "x"}) + "\n")
    with session_scope(container.session_factory) as db:
        source = register_source(db, "json", "json_logs", {"path": path.as_posix()})
        run = container.ingestion.ingest_source(db, source)
        assert run.status == "completed"
        assert run.records_received == 66
        assert run.records_accepted == 60
        assert run.records_rejected == 2
        assert run.duplicates == 4
        assert run.failures == 0
        assert run.redactions > 0
        assert len(run.error_details) == 2
        assert source.cursor_timestamp is not None

    with session_scope(container.session_factory) as db:
        rows = list(db.scalars(select(InteractionLog)))
        assert len(rows) == 60
        assert all("@example.com" not in r.user_prompt for r in rows)
        assert all("user_email" not in r.metadata_ for r in rows)
        assert any(r.redaction_status == "redacted" for r in rows)
        assert {r.redaction_status for r in rows} <= {"redacted", "clean"}
        # Second pass: the cursor window re-reads a little, nothing new is stored.
        source = db.merge(source)
        source.cursor_timestamp = None
        again = container.ingestion.ingest_source(db, source)
        assert again.records_accepted == 0 and again.duplicates == 64


def test_ingestion_respects_clock_and_cursor(
    tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    from tests.conftest import make_container

    midpoint = FIXED_NOW - timedelta(days=7)
    now = {"value": midpoint}
    container = make_container(clock=lambda: now["value"])
    path = tmp_path / "app.jsonl"
    write_jsonl(demo_records[:120], path)
    expected_first = sum(
        1 for r in demo_records[:120] if datetime.fromisoformat(r["timestamp"]) <= midpoint
    )
    with session_scope(container.session_factory) as db:
        source = register_source(db, "json", "json_logs", {"path": path.as_posix()})
        first = container.ingestion.ingest_source(db, source)
        assert first.records_accepted == expected_first > 0
        now["value"] = FIXED_NOW
        second = container.ingestion.ingest_source(db, source)
        assert first.records_accepted + second.records_accepted == 120


def test_ingestion_records_source_failure(container: Container, tmp_path: Path) -> None:
    with session_scope(container.session_factory) as db:
        source = register_source(db, "gone", "json_logs", {"path": str(tmp_path / "missing.jsonl")})
        run = container.ingestion.ingest_source(db, source)
        assert run.status == "failed" and run.failures == 1
        assert run.error_details[0]["stage"] == "source"
