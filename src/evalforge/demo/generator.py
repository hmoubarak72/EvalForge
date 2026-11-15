"""Deterministic generator of simulated production LLM logs.

Produces canonical-shaped dictionaries plus helpers that serialize them in the
three supported source formats (JSON Lines, OTLP/JSON trace export, and a
relational table), so the demo exercises every ingestion adapter.
"""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from evalforge.demo.corpus import REFUSAL_RESPONSE, Scenario, all_scenarios

_FIRST_NAMES = ("Priya", "Marcus", "Elena", "Tobias", "Aiko", "Lucas", "Fatima", "Noah")
_LAST_NAMES = ("Raman", "Oyelaran", "Castillo", "Lindqvist", "Tanaka", "Moreau", "Haddad", "Berg")
_REPHRASE_PREFIXES = (
    "That's not right. Let me rephrase: ",
    "That did not answer my question. Again: ",
    "Not helpful. To be precise: ",
)


def _fake_person(rng: random.Random) -> tuple[str, str, str]:
    first, last = rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)
    email = f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com"
    phone = f"+1 (555) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
    return f"{first} {last}", email, phone


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _base_record(
    scenario: Scenario,
    rng: random.Random,
    timestamp: datetime,
    index: int,
    response: str,
    session_id: str,
) -> dict[str, Any]:
    from evalforge.demo.corpus import FEATURE_MODELS

    latency = int(min(6000, max(180, rng.lognormvariate(7.0, 0.45))))
    record: dict[str, Any] = {
        "id": f"log-{index:05d}",
        "timestamp": timestamp.isoformat(),
        "application_feature": scenario.feature,
        "model_name": rng.choice(FEATURE_MODELS[scenario.feature]),
        "system_prompt": scenario.system_prompt,
        "user_prompt": scenario.prompt,
        "raw_response": response,
        "latency_ms": latency,
        "input_tokens": _tokens(scenario.system_prompt + scenario.prompt),
        "output_tokens": _tokens(response),
        "error_status": None,
        "user_feedback": None,
        "retry_count": 0,
        "content_filter_triggered": False,
        "confidence_score": None,
        "metadata": {"session_id": session_id, "app_version": rng.choice(["2.3.0", "2.3.1"])},
    }
    if scenario.source_document:
        record["metadata"]["source_document"] = scenario.source_document
    return record


def generate_interactions(
    count: int = 440,
    seed: int = 7,
    end: datetime | None = None,
    days: int = 14,
) -> list[dict[str, Any]]:
    """Generate ``count`` (or slightly more) simulated production log records.

    The traffic contains normal requests, ambiguous requests, prompt-injection
    attempts, retries after poor responses, negative feedback, high-latency
    calls, content-filter events, provider errors, unusual response lengths
    and PII (fake names, emails and phone numbers).
    """
    rng = random.Random(seed)
    end = (end or datetime.now(UTC)).astimezone(UTC)
    start = end - timedelta(days=days)
    scenarios = all_scenarios()
    weights = [
        0.7 if s.kind == "ambiguous" else 0.9 if s.kind == "injection" else 1.0 for s in scenarios
    ]
    records: list[dict[str, Any]] = []
    index = 0
    misrated_docqa = 0
    false_positive_filters = 0

    while len(records) < count:
        scenario = rng.choices(scenarios, weights=weights, k=1)[0]
        index += 1
        timestamp = start + timedelta(seconds=rng.uniform(600, days * 86400 - 600))
        session_id = f"sess-{rng.randrange(16**8):08x}"
        bad_probability = {"normal": 0.18, "ambiguous": 0.4, "injection": 0.22}[scenario.kind]
        is_bad = rng.random() < bad_probability
        response = scenario.bad_response if is_bad else scenario.good_response
        record = _base_record(scenario, rng, timestamp, index, response, session_id)

        if rng.random() > 0.15:
            record["confidence_score"] = round(
                rng.uniform(0.15, 0.48) if is_bad else rng.uniform(0.72, 0.98), 3
            )

        # PII in customer-support traffic (fake data, exercised by redaction).
        if (
            scenario.feature == "customer_support"
            and scenario.kind == "normal"
            and rng.random() < 0.35
        ):
            name, email, phone = _fake_person(rng)
            record["user_prompt"] += rng.choice(
                (
                    f" My name is {name}.",
                    f" Contact: {email}.",
                    f" Call me at {phone}.",
                    f" ({email}, {phone})",
                    f" My name is {name}, {email}.",
                )
            )
            record["metadata"]["user_email"] = email

        if is_bad:
            roll = rng.random()
            if (
                scenario.feature == "document_qa"
                and scenario.kind == "normal"
                and misrated_docqa < 3
            ):
                # A user wrongly applauds a hallucinated answer: only source
                # verification can catch this one.
                record["user_feedback"] = "thumbs_up"
                record["confidence_score"] = 0.81
                misrated_docqa += 1
            elif roll < 0.45:
                record["user_feedback"] = "thumbs_down"
            elif roll < 0.60:
                record["user_feedback"] = "edit"
            elif roll < 0.92:
                record["user_feedback"] = "retry"
        elif rng.random() < 0.3:
            record["user_feedback"] = "thumbs_up"

        if scenario.kind == "injection" and not is_bad and rng.random() < 0.6:
            record["content_filter_triggered"] = True
        elif scenario.kind == "normal" and false_positive_filters < 3 and rng.random() < 0.012:
            record["content_filter_triggered"] = True
            record["raw_response"] = "This request was blocked by the content filter."
            record["user_feedback"] = "thumbs_down"
            record["confidence_score"] = 0.2
            false_positive_filters += 1

        special = rng.random()
        if special < 0.06:
            record["latency_ms"] = rng.randint(9000, 21000)
        elif special < 0.09 and scenario.kind == "normal":
            record["error_status"] = rng.choice(["timeout", "rate_limited", "upstream_500"])
            record["raw_response"] = ""
            record["user_feedback"] = "retry"
            record["confidence_score"] = None
            record["latency_ms"] = rng.randint(15000, 30000)
        elif special < 0.115 and scenario.kind == "normal" and not is_bad:
            record["raw_response"] = (record["raw_response"] + " ") + " ".join(
                "To restate the same point in other words, the guidance above applies in general."
                for _ in range(70)
            )
            if rng.random() < 0.5:
                record["user_feedback"] = "thumbs_down"
        elif special < 0.135 and scenario.kind == "normal":
            record["raw_response"] = rng.choice(["Yes.", "No.", "It depends.", "OK."])
            record["user_feedback"] = record["user_feedback"] or "thumbs_down"
            record["confidence_score"] = 0.3

        record["output_tokens"] = _tokens(record["raw_response"])
        records.append(record)

        # Immediate retry with a rephrased prompt after a poor response.
        if record["user_feedback"] == "retry" and scenario.kind != "injection":
            index += 1
            retry_time = timestamp + timedelta(seconds=rng.randint(20, 95))
            if retry_time < end:
                retry = _base_record(
                    scenario, rng, retry_time, index, scenario.good_response, session_id
                )
                retry["user_prompt"] = rng.choice(_REPHRASE_PREFIXES) + scenario.prompt
                retry["retry_count"] = 1
                retry["confidence_score"] = round(rng.uniform(0.7, 0.95), 3)
                if rng.random() < 0.4:
                    retry["user_feedback"] = "thumbs_up"
                records.append(retry)

    records.sort(key=lambda r: r["timestamp"])
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSON Lines, using upstream-style field aliases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = dict(record)
            # Realistic upstream naming: exercised by the alias mapping.
            row["feature"] = row.pop("application_feature")
            row["model"] = row.pop("model_name")
            row["prompt"] = row.pop("user_prompt")
            row["response"] = row.pop("raw_response")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _otel_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def to_otel_export(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Serialize records as an OTLP/JSON trace export with GenAI attributes."""
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_feature.setdefault(record["application_feature"], []).append(record)
    resource_spans = []
    counter = itertools.count(1)
    for feature, items in sorted(by_feature.items()):
        spans = []
        for record in items:
            start = datetime.fromisoformat(record["timestamp"])
            start_ns = int(start.timestamp() * 1e9)
            attributes = {
                "gen_ai.request.model": record["model_name"],
                "gen_ai.system_prompt": record["system_prompt"],
                "gen_ai.prompt": record["user_prompt"],
                "gen_ai.completion": record["raw_response"],
                "gen_ai.usage.input_tokens": record["input_tokens"],
                "gen_ai.usage.output_tokens": record["output_tokens"],
                "app.feature": feature,
                "app.retry_count": record["retry_count"],
                "app.content_filter_triggered": record["content_filter_triggered"],
            }
            if record["user_feedback"]:
                attributes["app.user_feedback"] = record["user_feedback"]
            if record["confidence_score"] is not None:
                attributes["app.confidence_score"] = float(record["confidence_score"])
            if record["error_status"]:
                attributes["error.type"] = record["error_status"]
            for key, value in record["metadata"].items():
                attributes[key] = value
            spans.append(
                {
                    "traceId": f"{next(counter):032x}",
                    "spanId": f"{next(counter):016x}",
                    "name": "chat.completions",
                    "kind": 3,
                    "startTimeUnixNano": str(start_ns),
                    "endTimeUnixNano": str(start_ns + record["latency_ms"] * 1_000_000),
                    "attributes": [
                        {"key": k, "value": _otel_value(v)} for k, v in attributes.items()
                    ],
                    "status": {"code": 2, "message": record["error_status"]}
                    if record["error_status"]
                    else {"code": 1},
                }
            )
        # A non-LLM span that the adapter must ignore.
        spans.append(
            {
                "traceId": "0" * 32,
                "spanId": f"{len(spans):016x}",
                "name": "db.query",
                "startTimeUnixNano": "0",
                "endTimeUnixNano": "0",
                "attributes": [{"key": "db.system", "value": {"stringValue": "postgresql"}}],
            }
        )
        resource_spans.append(
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": feature}}]
                },
                "scopeSpans": [{"scope": {"name": "llm-gateway"}, "spans": spans}],
            }
        )
    return {"resourceSpans": resource_spans}


def write_otel(records: list[dict[str, Any]], path: Path) -> None:
    """Write an OTLP/JSON export file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_otel_export(records), ensure_ascii=False), encoding="utf-8")


def write_sqlite_table(records: list[dict[str, Any]], path: Path, table: str = "llm_calls") -> None:
    """Write records into a legacy-style relational table (direct DB source)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"""CREATE TABLE {table} (
                request_id TEXT PRIMARY KEY, created_at TEXT, feature TEXT, model TEXT,
                system_prompt TEXT, prompt TEXT, completion TEXT, duration_ms INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER, error_code TEXT,
                feedback TEXT, retries INTEGER, filtered INTEGER, confidence REAL, meta TEXT)"""
        )
        connection.executemany(
            f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["id"],
                    r["timestamp"],
                    r["application_feature"],
                    r["model_name"],
                    r["system_prompt"],
                    r["user_prompt"],
                    r["raw_response"],
                    r["latency_ms"],
                    r["input_tokens"],
                    r["output_tokens"],
                    r["error_status"],
                    r["user_feedback"],
                    r["retry_count"],
                    int(r["content_filter_triggered"]),
                    r["confidence_score"],
                    json.dumps(r["metadata"]),
                )
                for r in records
            ],
        )
        connection.commit()
    finally:
        connection.close()


__all__ = [
    "REFUSAL_RESPONSE",
    "generate_interactions",
    "to_otel_export",
    "write_jsonl",
    "write_otel",
    "write_sqlite_table",
]
