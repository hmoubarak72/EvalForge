"""OpenTelemetry-style trace export adapter.

Reads OTLP/JSON exports (``resourceSpans -> scopeSpans -> spans``) and maps
spans that carry GenAI semantic-convention attributes (``gen_ai.*``) onto the
canonical schema. Spans without a prompt attribute are ignored (they are not
LLM calls), which is not counted as a rejection.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalforge.adapters.sources.base import (
    LogSourceAdapter,
    RawRecord,
    SourceError,
    excerpt,
    parse_timestamp,
)
from evalforge.adapters.sources.json_logs import _short_error
from evalforge.domain.schemas import LLMInteraction

# OTel attribute -> canonical field
ATTRIBUTE_MAP: dict[str, str] = {
    "gen_ai.request.model": "model_name",
    "gen_ai.response.model": "model_name",
    "gen_ai.system_prompt": "system_prompt",
    "gen_ai.prompt": "user_prompt",
    "gen_ai.completion": "raw_response",
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.prompt_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "gen_ai.usage.completion_tokens": "output_tokens",
    "app.feature": "application_feature",
    "app.user_feedback": "user_feedback",
    "app.retry_count": "retry_count",
    "app.content_filter_triggered": "content_filter_triggered",
    "app.confidence_score": "confidence_score",
    "error.type": "error_status",
}


def _attribute_value(value: dict[str, Any]) -> Any:
    """Unwrap an OTLP ``AnyValue`` object."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_attribute_value(v) for v in value["arrayValue"].get("values", [])]
    return None


def flatten_attributes(attributes: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Convert an OTLP attribute list to a plain dict."""
    return {a["key"]: _attribute_value(a.get("value", {})) for a in attributes or [] if "key" in a}


def span_to_interaction(span: dict[str, Any], resource: dict[str, Any]) -> LLMInteraction | None:
    """Normalize one span; returns None for spans that are not LLM calls."""
    attrs = flatten_attributes(span.get("attributes"))
    if "gen_ai.prompt" not in attrs:
        return None
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "trace_id": span.get("traceId"),
        "span_id": span.get("spanId"),
        "span_name": span.get("name"),
    }
    for key, value in attrs.items():
        canonical = ATTRIBUTE_MAP.get(key)
        if canonical is None:
            metadata[key] = value
        elif canonical not in data and value is not None:
            data[canonical] = value
    start = int(span["startTimeUnixNano"])
    end = int(span.get("endTimeUnixNano", start))
    data["timestamp"] = parse_timestamp(start)
    data["latency_ms"] = max(0, (end - start) // 1_000_000)
    data["id"] = f"{span.get('traceId', '')}:{span.get('spanId', '')}"
    data.setdefault(
        "application_feature", resource.get("service.name") or span.get("name") or "unknown"
    )
    status = span.get("status") or {}
    if "error_status" not in data and status.get("code") in (2, "STATUS_CODE_ERROR"):
        data["error_status"] = status.get("message") or "error"
    data["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    return LLMInteraction.model_validate(data)


class OtelTraceAdapter(LogSourceAdapter):
    """Reads an OTLP/JSON trace export file.

    Config:
        ``path``: path to the export (single JSON document).
    """

    kind = "otel_traces"

    def read(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> Iterator[RawRecord]:
        path_value = self.config.get("path")
        if not path_value:
            raise SourceError(f"source '{self.name}': config.path is required")
        path = Path(str(path_value))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SourceError(f"source '{self.name}': cannot read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SourceError(f"source '{self.name}': invalid OTLP JSON: {exc.msg}") from exc
        for resource_span in document.get("resourceSpans", []):
            resource = flatten_attributes(resource_span.get("resource", {}).get("attributes"))
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    try:
                        interaction = span_to_interaction(span, resource)
                    except (ValidationError, ValueError, TypeError, KeyError) as exc:
                        yield RawRecord(
                            None,
                            error=_short_error(exc),
                            raw_excerpt=excerpt(span.get("spanId", "")),
                        )
                        continue
                    if interaction is None:
                        continue
                    if self.in_window(interaction.timestamp, since, until):
                        yield RawRecord(interaction)
