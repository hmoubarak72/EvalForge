"""Structured JSON / JSON Lines log adapter."""

from __future__ import annotations

import glob
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
from evalforge.domain.schemas import LLMInteraction

# Canonical field -> accepted aliases in upstream structured logs.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "request_id", "log_id", "event_id"),
    "timestamp": ("timestamp", "ts", "time", "created_at"),
    "application_feature": ("application_feature", "feature", "app_feature", "route"),
    "model_name": ("model_name", "model"),
    "system_prompt": ("system_prompt", "system"),
    "user_prompt": ("user_prompt", "prompt", "input", "question"),
    "raw_response": ("raw_response", "response", "completion", "output"),
    "latency_ms": ("latency_ms", "duration_ms", "latency"),
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "error_status": ("error_status", "error", "error_code"),
    "user_feedback": ("user_feedback", "feedback"),
    "retry_count": ("retry_count", "retries"),
    "content_filter_triggered": ("content_filter_triggered", "content_filter", "filtered"),
    "confidence_score": ("confidence_score", "confidence"),
    "metadata": ("metadata", "meta", "attributes"),
}

_KNOWN_KEYS = {alias for aliases in FIELD_ALIASES.values() for alias in aliases}


def normalize_mapping(raw: dict[str, Any]) -> LLMInteraction:
    """Map a loosely structured dict onto the canonical schema.

    Unknown top-level keys are preserved inside ``metadata`` so no upstream
    signal is silently dropped.
    """
    data: dict[str, Any] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in raw and raw[alias] is not None:
                data[canonical] = raw[alias]
                break
    if "timestamp" not in data:
        raise ValueError("missing required field: timestamp")
    data["timestamp"] = parse_timestamp(data["timestamp"])
    if "id" in data:
        data["id"] = str(data["id"])
    metadata = dict(data.get("metadata") or {})
    for key, value in raw.items():
        if key not in _KNOWN_KEYS:
            metadata.setdefault(key, value)
    data["metadata"] = metadata
    return LLMInteraction.model_validate(data)


class JsonLogAdapter(LogSourceAdapter):
    """Reads ``.jsonl`` (one object per line) or ``.json`` (array) files.

    Config:
        ``path``: file path or glob pattern.
    """

    kind = "json_logs"

    def read(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> Iterator[RawRecord]:
        pattern = self.config.get("path")
        if not pattern:
            raise SourceError(f"source '{self.name}': config.path is required")
        paths = sorted(glob.glob(str(pattern))) or (
            [str(pattern)] if Path(str(pattern)).exists() else []
        )
        if not paths:
            raise SourceError(f"source '{self.name}': no files match {pattern!r}")
        for path in paths:
            yield from self._read_file(Path(path), since, until)

    def _read_file(
        self, path: Path, since: datetime | None, until: datetime | None
    ) -> Iterator[RawRecord]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SourceError(f"source '{self.name}': cannot read {path}: {exc}") from exc
        for raw in self._iter_objects(text):
            if isinstance(raw, RawRecord):
                yield raw
                continue
            try:
                interaction = normalize_mapping(raw)
            except (ValidationError, ValueError, TypeError) as exc:
                yield RawRecord(None, error=_short_error(exc), raw_excerpt=excerpt(raw))
                continue
            if self.in_window(interaction.timestamp, since, until):
                yield RawRecord(interaction)

    @staticmethod
    def _iter_objects(text: str) -> Iterator[dict[str, Any] | RawRecord]:
        stripped = text.lstrip()
        if stripped.startswith("["):
            try:
                items = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield RawRecord(None, error=f"invalid JSON array: {exc.msg}")
                return
            for item in items:
                if isinstance(item, dict):
                    yield item
                else:
                    yield RawRecord(
                        None, error="record is not an object", raw_excerpt=excerpt(item)
                    )
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                yield RawRecord(None, error=f"invalid JSON: {exc.msg}", raw_excerpt=excerpt(line))
                continue
            if isinstance(item, dict):
                yield item
            else:
                yield RawRecord(None, error="record is not an object", raw_excerpt=excerpt(line))


def _short_error(exc: Exception) -> str:
    """Compact validation message without echoing raw input values."""
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
        ]
        return "; ".join(parts)
    return str(exc)
