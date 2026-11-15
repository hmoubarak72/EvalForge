"""Common interface for log-source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from evalforge.domain.schemas import LLMInteraction


class SourceError(Exception):
    """Raised when a source cannot be read at all (missing file, bad DSN, ...)."""


@dataclass(frozen=True)
class RawRecord:
    """Outcome of normalizing one raw log entry.

    Exactly one of ``interaction`` / ``error`` is set. Per-record failures are
    reported rather than raised so one bad line never aborts an ingestion run.
    """

    interaction: LLMInteraction | None
    error: str | None = None
    raw_excerpt: str = ""

    @property
    def ok(self) -> bool:
        return self.interaction is not None


class LogSourceAdapter(ABC):
    """Reads raw logs from one source and normalizes them to ``LLMInteraction``."""

    kind: str = "abstract"

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def read(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> Iterator[RawRecord]:
        """Yield normalized records with ``since < timestamp <= until``."""

    @staticmethod
    def in_window(timestamp: datetime, since: datetime | None, until: datetime | None) -> bool:
        """Return True when a timestamp falls inside the incremental window."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        if since is not None and timestamp <= since:
            return False
        return not (until is not None and timestamp > until)


def parse_timestamp(value: Any) -> datetime:
    """Parse ISO-8601 strings, epoch seconds/millis/nanos or datetimes to UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float) or (isinstance(value, str) and value.isdigit()):
        number = float(value)
        if number > 1e17:  # nanoseconds
            number /= 1e9
        elif number > 1e14:  # microseconds
            number /= 1e6
        elif number > 1e11:  # milliseconds
            number /= 1e3
        return datetime.fromtimestamp(number, tz=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise ValueError(f"unsupported timestamp value: {value!r}")


def excerpt(raw: Any, limit: int = 160) -> str:
    """Short, loggable excerpt of a raw record (never the full payload)."""
    text = str(raw)
    return text if len(text) <= limit else text[: limit - 3] + "..."
