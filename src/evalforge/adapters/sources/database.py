"""Direct database-read adapter.

Reads rows from an existing application table (any SQLAlchemy-supported
database) and maps columns to the canonical schema.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from evalforge.adapters.sources.base import LogSourceAdapter, RawRecord, SourceError, excerpt
from evalforge.adapters.sources.json_logs import _short_error, normalize_mapping

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class DatabaseLogAdapter(LogSourceAdapter):
    """Reads LLM call rows from a table.

    Config:
        ``url``: SQLAlchemy database URL of the upstream system.
        ``table``: table (or ``schema.table``) to read.
        ``column_map``: optional ``{canonical_field: column_name}`` overrides.
        ``batch_size``: rows fetched per round trip (default 500).
    """

    kind = "database"

    def read(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> Iterator[RawRecord]:
        url = self.config.get("url")
        table = self.config.get("table")
        if not url or not table:
            raise SourceError(f"source '{self.name}': config.url and config.table are required")
        if not _IDENTIFIER.match(str(table)):
            raise SourceError(f"source '{self.name}': invalid table name {table!r}")
        column_map: dict[str, str] = dict(self.config.get("column_map") or {})
        reverse_map = {column: canonical for canonical, column in column_map.items()}
        batch_size = int(self.config.get("batch_size", 500))

        engine = create_engine(str(url))
        try:
            with engine.connect() as connection:
                result = connection.execution_options(stream_results=True).execute(
                    text(f"SELECT * FROM {table}")
                )
                while True:
                    rows = result.mappings().fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        yield from self._normalize(dict(row), reverse_map, since, until)
        except SQLAlchemyError as exc:
            raise SourceError(f"source '{self.name}': database read failed: {exc}") from exc
        finally:
            engine.dispose()

    def _normalize(
        self,
        row: dict[str, Any],
        reverse_map: dict[str, str],
        since: datetime | None,
        until: datetime | None,
    ) -> Iterator[RawRecord]:
        mapped = {reverse_map.get(column, column): value for column, value in row.items()}
        metadata = mapped.get("metadata")
        if isinstance(metadata, str):
            try:
                mapped["metadata"] = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                mapped["metadata"] = {"raw_metadata": metadata}
        try:
            interaction = normalize_mapping(mapped)
        except (ValidationError, ValueError, TypeError) as exc:
            yield RawRecord(None, error=_short_error(exc), raw_excerpt=excerpt(row.get("id", "")))
            return
        if self.in_window(interaction.timestamp, since, until):
            yield RawRecord(interaction)
