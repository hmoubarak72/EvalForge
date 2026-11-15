"""Log-source adapters and their registry.

To add a new source, subclass :class:`LogSourceAdapter`, set ``kind`` and call
:func:`register_adapter` (or add it to ``_REGISTRY`` below).
"""

from __future__ import annotations

from typing import Any

from evalforge.adapters.sources.base import LogSourceAdapter, RawRecord, SourceError
from evalforge.adapters.sources.database import DatabaseLogAdapter
from evalforge.adapters.sources.json_logs import JsonLogAdapter
from evalforge.adapters.sources.otel import OtelTraceAdapter

_REGISTRY: dict[str, type[LogSourceAdapter]] = {
    JsonLogAdapter.kind: JsonLogAdapter,
    OtelTraceAdapter.kind: OtelTraceAdapter,
    DatabaseLogAdapter.kind: DatabaseLogAdapter,
}


def register_adapter(adapter_cls: type[LogSourceAdapter]) -> None:
    """Register a custom adapter class under its ``kind``."""
    _REGISTRY[adapter_cls.kind] = adapter_cls


def supported_kinds() -> list[str]:
    """Return the registered adapter kinds."""
    return sorted(_REGISTRY)


def build_adapter(kind: str, name: str, config: dict[str, Any]) -> LogSourceAdapter:
    """Instantiate the adapter registered for ``kind``."""
    try:
        adapter_cls = _REGISTRY[kind]
    except KeyError as exc:
        raise SourceError(
            f"unknown source kind {kind!r}; supported: {', '.join(supported_kinds())}"
        ) from exc
    return adapter_cls(name=name, config=config)


__all__ = [
    "DatabaseLogAdapter",
    "JsonLogAdapter",
    "LogSourceAdapter",
    "OtelTraceAdapter",
    "RawRecord",
    "SourceError",
    "build_adapter",
    "register_adapter",
    "supported_kinds",
]
