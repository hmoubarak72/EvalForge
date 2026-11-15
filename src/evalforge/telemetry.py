"""Structured JSON logging with correlation IDs.

Every pipeline run, API request and evaluation run binds a correlation ID via
:func:`bind_correlation_id`; structlog's contextvars integration then attaches
it to every log line emitted while that work is in progress.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog and the standard library root logger."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_level_number(level)),
        logger_factory=lambda *_: _StderrLogger(),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(
        level=_level_number(level), stream=sys.stderr, format="%(message)s", force=True
    )


class _StderrLogger:
    """Writes each rendered event to the *current* ``sys.stderr``.

    Resolving the stream at call time (instead of capturing it once) keeps
    logging working when stderr is redirected or replaced after configuration.
    """

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log = debug = info = warn = warning = error = critical = exception = fatal = msg


def _level_number(level: str) -> int:
    number = logging.getLevelName(level.upper())
    return number if isinstance(number, int) else logging.INFO


def get_logger(name: str) -> Any:
    """Return a lazy structlog logger tagged with a component name.

    The proxy resolves the active configuration on use, so module-level
    loggers created at import time still honour :func:`configure_logging`.
    """
    return structlog.get_logger(component=name)


def new_correlation_id() -> str:
    """Generate a fresh correlation ID."""
    return uuid.uuid4().hex


def current_correlation_id() -> str | None:
    """Return the correlation ID bound to the current context, if any."""
    value = structlog.contextvars.get_contextvars().get("correlation_id")
    return str(value) if value is not None else None


@contextmanager
def bind_correlation_id(correlation_id: str | None = None, **extra: Any) -> Iterator[str]:
    """Bind a correlation ID (and extra fields) for the duration of a block.

    If a correlation ID is already bound and none is passed, it is reused so
    nested stages share the ID of the enclosing pipeline run.
    """
    cid = correlation_id or current_correlation_id() or new_correlation_id()
    with structlog.contextvars.bound_contextvars(correlation_id=cid, **extra):
        yield cid
