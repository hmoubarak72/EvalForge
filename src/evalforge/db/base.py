"""Declarative base, portable column types and small helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite development mode).
JSONType = JSON().with_variant(JSONB(), "postgresql")

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetime that round-trips correctly on SQLite.

    PostgreSQL stores ``TIMESTAMP WITH TIME ZONE`` natively. SQLite drops the
    offset, so values are normalised to UTC on write and re-tagged on read.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base for all EvalForge tables."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}  # noqa: RUF012


def new_id() -> str:
    """Generate a UUID4 primary key as a string (portable across databases)."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Current UTC time, timezone-aware."""
    return datetime.now(UTC)
