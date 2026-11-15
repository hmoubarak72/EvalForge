"""Shared repository helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of results plus the total number of matching rows."""

    items: list[T]
    total: int
    limit: int
    offset: int


def paginate(session: Session, statement: Select[Any], limit: int, offset: int) -> Page[Any]:
    """Execute a select with LIMIT/OFFSET and a matching COUNT(*)."""
    total = session.scalar(select(func.count()).select_from(statement.order_by(None).subquery()))
    items = list(session.scalars(statement.limit(limit).offset(offset)))
    return Page(items=items, total=int(total or 0), limit=limit, offset=offset)
