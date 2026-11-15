"""Audit-event repository (append-only)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db.models import AuditEvent
from evalforge.repositories.base import Page, paginate
from evalforge.telemetry import current_correlation_id


class AuditRepository:
    """Records and lists audit events."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        actor: str = "system",
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append an audit event in the current transaction."""
        event = AuditEvent(
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload or {},
            correlation_id=current_correlation_id(),
        )
        self._session.add(event)
        return event

    def list_events(
        self, entity_type: str | None = None, limit: int = 50, offset: int = 0
    ) -> Page[AuditEvent]:
        """List audit events, newest first."""
        statement = select(AuditEvent).order_by(AuditEvent.timestamp.desc())
        if entity_type:
            statement = statement.where(AuditEvent.entity_type == entity_type)
        return paginate(self._session, statement, limit, offset)
