"""Health and readiness checks."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from evalforge import __version__
from evalforge.api.deps import ContainerDep
from evalforge.api.schemas import HealthOut, ReadinessOut
from evalforge.telemetry import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness probe")
def health() -> HealthOut:
    """The process is up."""
    return HealthOut(version=__version__)


@router.get("/ready", response_model=ReadinessOut, summary="Readiness probe")
def ready(container: ContainerDep, response: Response) -> ReadinessOut:
    """The service can reach its database."""
    try:
        with container.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database = True
    except SQLAlchemyError as exc:
        log.error("ready.database_unreachable", error=type(exc).__name__)
        database = False
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessOut(
        status="ready" if database else "not_ready",
        database=database,
        offline_mode=container.settings.offline_mode,
        scheduler_enabled=container.settings.scheduler.enabled,
    )
