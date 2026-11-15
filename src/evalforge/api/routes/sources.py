"""Ingestion sources and ingestion-run history."""

from __future__ import annotations

from fastapi import APIRouter, status

from evalforge.api.deps import ActorDep, PageDep, SessionDep
from evalforge.api.schemas import IngestionRunOut, PageOut, SourceCreate, SourceOut
from evalforge.container import register_source
from evalforge.repositories import AuditRepository, InteractionRepository

router = APIRouter(tags=["ingestion"])


@router.post(
    "/sources",
    response_model=SourceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update an ingestion source",
)
def create_source(body: SourceCreate, session: SessionDep, actor: ActorDep) -> SourceOut:
    """Register a log source. Posting an existing name updates that source."""
    source = register_source(session, body.name, body.kind, body.config, body.enabled)
    AuditRepository(session).record("source.upserted", "ingestion_source", source.id, actor=actor)
    return SourceOut.model_validate(source)


@router.get("/sources", response_model=list[SourceOut], summary="List ingestion sources")
def list_sources(session: SessionDep) -> list[SourceOut]:
    return [SourceOut.model_validate(s) for s in InteractionRepository(session).list_sources()]


@router.get(
    "/ingestion-runs", response_model=PageOut[IngestionRunOut], summary="List ingestion runs"
)
def list_ingestion_runs(session: SessionDep, page: PageDep) -> PageOut[IngestionRunOut]:
    result = InteractionRepository(session).list_ingestion_runs(page.limit, page.offset)
    return PageOut(
        items=[IngestionRunOut.model_validate(r) for r in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )
