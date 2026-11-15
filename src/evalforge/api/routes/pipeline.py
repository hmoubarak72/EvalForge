"""Pipeline triggers and run history."""

from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy.orm import Session

from evalforge.api.deps import ActorDep, ContainerDep, PageDep, SessionDep
from evalforge.api.schemas import (
    PageOut,
    PipelineRunCreate,
    PipelineRunOut,
    StageName,
    StageTrigger,
)
from evalforge.container import Container
from evalforge.db.session import session_scope
from evalforge.domain.errors import NotFoundError
from evalforge.repositories import RunRepository

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _load(container: Container, run_id: str) -> PipelineRunOut:
    with session_scope(container.session_factory) as session:
        run = RunRepository(session).get_pipeline_run(run_id)
        if run is None:
            raise NotFoundError("pipeline_run", run_id)
        return PipelineRunOut.model_validate(run)


@router.post(
    "/runs",
    response_model=PipelineRunOut,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger a full or partial pipeline run",
)
def create_pipeline_run(
    body: PipelineRunCreate, container: ContainerDep, actor: ActorDep
) -> PipelineRunOut:
    """Run the selected stages synchronously (all stages when omitted).

    A failed stage yields a run with ``status = "failed"`` and an ``error``;
    use the resume endpoint to continue it.
    """
    result = container.orchestrator.run(body.stages, trigger=f"api:{actor}", options=body.options())
    return _load(container, result.run_id)


@router.post(
    "/stages/{stage}",
    response_model=PipelineRunOut,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger a single stage",
)
def trigger_stage(
    stage: StageName, container: ContainerDep, actor: ActorDep, body: StageTrigger | None = None
) -> PipelineRunOut:
    """Run exactly one of: ingestion, sampling, classification, labeling, deduplication."""
    options = body.model_dump(exclude_none=True) if body else {}
    result = container.orchestrator.run([stage], trigger=f"api:{actor}", options=options)
    return _load(container, result.run_id)


@router.post("/runs/{run_id}/resume", response_model=PipelineRunOut, summary="Resume a failed run")
def resume_pipeline_run(run_id: str, container: ContainerDep, actor: ActorDep) -> PipelineRunOut:
    result = container.orchestrator.resume(run_id)
    return _load(container, result.run_id)


@router.get("/runs", response_model=PageOut[PipelineRunOut], summary="List pipeline runs")
def list_pipeline_runs(session: SessionDep, page: PageDep) -> PageOut[PipelineRunOut]:
    result = RunRepository(session).list_pipeline_runs(page.limit, page.offset)
    return PageOut(
        items=[PipelineRunOut.model_validate(r) for r in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get("/runs/{run_id}", response_model=PipelineRunOut, summary="Get a pipeline run")
def get_pipeline_run(run_id: str, session: SessionDep) -> PipelineRunOut:
    return _get(session, run_id)


def _get(session: Session, run_id: str) -> PipelineRunOut:
    run = RunRepository(session).get_pipeline_run(run_id)
    if run is None:
        raise NotFoundError("pipeline_run", run_id)
    return PipelineRunOut.model_validate(run)
