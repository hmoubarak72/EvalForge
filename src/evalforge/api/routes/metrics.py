"""Dataset coverage, growth, freshness, quality and reviewer metrics."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from evalforge.api.deps import ContainerDep, SessionDep

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/coverage", summary="Cluster coverage and cluster x difficulty heatmap")
def coverage(session: SessionDep, container: ContainerDep) -> dict[str, Any]:
    return container.metrics.coverage(session)


@router.get("/growth", summary="Dataset growth over time")
def growth(session: SessionDep, container: ContainerDep) -> dict[str, Any]:
    return container.metrics.growth(session)


@router.get("/freshness", summary="Share of cases sourced from recent traffic")
def freshness(
    session: SessionDep,
    container: ContainerDep,
    window_days: Annotated[int, Query(ge=1, le=3650)] = 30,
) -> dict[str, Any]:
    return container.metrics.freshness(session, window_days)


@router.get("/quality", summary="Dataset composition and auto-labeling quality")
def quality(session: SessionDep, container: ContainerDep) -> dict[str, Any]:
    return container.metrics.quality(session)


@router.get("/reviewers", summary="Reviewer throughput and inter-annotator agreement")
def reviewers(session: SessionDep, container: ContainerDep) -> dict[str, Any]:
    return container.metrics.reviewer_stats(session)
