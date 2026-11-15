"""Interactions and clusters (read-only)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from evalforge.api.deps import PageDep, SessionDep
from evalforge.api.schemas import (
    ClassificationOut,
    ClusterOut,
    InteractionDetailOut,
    InteractionOut,
    PageOut,
)
from evalforge.domain.errors import NotFoundError
from evalforge.repositories import InteractionRepository

router = APIRouter(tags=["interactions"])


@router.get("/interactions", response_model=PageOut[InteractionOut], summary="List interactions")
def list_interactions(
    session: SessionDep,
    page: PageDep,
    feature: Annotated[str | None, Query(description="Application feature")] = None,
    model: Annotated[str | None, Query(description="Model name")] = None,
    feedback: Annotated[str | None, Query(description="User feedback value")] = None,
    has_error: Annotated[bool | None, Query(description="Only errors / only successes")] = None,
    since: Annotated[datetime | None, Query(description="Timestamp >= since")] = None,
    until: Annotated[datetime | None, Query(description="Timestamp <= until")] = None,
) -> PageOut[InteractionOut]:
    result = InteractionRepository(session).list_interactions(
        feature=feature,
        model=model,
        feedback=feedback,
        has_error=has_error,
        since=since,
        until=until,
        limit=page.limit,
        offset=page.offset,
    )
    return PageOut(
        items=[InteractionOut.model_validate(i) for i in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/interactions/{interaction_id}",
    response_model=InteractionDetailOut,
    summary="Get an interaction with its classification",
)
def get_interaction(interaction_id: str, session: SessionDep) -> InteractionDetailOut:
    repo = InteractionRepository(session)
    interaction = repo.get(interaction_id)
    if interaction is None:
        raise NotFoundError("interaction", interaction_id)
    detail = InteractionDetailOut.model_validate(interaction)
    classification = repo.classification_for(interaction_id)
    if classification is not None:
        detail.classification = ClassificationOut.model_validate(classification)
    return detail


@router.get(
    "/classifications",
    response_model=PageOut[ClassificationOut],
    summary="List classifier outputs",
)
def list_classifications(
    session: SessionDep,
    page: PageDep,
    cluster_id: str | None = None,
    difficulty: str | None = None,
    outliers_only: bool = False,
) -> PageOut[ClassificationOut]:
    result = InteractionRepository(session).list_classifications(
        cluster_id=cluster_id,
        difficulty=difficulty,
        outliers_only=outliers_only,
        limit=page.limit,
        offset=page.offset,
    )
    return PageOut(
        items=[ClassificationOut.model_validate(c) for c in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get("/clusters", response_model=list[ClusterOut], summary="List clusters by traffic")
def list_clusters(session: SessionDep) -> list[ClusterOut]:
    return [ClusterOut.model_validate(c) for c in InteractionRepository(session).list_clusters()]


@router.get("/clusters/{cluster_id}", response_model=ClusterOut, summary="Get a cluster")
def get_cluster(cluster_id: str, session: SessionDep) -> ClusterOut:
    cluster = InteractionRepository(session).get_cluster(cluster_id)
    if cluster is None:
        raise NotFoundError("cluster", cluster_id)
    return ClusterOut.model_validate(cluster)
