"""Human review queue."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from evalforge.api.deps import ActorDep, ContainerDep, PageDep, SessionDep
from evalforge.api.schemas import (
    DecisionRequest,
    LabelEditRequest,
    NoteRequest,
    PageOut,
    ReviewDecisionOut,
    ReviewItemDetailOut,
    ReviewItemOut,
    ReviewOutcomeOut,
    SimilarCaseOut,
    TestCaseOut,
)
from evalforge.domain.errors import NotFoundError
from evalforge.repositories import DatasetRepository
from evalforge.services.review import ReviewOutcome

router = APIRouter(prefix="/review-queue", tags=["review"])


def _outcome(outcome: ReviewOutcome) -> ReviewOutcomeOut:
    return ReviewOutcomeOut(
        applied=outcome.applied,
        item=ReviewItemOut.model_validate(outcome.item),
        decision=ReviewDecisionOut.model_validate(outcome.decision),
        test_case=TestCaseOut.model_validate(outcome.test_case) if outcome.test_case else None,
    )


@router.get("", response_model=PageOut[ReviewItemOut], summary="List review-queue items")
def list_review_items(
    session: SessionDep,
    page: PageDep,
    status: Annotated[
        str | None, Query(description="pending, approved, edited or rejected")
    ] = "pending",
) -> PageOut[ReviewItemOut]:
    result = DatasetRepository(session).list_review_items(
        status=status or None, limit=page.limit, offset=page.offset
    )
    return PageOut(
        items=[ReviewItemOut.model_validate(i) for i in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/{item_id}",
    response_model=ReviewItemDetailOut,
    summary="Get a review item with labels, decisions and similar cases",
)
def get_review_item(
    item_id: str,
    session: SessionDep,
    container: ContainerDep,
    similar: Annotated[int, Query(ge=0, le=20, description="Similar cases to include")] = 5,
) -> ReviewItemDetailOut:
    item = DatasetRepository(session).get_review_item(item_id)
    if item is None:
        raise NotFoundError("review_queue_item", item_id)
    detail = ReviewItemDetailOut.model_validate(item)
    if similar:
        detail.similar_cases = [
            SimilarCaseOut(similarity=score, test_case=TestCaseOut.model_validate(case))
            for case, score in container.review.similar_cases(session, item.candidate, similar)
        ]
    return detail


@router.post("/{item_id}/approve", response_model=ReviewOutcomeOut, summary="Approve as-is")
def approve_item(
    item_id: str,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
    body: DecisionRequest | None = None,
) -> ReviewOutcomeOut:
    return _outcome(container.review.approve(session, item_id, actor, body.notes if body else ""))


@router.post(
    "/{item_id}/edit", response_model=ReviewOutcomeOut, summary="Approve with edited labels"
)
def edit_item(
    item_id: str,
    body: LabelEditRequest,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
) -> ReviewOutcomeOut:
    return _outcome(container.review.edit(session, item_id, actor, body.labels, body.notes))


@router.post("/{item_id}/reject", response_model=ReviewOutcomeOut, summary="Reject a candidate")
def reject_item(
    item_id: str,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
    body: DecisionRequest | None = None,
) -> ReviewOutcomeOut:
    return _outcome(container.review.reject(session, item_id, actor, body.notes if body else ""))


@router.post(
    "/{item_id}/notes", response_model=ReviewOutcomeOut, summary="Annotate without resolving"
)
def annotate_item(
    item_id: str,
    body: NoteRequest,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
) -> ReviewOutcomeOut:
    return _outcome(container.review.annotate(session, item_id, actor, body.notes))
