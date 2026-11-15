"""Candidates and test cases."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from evalforge.api.deps import ActorDep, ContainerDep, PageDep, SessionDep
from evalforge.api.schemas import (
    CandidateDetailOut,
    CandidateOut,
    LabelEditRequest,
    NoteRequest,
    PageOut,
    RetireRequest,
    TestCaseDetailOut,
    TestCaseOut,
)
from evalforge.domain.errors import NotFoundError
from evalforge.repositories import DatasetRepository

router = APIRouter(tags=["dataset"])


@router.get("/candidates", response_model=PageOut[CandidateOut], summary="List candidates")
def list_candidates(
    session: SessionDep,
    page: PageDep,
    status: Annotated[str | None, Query(description="Candidate status")] = None,
    cluster_id: str | None = None,
) -> PageOut[CandidateOut]:
    result = DatasetRepository(session).list_candidates(
        status=status, cluster_id=cluster_id, limit=page.limit, offset=page.offset
    )
    return PageOut(
        items=[CandidateOut.model_validate(c) for c in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/candidates/{candidate_id}", response_model=CandidateDetailOut, summary="Get a candidate"
)
def get_candidate(candidate_id: str, session: SessionDep) -> CandidateDetailOut:
    candidate = DatasetRepository(session).get_candidate(candidate_id)
    if candidate is None:
        raise NotFoundError("candidate", candidate_id)
    return CandidateDetailOut.model_validate(candidate)


@router.get("/test-cases", response_model=PageOut[TestCaseOut], summary="List test cases")
def list_test_cases(
    session: SessionDep,
    page: PageDep,
    category: str | None = None,
    difficulty: str | None = None,
    source: Annotated[
        str | None, Query(description="auto_labeled or human_reviewed (review status)")
    ] = None,
    status: Annotated[str | None, Query(description="active or retired")] = None,
    case_type: str | None = None,
    min_quality: Annotated[int | None, Query(ge=1, le=5)] = None,
    max_quality: Annotated[int | None, Query(ge=1, le=5)] = None,
    created_since: datetime | None = None,
    created_until: datetime | None = None,
    search: Annotated[str | None, Query(max_length=200, description="Prompt substring")] = None,
) -> PageOut[TestCaseOut]:
    result = DatasetRepository(session).list_test_cases(
        category=category,
        difficulty=difficulty,
        source=source,
        status=status,
        case_type=case_type,
        min_quality=min_quality,
        max_quality=max_quality,
        created_since=created_since,
        created_until=created_until,
        search=search,
        limit=page.limit,
        offset=page.offset,
    )
    return PageOut(
        items=[TestCaseOut.model_validate(c) for c in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/test-cases/{test_case_id}", response_model=TestCaseDetailOut, summary="Get a test case"
)
def get_test_case(test_case_id: str, session: SessionDep) -> TestCaseDetailOut:
    test_case = DatasetRepository(session).get_test_case(test_case_id)
    if test_case is None:
        raise NotFoundError("test_case", test_case_id)
    return TestCaseDetailOut.model_validate(test_case)


@router.patch(
    "/test-cases/{test_case_id}/labels",
    response_model=TestCaseDetailOut,
    summary="Edit labels (creates a new label version)",
)
def edit_test_case_labels(
    test_case_id: str,
    body: LabelEditRequest,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
) -> TestCaseDetailOut:
    test_case = container.review.edit_test_case(
        session, test_case_id, actor, body.labels, body.notes
    )
    return TestCaseDetailOut.model_validate(test_case)


@router.post(
    "/test-cases/{test_case_id}/notes",
    response_model=TestCaseDetailOut,
    summary="Add a reviewer note",
)
def add_test_case_note(
    test_case_id: str,
    body: NoteRequest,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
) -> TestCaseDetailOut:
    test_case = container.review.add_note(session, test_case_id, actor, body.notes)
    return TestCaseDetailOut.model_validate(test_case)


@router.post(
    "/test-cases/{test_case_id}/retire",
    response_model=TestCaseDetailOut,
    summary="Retire a test case from the active dataset",
)
def retire_test_case(
    test_case_id: str,
    body: RetireRequest,
    session: SessionDep,
    container: ContainerDep,
    actor: ActorDep,
) -> TestCaseDetailOut:
    test_case = container.review.retire_test_case(session, test_case_id, actor, body.reason)
    return TestCaseDetailOut.model_validate(test_case)
