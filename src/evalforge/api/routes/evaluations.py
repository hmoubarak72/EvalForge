"""Evaluation runs, reports and regression reports."""

from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy.orm import Session

from evalforge.api.deps import ActorDep, ContainerDep, PageDep, SessionDep
from evalforge.api.schemas import (
    EvaluationCreate,
    EvaluationReportOut,
    EvaluationResultOut,
    EvaluationRunOut,
    PageOut,
    RegressionReportOut,
)
from evalforge.db.models import EvaluationRun
from evalforge.domain.errors import NotFoundError
from evalforge.repositories import RunRepository

router = APIRouter(tags=["evaluation"])


def _run_or_404(session: Session, run_id: str) -> EvaluationRun:
    run = RunRepository(session).get_evaluation_run(run_id)
    if run is None:
        raise NotFoundError("evaluation_run", run_id)
    return run


@router.post(
    "/evaluations",
    response_model=EvaluationRunOut,
    status_code=status.HTTP_201_CREATED,
    summary="Run the dataset against a model endpoint",
)
def create_evaluation(
    body: EvaluationCreate, session: SessionDep, container: ContainerDep, actor: ActorDep
) -> EvaluationRunOut:
    """Evaluate the active dataset, persist all raw outputs and scores, and
    compare the run with the previous comparable run (regression report)."""
    run = container.evaluation.run(
        session,
        body.endpoint,
        name=body.name,
        grading_mode=body.grading_mode,
        comparison_group=body.comparison_group,
        baseline_run_id=body.baseline_run_id,
    )
    return EvaluationRunOut.model_validate(run)


@router.get(
    "/evaluations", response_model=PageOut[EvaluationRunOut], summary="List evaluation runs"
)
def list_evaluations(
    session: SessionDep,
    page: PageDep,
    comparison_group: str | None = None,
    model_name: str | None = None,
    status: str | None = None,
) -> PageOut[EvaluationRunOut]:
    result = RunRepository(session).list_evaluation_runs(
        comparison_group=comparison_group,
        model_name=model_name,
        status=status,
        limit=page.limit,
        offset=page.offset,
    )
    return PageOut(
        items=[EvaluationRunOut.model_validate(r) for r in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/evaluations/{run_id}", response_model=EvaluationRunOut, summary="Get an evaluation run"
)
def get_evaluation(run_id: str, session: SessionDep) -> EvaluationRunOut:
    return EvaluationRunOut.model_validate(_run_or_404(session, run_id))


@router.get(
    "/evaluations/{run_id}/report",
    response_model=EvaluationReportOut,
    summary="Structured evaluation report",
)
def get_evaluation_report(run_id: str, session: SessionDep) -> EvaluationReportOut:
    run = _run_or_404(session, run_id)
    return EvaluationReportOut(run=EvaluationRunOut.model_validate(run), report=run.report)


@router.get(
    "/evaluations/{run_id}/results",
    response_model=PageOut[EvaluationResultOut],
    summary="Per-case results (raw responses and scores)",
)
def list_evaluation_results(
    run_id: str, session: SessionDep, page: PageDep, passed: bool | None = None
) -> PageOut[EvaluationResultOut]:
    _run_or_404(session, run_id)
    result = RunRepository(session).list_results(
        run_id, passed=passed, limit=page.limit, offset=page.offset
    )
    return PageOut(
        items=[EvaluationResultOut.model_validate(r) for r in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


@router.get(
    "/evaluations/{run_id}/regression",
    response_model=RegressionReportOut,
    summary="Regression report versus the previous comparable run",
)
def get_regression_report(run_id: str, session: SessionDep) -> RegressionReportOut:
    _run_or_404(session, run_id)
    report = RunRepository(session).regression_report_for(run_id)
    if report is None:
        raise NotFoundError("regression_report_for_run", run_id)
    return RegressionReportOut.model_validate(report)


@router.get(
    "/regressions", response_model=PageOut[RegressionReportOut], summary="List regression reports"
)
def list_regression_reports(session: SessionDep, page: PageDep) -> PageOut[RegressionReportOut]:
    result = RunRepository(session).list_regression_reports(page.limit, page.offset)
    return PageOut(
        items=[RegressionReportOut.model_validate(r) for r in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )
