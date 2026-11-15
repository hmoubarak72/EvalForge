"""Repository for pipeline runs, evaluation runs/results and regression reports."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db.models import EvaluationResult, EvaluationRun, PipelineRun, RegressionReport
from evalforge.domain.enums import RunStatus
from evalforge.repositories.base import Page, paginate


class RunRepository:
    """Queries over run history (never mutates historical results)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- pipeline runs -------------------------------------------------------
    def get_pipeline_run(self, run_id: str) -> PipelineRun | None:
        return self._session.get(PipelineRun, run_id)

    def list_pipeline_runs(self, limit: int = 50, offset: int = 0) -> Page[PipelineRun]:
        statement = select(PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id)
        return paginate(self._session, statement, limit, offset)

    # -- evaluation runs -----------------------------------------------------
    def get_evaluation_run(self, run_id: str) -> EvaluationRun | None:
        return self._session.get(EvaluationRun, run_id)

    def list_evaluation_runs(
        self,
        *,
        comparison_group: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[EvaluationRun]:
        statement = select(EvaluationRun).order_by(
            EvaluationRun.started_at.desc(), EvaluationRun.id
        )
        if comparison_group:
            statement = statement.where(EvaluationRun.comparison_group == comparison_group)
        if model_name:
            statement = statement.where(EvaluationRun.model_name == model_name)
        if status:
            statement = statement.where(EvaluationRun.status == status)
        return paginate(self._session, statement, limit, offset)

    def previous_comparable_run(self, run: EvaluationRun) -> EvaluationRun | None:
        """Latest completed run in the same comparison group that started earlier."""
        statement = (
            select(EvaluationRun)
            .where(
                EvaluationRun.comparison_group == run.comparison_group,
                EvaluationRun.status == RunStatus.COMPLETED.value,
                EvaluationRun.id != run.id,
                EvaluationRun.started_at <= run.started_at,
            )
            .order_by(EvaluationRun.started_at.desc(), EvaluationRun.id)
            .limit(1)
        )
        return self._session.scalar(statement)

    def results_for_run(self, run_id: str) -> list[EvaluationResult]:
        statement = (
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.id)
        )
        return list(self._session.scalars(statement))

    def list_results(
        self, run_id: str, *, passed: bool | None = None, limit: int = 50, offset: int = 0
    ) -> Page[EvaluationResult]:
        statement = (
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.passed, EvaluationResult.id)
        )
        if passed is not None:
            statement = statement.where(EvaluationResult.passed.is_(passed))
        return paginate(self._session, statement, limit, offset)

    # -- regression reports --------------------------------------------------
    def regression_report_for(self, run_id: str) -> RegressionReport | None:
        return self._session.scalar(
            select(RegressionReport).where(RegressionReport.run_id == run_id)
        )

    def list_regression_reports(self, limit: int = 50, offset: int = 0) -> Page[RegressionReport]:
        statement = select(RegressionReport).order_by(
            RegressionReport.created_at.desc(), RegressionReport.id
        )
        return paginate(self._session, statement, limit, offset)
