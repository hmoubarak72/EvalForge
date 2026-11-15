"""Nightly scheduling with APScheduler.

The same job function is used by the in-process scheduler (started with the
API when ``EVALFORGE_SCHEDULER__ENABLED=true``) and by the standalone
``evalforge scheduler`` command used in Docker Compose.
"""

from __future__ import annotations

from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from evalforge.container import Container
from evalforge.db.session import session_scope
from evalforge.domain.errors import EvalForgeError
from evalforge.telemetry import get_logger

log = get_logger(__name__)

JOB_ID = "evalforge-nightly-pipeline"


def nightly_job(container: Container) -> dict[str, Any]:
    """Run the full pipeline, then (optionally) evaluate the default endpoint."""
    result = container.orchestrator.run(trigger="scheduled")
    summary: dict[str, Any] = {"pipeline_run_id": result.run_id, "status": result.status}
    if result.status != "completed" or not container.settings.scheduler.run_evaluation:
        return summary
    try:
        with session_scope(container.session_factory) as session:
            run = container.evaluation.run(session, name="nightly")
            summary["evaluation_run_id"] = run.id
            summary["pass_rate"] = run.pass_rate
    except EvalForgeError as exc:
        # e.g. an empty dataset on the very first night: report, do not crash.
        log.warning("scheduler.evaluation_skipped", reason=str(exc))
        summary["evaluation_skipped"] = str(exc)
    return summary


def build_scheduler(container: Container, blocking: bool = False) -> BaseScheduler:
    """Create a scheduler with the nightly job registered (not started)."""
    cfg = container.settings.scheduler
    scheduler: BaseScheduler = (
        BlockingScheduler(timezone=cfg.timezone)
        if blocking
        else BackgroundScheduler(timezone=cfg.timezone)
    )
    scheduler.add_job(
        nightly_job,
        CronTrigger.from_crontab(cfg.cron, timezone=cfg.timezone),
        args=[container],
        id=JOB_ID,
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    log.info("scheduler.configured", cron=cfg.cron, timezone=cfg.timezone)
    return scheduler
