"""CLI, migrations, scheduler job and dashboard smoke tests (file-based SQLite)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import inspect

from evalforge import cli
from evalforge.config import get_settings
from evalforge.container import build_container
from evalforge.db.session import make_engine, session_scope
from evalforge.domain.schemas import EndpointConfig
from evalforge.pipeline.scheduler import JOB_ID, build_scheduler, nightly_job
from evalforge.repositories import DatasetRepository
from tests.conftest import DB_COLUMN_MAP, write_demo_sources

pytestmark = pytest.mark.integration

REQUIRED_TABLES = {
    "interaction_logs", "ingestion_runs", "samples", "clusters", "interaction_classifications",
    "eval_candidates", "eval_test_cases", "test_case_labels", "review_queue_items",
    "review_decisions", "evaluation_runs", "evaluation_results", "regression_reports",
    "pipeline_runs", "audit_events",
}  # fmt: skip


@pytest.fixture(scope="module")
def cli_env(
    tmp_path_factory: pytest.TempPathFactory, demo_records: list[dict[str, Any]]
) -> Iterator[dict[str, Path]]:
    """Environment-configured app on a migrated SQLite file with demo sources."""
    root = tmp_path_factory.mktemp("cli")
    paths = write_demo_sources(demo_records, root)
    sources = [
        {"name": "json", "kind": "json_logs", "config": {"path": paths["json"].as_posix()}},
        {"name": "otel", "kind": "otel_traces", "config": {"path": paths["otel"].as_posix()}},
        {"name": "db", "kind": "database",
         "config": {"url": f"sqlite:///{paths['db'].as_posix()}", "table": "llm_calls",
                    "column_map": DB_COLUMN_MAP}},
    ]  # fmt: skip
    patch = pytest.MonkeyPatch()
    patch.setenv("EVALFORGE_DATABASE_URL", f"sqlite:///{(root / 'app.db').as_posix()}")
    patch.setenv("EVALFORGE_SOURCES", json.dumps(sources))
    patch.setenv("EVALFORGE_LOG_LEVEL", "ERROR")
    patch.setenv("EVALFORGE_OFFLINE_MODE", "true")
    patch.setenv("EVALFORGE_SAMPLING__SAMPLE_SIZE", "150")
    patch.setenv("EVALFORGE_LABELING__MAX_CANDIDATES_PER_RUN", "300")
    patch.chdir(root)  # no stray .env is picked up
    get_settings.cache_clear()
    try:
        yield paths
    finally:
        patch.undo()
        get_settings.cache_clear()


def test_cli_end_to_end(cli_env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["init-db"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "alembic"
    engine = make_engine(get_settings().database_url)
    assert set(inspect(engine).get_table_names()) >= REQUIRED_TABLES
    engine.dispose()

    assert cli.main(["sources"]) == 0
    assert {s["name"] for s in json.loads(capsys.readouterr().out)} == {"json", "otel", "db"}

    assert cli.main(["evaluate"]) == 2  # empty dataset: clean error, not a traceback
    assert "no active test cases" in capsys.readouterr().err

    assert cli.main(["pipeline", "--seed", "5"]) == 0
    pipeline = json.loads(capsys.readouterr().out)
    assert pipeline["status"] == "completed"
    assert pipeline["stage_results"]["sampling"]["sampled"] == 150
    assert pipeline["stage_results"]["deduplication"]["auto_accepted"] > 0

    assert cli.main(["evaluate", "--model", "mock-model-v1", "--name", "base"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["regression"] is None and first["overall"]["total"] > 0
    assert cli.main(["evaluate", "--model", "mock-model-v2"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["regression_severity"] in {"minor", "major"}
    assert cli.main(["report", second["evaluation_run_id"]]) == 0
    assert json.loads(capsys.readouterr().out)["regression"]["has_regression"] is True
    assert cli.main(["report", "missing"]) == 2

    assert cli.main(["add-source", "extra", "json_logs", "--config", '{"path": "x.jsonl"}']) == 0


def test_scheduler_job(cli_env: dict[str, Path]) -> None:
    container = build_container(get_settings())
    scheduler = build_scheduler(container)
    job = scheduler.get_job(JOB_ID)
    assert job is not None and str(job.trigger).startswith("cron[")
    summary = nightly_job(container)
    assert summary["status"] == "completed"
    assert summary["evaluation_run_id"] and 0 < summary["pass_rate"] <= 1


def test_dashboard_pages_render(cli_env: dict[str, Path]) -> None:
    from streamlit.testing.v1 import AppTest

    container = build_container(get_settings())
    with session_scope(container.session_factory) as session:
        assert DatasetRepository(session).active_test_cases()
        assert DatasetRepository(session).pending_review_items()
        container.evaluation.run(session, EndpointConfig(provider="mock", model="mock-model-v2"))

    script = Path(__file__).resolve().parents[1] / "src" / "evalforge" / "ui" / "dashboard.py"
    app = AppTest.from_file(str(script), default_timeout=120).run()
    assert not app.exception, app.exception
    pages = app.sidebar.radio[0].options
    assert pages == ["Overview", "Dataset explorer", "Review queue", "Evaluations", "Clusters",
                     "Reviewers"]  # fmt: skip
    for page in pages[1:]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
        assert app.header[0].value

    # Approving from the review queue adds a human-reviewed case.
    app.sidebar.radio[0].set_value("Review queue").run()
    with session_scope(container.session_factory) as session:
        pending = len(DatasetRepository(session).pending_review_items())
    next(b for b in app.button if b.label == "Approve as-is").click().run()
    assert not app.exception, app.exception
    with session_scope(container.session_factory) as session:
        dataset = DatasetRepository(session)
        assert len(dataset.pending_review_items()) == pending - 1
        assert any(c.source == "human_reviewed" for c in dataset.active_test_cases())
