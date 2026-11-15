"""Command-line interface.

Examples:
    evalforge init-db
    evalforge add-source app-logs json_logs --config '{"path": "logs/*.jsonl"}'
    evalforge pipeline --strategy signal_boosted --sample-size 80
    evalforge evaluate --provider mock --model mock-model-v1
    evalforge report <evaluation-run-id>
    evalforge scheduler
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from evalforge.config import get_settings
from evalforge.container import Container, build_container, register_source, sync_configured_sources
from evalforge.db.migrate import upgrade_to_head
from evalforge.db.session import session_scope
from evalforge.domain.errors import EvalForgeError, NotFoundError
from evalforge.domain.schemas import EndpointConfig
from evalforge.pipeline.orchestrator import STAGES
from evalforge.repositories import InteractionRepository, RunRepository
from evalforge.telemetry import configure_logging


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _container() -> Container:
    settings = get_settings()
    container = build_container(settings)
    with session_scope(container.session_factory) as session:
        sync_configured_sources(session, settings)
    return container


def cmd_init_db(_: argparse.Namespace) -> int:
    method = upgrade_to_head(get_settings().database_url)
    _print({"status": "ok", "method": method})
    return 0


def cmd_add_source(args: argparse.Namespace) -> int:
    container = _container()
    with session_scope(container.session_factory) as session:
        source = register_source(session, args.name, args.kind, json.loads(args.config))
        _print({"id": source.id, "name": source.name, "kind": source.kind})
    return 0


def cmd_sources(_: argparse.Namespace) -> int:
    container = _container()
    with session_scope(container.session_factory) as session:
        _print(
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "kind": s.kind,
                    "enabled": s.enabled,
                    "cursor": s.cursor_timestamp,
                }
                for s in InteractionRepository(session).list_sources()
            ]
        )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    container = _container()
    options = {
        key: value
        for key, value in (
            ("strategy", args.strategy),
            ("sample_size", args.sample_size),
            ("seed", args.seed),
        )
        if value is not None
    }
    result = container.orchestrator.run(args.stages or None, trigger="cli", options=options)
    _print(
        {
            "pipeline_run_id": result.run_id,
            "status": result.status,
            "error": result.error,
            "stage_results": result.stage_results,
        }
    )
    return 0 if result.status == "completed" else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    container = _container()
    endpoint = container.evaluation.default_endpoint()
    updates = {
        key: value
        for key, value in (
            ("provider", args.provider),
            ("model", args.model),
            ("base_url", args.base_url),
            ("api_key_env", args.api_key_env),
            ("concurrency", args.concurrency),
        )
        if value is not None
    }
    endpoint = EndpointConfig.model_validate({**endpoint.model_dump(), **updates})
    with session_scope(container.session_factory) as session:
        run = container.evaluation.run(
            session,
            endpoint,
            name=args.name or "",
            grading_mode=args.grading_mode,
            comparison_group=args.comparison_group,
        )
        regression = RunRepository(session).regression_report_for(run.id)
        _print(
            {
                "evaluation_run_id": run.id,
                "model": run.model_name,
                "dataset_version": run.dataset_version,
                "overall": run.report["overall"],
                "by_category": run.report["by_category"],
                "by_difficulty": run.report["by_difficulty"],
                "usage": run.report["usage"],
                "regression": regression.report["summary"] if regression else None,
                "regression_severity": regression.report["severity"] if regression else None,
            }
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    container = _container()
    with session_scope(container.session_factory) as session:
        runs = RunRepository(session)
        run = runs.get_evaluation_run(args.run_id)
        if run is None:
            raise NotFoundError("evaluation_run", args.run_id)
        regression = runs.regression_report_for(run.id)
        _print({"report": run.report, "regression": regression.report if regression else None})
    return 0


def cmd_scheduler(_: argparse.Namespace) -> int:
    from evalforge.pipeline.scheduler import build_scheduler

    scheduler = build_scheduler(_container(), blocking=True)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalforge", description="EvalForge command line")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="apply database migrations").set_defaults(func=cmd_init_db)

    add = sub.add_parser("add-source", help="create or update a log source")
    add.add_argument("name")
    add.add_argument("kind", choices=["json_logs", "otel_traces", "database"])
    add.add_argument("--config", default="{}", help="adapter config as JSON")
    add.set_defaults(func=cmd_add_source)

    sub.add_parser("sources", help="list log sources").set_defaults(func=cmd_sources)

    pipe = sub.add_parser("pipeline", help="run the pipeline (all stages by default)")
    pipe.add_argument("--stages", nargs="*", choices=STAGES)
    pipe.add_argument("--strategy", choices=["uniform", "stratified", "signal_boosted"])
    pipe.add_argument("--sample-size", type=int)
    pipe.add_argument("--seed", type=int)
    pipe.set_defaults(func=cmd_pipeline)

    evaluate = sub.add_parser("evaluate", help="run the dataset against a model endpoint")
    evaluate.add_argument("--provider", choices=["mock", "openai_compatible"])
    evaluate.add_argument("--model")
    evaluate.add_argument("--base-url")
    evaluate.add_argument("--api-key-env")
    evaluate.add_argument("--concurrency", type=int)
    evaluate.add_argument("--grading-mode", choices=["deterministic", "llm", "hybrid"])
    evaluate.add_argument("--comparison-group")
    evaluate.add_argument("--name")
    evaluate.set_defaults(func=cmd_evaluate)

    report = sub.add_parser("report", help="print an evaluation and regression report")
    report.add_argument("run_id")
    report.set_defaults(func=cmd_report)

    sub.add_parser("scheduler", help="run the nightly scheduler").set_defaults(func=cmd_scheduler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return int(args.func(args))
    except EvalForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
