"""Seed a complete EvalForge demo.

Generates two weeks of simulated production traffic (>= 300 interactions) in
three source formats, then replays it as a series of "nightly" pipeline runs
with a simulated clock so the dataset visibly grows over time. Also simulates
human review decisions and runs evaluations of two model versions so the
second one produces a detectable regression.

Usage:
    python scripts/seed_demo.py --reset
    python scripts/seed_demo.py --database-url postgresql+psycopg://... --reset

Everything runs offline with deterministic mock models; no API keys needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evalforge.config import Settings  # noqa: E402
from evalforge.container import Container, build_container, register_source  # noqa: E402
from evalforge.db.migrate import drop_everything, upgrade_to_head  # noqa: E402
from evalforge.db.session import make_engine, session_scope  # noqa: E402
from evalforge.demo.corpus import all_scenarios  # noqa: E402
from evalforge.demo.generator import (  # noqa: E402
    generate_interactions,
    write_jsonl,
    write_otel,
    write_sqlite_table,
)
from evalforge.domain.enums import ExpectedBehavior  # noqa: E402
from evalforge.domain.errors import ConflictError  # noqa: E402
from evalforge.domain.schemas import EndpointConfig, LabelEdit, LabelSet  # noqa: E402
from evalforge.domain.text_signals import jaccard, token_f1, tokenize  # noqa: E402
from evalforge.repositories import DatasetRepository, InteractionRepository  # noqa: E402
from evalforge.services.labelers import assertions_from_reference  # noqa: E402
from evalforge.telemetry import configure_logging  # noqa: E402

NIGHTS = (2, 4, 6, 8, 10, 12, 14)  # simulated days on which the nightly job runs
_EXPECTED = {
    "injection": ExpectedBehavior.REFUSE,
    "ambiguous": ExpectedBehavior.ASK_FOR_CLARIFICATION,
    "normal": ExpectedBehavior.ANSWER,
}


class SimulatedClock:
    """Mutable clock so historical nightly runs get historical timestamps."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        self.now += timedelta(milliseconds=1)  # keep event order strictly increasing
        return self.now


def write_sources(records: list[dict[str, Any]], data_dir: Path) -> dict[str, Path]:
    """Split the traffic across the three source formats, with some noise."""
    json_records = [r for i, r in enumerate(records) if i % 20 < 12]
    otel_records = [r for i, r in enumerate(records) if 12 <= i % 20 < 17]
    db_records = [r for i, r in enumerate(records) if i % 20 >= 17]
    # The same call sometimes reaches both the app log and the trace export.
    otel_records.extend(json_records[:8])

    jsonl_path = data_dir / "app_logs.jsonl"
    write_jsonl(json_records, jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for record in json_records[:5]:  # exact re-deliveries
            row = dict(record)
            row["feature"] = row.pop("application_feature")
            row["model"] = row.pop("model_name")
            row["prompt"] = row.pop("user_prompt")
            row["response"] = row.pop("raw_response")
            handle.write(json.dumps(row) + "\n")
        base_time = json_records[0]["timestamp"]
        handle.write('{"timestamp": "not-a-date", "feature": "x", "model": "m", "prompt": "p"}\n')
        handle.write(json.dumps({"timestamp": base_time, "feature": "customer_support"}) + "\n")
        handle.write(
            json.dumps(
                {
                    "timestamp": base_time,
                    "feature": "code_assistant",
                    "model": "gpt-4o",
                    "prompt": "hi",
                    "latency_ms": -5,
                }
            )
            + "\n"
        )
        handle.write("{this is not json}\n")
        handle.write(
            json.dumps(
                {
                    "timestamp": base_time,
                    "feature": "data_analysis",
                    "model": "gpt-4o",
                    "prompt": "hello",
                    "confidence": 7,
                }
            )
            + "\n"
        )

    otel_path = data_dir / "otel_traces.json"
    write_otel(otel_records, otel_path)
    db_path = data_dir / "legacy_llm_calls.db"
    write_sqlite_table(db_records, db_path)
    return {"json": jsonl_path, "otel": otel_path, "db": db_path}


def register_demo_sources(container: Container, paths: dict[str, Path]) -> None:
    with session_scope(container.session_factory) as session:
        register_source(session, "app-json-logs", "json_logs", {"path": paths["json"].as_posix()})
        register_source(
            session, "gateway-otel-traces", "otel_traces", {"path": paths["otel"].as_posix()}
        )
        register_source(
            session,
            "legacy-llm-calls-db",
            "database",
            {
                "url": f"sqlite:///{paths['db'].as_posix()}",
                "table": "llm_calls",
                "column_map": {
                    "id": "request_id",
                    "timestamp": "created_at",
                    "raw_response": "completion",
                    "latency_ms": "duration_ms",
                    "input_tokens": "prompt_tokens",
                    "output_tokens": "completion_tokens",
                    "error_status": "error_code",
                    "user_feedback": "feedback",
                    "retry_count": "retries",
                    "content_filter_triggered": "filtered",
                    "confidence_score": "confidence",
                    "metadata": "meta",
                    "application_feature": "feature",
                    "model_name": "model",
                    "user_prompt": "prompt",
                },
            },
        )


def _ground_truth(prompt: str) -> Any:
    tokens = set(tokenize(prompt))
    best_score, best = 0.0, None
    for scenario in all_scenarios():
        score = jaccard(tokens, set(tokenize(scenario.prompt)))
        if score > best_score:
            best_score, best = score, scenario
    return best if best_score >= 0.4 else None


def simulate_reviews(container: Container, limit: int, second_opinions: int) -> dict[str, int]:
    """Simulated reviewers who know the ground truth of the synthetic corpus."""
    counts = {"approved": 0, "edited": 0, "rejected": 0, "second_opinions": 0}
    with session_scope(container.session_factory) as session:
        items = DatasetRepository(session).pending_review_items()[:limit]
        for index, item in enumerate(items):
            candidate = item.candidate
            labels = LabelSet.model_validate(candidate.proposed_labels)
            scenario = _ground_truth(candidate.interaction.user_prompt)
            try:
                if scenario is None or labels.expected_behavior is not _EXPECTED[scenario.kind]:
                    container.review.reject(
                        session, item.id, "alice", "Auto-labels expect the wrong behaviour."
                    )
                    verdict = "rejected"
                elif scenario.kind != "normal" or (
                    labels.reference_answer
                    and token_f1(labels.reference_answer, scenario.good_response) >= 0.5
                ):
                    container.review.approve(session, item.id, "alice", "Labels look right.")
                    verdict = "approved"
                else:
                    edit = LabelEdit(
                        reference_answer=scenario.good_response,
                        required_assertions=assertions_from_reference(scenario.good_response),
                    )
                    container.review.edit(
                        session,
                        item.id,
                        "alice",
                        edit,
                        "Replaced the reference with the policy-correct answer.",
                    )
                    verdict = "edited"
            except ConflictError:
                container.review.reject(session, item.id, "alice", "Duplicate of an existing case.")
                verdict = "rejected"
            counts[verdict] += 1
            if index < second_opinions:
                # Bob double-reviews a few items; he disagrees on every third one.
                if verdict == "rejected" or index % 3 == 2:
                    container.review.reject(session, item.id, "bob", "Second opinion: reject.")
                else:
                    container.review.approve(session, item.id, "bob", "Second opinion: agree.")
                counts["second_opinions"] += 1
    return counts


def evaluate(container: Container, model: str, name: str) -> dict[str, Any]:
    with session_scope(container.session_factory) as session:
        run = container.evaluation.run(
            session, EndpointConfig(provider="mock", model=model), name=name
        )
        report = container.regression.compare_run(session, run)
        return {
            "run_id": run.id,
            "model": model,
            "dataset_version": run.dataset_version,
            "cases": run.dataset_size,
            "pass_rate": run.pass_rate,
            "average_quality_score": run.average_quality_score,
            "regression": report.report["summary"] if report else "no baseline yet",
            "severity": report.report["severity"] if report else None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", default=None, help="defaults to EVALFORGE_DATABASE_URL")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "demo"))
    parser.add_argument("--count", type=int, default=440, help="interactions to simulate (>=300)")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--reset", action="store_true", help="drop existing EvalForge tables first")
    parser.add_argument("--verbose", action="store_true", help="emit structured logs")
    args = parser.parse_args(argv)
    if args.count < 300:
        parser.error("--count must be at least 300")

    configure_logging("INFO" if args.verbose else "ERROR", json_logs=True)
    overrides: dict[str, Any] = {"offline_mode": True, "demo_mode": True}
    if args.database_url:
        overrides["database_url"] = args.database_url
    settings = Settings(**overrides)

    engine = make_engine(settings.database_url)
    if args.reset:
        drop_everything(engine)
    method = upgrade_to_head(settings.database_url)
    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(days=args.days)
    clock = SimulatedClock(start)
    container = build_container(settings, engine=engine, clock=clock)

    with session_scope(container.session_factory) as session:
        if InteractionRepository(session).count() and not args.reset:
            print("Database already contains interactions; re-run with --reset.", file=sys.stderr)
            return 2

    print(f"Schema ready ({method}); database: {settings.database_url}")
    records = generate_interactions(args.count, args.seed, end, args.days)
    paths = write_sources(records, Path(args.data_dir))
    register_demo_sources(container, paths)
    print(f"Generated {len(records)} interactions across 3 sources in {args.data_dir}")

    evaluations: list[dict[str, Any]] = []
    reviews = {"approved": 0, "edited": 0, "rejected": 0, "second_opinions": 0}
    print("\nNight  ingested  dupes  rejected  sampled  classified  labeled  auto  review  dup")
    for night in NIGHTS:
        if night > args.days:
            break
        clock.now = min(start + timedelta(days=night), end)
        result = container.orchestrator.run(
            trigger="scheduled", options={"sample_size": args.sample_size, "seed": 100 + night}
        )
        if result.status != "completed":
            print(f"Pipeline failed on night {night}: {result.error}", file=sys.stderr)
            return 1
        stages = result.stage_results
        print(
            f"{night:>5}  {stages['ingestion']['records_accepted']:>8}"
            f"  {stages['ingestion']['duplicates']:>5}  {stages['ingestion']['records_rejected']:>8}"
            f"  {stages['sampling']['sampled']:>7}  {stages['classification']['classified']:>10}"
            f"  {stages['labeling']['labeled']:>7}  {stages['deduplication']['auto_accepted']:>4}"
            f"  {stages['deduplication']['sent_to_review']:>6}"
            f"  {stages['deduplication']['duplicates'] + stages['labeling']['duplicates_skipped']:>3}"
        )
        if night in (6, 10, 14):  # reviewers work through part of the queue
            clock.now += timedelta(hours=7)
            for key, value in simulate_reviews(container, limit=7, second_opinions=3).items():
                reviews[key] += value
        if night == 8:
            clock.now += timedelta(hours=1)
            evaluations.append(evaluate(container, "mock-model-v1", "baseline (early dataset)"))
        if night == NIGHTS[-1] or night == args.days:
            clock.now += timedelta(hours=1)
            evaluations.append(evaluate(container, "mock-model-v1", "v1 on current dataset"))
            clock.now += timedelta(hours=1)
            evaluations.append(evaluate(container, "mock-model-v2", "v2 release candidate"))

    with session_scope(container.session_factory) as session:
        quality = container.metrics.quality(session)
        freshness = container.metrics.freshness(session)
        coverage = container.metrics.coverage(session)
        clusters = InteractionRepository(session).list_clusters()
        interactions = InteractionRepository(session).count()
        cluster_lines = [f"  - {c.name}  (traffic {c.size})" for c in clusters]

    print(f"\nInteractions stored: {interactions}")
    print(f"Clusters discovered: {len(clusters)}")
    print("\n".join(cluster_lines))
    print(
        f"\nDataset v{quality['dataset_version']}: {quality['active_test_cases']} active test cases"
    )
    print(f"  by source:      {quality['by_source']}")
    print(f"  by difficulty:  {quality['by_difficulty']}")
    print(
        f"  duplicates skipped: {quality['duplicates_skipped']}, "
        f"review queue pending: {quality['review_queue_pending']}"
    )
    print(
        f"  freshness (30d): {freshness['freshness_pct']}%   "
        f"sparse clusters: {len(coverage['sparse_clusters'])}"
    )
    print(f"  simulated reviews: {reviews}")
    print("\nEvaluation runs:")
    for entry in evaluations:
        print(
            f"  {entry['model']:<14} dataset v{entry['dataset_version']:<3} "
            f"cases={entry['cases']:<4} pass_rate={entry['pass_rate']:.1%} "
            f"avg_score={entry['average_quality_score']:.2f}"
        )
        print(
            f"      regression check: {entry['regression']}"
            + (f"  [severity: {entry['severity']}]" if entry["severity"] else "")
        )
    print("\nNext: `make api` (http://localhost:8000/docs) and `make dashboard`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
