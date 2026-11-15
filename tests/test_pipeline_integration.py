"""Integration tests: full pipeline, dedup, review, evaluation and regression
against a local in-memory database with deterministic mock models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select

from evalforge.adapters.llm.base import ChatResult, LLMError, Message
from evalforge.adapters.llm.openai_compat import OpenAICompatibleChatClient
from evalforge.container import Container
from evalforge.db.models import (
    AuditEvent,
    EvalCandidate,
    EvalTestCase,
    EvaluationResult,
    InteractionClassification,
    InteractionLog,
    RegressionReport,
    Sample,
)
from evalforge.db.session import session_scope
from evalforge.domain.enums import ExpectedBehavior
from evalforge.domain.errors import ConflictError, InvalidRequestError, NotFoundError
from evalforge.domain.schemas import EndpointConfig, LabelEdit
from evalforge.repositories import DatasetRepository, InteractionRepository, RunRepository
from tests.conftest import make_container, register_demo_sources, write_demo_sources

pytestmark = pytest.mark.integration


def count(container: Container, model: Any) -> int:
    with session_scope(container.session_factory) as db:
        return int(db.scalar(select(func.count()).select_from(model)) or 0)


def test_full_pipeline_produces_dataset_and_review_queue(seeded_container: Container) -> None:
    container = seeded_container
    with session_scope(container.session_factory) as db:
        run = RunRepository(db).list_pipeline_runs().items[0]
        stages = run.stage_results
        assert run.status == "completed" and run.correlation_id
        assert list(stages) == [
            "ingestion",
            "sampling",
            "classification",
            "labeling",
            "deduplication",
        ]
        assert stages["ingestion"]["records_accepted"] == 320
        assert stages["ingestion"]["redactions"] > 0
        assert stages["sampling"]["sampled"] == 200
        assert stages["classification"]["classified"] == 200
        assert stages["classification"]["new_clusters"] >= 5
        assert stages["deduplication"]["auto_accepted"] > 10
        assert stages["deduplication"]["sent_to_review"] > 0
        assert (
            stages["deduplication"]["duplicates"] + stages["labeling"]["duplicates_skipped"]
        ) > 0

        clusters = InteractionRepository(db).list_clusters()
        assert len(clusters) >= 5 and all(c.name and c.size > 0 for c in clusters)
        difficulties = {c.difficulty for c in db.scalars(select(InteractionClassification))}
        assert difficulties == {"simple", "moderate", "hard", "adversarial"}

        cases = DatasetRepository(db).active_test_cases()
        behaviors = {c.current_label.expected_behavior for c in cases if c.current_label}
        assert behaviors == {b.value for b in ExpectedBehavior}
        assert all(c.source == "auto_labeled" and c.added_in_version == 1 for c in cases)
        assert all(c.current_label and c.current_label.confidence >= 0.8 for c in cases)
        assert DatasetRepository(db).current_version() == 1
        # No stored text contains the synthetic PII.
        assert not any("@example.com" in c.user_prompt for c in cases)
        assert db.scalar(select(func.count()).select_from(AuditEvent)) > 0


def test_no_two_test_cases_exceed_the_similarity_threshold(seeded_container: Container) -> None:
    import numpy as np

    with session_scope(seeded_container.session_factory) as db:
        vectors = np.array([c.embedding for c in DatasetRepository(db).active_test_cases()])
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, 0.0)
    assert similarity.max() <= seeded_container.settings.dedup.similarity_threshold + 1e-9


def test_pipeline_is_idempotent_and_incremental(
    tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    container = make_container()
    register_demo_sources(container, write_demo_sources(demo_records, tmp_path))
    first = container.orchestrator.run(options={"sample_size": 80})
    snapshot = {m: count(container, m) for m in (InteractionLog, EvalTestCase, EvalCandidate)}
    second = container.orchestrator.run(options={"sample_size": 80})
    assert first.status == second.status == "completed"
    assert second.stage_results["ingestion"]["records_accepted"] == 0
    assert count(container, InteractionLog) == snapshot[InteractionLog]
    # The second run samples only interactions the first run did not touch.
    with session_scope(container.session_factory) as db:
        ids = list(db.scalars(select(Sample.interaction_id)))
    assert len(ids) == len(set(ids)) == 160
    assert count(container, EvalCandidate) > snapshot[EvalCandidate]
    assert count(container, EvalTestCase) >= snapshot[EvalTestCase]


def test_single_stage_runs_and_stage_validation(
    tmp_path: Path, demo_records: list[dict[str, Any]]
) -> None:
    container = make_container()
    register_demo_sources(container, write_demo_sources(demo_records, tmp_path))
    ingest = container.orchestrator.run(["ingestion"])
    assert list(ingest.stage_results) == ["ingestion"]
    sampled = container.orchestrator.run(
        ["sampling"], options={"strategy": "stratified", "sample_size": 50, "seed": 9}
    )
    assert sampled.stage_results["sampling"] == {
        "sampled": 50,
        "strategy": "stratified",
        "seed": 9,
        "reason_counts": {"stratum": 50},
    }
    classified = container.orchestrator.run(["classification"])  # picks up the backlog
    assert classified.stage_results["classification"]["classified"] == 50
    with pytest.raises(InvalidRequestError):
        container.orchestrator.run(["training"])
    with pytest.raises(NotFoundError):
        container.orchestrator.resume("missing")


def test_failed_stage_is_recorded_and_resumable(
    tmp_path: Path, demo_records: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    container = make_container()
    register_demo_sources(container, write_demo_sources(demo_records, tmp_path))
    original = container.labeling.run

    def explode(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("labeling model unavailable")

    monkeypatch.setattr(container.labeling, "run", explode)
    failed = container.orchestrator.run(options={"sample_size": 60})
    assert failed.status == "failed" and "labeling model unavailable" in (failed.error or "")
    assert list(failed.stage_results) == ["ingestion", "sampling", "classification"]

    monkeypatch.setattr(container.labeling, "run", original)
    resumed = container.orchestrator.resume(failed.run_id)
    assert resumed.status == "completed" and resumed.run_id == failed.run_id
    assert set(resumed.stage_results) >= {"labeling", "deduplication"}
    assert count(container, Sample) == 60  # completed stages were not repeated
    with pytest.raises(InvalidRequestError):
        container.orchestrator.resume(failed.run_id)


def test_review_workflow(tmp_path: Path, demo_records: list[dict[str, Any]]) -> None:
    container = make_container()
    register_demo_sources(container, write_demo_sources(demo_records, tmp_path))
    container.orchestrator.run(options={"sample_size": 200})
    review = container.review
    with session_scope(container.session_factory) as db:
        dataset = DatasetRepository(db)
        items = dataset.pending_review_items()
        assert len(items) >= 3 and all(i.reason for i in items)
        version = dataset.current_version()
        cases_before = len(dataset.active_test_cases())

        similar = review.similar_cases(db, items[0].candidate, k=3)
        assert len(similar) == 3 and similar[0][1] >= similar[-1][1]

        note = review.annotate(db, items[0].id, "alice", "checking with support")
        assert note.item.status == "pending"

        approved = review.approve(db, items[0].id, "alice", "ok")
        assert approved.applied and approved.test_case is not None
        assert approved.test_case.source == "human_reviewed"
        assert approved.test_case.current_label.confidence == 1.0
        assert approved.item.status == "approved" and approved.item.resolved_at

        edit = LabelEdit(expected_quality_score=5, reference_answer="A curated answer.")
        edited = review.edit(db, items[1].id, "alice", edit, "fixed reference")
        assert edited.item.status == "edited"
        assert edited.test_case.current_label.reference_answer == "A curated answer."
        assert edited.decision.edited_labels == {
            "expected_quality_score": 5,
            "reference_answer": "A curated answer.",
        }

        rejected = review.reject(db, items[2].id, "alice", "bad case")
        assert rejected.test_case is None and rejected.item.candidate.status == "rejected"

        # A second opinion never changes the dataset.
        second = review.reject(db, items[0].id, "bob", "disagree")
        assert not second.applied and second.item.status == "approved"
        review.approve(db, items[2].id, "bob")

        assert dataset.current_version() == version + 2
        assert len(dataset.active_test_cases()) == cases_before + 2

        case = approved.test_case
        updated = review.edit_test_case(
            db, case.id, "carol", LabelEdit(expected_behavior=ExpectedBehavior.REFUSE)
        )
        assert [lb.version for lb in updated.labels] == [1, 2]
        assert [lb.is_current for lb in updated.labels] == [False, True]
        assert updated.current_label.expected_behavior == "refuse"
        review.add_note(db, case.id, "carol", "double-check next quarter")
        assert "carol" in case.notes
        review.retire_test_case(db, case.id, "carol", "obsolete policy")
        with pytest.raises(ConflictError):
            review.retire_test_case(db, case.id, "carol")
        with pytest.raises(NotFoundError):
            review.approve(db, "missing", "alice")
        assert len(dataset.active_test_cases()) == cases_before + 1

        stats = container.metrics.reviewer_stats(db)
        assert stats["double_reviewed_items"] == 2
        assert stats["percent_agreement"] == 0.0 and stats["cohen_kappa"] is not None
        assert {r["reviewer"] for r in stats["reviewers"]} == {"alice", "bob"}
        quality = container.metrics.quality(db)
        assert quality["by_source"]["human_reviewed"] == 1
        assert quality["auto_label_accuracy_reviewed"] == pytest.approx(1 / 3, abs=0.01)


def test_metrics(seeded_container: Container) -> None:
    with session_scope(seeded_container.session_factory) as db:
        metrics = seeded_container.metrics
        coverage = metrics.coverage(db)
        assert coverage["clusters"] and coverage["total_test_cases"] > 0
        assert all(
            set(r["by_difficulty"]) == set(coverage["difficulties"]) for r in coverage["clusters"]
        )
        assert {r["status"] for r in coverage["clusters"]} <= {"sparse", "well_tested"}
        growth = metrics.growth(db)
        assert growth["series"][-1]["total"] == growth["total"] == coverage["total_test_cases"]
        assert growth["versions"][0]["version"] == 1
        fresh = metrics.freshness(db, window_days=30)
        assert fresh["freshness_pct"] == 100.0
        assert metrics.freshness(db, window_days=1)["freshness_pct"] < 100.0
        assert isinstance(metrics.underrepresented_features(db), frozenset)


async def test_evaluation_run_and_regression(seeded_container: Container) -> None:
    container = seeded_container
    with session_scope(container.session_factory) as db:
        baseline = await container.evaluation.arun(
            db, EndpointConfig(provider="mock", model="mock-model-v1"), name="baseline"
        )
        candidate = await container.evaluation.arun(
            db, EndpointConfig(provider="mock", model="mock-model-v2"), name="candidate"
        )

        assert baseline.status == candidate.status == "completed"
        assert baseline.dataset_size == len(DatasetRepository(db).active_test_cases())
        assert baseline.dataset_version == 1 and baseline.code_version
        assert baseline.endpoint_config["model"] == "mock-model-v1"
        assert "api_key" not in baseline.endpoint_config  # only the env-var name is stored
        assert baseline.pass_rate > candidate.pass_rate

        report = baseline.report
        assert set(report) >= {
            "overall",
            "by_category",
            "by_difficulty",
            "by_metric",
            "failed_cases",
            "usage",
            "run",
        }
        assert report["overall"]["total"] == baseline.dataset_size
        assert report["overall"]["passed"] + report["overall"]["failed"] == baseline.dataset_size
        assert sum(g["total"] for g in report["by_category"].values()) == baseline.dataset_size
        assert {"behavior", "required_assertions", "forbidden_assertions"} <= set(
            report["by_metric"]
        )
        assert report["usage"]["input_tokens"] > 0 and report["usage"]["estimated_cost_usd"] > 0
        assert all(f["failure_reasons"] for f in report["failed_cases"])

        runs = RunRepository(db)
        results = runs.results_for_run(candidate.id)
        assert len(results) == candidate.dataset_size and all(r.raw_response for r in results)
        assert runs.regression_report_for(baseline.id) is None  # nothing to compare with
        regression = runs.regression_report_for(candidate.id)
        assert regression is not None and regression.baseline_run_id == baseline.id
        assert regression.has_regression and regression.new_failures > 0
        assert regression.report["severity"] == "major"
        assert regression.report["category_shifts"] or regression.report["difficulty_shifts"]
        # Reports are immutable: comparing again returns the stored report.
        again = container.regression.compare_run(db, candidate)
        assert again is not None and again.id == regression.id
        assert db.scalar(select(func.count()).select_from(RegressionReport)) == 1
        assert db.scalar(select(func.count()).select_from(EvaluationResult)) == 2 * len(results)


async def test_evaluation_captures_endpoint_errors_and_isolates_groups(
    seeded_container: Container,
) -> None:
    class FlakyClient:
        model_name = "flaky"

        async def acomplete(self, messages: Sequence[Message], **_: Any) -> ChatResult:
            if "refund" in messages[-1]["content"].lower():
                raise LLMError("chat completion failed: HTTP 503")
            return ChatResult(text="I can't help with that.", input_tokens=3, output_tokens=4)

    container = make_container(target_client_factory=lambda _: FlakyClient())
    with session_scope(seeded_container.session_factory) as db:
        run = await container.evaluation.arun(
            db, EndpointConfig(provider="mock", model="flaky"), comparison_group="experiments"
        )
        assert run.status == "completed" and run.comparison_group == "experiments"
        assert run.report["overall"]["errors"] > 0
        errored = [r for r in RunRepository(db).results_for_run(run.id) if r.error]
        assert errored and all(not r.passed and r.quality_score == 1.0 for r in errored)
        # First run in its own comparison group: no regression report.
        assert RunRepository(db).regression_report_for(run.id) is None


def test_evaluation_requires_a_dataset(container: Container) -> None:
    with session_scope(container.session_factory) as db, pytest.raises(InvalidRequestError):
        container.evaluation.run(db)


async def test_openai_compatible_client_retries_and_parses() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        assert request.headers["Authorization"] == "Bearer test-key"
        if attempts["n"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )

    client = OpenAICompatibleChatClient(
        "http://llm.test/v1/",
        "m",
        "test-key",
        retry_backoff_s=0.0,
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
    )
    result = client.complete([{"role": "user", "content": "hi"}])
    assert (result.text, result.input_tokens, result.output_tokens, result.retries) == (
        "hello",
        7,
        2,
        1,
    )
    assert (await client.acomplete([{"role": "user", "content": "hi"}])).text == "hello"

    always_down = OpenAICompatibleChatClient(
        "http://llm.test/v1",
        "m",
        max_retries=1,
        retry_backoff_s=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    with pytest.raises(LLMError, match="HTTP 500"):
        always_down.complete([{"role": "user", "content": "hi"}])
    unauthorized = OpenAICompatibleChatClient(
        "http://llm.test/v1",
        "m",
        max_retries=3,
        retry_backoff_s=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    )
    with pytest.raises(LLMError, match="HTTP 401"):  # not retried
        unauthorized.complete([{"role": "user", "content": "hi"}])
