"""API endpoint tests (FastAPI TestClient, in-memory database, mock models)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from evalforge.api.app import create_app
from tests.conftest import (
    DB_COLUMN_MAP,
    make_container,
    make_settings,
    write_demo_sources,
)

API = "/api/v1"


@pytest.fixture(scope="module")
def client(
    tmp_path_factory: pytest.TempPathFactory, demo_records: list[dict[str, Any]]
) -> Iterator[TestClient]:
    """A client whose database has sources registered and one pipeline run."""
    container = make_container()
    paths = write_demo_sources(demo_records, tmp_path_factory.mktemp("api-sources"))
    with TestClient(create_app(container=container)) as test_client:
        sources = [
            {"name": "json", "kind": "json_logs", "config": {"path": paths["json"].as_posix()}},
            {"name": "otel", "kind": "otel_traces", "config": {"path": paths["otel"].as_posix()}},
            {
                "name": "db",
                "kind": "database",
                "config": {
                    "url": f"sqlite:///{paths['db'].as_posix()}",
                    "table": "llm_calls",
                    "column_map": DB_COLUMN_MAP,
                },
            },
        ]
        for source in sources:
            assert test_client.post(f"{API}/sources", json=source).status_code == 201
        response = test_client.post(f"{API}/pipeline/runs", json={"sample_size": 200, "seed": 3})
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "completed"
        yield test_client


def test_health_ready_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["database"] is True
    assert ready.headers["X-Correlation-ID"]
    echoed = client.get("/health", headers={"X-Correlation-ID": "abc123"})
    assert echoed.headers["X-Correlation-ID"] == "abc123"
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/api/v1/pipeline/runs",
        "/api/v1/review-queue/{item_id}/approve",
        "/api/v1/evaluations/{run_id}/regression",
        "/api/v1/metrics/coverage",
    ):
        assert path in paths


def test_sources_and_ingestion_runs(client: TestClient) -> None:
    sources = client.get(f"{API}/sources").json()
    assert {s["name"] for s in sources} == {"json", "otel", "db"}
    assert all(s["cursor_timestamp"] for s in sources)
    updated = client.post(
        f"{API}/sources",
        json={"name": "json", "kind": "json_logs", "config": sources[0]["config"], "enabled": True},
    )
    assert updated.status_code == 201 and len(client.get(f"{API}/sources").json()) == 3
    runs = client.get(f"{API}/ingestion-runs").json()
    assert runs["total"] == 3
    assert sum(r["records_accepted"] for r in runs["items"]) == 320
    bad = client.post(f"{API}/sources", json={"name": "x", "kind": "carrier_pigeon"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error"
    assert bad.json()["error"]["details"][0]["location"] == "body.kind"


def test_pipeline_runs_and_single_stages(client: TestClient) -> None:
    runs = client.get(f"{API}/pipeline/runs").json()
    run_id = runs["items"][-1]["id"]
    detail = client.get(f"{API}/pipeline/runs/{run_id}").json()
    assert set(detail["stage_results"]) == {
        "ingestion",
        "sampling",
        "classification",
        "labeling",
        "deduplication",
    }
    stage = client.post(f"{API}/pipeline/stages/ingestion")
    assert stage.status_code == 201 and stage.json()["stages"] == ["ingestion"]
    assert stage.json()["stage_results"]["ingestion"]["records_accepted"] == 0
    assert client.post(f"{API}/pipeline/stages/training").status_code == 422
    assert client.post(f"{API}/pipeline/runs", json={"sample_size": 0}).status_code == 422
    missing = client.get(f"{API}/pipeline/runs/nope")
    assert missing.status_code == 404
    assert missing.json()["error"] == {
        "code": "not_found",
        "message": "pipeline_run 'nope' was not found",
        "correlation_id": missing.headers["X-Correlation-ID"],
    }
    assert client.post(f"{API}/pipeline/runs/{run_id}/resume").status_code == 400


def test_interactions_pagination_filters_and_detail(client: TestClient) -> None:
    page = client.get(f"{API}/interactions", params={"limit": 25, "offset": 300}).json()
    assert page["total"] == 320 and len(page["items"]) == 20 and page["limit"] == 25
    support = client.get(f"{API}/interactions", params={"feature": "customer_support"}).json()
    assert 0 < support["total"] < 320
    assert all(i["application_feature"] == "customer_support" for i in support["items"])
    errors = client.get(f"{API}/interactions", params={"has_error": True}).json()
    assert errors["total"] > 0 and all(i["error_status"] for i in errors["items"])
    assert client.get(f"{API}/interactions", params={"limit": 0}).status_code == 422
    assert client.get(f"{API}/interactions", params={"limit": 501}).status_code == 422

    classified = client.get(f"{API}/classifications", params={"outliers_only": True}).json()
    assert classified["total"] > 0 and "embedding" not in classified["items"][0]
    interaction_id = classified["items"][0]["interaction_id"]
    detail = client.get(f"{API}/interactions/{interaction_id}").json()
    assert detail["classification"]["is_outlier"] is True
    assert isinstance(detail["metadata"], dict)
    assert "@example.com" not in detail["user_prompt"]
    assert client.get(f"{API}/interactions/nope").status_code == 404


def test_clusters_candidates_and_test_cases(client: TestClient) -> None:
    clusters = client.get(f"{API}/clusters").json()
    assert len(clusters) >= 5
    assert client.get(f"{API}/clusters/{clusters[0]['id']}").json()["name"] == clusters[0]["name"]
    assert client.get(f"{API}/clusters/nope").status_code == 404

    duplicates = client.get(f"{API}/candidates", params={"status": "duplicate"}).json()
    assert duplicates["total"] > 0
    assert all(c["max_similarity"] > 0.92 for c in duplicates["items"])
    candidate = client.get(f"{API}/candidates/{duplicates['items'][0]['id']}").json()
    assert candidate["interaction"]["user_prompt"] and candidate["classification"]["difficulty"]

    cases = client.get(f"{API}/test-cases", params={"limit": 500}).json()
    assert cases["total"] > 10
    adversarial = client.get(f"{API}/test-cases", params={"difficulty": "adversarial"}).json()
    assert 0 < adversarial["total"] < cases["total"]
    assert all(c["current_label"]["expected_behavior"] == "refuse" for c in adversarial["items"])
    top = client.get(f"{API}/test-cases", params={"min_quality": 5}).json()
    assert all(c["current_label"]["expected_quality_score"] == 5 for c in top["items"])
    assert client.get(f"{API}/test-cases", params={"source": "human_reviewed"}).json()["total"] == 0
    assert (
        client.get(f"{API}/test-cases", params={"created_since": "2031-01-01T00:00:00Z"}).json()[
            "total"
        ]
        == 0
    )
    searched = client.get(f"{API}/test-cases", params={"search": "refund"}).json()
    assert all("refund" in c["user_prompt"].lower() for c in searched["items"])

    case_id = cases["items"][0]["id"]
    edited = client.patch(
        f"{API}/test-cases/{case_id}/labels",
        json={"labels": {"expected_quality_score": 3}, "notes": "relaxed"},
        headers={"X-Actor": "dana"},
    )
    assert edited.status_code == 200
    body = edited.json()
    assert [label["version"] for label in body["labels"]] == [1, 2]
    assert body["current_label"]["created_by"] == "dana" and body["source"] == "human_reviewed"
    bad = client.patch(
        f"{API}/test-cases/{case_id}/labels", json={"labels": {"expected_quality_score": 9}}
    )
    assert bad.status_code == 422
    noted = client.post(f"{API}/test-cases/{case_id}/notes", json={"notes": "watch this"})
    assert "watch this" in noted.json()["notes"]
    assert client.post(f"{API}/test-cases/{case_id}/retire", json={}).json()["status"] == "retired"
    assert client.post(f"{API}/test-cases/{case_id}/retire", json={}).status_code == 409
    assert client.get(f"{API}/test-cases/nope").status_code == 404


def test_review_queue_actions(client: TestClient) -> None:
    queue = client.get(f"{API}/review-queue").json()
    assert queue["total"] >= 3
    first, second, third = (item["id"] for item in queue["items"][:3])
    detail = client.get(f"{API}/review-queue/{first}", params={"similar": 3}).json()
    assert detail["reason"] and len(detail["similar_cases"]) == 3
    assert detail["candidate"]["proposed_labels"]["expected_behavior"]
    assert detail["candidate"]["interaction"]["user_prompt"]

    noted = client.post(f"{API}/review-queue/{first}/notes", json={"notes": "needs context"})
    assert noted.status_code == 200 and noted.json()["item"]["status"] == "pending"
    approved = client.post(
        f"{API}/review-queue/{first}/approve", json={"notes": "ok"}, headers={"X-Actor": "alice"}
    ).json()
    assert approved["applied"] and approved["test_case"]["source"] == "human_reviewed"
    assert approved["decision"]["reviewer"] == "alice"
    opinion = client.post(f"{API}/review-queue/{first}/reject", headers={"X-Actor": "bob"}).json()
    assert opinion["applied"] is False and opinion["item"]["status"] == "approved"

    edited = client.post(
        f"{API}/review-queue/{second}/edit",
        json={"labels": {"reference_answer": "Curated.", "expected_behavior": "answer"}},
    ).json()
    assert edited["item"]["status"] == "edited"
    assert edited["test_case"]["current_label"]["reference_answer"] == "Curated."
    rejected = client.post(f"{API}/review-queue/{third}/reject", json={"notes": "noise"}).json()
    assert rejected["item"]["status"] == "rejected" and rejected["test_case"] is None

    assert client.get(f"{API}/review-queue").json()["total"] == queue["total"] - 3
    assert client.get(f"{API}/review-queue", params={"status": "approved"}).json()["total"] == 1
    assert client.post(f"{API}/review-queue/nope/approve").status_code == 404
    assert client.post(f"{API}/review-queue/{third}/notes", json={"notes": ""}).status_code == 422
    reviewers = client.get(f"{API}/metrics/reviewers").json()
    assert reviewers["double_reviewed_items"] == 1 and reviewers["percent_agreement"] == 0.0


def test_evaluations_reports_and_regression(client: TestClient) -> None:
    first = client.post(
        f"{API}/evaluations",
        json={"endpoint": {"provider": "mock", "model": "mock-model-v1"}, "name": "baseline"},
    )
    assert first.status_code == 201, first.text
    baseline = first.json()
    assert baseline["status"] == "completed" and 0.5 < baseline["pass_rate"] <= 1.0
    second = client.post(
        f"{API}/evaluations", json={"endpoint": {"provider": "mock", "model": "mock-model-v2"}}
    ).json()
    assert second["pass_rate"] < baseline["pass_rate"]

    listing = client.get(f"{API}/evaluations", params={"model_name": "mock-model-v2"}).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == second["id"]
    assert client.get(f"{API}/evaluations/{second['id']}").json()["dataset_size"] > 0
    report = client.get(f"{API}/evaluations/{second['id']}/report").json()
    assert report["run"]["id"] == second["id"]
    assert report["report"]["overall"]["failed"] == len(report["report"]["failed_cases"])
    failed = client.get(
        f"{API}/evaluations/{second['id']}/results", params={"passed": False, "limit": 500}
    ).json()
    assert failed["total"] == report["report"]["overall"]["failed"]
    assert all(r["failure_reasons"] and r["raw_response"] for r in failed["items"])

    regression = client.get(f"{API}/evaluations/{second['id']}/regression").json()
    assert regression["baseline_run_id"] == baseline["id"]
    assert regression["has_regression"] and regression["new_failures"] > 0
    assert regression["report"]["severity"] == "major"
    assert client.get(f"{API}/evaluations/{baseline['id']}/regression").status_code == 404
    assert client.get(f"{API}/regressions").json()["total"] == 1
    assert client.get(f"{API}/evaluations/nope/report").status_code == 404
    llm = client.post(f"{API}/evaluations", json={"grading_mode": "llm"})
    assert llm.status_code == 400 and "judge model" in llm.json()["error"]["message"]
    assert (
        client.post(
            f"{API}/evaluations", json={"endpoint": {"provider": "mock", "api_key": "x"}}
        ).status_code
        == 422
    )


def test_metrics_endpoints(client: TestClient) -> None:
    coverage = client.get(f"{API}/metrics/coverage").json()
    assert coverage["clusters"] and coverage["difficulties"] == [
        "simple",
        "moderate",
        "hard",
        "adversarial",
    ]
    growth = client.get(f"{API}/metrics/growth").json()
    assert growth["total"] > 0 and growth["series"]
    assert (
        client.get(f"{API}/metrics/freshness", params={"window_days": 30}).json()["freshness_pct"]
        == 100.0
    )
    assert client.get(f"{API}/metrics/freshness", params={"window_days": 0}).status_code == 422
    quality = client.get(f"{API}/metrics/quality").json()
    assert quality["active_test_cases"] > 0 and quality["duplicates_skipped"] > 0


def test_token_auth_protects_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVALFORGE_AUTH_TOKEN", "s3cret-token")
    settings = make_settings(auth={"mode": "token"})
    with TestClient(create_app(container=make_container(settings=settings))) as secured:
        body = {"name": "j", "kind": "json_logs", "config": {"path": str(tmp_path / "x.jsonl")}}
        assert secured.get(f"{API}/sources").status_code == 200  # reads stay open
        denied = secured.post(f"{API}/sources", json=body)
        assert denied.status_code == 401 and denied.json()["error"]["code"] == "unauthorized"
        assert (
            secured.post(f"{API}/sources", json=body, headers={"X-API-Key": "wrong"}).status_code
            == 401
        )
        assert secured.post(f"{API}/pipeline/runs", json={}).status_code == 401
        assert secured.post(f"{API}/evaluations", json={}).status_code == 401
        allowed = secured.post(f"{API}/sources", json=body, headers={"X-API-Key": "s3cret-token"})
        assert allowed.status_code == 201
        monkeypatch.delenv("EVALFORGE_AUTH_TOKEN")
        assert (
            secured.post(
                f"{API}/sources", json=body, headers={"X-API-Key": "s3cret-token"}
            ).status_code
            == 503
        )
