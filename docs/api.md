# API guide

Base URL: `http://localhost:8000`. Versioned routes live under `/api/v1`. Interactive
documentation: `/docs` (Swagger UI), `/redoc`, raw schema at `/openapi.json`.

## Conventions

- **Pagination:** `?limit=50&offset=0` (limit 1-500). Responses: `{"items": [...], "total": n, "limit": 50, "offset": 0}`.
- **Errors:** `{"error": {"code": "not_found", "message": "...", "correlation_id": "...", "details": [...]}}`

  | Status | `code` | When |
  | --- | --- | --- |
  | 400 | `invalid_request` | Well-formed but not executable (e.g. evaluating an empty dataset) |
  | 401 | `unauthorized` | Missing/invalid `X-API-Key` in token mode |
  | 404 | `not_found` | Unknown ID |
  | 409 | `conflict` | State conflict (e.g. retiring a retired case, approving an exact duplicate) |
  | 422 | `validation_error` | Body or query validation failed; `details` lists fields |
  | 500 | `internal_error` | Unexpected error (logged with the correlation ID) |

- **Headers:** `X-API-Key` (writes, token mode), `X-Actor` (name recorded on review decisions
  and audit events), `X-Correlation-ID` (optional in, always out).

## Typical session

```bash
API=http://localhost:8000/api/v1

# 1. Register a source and run the pipeline
curl -X POST $API/sources -H 'Content-Type: application/json' \
  -d '{"name": "app-logs", "kind": "json_logs", "config": {"path": "data/demo/app_logs.jsonl"}}'
curl -X POST $API/pipeline/runs -H 'Content-Type: application/json' \
  -d '{"strategy": "signal_boosted", "sample_size": 80, "seed": 7}'

# 2. Or trigger single stages
curl -X POST $API/pipeline/stages/ingestion
curl -X POST $API/pipeline/stages/sampling -H 'Content-Type: application/json' -d '{"sample_size": 50}'
curl -X POST $API/pipeline/stages/classification
curl -X POST $API/pipeline/stages/labeling
curl -X POST $API/pipeline/stages/deduplication

# 3. Explore
curl "$API/interactions?feature=customer_support&has_error=false&limit=5"
curl "$API/classifications?outliers_only=true"
curl $API/clusters
curl "$API/candidates?status=duplicate"
curl "$API/test-cases?difficulty=adversarial&source=auto_labeled&min_quality=4"

# 4. Review
curl "$API/review-queue?status=pending"
curl "$API/review-queue/<item-id>?similar=5"
curl -X POST $API/review-queue/<item-id>/approve -H 'X-Actor: alice' \
  -H 'Content-Type: application/json' -d '{"notes": "labels look right"}'
curl -X POST $API/review-queue/<item-id>/edit -H 'X-Actor: alice' -H 'Content-Type: application/json' \
  -d '{"labels": {"expected_behavior": "answer", "reference_answer": "Returns are accepted within 30 days.",
       "required_assertions": [{"type": "contains", "value": "30 days", "description": "States the window"}]},
       "notes": "fixed reference"}'
curl -X POST $API/review-queue/<item-id>/reject -H 'Content-Type: application/json' -d '{"notes": "not a useful case"}'
curl -X POST $API/review-queue/<item-id>/notes  -H 'Content-Type: application/json' -d '{"notes": "checking with support"}'

# 5. Evaluate and inspect
curl -X POST $API/evaluations -H 'Content-Type: application/json' \
  -d '{"endpoint": {"provider": "openai_compatible", "base_url": "https://api.openai.com/v1",
       "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY", "temperature": 0, "max_tokens": 1024,
       "timeout_s": 60, "max_retries": 2, "concurrency": 8},
       "name": "gpt-4o-mini nightly", "grading_mode": "deterministic"}'
curl $API/evaluations/<run-id>/report
curl "$API/evaluations/<run-id>/results?passed=false"
curl $API/evaluations/<run-id>/regression

# 6. Dataset health
curl $API/metrics/coverage
curl $API/metrics/growth
curl "$API/metrics/freshness?window_days=30"
curl $API/metrics/quality
curl $API/metrics/reviewers
```

## Endpoint reference

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (database reachable); 503 when not |
| POST | `/api/v1/sources` | Create or update (by name) a log source |
| GET | `/api/v1/sources` | List sources with cursors |
| GET | `/api/v1/ingestion-runs` | Ingestion history with metrics |
| POST | `/api/v1/pipeline/runs` | Run all or selected stages; body: `stages?`, `strategy?`, `sample_size?`, `seed?`, `source_ids?` |
| POST | `/api/v1/pipeline/stages/{stage}` | Run one stage |
| POST | `/api/v1/pipeline/runs/{id}/resume` | Resume a failed run |
| GET | `/api/v1/pipeline/runs`, `/{id}` | Run history and per-stage results |
| GET | `/api/v1/interactions`, `/{id}` | Filters: `feature`, `model`, `feedback`, `has_error`, `since`, `until`; detail includes the classification |
| GET | `/api/v1/classifications` | Filters: `cluster_id`, `difficulty`, `outliers_only` |
| GET | `/api/v1/clusters`, `/{id}` | Clusters ordered by traffic |
| GET | `/api/v1/candidates`, `/{id}` | Filters: `status`, `cluster_id`; detail includes every labeling pass |
| GET | `/api/v1/test-cases`, `/{id}` | Filters: `category`, `difficulty`, `source`, `status`, `case_type`, `min_quality`, `max_quality`, `created_since`, `created_until`, `search` |
| PATCH | `/api/v1/test-cases/{id}/labels` | New label version |
| POST | `/api/v1/test-cases/{id}/notes` | Add a note |
| POST | `/api/v1/test-cases/{id}/retire` | Retire a case |
| GET | `/api/v1/review-queue`, `/{id}` | Queue (default `status=pending`); detail includes decisions and similar cases |
| POST | `/api/v1/review-queue/{id}/approve` | Approve as-is |
| POST | `/api/v1/review-queue/{id}/edit` | Approve with edited labels |
| POST | `/api/v1/review-queue/{id}/reject` | Reject |
| POST | `/api/v1/review-queue/{id}/notes` | Annotate without resolving |
| POST | `/api/v1/evaluations` | Run the dataset against an endpoint; body: `endpoint?`, `name?`, `grading_mode?`, `comparison_group?`, `baseline_run_id?` |
| GET | `/api/v1/evaluations`, `/{id}` | Filters: `comparison_group`, `model_name`, `status` |
| GET | `/api/v1/evaluations/{id}/report` | Structured report |
| GET | `/api/v1/evaluations/{id}/results` | Per-case raw responses and scores; filter `passed` |
| GET | `/api/v1/evaluations/{id}/regression` | Regression report (404 for a first run) |
| GET | `/api/v1/regressions` | All regression reports |
| GET | `/api/v1/metrics/coverage` | Per-cluster coverage and heatmap data |
| GET | `/api/v1/metrics/growth` | Daily/cumulative counts and dataset versions |
| GET | `/api/v1/metrics/freshness` | Share of cases from the last `window_days` |
| GET | `/api/v1/metrics/quality` | Composition, auto-accept rate, review backlog |
| GET | `/api/v1/metrics/reviewers` | Throughput and inter-annotator agreement |

A review action on an already resolved item returns `"applied": false`: the decision is stored
as a second opinion for agreement metrics and the dataset is not changed.

## Assertion types

| `type` | `value` | Passes when |
| --- | --- | --- |
| `contains` | string | Response contains it (case-insensitive; numbers respect digit boundaries) |
| `contains_any` | list of strings | At least one is contained |
| `regex` | pattern | `re.search` matches |
| `min_length` / `max_length` | integer | Response length bound (characters) |
| `concept` | description | LLM grader says the concept is expressed; keyword coverage ≥ 50% as fallback |

Required assertions must pass; forbidden assertions must not.
