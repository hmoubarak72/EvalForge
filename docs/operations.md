# Operations guide

## Database

```bash
make migrate                                    # alembic upgrade head
python -m alembic revision --autogenerate -m "describe the change"   # after editing models
python -m alembic check                         # fails if models and migrations differ
python -m alembic downgrade -1                  # roll back one revision
```

`EVALFORGE_DATABASE_URL` selects the database. For PostgreSQL install the `postgres` extra and
use `postgresql+psycopg://user:password@host:5432/dbname`.

## Nightly schedule

- In-process: `EVALFORGE_SCHEDULER__ENABLED=true` on the API service.
- Dedicated worker: `evalforge scheduler` (the Compose `pipeline` service). Run one instance.
- `EVALFORGE_SCHEDULER__CRON` is a five-field crontab expression in
  `EVALFORGE_SCHEDULER__TIMEZONE`. `EVALFORGE_SCHEDULER__RUN_EVALUATION=true` evaluates the
  default endpoint after each successful pipeline run.
- Prefer an external scheduler? Call `evalforge pipeline && evalforge evaluate` from cron,
  a Kubernetes CronJob or an Airflow task; both commands exit non-zero on failure.

## Recovering a failed run

```bash
curl localhost:8000/api/v1/pipeline/runs?limit=5          # find the failed run and its error
curl -X POST localhost:8000/api/v1/pipeline/runs/<id>/resume
```

Completed stages are not repeated. Starting a fresh run is also safe: ingestion deduplicates,
sampling skips interactions that were already sampled, classification and labeling skip
interactions that already have results.

## Logs

JSON lines on stderr. Useful events:

| Event | Meaning |
| --- | --- |
| `pipeline.started` / `pipeline.stage_completed` / `pipeline.stage_failed` / `pipeline.finished` | Run lifecycle |
| `ingestion.finished` | Per-source metrics (received, accepted, rejected, duplicates, redactions, failures) |
| `ingestion.source_failed` | A source could not be read |
| `sampling.finished`, `classification.finished`, `labeling.finished`, `dedup.finished` | Stage summaries |
| `clustering.cluster_created`, `clustering.late_assignment` | Cluster changes |
| `classification.judge_fallback`, `labeling.llm_fallback`, `clustering.llm_naming_failed`, `scoring.grader_failed` | An LLM role fell back to its heuristic |
| `evaluation.started` / `evaluation.finished` / `evaluation.failed` | Evaluation lifecycle |
| `regression.compared` | Severity and new-failure count |
| `api.request`, `api.unhandled_error` | HTTP access and unexpected errors |

Filter by `correlation_id` to follow one pipeline run, evaluation or request end to end.

## Suggested alerts

| Signal | Source | Suggested condition |
| --- | --- | --- |
| Pipeline failed | `pipeline_runs.status` | Any `failed` run |
| Schema drift | `ingestion_runs` | Rejected / received above 2% |
| No new traffic | `ingestion_runs` | Zero accepted for 24 h |
| LLM fallback rate | logs | More than 5% of judge/labeler calls falling back |
| Review backlog | `GET /metrics/quality` | `review_queue_pending` above your reviewers' weekly capacity |
| Coverage gaps | `GET /metrics/coverage` | A high-traffic cluster stays `sparse` for a week |
| Model regression | `GET /regressions` | `severity = major` |

## Backups and retention

- Back up the PostgreSQL database; the dataset, labels, review decisions and run history are
  not reproducible from logs alone once reviewers have edited labels.
- `interaction_logs` grows fastest. Unsampled interactions older than your retention window
  can be deleted; sampled ones are referenced by test cases (`ON DELETE SET NULL`), so the
  dataset survives log deletion.
- `scripts/export_dataset.py` writes the active dataset as JSON Lines for archiving or use in
  another harness.

## Security checklist before exposing the service

- [ ] `EVALFORGE_AUTH__MODE=token` and a long random `EVALFORGE_AUTH_TOKEN`
- [ ] API and dashboard behind TLS and an identity-aware proxy
- [ ] Database credentials changed from the Compose defaults
- [ ] Redaction rules reviewed against your own identifiers; metadata keys reviewed
- [ ] Model-provider data-retention terms checked for the judge, labeler and embedding models
