# EvalForge

**Turn production LLM traffic into a self-growing, production-representative evaluation dataset.**

The hardest part of LLM evaluation is not the harness, it is the dataset. Hand-curated golden
sets go stale. EvalForge continuously mines production LLM logs, finds the representative,
edge-case and failure-prone interactions, labels them as evaluation cases, deduplicates them
and keeps a versioned eval dataset growing. It then runs any model endpoint against that
dataset, reports pass rates and quality by category and difficulty, and flags regressions
between runs. Low-confidence labels go to a human review queue in a Streamlit dashboard.

> **Demo result (fully offline, reproducible with `make seed`):** from two weeks of simulated
> production traffic (440 interactions, three log sources) EvalForge discovered 21 traffic
> clusters and built a dataset of **110 test cases** (91 auto-labeled, 19 human-reviewed),
> skipped 135 near-duplicates, and caught a **major regression** when a second model version
> dropped from a 93.6% to an 81.8% pass rate (15 new failures, concentrated in document QA and
> prompt-injection handling).

## Contents

- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Running the pipeline](#running-the-full-pipeline-locally)
- [Running evaluations](#running-evaluations)
- [Using the dashboard](#using-the-dashboard)
- [REST API](#rest-api)
- [Environment variables](#environment-variable-reference)
- [Extending EvalForge](#extending-evalforge)
- [Testing and quality](#testing-and-quality)
- [Limitations and production readiness](#limitations-and-production-readiness)

Further reading: [ARCHITECTURE.md](ARCHITECTURE.md) · [DECISIONS.md](DECISIONS.md) ·
[docs/api.md](docs/api.md) · [docs/demo-walkthrough.md](docs/demo-walkthrough.md) ·
[docs/operations.md](docs/operations.md)

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        J[JSON / JSONL logs]
        O[OpenTelemetry trace export]
        D[(Application database)]
    end

    subgraph Pipeline["Nightly pipeline (APScheduler)"]
        direction LR
        I["1 Ingest<br/>validate · redact PII · dedup"]
        S["2 Sample<br/>uniform · stratified · signal-boosted"]
        C["3 Classify<br/>embed · HDBSCAN · outliers<br/>difficulty · LLM judge"]
        L["4a Label<br/>reference / rubric · assertions<br/>multi-pass agreement"]
        R["4b Dedup + route<br/>cosine > 0.92 skipped"]
    end

    J --> I
    O --> I
    D --> I
    I --> LOGS[(interaction_logs)]
    LOGS --> S --> C --> L --> R
    R -- "high confidence" --> DS[(Eval dataset<br/>versioned)]
    R -- "low confidence / conflicts" --> Q[Review queue]
    Q -- "approve · edit · reject" --> DS

    DS --> E["5 Evaluation runner<br/>any OpenAI-compatible endpoint"]
    E --> RES[(Runs + raw results)]
    RES --> G["6 Regression tracker"]

    API[FastAPI service] --- LOGS
    API --- DS
    API --- RES
    UI[Streamlit dashboard] --- Q
    UI --- DS
    UI --- G
    DS -. "coverage gaps steer<br/>sampling and labeling" .-> S
```

Layers (each depends only on the ones below it):

| Layer | Package | Responsibility |
| --- | --- | --- |
| UI / API | `evalforge.ui`, `evalforge.api` | Streamlit dashboard, FastAPI routers and schemas |
| Orchestration | `evalforge.pipeline`, `evalforge.container` | Stage ordering, resumability, scheduler, dependency wiring |
| Domain services | `evalforge.services` | One module per concern: ingestion, redaction, sampling, clustering, classification, labeling, dedup, review, scoring, evaluation, regression, metrics |
| Repositories | `evalforge.repositories` | All SQL queries |
| Adapters | `evalforge.adapters` | Log sources and model providers behind small interfaces |
| Domain / DB | `evalforge.domain`, `evalforge.db` | Pydantic schemas, enums, text heuristics; SQLAlchemy models |

## Quickstart

Requirements: Python 3.11+ (tested on 3.11-3.13). Docker is optional.

### Option A: local, SQLite, no external services

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,postgres]"
cp .env.example .env            # optional; defaults work as-is

make seed                        # = python scripts/seed_demo.py --reset
make api                         # http://localhost:8000/docs
make dashboard                   # http://localhost:8501
```

`make seed` creates the schema with Alembic, writes simulated logs to `data/demo/`, replays
seven "nightly" pipeline runs with a simulated clock, simulates reviewer decisions and runs
three evaluations (the last one regresses). Everything uses deterministic mock models, so no
API key is needed and results are reproducible.

No `make` on Windows? Each target is a one-line command; see the [Makefile](Makefile), e.g.
`python -m uvicorn evalforge.api.app:app_factory --factory --port 8000` and
`python -m streamlit run src/evalforge/ui/dashboard.py`.

### Option B: Docker Compose with PostgreSQL

```bash
docker compose up --build -d                      # postgres, migrate, api, pipeline, dashboard
docker compose --profile demo run --rm seed        # load the demo
# API: http://localhost:8000/docs     Dashboard: http://localhost:8501
```

| Service | Role |
| --- | --- |
| `postgres` | Log store and application database |
| `migrate` | Applies Alembic migrations, then exits |
| `api` | FastAPI service |
| `pipeline` | Scheduler: nightly ingest → sample → classify → label → dedup, then an evaluation run |
| `dashboard` | Streamlit curation dashboard |
| `seed` (profile `demo`) | One-off demo loader |

## Running the full pipeline locally

1. **Register log sources** (once). Through the CLI, the API, or `EVALFORGE_SOURCES`:

   ```bash
   evalforge add-source app-logs json_logs --config '{"path": "logs/*.jsonl"}'
   evalforge add-source traces otel_traces --config '{"path": "exports/traces.json"}'
   evalforge add-source legacy database --config '{"url": "postgresql+psycopg://...", "table": "llm_calls", "column_map": {"user_prompt": "prompt"}}'
   ```

2. **Run the pipeline** (all stages, or a subset):

   ```bash
   evalforge pipeline                                         # full run
   evalforge pipeline --strategy stratified --sample-size 100 --seed 7
   evalforge pipeline --stages ingestion sampling             # partial run
   ```

   or `POST /api/v1/pipeline/runs`. Each stage commits separately and is idempotent; a failed
   run can be resumed with `POST /api/v1/pipeline/runs/{id}/resume`.

3. **Schedule it nightly**: set `EVALFORGE_SCHEDULER__ENABLED=true` (the API then runs the
   job in-process) or run `evalforge scheduler` as its own process (what Docker Compose does).
   The default cron is `0 2 * * *` UTC.

What a run does, in one paragraph: new log records are validated against the canonical
`LLMInteraction` schema, PII is redacted, and duplicates are dropped by fingerprint. A sample
is drawn (by default oversampling negative feedback, retries, high latency, low confidence,
content-filter events and unusual response lengths). Sampled prompts are embedded and
clustered with HDBSCAN into persistent, auto-named categories; outliers are flagged, difficulty
is estimated and a judge rates quality 1-5. The most valuable interactions become candidates
and are labeled in several passes (expected behaviour, reference answer or rubric, required and
forbidden assertions). Near-duplicates of existing cases are skipped; confident, conflict-free
candidates join the dataset automatically and the rest wait in the review queue.

## Running evaluations

```bash
# Offline: two simulated model versions (v2 regresses on document QA and injection safety)
evalforge evaluate --provider mock --model mock-model-v1
evalforge evaluate --provider mock --model mock-model-v2

# Any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, LiteLLM, Azure-compatible gateways)
export OPENAI_API_KEY=...            # the key is read from the env var named by --api-key-env
evalforge evaluate --provider openai_compatible --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini --api-key-env OPENAI_API_KEY --concurrency 8

evalforge report <evaluation-run-id>     # full report + regression report as JSON
```

or through the API:

```bash
curl -X POST localhost:8000/api/v1/evaluations -H 'Content-Type: application/json' \
  -d '{"endpoint": {"provider": "mock", "model": "mock-model-v2"}, "name": "release candidate"}'
curl localhost:8000/api/v1/evaluations/<id>/report
curl localhost:8000/api/v1/evaluations/<id>/regression
```

Every run stores the endpoint configuration (never the key itself), dataset version, code
version, all raw responses, latency, token usage, retries and errors. The report contains the
overall pass rate and average quality, pass/fail by category and by difficulty, metric-level
results, failed-case details with reasons, and cost/tokens/runtime.

**Pass criteria.** A case passes when the response shows the expected behaviour (answer /
refuse / ask for clarification), satisfies every required assertion, violates no forbidden
assertion, and scores at least `expected_quality_score - pass_score_tolerance`.

**Grading modes.** `deterministic` (default; assertions, behaviour detection, reference
similarity, rubric keyword coverage), `llm` (judge model rates helpfulness, accuracy,
completeness and semantic `concept` assertions) and `hybrid` (average of both).
LLM modes need `EVALFORGE_OFFLINE_MODE=false` and a configured `EVALFORGE_JUDGE__*` model.

**Regression tracking.** Each run is compared with the previous completed run in the same
`comparison_group` on the cases both runs share: new failures, newly passing cases, score
changes above a threshold, and category- and difficulty-level pass-rate shifts. Historical
runs and reports are never overwritten.

## Using the dashboard

`make dashboard` → http://localhost:8501

| View | What it shows |
| --- | --- |
| **Overview** | Counts, dataset growth over time (auto-labeled vs human-reviewed), distribution by category and difficulty, coverage heatmap (cluster × difficulty, sparse clusters highlighted), freshness (share of cases from the last 30 days) |
| **Dataset explorer** | Filter by category, difficulty, expected quality, review status, status, date and prompt text. Full interaction view: prompt, production response, labels, confidence, cluster, metadata. Edit labels (creates a new label version), add notes, retire a case |
| **Review queue** | Low-confidence candidates with the reason, auto-generated labels, confidence and pass agreement, the five most similar existing cases, and quick actions: approve as-is, edit then approve, reject, add note |
| **Evaluations** | Pass-rate trend, per-run report, regression banner with new failures, newly passing cases and category/difficulty shifts |
| **Clusters** | Traffic distribution by cluster and the most recent outliers |
| **Reviewers** | Throughput per reviewer and day, percent agreement and Cohen's kappa on double-reviewed items |

**Write protection.** Controlled by `EVALFORGE_AUTH__MODE`, shared by the API and dashboard:

- `local_dev` (default): writes are open to anyone who can reach the service. For local
  development only; the dashboard shows a warning banner.
- `token`: set `EVALFORGE_AUTH_TOKEN`. API writes require the `X-API-Key` header; the
  dashboard is read-only until the token is entered in the sidebar. Reads stay open.

## REST API

Interactive OpenAPI docs: http://localhost:8000/docs. Summary (all under `/api/v1`):

| Area | Endpoints |
| --- | --- |
| Health | `GET /health`, `GET /ready` (no prefix) |
| Sources | `POST /sources`, `GET /sources`, `GET /ingestion-runs` |
| Pipeline | `POST /pipeline/runs`, `POST /pipeline/stages/{stage}`, `POST /pipeline/runs/{id}/resume`, `GET /pipeline/runs[/{id}]` |
| Logs & classifier | `GET /interactions[/{id}]`, `GET /classifications`, `GET /clusters[/{id}]` |
| Dataset | `GET /candidates[/{id}]`, `GET /test-cases[/{id}]`, `PATCH /test-cases/{id}/labels`, `POST /test-cases/{id}/notes`, `POST /test-cases/{id}/retire` |
| Review | `GET /review-queue[/{id}]`, `POST /review-queue/{id}/approve\|edit\|reject\|notes` |
| Evaluation | `POST /evaluations`, `GET /evaluations[/{id}]`, `GET /evaluations/{id}/report\|results\|regression`, `GET /regressions` |
| Metrics | `GET /metrics/coverage\|growth\|freshness\|quality\|reviewers` |

List endpoints take `limit` (1-500) and `offset` and return `{items, total, limit, offset}`.
Errors use one envelope: `{"error": {"code", "message", "correlation_id", "details"?}}`.
Every response carries an `X-Correlation-ID` header that also appears in the structured logs.
Details and examples: [docs/api.md](docs/api.md).

## Environment-variable reference

All variables use the `EVALFORGE_` prefix; nested settings use `__`. The complete, commented
list is in [.env.example](.env.example). The most important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVALFORGE_DATABASE_URL` | `sqlite:///./evalforge.db` | SQLAlchemy URL. PostgreSQL: `postgresql+psycopg://user:pass@host/db` |
| `EVALFORGE_OFFLINE_MODE` | `true` | Force deterministic mock embeddings, judge and labeler |
| `EVALFORGE_DEMO_MODE` | `false` | Marks a demo environment (informational flag) |
| `EVALFORGE_LOG_LEVEL` / `EVALFORGE_LOG_JSON` | `INFO` / `true` | Structured logging |
| `EVALFORGE_SOURCES` | `[]` | JSON list of log sources synced at startup |
| `EVALFORGE_REDACTION__ENABLED` | `true` | PII redaction before persistence |
| `EVALFORGE_REDACTION__BUILTIN_RULES` | email, credit_card, ssn, phone, ip_address, api_key, iban | Built-in regex rules |
| `EVALFORGE_REDACTION__CUSTOM_RULES` | `[]` | Extra `{name, pattern, replacement}` rules |
| `EVALFORGE_REDACTION__NER_ENABLED` | `true` | Heuristic person-name detection |
| `EVALFORGE_SAMPLING__STRATEGY` | `signal_boosted` | `uniform`, `stratified` or `signal_boosted` |
| `EVALFORGE_SAMPLING__SAMPLE_SIZE` / `__SEED` | `60` / `42` | Per-run sample size and seed |
| `EVALFORGE_EMBEDDING__PROVIDER` / `__MODEL` | `mock` / `mock-embedding-256` | Embedding model |
| `EVALFORGE_CLUSTERING__MIN_CLUSTER_SIZE` | `5` | HDBSCAN minimum cluster size |
| `EVALFORGE_CLUSTERING__MIN_COHESION` | `0.55` | Minimum mean member-to-centroid similarity for a new cluster |
| `EVALFORGE_JUDGE__PROVIDER` / `__MODEL` | `mock` / `mock-judge` | Quality judge, cluster namer, LLM grader |
| `EVALFORGE_LABELING__PROVIDER` / `__MODEL` | `mock` / `mock-labeler` | Reference-answer and label generation |
| `EVALFORGE_LABELING__PASSES` | `3` | Labeling passes used to measure agreement |
| `EVALFORGE_LABELING__AUTO_ACCEPT_CONFIDENCE` | `0.8` | Confidence needed to skip human review |
| `EVALFORGE_LABELING__MAX_CANDIDATES_PER_RUN` | `40` | Cost cap on labeling per run |
| `EVALFORGE_DEDUP__SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity above which a candidate is a duplicate |
| `EVALFORGE_EVALUATION__*` | mock endpoint | Default endpoint: provider, base URL, model, API-key env var, temperature, max tokens, timeout, retries, concurrency, grading mode |
| `EVALFORGE_REGRESSION__*` | see `.env.example` | Score-change, category-shift, difficulty-shift and overall-drop thresholds |
| `EVALFORGE_SCHEDULER__ENABLED` / `__CRON` | `false` / `0 2 * * *` | Nightly pipeline schedule |
| `EVALFORGE_AUTH__MODE` | `local_dev` | `local_dev` or `token` |
| `EVALFORGE_AUTH_TOKEN` | unset | Shared write token for `token` mode |
| `OPENAI_API_KEY` (or any name) | unset | Secrets are referenced by name through `*_API_KEY_ENV` |

To use real models: set `EVALFORGE_OFFLINE_MODE=false` and, per role (`EMBEDDING`, `JUDGE`,
`LABELING`, `EVALUATION`), `__PROVIDER=openai_compatible`, `__BASE_URL`, `__MODEL` and
`__API_KEY_ENV`. Each LLM-backed role falls back to its deterministic implementation (and logs
a warning) when a model call or its JSON output is unusable.

## Extending EvalForge

### Add a log source

1. Subclass `LogSourceAdapter` and yield `RawRecord`s:

   ```python
   # my_package/kafka_source.py
   from evalforge.adapters.sources import LogSourceAdapter, RawRecord, register_adapter
   from evalforge.adapters.sources.json_logs import normalize_mapping


   class KafkaLogAdapter(LogSourceAdapter):
       kind = "kafka"

       def read(self, since=None, until=None):
           for message in consume(self.config["topic"]):  # your client
               try:
                   interaction = normalize_mapping(message)  # or build LLMInteraction directly
               except ValueError as exc:
                   yield RawRecord(None, error=str(exc))  # counted as rejected
                   continue
               if self.in_window(interaction.timestamp, since, until):
                   yield RawRecord(interaction)


   register_adapter(KafkaLogAdapter)
   ```

2. Import the module at startup and allow the new `kind` in `SourceConfig`/`SourceCreate`.
   Validation, redaction, deduplication, metrics and cursors are handled by `IngestionService`.

### Add a model provider

Implement the two small protocols in `evalforge.adapters.llm.base`:

```python
class MyChatClient:
    model_name = "my-model"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> ChatResult: ...
    async def acomplete(self, messages, *, temperature=None, max_tokens=None) -> ChatResult: ...


class MyEmbeddingClient:
    model_name = "my-embedder"

    def embed(self, texts) -> list[list[float]]: ...
```

Then either extend the factories in `evalforge/adapters/llm/__init__.py` with a new provider
name, or inject instances directly: `build_container(settings, embedder=..., judge_chat=...,
labeler_chat=..., target_client_factory=...)`. Anything that speaks the OpenAI HTTP API
already works through `openai_compatible` with a custom `base_url`.

Other extension points: `EntityRecognizer` (plug spaCy/Presidio into redaction),
`QualityJudge`, `Labeler`, `ClusterNamer`, and new assertion types in `services/scoring.py`.

## Testing and quality

```bash
make check        # ruff + mypy + pytest
make coverage     # 99 tests, ~95% line coverage of src/evalforge (UI excluded)
```

- Tests never call an external API: model roles use deterministic mocks, a scripted chat
  double, or `httpx.MockTransport`.
- Unit tests cover normalization, redaction, sampling, clustering wrappers, labeling logic,
  deduplication, scoring, regression detection and the API; integration tests run the whole
  pipeline, review workflow, evaluation, CLI, scheduler job and dashboard against local SQLite.
- Fixtures are realistic synthetic logs (normal, ambiguous, adversarial, retry, low-confidence,
  error, content-filter and PII-containing interactions) from `evalforge.demo`.
- CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, type check and tests on
  Python 3.11/3.12, then applies the migrations and runs the demo on PostgreSQL 16.

## Limitations and production readiness

What is production-grade: the schema and migrations, idempotent/resumable stages, redaction
before persistence, audit trail, immutable run history, structured logs with correlation IDs,
typed and tested code, offline reproducibility.

What you should know before relying on it:

- **Offline heuristics are a stand-in, not a judge.** In offline mode the quality judge,
  labeler and cluster namer are rule-based and the embeddings are hashed bag-of-words. They are
  good enough to demonstrate and test the system; real traffic needs real embedding, judge and
  labeling models. The demo numbers describe the simulated corpus, not real-world accuracy.
- **Deterministic grading is literal.** Assertions such as `contains "30 days"` fail on
  "thirty days". Use `hybrid` or `llm` grading for free-form answers.
- **Regex/heuristic PII redaction is best effort.** It covers common identifiers and
  self-introduced names. Regulated data needs a dedicated PII service plugged in through
  `EntityRecognizer`, plus a review of what upstream systems put in `metadata`.
- **Scale.** Embeddings are stored as JSON and similarity search is brute force in memory
  (fine to roughly 10^4-10^5 cases). The API runs pipeline and evaluation requests
  synchronously. See [ARCHITECTURE.md](ARCHITECTURE.md#scaling-considerations) for the path
  to pgvector, ClickHouse and background workers.
- **Auth is minimal.** A shared write token, no users, roles or SSO. Put the API and dashboard
  behind your identity-aware proxy before exposing them.
- **"Auto-label accuracy" is measured on reviewed items only**, i.e. the low-confidence queue,
  so it is a conservative lower bound. Spot-check auto-accepted cases periodically.
- **Single scheduler instance.** Run exactly one `pipeline` service; stages are idempotent but
  there is no distributed lock.

All assumptions and trade-offs are recorded in [DECISIONS.md](DECISIONS.md).

## License

MIT.
