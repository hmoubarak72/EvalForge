# Decisions and assumptions

The specification left a number of points open. This file records what was assumed, the
default chosen, and why. Nothing here was decided silently in the code.

## Stack and dependencies

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **HDBSCAN comes from `sklearn.cluster.HDBSCAN`** (scikit-learn ≥ 1.3). The standalone `hdbscan` package is an optional extra and is not required. | Same algorithm, one fewer compiled dependency (the standalone package has no wheels for some Python/OS combinations, including Python 3.13 on Windows). |
| D2 | **Model clients use plain `httpx`**, not a vendor SDK. | "OpenAI-compatible" is an HTTP contract; `httpx` works with OpenAI, vLLM, Ollama, LiteLLM and gateways alike and is trivial to fake in tests. |
| D3 | **APScheduler** (not Celery) runs the nightly job. | The spec prefers it unless distributed workers are needed. Stages are plain callables, so moving to a queue later does not change service code. |
| D4 | **SQLite is the development mode; PostgreSQL the deployment target.** IDs are string UUIDs, JSON columns are `JSONB` on PostgreSQL and `JSON` on SQLite, timestamps go through a `UTCDateTime` type. | One schema and one migration for both. Verified against PostgreSQL 16 (migrations, `alembic check`, full demo). |
| D5 | **Python 3.11+**; developed and tested on 3.13. | As specified. |

## Data model

| # | Decision | Rationale |
| --- | --- | --- |
| D6 | Two tables beyond the required 15: `ingestion_sources` and `dataset_versions`. | Sources need a place to live (and an incremental cursor); a monotonic dataset version is needed to say which dataset an evaluation ran against. |
| D7 | **Dataset version = integer bumped on every change** to the active dataset (pipeline batch, review approval, label edit, retirement). | Simple, auditable; each evaluation run stores the version it used. Full snapshots are not stored (see "Not built"). |
| D8 | **Labels are versioned** (`test_case_labels.version`, `is_current`). Edits never overwrite. | Auditability; evaluation results reference the exact label version used. |
| D9 | **Embeddings are stored as JSON arrays** on `interaction_classifications` and `eval_test_cases`. | Portable across SQLite and PostgreSQL. `SimilarityIndex` is the single seam to replace with pgvector/FAISS. |
| D10 | `interaction_logs` is append-only and referenced by ID only. | Lets the log store move to ClickHouse later without touching application tables. |
| D11 | Both **prompt and response embeddings** are computed and stored; **clustering and deduplication use the prompt embedding**. | Categories should describe what users ask; clustering on responses would group by answer style. The response embedding is stored for audit and future analyses (e.g. response-drift detection). |

## Ingestion and privacy

| # | Decision | Rationale |
| --- | --- | --- |
| D12 | **Fingerprint = SHA-256 of timestamp (ms), feature, model, system prompt, user prompt, response and session ID, computed after redaction.** The upstream ID is excluded. | The same call delivered by two sources (app log and trace) deduplicates; nothing unredacted is needed to compute it. |
| D13 | **Incremental ingestion** uses a per-source timestamp cursor with a 30-minute overlap. | Late-arriving records are not missed; the overlap is harmless because of D12. |
| D14 | **Redaction is on by default** and runs before persistence on the three text fields and recursively on metadata; a configurable list of metadata keys is dropped outright. Rejected-record error details store a short excerpt and validation messages, never whole payloads. | "Never persist unredacted sensitive fields in the default configuration." |
| D15 | **"Optional NER" is a pluggable `EntityRecognizer`.** The default is an offline, pattern-based person-name detector (introductions, titles, salutations). | Keeps the core demo free of model downloads. spaCy or Presidio can be injected without code changes elsewhere. It will miss names that are not introduced; documented as a limitation. |
| D16 | Credit-card matches must pass a **Luhn check**. | Avoids redacting order numbers and other long digit runs. |
| D17 | A source that cannot be read at all fails **that ingestion run** (recorded, logged) but not the pipeline; a bad record is counted as rejected and never aborts the run. | One broken source or line should not stop the nightly job. |

## Sampling and classification

| # | Decision | Rationale |
| --- | --- | --- |
| D18 | By default an interaction is **sampled at most once** (`exclude_previously_sampled`). | Each night works on fresh traffic; makes re-runs idempotent. |
| D19 | Signal-boosted sampling uses **weighted sampling without replacement** (Efraimidis-Spirakis). "Unusual response length" = outside median ± 3 scaled MAD. | Seedable, single pass, robust to outliers. |
| D20 | "Prioritise under-represented clusters in future sampling" is implemented in two places: sampling boosts **application features** whose dataset share lags their traffic share (cluster membership is unknown before classification), and candidate selection adds the **cluster coverage deficit** to the priority. | The cluster of an interaction only exists after stage 3. |
| D21 | **Clusters are persistent.** Each run clusters new embeddings together with a window of earlier ones, then maps every HDBSCAN group to an existing cluster (majority of already-labelled members, else centroid similarity) or creates a new one. | Category names must stay stable across nightly runs for coverage and trend reporting. |
| D22 | A new cluster is only created from a **cohesive** group (mean member-to-centroid similarity ≥ `min_cohesion`, default 0.55), and a point only joins a cluster above `member_threshold`. | Without this, small early runs merged unrelated topics into one cluster that then kept absorbing traffic (topic purity 0.79 → 0.97 on the demo after adding it). The thresholds are embedding-model specific and configurable. |
| D23 | **Late assignment:** an interaction that was novel when first seen joins a cluster once enough similar traffic exists; its candidate and test case follow. Its original `novel_prompt` outlier flag is kept. | Otherwise cold-start cases stay "unclustered" forever. The flag is historically true. |
| D24 | **Difficulty is rule-based** in all modes. | Deterministic and auditable; the signals behind each label are stored. |
| D25 | **Offline "LLM" roles are explicit heuristic implementations** (`HeuristicQualityJudge`, `HeuristicLabeler`, `DeterministicClusterNamer`) behind the same protocols as the LLM-backed ones, rather than a mock chat model pretending to produce JSON. | Honest about what runs offline; the LLM code paths are still tested with a scripted chat double. |
| D26 | When an LLM call or its JSON is unusable, the role **falls back to its heuristic and logs a warning**; the stored model name records that the fallback was used. | A flaky judge should degrade a run, not fail it, and the audit trail must show it. |

## Labeling, routing and deduplication

| # | Decision | Rationale |
| --- | --- | --- |
| D27 | **Candidate selection:** outliers, clear failures (quality ≤ 2.5) and strong exemplars (quality ≥ 4) become candidates; unremarkable mid-quality interactions do not. **Infrastructure errors and empty responses are excluded.** | A timeout says nothing about model behaviour. |
| D28 | **Positive vs negative cases.** A well-rated production response (≥ 4) is a positive exemplar and may serve as the reference; a poor one is a negative case whose reference must come from elsewhere (source document, LLM, or a human). | The spec's "keep doing this well" / "fix this" split. |
| D29 | **`expected_quality_score` is the bar a correct response must reach**, not a prediction of what a model will score. A case passes at `score ≥ expected − pass_score_tolerance` (default 1.0). | Gives the 1-5 label a concrete role in pass/fail. |
| D30 | **Assertions are structured** (`contains`, `contains_any`, `regex`, `min_length`, `max_length`, `concept`). `concept` is judged by the LLM grader and falls back to keyword coverage. Numeric `contains` checks use digit boundaries (20 does not match 120 or 2020). | Deterministic checks must be cheap, explainable and hard to satisfy by accident. |
| D31 | **Confidence = mean self-reported confidence × agreement** across passes; agreement = 0.4·behaviour + 0.25·quality + 0.35·assertion overlap. Any behaviour disagreement, a quality spread ≥ 2, low assertion overlap or an unverified reference is a **conflict** and forces human review regardless of confidence. | "Route high-confidence cases automatically … low-confidence or conflicting cases to a human." |
| D32 | Offline, the heuristic labeler **perturbs later passes with a seeded probability proportional to its own uncertainty**. | Makes multi-pass agreement meaningful and reproducible without a real model. |
| D33 | **Source verification** is deterministic in both modes: every figure in the reference must occur in the source document and at least half of its content words must overlap. | Catches hallucinated figures even when a user applauded the answer (the demo plants three such cases). |
| D34 | **Deduplication runs twice:** a cheap pre-check before labeling (saves model calls) and the authoritative check before routing, against active test cases *and* candidates already waiting for review. Exact repeats of retired cases are also skipped. | The reviewer should not see the same case twice. |
| D35 | Labeling is capped at `max_candidates_per_run` (default 40), highest priority first. | Bounds LLM cost per night. Unpicked classifications compete again next run. |

## Evaluation and regression

| # | Decision | Rationale |
| --- | --- | --- |
| D36 | **"Previous comparable run" = the latest completed run in the same `comparison_group`** (default `"default"`), regardless of model name; an explicit `baseline_run_id` overrides it. | The main use is comparing a new model version with the current one. Separate groups isolate experiments. |
| D37 | Runs are compared **only on shared test cases**; added and removed cases are reported separately. | The dataset grows between runs; comparing different case sets would fake regressions or improvements. |
| D38 | **Severity:** `major` if the overall pass rate drops by the threshold or any category/difficulty group regresses beyond its threshold; `minor` if there are new failures only; else `none`. Groups smaller than `min_group_size` are ignored for shifts. | Separates noise from signal. |
| D39 | A model-call failure is recorded as a **failed result** with the error; the run still completes. | Endpoint flakiness is itself a finding and must not hide other results. |
| D40 | **Cost is an estimate** from configurable per-1K-token prices. | Providers differ; the estimate is labelled as such. |
| D41 | The offline evaluation target is a **simulated model** (`mock-model-v1`, `mock-model-v2`) with a deterministic failure profile; v2 fixes some sporadic v1 failures but hallucinates on document QA and complies with more injections. | Produces a realistic, reproducible regression for the demo and tests. |

## API, dashboard and operations

| # | Decision | Rationale |
| --- | --- | --- |
| D42 | **Pipeline and evaluation endpoints run synchronously** and return the finished run. | Simplest correct behaviour for a demo-scale service; see ARCHITECTURE for the background-worker path. |
| D43 | **Auth:** `local_dev` (open writes, documented, dashboard warning) or `token` (shared secret from an env var, `X-API-Key`, constant-time comparison). Reads are always open. | The spec's minimum: "authentication or an explicitly documented local-development authorization mode." |
| D44 | The **dashboard uses the service layer directly**, not the HTTP API. | One process to run for review work, same business rules, no duplicated client. |
| D45 | The **first decisive review action resolves an item**; later decisions by other reviewers are stored as second opinions (`applied = false`). Agreement metrics treat approve and edit as "accept". | Enables inter-annotator agreement without letting a second click change the dataset. |
| D46 | Docker Compose runs classification, labeling and the eval runner in **one `pipeline` worker** rather than one container per stage. | The stages share a transaction boundary and a schedule; separate containers would add orchestration without benefit at this scale. |
| D47 | **Simulated clock.** Every service takes an injected clock; the demo seed replays 14 days of nightly runs with historical timestamps. | Makes growth-over-time charts meaningful in a demo and tests deterministic. |

## Not built (recommended next steps)

1. **pgvector / ANN index** for embeddings and similarity search.
2. **Background job queue** (Celery/Arq) for pipeline and evaluation requests, with a status endpoint.
3. **ClickHouse adapter** for `interaction_logs` at high volume.
4. **Dataset snapshots** so an old version can be re-evaluated exactly.
5. **Real authentication** (OIDC, per-user roles) and reviewer assignment.
6. **Streaming ingestion** (Kafka/OTLP receiver) instead of file and table polling.
7. **Calibration report** comparing auto-label confidence with reviewer outcomes, including a random audit sample of auto-accepted cases.
8. **Prometheus metrics endpoint** and alerting on regression severity.
