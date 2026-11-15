# Demo walkthrough

A script for a recorded demo of under four minutes, plus what the seed data contains.

## Setup

```bash
make seed        # python scripts/seed_demo.py --reset
make dashboard   # http://localhost:8501
make api         # http://localhost:8000/docs (optional)
```

`make seed` prints one line per simulated night and a summary. With the default seed:

| Item | Value |
| --- | --- |
| Simulated interactions | 440 over 14 days, from 3 sources (JSONL app log, OTLP trace export, legacy database table) |
| Rejected / duplicate records | 5 invalid lines and a handful of re-deliveries on every run |
| Nightly pipeline runs | 7 (every second day), 60 samples each |
| Clusters discovered | 21 |
| Dataset | version 26, 110 active test cases (91 auto-labeled, 19 human-reviewed) |
| Near-duplicates skipped | 135 |
| Review queue | 33 pending after simulated reviewers handled 21 items |
| Evaluations | v1 on the early dataset (90.8%), v1 on the full dataset (93.6%), v2 on the full dataset (81.8%) |
| Regression | major: 15 new failures, 2 newly passing; document-QA clusters and adversarial cases regress |

Numbers are deterministic for a given `--seed`, `--count` and code version; the timestamps are
relative to the moment you run the seed.

## What the simulated traffic contains

The corpus (`src/evalforge/demo/corpus.py`) has 148 distinct requests for a fictional retailer:

- **Customer support**: refunds, shipping, password reset, membership billing (with fake
  names, emails and phone numbers that redaction removes).
- **Code assistance**: Python functions, Python error debugging, SQL queries.
- **Document QA**: questions about four company documents, with the document attached as
  `metadata.source_document`.
- **Data analysis**: pandas how-tos and statistics interpretation.
- **Ambiguous requests**: "Fix it.", "Same as before please".
- **Prompt-injection and unsafe requests**: instruction overrides, system-prompt extraction,
  data-exfiltration and abuse requests.

Each request has a good and a bad response. The generator (`generator.py`) mixes in negative
feedback, retries with rephrased prompts, low model confidence, high latency, provider errors,
content-filter events (including false positives), very long and very short responses, and
three hallucinated document answers that a user wrongly rated thumbs-up.

## Suggested recording (about 3.5 minutes)

1. **Logs being processed (30 s).** Terminal: run `make seed`. Point at the per-night table:
   ingested, duplicates, rejected, sampled, labeled, auto-accepted, sent to review.
2. **Clustering reveals traffic patterns (30 s).** Dashboard → *Clusters*: auto-named
   categories ranked by traffic; scroll to the outlier table (novel prompts, retries,
   content-filter events).
3. **Auto-labeling creates test cases (45 s).** *Dataset explorer*: filter
   `difficulty = adversarial`, open a case and show expected behaviour `refuse`, the forbidden
   "must not leak the system prompt" assertion and the rubric. Then open a document-QA case:
   reference answer verified against the source, required fact, hallucination trap.
4. **Human in the loop (40 s).** *Review queue*: show the reason ("low_confidence …"), the
   label confidence and pass agreement, the similar existing cases; approve one, edit one.
5. **The eval runner catches a regression (45 s).** *Evaluations*: pass-rate trend dropping
   on the last run, red regression banner, new failures table, category and difficulty shifts.
6. **The dataset grows by itself (20 s).** *Overview*: growth chart, auto-labeled vs
   human-reviewed, coverage heatmap with sparse clusters, freshness.

## Narrative

> I built a system that automatically converts production LLM logs into a growing eval
> dataset. In two weeks of simulated production traffic it generated 110 test cases across
> 21 categories, 83% of them auto-labeled without human involvement, skipped 135
> near-duplicates, and its eval runner caught a regression in a new model version before
> release. The dataset keeps growing every night, steered toward the clusters it covers least.

Be precise about the claim when presenting: these figures come from a synthetic corpus and
deterministic offline labelers. Of the 21 low-confidence candidates the simulated reviewers
handled, 9 were approved unchanged, 10 needed an edited reference and 2 were rejected, which
is what the review queue is for.
