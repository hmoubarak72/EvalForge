"""Sampling strategies."""

from __future__ import annotations

import random
from collections import Counter

from evalforge.config import SamplingSettings
from evalforge.services.sampling import (
    SamplingRow,
    sample_signal_boosted,
    sample_stratified,
    sample_uniform,
    signal_weight,
    stratum_key,
)


def row(index: int, **overrides: object) -> SamplingRow:
    values: dict[str, object] = {
        "id": f"r{index}",
        "application_feature": "support",
        "model_name": "m1",
        "error_status": None,
        "user_feedback": None,
        "retry_count": 0,
        "latency_ms": 800,
        "confidence_score": 0.9,
        "content_filter_triggered": False,
        "response_length": 300,
    }
    values.update(overrides)
    return SamplingRow(**values)  # type: ignore[arg-type]


def population() -> list[SamplingRow]:
    rows = [row(i) for i in range(180)]
    rows += [row(200 + i, application_feature="code") for i in range(15)]
    rows += [row(300 + i, user_feedback="thumbs_down", confidence_score=0.2) for i in range(10)]
    rows += [row(400 + i, error_status="timeout", application_feature="docs") for i in range(3)]
    return rows


def test_uniform_is_seeded_and_without_replacement() -> None:
    rows = population()
    first = [s.row.id for s in sample_uniform(rows, 40, random.Random(5))]
    second = [s.row.id for s in sample_uniform(rows, 40, random.Random(5))]
    other = [s.row.id for s in sample_uniform(rows, 40, random.Random(6))]
    assert first == second != other
    assert len(set(first)) == 40
    assert len(sample_uniform(rows, 10_000, random.Random(1))) == len(rows)


def test_stratified_represents_every_stratum_proportionally() -> None:
    rows = population()
    keys = ["application_feature", "error_status", "user_feedback"]
    selections = sample_stratified(rows, 40, random.Random(3), keys)
    assert len(selections) == 40
    counts = Counter(s.stratum for s in selections)
    assert set(counts) == {stratum_key(r, keys) for r in rows}  # nobody left out
    biggest = stratum_key(rows[0], keys)
    assert counts[biggest] == max(counts.values())
    assert all(s.reasons == [f"stratum:{s.stratum}"] for s in selections)
    # Fewer slots than strata still returns exactly the requested size.
    assert len(sample_stratified(rows, 2, random.Random(3), keys)) == 2


def test_signal_weight_explains_every_boost() -> None:
    settings = SamplingSettings()
    weight, reasons = signal_weight(
        row(
            1,
            user_feedback="thumbs_down",
            retry_count=1,
            latency_ms=12000,
            confidence_score=0.1,
            content_filter_triggered=True,
            error_status="timeout",
            response_length=9000,
        ),
        settings,
        (100.0, 600.0),
        frozenset({"support"}),
    )
    assert reasons == [
        "negative_feedback:thumbs_down",
        "retry",
        "high_latency",
        "low_confidence",
        "content_filter",
        "error:timeout",
        "unusual_response_length",
        "underrepresented_feature",
    ]
    assert weight == 1.0 + 4 + 3 + 2 + 3 + 4 + 3 + 2 + 2
    assert signal_weight(row(2), settings, (100.0, 600.0)) == (1.0, [])


def test_signal_boosted_oversamples_problem_interactions() -> None:
    rows = population()
    flagged = {r.id for r in rows if r.user_feedback or r.error_status}
    share_in_population = len(flagged) / len(rows)
    hits = 0
    for seed in range(30):
        chosen = sample_signal_boosted(rows, 40, random.Random(seed), SamplingSettings())
        assert len({s.row.id for s in chosen}) == 40
        hits += sum(1 for s in chosen if s.row.id in flagged)
    assert hits / (30 * 40) > 2.5 * share_in_population
    sample = sample_signal_boosted(rows, 40, random.Random(1), SamplingSettings())
    assert all(s.reasons for s in sample)
    assert any(s.reasons == ["baseline"] for s in sample)
