"""Stage 2: sampling.

Three strategies operate on lightweight :class:`SamplingRow` records so they
are pure, seedable and easy to test:

* ``uniform``         - simple random sample (baseline diversity)
* ``stratified``      - proportional allocation across feature / model /
                        error status / feedback strata, at least one per stratum
* ``signal_boosted``  - weighted sampling without replacement that oversamples
                        negative feedback, retries, high latency, low
                        confidence, content-filter events, errors and unusual
                        response lengths

Every selected record stores the strategy, seed, weight, stratum and the
human-readable reasons it was picked.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from evalforge.config import SamplingSettings
from evalforge.db.base import utcnow
from evalforge.db.models import InteractionLog, Sample
from evalforge.domain.enums import NEGATIVE_FEEDBACK, SamplingStrategy
from evalforge.repositories import InteractionRepository
from evalforge.telemetry import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SamplingRow:
    """The subset of interaction fields the samplers need."""

    id: str
    application_feature: str
    model_name: str
    error_status: str | None
    user_feedback: str | None
    retry_count: int
    latency_ms: int
    confidence_score: float | None
    content_filter_triggered: bool
    response_length: int

    @classmethod
    def from_log(cls, row: InteractionLog) -> SamplingRow:
        return cls(
            id=row.id,
            application_feature=row.application_feature,
            model_name=row.model_name,
            error_status=row.error_status,
            user_feedback=row.user_feedback,
            retry_count=row.retry_count,
            latency_ms=row.latency_ms,
            confidence_score=row.confidence_score,
            content_filter_triggered=row.content_filter_triggered,
            response_length=row.response_length,
        )


@dataclass
class Selection:
    """One sampled row with the explanation of why it was chosen."""

    row: SamplingRow
    weight: float = 1.0
    stratum: str | None = None
    reasons: list[str] = field(default_factory=list)


def stratum_key(row: SamplingRow, keys: Sequence[str]) -> str:
    """Build the stratum label for a row, e.g. ``feature=x|error_status=none``."""
    parts = []
    for key in keys:
        value = getattr(row, key)
        if key == "error_status":
            value = "error" if value else "none"
        parts.append(f"{key}={value if value is not None else 'none'}")
    return "|".join(parts)


def sample_uniform(rows: Sequence[SamplingRow], size: int, rng: random.Random) -> list[Selection]:
    """Uniform random sample without replacement."""
    chosen = rng.sample(list(rows), min(size, len(rows)))
    return [Selection(row, 1.0, None, ["uniform_random"]) for row in chosen]


def sample_stratified(
    rows: Sequence[SamplingRow], size: int, rng: random.Random, keys: Sequence[str]
) -> list[Selection]:
    """Proportional stratified sample with at least one row per stratum."""
    size = min(size, len(rows))
    strata: dict[str, list[SamplingRow]] = {}
    for row in rows:
        strata.setdefault(stratum_key(row, keys), []).append(row)
    names = sorted(strata)
    allocation = {name: 0 for name in names}
    if size >= len(names):
        for name in names:
            allocation[name] = 1
    remaining = size - sum(allocation.values())
    total = len(rows)
    # Largest-remainder proportional allocation, capped by stratum size.
    quotas = {name: remaining * len(strata[name]) / total for name in names}
    for name in names:
        extra = min(math.floor(quotas[name]), len(strata[name]) - allocation[name])
        allocation[name] += extra
    leftover = size - sum(allocation.values())
    by_remainder = sorted(names, key=lambda n: (-(quotas[n] - math.floor(quotas[n])), n))
    while leftover > 0:
        progressed = False
        for name in by_remainder:
            if leftover == 0:
                break
            if allocation[name] < len(strata[name]):
                allocation[name] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    selections: list[Selection] = []
    for name in names:
        for row in rng.sample(strata[name], allocation[name]):
            selections.append(Selection(row, 1.0, name, [f"stratum:{name}"]))
    return selections


def length_bounds(rows: Sequence[SamplingRow]) -> tuple[float, float]:
    """Robust (median +/- 3 scaled MAD) bounds for response length."""
    lengths = [r.response_length for r in rows if r.response_length > 0]
    if len(lengths) < 5:
        return 0.0, float("inf")
    median = statistics.median(lengths)
    mad = statistics.median(abs(x - median) for x in lengths) or 1.0
    spread = 3.0 * 1.4826 * mad
    return max(0.0, median - spread), median + spread


def signal_weight(
    row: SamplingRow,
    settings: SamplingSettings,
    bounds: tuple[float, float],
    underrepresented_features: frozenset[str] = frozenset(),
) -> tuple[float, list[str]]:
    """Compute the sampling weight of a row and the signals that boosted it."""
    weight = 1.0
    reasons: list[str] = []
    if row.user_feedback in {f.value for f in NEGATIVE_FEEDBACK}:
        weight += settings.boost_negative_feedback
        reasons.append(f"negative_feedback:{row.user_feedback}")
    if row.retry_count > 0:
        weight += settings.boost_retry
        reasons.append("retry")
    if row.latency_ms >= settings.high_latency_ms:
        weight += settings.boost_high_latency
        reasons.append("high_latency")
    if (
        row.confidence_score is not None
        and row.confidence_score < settings.low_confidence_threshold
    ):
        weight += settings.boost_low_confidence
        reasons.append("low_confidence")
    if row.content_filter_triggered:
        weight += settings.boost_content_filter
        reasons.append("content_filter")
    if row.error_status:
        weight += settings.boost_error
        reasons.append(f"error:{row.error_status}")
    low, high = bounds
    if row.response_length > 0 and not low <= row.response_length <= high:
        weight += settings.boost_unusual_length
        reasons.append("unusual_response_length")
    if row.application_feature in underrepresented_features:
        weight += settings.boost_underrepresented_cluster
        reasons.append("underrepresented_feature")
    return weight, reasons


def sample_signal_boosted(
    rows: Sequence[SamplingRow],
    size: int,
    rng: random.Random,
    settings: SamplingSettings,
    underrepresented_features: frozenset[str] = frozenset(),
) -> list[Selection]:
    """Weighted sampling without replacement (Efraimidis-Spirakis A-Res)."""
    bounds = length_bounds(rows)
    keyed: list[tuple[float, Selection]] = []
    for row in rows:
        weight, reasons = signal_weight(row, settings, bounds, underrepresented_features)
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, Selection(row, weight, None, reasons or ["baseline"])))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [selection for _, selection in keyed[: min(size, len(rows))]]


class SamplingService:
    """Selects interactions for a pipeline run and persists the explanation."""

    def __init__(self, settings: SamplingSettings, clock: Callable[[], datetime] = utcnow) -> None:
        self._settings = settings
        self._clock = clock

    def sample(
        self,
        session: Session,
        pipeline_run_id: str,
        *,
        strategy: str | None = None,
        sample_size: int | None = None,
        seed: int | None = None,
        underrepresented_features: frozenset[str] = frozenset(),
    ) -> list[Sample]:
        """Sample eligible interactions; idempotent per pipeline run."""
        repo = InteractionRepository(session)
        existing = repo.samples_for_run(pipeline_run_id)
        if existing:
            log.info("sampling.resumed", pipeline_run_id=pipeline_run_id, samples=len(existing))
            return existing

        chosen_strategy = SamplingStrategy(strategy or self._settings.strategy)
        size = sample_size or self._settings.sample_size
        chosen_seed = self._settings.seed if seed is None else seed
        now = self._clock()
        since = (
            now - timedelta(days=self._settings.lookback_days)
            if self._settings.lookback_days
            else None
        )
        logs = repo.unsampled(
            exclude_previously_sampled=self._settings.exclude_previously_sampled,
            since=since,
            until=now,
        )
        rows = [SamplingRow.from_log(row) for row in logs]
        rng = random.Random(chosen_seed)
        if chosen_strategy is SamplingStrategy.UNIFORM:
            selections = sample_uniform(rows, size, rng)
        elif chosen_strategy is SamplingStrategy.STRATIFIED:
            selections = sample_stratified(rows, size, rng, self._settings.stratify_by)
        else:
            selections = sample_signal_boosted(
                rows, size, rng, self._settings, underrepresented_features
            )

        samples = [
            Sample(
                pipeline_run_id=pipeline_run_id,
                interaction_id=selection.row.id,
                strategy=chosen_strategy.value,
                seed=chosen_seed,
                weight=round(selection.weight, 4),
                stratum=selection.stratum,
                reasons=selection.reasons,
                created_at=now,
            )
            for selection in selections
        ]
        session.add_all(samples)
        session.flush()
        log.info(
            "sampling.finished",
            strategy=chosen_strategy.value,
            eligible=len(rows),
            sampled=len(samples),
            seed=chosen_seed,
        )
        return samples
