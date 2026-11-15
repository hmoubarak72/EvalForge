"""Dataset metrics: coverage, growth, freshness, quality and reviewer stats."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.db.base import utcnow
from evalforge.db.models import EvalTestCase, InteractionLog, ReviewDecision
from evalforge.domain.enums import (
    CandidateStatus,
    Difficulty,
    ReviewAction,
    ReviewStatus,
    TestCaseSource,
    TestCaseStatus,
)
from evalforge.repositories import DatasetRepository, InteractionRepository

MIN_CASES_PER_CLUSTER = 3
UNCATEGORIZED = "Novel / Unclustered"


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for paired categorical decisions (None if undefined)."""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[label] * right[label] for label in set(left) | set(right)) / (total * total)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return round((observed - expected) / (1.0 - expected), 3)


class DatasetMetricsService:
    """Read-only analytics over the dataset; shared by the API and dashboard."""

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock

    # -- coverage ------------------------------------------------------------
    def coverage(self, session: Session) -> dict[str, Any]:
        """Per-cluster coverage and the cluster x difficulty heatmap."""
        clusters = InteractionRepository(session).list_clusters()
        cases = [
            c
            for c in DatasetRepository(session).all_test_cases()
            if c.status == TestCaseStatus.ACTIVE.value
        ]
        total_traffic = sum(c.size for c in clusters) or 1
        total_cases = len(cases) or 1
        by_cluster: dict[str | None, list[EvalTestCase]] = defaultdict(list)
        for case in cases:
            by_cluster[case.cluster_id].append(case)
        difficulties = [d.value for d in Difficulty]

        rows: list[dict[str, Any]] = []
        for cluster in clusters:
            members = by_cluster.get(cluster.id, [])
            traffic_share = cluster.size / total_traffic
            dataset_share = len(members) / total_cases
            if not cases:
                deficit = 1.0
            elif traffic_share > 0:
                deficit = max(0.0, (traffic_share - dataset_share) / traffic_share)
            else:
                deficit = 0.0
            if len(members) < MIN_CASES_PER_CLUSTER:
                deficit = max(deficit, 0.5)
            counts = Counter(m.difficulty for m in members)
            rows.append(
                {
                    "cluster_id": cluster.id,
                    "cluster": cluster.name,
                    "traffic_size": cluster.size,
                    "traffic_share": round(traffic_share, 4),
                    "test_cases": len(members),
                    "dataset_share": round(dataset_share if cases else 0.0, 4),
                    "deficit": round(deficit, 4),
                    "status": "sparse" if deficit >= 0.5 else "well_tested",
                    "by_difficulty": {d: counts.get(d, 0) for d in difficulties},
                }
            )
        unclustered = by_cluster.get(None, [])
        if unclustered:
            counts = Counter(m.difficulty for m in unclustered)
            rows.append(
                {
                    "cluster_id": None,
                    "cluster": UNCATEGORIZED,
                    "traffic_size": 0,
                    "traffic_share": 0.0,
                    "test_cases": len(unclustered),
                    "dataset_share": round(len(unclustered) / total_cases, 4),
                    "deficit": 0.0,
                    "status": "well_tested",
                    "by_difficulty": {d: counts.get(d, 0) for d in difficulties},
                }
            )
        return {
            "clusters": rows,
            "difficulties": difficulties,
            "sparse_clusters": [r["cluster"] for r in rows if r["status"] == "sparse"],
            "total_test_cases": len(cases),
        }

    def cluster_deficits(self, session: Session) -> dict[str, float]:
        """Coverage deficit per cluster ID (0 = well covered, 1 = untested)."""
        return {
            row["cluster_id"]: row["deficit"]
            for row in self.coverage(session)["clusters"]
            if row["cluster_id"]
        }

    def underrepresented_features(self, session: Session) -> frozenset[str]:
        """Application features whose dataset share lags their traffic share."""
        traffic = dict(
            session.execute(
                select(InteractionLog.application_feature, func.count()).group_by(
                    InteractionLog.application_feature
                )
            ).all()
        )
        dataset = dict(
            session.execute(
                select(EvalTestCase.application_feature, func.count())
                .where(EvalTestCase.status == TestCaseStatus.ACTIVE.value)
                .group_by(EvalTestCase.application_feature)
            ).all()
        )
        total_traffic = sum(traffic.values())
        total_cases = sum(dataset.values())
        if not total_traffic or not total_cases:
            return frozenset()
        return frozenset(
            feature
            for feature, count in traffic.items()
            if dataset.get(feature, 0) / total_cases < 0.5 * count / total_traffic
        )

    # -- growth / freshness ----------------------------------------------------
    def growth(self, session: Session) -> dict[str, Any]:
        """Daily and cumulative test-case counts, split by label source."""
        cases = DatasetRepository(session).all_test_cases()
        daily: dict[str, Counter[str]] = defaultdict(Counter)
        for case in cases:
            daily[case.created_at.date().isoformat()][case.source] += 1
        series = []
        cumulative = 0
        cumulative_by_source: Counter[str] = Counter()
        for day in sorted(daily):
            added = sum(daily[day].values())
            cumulative += added
            cumulative_by_source.update(daily[day])
            series.append(
                {
                    "date": day,
                    "added": added,
                    "total": cumulative,
                    "auto_labeled": cumulative_by_source[TestCaseSource.AUTO_LABELED.value],
                    "human_reviewed": cumulative_by_source[TestCaseSource.HUMAN_REVIEWED.value],
                }
            )
        versions = [
            {
                "version": v.version,
                "case_count": v.case_count,
                "reason": v.reason,
                "created_at": v.created_at.isoformat(),
            }
            for v in DatasetRepository(session).list_versions()
        ]
        return {"series": series, "total": cumulative, "versions": versions}

    def freshness(self, session: Session, window_days: int = 30) -> dict[str, Any]:
        """Share of active cases sourced from recent production traffic."""
        cutoff = self._clock() - timedelta(days=window_days)
        cases = [
            c
            for c in DatasetRepository(session).all_test_cases()
            if c.status == TestCaseStatus.ACTIVE.value
        ]
        fresh = sum(1 for c in cases if c.source_timestamp and c.source_timestamp >= cutoff)
        return {
            "window_days": window_days,
            "active_cases": len(cases),
            "fresh_cases": fresh,
            "freshness_pct": round(100.0 * fresh / len(cases), 2) if cases else 0.0,
        }

    # -- quality -----------------------------------------------------------------
    def quality(self, session: Session) -> dict[str, Any]:
        """Dataset composition and auto-labeling quality indicators."""
        repo = DatasetRepository(session)
        cases = [c for c in repo.all_test_cases() if c.status == TestCaseStatus.ACTIVE.value]
        status_counts = repo.candidate_status_counts()
        labels = [c.current_label for c in cases if c.current_label is not None]
        review_items = repo.list_review_items(limit=100000).items
        resolved = [i for i in review_items if i.status != ReviewStatus.PENDING.value]
        approved_as_is = sum(1 for i in resolved if i.status == ReviewStatus.APPROVED.value)
        decided = (
            status_counts.get(CandidateStatus.AUTO_ACCEPTED.value, 0)
            + status_counts.get(CandidateStatus.NEEDS_REVIEW.value, 0)
            + status_counts.get(CandidateStatus.APPROVED.value, 0)
            + status_counts.get(CandidateStatus.REJECTED.value, 0)
        )
        return {
            "active_test_cases": len(cases),
            "dataset_version": repo.current_version(),
            "by_source": dict(Counter(c.source for c in cases)),
            "by_category": dict(Counter(c.category for c in cases)),
            "by_difficulty": dict(Counter(c.difficulty for c in cases)),
            "by_case_type": dict(Counter(c.case_type for c in cases)),
            "by_expected_behavior": dict(Counter(lb.expected_behavior for lb in labels)),
            "candidate_status": status_counts,
            "average_label_confidence": round(sum(lb.confidence for lb in labels) / len(labels), 3)
            if labels
            else None,
            "auto_accept_rate": round(
                status_counts.get(CandidateStatus.AUTO_ACCEPTED.value, 0) / decided, 3
            )
            if decided
            else None,
            "duplicates_skipped": status_counts.get(CandidateStatus.DUPLICATE.value, 0),
            "review_queue_pending": sum(
                1 for i in review_items if i.status == ReviewStatus.PENDING.value
            ),
            # Share of human-reviewed candidates whose auto-labels were accepted
            # unchanged. Measured on the low-confidence queue, so it is a
            # conservative lower bound for auto-label accuracy overall.
            "auto_label_accuracy_reviewed": round(approved_as_is / len(resolved), 3)
            if resolved
            else None,
        }

    # -- reviewers ---------------------------------------------------------------
    def reviewer_stats(self, session: Session) -> dict[str, Any]:
        """Reviewer throughput and basic inter-annotator agreement."""
        decisions = [
            d
            for d in DatasetRepository(session).all_review_decisions()
            if d.action != ReviewAction.NOTE.value
        ]
        per_reviewer: dict[str, Counter[str]] = defaultdict(Counter)
        per_day: dict[str, Counter[str]] = defaultdict(Counter)
        by_item: dict[str, dict[str, ReviewDecision]] = defaultdict(dict)
        for decision in decisions:
            per_reviewer[decision.reviewer][decision.action] += 1
            per_day[decision.created_at.date().isoformat()][decision.reviewer] += 1
            by_item[decision.review_item_id][decision.reviewer] = decision  # latest wins
        pairs: list[tuple[str, str]] = []
        for reviewers in by_item.values():
            for left, right in combinations(sorted(reviewers), 2):
                pairs.append((self._verdict(reviewers[left]), self._verdict(reviewers[right])))
        agreement = round(sum(1 for a, b in pairs if a == b) / len(pairs), 3) if pairs else None
        return {
            "reviewers": [
                {"reviewer": name, "decisions": sum(counts.values()), **dict(counts)}
                for name, counts in sorted(per_reviewer.items())
            ],
            "throughput_by_day": [
                {"date": day, "reviewer": reviewer, "decisions": count}
                for day in sorted(per_day)
                for reviewer, count in sorted(per_day[day].items())
            ],
            "double_reviewed_items": sum(1 for r in by_item.values() if len(r) >= 2),
            "percent_agreement": agreement,
            "cohen_kappa": cohen_kappa(pairs),
        }

    @staticmethod
    def _verdict(decision: ReviewDecision) -> str:
        """Collapse approve/edit into 'accept' for agreement purposes."""
        return "reject" if decision.action == ReviewAction.REJECT.value else "accept"
