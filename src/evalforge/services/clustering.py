"""Embedding-based clustering with persistent cluster identities.

HDBSCAN is run over the new embeddings plus a window of previously classified
embeddings. Each discovered group is then mapped onto a *persistent* cluster
row (by majority vote of already-labelled members, else centroid similarity)
so categories stay stable from one nightly run to the next and coverage can be
tracked over time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sqlalchemy import update
from sqlalchemy.orm import Session

from evalforge.adapters.llm.base import ChatClient, LLMError, parse_json_object
from evalforge.config import ClusteringSettings
from evalforge.db.models import Cluster, EvalCandidate, EvalTestCase
from evalforge.domain.text_signals import top_keywords
from evalforge.repositories import InteractionRepository
from evalforge.telemetry import get_logger

log = get_logger(__name__)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row (zero rows are left untouched)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between the rows of ``a`` and ``b``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    return normalize_rows(a) @ normalize_rows(b).T


def run_hdbscan(vectors: np.ndarray, settings: ClusteringSettings) -> tuple[np.ndarray, np.ndarray]:
    """Cluster vectors with HDBSCAN.

    Returns ``(labels, probabilities)`` where label ``-1`` marks noise. Inputs
    are L2-normalized (so Euclidean distance is monotonic in cosine distance)
    and optionally reduced with PCA, which makes density estimation far more
    reliable than in the raw embedding space.
    """
    count = vectors.shape[0]
    if count < max(2 * settings.min_cluster_size, 4):
        return np.full(count, -1, dtype=int), np.zeros(count)
    data = normalize_rows(vectors.astype(float))
    target_dim = settings.reduce_dimensions
    if target_dim and data.shape[1] > target_dim and count > target_dim + 1:
        data = PCA(n_components=target_dim, random_state=settings.seed).fit_transform(data)
    model = HDBSCAN(
        min_cluster_size=settings.min_cluster_size,
        min_samples=settings.min_samples,
        cluster_selection_epsilon=settings.cluster_selection_epsilon,
        cluster_selection_method=settings.cluster_selection_method,
        metric="euclidean",
        copy=True,
    )
    labels = model.fit_predict(data)
    return labels.astype(int), np.asarray(model.probabilities_, dtype=float)


@dataclass(frozen=True)
class ClusterName:
    """A generated cluster name with provenance."""

    name: str
    description: str
    keywords: list[str]
    method: str


class ClusterNamer(Protocol):
    """Names a cluster from representative prompts."""

    def name(self, prompts: Sequence[str]) -> ClusterName: ...


class DeterministicClusterNamer:
    """Offline namer: the most frequent content words of the representatives."""

    def name(self, prompts: Sequence[str]) -> ClusterName:
        keywords = top_keywords(list(prompts), limit=5)
        label = " / ".join(word.replace("_", " ").capitalize() for word in keywords[:3]) or "Misc"
        description = f"Auto-named from {len(prompts)} representative prompts; keywords: " + (
            ", ".join(keywords) or "n/a"
        )
        return ClusterName(label, description, keywords, "deterministic")


class LLMClusterNamer:
    """Names clusters with an LLM, falling back to the deterministic namer."""

    def __init__(self, chat: ChatClient, fallback: ClusterNamer | None = None) -> None:
        self._chat = chat
        self._fallback = fallback or DeterministicClusterNamer()

    def name(self, prompts: Sequence[str]) -> ClusterName:
        examples = "\n".join(f"- {p[:300]}" for p in prompts[:8])
        messages = [
            {
                "role": "system",
                "content": "You name clusters of user requests. Reply with JSON only: "
                '{"name": "<2-5 word Title Case category>", "description": "<one sentence>"}',
            },
            {"role": "user", "content": f"Representative requests:\n{examples}"},
        ]
        try:
            parsed = parse_json_object(self._chat.complete(messages, temperature=0.0).text)
            name = str(parsed["name"]).strip()[:120]
            if not name:
                raise LLMError("empty cluster name")
        except (LLMError, KeyError) as exc:
            log.warning("clustering.llm_naming_failed", error=str(exc))
            return self._fallback.name(prompts)
        return ClusterName(
            name,
            str(parsed.get("description", "")).strip(),
            top_keywords(list(prompts), limit=5),
            f"llm:{self._chat.model_name}",
        )


@dataclass(frozen=True)
class ClusterItem:
    """An interaction to be placed into a cluster."""

    interaction_id: str
    vector: list[float]
    prompt: str


@dataclass(frozen=True)
class Assignment:
    """Cluster assignment of one interaction (``cluster_id`` None = novel)."""

    cluster_id: str | None
    probability: float


@dataclass
class AssignmentResult:
    """Assignments plus bookkeeping for stage metrics."""

    assignments: dict[str, Assignment]
    new_cluster_ids: list[str]
    late_assignments: int = 0


class ClusterService:
    """Discovers clusters and maintains the persistent cluster table."""

    def __init__(self, settings: ClusteringSettings, namer: ClusterNamer) -> None:
        self._settings = settings
        self._namer = namer

    def assign(
        self, session: Session, items: Sequence[ClusterItem], pipeline_run_id: str | None
    ) -> AssignmentResult:
        """Assign each item to a persistent cluster or mark it as novel."""
        if not items:
            return AssignmentResult({}, [])
        repo = InteractionRepository(session)
        context = repo.recent_classifications(self._settings.context_window)
        context = [c for c in context if c.embedding]
        n_ctx = len(context)
        matrix = np.array([c.embedding for c in context] + [i.vector for i in items], dtype=float)
        labels, probabilities = run_hdbscan(matrix, self._settings)

        clusters = {c.id: c for c in repo.list_clusters()}
        assignments: dict[str, Assignment] = {}
        new_cluster_ids: list[str] = []
        additions: dict[str, list[int]] = {}
        rejected: list[int] = []
        adopted: dict[str, list[str]] = {}
        late: dict[str, list[int]] = {}

        for label in sorted(set(labels.tolist()) - {-1}):
            members = np.flatnonzero(labels == label)
            new_members = [int(i) for i in members if i >= n_ctx]
            if not new_members:
                continue
            ctx_members = [int(i) for i in members if i < n_ctx]
            centroid = normalize_rows(matrix[members].mean(axis=0, keepdims=True))[0]
            target = self._match_existing(context, ctx_members, centroid, clusters)
            if target is None:
                cohesion = float((normalize_rows(matrix[members]) @ centroid).mean())
                if cohesion < self._settings.min_cohesion:
                    # A loose group mixes unrelated topics: do not mint a cluster
                    # from it; its points are handled as noise below.
                    rejected.extend(new_members)
                    continue
                prompts = self._representatives(matrix, members, centroid, context, items, n_ctx)
                target = self._create_cluster(session, repo, prompts, centroid, pipeline_run_id)
                clusters[target.id] = target
                new_cluster_ids.append(target.id)
                target_centroid = centroid
            else:
                target_centroid = np.array(target.centroid, dtype=float)
            fits = normalize_rows(matrix[new_members]) @ target_centroid
            for index, fit in zip(new_members, fits, strict=True):
                if float(fit) < self._settings.member_threshold:
                    rejected.append(index)
                    continue
                item = items[index - n_ctx]
                assignments[item.interaction_id] = Assignment(
                    target.id, round(float(probabilities[index]), 4)
                )
                additions.setdefault(target.id, []).append(index)
            # Late assignment: earlier interactions that were novel when first
            # seen join the cluster now that enough similar traffic exists.
            for index in ctx_members:
                earlier = context[index]
                if earlier.cluster_id is None and (
                    float(normalize_rows(matrix[[index]])[0] @ target_centroid)
                    >= self._settings.member_threshold
                ):
                    earlier.cluster_id = target.id
                    adopted.setdefault(target.id, []).append(earlier.interaction_id)
                    additions.setdefault(target.id, [])
                    late.setdefault(target.id, []).append(index)

        # Noise points: attach to the nearest known cluster if close enough.
        noise = [int(i) for i in np.flatnonzero(labels == -1) if i >= n_ctx] + rejected
        if noise and clusters:
            ordered = list(clusters.values())
            centroids = np.array([c.centroid for c in ordered], dtype=float)
            similarities = cosine_similarity_matrix(matrix[noise], centroids)
            for row, index in enumerate(noise):
                best = int(np.argmax(similarities[row]))
                score = float(similarities[row, best])
                item = items[index - n_ctx]
                if score >= self._settings.assign_threshold:
                    assignments[item.interaction_id] = Assignment(ordered[best].id, round(score, 4))
                    additions.setdefault(ordered[best].id, []).append(index)
                else:
                    assignments[item.interaction_id] = Assignment(None, 0.0)
        else:
            for index in noise:
                assignments[items[index - n_ctx].interaction_id] = Assignment(None, 0.0)

        for cluster_id, indices in additions.items():
            cluster = clusters[cluster_id]
            late_indices = late.get(cluster_id, [])
            self._absorb(
                cluster,
                matrix[indices + late_indices],
                [items[i - n_ctx].interaction_id for i in indices] + adopted.get(cluster_id, []),
            )
            self._sync_adopted(session, cluster, adopted.get(cluster_id, []))
        session.flush()
        return AssignmentResult(
            assignments, new_cluster_ids, sum(len(ids) for ids in adopted.values())
        )

    @staticmethod
    def _sync_adopted(session: Session, cluster: Cluster, interaction_ids: list[str]) -> None:
        """Propagate a late cluster assignment to candidates and test cases."""
        if not interaction_ids:
            return
        session.execute(
            update(EvalCandidate)
            .where(EvalCandidate.interaction_id.in_(interaction_ids))
            .values(cluster_id=cluster.id)
        )
        session.execute(
            update(EvalTestCase)
            .where(
                EvalTestCase.source_interaction_id.in_(interaction_ids),
                EvalTestCase.cluster_id.is_(None),
            )
            .values(cluster_id=cluster.id, category=cluster.name)
        )
        log.info("clustering.late_assignment", cluster=cluster.name, count=len(interaction_ids))

    def _match_existing(
        self,
        context: Sequence[object],
        ctx_members: list[int],
        centroid: np.ndarray,
        clusters: dict[str, Cluster],
    ) -> Cluster | None:
        votes = Counter(
            context[i].cluster_id  # type: ignore[attr-defined]
            for i in ctx_members
            if context[i].cluster_id in clusters  # type: ignore[attr-defined]
        )
        if votes:
            cluster_id, count = votes.most_common(1)[0]
            if count >= 0.5 * len(ctx_members):
                return clusters[cluster_id]
        if not clusters:
            return None
        ordered = list(clusters.values())
        similarities = cosine_similarity_matrix(
            centroid[None, :], np.array([c.centroid for c in ordered], dtype=float)
        )[0]
        best = int(np.argmax(similarities))
        if float(similarities[best]) >= self._settings.match_threshold:
            return ordered[best]
        return None

    @staticmethod
    def _representatives(
        matrix: np.ndarray,
        members: np.ndarray,
        centroid: np.ndarray,
        context: Sequence[object],
        items: Sequence[ClusterItem],
        n_ctx: int,
    ) -> list[str]:
        similarities = normalize_rows(matrix[members]) @ centroid
        ranked = [int(members[i]) for i in np.argsort(-similarities)]
        prompts: list[str] = []
        for index in ranked:
            prompt = (
                items[index - n_ctx].prompt
                if index >= n_ctx
                else context[index].interaction.user_prompt  # type: ignore[attr-defined]
            )
            if prompt not in prompts:
                prompts.append(prompt)
            if len(prompts) >= 8:
                break
        return prompts

    def _create_cluster(
        self,
        session: Session,
        repo: InteractionRepository,
        prompts: list[str],
        centroid: np.ndarray,
        pipeline_run_id: str | None,
    ) -> Cluster:
        generated = self._namer.name(prompts)
        name, suffix = generated.name, 2
        if repo.cluster_name_exists(name) and len(generated.keywords) > 3:
            # Prefer a name built from the next keywords over a numeric suffix.
            name = " / ".join(w.replace("_", " ").capitalize() for w in generated.keywords[1:4])
        while repo.cluster_name_exists(name):
            name = f"{generated.name} #{suffix}"
            suffix += 1
        cluster = Cluster(
            name=name,
            description=generated.description,
            keywords=generated.keywords,
            naming_method=generated.method,
            centroid=[round(float(x), 6) for x in centroid],
            size=0,
            representative_interaction_ids=[],
            first_pipeline_run_id=pipeline_run_id,
        )
        session.add(cluster)
        session.flush()
        log.info("clustering.cluster_created", cluster=name, method=generated.method)
        return cluster

    @staticmethod
    def _absorb(cluster: Cluster, vectors: np.ndarray, interaction_ids: Sequence[str]) -> None:
        old = np.array(cluster.centroid, dtype=float) * max(cluster.size, 0)
        merged = old + vectors.sum(axis=0) if cluster.size else vectors.sum(axis=0)
        centroid = normalize_rows(merged[None, :])[0]
        cluster.centroid = [round(float(x), 6) for x in centroid]
        cluster.size = cluster.size + len(interaction_ids)
        representatives = list(cluster.representative_interaction_ids)
        for interaction_id in interaction_ids:
            if len(representatives) >= 10:
                break
            representatives.append(interaction_id)
        cluster.representative_interaction_ids = representatives
