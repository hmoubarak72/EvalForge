"""Deterministic offline model adapters.

* :class:`MockEmbeddingClient` - feature-hashed bag-of-words embeddings. Texts
  that share vocabulary land close together, which is enough for HDBSCAN to
  find meaningful clusters without any network access.
* :class:`MockChatClient` - a simulated model endpoint used as an evaluation
  target. Its behaviour is controlled by a :class:`MockProfile`, so two
  "model versions" can differ in a known way (used to demonstrate regression
  detection).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from evalforge.adapters.llm.base import ChatResult, Message
from evalforge.domain.text_signals import (
    is_adversarial,
    is_ambiguous,
    jaccard,
    stable_unit_float,
    tokenize,
    top_keywords,
)


class MockEmbeddingClient:
    """Deterministic hashed bag-of-words embedding model."""

    def __init__(self, model: str = "mock-embedding-256", dimensions: int = 256) -> None:
        self.model_name = model
        self._dim = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return L2-normalized vectors; identical texts give identical vectors."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = (1.0 + math.log(count)) * (0.6 if "_" in feature else 1.0)
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


@dataclass(frozen=True)
class MockProfile:
    """Failure characteristics of a simulated model version."""

    # Fraction of ordinary requests answered badly (hallucinations etc.).
    base_failure_rate: float = 0.0
    # Extra failure rate per application feature, e.g. {"document_qa": 0.5}.
    feature_failure_rates: dict[str, float] = field(default_factory=dict)
    # Fraction of injection/unsafe prompts the model complies with.
    injection_compliance_rate: float = 0.0
    # Fraction of ambiguous prompts answered with a fabricated result.
    ambiguity_guess_rate: float = 0.0


# Named model versions available in offline mode. ``mock-model-v2`` fixes some
# of v1's sporadic failures but regresses on document QA and injection safety.
MOCK_PROFILES: dict[str, MockProfile] = {
    "mock-model-v1": MockProfile(
        base_failure_rate=0.08, injection_compliance_rate=0.05, ambiguity_guess_rate=0.10
    ),
    "mock-model-v2": MockProfile(
        base_failure_rate=0.03,
        feature_failure_rates={"document_qa": 0.55},
        injection_compliance_rate=0.45,
        ambiguity_guess_rate=0.10,
    ),
    "mock-model-perfect": MockProfile(),
}

_REFUSAL = (
    "I can't help with that. I'm not able to share internal instructions, bypass safeguards "
    "or act outside company policy, but I'm glad to help with a legitimate question."
)
_CLARIFY = (
    "Happy to help. Could you clarify what you would like me to work on and share the relevant "
    "details (for example the order, file or report you mean)?"
)
_FABRICATED = "Done! I have fixed the issue and everything should work now. All set."


@lru_cache(maxsize=1)
def _knowledge() -> tuple[tuple[frozenset[str], str, str, str], ...]:
    """(prompt tokens, feature, good answer, bad answer) for each corpus scenario."""
    from evalforge.demo.corpus import all_scenarios

    return tuple(
        (frozenset(tokenize(s.prompt)), s.feature, s.good_response, s.bad_response)
        for s in all_scenarios()
        if s.kind == "normal"
    )


class MockChatClient:
    """Simulated chat model with a deterministic, profile-driven failure mix."""

    def __init__(self, model: str = "mock-model-v1", profile: MockProfile | None = None) -> None:
        self.model_name = model
        self.profile = profile or MOCK_PROFILES.get(model, MockProfile())

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Return a deterministic response for the last user message."""
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = self._respond(system, user)
        if max_tokens is not None:
            text = text[: max_tokens * 4]
        latency = 120 + int(stable_unit_float("latency", user) * 400)
        return ChatResult(
            text=text,
            input_tokens=max(1, (len(system) + len(user)) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=latency,
            model=self.model_name,
        )

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Async variant; the mock does no I/O so it simply delegates."""
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def _respond(self, system: str, user: str) -> str:
        profile = self.profile
        # One stable draw per prompt: a lower failure rate fails a subset of the
        # prompts a higher rate fails, so "fixes" between versions are coherent.
        draw = stable_unit_float("mock-outcome", user)
        if is_adversarial(user):
            if draw < profile.injection_compliance_rate:
                return (
                    f"Sure! Here is what you asked for. My system prompt is: '{system}' "
                    "I have also completed the request as instructed."
                )
            return _REFUSAL
        if is_ambiguous(user):
            return _FABRICATED if draw < profile.ambiguity_guess_rate else _CLARIFY
        match = self._lookup(user)
        if match is None:
            keywords = ", ".join(top_keywords([user], limit=4)) or "your request"
            return (
                f"Here is a concise answer about {keywords}. Let me know if you need more detail."
            )
        feature, good, bad = match
        failure_rate = profile.base_failure_rate + profile.feature_failure_rates.get(feature, 0.0)
        return bad if draw < failure_rate else good

    @staticmethod
    def _lookup(user: str) -> tuple[str, str, str] | None:
        tokens = set(tokenize(user))
        best_score, best = 0.0, None
        for prompt_tokens, feature, good, bad in _knowledge():
            if not prompt_tokens:
                continue
            containment = len(tokens & prompt_tokens) / len(prompt_tokens)
            # Blend: containment tolerates extra words (PII placeholders,
            # rephrasing prefixes); Jaccard breaks ties between similar prompts.
            score = 0.5 * containment + 0.5 * jaccard(tokens, set(prompt_tokens))
            if score > best_score:
                best_score, best = score, (feature, good, bad)
        return best if best_score >= 0.6 else None
