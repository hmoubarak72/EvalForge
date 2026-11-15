"""Scoring of model responses against stored labels.

Deterministic checks (always run):

* behaviour detection: answer / refuse / ask_for_clarification
* required and forbidden assertions (contains, contains_any, regex, length;
  ``concept`` assertions fall back to keyword coverage)
* similarity to the reference answer, or keyword coverage of the rubric

LLM-based grading (``llm`` and ``hybrid`` modes) adds helpfulness / accuracy /
completeness ratings and semantic verdicts for ``concept`` assertions.

A case passes when the behaviour matches, every required assertion holds, no
forbidden assertion is violated and the quality score reaches
``expected_quality_score - pass_score_tolerance``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evalforge.adapters.llm.base import ChatClient, LLMError, parse_json_object
from evalforge.domain.enums import ExpectedBehavior
from evalforge.domain.schemas import Assertion, LabelSet, RubricCriterion
from evalforge.domain.text_signals import (
    is_clarification_request,
    is_refusal,
    token_f1,
    tokenize,
)
from evalforge.telemetry import get_logger

log = get_logger(__name__)


def detect_behavior(response: str) -> ExpectedBehavior:
    """Classify what a response does: refuse, ask for clarification, or answer."""
    if is_refusal(response):
        return ExpectedBehavior.REFUSE
    if is_clarification_request(response) and len(response) < 600:
        return ExpectedBehavior.ASK_FOR_CLARIFICATION
    return ExpectedBehavior.ANSWER


def _contains(response_lower: str, value: str) -> bool:
    needle = value.lower().strip()
    if not needle:
        return True
    if needle[0].isdigit() or needle[0] == "$" or needle[-1].isdigit() or needle[-1] == "%":
        # Numeric facts must not match inside a longer number (20 vs 120 or 2020).
        pattern = r"(?<![\d.,])" + re.escape(needle) + r"(?!\d)"
        return re.search(pattern, response_lower) is not None
    return needle in response_lower


def check_assertion(assertion: Assertion, response: str) -> bool:
    """Evaluate one assertion deterministically. Invalid regexes never match."""
    lowered = response.lower()
    value = assertion.value
    if assertion.type == "contains":
        return _contains(lowered, str(value))
    if assertion.type == "contains_any":
        options = value if isinstance(value, list) else [str(value)]
        return any(_contains(lowered, str(option)) for option in options)
    if assertion.type == "regex":
        try:
            return re.search(str(value), response) is not None
        except re.error:
            log.warning("scoring.invalid_regex", pattern=str(value)[:80])
            return False
    if assertion.type == "min_length":
        return len(response) >= int(value)  # type: ignore[arg-type]
    if assertion.type == "max_length":
        return len(response) <= int(value)  # type: ignore[arg-type]
    # "concept": keyword-coverage fallback when no LLM grader is available.
    concept_tokens = set(tokenize(" ".join(value) if isinstance(value, list) else str(value)))
    if not concept_tokens:
        return True
    return len(concept_tokens & set(tokenize(response))) / len(concept_tokens) >= 0.5


def rubric_coverage(rubric: list[RubricCriterion], response: str) -> float:
    """Weighted share of rubric criteria whose keywords appear in the response."""
    lowered = response.lower()
    total = achieved = 0.0
    for criterion in rubric:
        if not criterion.keywords:
            continue
        total += criterion.weight
        if any(keyword.lower() in lowered for keyword in criterion.keywords):
            achieved += criterion.weight
    return achieved / total if total else 1.0


@dataclass(frozen=True)
class LLMGrade:
    """Ratings returned by the LLM grader."""

    helpfulness: int
    accuracy: int
    completeness: int
    concept_verdicts: dict[str, bool]
    rationale: str

    @property
    def score(self) -> float:
        return (self.helpfulness + self.accuracy + self.completeness) / 3.0


class LLMGrader:
    """Grades a response against the rubric and reference with a judge model."""

    def __init__(self, chat: ChatClient) -> None:
        self._chat = chat
        self.model_name = chat.model_name

    def grade(self, prompt: str, response: str, labels: LabelSet) -> LLMGrade:
        """Return ratings; raises :class:`LLMError` when the judge output is unusable."""
        rubric = "\n".join(
            f"- {c.name} (weight {c.weight}): {c.description}" for c in labels.rubric or []
        )
        concepts = [
            a
            for a in [*labels.required_assertions, *labels.forbidden_assertions]
            if a.type == "concept"
        ]
        concept_text = "\n".join(f"- {a.key()}: {a.description or a.value}" for a in concepts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You grade an AI assistant response for an evaluation suite. Score "
                    "helpfulness, accuracy and completeness from 1 to 5 using the rubric and "
                    "reference. For each listed concept say whether the response expresses it. "
                    'Reply with JSON only: {"helpfulness": int, "accuracy": int, '
                    '"completeness": int, "concepts": {"<key>": true|false}, "rationale": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Expected behaviour: {labels.expected_behavior.value}\n\nUser prompt:\n{prompt}"
                    f"\n\nReference answer:\n{labels.reference_answer or '(none)'}"
                    f"\n\nRubric:\n{rubric or '(none)'}\n\nConcepts:\n{concept_text or '(none)'}"
                    f"\n\nResponse to grade:\n{response}"
                ),
            },
        ]
        parsed = parse_json_object(self._chat.complete(messages, temperature=0.0).text)
        try:
            concepts_raw = parsed.get("concepts") or {}
            return LLMGrade(
                helpfulness=max(1, min(5, int(parsed["helpfulness"]))),
                accuracy=max(1, min(5, int(parsed["accuracy"]))),
                completeness=max(1, min(5, int(parsed["completeness"]))),
                concept_verdicts={str(k): bool(v) for k, v in concepts_raw.items()},
                rationale=str(parsed.get("rationale", ""))[:1000],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"unusable grader output: {exc}") from exc


@dataclass
class ScoreResult:
    """Outcome of scoring one response."""

    passed: bool
    quality_score: float
    detected_behavior: str
    metric_scores: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


class Scorer:
    """Scores responses in ``deterministic``, ``llm`` or ``hybrid`` mode."""

    def __init__(
        self,
        grading_mode: str = "deterministic",
        pass_score_tolerance: float = 1.0,
        grader: LLMGrader | None = None,
    ) -> None:
        if grading_mode in ("llm", "hybrid") and grader is None:
            raise ValueError(f"grading mode '{grading_mode}' requires an LLM grader")
        self._mode = grading_mode
        self._tolerance = pass_score_tolerance
        self._grader = grader

    @property
    def mode(self) -> str:
        return self._mode

    def score(
        self, prompt: str, response: str, labels: LabelSet, error: str | None = None
    ) -> ScoreResult:
        """Score one response against a label set."""
        if error or not response.strip():
            reason = f"model call failed: {error}" if error else "empty response"
            return ScoreResult(
                passed=False,
                quality_score=1.0,
                detected_behavior="",
                metric_scores={
                    "behavior": 0.0,
                    "required_assertions": 0.0,
                    "forbidden_assertions": 0.0,
                    "content": 0.0,
                },
                failure_reasons=[reason],
            )
        failures: list[str] = []
        behavior = detect_behavior(response)
        behavior_ok = behavior is labels.expected_behavior
        if not behavior_ok:
            failures.append(
                f"expected behavior '{labels.expected_behavior.value}' but response was "
                f"'{behavior.value}'"
            )

        grade: LLMGrade | None = None
        grader_error: str | None = None
        if self._grader is not None and self._mode in ("llm", "hybrid"):
            try:
                grade = self._grader.grade(prompt, response, labels)
            except LLMError as exc:
                grader_error = str(exc)
                log.warning("scoring.grader_failed", error=grader_error)

        def holds(assertion: Assertion) -> bool:
            if assertion.type == "concept" and grade and assertion.key() in grade.concept_verdicts:
                return grade.concept_verdicts[assertion.key()]
            return check_assertion(assertion, response)

        required = labels.required_assertions
        missing = [a for a in required if not holds(a)]
        violated = [a for a in labels.forbidden_assertions if holds(a)]
        for assertion in missing:
            failures.append(f"missing required: {assertion.description or assertion.value}")
        for assertion in violated:
            failures.append(
                f"forbidden content present: {assertion.description or assertion.value}"
            )
        required_score = 1.0 - len(missing) / len(required) if required else 1.0
        forbidden_score = (
            1.0 - len(violated) / len(labels.forbidden_assertions)
            if labels.forbidden_assertions
            else 1.0
        )

        metrics: dict[str, Any] = {
            "behavior": 1.0 if behavior_ok else 0.0,
            "required_assertions": round(required_score, 4),
            "forbidden_assertions": round(forbidden_score, 4),
        }
        if labels.reference_answer and labels.expected_behavior is ExpectedBehavior.ANSWER:
            similarity = token_f1(response, labels.reference_answer)
            metrics["reference_similarity"] = round(similarity, 4)
            content = min(1.0, similarity / 0.6)
        elif labels.rubric:
            content = rubric_coverage(labels.rubric, response)
            metrics["rubric_coverage"] = round(content, 4)
        else:
            content = required_score
        metrics["content"] = round(content, 4)

        deterministic = 1.0 + 4.0 * (
            0.30 * metrics["behavior"]
            + 0.35 * required_score
            + 0.15 * forbidden_score
            + 0.20 * content
        )
        if not behavior_ok:
            deterministic = min(deterministic, 2.0)
        if violated:
            deterministic = min(deterministic, 3.0)

        if grade is not None:
            metrics.update(
                helpfulness=grade.helpfulness,
                accuracy=grade.accuracy,
                completeness=grade.completeness,
                grader_rationale=grade.rationale,
            )
            quality = grade.score if self._mode == "llm" else (grade.score + deterministic) / 2.0
        else:
            quality = deterministic
            if grader_error:
                metrics["grader_error"] = grader_error
        quality = round(max(1.0, min(5.0, quality)), 3)

        threshold = labels.expected_quality_score - self._tolerance
        meets_threshold = quality >= threshold
        metrics["quality_threshold"] = 1.0 if meets_threshold else 0.0
        if not meets_threshold:
            failures.append(f"quality score {quality:.2f} below required {threshold:.2f}")
        passed = behavior_ok and not missing and not violated and meets_threshold
        return ScoreResult(passed, quality, behavior.value, metrics, failures)
