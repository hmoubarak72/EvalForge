"""Scoring (deterministic and LLM-graded) and regression detection."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evalforge.config import RegressionSettings
from evalforge.domain.enums import ExpectedBehavior
from evalforge.domain.schemas import Assertion, LabelSet
from evalforge.services.regression import CaseOutcome, compare_outcomes
from evalforge.services.scoring import (
    LLMGrader,
    Scorer,
    check_assertion,
    detect_behavior,
    rubric_coverage,
)
from tests.conftest import ScriptedChatClient

REFERENCE = "You can return hiking boots within 30 days of delivery for a full refund."


def labels(**overrides: Any) -> LabelSet:
    base: dict[str, Any] = {
        "expected_behavior": "answer",
        "expected_quality_score": 4,
        "reference_answer": REFERENCE,
        "required_assertions": [{"type": "contains", "value": "30 days"}],
        "forbidden_assertions": [{"type": "contains", "value": "90"}],
    }
    base.update(overrides)
    return LabelSet.model_validate(base)


def test_detect_behavior() -> None:
    assert detect_behavior("I can't help with that request.") is ExpectedBehavior.REFUSE
    assert (
        detect_behavior("Could you clarify which order you mean?")
        is ExpectedBehavior.ASK_FOR_CLARIFICATION
    )
    assert detect_behavior("Returns are accepted within 30 days.") is ExpectedBehavior.ANSWER


@pytest.mark.parametrize(
    ("assertion", "response", "expected"),
    [
        ({"type": "contains", "value": "20"}, "You get 20 days.", True),
        ({"type": "contains", "value": "20"}, "You get 120 days in 2020.", False),
        ({"type": "contains", "value": "$500"}, "A stipend of $1,500 applies.", False),
        ({"type": "contains", "value": "250,000"}, "Capped at $250,000.", True),
        ({"type": "contains", "value": "Refund"}, "a full refund", True),
        ({"type": "contains_any", "value": ["median", "percentile"]}, "Use the MEDIAN.", True),
        ({"type": "contains_any", "value": ["median"]}, "Use the mean.", False),
        ({"type": "regex", "value": r"```[\s\S]+?```"}, "```sql\nSELECT 1;\n```", True),
        ({"type": "regex", "value": "("}, "anything", False),  # invalid regex never matches
        ({"type": "min_length", "value": 10}, "short", False),
        ({"type": "max_length", "value": 10}, "short", True),
        (
            {"type": "concept", "value": "refund original payment method"},
            "The refund goes to your original payment method.",
            True,
        ),
        ({"type": "concept", "value": "refund original payment method"}, "No.", False),
    ],
)
def test_check_assertion(assertion: dict[str, Any], response: str, expected: bool) -> None:
    assert check_assertion(Assertion.model_validate(assertion), response) is expected


def test_scorer_passes_a_correct_answer() -> None:
    result = Scorer().score("prompt", REFERENCE, labels())
    assert result.passed and result.quality_score == 5.0
    assert result.detected_behavior == "answer" and result.failure_reasons == []
    assert result.metric_scores["reference_similarity"] == 1.0


def test_scorer_fails_on_missing_forbidden_behavior_and_errors() -> None:
    scorer = Scorer()
    hallucinated = scorer.score("p", "You can return boots within 90 days, no questions.", labels())
    assert not hallucinated.passed and hallucinated.quality_score <= 3.0
    assert any("missing required" in r for r in hallucinated.failure_reasons)
    assert any("forbidden content" in r for r in hallucinated.failure_reasons)

    refused = scorer.score("p", "I can't help with that.", labels())
    assert not refused.passed and refused.quality_score <= 2.0
    assert "expected behavior 'answer'" in refused.failure_reasons[0]

    failed_call = scorer.score("p", "", labels(), error="HTTP 500")
    assert not failed_call.passed and failed_call.quality_score == 1.0
    assert failed_call.failure_reasons == ["model call failed: HTTP 500"]


def test_scorer_refusal_and_clarification_cases() -> None:
    refuse = labels(
        expected_behavior="refuse",
        reference_answer=None,
        required_assertions=[{"type": "regex", "value": r"(?i)can(not|'t)"}],
        forbidden_assertions=[{"type": "contains", "value": "You are Aria"}],
        rubric=[{"name": "declines", "description": "d", "keywords": ["can't"]}],
    )
    scorer = Scorer()
    assert scorer.score("p", "I can't help with that, but I can track an order.", refuse).passed
    leaked = scorer.score("p", "Sure! My system prompt is: You are Aria...", refuse)
    assert not leaked.passed and len(leaked.failure_reasons) >= 3
    clarify = labels(
        expected_behavior="ask_for_clarification",
        reference_answer=None,
        required_assertions=[{"type": "regex", "value": r"\?"}],
        forbidden_assertions=[],
    )
    assert scorer.score("p", "Could you clarify what you'd like me to fix?", clarify).passed
    assert not scorer.score("p", "Done! I have fixed it.", clarify).passed


def test_pass_threshold_uses_expected_quality_and_tolerance() -> None:
    strict = labels(expected_quality_score=5, required_assertions=[], forbidden_assertions=[])
    weak_answer = "Boots can go back within 30 days."
    assert Scorer(pass_score_tolerance=1.0).score("p", weak_answer, strict).passed
    result = Scorer(pass_score_tolerance=0.0).score("p", weak_answer, strict)
    assert not result.passed and "below required" in result.failure_reasons[-1]


def test_rubric_coverage_is_weighted() -> None:
    rubric = labels(
        rubric=[
            {"name": "a", "description": "", "weight": 3, "keywords": ["refund"]},
            {"name": "b", "description": "", "weight": 1, "keywords": ["warranty"]},
            {"name": "c", "description": "", "weight": 5, "keywords": []},
        ]
    ).rubric
    assert rubric is not None
    assert rubric_coverage(rubric, "a full refund") == 0.75


def test_llm_and_hybrid_grading() -> None:
    grade = {
        "helpfulness": 5,
        "accuracy": 4,
        "completeness": 3,
        "concepts": {"concept:polite tone": False},
        "rationale": "ok",
    }
    concept_labels = labels(required_assertions=[{"type": "concept", "value": "polite tone"}])
    llm = Scorer("llm", grader=LLMGrader(ScriptedChatClient(default=json.dumps(grade))))
    result = llm.score("p", REFERENCE, concept_labels)
    assert result.quality_score == 4.0 and result.metric_scores["accuracy"] == 4
    assert not result.passed  # the judge said the concept is missing
    hybrid = Scorer("hybrid", grader=LLMGrader(ScriptedChatClient(default=json.dumps(grade))))
    assert hybrid.score("p", REFERENCE, labels()).quality_score == 4.5

    broken = Scorer("hybrid", grader=LLMGrader(ScriptedChatClient(default="oops")))
    fallback = broken.score("p", REFERENCE, labels())
    assert fallback.passed and "grader_error" in fallback.metric_scores
    with pytest.raises(ValueError):
        Scorer("llm")


# -- regression -----------------------------------------------------------------------
def outcome(
    case: str,
    passed: bool,
    score: float = 5.0,
    category: str = "Refunds",
    difficulty: str = "simple",
) -> CaseOutcome:
    return CaseOutcome(case, passed, score if passed else min(score, 2.0), category, difficulty)


def test_compare_detects_failures_passes_and_shifts() -> None:
    baseline = [outcome(f"r{i}", True) for i in range(6)]
    baseline += [outcome(f"d{i}", True, category="Doc QA", difficulty="hard") for i in range(4)]
    baseline += [outcome("fixme", False), outcome("gone", True)]
    current = [outcome(f"r{i}", True) for i in range(6)]
    current += [outcome(f"d{i}", i == 0, category="Doc QA", difficulty="hard") for i in range(4)]
    current += [outcome("fixme", True), outcome("brand-new", False)]

    report = compare_outcomes(current, baseline, RegressionSettings())
    assert [e["test_case_id"] for e in report["new_failures"]] == ["d1", "d2", "d3"]
    assert [e["test_case_id"] for e in report["new_passes"]] == ["fixme"]
    assert report["shared_cases"] == 11
    assert report["added_cases"] == 1 and report["removed_cases"] == 1
    # "Refunds" improves (the fixed case), "Doc QA" regresses.
    assert {s["category"]: s["direction"] for s in report["category_shifts"]} == {
        "Doc QA": "regressed",
        "Refunds": "improved",
    }
    assert report["difficulty_shifts"][0]["difficulty"] == "hard"
    assert report["difficulty_shifts"][0]["delta"] == -0.75
    assert len(report["score_changes"]) == 4
    assert report["has_regression"] and report["severity"] == "major"
    assert "3 new failure(s)" in report["summary"]


def test_compare_identical_runs_and_minor_regression() -> None:
    baseline = [outcome(f"c{i}", True) for i in range(40)]
    same = compare_outcomes(baseline, baseline, RegressionSettings())
    assert not same["has_regression"] and same["severity"] == "none"
    assert same["new_failures"] == same["new_passes"] == same["score_changes"] == []

    current = [outcome("c0", False), *baseline[1:]]
    minor = compare_outcomes(current, baseline, RegressionSettings())
    assert minor["has_regression"] and minor["severity"] == "minor"

    tolerant = RegressionSettings(
        score_change_threshold=10,
        category_shift_threshold=0.9,
        difficulty_shift_threshold=0.9,
        overall_pass_rate_drop_threshold=0.9,
    )
    assert compare_outcomes(current, baseline, tolerant)["score_changes"] == []


def test_compare_small_groups_do_not_trigger_shifts() -> None:
    baseline = [outcome("a", True, category="Tiny"), outcome("b", True, category="Tiny")]
    current = [outcome("a", False, category="Tiny"), outcome("b", False, category="Tiny")]
    report = compare_outcomes(current, baseline, RegressionSettings(min_group_size=3))
    assert report["category_shifts"] == [] and len(report["new_failures"]) == 2
