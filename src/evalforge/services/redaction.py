"""PII redaction applied before any interaction is persisted.

Two layers:

1. Regex rules (built-in plus custom rules from configuration).
2. Optional named-entity detection behind the :class:`EntityRecognizer`
   protocol. The default recognizer is a lightweight, offline, pattern-based
   person-name detector; a spaCy/Presidio recognizer can be injected instead.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from evalforge.config import RedactionSettings
from evalforge.domain.enums import RedactionStatus
from evalforge.domain.schemas import LLMInteraction


def _luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, char in enumerate(digits):
        value = int(char)
        if i % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _credit_card_replacer(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_valid(digits):
        return "[CREDIT_CARD]"
    return match.group(0)


@dataclass(frozen=True)
class CompiledRule:
    """A compiled redaction rule."""

    name: str
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]


# Order matters: more specific patterns run first.
BUILTIN_RULES: dict[str, CompiledRule] = {
    "api_key": CompiledRule(
        "api_key",
        re.compile(
            r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b|\bAKIA[0-9A-Z]{16}\b|"
            r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bBearer\s+[A-Za-z0-9._\-]{20,}"
        ),
        "[API_KEY]",
    ),
    "email": CompiledRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL]",
    ),
    "iban": CompiledRule("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    "credit_card": CompiledRule(
        "credit_card", re.compile(r"\b\d(?:[ \-]?\d){12,18}\b"), _credit_card_replacer
    ),
    "ssn": CompiledRule("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    "phone": CompiledRule(
        "phone",
        re.compile(
            r"(?<![\w$])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\w)"
        ),
        "[PHONE]",
    ),
    "ip_address": CompiledRule(
        "ip_address",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        "[IP_ADDRESS]",
    ),
}


class EntityRecognizer(Protocol):
    """Finds sensitive entity spans that regexes cannot express."""

    def redact(self, text: str) -> tuple[str, int]:
        """Return the redacted text and the number of replaced spans."""
        ...


class HeuristicNameRecognizer:
    """Offline person-name detector based on introduction patterns and titles."""

    _NAME = r"[A-Z][a-z]+(?:[ \-][A-Z][a-z]+){0,2}"
    _PATTERNS = (
        re.compile(rf"\b((?:[Mm]y name is|[Tt]his is|I am|I'm|[Nn]ame:)\s+)({_NAME})"),
        re.compile(rf"\b((?:Mr|Mrs|Ms|Dr|Prof)\.?\s+)({_NAME})"),
        re.compile(rf"\b((?:[Dd]ear|[Rr]egards,|[Ss]incerely,|[Tt]hanks,)\s+)({_NAME})"),
    )
    _NOT_NAMES = frozenset({"Sorry", "Happy", "Sure", "Not", "Here", "Still", "Very", "Glad"})

    def redact(self, text: str) -> tuple[str, int]:
        count = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal count
            if match.group(2).split()[0] in self._NOT_NAMES:
                return match.group(0)
            count += 1
            return f"{match.group(1)}[NAME]"

        for pattern in self._PATTERNS:
            text = pattern.sub(_replace, text)
        return text, count


@dataclass(frozen=True)
class RedactionResult:
    """Redacted interaction plus the number of replaced spans."""

    interaction: LLMInteraction
    redaction_count: int


class Redactor:
    """Applies configured redaction rules to text and whole interactions."""

    TEXT_FIELDS = ("system_prompt", "user_prompt", "raw_response")

    def __init__(
        self, settings: RedactionSettings, recognizer: EntityRecognizer | None = None
    ) -> None:
        self._enabled = settings.enabled
        self._rules: list[CompiledRule] = []
        unknown = [name for name in settings.builtin_rules if name not in BUILTIN_RULES]
        if unknown:
            raise ValueError(f"unknown built-in redaction rules: {', '.join(unknown)}")
        for name in BUILTIN_RULES:  # preserve the safe built-in ordering
            if name in settings.builtin_rules:
                self._rules.append(BUILTIN_RULES[name])
        for rule in settings.custom_rules:
            self._rules.append(CompiledRule(rule.name, re.compile(rule.pattern), rule.replacement))
        self._recognizer: EntityRecognizer | None = (
            recognizer or HeuristicNameRecognizer() if settings.ner_enabled else None
        )
        self._drop_keys = {key.lower() for key in settings.drop_metadata_keys}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def redact_text(self, text: str) -> tuple[str, int]:
        """Redact one string; returns the new text and the replacement count."""
        if not self._enabled or not text:
            return text, 0
        total = 0
        for rule in self._rules:
            if callable(rule.replacement):
                replacer = rule.replacement
                hits = 0

                def _counting(match: re.Match[str], _fn: Any = replacer) -> str:
                    nonlocal hits
                    replaced = str(_fn(match))
                    hits += replaced != match.group(0)
                    return replaced

                text = rule.pattern.sub(_counting, text)
                total += hits
            else:
                text, count = rule.pattern.subn(rule.replacement, text)
                total += count
        if self._recognizer is not None:
            text, count = self._recognizer.redact(text)
            total += count
        return text, total

    def _redact_value(self, value: Any) -> tuple[Any, int]:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return self._redact_mapping(value)
        if isinstance(value, list):
            total = 0
            items = []
            for item in value:
                redacted, count = self._redact_value(item)
                items.append(redacted)
                total += count
            return items, total
        return value, 0

    def _redact_mapping(self, mapping: dict[str, Any]) -> tuple[dict[str, Any], int]:
        total = 0
        out: dict[str, Any] = {}
        for key, value in mapping.items():
            if str(key).lower() in self._drop_keys:
                total += 1
                continue
            out[key], count = self._redact_value(value)
            total += count
        return out, total

    def redact_interaction(self, interaction: LLMInteraction) -> RedactionResult:
        """Return a redacted copy of an interaction with ``redaction_status`` set."""
        if not self._enabled:
            return RedactionResult(
                interaction.model_copy(update={"redaction_status": RedactionStatus.SKIPPED}), 0
            )
        updates: dict[str, Any] = {}
        total = 0
        for field_name in self.TEXT_FIELDS:
            redacted, count = self.redact_text(getattr(interaction, field_name))
            updates[field_name] = redacted
            total += count
        metadata, count = self._redact_mapping(interaction.metadata)
        updates["metadata"] = metadata
        total += count
        updates["redaction_status"] = RedactionStatus.REDACTED if total else RedactionStatus.CLEAN
        updates["response_length"] = len(updates["raw_response"])
        return RedactionResult(interaction.model_copy(update=updates), total)
