"""Provider-agnostic interfaces for embedding and chat models."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]


class LLMError(Exception):
    """Raised when a model call fails after retries or returns unusable output."""


@dataclass(frozen=True)
class ChatResult:
    """Result of one chat completion."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    retries: int = 0
    model: str = ""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Turns texts into fixed-size vectors."""

    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""
        ...


@runtime_checkable
class ChatClient(Protocol):
    """Chat-completion client with sync and async entry points."""

    model_name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Blocking completion."""
        ...

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Async completion (used by the concurrent evaluation runner)."""
        ...


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output.

    Raises:
        LLMError: if no valid JSON object can be found.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?|```$", "", candidate.strip()).strip()
    match = _JSON_BLOCK.search(candidate)
    if not match:
        raise LLMError("model output did not contain a JSON object")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"model output was not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model output JSON was not an object")
    return parsed
