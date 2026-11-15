"""OpenAI-compatible HTTP clients (chat completions and embeddings).

Works with any server implementing the ``/chat/completions`` and
``/embeddings`` routes: OpenAI, Azure-compatible gateways, vLLM, Ollama,
LiteLLM, and similar. Uses plain ``httpx`` so no vendor SDK is required.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import httpx

from evalforge.adapters.llm.base import ChatResult, LLMError, Message
from evalforge.telemetry import get_logger

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


class OpenAICompatibleChatClient:
    """Chat client for OpenAI-compatible endpoints with retry and backoff."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._backoff = retry_backoff_s
        self._transport = transport
        self._async_transport = async_transport

    def _payload(
        self, messages: Sequence[Message], temperature: float | None, max_tokens: int | None
    ) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }

    def _parse(self, body: dict[str, Any], started: float, retries: int) -> ChatResult:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("malformed chat completion response") from exc
        usage = body.get("usage") or {}
        return ChatResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            retries=retries,
            model=str(body.get("model", self.model_name)),
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Blocking chat completion with bounded retries."""
        payload = self._payload(messages, temperature, max_tokens)
        started = time.perf_counter()
        last_error: Exception | None = None
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            for attempt in range(self._max_retries + 1):
                try:
                    response = client.post(self._url, json=payload, headers=_headers(self._api_key))
                    response.raise_for_status()
                    return self._parse(response.json(), started, attempt)
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt >= self._max_retries or not _should_retry(exc):
                        break
                    log.warning("llm.retry", attempt=attempt + 1, reason=_describe(exc))
                    time.sleep(self._backoff * (2**attempt))
        raise LLMError(f"chat completion failed: {_describe(last_error or Exception())}")

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Async chat completion with bounded retries."""
        payload = self._payload(messages, temperature, max_tokens)
        started = time.perf_counter()
        last_error: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._async_transport
        ) as client:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await client.post(
                        self._url, json=payload, headers=_headers(self._api_key)
                    )
                    response.raise_for_status()
                    return self._parse(response.json(), started, attempt)
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt >= self._max_retries or not _should_retry(exc):
                        break
                    log.warning("llm.retry", attempt=attempt + 1, reason=_describe(exc))
                    await asyncio.sleep(self._backoff * (2**attempt))
        raise LLMError(f"chat completion failed: {_describe(last_error or Exception())}")


class OpenAICompatibleEmbeddingClient:
    """Embedding client for OpenAI-compatible ``/embeddings`` endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._batch_size = batch_size
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._transport = transport

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches, preserving input order."""
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                vectors.extend(self._embed_batch(client, batch))
        return vectors

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = client.post(
                    self._url,
                    json={"model": self.model_name, "input": batch},
                    headers=_headers(self._api_key),
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                return [list(map(float, item["embedding"])) for item in data]
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self._max_retries or not _should_retry(exc):
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError("malformed embeddings response") from exc
        raise LLMError(f"embedding request failed: {_describe(last_error or Exception())}")
