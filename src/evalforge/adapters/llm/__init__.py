"""Model-provider adapters and factories.

To add a provider, implement the :class:`ChatClient` / :class:`EmbeddingClient`
protocols and extend the two factory functions below.
"""

from __future__ import annotations

from evalforge.adapters.llm.base import (
    ChatClient,
    ChatResult,
    EmbeddingClient,
    LLMError,
    Message,
    parse_json_object,
)
from evalforge.adapters.llm.mock import (
    MOCK_PROFILES,
    MockChatClient,
    MockEmbeddingClient,
    MockProfile,
)
from evalforge.adapters.llm.openai_compat import (
    OpenAICompatibleChatClient,
    OpenAICompatibleEmbeddingClient,
)
from evalforge.config import EmbeddingSettings, ModelProviderSettings, Settings
from evalforge.domain.schemas import EndpointConfig


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create the embedding client for the configured provider."""
    cfg: EmbeddingSettings = settings.embedding
    if settings.provider_for(cfg) == "mock":
        model = cfg.model if cfg.provider == "mock" else f"mock-embedding-{cfg.dimensions}"
        return MockEmbeddingClient(model=model, dimensions=cfg.dimensions)
    return OpenAICompatibleEmbeddingClient(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.resolve_api_key(),
        batch_size=cfg.batch_size,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )


def build_chat_client(settings: Settings, role: ModelProviderSettings) -> ChatClient | None:
    """Create a chat client for a pipeline role (judge, labeler).

    Returns ``None`` in offline mode or when the role uses the mock provider;
    callers then use their deterministic heuristic implementation.
    """
    if settings.provider_for(role) == "mock":
        return None
    return OpenAICompatibleChatClient(
        base_url=role.base_url,
        model=role.model,
        api_key=role.resolve_api_key(),
        temperature=role.temperature,
        max_tokens=role.max_tokens,
        timeout_s=role.timeout_s,
        max_retries=role.max_retries,
    )


def build_target_client(endpoint: EndpointConfig) -> ChatClient:
    """Create the client for a model endpoint under evaluation."""
    if endpoint.provider == "mock":
        return MockChatClient(model=endpoint.model)
    import os

    return OpenAICompatibleChatClient(
        base_url=endpoint.base_url,
        model=endpoint.model,
        api_key=os.environ.get(endpoint.api_key_env),
        temperature=endpoint.temperature,
        max_tokens=endpoint.max_tokens,
        timeout_s=endpoint.timeout_s,
        max_retries=endpoint.max_retries,
        retry_backoff_s=endpoint.retry_backoff_s,
    )


__all__ = [
    "MOCK_PROFILES",
    "ChatClient",
    "ChatResult",
    "EmbeddingClient",
    "LLMError",
    "Message",
    "MockChatClient",
    "MockEmbeddingClient",
    "MockProfile",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleEmbeddingClient",
    "build_chat_client",
    "build_embedding_client",
    "build_target_client",
    "parse_json_object",
]
