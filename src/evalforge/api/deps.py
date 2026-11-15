"""FastAPI dependencies: container, database session, pagination, authorization."""

from __future__ import annotations

import hmac
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from evalforge.container import Container
from evalforge.db.session import session_scope


def get_container(request: Request) -> Container:
    """Return the application container created at startup."""
    container: Container = request.app.state.container
    return container


def get_session(container: Annotated[Container, Depends(get_container)]) -> Iterator[Session]:
    """Request-scoped transactional session (commit on success, else rollback)."""
    with session_scope(container.session_factory) as session:
        yield session


@dataclass(frozen=True)
class Pagination:
    """Validated limit/offset query parameters."""

    limit: int
    offset: int


def pagination(
    limit: Annotated[int, Query(ge=1, le=500, description="Page size")] = 50,
    offset: Annotated[int, Query(ge=0, description="Rows to skip")] = 0,
) -> Pagination:
    return Pagination(limit, offset)


def require_write_access(
    container: Annotated[Container, Depends(get_container)],
    x_api_key: Annotated[str | None, Header()] = None,
    x_actor: Annotated[str | None, Header(max_length=120)] = None,
) -> str:
    """Authorize a state-changing request and return the acting user's name.

    * ``local_dev`` mode: every write is allowed (documented, development only).
    * ``token`` mode: the ``X-API-Key`` header must match the shared token.
    """
    auth = container.settings.auth
    if auth.mode == "token":
        expected = auth.resolve_token()
        if not expected:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"token auth is enabled but {auth.token_env} is not set",
            )
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")
    return (x_actor or "").strip() or ("local-dev" if auth.mode == "local_dev" else "api")


ContainerDep = Annotated[Container, Depends(get_container)]
SessionDep = Annotated[Session, Depends(get_session)]
PageDep = Annotated[Pagination, Depends(pagination)]
ActorDep = Annotated[str, Depends(require_write_access)]
