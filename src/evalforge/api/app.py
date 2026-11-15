"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from evalforge import __version__
from evalforge.api.routes import (
    datasets,
    evaluations,
    health,
    interactions,
    metrics,
    pipeline,
    review,
    sources,
)
from evalforge.api.schemas import ErrorResponse
from evalforge.config import Settings, get_settings
from evalforge.container import Container, build_container, sync_configured_sources
from evalforge.db.session import session_scope
from evalforge.domain.errors import (
    ConflictError,
    EvalForgeError,
    InvalidRequestError,
    NotFoundError,
)
from evalforge.telemetry import (
    bind_correlation_id,
    configure_logging,
    current_correlation_id,
    get_logger,
)

log = get_logger(__name__)

API_PREFIX = "/api/v1"
_STATUS_BY_ERROR: dict[type[EvalForgeError], int] = {
    NotFoundError: 404,
    ConflictError: 409,
    InvalidRequestError: 400,
}
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 422)
}


def _error(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "correlation_id": current_correlation_id(),
    }
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content={"error": body})


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Create the API. Pass a pre-built container in tests."""
    settings = settings or (container.settings if container else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        app.state.container = container or build_container(settings)
        with session_scope(app.state.container.session_factory) as session:
            sync_configured_sources(session, settings)
        scheduler = None
        if settings.scheduler.enabled:
            from evalforge.pipeline.scheduler import build_scheduler

            scheduler = build_scheduler(app.state.container)
            scheduler.start()
        log.info("api.started", version=__version__, offline_mode=settings.offline_mode)
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(
        title="EvalForge API",
        version=__version__,
        description=(
            "Turn production LLM traffic into a self-growing, production-representative "
            "evaluation dataset. Write endpoints require the `X-API-Key` header when "
            "`EVALFORGE_AUTH__MODE=token`."
        ),
        lifespan=lifespan,
        responses=_ERROR_RESPONSES,
    )
    if container is not None:  # make the container available without lifespan (tests)
        app.state.container = container

    @app.middleware("http")
    async def correlation_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Correlation-ID")
        with bind_correlation_id(incoming, path=request.url.path, method=request.method) as cid:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = cid
            log.info("api.request", status_code=response.status_code)
            return response

    @app.exception_handler(EvalForgeError)
    async def handle_domain_error(_: Request, exc: EvalForgeError) -> JSONResponse:
        return _error(_STATUS_BY_ERROR.get(type(exc), 400), exc.code, str(exc))

    @app.exception_handler(HTTPException)
    async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
        codes = {401: "unauthorized", 403: "forbidden", 404: "not_found", 503: "unavailable"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return _error(422, "validation_error", "request validation failed", details)

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("api.unhandled_error", error=type(exc).__name__, exc_info=True)
        return _error(500, "internal_error", "an unexpected error occurred")

    app.include_router(health.router)
    for module in (sources, pipeline, interactions, datasets, review, evaluations, metrics):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn evalforge.api.app:app_factory --factory``."""
    return create_app()
