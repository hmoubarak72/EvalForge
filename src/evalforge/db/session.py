"""Engine and session factories (constructed explicitly, never global)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from evalforge.db.base import Base


def make_engine(database_url: str, echo: bool = False) -> Engine:
    """Create an engine for PostgreSQL or the SQLite development mode."""
    if database_url.startswith("sqlite"):
        kwargs: dict[str, object] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in database_url or database_url in ("sqlite://", "sqlite:///"):
            kwargs["poolclass"] = StaticPool
        engine = create_engine(database_url, echo=echo, **kwargs)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(database_url, echo=echo, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to an engine."""
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=True)


def create_all(engine: Engine) -> None:
    """Create all tables directly (tests and quick local demos).

    Production deployments should use the Alembic migrations instead.
    """
    import evalforge.db.models  # noqa: F401  (register models on the metadata)

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
