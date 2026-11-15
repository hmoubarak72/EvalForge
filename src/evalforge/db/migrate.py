"""Programmatic schema management (used by the CLI and the demo seed)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, text

from evalforge.db.base import Base
from evalforge.db.session import create_all
from evalforge.telemetry import get_logger

log = get_logger(__name__)


def find_alembic_ini(start: Path | None = None) -> Path | None:
    """Locate ``alembic.ini`` in the working directory or the repository root."""
    candidates = [
        (start or Path.cwd()) / "alembic.ini",
        Path(__file__).resolve().parents[3] / "alembic.ini",
    ]
    return next((path for path in candidates if path.exists()), None)


def upgrade_to_head(database_url: str) -> str:
    """Bring the schema to the latest revision.

    Uses the Alembic migrations when the repository layout is available and
    falls back to ``create_all`` otherwise (e.g. an installed wheel).
    Returns the method used: ``"alembic"`` or ``"create_all"``.
    """
    ini = find_alembic_ini()
    if ini is None:
        from evalforge.db.session import make_engine

        log.warning("migrate.alembic_ini_missing", fallback="create_all")
        engine = make_engine(database_url)
        try:
            create_all(engine)
        finally:
            engine.dispose()
        return "create_all"
    from alembic import command
    from alembic.config import Config

    config = Config(str(ini))
    config.set_main_option("script_location", str(ini.parent / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["configure_logging"] = False
    command.upgrade(config, "head")
    return "alembic"


def drop_everything(engine: Engine) -> None:
    """Drop all EvalForge tables and Alembic's version table (demo reset only)."""
    import evalforge.db.models  # noqa: F401

    Base.metadata.drop_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
