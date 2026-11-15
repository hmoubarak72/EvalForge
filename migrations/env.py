"""Alembic environment.

The database URL comes from ``-x database_url=...``, a URL set programmatically
on the Alembic config, or the application settings (EVALFORGE_DATABASE_URL).
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import evalforge.db.models  # noqa: F401  (register all tables)
from evalforge.config import get_settings
from evalforge.db.base import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    return (
        x_args.get("database_url")
        or config.get_main_option("sqlalchemy.url")
        or get_settings().database_url
    )


def run_migrations_offline() -> None:
    """Emit SQL without connecting to the database."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
