"""Alembic environment configuration for notebook-project.

Wired to Flask-Migrate: reads the Flask app's SQLAlchemy metadata so
``flask db migrate`` and ``flask db upgrade`` work correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

logging.basicConfig(level=logging.WARNING)

config = context.config
root = Path(__file__).resolve().parent.parent
config.set_main_option("script_location", str(root / "migrations"))

from src.app import create_app  # noqa: E402

app = create_app()
target_metadata = app.extensions["migrate"].db.metadata

if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", app.config["SQLALCHEMY_DATABASE_URI"])


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL without a DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to the DB and apply)."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
