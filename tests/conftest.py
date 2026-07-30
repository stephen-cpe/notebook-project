"""Shared pytest fixtures.

Tests run fully offline:
- ``AI_MOCK=true`` and ``CI=true`` force mock AI + in-memory ChromaDB.
- SQLAlchemy uses SQLite in-memory (per-test isolation).
- PostgreSQL-only validation is bypassed per-test via ``is_test()``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Ensure offline/test flags are set BEFORE importing the app.
os.environ.setdefault("AI_MOCK", "true")
os.environ.setdefault("CI", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("HF_TOKEN", "")
# Keep Chroma Cloud credentials empty in tests so the local/ephemeral backend
# is always selected (CI=true wins anyway, but this prevents the real .env
# from leaking cloud creds into config-default assertions).
os.environ.setdefault("CHROMA_CLOUD_API_KEY", "")
os.environ.setdefault("CHROMA_CLOUD_CONNECTION_STRING", "")
os.environ.setdefault("CHROMA_DB", "local")

# Make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()


@pytest.fixture()
def app() -> Generator[object]:
    """Create a fresh app + isolated DB for each test.

    Uses a temporary file-based SQLite database instead of ``:memory:`` so
    that background job threads and multiple app contexts share the same
    database (P0-2.26). In-memory ``:memory:`` SQLite gives each connection
    its own private database, which breaks session-invalidation and threaded
    job tests. The temp file is removed on teardown.
    """
    import tempfile

    from src.app import create_app
    from src.config import Config
    from src.extensions import db

    # Reset ChromaDB vector store so collections from prior tests don't leak
    # (EphemeralClient is process-global; without this, ingestion dedup
    # short-circuits and ContentRegistry entries aren't created in the new DB).
    from src.services.vector_store import get_vector_store, reset_vector_store

    reset_vector_store()
    get_vector_store().reset()

    # Use a temp file SQLite DB so all threads/contexts share one database.
    tmp_dir = tempfile.mkdtemp(prefix="nbtest_")
    tmp_db = os.path.join(tmp_dir, "test.db")
    test_uri = f"sqlite:///{tmp_db}"
    os.environ["TEST_DATABASE_URL"] = test_uri

    cfg = Config()
    application = create_app(cfg)

    # Enable SQLite foreign-key enforcement (cascades) for tests.
    @event.listens_for(Engine, "connect")
    def _enable_fk(dbapi_conn, _records):  # noqa: ANN001
        try:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        except Exception:  # noqa: BLE001
            pass

    with application.app_context():
        db.create_all()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()

    # Clean up the temp DB file + dir.
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def client(app: object) -> object:
    """Flask test client."""
    from flask import Flask

    assert isinstance(app, Flask)
    return app.test_client()
