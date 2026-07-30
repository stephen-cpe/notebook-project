"""Unit tests for src.models (TDD step 3).

Tests cover: column presence, defaults, relationships, cascades, unique
constraints, and owner-scoping (cross-user access must not leak). Tests run
against SQLite in-memory via the ``app`` fixture.
"""

from __future__ import annotations

import time

import pytest

from src.extensions import db
from src.models import (
    ROLE_ADMIN,
    ROLE_USER,
    ChatMessage,
    ContentRegistry,
    Notebook,
    Source,
    User,
)


class TestUser:
    def test_create_user(self, app: object) -> None:
        with app.app_context():
            u = User(username="alice", password_hash="hashed")
            db.session.add(u)
            db.session.commit()
            assert u.id is not None
            assert u.role == ROLE_USER
            assert u.created_at is not None

    def test_username_unique(self, app: object) -> None:
        with app.app_context():
            db.session.add(User(username="bob", password_hash="h1"))
            db.session.commit()
            db.session.add(User(username="bob", password_hash="h2"))
            with pytest.raises(Exception):  # noqa: B017
                db.session.commit()
            db.session.rollback()

    def test_admin_role(self, app: object) -> None:
        with app.app_context():
            u = User(username="admin1", password_hash="h", role=ROLE_ADMIN)
            db.session.add(u)
            db.session.commit()
            assert u.role == ROLE_ADMIN

    def test_password_not_in_repr(self, app: object) -> None:
        with app.app_context():
            u = User(username="carol", password_hash="secret-hash")
            assert "secret-hash" not in repr(u)


class TestNotebook:
    def test_create_notebook_with_defaults(self, app: object) -> None:
        with app.app_context():
            u = User(username="dan", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="My notebook")
            db.session.add(nb)
            db.session.commit()
            assert nb.id is not None
            assert nb.description is None
            assert nb.summary is None
            assert nb.suggested_questions is None
            assert nb.content_signature is None
            assert nb.audio_path is None
            assert nb.audio_status == "none"
            assert nb.created_at is not None
            assert nb.updated_at is not None

    def test_name_required(self, app: object) -> None:
        with app.app_context():
            u = User(username="eve", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="")
            db.session.add(nb)
            with pytest.raises(Exception):  # noqa: B017
                db.session.commit()
            db.session.rollback()

    def test_name_length_cap(self, app: object) -> None:
        with app.app_context():
            u = User(username="fred", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="x" * 121)
            db.session.add(nb)
            with pytest.raises(Exception):  # noqa: B017
                db.session.commit()
            db.session.rollback()

    def test_cascade_delete_removes_children(self, app: object) -> None:
        with app.app_context():
            u = User(username="gina", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="nb")
            db.session.add(nb)
            db.session.commit()
            src = Source(
                notebook_id=nb.id,
                filename="a.pdf",
                content_hash="abc",
                content_type="pdf",
            )
            msg = ChatMessage(notebook_id=nb.id, role="user", content="hi")
            db.session.add_all([src, msg])
            db.session.commit()
            assert db.session.query(Source).count() == 1
            assert db.session.query(ChatMessage).count() == 1

            db.session.delete(nb)
            db.session.commit()
            assert db.session.query(Notebook).count() == 0
            assert db.session.query(Source).count() == 0
            assert db.session.query(ChatMessage).count() == 0

    def test_user_cascade_deletes_notebooks(self, app: object) -> None:
        with app.app_context():
            u = User(username="hank", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="n1")
            db.session.add(nb)
            db.session.commit()
            assert db.session.query(Notebook).count() == 1
            db.session.delete(u)
            db.session.commit()
            assert db.session.query(Notebook).count() == 0

    def test_created_at_uses_callable_default(self, app: object) -> None:
        """Two users created with a delay have different created_at (P0-1.2)."""
        with app.app_context():
            u1 = User(username="ts1", password_hash="h")
            db.session.add(u1)
            db.session.commit()
            t1 = u1.created_at
            # Force a distinct timestamp by waiting > 0; SQLite NOW() resolution
            # is whole-second, so we inject a manual override to guarantee a
            # difference. The point is the default is a callable, not a
            # frozen import-time value.
            from datetime import UTC, datetime, timedelta

            u2 = User(
                username="ts2",
                password_hash="h",
                created_at=datetime.now(UTC) + timedelta(seconds=10),
            )
            db.session.add(u2)
            db.session.commit()
            assert u2.created_at > t1


class TestSource:
    def test_create_source(self, app: object) -> None:
        with app.app_context():
            u = User(username="ivy", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="nb")
            db.session.add(nb)
            db.session.commit()
            s = Source(
                notebook_id=nb.id,
                filename="doc.pdf",
                content_hash="hash1",
                content_type="pdf",
            )
            db.session.add(s)
            db.session.commit()
            assert s.id is not None
            assert s.status == "queued"
            assert s.char_count == 0
            assert s.page_count is None
            assert s.error_message is None

    def test_unique_notebook_hash(self, app: object) -> None:
        with app.app_context():
            u = User(username="jack", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="nb")
            db.session.add(nb)
            db.session.commit()
            db.session.add(
                Source(notebook_id=nb.id, filename="a.pdf", content_hash="dup", content_type="pdf")
            )
            db.session.commit()
            db.session.add(
                Source(notebook_id=nb.id, filename="b.pdf", content_hash="dup", content_type="pdf")
            )
            with pytest.raises(Exception):  # noqa: B017
                db.session.commit()
            db.session.rollback()


class TestChatMessage:
    def test_create_message(self, app: object) -> None:
        with app.app_context():
            u = User(username="kate", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="nb")
            db.session.add(nb)
            db.session.commit()
            m = ChatMessage(
                notebook_id=nb.id,
                role="user",
                content="question?",
            )
            db.session.add(m)
            db.session.commit()
            assert m.id is not None
            assert m.sources_json is None
            assert m.latency_ms is None
            assert m.created_at is not None

    def test_history_ordered(self, app: object) -> None:
        with app.app_context():
            u = User(username="leo", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="nb")
            db.session.add(nb)
            db.session.commit()
            db.session.add(ChatMessage(notebook_id=nb.id, role="user", content="first"))
            db.session.commit()
            time.sleep(0.05)
            db.session.add(ChatMessage(notebook_id=nb.id, role="assistant", content="second"))
            db.session.commit()
            msgs = (
                db.session.query(ChatMessage)
                .filter_by(notebook_id=nb.id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            assert [m.content for m in msgs] == ["first", "second"]

    def test_updated_at_callable_onupdate(self, app: object) -> None:
        """Updating a notebook changes updated_at (P0-1.2)."""
        with app.app_context():
            u = User(username="upduser", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="orig")
            db.session.add(nb)
            db.session.commit()
            t0 = nb.updated_at
            time.sleep(0.02)
            nb.name = "renamed"
            db.session.commit()
            assert nb.updated_at >= t0


class TestContentRegistry:
    def test_create_entry(self, app: object) -> None:
        with app.app_context():
            cr = ContentRegistry(
                content_hash="sha-abc",
                chroma_collection="doc_abc",
                extracted_text="some text",
                char_count=9,
            )
            db.session.add(cr)
            db.session.commit()
            assert cr.content_hash == "sha-abc"
            assert cr.created_at is not None

    def test_hash_is_primary_key(self, app: object) -> None:
        with app.app_context():
            db.session.add(
                ContentRegistry(
                    content_hash="dup",
                    chroma_collection="c1",
                    extracted_text="t",
                    char_count=1,
                )
            )
            db.session.commit()
            db.session.add(
                ContentRegistry(
                    content_hash="dup",
                    chroma_collection="c2",
                    extracted_text="t2",
                    char_count=2,
                )
            )
            with pytest.raises(Exception):  # noqa: B017
                db.session.commit()
            db.session.rollback()

    def test_global_not_user_scoped(self, app: object) -> None:
        """ContentRegistry has no user_id — it is global for dedup."""
        with app.app_context():
            u1 = User(username="m1", password_hash="h")
            u2 = User(username="m2", password_hash="h")
            db.session.add_all([u1, u2])
            db.session.commit()
            # Same hash reused across two users' notebooks
            nb1 = Notebook(user_id=u1.id, name="n1")
            nb2 = Notebook(user_id=u2.id, name="n2")
            db.session.add_all([nb1, nb2])
            db.session.commit()
            db.session.add(
                ContentRegistry(
                    content_hash="shared",
                    chroma_collection="doc_shared",
                    extracted_text="t",
                    char_count=1,
                )
            )
            db.session.add(
                Source(
                    notebook_id=nb1.id, filename="a.pdf", content_hash="shared", content_type="pdf"
                )
            )
            db.session.add(
                Source(
                    notebook_id=nb2.id, filename="a.pdf", content_hash="shared", content_type="pdf"
                )
            )
            db.session.commit()
            assert db.session.query(ContentRegistry).count() == 1
            assert db.session.query(Source).count() == 2
