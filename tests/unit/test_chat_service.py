"""Unit tests for src.services.chat_service (TDD step 15).

Tests the chat orchestration flow: guardrails → retrieve → generate → persist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.auth_service import hash_password
from src.services.chat_service import ChatService

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _setup_notebook_with_source(app: object, username: str = "chatsvc") -> int:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Chat Svc NB")
        db.session.add(nb)
        db.session.commit()
        h = "c" * 64
        db.session.add(
            Source(
                notebook_id=nb.id,
                filename="doc.txt",
                content_hash=h,
                content_type="txt",
                status="ready",
            )
        )
        db.session.commit()
        content_registry_repo.get_or_create(
            content_hash=h,
            chroma_collection="doc_c",
            extracted_text="This document discusses databases, SQL, PostgreSQL, and MongoDB.",
            char_count=60,
        )
        return nb.id


class TestChatSync:
    def test_returns_answer_sources_latency(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _setup_notebook_with_source(app, "chatsvc1")

        svc = ChatService()
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            result = svc.chat_sync(nb, "What databases are mentioned?")
            assert "answer" in result
            assert "sources" in result
            assert "latency_ms" in result
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0

    def test_out_of_scope_returns_refusal(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _setup_notebook_with_source(app, "chatsvc2")

        svc = ChatService()
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            result = svc.chat_sync(nb, "What is the weather like?")
            assert "notebook" in result["answer"].lower() or "sources" in result["answer"].lower()

    def test_no_sources_returns_no_info(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        with app.app_context():
            u = User(username="chatsvc3", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Empty NB")
            db.session.add(nb)
            db.session.commit()
            svc = ChatService()
            result = svc.chat_sync(nb, "What is machine learning?")
            assert "answer" in result
            assert "sources" in result

    def test_persists_messages(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _setup_notebook_with_source(app, "chatsvc4")

        svc = ChatService()
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            svc.chat_sync(nb, "What databases are mentioned?")
            from src.models import ChatMessage

            msgs = db.session.query(ChatMessage).filter_by(notebook_id=nb_id).all()
            assert len(msgs) == 2
            assert msgs[0].role == "user"
            assert msgs[1].role == "assistant"


class TestChatStream:
    def test_yields_sse_frames(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _setup_notebook_with_source(app, "chatsvc5")

        svc = ChatService()
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            frames = list(svc.chat_stream(nb, "What databases are mentioned?"))
            assert len(frames) >= 2
            assert all(f.startswith("data: ") for f in frames)
            import json

            last = json.loads(frames[-1].replace("data: ", ""))
            assert last.get("done") is True
            assert "latency_ms" in last
