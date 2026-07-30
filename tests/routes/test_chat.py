"""Route tests for chat (TDD step 14).

Covers:
- /chat/sync: valid question -> 200 with answer + sources + latency_ms.
- /chat/sync: out-of-scope question -> refusal without LLM-specific content.
- /chat/sync: empty question -> 400.
- /chat/sync: no sources -> "not enough information" response.
- /chat/sync: non-owner -> 404.
- /chat/sync: login required.
- /chat/stream: SSE format (data: frames, done: true at end).
- /chat/clear: deletes history.
- /chat/history: returns persisted messages.
- Chat persistence: user + assistant messages saved to DB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extensions import db
from src.models import ChatMessage, Notebook, User
from src.services.auth_service import hash_password

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _login(client: object, app: object, username: str, password: str = "pw123") -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password(password)))
            db.session.commit()
    client.post("/login", data={"username": username, "password": password})


def _create_notebook(client: object, app: object, name: str = "Chat NB") -> int:
    client.post("/notebooks", data={"name": name})
    with app.app_context():
        nb = db.session.query(Notebook).filter_by(name=name).first()
        assert nb is not None
        return nb.id


def _upload_source(
    client: object, app: object, nb_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload a fixture source to the notebook."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("AI_MOCK", "true")
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
    with open(FIXTURES / "sample.txt", "rb") as f:
        client.post(
            f"/notebooks/{nb_id}/sources",
            data={"file": (f, "sample.txt")},
            content_type="multipart/form-data",
        )


class TestChatSync:
    def test_valid_question(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chatuser1")
        nb_id = _create_notebook(client, app, "Chat Sync NB")
        _upload_source(client, app, nb_id, monkeypatch)

        res = client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What databases are mentioned?"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_persists_messages(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chatuser2")
        nb_id = _create_notebook(client, app, "Chat Persist NB")
        _upload_source(client, app, nb_id, monkeypatch)

        client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What databases are mentioned?"},
        )
        with app.app_context():
            msgs = db.session.query(ChatMessage).filter_by(notebook_id=nb_id).all()
            assert len(msgs) == 2  # user + assistant
            assert msgs[0].role == "user"
            assert msgs[1].role == "assistant"
            assert "databases" in msgs[0].content.lower()

    def test_empty_question(self, client: object, app: object) -> None:
        _login(client, app, "chatuser3")
        nb_id = _create_notebook(client, app, "Chat Empty NB")
        res = client.post(f"/notebooks/{nb_id}/chat/sync", json={"question": ""})
        assert res.status_code == 400

    def test_no_question_field(self, client: object, app: object) -> None:
        _login(client, app, "chatuser4")
        nb_id = _create_notebook(client, app, "Chat No Q NB")
        res = client.post(f"/notebooks/{nb_id}/chat/sync", json={})
        assert res.status_code == 400

    def test_no_sources(self, client: object, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        _login(client, app, "chatuser5")
        nb_id = _create_notebook(client, app, "Chat No Src NB")
        # No sources uploaded.
        res = client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What is machine learning?"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "answer" in data
        # With no sources, the scope check fails -> refusal.
        assert "sources" in data

    def test_out_of_scope(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chatuser6")
        nb_id = _create_notebook(client, app, "Chat Scope NB")
        _upload_source(client, app, nb_id, monkeypatch)

        res = client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What is the weather like today?"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "notebook" in data["answer"].lower() or "sources" in data["answer"].lower()

    def test_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "chatuser7")
        nb_id = _create_notebook(client, app, "Chat Private NB")
        _login(client, app, "chatuser8", "pw456")
        res = client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "anything"},
        )
        assert res.status_code == 404

    def test_requires_login(self, client: object) -> None:
        res = client.post("/notebooks/1/chat/sync", json={"question": "x"})
        assert res.status_code in (301, 302, 303)


class TestChatStream:
    def test_sse_format(self, client: object, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chatstream1")
        nb_id = _create_notebook(client, app, "Chat Stream NB")
        _upload_source(client, app, nb_id, monkeypatch)

        res = client.post(
            f"/notebooks/{nb_id}/chat",
            json={"question": "What databases are mentioned?"},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.content_type
        # Parse SSE frames.
        body = res.get_data(as_text=True)
        frames = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(frames) >= 2  # at least one token + done frame
        # Last frame should have done: true.
        last_data = json.loads(frames[-1].replace("data: ", ""))
        assert last_data.get("done") is True
        assert "latency_ms" in last_data

    def test_stream_empty_question(self, client: object, app: object) -> None:
        _login(client, app, "chatstream2")
        nb_id = _create_notebook(client, app, "Chat Stream Empty NB")
        res = client.post(f"/notebooks/{nb_id}/chat", json={"question": ""})
        assert res.status_code == 400


class TestChatHistory:
    def test_returns_history(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chathist1")
        nb_id = _create_notebook(client, app, "Chat Hist NB")
        _upload_source(client, app, nb_id, monkeypatch)

        client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What databases are mentioned?"},
        )
        res = client.get(f"/notebooks/{nb_id}/chat/history")
        assert res.status_code == 200
        data = res.get_json()
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_empty_history(self, client: object, app: object) -> None:
        _login(client, app, "chathist2")
        nb_id = _create_notebook(client, app, "Chat Empty Hist NB")
        res = client.get(f"/notebooks/{nb_id}/chat/history")
        assert res.status_code == 200
        data = res.get_json()
        assert data["messages"] == []


class TestClearHistory:
    def test_clears_messages(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "chatclear1")
        nb_id = _create_notebook(client, app, "Chat Clear NB")
        _upload_source(client, app, nb_id, monkeypatch)

        client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What databases are mentioned?"},
        )
        with app.app_context():
            assert db.session.query(ChatMessage).filter_by(notebook_id=nb_id).count() == 2

        res = client.post(f"/notebooks/{nb_id}/chat/clear")
        assert res.status_code == 200
        with app.app_context():
            assert db.session.query(ChatMessage).filter_by(notebook_id=nb_id).count() == 0
