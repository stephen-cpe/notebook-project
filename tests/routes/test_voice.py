"""Route tests for voice conversation endpoints.

All tests run under AI_MOCK=true with VOICE_ENABLED=true so the blueprint is
registered; STT/TTS/chat are mocked (no real model download or network).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from src.extensions import db
from src.models import Notebook, User
from src.services.auth_service import hash_password


@pytest.fixture()
def voice_app(monkeypatch: pytest.MonkeyPatch) -> object:
    """Create an app with VOICE_ENABLED=true and mocked STT/voice services."""
    monkeypatch.setenv("AI_MOCK", "true")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("VOICE_ENABLED", "true")
    # Import after env is set so Config picks it up.
    import os

    os.environ["VOICE_ENABLED"] = "true"

    import tempfile

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    from src.app import create_app
    from src.config import Config
    from src.extensions import db as _db

    tmp_dir = tempfile.mkdtemp(prefix="nbvoicetest_")
    tmp_db = os.path.join(tmp_dir, "test.db")
    os.environ["TEST_DATABASE_URL"] = f"sqlite:///{tmp_db}"

    from src.services.vector_store import get_vector_store, reset_vector_store

    reset_vector_store()
    get_vector_store().reset()

    cfg = Config()
    application = create_app(cfg)

    @event.listens_for(Engine, "connect")
    def _enable_fk(dbapi_conn, _records):  # noqa: ANN001
        try:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        except Exception:  # noqa: BLE001
            pass

    with application.app_context():
        _db.create_all()

    # Patch the voice service to avoid real STT/chat/TTS.
    fake_result = MagicMock()
    fake_result.transcript = "What databases are mentioned?"
    fake_result.answer = "PostgreSQL and MongoDB."
    fake_result.sources = []
    fake_result.latency_ms = 100
    fake_result.reply_audio_path = None
    fake_result.reply_audio_url = None
    fake_result.error = None

    mock_svc = MagicMock()
    mock_svc.run_voice_turn.return_value = fake_result
    import src.services.voice_service as vs

    monkeypatch.setattr(vs, "get_voice_service", lambda: mock_svc)

    yield application

    with application.app_context():
        _db.session.remove()
        _db.drop_all()
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)


def _login(client: object, app: object, username: str = "voiceuser") -> int:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password("pw123456")))
            db.session.commit()
        nb = Notebook(
            user_id=db.session.query(User).filter_by(username=username).one().id, name="Voice NB"
        )
        db.session.add(nb)
        db.session.commit()
        nb_id = nb.id
    client.post("/login", data={"username": username, "password": "pw123456"})
    return nb_id


class TestVoiceTurn:
    def test_happy_path(self, voice_app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        client = voice_app.test_client()
        nb_id = _login(client, voice_app, "voice1")
        res = client.post(
            f"/notebooks/{nb_id}/voice/turn",
            data={"audio": (io.BytesIO(b"fake audio bytes"), "rec.webm")},
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["transcript"] == "What databases are mentioned?"
        assert "PostgreSQL" in data["answer"]
        assert "reply_audio_url" in data

    def test_login_required(self, voice_app: object) -> None:
        client = voice_app.test_client()
        res = client.post(
            "/notebooks/1/voice/turn",
            data={"audio": (io.BytesIO(b"x"), "r.webm")},
            content_type="multipart/form-data",
        )
        assert res.status_code in (301, 302, 303)

    def test_non_owner_404(self, voice_app: object) -> None:
        client = voice_app.test_client()
        _login(client, voice_app, "voice2")
        # Another user's notebook id (nonexistent).
        res = client.post(
            "/notebooks/99999/voice/turn",
            data={"audio": (io.BytesIO(b"x"), "r.webm")},
            content_type="multipart/form-data",
        )
        assert res.status_code == 404

    def test_missing_audio_400(self, voice_app: object) -> None:
        client = voice_app.test_client()
        nb_id = _login(client, voice_app, "voice3")
        res = client.post(f"/notebooks/{nb_id}/voice/turn", data={})
        assert res.status_code == 400

    def test_no_speech_returns_422(
        self, voice_app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.services.voice_service as vs

        fake = MagicMock()
        fake.error = "no_speech"
        fake.transcript = ""
        fake.answer = ""
        fake.sources = []
        fake.latency_ms = 1
        fake.reply_audio_path = None
        fake.reply_audio_url = None
        mock_svc = MagicMock()
        mock_svc.run_voice_turn.return_value = fake
        monkeypatch.setattr(vs, "get_voice_service", lambda: mock_svc)
        client = voice_app.test_client()
        nb_id = _login(client, voice_app, "voice4")
        res = client.post(
            f"/notebooks/{nb_id}/voice/turn",
            data={"audio": (io.BytesIO(b"x"), "r.webm")},
            content_type="multipart/form-data",
        )
        assert res.status_code == 422


class TestVoiceReply:
    def test_filename_traversal_rejected(self, voice_app: object) -> None:
        client = voice_app.test_client()
        nb_id = _login(client, voice_app, "voice5")
        res = client.get(f"/notebooks/{nb_id}/voice/reply/..%2F..%2Fetc%2Fpasswd")
        assert res.status_code in (400, 404)


class TestVoiceDisabled:
    def test_voice_turn_404_when_disabled(self, app: object) -> None:
        """With VOICE_ENABLED=false (default), the voice route is not registered."""
        client = app.test_client()
        res = client.post(
            "/notebooks/1/voice/turn",
            data={"audio": (io.BytesIO(b"x"), "r.webm")},
            content_type="multipart/form-data",
        )
        # Route not registered -> 404 (or login redirect). Both are acceptable
        # for "disabled". Assert it's not 200.
        assert res.status_code != 200


# ---------------------------------------------------------------------------
# SocketIO streaming
# ---------------------------------------------------------------------------


class TestVoiceSocketIO:
    """End-to-end SocketIO namespace tests using socketio.test_client.

    The voice service is mocked so no real STT/TTS/network runs.
    """

    def test_connect_requires_auth(self, voice_app: object) -> None:
        sio = voice_app.extensions["socketio"]
        assert sio is not None
        client = sio.test_client(voice_app, namespace="/voice")
        # Unauthenticated -> connection rejected.
        assert client.is_connected(namespace="/voice") is False
        # disconnect() raises if not connected; the rejection is the assertion.

    def test_connect_authenticated_then_start_stop(
        self, voice_app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sio = voice_app.extensions["socketio"]
        # Use the Flask test client to log in and seed a notebook (carries the
        # session cookie), then open a SocketIO test client reusing the cookie.
        flask_client = voice_app.test_client()
        nb_id = _login(flask_client, voice_app, "sio1")
        # flask_socketio.test_client reuses the Flask test client's session
        # cookie so the namespace's auth check (current_user.is_authenticated)
        # passes.
        sio_client = sio.test_client(voice_app, namespace="/voice", flask_test_client=flask_client)
        assert sio_client.is_connected(namespace="/voice") is True

        # voice:start with a valid (owned) notebook.
        sio_client.emit("voice:start", {"notebook_id": nb_id}, namespace="/voice")
        import time

        time.sleep(0.1)
        received = sio_client.get_received(namespace="/voice")
        states = [m for m in received if m["name"] == "voice:status"]
        assert any(m["args"][0]["state"] == "ready" for m in states), f"received={received}"

        # Audio is sent via HTTP POST /voice/turn, not via SocketIO binary.
        # voice:audio_chunk returns an error (binary not accepted).
        sio_client.emit("voice:audio_chunk", b"fake-audio-bytes", namespace="/voice")
        time.sleep(0.1)
        received = sio_client.get_received(namespace="/voice")
        assert any(m["name"] == "voice:error" for m in received), f"received={received}"

        # voice:stop emits a "sending" status (audio goes via HTTP).
        sio_client.emit("voice:stop", namespace="/voice")
        time.sleep(0.1)
        received = sio_client.get_received(namespace="/voice")
        states = [m for m in received if m["name"] == "voice:status"]
        assert any(m["args"][0]["state"] == "sending" for m in states), f"received={received}"

    def test_start_invalid_notebook_emits_error(self, voice_app: object) -> None:
        sio = voice_app.extensions["socketio"]
        flask_client = voice_app.test_client()
        _login(flask_client, voice_app, "sio2")
        sio_client = sio.test_client(voice_app, namespace="/voice", flask_test_client=flask_client)
        assert sio_client.is_connected(namespace="/voice") is True
        # Non-owned/nonexistent notebook id -> require_owner aborts (404) -> error.
        sio_client.emit("voice:start", {"notebook_id": 999999}, namespace="/voice")
        import time

        time.sleep(0.1)
        received = sio_client.get_received(namespace="/voice")
        names = [m["name"] for m in received]
        assert "voice:error" in names, f"received={received}"

    def test_cancel_emits_done_cancelled(self, voice_app: object) -> None:
        sio = voice_app.extensions["socketio"]
        flask_client = voice_app.test_client()
        nb_id = _login(flask_client, voice_app, "sio3")
        sio_client = sio.test_client(voice_app, namespace="/voice", flask_test_client=flask_client)
        assert sio_client.is_connected(namespace="/voice") is True
        sio_client.emit("voice:start", {"notebook_id": nb_id}, namespace="/voice")
        import time

        time.sleep(0.1)
        sio_client.get_received(namespace="/voice")
        sio_client.emit("voice:cancel", namespace="/voice")
        time.sleep(0.1)
        received = sio_client.get_received(namespace="/voice")
        done = [m for m in received if m["name"] == "voice:done"]
        assert done and done[0]["args"][0]["state"] == "cancelled", f"received={received}"
