"""Unit tests for src.realtime (SocketIO voice namespace).

These tests exercise the ``/voice`` namespace handlers without a real
SocketIO server: ``flask_socketio.emit`` and ``flask.request`` are patched so
the handler logic runs offline. If ``flask_socketio`` is not installed (e.g.
on machines without the full requirements), a minimal fake module is injected
into ``sys.modules`` so the handlers' lazy imports still resolve.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.extensions import db
from src.models import Notebook, User
from src.realtime import (
    VoiceNamespace,
    _run_turn,
    register_voice_namespace,
)
from src.services.auth_service import hash_password


def _inject_fake_socketio() -> None:
    """Provide a minimal ``flask_socketio`` module only when not installed.

    When the real package is installed (CI installs requirements.txt), it is
    left untouched so other tests (e.g. ``tests/routes/test_voice.py``) can
    use the real SocketIO test client.
    """
    if "flask_socketio" in sys.modules:
        return
    try:
        import flask_socketio  # noqa: F401

        return  # real module available; nothing to inject
    except ImportError:
        pass

    fake = types.ModuleType("flask_socketio")

    class _FakeNamespace:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    fake.emit = MagicMock()
    fake.Namespace = _FakeNamespace
    sys.modules["flask_socketio"] = fake


_inject_fake_socketio()


def _make_user_and_notebook(app: object, username: str = "rtuser") -> tuple[int, int]:
    """Create a user + notebook, returning (user_id, notebook_id)."""
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Realtime NB")
        db.session.add(nb)
        db.session.commit()
        return u.id, nb.id


def _fake_request(sid: str = "sid-123") -> MagicMock:
    req = MagicMock()
    req.sid = sid
    return req


def _fake_user(user_id: int, authenticated: bool = True) -> MagicMock:
    u = MagicMock()
    u.is_authenticated = authenticated
    u.username = "rtuser"
    u.get_id.return_value = str(user_id)
    return u


# require_owner (src/routes/_helpers) binds `current_user` at import time, so
# that module's attribute must be patched (not flask_login's).
def _patch_helpers_user(user_id: int) -> MagicMock:
    fake = _fake_user(user_id)
    fake.is_admin = False
    return patch("src.routes._helpers.current_user", fake)


class TestVoiceNamespaceConnect:
    def test_authenticated_returns_true(self, app: object) -> None:
        user_id, _ = _make_user_and_notebook(app)
        ns = VoiceNamespace()
        with (
            patch("flask.request", _fake_request()),
            patch("flask_login.current_user", _fake_user(user_id)),
        ):
            assert ns.on_connect() is True

    def test_unauthenticated_returns_false(self, app: object) -> None:
        ns = VoiceNamespace()
        with patch("flask_login.current_user", _fake_user(1, authenticated=False)):
            assert ns.on_connect() is False


class TestVoiceNamespaceDisconnect:
    def test_cleans_up_session(self, app: object) -> None:
        import src.realtime as rt

        user_id, nb_id = _make_user_and_notebook(app)
        ns = VoiceNamespace()
        with (
            patch("flask.request", _fake_request()),
            patch("flask_login.current_user", _fake_user(user_id)),
        ):
            rt._SESSIONS["sid-123"] = {"notebook_id": nb_id, "user_id": user_id}
            ns.on_disconnect()
            assert "sid-123" not in rt._SESSIONS


class TestVoiceNamespaceStart:
    def test_missing_notebook_id_emits_error(self, app: object) -> None:
        ns = VoiceNamespace()
        emit = MagicMock()
        with (
            patch("flask.request", _fake_request()),
            patch("flask_socketio.emit", emit),
        ):
            ns.on_voice_start({})
        emit.assert_called_once_with("voice:error", {"error": "notebook_id required"})

    def test_not_owner_emits_error(self, app: object) -> None:
        user_id, _ = _make_user_and_notebook(app, "rtuser1")
        _make_user_and_notebook(app, "rtuser2")  # other user's notebook
        ns = VoiceNamespace()
        emit = MagicMock()
        with (
            patch("flask.request", _fake_request()),
            _patch_helpers_user(user_id),
            patch("flask_socketio.emit", emit),
            app.app_context(),
        ):
            ns.on_voice_start({"notebook_id": 99999})
        emit.assert_called_once_with("voice:error", {"error": "not owner"})

    def test_owner_creates_session(self, app: object) -> None:
        import src.realtime as rt

        user_id, nb_id = _make_user_and_notebook(app)
        ns = VoiceNamespace()
        emit = MagicMock()
        with (
            patch("flask.request", _fake_request("sid-owner")),
            patch("flask_login.current_user", _fake_user(user_id)),
            _patch_helpers_user(user_id),
            patch("flask_socketio.emit", emit),
            app.app_context(),
        ):
            ns.on_voice_start({"notebook_id": nb_id})

        assert "sid-owner" in rt._SESSIONS
        assert rt._SESSIONS["sid-owner"]["notebook_id"] == nb_id
        assert rt._SESSIONS["sid-owner"]["user_id"] == user_id
        assert rt._SESSIONS["sid-owner"]["cancelled"] is False
        emit.assert_any_call("voice:status", {"state": "ready"})


class TestVoiceNamespaceAudio:
    def test_audio_chunk_emits_http_hint(self, app: object) -> None:
        ns = VoiceNamespace()
        emit = MagicMock()
        with patch("flask_socketio.emit", emit):
            ns.on_voice_audio_chunk(b"data")
        emit.assert_called_once_with(
            "voice:error", {"error": "audio must be sent via HTTP POST /voice/turn"}
        )

    def test_stop_emits_sending_status(self, app: object) -> None:
        ns = VoiceNamespace()
        emit = MagicMock()
        with patch("flask_socketio.emit", emit):
            ns.on_voice_stop()
        emit.assert_called_once_with("voice:status", {"state": "sending"})

    def test_cancel_marks_cancelled(self, app: object) -> None:
        import src.realtime as rt

        user_id, nb_id = _make_user_and_notebook(app)
        rt._SESSIONS["sid-cancel"] = {"notebook_id": nb_id, "user_id": user_id}
        ns = VoiceNamespace()
        emit = MagicMock()
        with (
            patch("flask.request", _fake_request("sid-cancel")),
            patch("flask_socketio.emit", emit),
        ):
            ns.on_voice_cancel()

        assert "sid-cancel" not in rt._SESSIONS
        emit.assert_called_once_with("voice:done", {"state": "cancelled"})


class TestRunTurn:
    def test_cancelled_emits_done(self, app: object) -> None:
        user_id, nb_id = _make_user_and_notebook(app)
        emit = MagicMock()
        with (
            patch("flask_socketio.emit", emit),
            app.app_context(),
        ):
            _run_turn(
                app,
                "sid-x",
                {
                    "notebook_id": nb_id,
                    "user_id": user_id,
                    "cancelled": True,
                    "buffer": bytearray(b"abc"),
                },
            )
        emit.assert_any_call("voice:done", {"state": "cancelled"}, namespace="/voice", to="sid-x")

    def test_empty_buffer_emits_error(self, app: object) -> None:
        user_id, nb_id = _make_user_and_notebook(app)
        emit = MagicMock()
        with (
            patch("flask_socketio.emit", emit),
            app.app_context(),
        ):
            _run_turn(
                app,
                "sid-x",
                {
                    "notebook_id": nb_id,
                    "user_id": user_id,
                    "cancelled": False,
                    "buffer": bytearray(),
                },
            )
        emit.assert_any_call(
            "voice:error", {"error": "no audio received"}, namespace="/voice", to="sid-x"
        )

    def test_happy_path_streams_audio(self, app: object, tmp_path: Path) -> None:
        user_id, nb_id = _make_user_and_notebook(app)
        emit = MagicMock()

        # A fake reply MP3 on disk.
        reply_path = tmp_path / "reply.mp3"
        reply_path.write_bytes(b"fake-mp3-bytes")

        fake_result = SimpleNamespace(
            error=None,
            transcript="hello world",
            answer="The answer is in the sources.",
            sources=[{"filename": "doc.txt", "page": 1}],
            reply_audio_path=str(reply_path),
        )
        fake_service = MagicMock()
        fake_service.run_voice_turn.return_value = fake_result

        with (
            patch("flask_socketio.emit", emit),
            patch("src.services.voice_service.get_voice_service", return_value=fake_service),
            app.app_context(),
        ):
            _run_turn(
                app,
                "sid-happy",
                {
                    "notebook_id": nb_id,
                    "user_id": user_id,
                    "cancelled": False,
                    "buffer": bytearray(b"audio"),
                },
            )

        # Transcribe -> answer -> sources -> audio chunks -> done.
        emit.assert_any_call(
            "voice:status", {"state": "transcribing"}, namespace="/voice", to="sid-happy"
        )
        emit.assert_any_call(
            "voice:transcript",
            {"text": "hello world", "final": True},
            namespace="/voice",
            to="sid-happy",
        )
        emit.assert_any_call(
            "voice:answer",
            {
                "text": "The answer is in the sources.",
                "sources": [{"filename": "doc.txt", "page": 1}],
            },
            namespace="/voice",
            to="sid-happy",
        )
        emit.assert_any_call(
            "voice:status", {"state": "speaking"}, namespace="/voice", to="sid-happy"
        )
        emit.assert_any_call("voice:status", {"state": "done"}, namespace="/voice", to="sid-happy")
        emit.assert_any_call("voice:done", {"state": "done"}, namespace="/voice", to="sid-happy")
        # Audio bytes were emitted as binary events.
        assert any(call.args[0] == "voice:audio_chunk" for call in emit.call_args_list)

    def test_no_speech_emits_error(self, app: object) -> None:
        user_id, nb_id = _make_user_and_notebook(app)
        emit = MagicMock()

        fake_result = SimpleNamespace(
            error="no_speech", transcript="", answer="", sources=[], reply_audio_path=None
        )
        fake_service = MagicMock()
        fake_service.run_voice_turn.return_value = fake_result

        with (
            patch("flask_socketio.emit", emit),
            patch("src.services.voice_service.get_voice_service", return_value=fake_service),
            app.app_context(),
        ):
            _run_turn(
                app,
                "sid-x",
                {
                    "notebook_id": nb_id,
                    "user_id": user_id,
                    "cancelled": False,
                    "buffer": bytearray(b"audio"),
                },
            )
        emit.assert_any_call("voice:error", {"error": "no_speech"}, namespace="/voice", to="sid-x")

    def test_generic_error_emits_error(self, app: object) -> None:
        user_id, nb_id = _make_user_and_notebook(app)
        emit = MagicMock()

        fake_result = SimpleNamespace(
            error="tts_failed", transcript="t", answer="a", sources=[], reply_audio_path=None
        )
        fake_service = MagicMock()
        fake_service.run_voice_turn.return_value = fake_result

        with (
            patch("flask_socketio.emit", emit),
            patch("src.services.voice_service.get_voice_service", return_value=fake_service),
            app.app_context(),
        ):
            _run_turn(
                app,
                "sid-x",
                {
                    "notebook_id": nb_id,
                    "user_id": user_id,
                    "cancelled": False,
                    "buffer": bytearray(b"audio"),
                },
            )
        emit.assert_any_call("voice:error", {"error": "tts_failed"}, namespace="/voice", to="sid-x")


class TestRegisterVoiceNamespace:
    def test_registers_namespace(self) -> None:
        socketio = MagicMock()
        register_voice_namespace(socketio)
        socketio.on_namespace.assert_called_once()

    def test_handles_missing_flask_socketio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        # Temporarily remove flask_socketio (real or fake) so the import fails.
        monkeypatch.delitem(sys.modules, "flask_socketio", raising=False)
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "flask_socketio":
                raise ImportError("flask_socketio not installed")
            return real_import(name, *args, **kwargs)

        socketio = MagicMock()
        monkeypatch.setattr(builtins, "__import__", _fake_import)
        register_voice_namespace(socketio)
        socketio.on_namespace.assert_not_called()
