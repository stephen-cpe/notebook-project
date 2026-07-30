"""Unit tests for src.services.voice_service.

All tests use mocked STT + chat + TTS (AI_MOCK=true, no network).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.extensions import db
from src.models import Notebook, User
from src.services.auth_service import hash_password
from src.services.exceptions import AudioTooLongError
from src.services.stt_service import SpeechToTextResult
from src.services.voice_service import VoiceService, get_voice_service, reset_voice_service


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_voice_service()
    yield
    reset_voice_service()


def _make_notebook(app: object, username: str = "voicesvc") -> int:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw123456"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Voice NB")
        db.session.add(nb)
        db.session.commit()
        return nb.id


class TestRunVoiceTurn:
    def test_full_mocked_pipeline(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "voicesvc1")

        stt = MagicMock()
        stt.transcribe.return_value = SpeechToTextResult(
            text="What databases are mentioned?", language="en", duration_ms=4200, provider="mock"
        )
        chat = MagicMock()
        chat.chat_sync.return_value = {
            "answer": "PostgreSQL and MongoDB.",
            "sources": [{"filename": "doc.pdf", "page": 1}],
            "latency_ms": 100,
        }
        svc = VoiceService(stt=stt, chat=chat)

        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            result = svc.run_voice_turn(nb, str(tmp_path / "in.webm"), "Ava")
        assert result.transcript == "What databases are mentioned?"
        assert "PostgreSQL" in result.answer
        assert result.error is None
        assert result.reply_audio_url is not None
        assert result.reply_audio_path is not None
        assert Path(result.reply_audio_path).exists()

    def test_empty_transcript_short_circuit(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        nb_id = _make_notebook(app, "voicesvc2")
        stt = MagicMock()
        stt.transcribe.return_value = SpeechToTextResult(
            text="   ", language="en", duration_ms=4200, provider="mock"
        )
        chat = MagicMock()
        svc = VoiceService(stt=stt, chat=chat)
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            result = svc.run_voice_turn(nb, str(tmp_path / "in.webm"), "Ava")
        assert result.error == "no_speech"
        assert result.transcript == ""
        chat.chat_sync.assert_not_called()

    def test_stt_failure_returns_error(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        nb_id = _make_notebook(app, "voicesvc3")
        stt = MagicMock()
        stt.transcribe.side_effect = AudioTooLongError("too long")
        chat = MagicMock()
        svc = VoiceService(stt=stt, chat=chat)
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            result = svc.run_voice_turn(nb, str(tmp_path / "in.webm"), "Ava")
        assert result.error == "too long"
        chat.chat_sync.assert_not_called()

    def test_llm_failure_returns_transcript_and_error(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        nb_id = _make_notebook(app, "voicesvc4")
        stt = MagicMock()
        stt.transcribe.return_value = SpeechToTextResult(
            text="hello", language="en", duration_ms=4200, provider="mock"
        )
        chat = MagicMock()
        chat.chat_sync.side_effect = RuntimeError("LLM down")
        svc = VoiceService(stt=stt, chat=chat)
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            result = svc.run_voice_turn(nb, str(tmp_path / "in.webm"), "Ava")
        assert result.transcript == "hello"
        assert result.error is not None
        assert "chat_failed" in result.error

    def test_tts_failure_still_returns_text(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "voicesvc5")
        stt = MagicMock()
        stt.transcribe.return_value = SpeechToTextResult(
            text="hi", language="en", duration_ms=4200, provider="mock"
        )
        chat = MagicMock()
        chat.chat_sync.return_value = {"answer": "text answer", "sources": [], "latency_ms": 1}
        svc = VoiceService(stt=stt, chat=chat)
        # Force TTS to fail.
        monkeypatch.setattr(
            "src.services.voice_service.synthesize_utterance",
            lambda *a, **k: False,
        )
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            result = svc.run_voice_turn(nb, str(tmp_path / "in.webm"), "Ava")
        assert result.answer == "text answer"
        assert result.error == "tts_failed"
        assert result.reply_audio_path is None


class TestVoiceSingleton:
    def test_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        a = get_voice_service()
        b = get_voice_service()
        assert a is b
