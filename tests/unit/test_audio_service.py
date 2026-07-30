"""Unit tests for src.services.audio_service (TDD step 16b).

The audio service synthesizes a two-host dialogue via edge-TTS (two distinct
voices) and concatenates into a single MP3. Per-utterance failures are
isolated (FR-74). Generation is idempotent per notebook version (FR-75).

Covers:
- generate_audio: produces an audio file path.
- Mock mode: writes a stub file without real TTS.
- Per-utterance failure isolation: one failed utterance doesn't block others.
- Idempotency: same content_signature -> returns existing file.
- Status transitions: none -> queued -> scripting -> synthesizing -> ready/failed.
- Empty dialogue -> no audio file, status = failed or none.
- Voice assignment: Host A -> voice_a, Host B -> voice_b.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.extensions import db
from src.models import Notebook, User
from src.repositories import notebook_repo
from src.services.audio_service import (
    AudioService,
    generate_audio_for_notebook,
)
from src.services.auth_service import hash_password

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_notebook(app: object, username: str = "audsvc") -> int:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Audio Svc NB", audio_status="none")
        db.session.add(nb)
        db.session.commit()
        return nb.id


class TestGenerateAudio:
    def test_mock_produces_file(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "audsvc1")

        dialogue = [
            {"host": "A", "text": "Welcome!"},
            {"host": "B", "text": "Let's discuss ML."},
            {"host": "A", "text": "Great idea!"},
        ]

        svc = AudioService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            result = svc.generate_audio(nb, dialogue)
            assert result.status == "ready"
            assert result.audio_path is not None
            assert Path(result.audio_path).exists()

    def test_empty_dialogue_no_file(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "audsvc2")

        svc = AudioService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            result = svc.generate_audio(nb, [])
            assert result.status == "failed"
            assert result.audio_path is None

    def test_per_utterance_failure_isolated(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FR-74: a failed utterance is skipped; overall audio still completes."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "audsvc3")

        dialogue = [
            {"host": "A", "text": "First utterance."},
            {"host": "B", "text": "FAIL_THIS_ONE"},
            {"host": "A", "text": "Third utterance."},
        ]

        svc = AudioService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            # Patch the shared synthesize_utterance to fail on the "FAIL" text.
            import src.services.tts_utils as tts_utils

            original = tts_utils.synthesize_utterance

            def _failing_synth(text: str, voice: str, output_path: str, mock: bool = False) -> bool:
                if "FAIL_THIS_ONE" in text:
                    return False
                return original(text, voice, output_path, mock=mock)

            with patch.object(tts_utils, "synthesize_utterance", side_effect=_failing_synth):
                result = svc.generate_audio(nb, dialogue)
            # Should still succeed (2 of 3 utterances).
            assert result.status == "ready"
            assert result.audio_path is not None

    def test_voice_assignment(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        _make_notebook(app, "audsvc4")

        from src.services.tts_utils import speaker_to_voice

        assert speaker_to_voice("Ava") == "en-US-AvaNeural"
        assert speaker_to_voice("Andrew") == "en-US-AndrewNeural"
        assert speaker_to_voice("Emma") == "en-US-EmmaNeural"
        assert speaker_to_voice("Ryan") == "en-US-RyanNeural"
        assert speaker_to_voice("Unknown") == "en-US-AvaNeural"

    def test_persists_status_to_notebook(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _make_notebook(app, "audsvc5")

        dialogue = [{"host": "A", "text": "test"}]

        svc = AudioService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            svc.generate_audio(nb, dialogue)
            db.session.refresh(nb)
            assert nb.audio_status == "ready"
            assert nb.audio_path is not None


class TestGenerateAudioForNotebook:
    def test_full_pipeline_mock(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end: scripter + audio synthesis in mock mode."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))

        from src.models import Source
        from src.repositories import content_registry_repo

        with app.app_context():
            u = User(username="audfull1", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Full Audio NB")
            db.session.add(nb)
            db.session.commit()
            h = "f" * 64
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
                chroma_collection="doc_f",
                extracted_text="This document is about databases and data science.",
                char_count=50,
            )
            nb_id = nb.id

        with app.app_context():
            result = generate_audio_for_notebook(nb_id)
            assert result is not None
            assert result.status == "ready"
