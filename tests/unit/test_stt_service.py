"""Unit tests for src.services.stt_service.

All tests run under AI_MOCK=true with no real Whisper model download.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.services.exceptions import AudioTooLongError, UnsupportedAudioFormatError
from src.services.stt_service import STTService, get_stt_service, reset_stt_service


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_stt_service()
    yield
    reset_stt_service()


def _make_wav(path: Path) -> str:
    """Create a tiny valid WAV file (mock STT reads bytes, doesn't decode)."""
    # Minimal RIFF/WAVE header + a few zero samples.
    header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x80\xbb\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    )
    path.write_bytes(header)
    return str(path)


class TestSttMock:
    def test_is_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        svc = STTService()
        assert svc.is_mock is True

    def test_mock_transcribe_returns_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("VOICE_MAX_UPLOAD_MB", "10")
        svc = STTService()
        wav = _make_wav(tmp_path / "in.wav")
        result = svc.transcribe(wav)
        assert result.text.startswith("[mock transcription")
        assert result.provider == "mock"
        assert result.language == "en"


class TestSttLimits:
    def test_oversized_upload_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("VOICE_MAX_UPLOAD_MB", "1")
        svc = STTService()
        big = tmp_path / "big.wav"
        big.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB > 1 MB limit
        with pytest.raises(AudioTooLongError):
            svc.transcribe(str(big))

    def test_unsupported_format_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        svc = STTService()
        # pydub is installed in dev; feed it garbage so AudioSegment.from_file
        # raises, which the service translates to UnsupportedAudioFormatError.
        bad = tmp_path / "bad.ogg"
        bad.write_bytes(b"definitely not real audio bytes")
        with pytest.raises(UnsupportedAudioFormatError):
            svc.transcribe(str(bad))


class TestSttSingleton:
    def test_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CI", "true")
        a = get_stt_service()
        b = get_stt_service()
        assert a is b


class TestFasterWhisperBackend:
    def test_transcribe_via_mocked_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The faster-whisper backend transcribes via the lazy-loaded model."""
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("VOICE_STT_PROVIDER", "local")
        svc = STTService()
        assert svc.is_mock is False

        # Patch _make_backend to return a backend whose model transcribes.
        from src.services.stt_service import _FasterWhisperSttBackend

        class _FakeSegment:
            text = "hello world"

        class _FakeInfo:
            language = "en"
            duration = 4.2

        backend = _FasterWhisperSttBackend(model="base.en", device="cpu", compute_type="int8")
        backend._model = MagicMock()
        backend._model.transcribe.return_value = (iter([_FakeSegment()]), _FakeInfo())
        svc._backend = backend

        wav = _make_wav(tmp_path / "in.wav")
        result = svc.transcribe(wav)
        assert result.text == "hello world"
        assert result.provider == "faster-whisper"
        assert result.language == "en"
