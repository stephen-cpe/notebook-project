"""Speech-to-text service.

Transcribes recorded audio to text using either:
- a deterministic mock backend (``AI_MOCK=true`` — offline, no model download); or
- ``faster-whisper`` local backend (lazy-loaded; downloads a model on first use).

Audio normalization: webm/ogg/mp4/wav inputs are decoded to 16 kHz mono 16-bit
PCM via pydub (which requires ffmpeg on PATH). Inputs longer than
``VOICE_MAX_RECORDING_SECONDS`` or larger than ``VOICE_MAX_UPLOAD_MB`` are
rejected with typed errors.

Lazy singleton: ``get_stt_service()`` / ``reset_stt_service()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.services.exceptions import (
    AudioTooLongError,
    SpeechToTextError,
    UnsupportedAudioFormatError,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config


@dataclass
class SpeechToTextResult:
    """Outcome of a transcription."""

    text: str
    language: str | None
    duration_ms: int | None
    provider: str


class _SttBackend(Protocol):
    """Internal contract for STT backends."""

    def transcribe(self, pcm_path: str) -> SpeechToTextResult: ...


class STTService:
    """Speech-to-text with mock support + lazy faster-whisper backend."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self._mock: bool = bool(config.ai_mock)
        self._provider: str = config.voice_stt_provider
        self._max_seconds: int = config.voice_max_recording_seconds
        self._max_upload_mb: int = config.voice_max_upload_mb
        self._backend: _SttBackend | None = None

    @property
    def is_mock(self) -> bool:
        """True when the service is using the deterministic mock backend."""
        return self._mock

    def transcribe(self, audio_path: str) -> SpeechToTextResult:
        """Transcribe an audio file to text.

        Normalizes the input to 16 kHz mono PCM, enforces duration/size limits,
        then delegates to the configured backend. Raises typed errors for
        too-long / unsupported audio; never silently returns garbage.
        """
        # Size guard.
        size = Path(audio_path).stat().st_size
        if size > self._max_upload_mb * 1024 * 1024:
            raise AudioTooLongError(f"Audio upload exceeds {self._max_upload_mb} MB limit.")

        pcm_path = self._normalize_audio(audio_path)
        try:
            duration_ms = self._estimate_duration_ms(pcm_path)
            if duration_ms is not None and duration_ms > self._max_seconds * 1000:
                raise AudioTooLongError(f"Audio exceeds {self._max_seconds}s recording limit.")
            if self._mock:
                return _MockSttBackend().transcribe(pcm_path)
            if self._backend is None:
                self._backend = self._make_backend()
            return self._backend.transcribe(pcm_path)
        finally:
            # Clean up the normalized PCM temp file (not the original).
            if pcm_path != audio_path:
                with contextlib.suppress(Exception):
                    Path(pcm_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_audio(self, audio_path: str) -> str:
        """Decode any supported format to 16 kHz mono 16-bit WAV via pydub.

        If the input is already a WAV file it is passed through unchanged
        (WAV is already PCM, so pydub conversion is unnecessary and avoids an
        ffmpeg dependency for the common browser-recording case after the
        browser encodes to WAV). For non-WAV inputs (webm/ogg/mp4), pydub
        (which requires ffmpeg on PATH) decodes and resamples.
        Raises ``UnsupportedAudioFormatError`` when decoding fails.
        """
        # WAV passthrough: no ffmpeg/pydub conversion needed.
        if audio_path.lower().endswith(".wav"):
            return audio_path

        try:
            from pydub import AudioSegment  # noqa: F401
        except ImportError:
            raise UnsupportedAudioFormatError(
                "pydub is not installed; cannot decode non-WAV audio."
            ) from None

        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not on PATH; STT normalization may fail for non-WAV inputs.")

        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedAudioFormatError(
                f"Could not decode audio ({Path(audio_path).suffix}): {exc}"
            ) from exc

        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        fd, tmp_name = tempfile.mkstemp(suffix="_stt.wav", dir=str(Path(audio_path).parent))
        os.close(fd)
        audio.export(tmp_name, format="wav")
        return tmp_name

    @staticmethod
    def _estimate_duration_ms(pcm_path: str) -> int | None:
        """Best-effort duration estimate from the WAV header (pydub)."""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(pcm_path, format="wav")
            return len(audio)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _make_backend(self) -> _SttBackend:
        provider = self._provider
        if provider == "local":
            return _FasterWhisperSttBackend(
                model=self._config.voice_stt_model,
                device=self._config.voice_stt_device,
                compute_type=self._config.voice_stt_compute_type,
                language=self._config.voice_stt_language or None,
            )
        raise ValueError(
            f"Unknown VOICE_STT_PROVIDER={provider!r}. Expected 'local' (faster-whisper)."
        )


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


class _MockSttBackend:
    """Deterministic mock: returns text derived from the file bytes (offline)."""

    def transcribe(self, pcm_path: str) -> SpeechToTextResult:
        data = Path(pcm_path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:8]
        return SpeechToTextResult(
            text=f"[mock transcription {digest}] What databases are mentioned in the sources?",
            language="en",
            duration_ms=4200,
            provider="mock",
        )


class _FasterWhisperSttBackend:
    """Local faster-whisper backend. Lazy-loads the model (heavy)."""

    def __init__(
        self,
        model: str,
        device: str,
        compute_type: str,
        language: str | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: Any = None

    def _load_model(self) -> Any:  # noqa: ANN401
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model %s (device=%s, compute_type=%s)",
            self._model_name,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )
        return self._model

    def transcribe(self, pcm_path: str) -> SpeechToTextResult:
        model = self._load_model()
        try:
            segments, info = model.transcribe(pcm_path, language=self._language)
            text = " ".join(s.text for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            raise SpeechToTextError(f"faster-whisper transcription failed: {exc}") from exc
        return SpeechToTextResult(
            text=text,
            language=getattr(info, "language", None),
            duration_ms=int(getattr(info, "duration", 0) * 1000) or None,
            provider="faster-whisper",
        )


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_service: STTService | None = None


def get_stt_service() -> STTService:
    """Return a process-wide ``STTService`` (created lazily)."""
    global _service
    if _service is None:
        _service = STTService()
    return _service


def reset_stt_service() -> None:
    """Reset the cached service (used by tests that change config)."""
    global _service
    _service = None
