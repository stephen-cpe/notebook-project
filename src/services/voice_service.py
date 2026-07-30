"""Voice service -- orchestrates a single voice conversation turn.

Flow (``run_voice_turn``):
1. Transcribe the user's audio via ``STTService``.
2. If the transcript is empty, return an error result (no LLM call).
3. Reuse ``ChatService.chat_sync`` for RAG retrieval + LLM answer + persistence
   so voice turns are grounded in the notebook sources and saved to history.
4. Synthesize the answer to speech via the shared TTS helper.
5. Store the reply audio under ``DATA_DIR/voice/<notebook_id>/<uuid>.mp3``.

``run_voice_turn`` never raises: failures are captured in ``VoiceTurnResult.error``
so a TTS failure still returns the text answer + sources.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import Notebook
from src.services.chat_service import ChatService
from src.services.exceptions import AudioTooLongError, SpeechToTextError
from src.services.stt_service import STTService, get_stt_service
from src.services.tts_utils import clean_for_tts, speaker_to_voice, synthesize_utterance

logger = logging.getLogger(__name__)


@dataclass
class VoiceTurnResult:
    """Outcome of a single voice turn."""

    transcript: str
    answer: str
    sources: list[dict[str, Any]]
    reply_audio_path: str | None
    reply_audio_url: str | None
    latency_ms: int
    error: str | None = None


class VoiceService:
    """Orchestrates the STT -> chat -> TTS voice turn pipeline."""

    def __init__(
        self,
        config: Any | None = None,  # noqa: ANN401
        stt: STTService | None = None,
        chat: ChatService | None = None,
    ) -> None:
        if config is None:
            from src.config import Config

            config = Config()
        self._config = config
        self._data_dir: str = config.data_dir
        self._mock: bool = bool(config.ai_mock)
        self._stt = stt or get_stt_service()
        self._chat = chat or ChatService()

    def run_voice_turn(
        self,
        notebook: Notebook,
        audio_path: str,
        speaker: str,
    ) -> VoiceTurnResult:
        """Run one voice turn. Never raises."""
        start = time.time()
        error: str | None = None
        transcript = ""
        answer = ""
        sources: list[dict[str, Any]] = []
        reply_path: str | None = None
        reply_url: str | None = None

        # 1. Transcribe.
        try:
            stt_result = self._stt.transcribe(audio_path)
            transcript = stt_result.text.strip()
            logger.info(
                "voice turn: notebook=%d stt_provider=%s transcript_len=%d",
                notebook.id,
                stt_result.provider,
                len(transcript),
            )
        except (AudioTooLongError, SpeechToTextError) as exc:
            logger.warning("voice turn STT failed: %s", exc)
            return VoiceTurnResult(
                transcript="",
                answer="",
                sources=[],
                reply_audio_path=None,
                reply_audio_url=None,
                latency_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("voice turn STT unexpected error: %s", exc)
            return VoiceTurnResult(
                transcript="",
                answer="",
                sources=[],
                reply_audio_path=None,
                reply_audio_url=None,
                latency_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )

        # 2. Empty transcript -> no LLM call.
        if not transcript:
            return VoiceTurnResult(
                transcript="",
                answer="",
                sources=[],
                reply_audio_path=None,
                reply_audio_url=None,
                latency_ms=int((time.time() - start) * 1000),
                error="no_speech",
            )

        # 3. Chat (RAG + LLM + persistence).
        try:
            chat_result = self._chat.chat_sync(notebook, transcript)
            answer = chat_result.get("answer", "")
            sources = chat_result.get("sources", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.exception("voice turn chat failed: %s", exc)
            return VoiceTurnResult(
                transcript=transcript,
                answer="",
                sources=[],
                reply_audio_path=None,
                reply_audio_url=None,
                latency_ms=int((time.time() - start) * 1000),
                error=f"chat_failed: {exc}",
            )

        # 4. TTS — synthesize the answer.
        if answer:
            try:
                voice = speaker_to_voice(speaker)
                voice_dir = Path(self._data_dir) / "voice" / str(notebook.id)
                voice_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{uuid.uuid4().hex}.mp3"
                reply_path = str(voice_dir / fname)
                # Clean the answer for natural speech: strip markdown markers
                # and citation brackets so the speaker doesn't read "asterisk
                # asterisk" or "Source 3" verbatim. The chat UI keeps the
                # original formatted answer.
                spoken_text = clean_for_tts(answer)
                ok = synthesize_utterance(spoken_text, voice, reply_path, mock=self._mock)
                if ok:
                    reply_url = f"/notebooks/{notebook.id}/voice/reply/{fname}"
                else:
                    reply_path = None
                    error = "tts_failed"
            except Exception as exc:  # noqa: BLE001
                logger.exception("voice turn TTS failed: %s", exc)
                reply_path = None
                error = f"tts_failed: {exc}"

        return VoiceTurnResult(
            transcript=transcript,
            answer=answer,
            sources=sources,
            reply_audio_path=reply_path,
            reply_audio_url=reply_url,
            latency_ms=int((time.time() - start) * 1000),
            error=error,
        )


_service: VoiceService | None = None


def get_voice_service() -> VoiceService:
    """Return a process-wide ``VoiceService`` (created lazily)."""
    global _service
    if _service is None:
        _service = VoiceService()
    return _service


def reset_voice_service() -> None:
    """Reset the cached service (used by tests that change config)."""
    global _service
    _service = None
