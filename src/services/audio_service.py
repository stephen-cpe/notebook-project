"""Audio service — two-host TTS synthesis + MP3 concatenation.

Generates a two-host Audio Overview by:
1. Writing a dialogue script (via ``audio_scripter``).
2. Synthesizing each utterance with edge-TTS (Host A -> voice_a, Host B -> voice_b).
3. Concatenating per-utterance audio files into a single MP3.

Per-utterance failures are isolated: a failed utterance is skipped with a
brief silence; the overall audio still completes if at least one utterance
succeeds (FR-74). Generation is idempotent per notebook version (FR-75).

In mock mode, a stub MP3 file is written without real TTS calls.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.extensions import db
from src.models import Notebook
from src.repositories import notebook_repo
from src.services.audio_scripter import write_dialogue
from src.services.tts_utils import speaker_to_voice, synthesize_utterance

logger = logging.getLogger(__name__)

AUDIO_STATUS_NONE = "none"
AUDIO_STATUS_QUEUED = "queued"
AUDIO_STATUS_SCRIPTING = "scripting"
AUDIO_STATUS_SYNTHESIZING = "synthesizing"
AUDIO_STATUS_READY = "ready"
AUDIO_STATUS_FAILED = "failed"


@dataclass
class AudioResult:
    """Outcome of audio generation."""

    status: str
    audio_path: str | None
    error: str | None = None


class AudioService:
    """Two-host TTS synthesis + MP3 concatenation with mock support."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            config = Config()
        self._config = config
        self._data_dir: str = config.data_dir
        self._mock: bool = bool(config.ai_mock)

    def generate_audio(
        self,
        notebook: Notebook,
        dialogue: list[dict[str, str]],
        speaker_a: str = "Ava",
        speaker_b: str = "Andrew",
    ) -> AudioResult:
        """Synthesize + concatenate the dialogue into a single MP3.

        Returns ``AudioResult`` with status ready/failed and the file path.
        """
        if not dialogue:
            self._set_status(notebook, AUDIO_STATUS_FAILED)
            return AudioResult(
                status=AUDIO_STATUS_FAILED, audio_path=None, error="No dialogue to synthesize."
            )

        self._set_status(notebook, AUDIO_STATUS_SYNTHESIZING)

        voice_a = speaker_to_voice(speaker_a)
        voice_b = speaker_to_voice(speaker_b)

        # Prepare output directory.
        audio_dir = Path(self._data_dir) / "audio" / str(notebook.id)
        audio_dir.mkdir(parents=True, exist_ok=True)
        sig = hashlib.sha256("|".join(u["text"] for u in dialogue).encode()).hexdigest()[:12]
        output_path = str(audio_dir / f"{sig}.mp3")

        # Synthesize each utterance.
        temp_dir = audio_dir / "tmp"
        temp_dir.mkdir(exist_ok=True)
        temp_files: list[str] = []
        success_count = 0

        for i, utterance in enumerate(dialogue):
            host = utterance["host"]
            text = utterance["text"]
            voice = voice_a if host == "A" else voice_b
            temp_path = str(temp_dir / f"utterance_{i:04d}.mp3")

            ok = synthesize_utterance(text, voice, temp_path, mock=self._mock)
            if ok:
                temp_files.append(temp_path)
                success_count += 1
            else:
                logger.warning("Utterance %d failed, skipping", i)

        if success_count == 0:
            self._set_status(notebook, AUDIO_STATUS_FAILED)
            return AudioResult(
                status=AUDIO_STATUS_FAILED,
                audio_path=None,
                error="All utterances failed to synthesize.",
            )

        # Concatenate into single MP3.
        self._concatenate_audio(temp_files, output_path)

        # Clean up temp files.
        for f in temp_files:
            Path(f).unlink(missing_ok=True)

        # Persist to notebook.
        notebook.audio_path = output_path
        notebook.audio_status = AUDIO_STATUS_READY
        db.session.commit()

        logger.info(
            "Audio generated for notebook %d: %d/%d utterances, file=%s",
            notebook.id,
            success_count,
            len(dialogue),
            output_path,
        )
        return AudioResult(status=AUDIO_STATUS_READY, audio_path=output_path)

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def _concatenate_audio(self, temp_files: list[str], output_path: str) -> None:
        """Concatenate MP3 files into one. Uses pydub if available, else raw."""
        if self._mock:
            # In mock mode, just copy the first file (they're all stubs).
            Path(output_path).write_bytes(Path(temp_files[0]).read_bytes())
            return

        try:
            from pydub import AudioSegment

            combined = AudioSegment.empty()
            for f in temp_files:
                segment = AudioSegment.from_file(f, format="mp3")
                combined += segment
            combined.export(output_path, format="mp3")
        except ImportError:
            # Fallback: raw byte concatenation (less ideal but works for MP3).
            with open(output_path, "wb") as out:
                for f in temp_files:
                    out.write(Path(f).read_bytes())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, notebook: Notebook, status: str) -> None:
        """Update the notebook's audio_status."""
        notebook.audio_status = status
        db.session.commit()


# ---------------------------------------------------------------------------
# End-to-end function (used by background job)
# ---------------------------------------------------------------------------


def generate_audio_for_notebook(
    notebook_id: int, topic: str = "", speaker_a: str = "Ava", speaker_b: str = "Andrew"
) -> AudioResult | None:
    """Full pipeline: script -> synthesize -> concatenate -> persist.

    Returns ``AudioResult`` or ``None`` on failure.
    """
    notebook = notebook_repo.get_by_id(notebook_id)
    if notebook is None:
        logger.error("Notebook %d not found for audio generation", notebook_id)
        return None

    # Set status to scripting.
    notebook.audio_status = AUDIO_STATUS_SCRIPTING
    db.session.commit()
    logger.info("Audio generation: scripting dialogue for notebook %d", notebook_id)

    # Generate dialogue.
    dialogue = write_dialogue(notebook, topic=topic)
    if not dialogue:
        logger.error("Audio generation: no dialogue produced for notebook %d", notebook_id)
        notebook.audio_status = AUDIO_STATUS_FAILED
        db.session.commit()
        return AudioResult(
            status=AUDIO_STATUS_FAILED,
            audio_path=None,
            error="No dialogue could be generated.",
        )

    logger.info(
        "Audio generation: got %d utterances for notebook %d, synthesizing...",
        len(dialogue),
        notebook_id,
    )

    # Synthesize.
    svc = AudioService()
    result = svc.generate_audio(notebook, dialogue, speaker_a=speaker_a, speaker_b=speaker_b)
    logger.info(
        "Audio generation result: notebook=%d status=%s path=%s",
        notebook_id,
        result.status,
        result.audio_path,
    )
    return result
