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
from src.models import (
    AUDIO_STATUS_FAILED,
    AUDIO_STATUS_NONE,
    AUDIO_STATUS_READY,
    AUDIO_STATUS_SCRIPTING,
    AUDIO_STATUS_SYNTHESIZING,
    Notebook,
)
from src.repositories import notebook_repo
from src.services.audio_scripter import write_dialogue
from src.services.tts_utils import speaker_to_voice, synthesize_utterance

logger = logging.getLogger(__name__)


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
        job_id: str | None = None,
        generation: int | None = None,
    ) -> AudioResult:
        """Synthesize + concatenate the dialogue into a single MP3.

        Returns ``AudioResult`` with status ready/failed and the file path.
        Each job synthesizes into its own temp directory so concurrent jobs
        never overwrite each other's files.
        """
        import uuid

        if not dialogue:
            self._set_failed(notebook, "No dialogue to synthesize.")
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

        # Job-isolated temp directory (never the shared fixed "tmp/" name).
        jid = job_id or uuid.uuid4().hex[:8]
        temp_dir = audio_dir / f"tmp_{jid}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
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
                self._set_failed(notebook, "All utterances failed to synthesize.")
                return AudioResult(
                    status=AUDIO_STATUS_FAILED,
                    audio_path=None,
                    error="All utterances failed to synthesize.",
                )

            # Concatenate into single MP3.
            self._concatenate_audio(temp_files, output_path)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        if self._is_superseded(notebook, generation):
            Path(output_path).unlink(missing_ok=True)
            logger.warning(
                "Audio result for notebook %d discarded (superseded generation)", notebook.id
            )
            return AudioResult(
                status=AUDIO_STATUS_FAILED,
                audio_path=None,
                error="Superseded by a newer job.",
            )

        # Persist to notebook, removing the superseded artifact if replaced.
        previous = notebook.audio_path
        notebook.audio_path = output_path
        notebook.audio_status = AUDIO_STATUS_READY
        notebook.audio_error = None
        db.session.commit()
        if previous and previous != output_path:
            Path(previous).unlink(missing_ok=True)

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

    def _set_failed(self, notebook: Notebook, error: str) -> None:
        """Mark the notebook's audio job failed with a user-visible reason."""
        notebook.audio_status = AUDIO_STATUS_FAILED
        notebook.audio_error = error
        db.session.commit()

    @staticmethod
    def _is_superseded(notebook: Notebook, generation: int | None) -> bool:
        """True when this job's result must be discarded.

        Refreshes the notebook row so a relaunch or delete that committed
        after this job started is observed. A deleted/unreadable row also
        counts as superseded.
        """
        if generation is None:
            return False
        try:
            db.session.refresh(notebook)
        except Exception:  # noqa: BLE001
            return True
        return notebook.audio_generation != generation or notebook.audio_status == AUDIO_STATUS_NONE


# ---------------------------------------------------------------------------
# End-to-end function (used by background job)
# ---------------------------------------------------------------------------


def generate_audio_for_notebook(
    notebook_id: int,
    topic: str = "",
    speaker_a: str = "Ava",
    speaker_b: str = "Andrew",
    job_id: str | None = None,
    generation: int | None = None,
) -> AudioResult | None:
    """Full pipeline: script -> synthesize -> concatenate -> persist.

    Returns ``AudioResult`` or ``None`` on failure. A job whose generation no
    longer matches (relaunch/delete raced it) exits without persisting.
    """
    notebook = notebook_repo.get_by_id(notebook_id)
    if notebook is None:
        logger.error("Notebook %d not found for audio generation", notebook_id)
        return None
    if generation is not None and notebook.audio_generation != generation:
        logger.info(
            "Audio job for notebook %d is stale (job %s, current %s); exiting.",
            notebook_id,
            generation,
            notebook.audio_generation,
        )
        return None

    # Set status to scripting.
    notebook.audio_status = AUDIO_STATUS_SCRIPTING
    notebook.audio_error = None
    db.session.commit()
    logger.info("Audio generation: scripting dialogue for notebook %d", notebook_id)

    # Generate dialogue.
    dialogue = write_dialogue(notebook, topic=topic)
    if not dialogue:
        logger.error("Audio generation: no dialogue produced for notebook %d", notebook_id)
        notebook.audio_status = AUDIO_STATUS_FAILED
        notebook.audio_error = "No dialogue could be generated."
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
    result = svc.generate_audio(
        notebook,
        dialogue,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        job_id=job_id,
        generation=generation,
    )
    logger.info(
        "Audio generation result: notebook=%d status=%s path=%s",
        notebook_id,
        result.status,
        result.audio_path,
    )
    return result
