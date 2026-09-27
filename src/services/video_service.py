"""Video service — narrated slide presentation + MP4 generation.

Pipeline:
1. Generate slide images (Pillow, dark theme) — minimal text, clean layout
2. Synthesize TTS audio from narration text via edge-TTS
3. Combine images + audio into MP4 via ffmpeg

Slides are visual anchors only (heading + 2-3 short bullets). The speaker's
narration carries the depth — the listener focuses on the expert, not reading.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.config import Config
from src.extensions import db
from src.models import (
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_NONE,
    VIDEO_STATUS_READY,
    VIDEO_STATUS_SCRIPTING,
    VIDEO_STATUS_SYNTHESIZING,
    Notebook,
)
from src.repositories import notebook_repo
from src.services.tts_utils import speaker_to_voice, synthesize_utterance
from src.services.video_scripter import VideoScripter

logger = logging.getLogger(__name__)

SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720
BG_COLOR = (10, 14, 23)
ACCENT_COLOR = (13, 202, 240)
TEXT_COLOR = (224, 230, 240)
MUTED_COLOR = (139, 149, 167)

#: Slide duration when its narration is missing (seconds).
DEFAULT_SLIDE_SECONDS = 5.0
#: Breathing room appended after each narration so slides never cut audio off.
SLIDE_TAIL_SECONDS = 0.5

MARGIN_LEFT = 100
MARGIN_RIGHT = 100
MARGIN_TOP = 80
BULLET_LEFT = 120
LINE_SPACING = 56


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _load_fonts() -> tuple[Any, Any, Any]:
    font_heading: Any
    font_bullet: Any
    font_subtitle: Any
    try:
        font_heading = ImageFont.truetype("arial.ttf", 44)
        font_bullet = ImageFont.truetype("arial.ttf", 30)
        font_subtitle = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font_heading = ImageFont.load_default()
        font_bullet = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
    return font_heading, font_bullet, font_subtitle


def _wrap_text(text: str, font: Any, max_width: int) -> list[str]:  # noqa: ANN401
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


@dataclass
class VideoResult:
    status: str
    video_path: str | None
    error: str | None = None


class VideoService:
    """Narrated slide presentation generator with TTS + ffmpeg."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            config = Config()
        self._config = config
        self._data_dir: str = config.data_dir
        self._mock: bool = bool(config.ai_mock)

    def generate_video(
        self,
        notebook: Notebook,
        slides: list[dict[str, Any]],
        speaker: str,
        job_id: str | None = None,
        generation: int | None = None,
    ) -> VideoResult:
        """Generate slide images + TTS narration + combine into MP4."""
        import uuid

        if not slides:
            self._set_failed(notebook, "No slides to render.")
            return VideoResult(
                status=VIDEO_STATUS_FAILED, video_path=None, error="No slides to render."
            )

        if not self._mock and not _ffmpeg_available():
            self._set_failed(notebook, "ffmpeg is not installed.")
            return VideoResult(
                status=VIDEO_STATUS_FAILED,
                video_path=None,
                error="ffmpeg is not installed. Install ffmpeg and add it to your PATH.",
            )

        self._set_status(notebook, VIDEO_STATUS_SYNTHESIZING)

        video_dir = Path(self._data_dir) / "video" / str(notebook.id)
        video_dir.mkdir(parents=True, exist_ok=True)
        sig = hashlib.sha256(
            "|".join(s.get("narration", s.get("heading", "")) for s in slides).encode()
        ).hexdigest()[:12]
        output_path = str(video_dir / f"{sig}.mp4")

        if self._mock:
            return self._mock_generate(output_path, notebook, generation)

        # Job-isolated temp directory (never the shared fixed "tmp/" name).
        jid = job_id or uuid.uuid4().hex[:8]
        temp_dir = video_dir / f"tmp_{jid}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            slide_files: list[str] = []
            audio_files: list[str] = []

            for i, slide in enumerate(slides):
                img_path = str(temp_dir / f"slide_{i:04d}.png")
                self._render_slide(slide, img_path)
                slide_files.append(img_path)

                narration = slide.get("narration", slide.get("heading", ""))
                audio_path = str(temp_dir / f"audio_{i:04d}.mp3")
                ok = self._synthesize(narration, speaker, audio_path)
                if ok:
                    audio_files.append(audio_path)
                else:
                    audio_files.append("")

            if not any(a and Path(a).exists() for a in audio_files):
                self._set_failed(notebook, "All narrations failed to synthesize.")
                return VideoResult(
                    status=VIDEO_STATUS_FAILED,
                    video_path=None,
                    error="All narrations failed to synthesize.",
                )

            self._combine_to_mp4(slide_files, audio_files, output_path, scratch_dir=str(temp_dir))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if self._is_superseded(notebook, generation):
            Path(output_path).unlink(missing_ok=True)
            logger.warning(
                "Video result for notebook %d discarded (superseded generation)", notebook.id
            )
            return VideoResult(
                status=VIDEO_STATUS_FAILED,
                video_path=None,
                error="Superseded by a newer job.",
            )

        # Persist, removing the superseded artifact if replaced.
        previous = notebook.video_path
        notebook.video_path = output_path
        notebook.video_status = VIDEO_STATUS_READY
        notebook.video_error = None
        db.session.commit()
        if previous and previous != output_path:
            Path(previous).unlink(missing_ok=True)

        logger.info(
            "Video generated for notebook %d: %d slides, file=%s",
            notebook.id,
            len(slides),
            output_path,
        )
        return VideoResult(status=VIDEO_STATUS_READY, video_path=output_path)

    def _render_slide(self, slide: dict[str, Any], output_path: str) -> None:
        img = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        font_heading, font_bullet, font_subtitle = _load_fonts()

        heading = slide.get("heading", "")
        slide_type = slide.get("type", "content")
        bullets = slide.get("bullets", [])
        max_text_width = SLIDE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

        if slide_type == "title":
            heading_lines = _wrap_text(heading, font_heading, max_text_width)
            y = 200
            for line in heading_lines:
                draw.text((MARGIN_LEFT, y), line, fill=ACCENT_COLOR, font=font_heading)
                y += 56
            if bullets:
                subtitle = bullets[0] if isinstance(bullets, list) else str(bullets)
                sub_lines = _wrap_text(subtitle, font_subtitle, max_text_width)
                y += 20
                for line in sub_lines:
                    draw.text((MARGIN_LEFT, y), line, fill=MUTED_COLOR, font=font_subtitle)
                    y += 36
        else:
            heading_lines = _wrap_text(heading, font_heading, max_text_width)
            y = MARGIN_TOP
            for line in heading_lines:
                draw.text((MARGIN_LEFT, y), line, fill=ACCENT_COLOR, font=font_heading)
                y += 56

            y += 30
            for bullet in bullets:
                bullet_text = str(bullet).strip()
                bullet_lines = _wrap_text(
                    bullet_text, font_bullet, max_text_width - (BULLET_LEFT - MARGIN_LEFT)
                )
                for bl in bullet_lines:
                    draw.text((BULLET_LEFT, y), f"• {bl}", fill=TEXT_COLOR, font=font_bullet)
                    y += LINE_SPACING
                y += 8

        img.save(output_path, "PNG")

    def _synthesize(self, text: str, speaker: str, output_path: str) -> bool:
        voice = speaker_to_voice(speaker)
        return synthesize_utterance(text, voice, output_path, mock=self._mock)

    def _combine_to_mp4(
        self,
        slide_files: list[str],
        audio_files: list[str],
        output_path: str,
        scratch_dir: str | None = None,
    ) -> None:
        """Combine slides + narration into one MP4 with an aligned timeline.

        Each slide is shown for exactly its narration duration plus a short
        tail; slides without narration show for the default duration backed
        by generated silence. The audio track is built from the same segment
        durations, so narration never drifts to the wrong slide and the final
        video is never truncated mid-sentence. ``-shortest`` remains only as
        a guard against rounding differences.

        ``scratch_dir`` holds the ffmpeg concat manifests and generated
        silence files. It must be job-isolated (e.g. the job's ``tmp_<jid>``
        dir): these files used to live next to ``output_path`` under
        content-derived names, so two overlapping jobs for one notebook
        overwrote and deleted each other's manifests mid-run. Callers that
        omit it keep the legacy layout (single-job use, e.g. tests).
        """
        output = Path(output_path)
        scratch = Path(scratch_dir) if scratch_dir else output.parent
        durations: list[float] = []
        for i in range(len(slide_files)):
            audio = audio_files[i] if i < len(audio_files) else ""
            if audio and Path(audio).exists():
                durations.append(self._get_audio_duration(audio) + SLIDE_TAIL_SECONDS)
            else:
                durations.append(DEFAULT_SLIDE_SECONDS)

        concat_file = str(scratch / (output.stem + ".txt"))
        lines: list[str] = []
        for i, img in enumerate(slide_files):
            abs_img = str(Path(img).resolve())
            lines.append(f"file '{abs_img}'")
            lines.append(f"duration {durations[i]:.1f}")
        abs_last = str(Path(slide_files[-1]).resolve())
        lines.append(f"file '{abs_last}'")

        Path(concat_file).write_text("\n".join(lines), encoding="utf-8")

        audio_present = any(a and Path(a).exists() for a in audio_files)
        generated_silences: list[str] = []

        try:
            if audio_present:
                pad_file = str(scratch / "silence_pad.mp3")
                self._make_silence(SLIDE_TAIL_SECONDS, pad_file)
                generated_silences.append(pad_file)
                track: list[str] = []
                for i in range(len(slide_files)):
                    audio = audio_files[i] if i < len(audio_files) else ""
                    if audio and Path(audio).exists():
                        track += [audio, pad_file]
                    else:
                        sil = str(scratch / f"silence_slide_{i:04d}.mp3")
                        self._make_silence(durations[i], sil)
                        generated_silences.append(sil)
                        track.append(sil)
                audio_concat = str(scratch / (output.stem + ".audio.txt"))
                audio_lines = [f"file '{str(Path(a).resolve())}'" for a in track]
                Path(audio_concat).write_text("\n".join(audio_lines), encoding="utf-8")

                subprocess.run(  # noqa: S603
                    [  # noqa: S607
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        concat_file,
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        audio_concat,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                        "-vf",
                        f"scale={SLIDE_WIDTH}:{SLIDE_HEIGHT}",
                        output_path,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
                Path(audio_concat).unlink(missing_ok=True)
            else:
                subprocess.run(  # noqa: S603
                    [  # noqa: S607
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        concat_file,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-vf",
                        f"scale={SLIDE_WIDTH}:{SLIDE_HEIGHT}",
                        output_path,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
        finally:
            Path(concat_file).unlink(missing_ok=True)
            for sil in generated_silences:
                Path(sil).unlink(missing_ok=True)

    @staticmethod
    def _make_silence(duration: float, output_path: str) -> None:
        """Generate a silent MP3 of ``duration`` seconds via ffmpeg."""
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{max(duration, 0.1):.1f}",
                "-c:a",
                "libmp3lame",
                output_path,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )

    @staticmethod
    def _get_audio_duration(path: str) -> float:
        try:
            result = subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 5.0

    def _mock_generate(
        self, output_path: str, notebook: Notebook, generation: int | None = None
    ) -> VideoResult:
        if self._is_superseded(notebook, generation):
            logger.warning(
                "Video result for notebook %d discarded (superseded generation)", notebook.id
            )
            return VideoResult(
                status=VIDEO_STATUS_FAILED, video_path=None, error="Superseded by a newer job."
            )
        Path(output_path).write_bytes(b"stub mp4")
        previous = notebook.video_path
        notebook.video_path = output_path
        notebook.video_status = VIDEO_STATUS_READY
        notebook.video_error = None
        db.session.commit()
        if previous and previous != output_path:
            Path(previous).unlink(missing_ok=True)
        return VideoResult(status=VIDEO_STATUS_READY, video_path=output_path)

    def _set_status(self, notebook: Notebook, status: str) -> None:
        notebook.video_status = status
        db.session.commit()

    def _set_failed(self, notebook: Notebook, error: str) -> None:
        """Mark the notebook's video job failed with a user-visible reason."""
        notebook.video_status = VIDEO_STATUS_FAILED
        notebook.video_error = error
        db.session.commit()

    @staticmethod
    def _is_superseded(notebook: Notebook, generation: int | None) -> bool:
        """True when this job's result must be discarded (relaunch/delete raced it)."""
        if generation is None:
            return False
        try:
            db.session.refresh(notebook)
        except Exception:  # noqa: BLE001
            return True
        return notebook.video_generation != generation or notebook.video_status == VIDEO_STATUS_NONE


def generate_video_for_notebook(
    notebook_id: int,
    topic: str = "",
    speaker: str = "Ava",
    job_id: str | None = None,
    generation: int | None = None,
) -> VideoResult | None:
    """Full pipeline: script -> render slides -> TTS narration -> combine -> persist.

    Returns ``VideoResult`` or ``None`` on failure. A job whose generation no
    longer matches (relaunch/delete raced it) exits without persisting.
    """
    notebook = notebook_repo.get_by_id(notebook_id)
    if notebook is None:
        logger.error("Notebook %d not found for video generation", notebook_id)
        return None
    if generation is not None and notebook.video_generation != generation:
        logger.info(
            "Video job for notebook %d is stale (job %s, current %s); exiting.",
            notebook_id,
            generation,
            notebook.video_generation,
        )
        return None

    notebook.video_status = VIDEO_STATUS_SCRIPTING
    notebook.video_error = None
    db.session.commit()
    logger.info("Video generation: scripting for notebook %d", notebook_id)

    scripter = VideoScripter()
    slides = scripter.write_script(notebook, topic=topic)
    if not slides:
        logger.error("Video generation: no slides produced for notebook %d", notebook_id)
        notebook.video_status = VIDEO_STATUS_FAILED
        notebook.video_error = "No slides could be generated."
        db.session.commit()
        return VideoResult(
            status=VIDEO_STATUS_FAILED,
            video_path=None,
            error="No slides could be generated.",
        )

    logger.info(
        "Video generation: got %d slides for notebook %d, rendering...",
        len(slides),
        notebook_id,
    )

    svc = VideoService()
    try:
        result = svc.generate_video(notebook, slides, speaker, job_id=job_id, generation=generation)
    except Exception:  # noqa: BLE001
        logger.exception("Video generation crashed for notebook %d", notebook_id)
        try:
            fresh = notebook_repo.get_by_id(notebook_id)
            if fresh is not None and (generation is None or fresh.video_generation == generation):
                fresh.video_status = VIDEO_STATUS_FAILED
                fresh.video_error = "Generation crashed unexpectedly."
                db.session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Could not mark video failed for notebook %d", notebook_id)
        return VideoResult(
            status=VIDEO_STATUS_FAILED, video_path=None, error="Generation crashed unexpectedly."
        )
    logger.info("Video generation result: notebook=%d status=%s", notebook_id, result.status)
    return result
