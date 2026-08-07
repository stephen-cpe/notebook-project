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
from src.models import Notebook
from src.repositories import notebook_repo
from src.services.tts_utils import speaker_to_voice, synthesize_utterance
from src.services.video_scripter import VideoScripter

logger = logging.getLogger(__name__)

VIDEO_STATUS_NONE = "none"
VIDEO_STATUS_QUEUED = "queued"
VIDEO_STATUS_SCRIPTING = "scripting"
VIDEO_STATUS_SYNTHESIZING = "synthesizing"
VIDEO_STATUS_READY = "ready"
VIDEO_STATUS_FAILED = "failed"

SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720
BG_COLOR = (10, 14, 23)
ACCENT_COLOR = (13, 202, 240)
TEXT_COLOR = (224, 230, 240)
MUTED_COLOR = (139, 149, 167)

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
    ) -> VideoResult:
        """Generate slide images + TTS narration + combine into MP4."""
        if not slides:
            self._set_status(notebook, VIDEO_STATUS_FAILED)
            return VideoResult(
                status=VIDEO_STATUS_FAILED, video_path=None, error="No slides to render."
            )

        if not self._mock and not _ffmpeg_available():
            self._set_status(notebook, VIDEO_STATUS_FAILED)
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
            return self._mock_generate(output_path, notebook)

        temp_dir = video_dir / "tmp"
        temp_dir.mkdir(exist_ok=True)

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

            self._combine_to_mp4(slide_files, audio_files, output_path)

            for f in slide_files + audio_files:
                Path(f).unlink(missing_ok=True)

            notebook.video_path = output_path
            notebook.video_status = VIDEO_STATUS_READY
            db.session.commit()

            logger.info(
                "Video generated for notebook %d: %d slides, file=%s",
                notebook.id,
                len(slides),
                output_path,
            )
            return VideoResult(status=VIDEO_STATUS_READY, video_path=output_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Video generation failed for notebook %d: %s", notebook.id, exc)
            self._set_status(notebook, VIDEO_STATUS_FAILED)
            return VideoResult(status=VIDEO_STATUS_FAILED, video_path=None, error=str(exc))

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
        self, slide_files: list[str], audio_files: list[str], output_path: str
    ) -> None:
        concat_file = str(Path(output_path).with_suffix(".txt"))
        lines: list[str] = []
        for i, img in enumerate(slide_files):
            duration = 5.0
            if i < len(audio_files) and audio_files[i] and Path(audio_files[i]).exists():
                duration = self._get_audio_duration(audio_files[i]) + 0.5
            abs_img = str(Path(img).resolve())
            lines.append(f"file '{abs_img}'")
            lines.append(f"duration {duration:.1f}")
        abs_last = str(Path(slide_files[-1]).resolve())
        lines.append(f"file '{abs_last}'")

        Path(concat_file).write_text("\n".join(lines), encoding="utf-8")

        audio_present = any(a and Path(a).exists() for a in audio_files)

        if audio_present:
            audio_concat = str(Path(output_path).with_suffix(".audio.txt"))
            audio_lines = [
                f"file '{str(Path(a).resolve())}'" for a in audio_files if a and Path(a).exists()
            ]
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

        Path(concat_file).unlink(missing_ok=True)

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

    def _mock_generate(self, output_path: str, notebook: Notebook) -> VideoResult:
        Path(output_path).write_bytes(b"stub mp4")
        notebook.video_path = output_path
        notebook.video_status = VIDEO_STATUS_READY
        db.session.commit()
        return VideoResult(status=VIDEO_STATUS_READY, video_path=output_path)

    def _set_status(self, notebook: Notebook, status: str) -> None:
        notebook.video_status = status
        db.session.commit()


def generate_video_for_notebook(
    notebook_id: int, topic: str = "", speaker: str = "Ava"
) -> VideoResult | None:
    """Full pipeline: script -> render slides -> TTS narration -> combine -> persist."""
    notebook = notebook_repo.get_by_id(notebook_id)
    if notebook is None:
        logger.error("Notebook %d not found for video generation", notebook_id)
        return None

    notebook.video_status = VIDEO_STATUS_SCRIPTING
    db.session.commit()
    logger.info("Video generation: scripting for notebook %d", notebook_id)

    scripter = VideoScripter()
    slides = scripter.write_script(notebook, topic=topic)
    if not slides:
        logger.error("Video generation: no slides produced for notebook %d", notebook_id)
        notebook.video_status = VIDEO_STATUS_FAILED
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
    result = svc.generate_video(notebook, slides, speaker)
    logger.info("Video generation result: notebook=%d status=%s", notebook_id, result.status)
    return result
