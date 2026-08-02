"""Unit tests for src.services.video_service (TDD step X).

The video service generates narrated slide presentations (MP4) via:
1. Slide rendering (Pillow)
2. TTS narration (edge-TTS)
3. Combination via ffmpeg
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.auth_service import hash_password
from src.services.video_service import (
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_NONE,
    VIDEO_STATUS_QUEUED,
    VIDEO_STATUS_READY,
    VIDEO_STATUS_SCRIPTING,
    VIDEO_STATUS_SYNTHESIZING,
    VideoService,
    _ffmpeg_available,
    _load_fonts,
    _wrap_text,
    generate_video_for_notebook,
)


class TestVideoStatus:
    def test_constants(self) -> None:
        assert VIDEO_STATUS_NONE == "none"
        assert VIDEO_STATUS_QUEUED == "queued"
        assert VIDEO_STATUS_SCRIPTING == "scripting"
        assert VIDEO_STATUS_SYNTHESIZING == "synthesizing"
        assert VIDEO_STATUS_READY == "ready"
        assert VIDEO_STATUS_FAILED == "failed"


class TestWrapText:
    def test_wraps_to_width(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create a mock font with predictable getbbox
        mock_font = MagicMock()
        # Return width based on text length * 10 (simulating char width)
        mock_font.getbbox.side_effect = lambda text: (0, 0, len(text) * 10, 20)

        lines = _wrap_text("hello world", mock_font, 50)
        # "hello world" = 11 chars * 10 = 110 > 50, so should wrap
        assert len(lines) >= 1

    def test_empty_string(self) -> None:
        mock_font = MagicMock()
        lines = _wrap_text("", mock_font, 100)
        assert lines == []

    def test_single_word_fits(self) -> None:
        mock_font = MagicMock()
        mock_font.getbbox.return_value = (0, 0, 30, 20)
        lines = _wrap_text("hi", mock_font, 100)
        assert lines == ["hi"]

    def test_long_word_wraps(self) -> None:
        mock_font = MagicMock()
        # call1: "aaaa" -> 100 > 40, too wide; call2: "aaaa bbbb" -> 200 > 40
        mock_font.getbbox.side_effect = [(0, 0, 100, 20), (0, 0, 200, 20)]
        lines = _wrap_text("aaaa bbbb", mock_font, 40)
        assert len(lines) == 2


class TestLoadFonts:
    def test_loads_fonts_or_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # This just tests the function runs without error
        fonts = _load_fonts()
        assert len(fonts) == 3
        for f in fonts:
            assert f is not None


class TestFfmpegAvailable:
    def test_returns_bool(self) -> None:
        result = _ffmpeg_available()
        assert isinstance(result, bool)


class TestVideoService:
    def _make_notebook(self, app: object) -> Notebook:
        with app.app_context():
            u = User(username="vsvc", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Video Service Test")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id
        with app.app_context():
            return db.session.get(Notebook, nb_id)

    def test_mock_generate_creates_file(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("DATA_DIR", tempfile.gettempdir())

        nb = self._make_notebook(app)
        slides = [
            {
                "type": "title",
                "heading": "Test",
                "bullets": ["Overview"],
                "narration": "Welcome to the test.",
            },
            {
                "type": "content",
                "heading": "Slide 2",
                "bullets": ["Point 1", "Point 2"],
                "narration": "Here is the content.",
            },
        ]

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()
            result = svc.generate_video(nb, slides, "Ava")

        assert result.status == "ready"
        assert result.video_path is not None
        assert Path(result.video_path).exists()

    def test_no_slides_returns_failed(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()
            result = svc.generate_video(nb, [], "Ava")

        assert result.status == "failed"
        assert result.error == "No slides to render."

    def test_missing_ffmpeg_returns_failed(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        nb = self._make_notebook(app)
        slides = [{"type": "title", "heading": "Test", "bullets": [], "narration": "Welcome"}]

        with patch("src.services.video_service._ffmpeg_available", return_value=False):
            with app.app_context():
                nb = db.session.merge(nb)
                svc = VideoService()
                result = svc.generate_video(nb, slides, "Ava")

        assert result.status == "failed"
        assert "ffmpeg is not installed" in (result.error or "")

    def test_render_slide_creates_image(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "slide.png")
                slide = {
                    "type": "content",
                    "heading": "Test Slide",
                    "bullets": ["Bullet 1", "Bullet 2"],
                    "narration": "Test narration",
                }
                svc._render_slide(slide, output_path)

                assert Path(output_path).exists()
                assert Path(output_path).stat().st_size > 0

    def test_render_title_slide(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "title.png")
                slide = {
                    "type": "title",
                    "heading": "Title Slide",
                    "bullets": ["Subtitle here"],
                    "narration": "Welcome!",
                }
                svc._render_slide(slide, output_path)

                assert Path(output_path).exists()

    def test_synthesize_mock(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "audio.mp3")
                ok = svc._synthesize("Test narration", "Ava", output_path)

                assert ok is True
                assert Path(output_path).exists()

    def test_combine_to_mp4_creates_concat_files(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            with tempfile.TemporaryDirectory() as tmpdir:
                slide_files = [str(Path(tmpdir) / f"slide_{i}.png") for i in range(2)]
                for f in slide_files:
                    Path(f).write_bytes(b"fake png")

                audio_files = [str(Path(tmpdir) / f"audio_{i}.mp3") for i in range(2)]
                for f in audio_files:
                    Path(f).write_bytes(b"fake mp3")

                output_path = str(Path(tmpdir) / "output.mp4")

                with patch("src.services.video_service.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(stdout="")
                    svc._combine_to_mp4(slide_files, audio_files, output_path)

                # ffprobe duration lookups (2) + one ffmpeg invocation.
                ffmpeg_calls = [c for c in mock_run.call_args_list if "ffmpeg" in c.args[0]]
                assert len(ffmpeg_calls) == 1
                assert "ffprobe" in mock_run.call_args_list[0].args[0]
                # Concat list files were written and cleaned up.
                assert not Path(output_path).with_suffix(".txt").exists()
                assert not Path(output_path).with_suffix(".audio.txt").exists()

    def test_combine_to_mp4_without_audio(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            with tempfile.TemporaryDirectory() as tmpdir:
                slide_files = [str(Path(tmpdir) / f"slide_{i}.png") for i in range(2)]
                for f in slide_files:
                    Path(f).write_bytes(b"fake png")

                # No audio files present -> audio-less ffmpeg invocation.
                audio_files: list[str] = [""]
                output_path = str(Path(tmpdir) / "output.mp4")

                with patch("src.services.video_service.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(stdout="")
                    svc._combine_to_mp4(slide_files, audio_files, output_path)

                assert mock_run.call_count == 1
                args = mock_run.call_args.args[0]
                assert "ffmpeg" in args
                assert not Path(output_path).with_suffix(".txt").exists()

    def test_get_audio_duration_handles_error(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()

            # Non-existent file should return default 5.0
            duration = svc._get_audio_duration("/nonexistent/path.mp3")
            assert duration == 5.0

    def test_set_status(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        nb = self._make_notebook(app)

        with app.app_context():
            nb = db.session.merge(nb)
            svc = VideoService()
            svc._set_status(nb, "ready")
            db.session.refresh(nb)
            assert nb.video_status == "ready"


class TestGenerateVideoForNotebook:
    def _make_notebook_with_source(self, app: object, username: str) -> int:
        with app.app_context():
            u = User(username=username, password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name=f"{username} NB")
            db.session.add(nb)
            db.session.commit()
            h = "g" * 64
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
                chroma_collection="doc_g",
                extracted_text="Machine learning and neural networks are core AI topics.",
                char_count=55,
            )
            return nb.id

    def test_full_pipeline_mock(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("DATA_DIR", tempfile.gettempdir())

        nb_id = self._make_notebook_with_source(app, "genv1")

        with app.app_context():
            result = generate_video_for_notebook(nb_id, topic="AI", speaker="Ava")

        assert result is not None
        assert result.status == "ready"
        assert result.video_path is not None

    def test_not_found_returns_none(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")

        with app.app_context():
            result = generate_video_for_notebook(99999)

        assert result is None

    def test_no_slides_returns_failed(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")

        with app.app_context():
            u = User(username="genv2", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="No Slides Test")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with app.app_context():
            result = generate_video_for_notebook(nb_id)

        assert result is not None
        assert result.status == "failed"
        assert "No slides could be generated" in (result.error or "")

    def test_updates_status_during_pipeline(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("DATA_DIR", tempfile.gettempdir())

        nb_id = self._make_notebook_with_source(app, "genv3")

        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb.video_status == "none"

            # The function updates status internally
            result = generate_video_for_notebook(nb_id)

            nb = db.session.get(Notebook, nb_id)
            assert nb.video_status == "ready"
            assert result.status == "ready"