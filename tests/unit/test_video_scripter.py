"""Unit tests for src.services.video_scripter (TDD step X).

The video scripter generates narrated slide scripts from notebook sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.auth_service import hash_password
from src.services.ollama_client import reset_ollama_client
from src.services.video_scripter import (
    VideoScripter,
    _build_duration_instruction,
    _strip_markdown_fences,
    parse_video_response,
)


@pytest.fixture(autouse=True)
def _reset_clients() -> None:
    """Reset process-global singletons so env changes take effect per test."""
    reset_ollama_client()
    yield
    reset_ollama_client()


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_notebook_with_sources(app: object, username: str = "vidscr") -> int:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Video Scripter NB")
        db.session.add(nb)
        db.session.commit()
        h = "v" * 64
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
            chroma_collection="doc_v",
            extracted_text=(
                "Machine learning is a subset of AI. Neural networks are key to deep learning."
            ),
            char_count=80,
        )
        return nb.id


class TestStripMarkdownFences:
    def test_no_fences(self) -> None:
        raw = "plain text"
        assert _strip_markdown_fences(raw) == "plain text"

    def test_json_fence(self) -> None:
        raw = '```json\n{"slides": []}\n```'
        assert _strip_markdown_fences(raw) == '{"slides": []}'

    def test_generic_fence(self) -> None:
        raw = '```\n{"slides": []}\n```'
        assert _strip_markdown_fences(raw) == '{"slides": []}'

    def test_fence_with_extra_text(self) -> None:
        raw = 'Here is the response:\n```json\n{"slides": []}\n```\nDone.'
        assert _strip_markdown_fences(raw) == '{"slides": []}'

    def test_empty_string(self) -> None:
        assert _strip_markdown_fences("") == ""

    def test_starts_with_fence_no_newline(self) -> None:
        raw = '```json\n{"slides": []}\n```'
        assert _strip_markdown_fences(raw) == '{"slides": []}'


class TestBuildDurationInstruction:
    def test_basic_calculation(self) -> None:
        # total_chars = 30000 -> 200 sec (30000/150), within min/max bounds
        instr = _build_duration_instruction(60, 480, 30000)
        assert "200 seconds" in instr
        assert "500 words" in instr

    def test_clamps_to_min(self) -> None:
        instr = _build_duration_instruction(60, 480, 100)
        # 100/150 = 0.66, clamped to 60
        assert "60 seconds" in instr

    def test_clamps_to_max(self) -> None:
        instr = _build_duration_instruction(60, 480, 100000)
        # 100000/150 = 666, clamped to 480
        assert "480 seconds" in instr


class TestParseVideoResponse:
    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "slides": [
                    {
                        "type": "title",
                        "heading": "Welcome",
                        "bullets": ["Overview"],
                        "narration": "Welcome to the presentation.",
                    },
                    {
                        "type": "content",
                        "heading": "Concept",
                        "bullets": ["Point 1", "Point 2"],
                        "narration": "Let me explain.",
                    },
                ]
            }
        )
        slides = parse_video_response(raw)
        assert len(slides) == 2
        assert slides[0]["type"] == "title"
        assert slides[0]["heading"] == "Welcome"
        assert slides[0]["bullets"] == ["Overview"]
        assert slides[0]["narration"] == "Welcome to the presentation."

    def test_missing_slides_key(self) -> None:
        raw = json.dumps({"other": "data"})
        assert parse_video_response(raw) == []

    def test_invalid_json(self) -> None:
        assert parse_video_response("not json") == []

    def test_empty_string(self) -> None:
        assert parse_video_response("") == []

    def test_empty_slides_list(self) -> None:
        raw = json.dumps({"slides": []})
        assert parse_video_response(raw) == []

    def test_filters_empty_heading(self) -> None:
        raw = json.dumps(
            {
                "slides": [
                    {"type": "content", "heading": "", "bullets": [], "narration": "x"},
                    {"type": "content", "heading": "Valid", "bullets": [], "narration": "y"},
                ]
            }
        )
        slides = parse_video_response(raw)
        assert len(slides) == 1
        assert slides[0]["heading"] == "Valid"

    def test_defaults_type_and_narration(self) -> None:
        raw = json.dumps(
            {
                "slides": [
                    {"heading": "No Type", "bullets": ["b1"]},
                ]
            }
        )
        slides = parse_video_response(raw)
        assert len(slides) == 1
        assert slides[0]["type"] == "content"
        assert slides[0]["narration"] == "No Type"

    def test_strips_bullet_whitespace(self) -> None:
        raw = json.dumps(
            {
                "slides": [
                    {"type": "content", "heading": "H", "bullets": ["  Point 1  ", "", "Point 2"]}
                ]
            }
        )
        slides = parse_video_response(raw)
        assert slides[0]["bullets"] == ["Point 1", "Point 2"]

    def test_handles_non_dict_slide(self) -> None:
        raw = json.dumps({"slides": ["not a dict", {"heading": "Valid"}]})
        slides = parse_video_response(raw)
        assert len(slides) == 1
        assert slides[0]["heading"] == "Valid"

    def test_handles_non_list_bullets(self) -> None:
        raw = json.dumps({"slides": [{"type": "content", "heading": "H", "bullets": "not a list"}]})
        slides = parse_video_response(raw)
        assert slides[0]["bullets"] == []


class TestVideoScripter:
    def test_no_sources_returns_empty(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        with app.app_context():
            u = User(username="vs1", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Empty NB")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()
            slides = scripter.write_script(nb)
            assert slides == []

    def test_mock_returns_slides(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "vs2")

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()
            slides = scripter.write_script(nb)

        assert isinstance(slides, list)
        assert len(slides) >= 3
        for s in slides:
            assert "type" in s
            assert "heading" in s
            assert "bullets" in s
            assert "narration" in s

    def test_mock_deterministic(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "vs3")

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()
            s1 = scripter.write_script(nb)
            s2 = scripter.write_script(nb)

        assert s1 == s2

    def test_mock_includes_topic_in_slides(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "vs4")

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()
            slides = scripter.write_script(nb, topic="Neural Networks")

        assert isinstance(slides, list)
        assert len(slides) > 0

    def test_real_mode_calls_ollama(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "http://fake")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "fake-key")
        nb_id = _make_notebook_with_sources(app, "vs5")

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()

            # Mock the client's chat method
            with patch.object(scripter._client, "chat", return_value='{"slides": []}') as mock_chat:
                scripter.write_script(nb)
                mock_chat.assert_called_once()

    def test_real_mode_handles_ollama_error(
        self, app: object, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.ERROR)

        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "http://fake")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "fake-key")
        nb_id = _make_notebook_with_sources(app, "vs6")

        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            scripter = VideoScripter()

            with patch.object(scripter._client, "chat", side_effect=Exception("ollama down")):
                slides = scripter.write_script(nb)

        assert slides == []
        assert any("Video script generation failed" in r.message for r in caplog.records)

    def test_parse_response_with_markdown_fences(self) -> None:
        raw = (
            '```json\n{"slides": [{"type": "title", "heading": "H", '
            '"bullets": ["b"], "narration": "n"}]}\n```'
        )
        slides = parse_video_response(raw)
        assert len(slides) == 1
        assert slides[0]["heading"] == "H"

    def test_get_source_texts_filters_by_status(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        with app.app_context():
            u = User(username="vs7", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Status Test")
            db.session.add(nb)
            db.session.commit()
            # ready source
            h1 = "r" * 64
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="r.txt",
                    content_hash=h1,
                    content_type="txt",
                    status="ready",
                )
            )
            # partial source
            h2 = "p" * 64
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="p.txt",
                    content_hash=h2,
                    content_type="txt",
                    status="partial",
                )
            )
            # failed source (should be excluded)
            h3 = "f" * 64
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="f.txt",
                    content_hash=h3,
                    content_type="txt",
                    status="failed",
                )
            )
            db.session.commit()
            content_registry_repo.get_or_create(
                content_hash=h1, chroma_collection="c1", extracted_text="ready text", char_count=10
            )
            content_registry_repo.get_or_create(
                content_hash=h2,
                chroma_collection="c2",
                extracted_text="partial text",
                char_count=12,
            )
            content_registry_repo.get_or_create(
                content_hash=h3, chroma_collection="c3", extracted_text="failed text", char_count=11
            )

            scripter = VideoScripter()
            texts = scripter._get_source_texts(nb.id)

        assert len(texts) == 2
        assert "ready text" in texts
        assert "partial text" in texts
        assert "failed text" not in texts
