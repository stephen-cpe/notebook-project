"""Unit tests for src.services.tts_utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.tts_utils import (
    DEFAULT_VOICE,
    clean_for_tts,
    speaker_to_voice,
    synthesize_utterance,
)


class TestSpeakerToVoice:
    def test_known_speakers(self) -> None:
        assert speaker_to_voice("Ava") == "en-US-AvaNeural"
        assert speaker_to_voice("Andrew") == "en-US-AndrewNeural"
        assert speaker_to_voice("Emma") == "en-US-EmmaNeural"
        assert speaker_to_voice("Ryan") == "en-US-RyanNeural"

    def test_unknown_speaker_falls_back(self) -> None:
        assert speaker_to_voice("Mystery") == DEFAULT_VOICE
        assert speaker_to_voice("") == DEFAULT_VOICE


class TestSynthesizeUtterance:
    def test_mock_writes_stub_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.mp3"
        ok = synthesize_utterance("hello", "en-US-AvaNeural", str(out), mock=True)
        assert ok is True
        assert out.exists()
        assert out.read_bytes().startswith(b"ID3")

    def test_mock_failure_returns_false(self, tmp_path: Path) -> None:
        # Point output at an unwritable path to force a failure.
        out = tmp_path / "nodir" / "out.mp3"
        ok = synthesize_utterance("hello", "en-US-AvaNeural", str(out), mock=True)
        assert ok is False
        assert not out.exists()


class TestSynthesizeRealPath:
    def test_real_synthesize_failure_returns_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The non-mock path logs and returns False on edge_tts failure (P0)."""

        out = tmp_path / "out.mp3"

        async def _boom(*a: object, **k: object) -> None:
            raise RuntimeError("edge-tts down")

        monkeypatch.setattr("src.services.tts_utils._edge_tts_synthesize", _boom)
        ok = synthesize_utterance("hi", "en-US-AvaNeural", str(out), mock=False)
        assert ok is False
        assert not out.exists()


class TestCleanForTts:
    def test_strips_citation_brackets(self) -> None:
        text = "It is reliable [Source 3]."
        assert clean_for_tts(text) == "It is reliable."

    def test_strips_multiple_sources_in_bracket(self) -> None:
        text = "GitHub is a service [Source 1, Source 2, Source 4]."
        assert "[Source" not in clean_for_tts(text)

    def test_strips_bold_markers(self) -> None:
        text = "**Collaboration tools:** include forking."
        assert clean_for_tts(text) == "Collaboration tools: include forking."

    def test_strips_italic_markers(self) -> None:
        text = "It is *very* important."
        assert clean_for_tts(text) == "It is very important."

    def test_strips_code_fences(self) -> None:
        text = "```python\nprint('hi')\n```\nDone."
        assert "```" not in clean_for_tts(text)
        assert "python" not in clean_for_tts(text)

    def test_strips_inline_code(self) -> None:
        text = "Use `git push` to upload."
        assert clean_for_tts(text) == "Use git push to upload."

    def test_strips_headings(self) -> None:
        text = "## Overview\nThis is the overview."
        assert clean_for_tts(text) == "Overview\nThis is the overview."

    def test_strips_bullet_markers(self) -> None:
        text = "- First item\n- Second item"
        result = clean_for_tts(text)
        assert "- " not in result

    def test_strips_markdown_links(self) -> None:
        text = "See [GitHub](https://github.com) for details."
        assert clean_for_tts(text) == "See GitHub for details."

    def test_empty_string(self) -> None:
        assert clean_for_tts("") == ""

    def test_collapses_blank_lines(self) -> None:
        text = "Para one.\n\n\n\nPara two."
        assert clean_for_tts(text) == "Para one.\n\nPara two."

    def test_complex_answer(self) -> None:
        text = (
            "GitHub is a cloud-based Git repository hosting service "
            "[Source 1, Source 2, Source 4]. It offers:\n\n"
            "* **Collaboration tools:** These include forking [Source 1].\n"
            "* **Management policies:** Protected branches [Source 3]."
        )
        result = clean_for_tts(text)
        assert "[Source" not in result
        assert "**" not in result
        assert "* " not in result
