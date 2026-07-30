"""Video scripter -- generates narration + minimal slides from notebook sources.

The speaker is an expert tutor who explains and elaborates, while slides
carry only key headings and 2-3 short bullet points. The listener focuses
on the speaker, not reading.

Produces a structured JSON with both narration entries and slide content.
The narration is used for TTS; slides are rendered as images.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from src.extensions import db
from src.models import Notebook, Source
from src.repositories import content_registry_repo
from src.services.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)

_MD_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*)\n?```", re.DOTALL)

VIDEO_SYSTEM_PROMPT = (
    "You are an expert educator creating a narrated video presentation. "
    "Given source texts, produce a slide-by-slide script with narration. "
    "\n\n"
    "RULES:\n"
    "- Generate exactly 5-8 slides (title + 4-7 content slides + summary).\n"
    "- Each slide has a short heading (max 8 words) and 2-3 very short bullet points (max 6 words each).\n"
    "- Bullets are visual anchors only — the narration carries the depth.\n"
    "- Each slide has a narration: 2-4 conversational sentences (max 60 words) that explain, "
    "connect, and elaborate. Do NOT just read the bullets aloud. Use analogies, transitions, "
    "and an enthusiastic expert tone.\n"
    "- The title slide narration should introduce the topic enthusiastically.\n"
    "- The summary slide narration should recap key takeaways.\n"
    "{duration_instruction}"
    'Respond as JSON: {{"slides": ['
    '{{"type": "title", "heading": "Short Title", "bullets": ["Brief subtitle"], '
    '"narration": "Welcome! Today we are going to explore..."}}, '
    '{{"type": "content", "heading": "Key Concept", "bullets": ["Point 1", "Point 2"], '
    '"narration": "Let me explain this in more detail..."}}, '
    '{{"type": "summary", "heading": "Key Takeaways", "bullets": ["Takeaway 1", "Takeaway 2"], '
    '"narration": "To wrap up, here is what we covered..."}}'
    "...]}}"
)


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    m = _MD_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            return text[first_newline + 1 :].strip()
    return text


def _build_duration_instruction(min_sec: int, max_sec: int, total_chars: int) -> str:
    target_sec = min(max_sec, max(min_sec, total_chars // 150))
    target_words = target_sec * 2.5
    return (
        f"- Total narration across all slides should be about {target_sec} seconds "
        f"when spoken (~{int(target_words)} words total). "
    )


def parse_video_response(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM's JSON response into a list of slide dicts."""
    if not raw:
        return []
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
        slides = data.get("slides", [])
        if not isinstance(slides, list):
            return []
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse video JSON: %.200s...", cleaned[:200])
        return []

    result: list[dict[str, Any]] = []
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        heading = str(slide.get("heading", "")).strip()
        if not heading:
            continue
        entry: dict[str, Any] = {
            "type": slide.get("type", "content"),
            "heading": heading,
        }
        bullets = slide.get("bullets", [])
        if isinstance(bullets, list):
            entry["bullets"] = [str(b).strip() for b in bullets if str(b).strip()]
        else:
            entry["bullets"] = []
        narration = str(slide.get("narration", "")).strip()
        entry["narration"] = narration if narration else heading
        result.append(entry)
    return result


class VideoScripter:
    """Generates narrated video slide scripts via Ollama Cloud (or mock)."""

    def __init__(self) -> None:
        from src.config import Config

        self._config = Config()
        self._client = get_ollama_client()

    def write_script(self, notebook: Notebook, topic: str = "") -> list[dict[str, Any]]:
        """Generate the slide script with narration."""
        source_texts = self._get_source_texts(notebook.id)
        if not source_texts:
            logger.info("No sources for notebook %d; video script is empty", notebook.id)
            return []

        if self._client._mock:  # noqa: SLF001
            return self._mock_script(notebook.id, source_texts)

        try:
            total_chars = sum(len(t) for t in source_texts)
            duration_instruction = _build_duration_instruction(
                self._config.audio_min_duration_seconds,
                self._config.audio_max_duration_seconds,
                total_chars,
            )
            system_prompt = VIDEO_SYSTEM_PROMPT.format(
                duration_instruction=duration_instruction,
            )

            user_content = (
                "Create a narrated video presentation based on these source texts:\n\n"
                + "\n\n".join(source_texts[:3])
            )
            if topic:
                user_content = f"Focus the presentation on: {topic}\n\n" + user_content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            raw = self._client.chat(messages)
            logger.info(
                "Video script LLM response: %d chars, first=%.100s...",
                len(raw),
                raw[:100],
            )
            slides = parse_video_response(raw)
            if not slides:
                logger.warning(
                    "parse_video_response returned empty for notebook %d. Raw response: %.300s...",
                    notebook.id,
                    raw[:300],
                )
            return slides
        except Exception as exc:  # noqa: BLE001
            logger.error("Video script generation failed for notebook %d: %s", notebook.id, exc)
            return []

    def _get_source_texts(self, notebook_id: int) -> list[str]:
        sources = (
            db.session.query(Source)
            .filter(
                Source.notebook_id == notebook_id,
                Source.status.in_(["ready", "partial"]),
            )
            .all()
        )
        texts: list[str] = []
        for s in sources:
            entry = content_registry_repo.get_by_hash(s.content_hash)
            if entry and entry.extracted_text:
                texts.append(entry.extracted_text)
        return texts

    @staticmethod
    def _mock_script(notebook_id: int, source_texts: list[str]) -> list[dict[str, Any]]:
        digest = hashlib.sha256(f"{notebook_id}:{source_texts[0][:50]}".encode()).hexdigest()[:8]
        return [
            {
                "type": "title",
                "heading": f"Overview of Your Sources [{digest}]",
                "bullets": ["A summary of key topics"],
                "narration": (
                    "Welcome! Today we are going to explore the key topics from your uploaded "
                    "documents. I will walk you through the main concepts and findings, "
                    "explaining each one in a way that is easy to understand."
                ),
            },
            {
                "type": "content",
                "heading": "Key Concepts",
                "bullets": ["Important themes", "Core ideas"],
                "narration": (
                    "Let us start with the key concepts. The documents cover several important "
                    "topics and themes. Each source provides unique insights into the subject "
                    "matter, and together they paint a comprehensive picture."
                ),
            },
            {
                "type": "content",
                "heading": "Main Findings",
                "bullets": ["Primary focus", "Supporting details"],
                "narration": (
                    "Now let us look at the main findings. The primary focus is on the concepts "
                    "and relationships described in the text. There are several important "
                    "details worth highlighting that connect these ideas together."
                ),
            },
            {
                "type": "content",
                "heading": "Practical Applications",
                "bullets": ["Real-world use", "Key benefits"],
                "narration": (
                    "How does this apply in practice? The concepts we have discussed have "
                    "real-world applications that make them valuable. Understanding these "
                    "connections helps you see why these topics matter."
                ),
            },
            {
                "type": "summary",
                "heading": "Key Takeaways",
                "bullets": ["Main themes", "What to remember"],
                "narration": (
                    "To wrap up, we have covered the main themes and supporting details from "
                    "your sources. The key takeaways include the core concepts and their "
                    "practical significance. Thank you for watching!"
                ),
            },
        ]
