"""Shared TTS helpers.

Extracts the duplicated edge-TTS synthesis + speaker->voice mapping out of
``AudioService`` and ``VideoService`` so the voice conversation pipeline can
reuse the same code path instead of forking a third copy.

- ``speaker_to_voice(speaker)`` maps a friendly speaker name to an edge-TTS
  voice id (en-US Neural voices).
- ``synthesize_utterance(text, voice, output_path, mock)`` synthesizes a single
  utterance to an MP3 file. Returns True on success, False on failure (errors
  are logged, never raised).
- ``clean_for_tts(text)`` strips markdown formatting and citation brackets
  from an LLM answer so the spoken version reads naturally (no "asterisk
  asterisk" or "Source 3" verbatim).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SPEAKER_TO_VOICE: dict[str, str] = {
    "Ava": "en-US-AvaNeural",
    "Andrew": "en-US-AndrewNeural",
    "Emma": "en-US-EmmaNeural",
    "Ryan": "en-US-RyanNeural",
}

DEFAULT_VOICE = "en-US-AvaNeural"


def speaker_to_voice(speaker: str) -> str:
    """Map a friendly speaker name to an edge-TTS voice id.

    Falls back to ``DEFAULT_VOICE`` for unknown names so a bad setting never
    crashes synthesis.
    """
    return _SPEAKER_TO_VOICE.get(speaker, DEFAULT_VOICE)


def clean_for_tts(text: str) -> str:
    """Strip markdown + citation brackets from ``text`` for natural speech.

    The chat UI keeps the original formatted answer (with citations and
    markdown); this helper produces a spoken-friendly version:

    - Citation brackets ``[Source 3]`` / ``[Source 1, Source 2]`` are removed
      entirely (sources are shown as badges in the UI — no need to read them).
    - Bold/italic markers ``**word**`` / ``*word*`` / ``__word__`` are stripped
      to just ``word`` (no "asterisk asterisk").
    - Markdown headings ``##`` are removed.
    - Bullet markers ``- `` / ``* `` at line starts are removed.
    - Code fences and inline code backticks are removed.
    - Markdown links ``[text](url)`` become just ``text``.
    - Multiple blank lines and trailing/leading whitespace are collapsed.
    """
    if not text:
        return ""

    result = text

    # 1. Remove citation brackets: [Source 3], [Source 1, Source 2, Source 4]
    result = re.sub(
        r"\[(?:Source\s+\d+(?:\s*,\s*Source\s+\d+)*)\]", "", result, flags=re.IGNORECASE
    )

    # 2. Remove code fences ```...```
    result = re.sub(r"```[^\n]*\n?", "", result)
    # 3. Remove inline code backticks
    result = re.sub(r"`([^`]*)`", r"\1", result)

    # 4. Remove markdown headings: ## Heading -> Heading
    result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)

    # 5. Remove bold/italic markers: **word**, *word*, __word__, _word_
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", result)
    result = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", result)

    # 6. Remove markdown links: [text](url) -> text
    result = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", result)

    # 7. Remove bullet markers at line starts: "- " or "* " -> ""
    result = re.sub(r"^[\-\*]\s+", "", result, flags=re.MULTILINE)

    # 8. Remove horizontal rules: --- or ***
    result = re.sub(r"^[\-\*]{3,}\s*$", "", result, flags=re.MULTILINE)

    # 9. Fix spacing left by removed citations: " ." -> ".", " ," -> ","
    result = re.sub(r"\s+([.,;:!?)])", r"\1", result)

    # 10. Collapse multiple blank lines into one, strip leading/trailing space.
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    return result


def synthesize_utterance(text: str, voice: str, output_path: str, mock: bool = False) -> bool:
    """Synthesize a single utterance to ``output_path`` (MP3).

    In mock mode (``mock=True``), writes a minimal stub MP3 without any network
    call (offline/test friendly). Returns True on success, False on failure.
    """
    if mock:
        return _mock_synthesize(output_path)
    try:
        asyncio.run(_edge_tts_synthesize(text, voice, output_path))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("TTS failed for voice %s: %s", voice, exc)
        return False


def _mock_synthesize(output_path: str) -> bool:
    """Write a stub MP3 file (minimal valid MP3 header)."""
    try:
        Path(output_path).write_bytes(
            b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 100
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Mock synthesize failed: %s", exc)
        return False


async def _edge_tts_synthesize(text: str, voice: str, output_path: str) -> None:
    """Call edge-TTS to synthesize a single utterance."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
