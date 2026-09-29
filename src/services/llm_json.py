"""Shared LLM JSON extraction helper.

Every service that asks the LLM for JSON previously used a fragile pattern:
find ``{`` … ``rfind('}')`` … ``json.loads``. That fails on markdown fences,
trailing prose, or truncation — and the ``except`` silently swallowed it.

``extract_json`` / ``extract_json_array`` instead:
1. Strip markdown code fences.
2. Use ``json.JSONDecoder().raw_decode`` from the first ``{`` / ``[`` so
   trailing prose is ignored instead of corrupting the parse.
3. Log a WARNING with the raw response on failure so it is diagnosable.

Ported from study-and-learn (``src/services/llm_json.py``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches ```json ... ``` or ``` ... ``` fences.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def extract_json(response: str) -> Any:  # noqa: ANN401
    """Extract the first JSON object from an LLM response.

    Returns the parsed value, or ``None`` when no valid JSON is found
    (with a WARNING log including the first 300 chars for diagnosis).
    """
    if not response or not response.strip():
        return None
    text = response.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        logger.warning("Failed to extract JSON: no '{' found in response: %r", text[:200])
        return None
    try:
        decoder = json.JSONDecoder()
        result, _end = decoder.raw_decode(text[start:])
        return result
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to extract JSON from LLM response: %s. Response (first 300): %r",
            e,
            text[:300],
        )
        return None


def extract_json_array(response: str) -> list[Any] | None:
    """Extract the first JSON array from an LLM response (or ``None``)."""
    if not response or not response.strip():
        return None
    text = response.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    if start == -1:
        logger.warning("Failed to extract JSON array: no '[' found in response: %r", text[:200])
        return None
    try:
        decoder = json.JSONDecoder()
        result, _end = decoder.raw_decode(text[start:])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to extract JSON array from LLM response: %s. Response (first 300): %r",
            e,
            text[:300],
        )
        return None
