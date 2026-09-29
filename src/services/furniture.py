"""Running-header/footer dedup without vectors (ported from pdf2md).

PDFs repeat headers, footers, and page numbers on every page. Embedding
them pollutes every chunk and drags down retrieval. This strips them with
pure string matching (no embedding model, no vector store):

- Candidate furniture = first 2 + last 2 non-empty lines of each page.
- A line is furniture when it recurs on >=60% of pages (exact match, or
  ``difflib`` >= 0.9 fuzzy match for page numbers like "Page 3 of 12").
- Lines containing figure placeholders (``FIG`` / ``[Image`` / ``[Page``)
  are never furniture.
- Only strips from page edges (never mid-page body text).
"""

from __future__ import annotations

import difflib
import logging

logger = logging.getLogger(__name__)

_RECURRENCE_THRESHOLD = 0.60
_FUZZY_THRESHOLD = 0.9
_NEVER_FURNITURE_MARKERS = ("FIG", "[Image", "[Page", "[Document")


def _is_protected(line: str) -> bool:
    """Return True for lines that must never be stripped (figures/markers)."""
    upper = line.upper()
    return any(m.upper() in upper for m in _NEVER_FURNITURE_MARKERS)


def dedup_furniture(page_texts: list[str]) -> list[str]:
    """Strip recurring headers/footers from per-page texts.

    Args:
        page_texts: one string per page (1-based order preserved).

    Returns:
        New list with furniture lines removed from page edges. Input shorter
        than 3 pages is returned unchanged (recurrence needs evidence).
    """
    if len(page_texts) < 3:
        return list(page_texts)
    # Collect edge-line candidates per page.
    edge_lines: list[list[str]] = []
    for text in page_texts:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        edges = (lines[:2] + lines[-2:]) if lines else []
        edge_lines.append([ln for ln in edges if ln and not _is_protected(ln)])
    # Count recurrence across pages (exact or fuzzy).
    candidates: dict[str, int] = {}
    for edges in edge_lines:
        seen: set[str] = set()
        for ln in edges:
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates[key] = candidates.get(key, 0) + 1
    total = len(page_texts)
    furniture = {
        key
        for key, count in candidates.items()
        if count / total >= _RECURRENCE_THRESHOLD or _fuzzy_recurs(key, candidates, total)
    }
    if not furniture:
        return list(page_texts)
    cleaned: list[str] = []
    for text in page_texts:
        lines = (text or "").splitlines()
        # Strip leading furniture.
        start = 0
        while start < len(lines) and lines[start].strip().lower() in furniture:
            start += 1
        # Strip trailing furniture.
        end = len(lines)
        while end > start and lines[end - 1].strip().lower() in furniture:
            end -= 1
        cleaned.append("\n".join(lines[start:end]))
    logger.info("dedup_furniture: stripped %d recurring lines", len(furniture))
    return cleaned


def _fuzzy_recurs(key: str, candidates: dict[str, int], total: int) -> bool:
    """Return True when fuzzy variants of ``key`` recur on most pages."""
    similar = sum(
        count
        for other, count in candidates.items()
        if other != key and difflib.SequenceMatcher(None, key, other).ratio() >= _FUZZY_THRESHOLD
    )
    return (similar + candidates.get(key, 0)) / total >= _RECURRENCE_THRESHOLD
