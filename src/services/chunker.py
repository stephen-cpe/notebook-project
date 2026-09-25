"""Text chunker — RecursiveCharacterTextSplitter wrapper.

Splits extracted text into overlapping chunks for embedding. Uses LangChain's
``RecursiveCharacterTextSplitter`` (chunk_size=1000, overlap=200) with
hierarchical separators. Handles empty/whitespace input gracefully.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    return _splitter


def chunk_text(text: str) -> list[str]:
    """Split ``text`` into overlapping chunks. Empty input returns ``[]``."""
    if not text or not text.strip():
        return []
    return _get_splitter().split_text(text)


def chunk_units(units: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    """Split per-location units, preserving each chunk's page.

    Each input is ``(text, page)`` where ``page`` is a real PDF page / PPTX
    slide number or ``None``. Returns ``(chunk, page)`` pairs so citations
    carry true locations instead of fabricated chunk indices.
    """
    out: list[tuple[str, int | None]] = []
    for text, page in units:
        for chunk in chunk_text(text):
            out.append((chunk, page))
    return out


def chunk_with_metadata(
    text: str, base_metadata: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Split text and attach ``base_metadata`` to each chunk's dict."""
    chunks = chunk_text(text)
    base = base_metadata or {}
    return [{"text": c, **base, "chunk_index": i} for i, c in enumerate(chunks)]
