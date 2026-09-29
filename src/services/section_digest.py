"""Full-coverage section digests — map-reduce over source text.

This is the step where 100% of an uploaded file genuinely gets read: the
extracted text is split into sections, the LLM faithfully summarizes *every*
section (the map pass), and the section summaries are stitched into one
digest (the reduce step). The digest is cached in ``content_registry`` keyed
by content hash + pipeline fingerprint, so it is built once ever and shared
across notebooks/users that upload the same file.

- Chat/overviews prepend the digest (everything is *represented*) on top of
  top-k retrieved passages (precise, quotable evidence).
- Builds run in background threads (summary/audio/video jobs), never in the
  chat request path: chat only reads the cache and falls back to the cheap
  extractive digest when the cache is cold.
- Never raises: any failure degrades to ``""`` and callers fall back.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config

MAP_SYSTEM_PROMPT = (
    "You are a faithful reading-digest writer preparing notes for another AI "
    "that will answer questions about this document. Summarize the section "
    "below in 100-150 words. Preserve all facts, numbers, names, and claims "
    "exactly as stated. Do not add information, do not paraphrase away "
    "precision, do not give opinions. If the section is mostly boilerplate "
    "(headers, footers, tables of contents), say so in one sentence."
)

_building: set[str] = set()
_building_lock = threading.Lock()


def split_into_sections(text: str, max_section_chars: int = 6000) -> list[str]:
    """Split ``text`` into sections of at most ``max_section_chars``.

    Splits on blank lines first (paragraphs stay intact); a single paragraph
    longer than the limit is hard-split on sentence boundaries. Empty input
    yields ``[]``. Every non-blank character of the input appears in exactly
    one section.
    """
    text = text or ""
    if not text.strip():
        return []
    max_section_chars = max(500, int(max_section_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_section_chars:
            if current:
                sections.append("\n\n".join(current))
                current, current_len = [], 0
            sections.extend(_hard_split_paragraph(para, max_section_chars))
            continue
        if current and current_len + 2 + len(para) > max_section_chars:
            sections.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += (2 if current_len else 0) + len(para)
    if current:
        sections.append("\n\n".join(current))
    return sections


def _hard_split_paragraph(para: str, max_chars: int) -> list[str]:
    """Split an overlong paragraph on sentence boundaries (best-effort)."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    parts: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > max_chars:
            parts.append(current)
            current = ""
        current = f"{current} {sent}".strip() if current else sent
        while len(current) > max_chars:
            parts.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        parts.append(current)
    return parts


def _summarize_section(section: str, index: int, total: int, filename: str = "") -> str:
    """Summarize one section via the chat model ("" on any failure)."""
    try:
        from src.services.ollama_client import get_ollama_client

        client = get_ollama_client()
        source = f" from {filename}" if filename else ""
        user_content = f"Summarize section {index} of {total}{source}:\n\n{section}"
        out = client.chat(
            [
                {"role": "system", "content": MAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        return (out or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Section digest map failed (section %d): %s", index, exc)
        return ""


def _digest_key(content_hash: str, pipeline: str) -> str:
    """Stable key fragment for logs (never the full hash)."""
    short = hashlib.sha256(f"{content_hash}|{pipeline}".encode()).hexdigest()[:12]
    return short


def get_cached_digest(content_hash: str, config: Config | None = None) -> str:
    """Return the cached digest when present and pipeline-current, else ""."""
    try:
        from src.repositories import content_registry_repo
        from src.services.pipeline_version import compute_pipeline_version

        if config is None:
            from src.config import Config as _Config

            config = _Config()
        entry = content_registry_repo.get_by_hash(content_hash)
        if entry is None:
            return ""
        current = compute_pipeline_version(config)
        if entry.digest_pipeline != current:
            return ""
        return entry.section_digest or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Digest cache read failed for %s: %s", content_hash[:12], exc)
        return ""


def build_source_digest(content_hash: str, filename: str = "", config: Config | None = None) -> str:
    """Build, cache, and return the section digest for ``content_hash``.

    Reads the extracted text from the registry, maps every section through
    the LLM, and stitches the results. Returns "" when there is no text,
    the build fails, or no sections survive. Never raises.
    """
    try:
        from src.config import Config as _Config
        from src.repositories import content_registry_repo
        from src.services.pipeline_version import compute_pipeline_version

        if config is None:
            config = _Config()
        entry = content_registry_repo.get_by_hash(content_hash)
        text = (entry.extracted_text if entry and entry.extracted_text else "").strip()
        if not text:
            return ""
        max_sections = max(1, int(config.rag_map_max_sections))
        section_chars = max(500, int(config.rag_map_section_chars))
        sections = split_into_sections(text, section_chars)
        if len(sections) > max_sections:
            # Widen sections instead of dropping the tail: full coverage
            # with fewer, larger map calls.
            section_chars = max(section_chars, -(-len(text) // max_sections))
            sections = split_into_sections(text, section_chars)
            logger.info(
                "Digest for %s: widened sections to %d chars to fit %d sections",
                _digest_key(content_hash, ""),
                section_chars,
                len(sections),
            )
        total = len(sections)
        if total == 0:
            return ""
        parts: list[str] = []
        for i, section in enumerate(sections, start=1):
            summary = _summarize_section(section, i, total, filename)
            if summary:
                parts.append(f"[Section {i}/{total}]\n{summary}")
        if not parts:
            return ""
        digest = "\n\n".join(parts)
        pipeline = compute_pipeline_version(config)
        content_registry_repo.save_digest(content_hash, digest, pipeline)
        logger.info(
            "Built section digest for %s: %d sections, %d chars",
            _digest_key(content_hash, pipeline),
            total,
            len(digest),
        )
        return digest
    except Exception as exc:  # noqa: BLE001
        logger.warning("Section digest build failed for %s: %s", content_hash[:12], exc)
        return ""


def ensure_source_digest(
    content_hash: str, filename: str = "", config: Config | None = None
) -> str:
    """Return the cached digest, building it (once) when cold or stale.

    A process-wide in-progress guard prevents duplicate concurrent builds of
    the same hash; concurrent losers get whatever the winner cached (possibly
    "" if they arrive mid-build — the next call retries). Never raises.
    """
    try:
        cached = get_cached_digest(content_hash, config)
        if cached:
            return cached
        with _building_lock:
            if content_hash in _building:
                return ""
            _building.add(content_hash)
        try:
            return build_source_digest(content_hash, filename, config)
        finally:
            with _building_lock:
                _building.discard(content_hash)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Digest ensure failed for %s: %s", content_hash[:12], exc)
        return ""


def ensure_notebook_digests(notebook_id: int, config: Config | None = None) -> int:
    """Ensure digests for all ready/partial sources of a notebook.

    Intended for background threads (summary/audio/video jobs), not the chat
    request path. Returns the number of sources with a usable digest
    afterwards. Never raises.
    """
    try:
        from sqlalchemy import select

        from src.extensions import db
        from src.models import Source

        if config is None:
            from src.config import Config as _Config

            config = _Config()
        sources = db.session.scalars(
            select(Source).where(
                Source.notebook_id == notebook_id,
                Source.status.in_(["ready", "partial"]),
            )
        ).all()
        usable = 0
        for src in sources:
            if ensure_source_digest(src.content_hash, src.filename, config):
                usable += 1
        logger.info("Notebook %d digests: %d/%d sources covered", notebook_id, usable, len(sources))
        return usable
    except Exception as exc:  # noqa: BLE001
        logger.warning("Notebook digest ensure failed for %d: %s", notebook_id, exc)
        return 0


def reset_digest_build_state() -> None:
    """Clear the in-progress guard (used by tests)."""
    with _building_lock:
        _building.clear()


def notebook_digest_text(notebook_id: int, max_chars: int, config: Config | None = None) -> str:
    """Stitch cached section digests for a notebook's sources (upload order).

    Returns "" when no source has a usable cached digest (cold cache — the
    caller falls back to the cheap extractive digest). Whole per-source
    blocks only; never exceeds ``max_chars``. Never raises.
    """
    try:
        from sqlalchemy import select

        from src.extensions import db
        from src.models import Source

        if config is None:
            from src.config import Config as _Config

            config = _Config()
        max_chars = max(0, int(max_chars))
        if max_chars <= 0:
            return ""
        sources = db.session.scalars(
            select(Source)
            .where(
                Source.notebook_id == notebook_id,
                Source.status.in_(["ready", "partial"]),
            )
            .order_by(Source.created_at)
        ).all()
        blocks: list[str] = []
        running = 0
        for src in sources:
            digest = get_cached_digest(src.content_hash, config)
            if not digest:
                continue
            block = f"[Full digest — {src.filename}]\n{digest}"
            if running + len(block) > max_chars:
                break
            blocks.append(block)
            running += len(block)
        return "\n\n".join(blocks)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Notebook digest stitch failed for %d: %s", notebook_id, exc)
        return ""
