"""Source selection for LLM context assembly.

Shared helper that gathers a notebook's ready/partial source texts from the
ContentRegistry (deterministically ordered by upload time) and selects as many
as fit within a character budget. Used by the summary, audio, and video
overview generators so they share one consistent, deterministic, budgeted
context-building path instead of each silently capping to a hardcoded count.

- ``select_sources_within_budget(notebook_id, max_chars)`` returns the list of
  source texts (ordered by upload time) that fit within ``max_chars``. The
  budget is hard: an oversized source is truncated (with a marker) so a
  single 25 MB document can never become a 25-million-character prompt.
  Dropped sources are logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select

from src.extensions import db
from src.models import Source
from src.repositories import content_registry_repo

logger = logging.getLogger(__name__)


@dataclass
class SourceSelection:
    """Result of selecting sources within a character budget.

    ``texts`` are the selected source texts in upload order; ``total_chars`` is
    the sum of their lengths; ``used_count`` / ``total_count`` describe how many
    sources were included vs. available.
    """

    texts: list[str]
    total_chars: int
    used_count: int
    total_count: int


def select_sources_within_budget(notebook_id: int, max_chars: int) -> SourceSelection:
    """Select notebook source texts that fit within ``max_chars``.

    Sources are ordered by ``created_at`` (upload order) and included until the
    running character total exceeds ``max_chars``. The budget is a hard limit:
    a source that does not fit is truncated to the remaining budget (with a
    ``[truncated]`` marker) instead of being included whole, so one oversized
    document can never blow past the configured budget. Sources with no cached
    text in the ContentRegistry are skipped (not counted in ``total_count``).

    Args:
        notebook_id: the notebook whose sources are selected.
        max_chars: the character budget for the assembled context.

    Returns:
        A ``SourceSelection`` describing the selected texts and counts.
        ``total_count`` is the number of sources with cached text; ``used_count``
        is how many of those fit within the budget.
    """
    sources = db.session.scalars(
        select(Source)
        .where(
            Source.notebook_id == notebook_id,
            Source.status.in_(["ready", "partial"]),
        )
        .order_by(Source.created_at)
    ).all()

    # Resolve cached text for each source in upload order. Sources without
    # cached text in the ContentRegistry are skipped entirely (not counted
    # in total_count, since they contribute nothing to the LLM context).
    available_texts: list[str] = []
    for src in sources:
        entry = content_registry_repo.get_by_hash(src.content_hash)
        if entry is not None and entry.extracted_text:
            available_texts.append(entry.extracted_text)

    total_count = len(available_texts)

    # Select as many sources as fit within the budget. The budget is hard:
    # a source that would overflow it is truncated to the remaining room
    # (with a marker) so oversized documents still contribute a head excerpt
    # without ever exceeding max_chars in total.
    texts: list[str] = []
    running = 0
    for text in available_texts:
        room = max_chars - running
        if room <= 0:
            break
        if len(text) > room:
            texts.append(text[:room] + "\n\n[truncated: source exceeds context budget]")
            running += room
            break
        texts.append(text)
        running += len(text)

    if total_count > len(texts):
        logger.info(
            "select_sources_within_budget: notebook=%d used %d of %d sources (%d chars, budget=%d)",
            notebook_id,
            len(texts),
            total_count,
            running,
            max_chars,
        )

    return SourceSelection(
        texts=texts,
        total_chars=running,
        used_count=len(texts),
        total_count=total_count,
    )
