"""Token/context budgeting for the RAG pipeline.

Retrieval depth scales with the LLM context window instead of hardcoded
``top_k`` values, so selecting a larger-context model automatically raises
the retrieval budget without code changes.

Budgeting model (ported from study-and-learn ``rag_budget.py``)::

    reserved  = num_ctx * 0.20      (prompt template + generated output)
    budget    = num_ctx - reserved  (tokens available for retrieved context)
    max_chars = budget * 4          (~4 chars per token for English prose)

All functions are pure (no I/O) except the ``Config``-reading wrappers.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

AVG_CHARS_PER_TOKEN = 4
RESERVED_TOKEN_FRACTION = 0.20


def get_num_ctx(config: Config | None = None) -> int:
    """Return the configured LLM context window in tokens (256K default)."""
    if config is not None:
        return int(config.ollama_num_ctx)
    try:
        return int(os.environ.get("OLLAMA_NUM_CTX", "262144"))
    except (TypeError, ValueError):
        return 262144


def _budget_chars(num_ctx: int | None = None, config: Config | None = None) -> int:
    """Return the character budget available for retrieved context."""
    ctx = num_ctx if num_ctx is not None else get_num_ctx(config)
    return int(ctx * (1.0 - RESERVED_TOKEN_FRACTION) * AVG_CHARS_PER_TOKEN)


def get_context_budget_chars(
    fraction: float = 1.0,
    *,
    max_chars_cap: int | None = None,
    num_ctx: int | None = None,
    config: Config | None = None,
) -> int:
    """Return the character budget for retrieved context.

    Args:
        fraction: fraction of the available budget to use (1.0 = all).
        max_chars_cap: hard ceiling; when None, read from config
            (``rag_max_context_chars``) or env ``RAG_MAX_CONTEXT_CHARS``.
        num_ctx: override for the context window (used by tests).
        config: optional ``Config`` for cap + num_ctx defaults.
    """
    budget = _budget_chars(num_ctx, config) * max(0.0, min(fraction, 1.0))
    if max_chars_cap is None:
        if config is not None:
            max_chars_cap = int(config.rag_max_context_chars)
        else:
            try:
                max_chars_cap = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "700000"))
            except (TypeError, ValueError):
                max_chars_cap = 700000
    return int(min(budget, max_chars_cap))


def get_top_k_for_budget(
    *,
    per_collection_top_k: int | None = None,
    default_top_k: int = 20,
    avg_chunk_chars: int = 1000,
    num_ctx: int | None = None,
    config: Config | None = None,
) -> int:
    """Return a retrieval ``top_k`` matching the context budget.

    The result is the larger of the configured default and the number of
    average-sized chunks fitting in the (uncapped) context budget, so a
    small default is raised automatically for large-context models but an
    explicitly large admin value is never lowered.
    """
    if per_collection_top_k is None:
        if config is not None:
            per_collection_top_k = int(config.rag_top_k)
        else:
            try:
                per_collection_top_k = int(os.environ.get("RAG_TOP_K", str(default_top_k)))
            except (TypeError, ValueError):
                per_collection_top_k = default_top_k
    budget_chars = _budget_chars(num_ctx, config)
    budget_top_k = max(1, budget_chars // max(1, avg_chunk_chars))
    return max(int(per_collection_top_k), int(budget_top_k))


def per_collection_k_for(top_k: int, n_sources: int, floor: int = 12) -> int:
    """Scale per-source retrieval depth so big single docs can fill the window.

    A single giant PDF needs deep per-collection reads (up to ``top_k``),
    while 50 small sources each contribute a few of their best chunks. The
    global ``top_k`` merge still bounds the final prompt.
    """
    if top_k <= 0:
        return floor
    share = top_k // max(1, n_sources) + 2
    return max(floor, min(top_k, share))


def truncate_results_to_budget(
    results: list[dict[str, object]], max_chars: int
) -> tuple[list[dict[str, object]], int]:
    """Keep whole result chunks until ``max_chars`` would overflow.

    Whole chunks only (never a cut-off chunk — citations stay clean).
    Returns ``(kept_results, kept_chars)``.
    """
    if max_chars <= 0:
        return [], 0
    kept: list[dict[str, object]] = []
    running = 0
    for r in results:
        text = str(r.get("text", r.get("document", "")))
        if running + len(text) > max_chars:
            break
        kept.append(r)
        running += len(text)
    return kept, running


def estimate_coverage_ratio(retrieved_chars: int, total_chars: int) -> float:
    """Return ``retrieved / total`` clamped to [0.0, 1.0] (1.0 when empty)."""
    if total_chars <= 0:
        return 1.0
    return max(0.0, min(1.0, retrieved_chars / float(total_chars)))
