"""Content cleanup service — reference-counted deletion of orphaned content.

When a Source is deleted, the underlying ChromaDB collection,
ContentRegistry entry, and cached figure thumbnails are shared across
notebooks/users (dedup, NFR-22). They must only be removed when no remaining
Source references the same ``content_hash``. This service centralizes that
reference-counted cleanup so the delete routes stay thin.

Functions never raise: cleanup is best-effort and logged on failure so a
storage hiccup never blocks a user-facing delete.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from src.repositories import content_registry_repo, source_repo
from src.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def cleanup_orphaned_content(content_hash: str, exclude_source_id: int | None = None) -> bool:
    """Remove Chroma collection + ContentRegistry row + figures if no Sources remain.

    Args:
        content_hash: the shared content hash to check.
        exclude_source_id: a Source row being deleted (excluded from the count).

    Returns:
        True if the orphaned content was removed, False if other Sources still
        reference the hash (or on best-effort failure).
    """
    remaining = source_repo.count_by_content_hash(content_hash, exclude_source_id=exclude_source_id)
    if remaining > 0:
        logger.debug(
            "cleanup_orphaned_content: hash %s still referenced by %d source(s); keeping.",
            content_hash[:12],
            remaining,
        )
        return False

    # No remaining references — delete the Chroma collection first. The
    # registry row holds the only rebuild copy, so it is removed only after
    # the vector deletion is confirmed. A failed vector deletion keeps the
    # registry (and returns False) so a later cleanup can retry instead of
    # destroying the only recovery copy.
    try:
        vector_deleted = get_vector_store().delete_collection(content_hash)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup_orphaned_content: delete_collection failed: %s", exc)
        vector_deleted = False
    if not vector_deleted:
        logger.warning(
            "cleanup_orphaned_content: keeping registry for hash %s; "
            "vector deletion unconfirmed, will retry on next cleanup.",
            content_hash[:12],
        )
        return False

    try:
        content_registry_repo.delete_entry(content_hash)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup_orphaned_content: delete_entry failed: %s", exc)
        return False

    # The registry row was the last shared reference: cached figure
    # thumbnails for this hash are orphaned too. Best-effort; a failure here
    # must not undo the confirmed vector/registry cleanup above.
    _remove_figure_dir(content_hash)

    logger.info("cleanup_orphaned_content: removed orphaned content for hash %s", content_hash[:12])
    return True


def _remove_figure_dir(content_hash: str) -> None:
    """Best-effort removal of ``DATA_DIR/figures/<hash>/`` (never raises)."""
    try:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", content_hash or ""):
            return
        from src.config import Config

        figures_dir = Path(Config().data_dir) / "figures" / content_hash
        if figures_dir.exists():
            shutil.rmtree(figures_dir, ignore_errors=True)
            logger.info("cleanup_orphaned_content: removed figures for hash %s", content_hash[:12])
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup_orphaned_content: figure cleanup failed: %s", exc)


def cleanup_notebook_media(notebook_id: int, data_dir: str) -> None:
    """Delete a notebook's audio and video files (notebook-specific, not shared).

    Called on notebook deletion so the on-disk MP3/MP4 artifacts don't leak.
    Best-effort: missing files are ignored.
    """
    for sub in ("audio", "video", "voice"):
        media_dir = Path(data_dir) / sub / str(notebook_id)
        if media_dir.exists():
            # Recursive: job temp dirs (tmp_<jobid>/) live under these and a
            # plain rmdir would silently leave them behind.
            shutil.rmtree(media_dir, ignore_errors=True)
            logger.info("cleanup_notebook_media: removed %s", media_dir)


def cleanup_notebook_orphaned_content(hashes: list[str]) -> None:
    """After a notebook's sources are cascade-deleted, remove orphaned content.

    ``hashes`` is the snapshot of content hashes the notebook used (captured
    BEFORE the notebook row was deleted via ``source_repo.list_hashes_by_notebook``).
    For each hash, removes the Chroma/registry entries if no other notebook's
    sources still reference them.
    """
    for h in hashes:
        cleanup_orphaned_content(h)
