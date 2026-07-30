"""Content registry repository — DB access for the ContentRegistry table.

The ContentRegistry maps a content hash to its ChromaDB collection name and
cached extracted text, enabling cross-user dedup and corruption recovery.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from src.extensions import db
from src.models import ContentRegistry


def create_entry(
    content_hash: str,
    chroma_collection: str,
    extracted_text: str,
    char_count: int,
) -> ContentRegistry:
    """Insert a new registry entry and return it."""
    entry = ContentRegistry(
        content_hash=content_hash,
        chroma_collection=chroma_collection,
        extracted_text=extracted_text,
        char_count=char_count,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


def get_by_hash(content_hash: str) -> ContentRegistry | None:
    """Fetch a registry entry by content hash (PK)."""
    return db.session.get(ContentRegistry, content_hash)


def get_or_create(
    content_hash: str,
    chroma_collection: str,
    extracted_text: str,
    char_count: int,
) -> ContentRegistry:
    """Return an existing entry, or create one if it does not exist.

    Race-safe (P0-1.11): concurrent inserts on the primary key are handled by
    catching ``IntegrityError``, rolling back, and re-fetching the existing
    row inserted by the winning transaction.
    """
    existing = get_by_hash(content_hash)
    if existing is not None:
        return existing
    try:
        return create_entry(content_hash, chroma_collection, extracted_text, char_count)
    except Exception as exc:  # noqa: BLE001
        # IntegrityError (PK conflict) on a race; re-fetch the winner's row.
        from sqlalchemy.exc import IntegrityError

        if isinstance(exc, IntegrityError):
            db.session.rollback()
            existing = get_by_hash(content_hash)
            if existing is not None:
                return existing
        raise


def list_all() -> Sequence[ContentRegistry]:
    """Return all registry entries (admin/diagnostic view)."""
    return db.session.scalars(select(ContentRegistry).order_by(ContentRegistry.created_at)).all()


def delete_entry(content_hash: str) -> None:
    """Delete a registry entry (no error if missing)."""
    entry = get_by_hash(content_hash)
    if entry is not None:
        db.session.delete(entry)
        db.session.commit()
