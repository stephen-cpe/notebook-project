"""Content registry repository — DB access for the ContentRegistry table.

The ContentRegistry maps a content hash to its ChromaDB collection name and
cached extracted text, enabling cross-user dedup and corruption recovery.
"""

from __future__ import annotations

from src.extensions import db
from src.models import ContentRegistry


def create_entry(
    content_hash: str,
    chroma_collection: str,
    extracted_text: str,
    char_count: int,
    embedding_fingerprint: str = "",
) -> ContentRegistry:
    """Insert a new registry entry and return it."""
    entry = ContentRegistry(
        content_hash=content_hash,
        chroma_collection=chroma_collection,
        embedding_fingerprint=embedding_fingerprint,
        extracted_text=extracted_text,
        char_count=char_count,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


def get_by_hash(content_hash: str) -> ContentRegistry | None:
    """Fetch a registry entry by content hash (PK)."""
    return db.session.get(ContentRegistry, content_hash)


def update_fingerprint(
    entry: ContentRegistry, embedding_fingerprint: str, chroma_collection: str
) -> ContentRegistry:
    """Record a re-embedding of ``entry`` under a new backend fingerprint."""
    entry.embedding_fingerprint = embedding_fingerprint
    entry.chroma_collection = chroma_collection
    db.session.commit()
    return entry


def get_or_create(
    content_hash: str,
    chroma_collection: str,
    extracted_text: str,
    char_count: int,
    embedding_fingerprint: str = "",
) -> ContentRegistry:
    """Return an existing entry, or create one if it does not exist.

    Race-safe: concurrent inserts on the primary key are handled by catching
    ``IntegrityError``, rolling back, and re-fetching the existing row inserted
    by the winning transaction. A stale embedding fingerprint on an existing
    row is refreshed so later reads know which backend version the vectors
    belong to.
    """
    existing = get_by_hash(content_hash)
    if existing is not None:
        if embedding_fingerprint and existing.embedding_fingerprint != embedding_fingerprint:
            return update_fingerprint(existing, embedding_fingerprint, chroma_collection)
        return existing
    try:
        return create_entry(
            content_hash, chroma_collection, extracted_text, char_count, embedding_fingerprint
        )
    except Exception as exc:  # noqa: BLE001
        # IntegrityError (PK conflict) on a race; re-fetch the winner's row.
        from sqlalchemy.exc import IntegrityError

        if isinstance(exc, IntegrityError):
            db.session.rollback()
            existing = get_by_hash(content_hash)
            if existing is not None:
                if (
                    embedding_fingerprint
                    and existing.embedding_fingerprint != embedding_fingerprint
                ):
                    return update_fingerprint(existing, embedding_fingerprint, chroma_collection)
                return existing
        raise


def delete_entry(content_hash: str) -> None:
    """Delete a registry entry (no error if missing)."""
    entry = get_by_hash(content_hash)
    if entry is not None:
        db.session.delete(entry)
        db.session.commit()
