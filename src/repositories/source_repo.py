"""Source repository — DB access for Source (no business logic)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from src.extensions import db
from src.models import Source


def create_source(
    notebook_id: int,
    filename: str,
    content_hash: str,
    content_type: str,
) -> Source:
    """Insert a new source row (status=queued) and return it."""
    src = Source(
        notebook_id=notebook_id,
        filename=filename,
        content_hash=content_hash,
        content_type=content_type,
    )
    db.session.add(src)
    db.session.commit()
    return src


def get_by_id(source_id: int) -> Source | None:
    """Fetch a source by primary key."""
    return db.session.get(Source, source_id)


def list_by_notebook(notebook_id: int) -> Sequence[Source]:
    """Return all sources for a notebook, newest first."""
    return db.session.scalars(
        select(Source).where(Source.notebook_id == notebook_id).order_by(Source.created_at.desc())
    ).all()


def update_status(
    source: Source,
    status: str,
    char_count: int | None = None,
    page_count: int | None = None,
    error_message: str | None = None,
) -> Source:
    """Update the ingestion status + metadata of a source."""
    source.status = status
    if char_count is not None:
        source.char_count = char_count
    if page_count is not None:
        source.page_count = page_count
    if error_message is not None:
        source.error_message = error_message
    db.session.commit()
    return source


def delete_source(source: Source) -> None:
    """Delete a source row."""
    db.session.delete(source)
    db.session.commit()


def rename_source(source: Source, new_filename: str) -> Source:
    """Update the display filename of a source (does not re-ingest)."""
    source.filename = new_filename
    db.session.commit()
    return source


def get_by_notebook_and_hash(notebook_id: int, content_hash: str) -> Source | None:
    """Return the source for a (notebook_id, content_hash) pair, if it exists."""
    return db.session.scalar(
        select(Source).where(
            Source.notebook_id == notebook_id,
            Source.content_hash == content_hash,
        )
    )


def count_by_notebook(notebook_id: int) -> int:
    """Return the number of sources in a notebook."""
    return db.session.query(Source).filter(Source.notebook_id == notebook_id).count()


def count_by_content_hash(content_hash: str, exclude_source_id: int | None = None) -> int:
    """Count sources referencing ``content_hash`` (for ref-counted cleanup).

    Optionally exclude ``exclude_source_id`` (the source being deleted) so the
    count reflects remaining references after deletion.
    """
    q = db.session.query(Source).filter(Source.content_hash == content_hash)
    if exclude_source_id is not None:
        q = q.filter(Source.id != exclude_source_id)
    return q.count()


def list_hashes_by_notebook(notebook_id: int) -> list[str]:
    """Return the distinct content hashes used by a notebook's sources.

    Used by notebook deletion to know which hashes may need cleanup after the
    notebook's sources are cascade-deleted.
    """
    rows = db.session.query(Source.content_hash).filter(Source.notebook_id == notebook_id).all()
    return [r[0] for r in rows]
