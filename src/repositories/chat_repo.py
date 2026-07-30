"""Chat message repository — DB access for ChatMessage (no business logic)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from src.extensions import db
from src.models import ChatMessage


def create_message(
    notebook_id: int,
    role: str,
    content: str,
    sources_json: str | None = None,
    latency_ms: int | None = None,
) -> ChatMessage:
    """Insert a chat message and return it."""
    msg = ChatMessage(
        notebook_id=notebook_id,
        role=role,
        content=content,
        sources_json=sources_json,
        latency_ms=latency_ms,
    )
    db.session.add(msg)
    db.session.commit()
    return msg


def list_by_notebook(notebook_id: int, limit: int = 100) -> Sequence[ChatMessage]:
    """Return chat history for a notebook, oldest first."""
    return db.session.scalars(
        select(ChatMessage)
        .where(ChatMessage.notebook_id == notebook_id)
        .order_by(ChatMessage.created_at)
        .limit(limit)
    ).all()


def delete_by_notebook(notebook_id: int) -> int:
    """Delete all chat messages for a notebook. Returns count deleted."""
    count = db.session.query(ChatMessage).filter(ChatMessage.notebook_id == notebook_id).count()
    db.session.query(ChatMessage).filter(ChatMessage.notebook_id == notebook_id).delete()
    db.session.commit()
    return count
