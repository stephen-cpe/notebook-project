"""SQLAlchemy ORM models.

Layered architecture note: models hold structure only (no business logic).
Relationships cascade so deleting a User removes their Notebooks, which in turn
remove their Sources and ChatMessages. ``ContentRegistry`` is global (no
user_id) to enable cross-user dedup (NFR-22, SRS §7).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .extensions import db

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_DISABLED = "disabled"

AUDIO_STATUS_NONE = "none"
AUDIO_STATUS_QUEUED = "queued"
AUDIO_STATUS_SCRIPTING = "scripting"
AUDIO_STATUS_SYNTHESIZING = "synthesizing"
AUDIO_STATUS_READY = "ready"
AUDIO_STATUS_FAILED = "failed"

VIDEO_STATUS_NONE = "none"
VIDEO_STATUS_QUEUED = "queued"
VIDEO_STATUS_SCRIPTING = "scripting"
VIDEO_STATUS_SYNTHESIZING = "synthesizing"
VIDEO_STATUS_READY = "ready"
VIDEO_STATUS_FAILED = "failed"

SOURCE_STATUS_QUEUED = "queued"
SOURCE_STATUS_EXTRACTING = "extracting"
SOURCE_STATUS_EMBEDDING = "embedding"
SOURCE_STATUS_READY = "ready"
SOURCE_STATUS_FAILED = "failed"
SOURCE_STATUS_PARTIAL = "partial"


class User(db.Model):  # type: ignore[name-defined, misc]
    """An application user (regular or admin)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=ROLE_USER)
    avatar: Mapped[str] = mapped_column(String(32), nullable=False, default="avatar-0.png")
    audio_speaker_a: Mapped[str] = mapped_column(String(32), nullable=False, default="Ava")
    audio_speaker_b: Mapped[str] = mapped_column(String(32), nullable=False, default="Andrew")
    video_speaker: Mapped[str] = mapped_column(String(32), nullable=False, default="Ava")
    voice_speaker: Mapped[str] = mapped_column(String(32), nullable=False, default="Ava")
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)

    notebooks: Mapped[list[Notebook]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


class Notebook(db.Model):  # type: ignore[name-defined, misc]
    """A named, user-owned collection of sources."""

    __tablename__ = "notebooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_questions: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_status: Mapped[str] = mapped_column(String(20), nullable=False, default=AUDIO_STATUS_NONE)
    audio_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_status: Mapped[str] = mapped_column(String(20), nullable=False, default=VIDEO_STATUS_NONE)
    video_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC), nullable=False, onupdate=lambda: datetime.now(UTC)
    )

    user: Mapped[User] = relationship(back_populates="notebooks")
    sources: Mapped[list[Source]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("length(name) >= 1", name="notebook_name_nonempty"),
        CheckConstraint("length(name) <= 120", name="notebook_name_length"),
        Index("ix_notebooks_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<Notebook id={self.id} name={self.name!r}>"


class Source(db.Model):  # type: ignore[name-defined, misc]
    """An uploaded file attached to a notebook."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    notebook_id: Mapped[int] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(String(10), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=SOURCE_STATUS_QUEUED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)

    notebook: Mapped[Notebook] = relationship(back_populates="sources")

    __table_args__ = (
        Index("ix_sources_notebook_hash", "notebook_id", "content_hash", unique=True),
        Index("ix_sources_notebook_id", "notebook_id"),
        Index("ix_sources_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<Source id={self.id} filename={self.filename!r} status={self.status!r}>"


class ChatMessage(db.Model):  # type: ignore[name-defined, misc]
    """A single user or assistant message in a notebook's chat history."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    notebook_id: Mapped[int] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)

    notebook: Mapped[Notebook] = relationship(back_populates="chat_messages")

    __table_args__ = (Index("ix_chat_notebook_created", "notebook_id", "created_at"),)

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role!r}>"


class ContentRegistry(db.Model):  # type: ignore[name-defined, misc]
    """Global hash -> (chroma collection, extracted text) map for dedup + recovery.

    Not user-scoped: the same file content can be reused across users.
    """

    __tablename__ = "content_registry"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    chroma_collection: Mapped[str] = mapped_column(String(80), nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)

    def __repr__(self) -> str:
        return f"<ContentRegistry hash={self.content_hash[:12]!r}...>"
