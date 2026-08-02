"""voice and audio/video error columns + content_hash index

Revision ID: 0002_voice_and_errors
Revises: 0001_initial
Create Date: 2026-07-28

Adds:
- users.voice_speaker (default 'Ava')
- notebooks.audio_error, notebooks.video_error (nullable text)
- chat_messages.metadata_json (nullable text)
- sources.content_hash index (for ref-counted cleanup queries)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_voice_and_errors"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Users: voice conversation speaker preference.
    op.add_column(
        "users",
        sa.Column("voice_speaker", sa.String(length=32), nullable=False, server_default="Ava"),
    )

    # Notebooks: persisted failure reasons for audio/video generation (P1-2.25).
    op.add_column("notebooks", sa.Column("audio_error", sa.Text(), nullable=True))
    op.add_column("notebooks", sa.Column("video_error", sa.Text(), nullable=True))

    # ChatMessages: voice/modality metadata (Task 1).
    op.add_column("chat_messages", sa.Column("metadata_json", sa.Text(), nullable=True))

    # Sources: index on content_hash for ref-counted cleanup (P0-1.3).
    op.create_index("ix_sources_content_hash", "sources", ["content_hash"])


def downgrade() -> None:
    op.drop_index("ix_sources_content_hash", table_name="sources")
    op.drop_column("chat_messages", "metadata_json")
    op.drop_column("notebooks", "video_error")
    op.drop_column("notebooks", "audio_error")
    op.drop_column("users", "voice_speaker")
