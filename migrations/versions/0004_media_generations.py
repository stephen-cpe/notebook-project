"""media generation counters for stale-job detection

Revision ID: 0004_media_generations
Revises: 0003_registry_fingerprint
Create Date: 2026-09-25

Adds:
- notebooks.audio_generation / notebooks.video_generation (non-null, default 0):
  incremented on every launch/delete so a superseded background job can
  recognize its result as stale and discard it instead of resurrecting
  deleted files or overwriting a newer generation's output.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_media_generations"
down_revision: str | None = "0003_registry_fingerprint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notebooks",
        sa.Column("audio_generation", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "notebooks",
        sa.Column("video_generation", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("notebooks", "video_generation")
    op.drop_column("notebooks", "audio_generation")
