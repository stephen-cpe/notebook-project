"""answer difficulty preference

Revision ID: 0005_user_difficulty
Revises: 0004_media_generations
Create Date: 2026-09-29

Adds:
- users.difficulty (non-null, default 'Normal'): controls answer reading
  level for chat/summary/audio/video (Easy/Normal/Hard).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_user_difficulty"
down_revision: str | None = "0004_media_generations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("difficulty", sa.String(16), nullable=False, server_default="Normal"),
    )


def downgrade() -> None:
    op.drop_column("users", "difficulty")
