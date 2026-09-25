"""content_registry embedding fingerprint for versioned collections

Revision ID: 0003_registry_fingerprint
Revises: 0002_voice_and_errors
Create Date: 2026-09-25

Adds:
- content_registry.embedding_fingerprint (non-null, default ''): records which
  embedding backend (provider/model/dim/chunker) a content hash was embedded
  with. A mismatch triggers re-embedding into a versioned collection instead
  of silently mixing incompatible vector spaces.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_registry_fingerprint"
down_revision: str | None = "0002_voice_and_errors"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "content_registry",
        sa.Column("embedding_fingerprint", sa.String(length=16), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("content_registry", "embedding_fingerprint")
