"""cached full-coverage section digests

Revision ID: 0006_source_digest
Revises: 0005_user_difficulty
Create Date: 2026-09-30

Adds:
- content_registry.section_digest (nullable TEXT): the stitched per-section
  LLM digest for a content hash, built once and reused across notebooks/users.
- content_registry.digest_pipeline (nullable VARCHAR(16)): the pipeline
  fingerprint the digest was built under; a mismatch triggers a rebuild so
  digests never silently mix settings versions.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_source_digest"
down_revision: str | None = "0005_user_difficulty"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "content_registry",
        sa.Column("section_digest", sa.Text(), nullable=True),
    )
    op.add_column(
        "content_registry",
        sa.Column("digest_pipeline", sa.String(16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("content_registry", "digest_pipeline")
    op.drop_column("content_registry", "section_digest")
