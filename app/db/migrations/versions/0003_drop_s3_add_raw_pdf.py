"""drop s3 columns, add raw_pdf bytea

Revision ID: 0003_drop_s3_add_raw_pdf
Revises: 0002_chunk_metadata_v2
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_drop_s3_add_raw_pdf"
down_revision: str | None = "0002_chunk_metadata_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("raw_pdf", sa.LargeBinary(), nullable=True))
    op.drop_column("documents", "s3_bucket")
    op.drop_column("documents", "s3_key")


def downgrade() -> None:
    op.add_column("documents", sa.Column("s3_bucket", sa.Text(), nullable=False, server_default=""))
    op.add_column("documents", sa.Column("s3_key", sa.Text(), nullable=False, server_default=""))
    op.drop_column("documents", "raw_pdf")
