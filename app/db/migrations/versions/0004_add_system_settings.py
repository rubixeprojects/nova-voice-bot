"""add system_settings table

Revision ID: 0004_add_system_settings
Revises: 0003_drop_s3_add_raw_pdf
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_system_settings"
down_revision: str | None = "0003_drop_s3_add_raw_pdf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "system_settings",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.execute(
        "INSERT INTO system_settings (key, value) VALUES ('active_language', 'as')"
    )


def downgrade() -> None:
    op.drop_table("system_settings")
