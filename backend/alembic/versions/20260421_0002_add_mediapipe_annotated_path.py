"""Add ``mediapipe_annotated_path`` to videos.

Tracks the stable location of the annotated expert video produced by the
one-time expert preprocessing flow (rendered into
``storage/expert/mediapipe/{expert_code}/annotated.mp4``).

Revision ID: 20260421_0002
Revises: 20260421_0001
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260421_0002"
down_revision = "20260421_0001"
branch_labels = None
depends_on = None


_TABLE_NAME = "videos"
_COLUMN_NAME = "mediapipe_annotated_path"


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(_TABLE_NAME)}


def upgrade() -> None:
    if _COLUMN_NAME in _existing_columns():
        return
    op.add_column(
        _TABLE_NAME,
        sa.Column(_COLUMN_NAME, sa.Text(), nullable=True),
    )


def downgrade() -> None:
    if _COLUMN_NAME not in _existing_columns():
        return
    op.drop_column(_TABLE_NAME, _COLUMN_NAME)
