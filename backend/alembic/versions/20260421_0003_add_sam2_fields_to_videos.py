"""Add SAM 2 expert-reference columns to videos.

These columns are populated by the one-time expert SAM 2 preprocessing
flow (``app.services.expert_sam2_service``) and consumed by the learner
comparison pipeline. They mirror the MediaPipe columns added in
``20260421_0001_add_mediapipe_fields_to_videos.py`` and are all nullable
so existing rows (learner uploads, unprocessed experts) are unaffected.

Each ``ADD COLUMN`` is guarded by a live inspector check so re-running
is a no-op — same defensive style as the MediaPipe migration.

Revision ID: 20260421_0003
Revises: 20260421_0002
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260421_0003"
down_revision = "20260421_0002"
branch_labels = None
depends_on = None


_TABLE_NAME = "videos"

_NEW_COLUMNS: tuple[tuple[str, sa.types.TypeEngine, bool], ...] = (
    ("sam2_source_path", sa.Text(), True),
    ("sam2_raw_path", sa.Text(), True),
    ("sam2_summary_path", sa.Text(), True),
    ("sam2_metadata_path", sa.Text(), True),
    ("sam2_annotated_path", sa.Text(), True),
    ("sam2_status", sa.String(length=20), True),
    ("sam2_processed_at", sa.DateTime(timezone=True), True),
    ("sam2_pipeline_version", sa.String(length=50), True),
    ("sam2_model_name", sa.String(length=100), True),
    ("sam2_fps", sa.Numeric(), True),
    ("sam2_frame_count", sa.Integer(), True),
    ("sam2_detection_rate", sa.Numeric(), True),
)


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(_TABLE_NAME)}


def upgrade() -> None:
    existing = _existing_columns()
    for column_name, column_type, nullable in _NEW_COLUMNS:
        if column_name in existing:
            continue
        op.add_column(
            _TABLE_NAME,
            sa.Column(column_name, column_type, nullable=nullable),
        )


def downgrade() -> None:
    existing = _existing_columns()
    for column_name, _column_type, _nullable in reversed(_NEW_COLUMNS):
        if column_name not in existing:
            continue
        op.drop_column(_TABLE_NAME, column_name)
