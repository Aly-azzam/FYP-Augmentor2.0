"""Add MediaPipe expert-reference columns to videos.

These columns are populated by the one-time expert preprocessing flow
(``app.services.expert_mediapipe_service``) and consumed by the learner
comparison pipeline. They are all nullable, so existing videos (including
learner uploads) are unaffected.

This migration is written to be safe against databases that were
originally created via ``Base.metadata.create_all`` (and therefore
already contain some or none of these columns): each ``ADD COLUMN`` is
guarded by a live-inspector check so re-running is a no-op.

Revision ID: 20260421_0001
Revises:
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260421_0001"
down_revision = None
branch_labels = None
depends_on = None


_TABLE_NAME = "videos"

_NEW_COLUMNS: tuple[tuple[str, sa.types.TypeEngine, bool], ...] = (
    ("mediapipe_source_path", sa.Text(), True),
    ("mediapipe_detections_path", sa.Text(), True),
    ("mediapipe_features_path", sa.Text(), True),
    ("mediapipe_metadata_path", sa.Text(), True),
    ("mediapipe_status", sa.String(length=20), True),
    ("mediapipe_processed_at", sa.DateTime(timezone=True), True),
    ("mediapipe_pipeline_version", sa.String(length=50), True),
    ("mediapipe_fps", sa.Numeric(), True),
    ("mediapipe_frame_count", sa.Integer(), True),
    ("mediapipe_detection_rate", sa.Numeric(), True),
    ("mediapipe_selected_hand_policy", sa.String(length=50), True),
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
