"""Audit/recover DB registry rows from existing storage artifacts.

Usage:
  python -m app.scripts.recover_storage_registry
  python -m app.scripts.recover_storage_registry --apply --create-missing-chapters
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.video import Video


RECOVERY_COURSE_TITLE = "Recovered Course (storage re-registered)"


@dataclass
class ExpertStorageRecord:
    chapter_id: str
    mediapipe_folder: Optional[Path]
    sam2_folder: Optional[Path]
    mediapipe_metadata: Optional[dict[str, Any]]
    sam2_metadata: Optional[dict[str, Any]]
    source_path: Optional[Path]


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _path_from_metadata(metadata: Optional[dict[str, Any]], key: str) -> Optional[Path]:
    if metadata is None:
        return None
    raw = metadata.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def _coerce_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _relative_storage_key(path: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve()).as_posix()
    except ValueError:
        return None


def _scan_expert_storage() -> list[ExpertStorageRecord]:
    storage_root = Path(settings.STORAGE_ROOT)
    expert_mp_root = storage_root / "expert" / "mediapipe"
    expert_s2_root = storage_root / "expert" / "sam2"

    chapter_ids: set[str] = set()
    for root in (expert_mp_root, expert_s2_root):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                chapter_ids.add(child.name)

    records: list[ExpertStorageRecord] = []
    for chapter_id in sorted(chapter_ids):
        mp_folder = expert_mp_root / chapter_id
        s2_folder = expert_s2_root / chapter_id

        mp_folder = mp_folder if mp_folder.is_dir() else None
        s2_folder = s2_folder if s2_folder.is_dir() else None

        mp_meta = _load_json(mp_folder / "metadata.json") if mp_folder else None
        s2_meta = _load_json(s2_folder / "metadata.json") if s2_folder else None

        candidate_paths: list[Path] = []
        for metadata in (mp_meta, s2_meta):
            src = _path_from_metadata(metadata, "source_video_path")
            if src is not None:
                candidate_paths.append(src)
        if mp_folder is not None:
            candidate_paths.append(mp_folder / "source.mp4")
        if s2_folder is not None:
            candidate_paths.append(s2_folder / "source.mp4")

        source_path = next((p for p in candidate_paths if p.is_file()), None)

        records.append(
            ExpertStorageRecord(
                chapter_id=chapter_id,
                mediapipe_folder=mp_folder,
                sam2_folder=s2_folder,
                mediapipe_metadata=mp_meta,
                sam2_metadata=s2_meta,
                source_path=source_path,
            )
        )
    return records


def _ensure_recovery_course(session) -> Course:
    course = session.execute(
        select(Course).where(Course.title == RECOVERY_COURSE_TITLE)
    ).scalar_one_or_none()
    if course is not None:
        return course

    course = Course(title=RECOVERY_COURSE_TITLE, description="Auto-created by storage recovery script.")
    session.add(course)
    session.flush()
    return course


def _ensure_chapter(session, chapter_id: str, *, create_missing: bool) -> Optional[Chapter]:
    chapter = session.execute(select(Chapter).where(Chapter.id == chapter_id)).scalar_one_or_none()
    if chapter is not None:
        return chapter
    if not create_missing:
        return None

    course = _ensure_recovery_course(session)
    max_order = session.execute(
        select(Chapter.chapter_order)
        .where(Chapter.course_id == course.id)
        .order_by(Chapter.chapter_order.desc())
        .limit(1)
    ).scalar_one_or_none()
    next_order = int(max_order or 0) + 1

    chapter = Chapter(
        id=chapter_id,
        course_id=course.id,
        title=f"Recovered Chapter {chapter_id[:8]}",
        description="Auto-created by storage recovery script.",
        chapter_order=next_order,
    )
    session.add(chapter)
    session.flush()
    return chapter


def _upsert_expert_video(session, record: ExpertStorageRecord, *, create_missing_chapters: bool) -> str:
    chapter = _ensure_chapter(session, record.chapter_id, create_missing=create_missing_chapters)
    if chapter is None:
        return "skip_missing_chapter"
    if record.source_path is None:
        return "skip_missing_source_video"

    source_storage_key = _relative_storage_key(record.source_path)
    if source_storage_key is None:
        return "skip_source_outside_storage_root"

    video = session.execute(
        select(Video).where(
            Video.chapter_id == record.chapter_id,
            Video.video_role == "expert",
        )
    ).scalar_one_or_none()
    created = video is None
    if created:
        video = Video(
            chapter_id=record.chapter_id,
            owner_user_id=None,
            video_role="expert",
            source_video_id=None,
            file_path=source_storage_key,
            file_name=record.source_path.name,
            mime_type="video/mp4",
            file_size_bytes=record.source_path.stat().st_size,
            storage_provider="local",
        )
        session.add(video)
    else:
        video.file_path = source_storage_key
        video.file_name = record.source_path.name
        video.mime_type = "video/mp4"
        video.file_size_bytes = record.source_path.stat().st_size

    if record.mediapipe_folder is not None:
        mp = record.mediapipe_metadata or {}
        video.mediapipe_source_path = source_storage_key
        mp_key = _relative_storage_key(record.mediapipe_folder / "detections.json")
        video.mediapipe_detections_path = mp_key if (record.mediapipe_folder / "detections.json").is_file() else None
        mp_key = _relative_storage_key(record.mediapipe_folder / "features.json")
        video.mediapipe_features_path = mp_key if (record.mediapipe_folder / "features.json").is_file() else None
        mp_key = _relative_storage_key(record.mediapipe_folder / "metadata.json")
        video.mediapipe_metadata_path = mp_key if (record.mediapipe_folder / "metadata.json").is_file() else None
        mp_key = _relative_storage_key(record.mediapipe_folder / "annotated.mp4")
        video.mediapipe_annotated_path = mp_key if (record.mediapipe_folder / "annotated.mp4").is_file() else None
        video.mediapipe_status = "completed" if video.mediapipe_features_path else "failed"
        video.mediapipe_processed_at = _coerce_datetime(mp.get("created_at"))
        video.mediapipe_pipeline_version = "recovered"
        video.mediapipe_fps = _coerce_decimal(mp.get("fps"))
        video.mediapipe_frame_count = (
            int(mp.get("frame_count")) if isinstance(mp.get("frame_count"), (int, float)) else None
        )
        video.mediapipe_detection_rate = _coerce_decimal(mp.get("detection_rate"))
        if isinstance(mp.get("selected_hand_policy"), str):
            video.mediapipe_selected_hand_policy = mp.get("selected_hand_policy")

    if record.sam2_folder is not None:
        s2 = record.sam2_metadata or {}
        video.sam2_source_path = source_storage_key
        s2_key = _relative_storage_key(record.sam2_folder / "raw.json")
        video.sam2_raw_path = s2_key if (record.sam2_folder / "raw.json").is_file() else None
        s2_key = _relative_storage_key(record.sam2_folder / "summary.json")
        video.sam2_summary_path = s2_key if (record.sam2_folder / "summary.json").is_file() else None
        s2_key = _relative_storage_key(record.sam2_folder / "metadata.json")
        video.sam2_metadata_path = s2_key if (record.sam2_folder / "metadata.json").is_file() else None
        s2_key = _relative_storage_key(record.sam2_folder / "annotated.mp4")
        video.sam2_annotated_path = s2_key if (record.sam2_folder / "annotated.mp4").is_file() else None
        video.sam2_status = "completed" if video.sam2_raw_path else "failed"
        video.sam2_processed_at = _coerce_datetime(s2.get("created_at"))
        video.sam2_pipeline_version = "recovered"
        if isinstance(s2.get("model_name"), str):
            video.sam2_model_name = s2.get("model_name")
        video.sam2_fps = _coerce_decimal(s2.get("fps"))
        video.sam2_frame_count = (
            int(s2.get("frame_count")) if isinstance(s2.get("frame_count"), (int, float)) else None
        )
        video.sam2_detection_rate = _coerce_decimal(s2.get("detection_rate"))

    if created:
        return "created_expert_video"
    return "updated_expert_video"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/recover storage-backed expert records.")
    parser.add_argument("--apply", action="store_true", help="Persist recovery changes to DB.")
    parser.add_argument(
        "--create-missing-chapters",
        action="store_true",
        help="When --apply is set, auto-create missing chapters under a recovery course.",
    )
    args = parser.parse_args()

    storage_root = Path(settings.STORAGE_ROOT)
    expert_records = _scan_expert_storage()
    mp_run_count = len(list((storage_root / "mediapipe" / "runs").glob("*/metadata.json")))
    sam2_run_count = len(list((storage_root / "sam2" / "runs").glob("*/metadata.json")))

    session = SessionLocal()
    try:
        db_counts = {
            "courses": session.query(Course).count(),
            "chapters": session.query(Chapter).count(),
            "videos": session.query(Video).count(),
            "expert_videos": session.query(Video).filter(Video.video_role == "expert").count(),
            "learner_videos": session.query(Video).filter(Video.video_role == "learner").count(),
        }

        expert_audit: list[dict[str, Any]] = []
        for record in expert_records:
            chapter_exists = (
                session.execute(select(Chapter.id).where(Chapter.id == record.chapter_id)).scalar_one_or_none()
                is not None
            )
            expert_row_exists = (
                session.execute(
                    select(Video.id).where(
                        Video.chapter_id == record.chapter_id,
                        Video.video_role == "expert",
                    )
                ).scalar_one_or_none()
                is not None
            )
            expert_audit.append(
                {
                    "chapter_id": record.chapter_id,
                    "chapter_exists_in_db": chapter_exists,
                    "expert_video_row_exists_in_db": expert_row_exists,
                    "mediapipe_folder_exists": record.mediapipe_folder is not None,
                    "sam2_folder_exists": record.sam2_folder is not None,
                    "source_video_exists": record.source_path is not None,
                    "source_video_path_on_disk": (
                        str(record.source_path) if record.source_path is not None else None
                    ),
                }
            )

        actions: list[dict[str, str]] = []
        db_counts_after_apply: Optional[dict[str, int]] = None
        if args.apply:
            for record in expert_records:
                outcome = _upsert_expert_video(
                    session,
                    record,
                    create_missing_chapters=args.create_missing_chapters,
                )
                actions.append({"chapter_id": record.chapter_id, "outcome": outcome})
            session.commit()
            db_counts_after_apply = {
                "courses": session.query(Course).count(),
                "chapters": session.query(Chapter).count(),
                "videos": session.query(Video).count(),
                "expert_videos": session.query(Video).filter(Video.video_role == "expert").count(),
                "learner_videos": session.query(Video).filter(Video.video_role == "learner").count(),
            }
        else:
            session.rollback()

        report = {
            "database_counts": db_counts,
            "storage_counts": {
                "expert_records_found": len(expert_records),
                "mediapipe_runtime_runs_found": mp_run_count,
                "sam2_runtime_runs_found": sam2_run_count,
            },
            "expert_audit": expert_audit,
            "apply_mode": bool(args.apply),
            "create_missing_chapters": bool(args.create_missing_chapters),
            "actions": actions,
            "database_counts_after_apply": db_counts_after_apply,
            "notes": [
                "This script only re-registers expert rows when a source video file still exists on disk.",
                "Learner runtime run folders under storage/mediapipe/runs and storage/sam2/runs are audit-only here.",
                "Run with --apply to persist changes; default mode is read-only.",
            ],
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        print(json.dumps(report, indent=2))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
