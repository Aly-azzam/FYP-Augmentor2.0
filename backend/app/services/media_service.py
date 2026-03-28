from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.core.constants import ALLOWED_VIDEO_EXTENSIONS
from app.models.chapter import Chapter
from app.models.expert_video import ExpertVideo


@dataclass
class ExpertVideoAsset:
    filename: str
    storage_key: str
    url: str
    chapter_id: Optional[str] = None
    expert_video_id: Optional[str] = None


def normalize_storage_key(file_path: str | Path) -> str:
    raw_path = Path(str(file_path).replace("\\", "/"))

    if raw_path.is_absolute():
        try:
            relative_path = raw_path.resolve().relative_to(settings.STORAGE_ROOT.resolve())
        except ValueError as exc:
            raise ValueError("File path is outside the configured storage root.") from exc
        return relative_path.as_posix()

    path_parts = [part for part in raw_path.parts if part not in ("", ".")]
    if path_parts and path_parts[0].lower() == "storage":
        path_parts = path_parts[1:]

    if not path_parts:
        raise ValueError("Storage key cannot be empty.")

    return Path(*path_parts).as_posix()


def storage_path_exists(storage_key: str) -> bool:
    return (settings.STORAGE_ROOT / storage_key).exists()


def build_storage_url(file_path: str | Path) -> str:
    storage_key = normalize_storage_key(file_path)
    return f"/storage/{quote(storage_key, safe='/')}"


def find_first_expert_video_file() -> Optional[Path]:
    expert_root = settings.STORAGE_ROOT / "expert"
    if not expert_root.exists() or not expert_root.is_dir():
        return None

    expert_files = sorted(
        file_path
        for file_path in expert_root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
    )
    return expert_files[0] if expert_files else None


def get_default_expert_video_asset(db: Session) -> Optional[ExpertVideoAsset]:
    result = db.execute(
        select(ExpertVideo)
        .options(joinedload(ExpertVideo.chapter))
        .join(Chapter, ExpertVideo.chapter_id == Chapter.id)
        .order_by(Chapter.order.asc(), ExpertVideo.created_at.asc())
    )
    expert_videos = result.scalars().all()

    for expert_video in expert_videos:
        try:
            storage_key = normalize_storage_key(expert_video.file_path)
        except ValueError:
            continue

        if not storage_path_exists(storage_key):
            continue

        return ExpertVideoAsset(
            filename=Path(storage_key).name,
            storage_key=storage_key,
            url=build_storage_url(storage_key),
            chapter_id=str(expert_video.chapter_id),
            expert_video_id=str(expert_video.id),
        )

    discovered_file = find_first_expert_video_file()
    if discovered_file is None:
        return None

    storage_key = normalize_storage_key(discovered_file)
    return ExpertVideoAsset(
        filename=discovered_file.name,
        storage_key=storage_key,
        url=build_storage_url(storage_key),
    )
