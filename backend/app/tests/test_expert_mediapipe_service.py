"""Tests for the expert MediaPipe preprocessing service.

These tests avoid running the real MediaPipe pipeline (which needs the
hand_landmarker model + opencv + a real video) by patching
``run_pipeline``. They exercise the orchestration logic:
    * folder creation + safety
    * source video copy
    * metadata parsing
    * DB update path (via a stubbed Session)
    * reuse behavior on already-completed references
    * failure cleanup + status rollback
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.chapter import Chapter
from app.models.video import Video
from app.schemas.mediapipe.mediapipe_schema import MediaPipeRunMeta
from app.services import expert_mediapipe_service as svc
from app.services.mediapipe.run_service import (
    ANNOTATED_VIDEO_FILENAME,
    DETECTIONS_FILENAME,
    FEATURES_FILENAME,
    METADATA_FILENAME,
    MediaPipeRunArtifacts,
    MediaPipeRunError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect settings.STORAGE_ROOT to a fresh temp directory."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    # Patch both the Settings instance and the module-level reference used
    # inside expert_mediapipe_service / run_service.
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "STORAGE_ROOT", storage_root)
    return storage_root


@pytest.fixture
def sample_source_video(tmp_path: Path) -> Path:
    path = tmp_path / "raw_expert.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake_video_bytes")
    return path


def _make_expert_video(
    *,
    file_path: str,
    chapter_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Video:
    video = Video(
        id=str(uuid4()),
        owner_user_id=None,
        chapter_id=chapter_id,
        video_role="expert",
        source_video_id=None,
        file_path=file_path,
        file_name="expert.mp4",
        mime_type="video/mp4",
        file_size_bytes=1024,
        storage_provider="local",
    )
    if status is not None:
        video.mediapipe_status = status
    return video


def _fake_metadata(run_id: str, source_video_path: str) -> MediaPipeRunMeta:
    return MediaPipeRunMeta(
        run_id=run_id,
        source_video_path=source_video_path,
        fps=30.0,
        frame_count=90,
        width=640,
        height=480,
        created_at="2026-04-21T00:00:00+00:00",
        selected_hand_policy="prefer_right_else_first",
        total_frames=90,
        frames_with_detection=81,
        detection_rate=0.9,
        right_hand_selected_count=81,
        left_hand_selected_count=0,
    )


def _write_fake_run_outputs(
    run_dir: Path,
    metadata: MediaPipeRunMeta,
    *,
    include_annotated: bool = True,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    detections_path = run_dir / DETECTIONS_FILENAME
    features_path = run_dir / FEATURES_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    annotated_path = run_dir / ANNOTATED_VIDEO_FILENAME

    detections_path.write_text(
        json.dumps(
            {
                "run_id": metadata.run_id,
                "source_video_path": metadata.source_video_path,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "selected_hand_policy": metadata.selected_hand_policy,
                "frames": [],
            }
        )
    )
    features_path.write_text(
        json.dumps(
            {
                "run_id": metadata.run_id,
                "source_video_path": metadata.source_video_path,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "trajectory_history_size": 30,
                "frames": [],
            }
        )
    )
    metadata_path.write_text(metadata.model_dump_json())
    if include_annotated:
        annotated_path.write_bytes(b"fake-annotated-video-bytes")
    return {
        "detections": detections_path,
        "features": features_path,
        "metadata": metadata_path,
        "annotated": annotated_path,
    }


@dataclass
class _FakeRunPipelineCall:
    video_path: Path
    run_id: Optional[str]
    render_annotation: bool


class _RunPipelineRecorder:
    """Test double for ``run_pipeline`` that writes valid artifacts on disk."""

    def __init__(self, storage_root: Path, *, should_fail: bool = False):
        self.storage_root = storage_root
        self.calls: list[_FakeRunPipelineCall] = []
        self.should_fail = should_fail

    def __call__(
        self,
        video_path: str | Path,
        *,
        run_id: Optional[str] = None,
        render_annotation: bool = True,
        **_kwargs: Any,
    ) -> MediaPipeRunArtifacts:
        self.calls.append(
            _FakeRunPipelineCall(
                video_path=Path(video_path),
                run_id=run_id,
                render_annotation=render_annotation,
            )
        )

        if self.should_fail:
            raise MediaPipeRunError("synthetic pipeline failure")

        effective_run_id = run_id or "fake_run"
        run_dir = self.storage_root / "mediapipe" / "runs" / effective_run_id
        metadata = _fake_metadata(effective_run_id, str(video_path))
        paths = _write_fake_run_outputs(run_dir, metadata)

        return MediaPipeRunArtifacts(
            run_id=effective_run_id,
            run_dir=run_dir,
            detections_path=paths["detections"],
            features_path=paths["features"],
            metadata_path=paths["metadata"],
            annotated_video_path=paths["annotated"],
            source_video_path=Path(video_path),
            metadata=metadata,
        )


class _StubSession:
    """Minimal Session stub for the orchestration tests."""

    def __init__(self, video: Video, chapter: Optional[Chapter] = None):
        self.video = video
        self.chapter = chapter
        self.commit_count = 0
        self.rollback_count = 0
        self.added: list[Any] = []

    def execute(self, statement: Any):
        result = MagicMock()
        target_model: Any = None
        try:
            # Scan SQLAlchemy select's column descriptions to figure out
            # which model is being queried. This is enough for our stub.
            target_model = statement.column_descriptions[0]["entity"]
        except Exception:
            pass

        if target_model is Video:
            result.scalar_one_or_none.return_value = self.video
        elif target_model is Chapter:
            result.scalar_one_or_none.return_value = self.chapter
        else:
            result.scalar_one_or_none.return_value = None
        return result

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def refresh(self, _obj: Any) -> None:  # no-op for in-memory objects
        return None


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_ensure_expert_mediapipe_dir_creates_folder(tmp_storage_root: Path) -> None:
    folder = svc.ensure_expert_mediapipe_dir("expert_001")
    assert folder.is_dir()
    assert folder == tmp_storage_root / "expert" / "mediapipe" / "expert_001"


@pytest.mark.parametrize("bad_code", ["", "..", "a/b", "a\\b", "../escape"])
def test_ensure_expert_mediapipe_dir_rejects_unsafe_codes(
    tmp_storage_root: Path, bad_code: str
) -> None:
    with pytest.raises(svc.ExpertMediaPipeError):
        svc.ensure_expert_mediapipe_dir(bad_code)


def test_save_expert_source_video_copies_to_stable_folder(
    tmp_storage_root: Path, sample_source_video: Path
) -> None:
    expert_folder = svc.ensure_expert_mediapipe_dir("expert_copy")
    video = _make_expert_video(file_path=str(sample_source_video))

    destination = svc.save_expert_source_video(video, expert_folder)

    assert destination == expert_folder / svc.SOURCE_FILENAME
    assert destination.is_file()
    assert destination.read_bytes() == sample_source_video.read_bytes()


def test_load_mediapipe_metadata_parses_summary(
    tmp_storage_root: Path,
) -> None:
    expert_folder = svc.ensure_expert_mediapipe_dir("expert_meta")
    metadata = _fake_metadata("run_meta", str(expert_folder / "source.mp4"))
    metadata_path = expert_folder / METADATA_FILENAME
    metadata_path.write_text(metadata.model_dump_json())

    loaded = svc.load_mediapipe_metadata(metadata_path)
    assert loaded.fps == 30.0
    assert loaded.frame_count == 90
    assert loaded.detection_rate == pytest.approx(0.9)
    assert loaded.selected_hand_policy == "prefer_right_else_first"


def test_cleanup_on_failure_removes_partial_outputs_but_can_keep_source(
    tmp_storage_root: Path,
) -> None:
    expert_folder = svc.ensure_expert_mediapipe_dir("expert_cleanup")
    for filename in (
        svc.SOURCE_FILENAME,
        DETECTIONS_FILENAME,
        FEATURES_FILENAME,
        METADATA_FILENAME,
        ANNOTATED_VIDEO_FILENAME,
    ):
        (expert_folder / filename).write_text("partial")

    svc.cleanup_on_failure(expert_folder, keep_source=True)

    assert (expert_folder / svc.SOURCE_FILENAME).exists()
    assert not (expert_folder / DETECTIONS_FILENAME).exists()
    assert not (expert_folder / FEATURES_FILENAME).exists()
    assert not (expert_folder / METADATA_FILENAME).exists()
    assert not (expert_folder / ANNOTATED_VIDEO_FILENAME).exists()


# ---------------------------------------------------------------------------
# Orchestration tests (mocked run_pipeline)
# ---------------------------------------------------------------------------


def test_register_expert_mediapipe_reference_happy_path(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # We only need an object with `.title` for the response builder; a
    # SimpleNamespace avoids SQLAlchemy's instance-state requirements.
    chapter = SimpleNamespace(id=str(uuid4()), title="Suture Tying")

    video = _make_expert_video(
        file_path=str(sample_source_video),
        chapter_id=chapter.id,
    )
    session = _StubSession(video=video, chapter=chapter)

    recorder = _RunPipelineRecorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_pipeline", recorder)

    reference = svc.register_expert_mediapipe_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
    )

    expert_folder = tmp_storage_root / "expert" / "mediapipe" / str(video.id)
    assert expert_folder.is_dir()
    for filename in (
        svc.SOURCE_FILENAME,
        DETECTIONS_FILENAME,
        FEATURES_FILENAME,
        METADATA_FILENAME,
        ANNOTATED_VIDEO_FILENAME,
    ):
        assert (expert_folder / filename).is_file()

    assert len(recorder.calls) == 1
    assert recorder.calls[0].render_annotation is True
    assert reference.annotated_path is not None
    assert reference.annotated_path.endswith(ANNOTATED_VIDEO_FILENAME)
    assert video.mediapipe_annotated_path is not None

    assert reference.reused_existing is False
    assert reference.expert_code == str(video.id)
    assert reference.mediapipe_status == svc.STATUS_COMPLETED
    assert reference.summary.fps == 30.0
    assert reference.summary.detection_rate == pytest.approx(0.9)
    assert reference.summary.selected_hand_policy == "prefer_right_else_first"
    assert reference.title == "Suture Tying"

    # The DB row was updated with MediaPipe paths + summary.
    assert video.mediapipe_status == svc.STATUS_COMPLETED
    assert video.mediapipe_fps is not None and float(video.mediapipe_fps) == 30.0
    assert video.mediapipe_frame_count == 90
    assert video.mediapipe_detection_rate is not None
    assert video.mediapipe_processed_at is not None

    assert session.commit_count >= 2  # one for "processing", one for "completed"

    # Temporary runs folder was cleaned up.
    temp_run_dir = tmp_storage_root / "mediapipe" / "runs" / recorder.calls[0].run_id
    assert not temp_run_dir.exists()


def test_register_expert_mediapipe_reference_reuses_completed_outputs(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(
        file_path=str(sample_source_video),
        status=svc.STATUS_COMPLETED,
    )
    session = _StubSession(video=video)

    # Pre-populate the stable expert folder so the reuse path kicks in.
    expert_folder = svc.ensure_expert_mediapipe_dir(str(video.id))
    metadata = _fake_metadata("prebuilt", str(expert_folder / svc.SOURCE_FILENAME))
    (expert_folder / svc.SOURCE_FILENAME).write_bytes(b"pre-existing")
    _write_fake_run_outputs(expert_folder, metadata)

    recorder = _RunPipelineRecorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_pipeline", recorder)

    reference = svc.register_expert_mediapipe_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
        overwrite=False,
    )

    assert reference.reused_existing is True
    assert not recorder.calls  # pipeline was NOT invoked


def test_register_expert_mediapipe_reference_overwrite_reruns_pipeline(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(
        file_path=str(sample_source_video),
        status=svc.STATUS_COMPLETED,
    )
    session = _StubSession(video=video)

    expert_folder = svc.ensure_expert_mediapipe_dir(str(video.id))
    metadata = _fake_metadata("old_run", str(expert_folder / svc.SOURCE_FILENAME))
    _write_fake_run_outputs(expert_folder, metadata)
    (expert_folder / svc.SOURCE_FILENAME).write_bytes(b"old")

    recorder = _RunPipelineRecorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_pipeline", recorder)

    reference = svc.register_expert_mediapipe_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
        overwrite=True,
    )

    assert reference.reused_existing is False
    assert len(recorder.calls) == 1


def test_register_expert_mediapipe_reference_failure_marks_status_failed(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(file_path=str(sample_source_video))
    session = _StubSession(video=video)

    recorder = _RunPipelineRecorder(tmp_storage_root, should_fail=True)
    monkeypatch.setattr(svc, "run_pipeline", recorder)

    with pytest.raises(svc.ExpertMediaPipeError):
        svc.register_expert_mediapipe_reference(
            session,  # type: ignore[arg-type]
            expert_video_id=video.id,
        )

    assert video.mediapipe_status == svc.STATUS_FAILED

    expert_folder = tmp_storage_root / "expert" / "mediapipe" / str(video.id)
    assert expert_folder.is_dir()
    # Partial outputs are removed on failure.
    for filename in (
        DETECTIONS_FILENAME,
        FEATURES_FILENAME,
        METADATA_FILENAME,
        ANNOTATED_VIDEO_FILENAME,
    ):
        assert not (expert_folder / filename).exists()
    # The source is retained so retries don't need to re-copy.
    assert (expert_folder / svc.SOURCE_FILENAME).exists()


def test_register_expert_mediapipe_reference_rejects_non_expert_role(
    tmp_storage_root: Path,
    sample_source_video: Path,
) -> None:
    video = _make_expert_video(file_path=str(sample_source_video))
    video.video_role = "learner_upload"
    session = _StubSession(video=video)

    with pytest.raises(svc.ExpertMediaPipeError):
        svc.register_expert_mediapipe_reference(
            session,  # type: ignore[arg-type]
            expert_video_id=video.id,
        )


def test_register_expert_mediapipe_reference_requires_an_identifier(
    tmp_storage_root: Path,
) -> None:
    session = _StubSession(video=_make_expert_video(file_path="dummy"))
    with pytest.raises(svc.ExpertMediaPipeError):
        svc.register_expert_mediapipe_reference(session)  # type: ignore[arg-type]
