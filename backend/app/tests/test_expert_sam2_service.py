"""Tests for the SAM 2 backend flow (contract + expert orchestration).

These tests avoid the real SAM 2 inference (which needs torch, a CUDA
checkpoint, and ffmpeg) by:

    * exercising the pure contract helpers directly, and
    * patching ``run_sam2_from_mediapipe_prompt`` in the expert service
      with a recorder that writes valid artifacts on disk.

They cover:
    * raw.json / summary.json derivation from in-memory documents
    * automatic prompt creation from a synthetic MediaPipe features.json
    * expert folder creation + safety checks
    * happy-path register_expert_sam2_reference
    * reuse of already-completed outputs
    * overwrite behavior
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

from app.core.sam2_constants import (
    SAM2_ANNOTATED_FILENAME,
    SAM2_METADATA_FILENAME,
    SAM2_RAW_FILENAME,
    SAM2_SOURCE_FILENAME,
    SAM2_SUMMARY_FILENAME,
)
from app.models.chapter import Chapter
from app.models.video import Video
from app.schemas.sam2.sam2_contract_schema import (
    SAM2RawDocument,
    SAM2SummaryDocument,
)
from app.schemas.sam2.sam2_schema import (
    SAM2AreaStats,
    SAM2BoundingBox,
    SAM2DetectionsDocument,
    SAM2FeaturesDocument,
    SAM2FrameFeatures,
    SAM2FrameMask,
    SAM2InitPoint,
    SAM2InitPrompt,
    SAM2RunMeta,
    SAM2WorkingRegion,
)
from app.services import expert_sam2_service as svc
from app.services.sam2.pipeline_service import (
    SAM2ContractArtifacts,
    build_sam2_raw_document,
    build_sam2_summary_document,
)
from app.services.sam2.sam2_service import (
    SAM2InitError,
    SAM2RunArtifacts,
    build_sam2_auto_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_storage_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect settings.STORAGE_ROOT to a fresh temp directory."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
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
    sam2_status: Optional[str] = None,
    mediapipe_features_path: Optional[str] = None,
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
    if sam2_status is not None:
        video.sam2_status = sam2_status
    if mediapipe_features_path is not None:
        video.mediapipe_features_path = mediapipe_features_path
        video.mediapipe_status = "completed"
    return video


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

    def refresh(self, _obj: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Fake in-memory / on-disk SAM 2 artifacts for the recorder
# ---------------------------------------------------------------------------

def _fake_run_meta(run_id: str, source_video_path: str) -> SAM2RunMeta:
    return SAM2RunMeta(
        run_id=run_id,
        source_video_path=source_video_path,
        pipeline_name="sam2_video_segmentation",
        model_name="sam2_hiera_tiny",
        device="cpu",
        created_at="2026-04-21T00:00:00+00:00",
        fps=30.0,
        frame_count=60,
        width=640,
        height=480,
        analysis_start_frame_index=0,
        analysis_end_frame_index=59,
        frame_stride=1,
        total_frames_processed=60,
        frames_with_mask=45,
        detection_rate=0.75,
        target_object_id=1,
        init_prompt=SAM2InitPrompt(
            type="point",
            source="mediapipe",
            frame_index=0,
            target_object_id=1,
            point=SAM2InitPoint(x=320.0, y=240.0, label=1),
        ),
        warnings=[],
    )


def _fake_detections(run_id: str, source_video_path: str) -> SAM2DetectionsDocument:
    frames: list[SAM2FrameMask] = []
    for i in range(5):
        has_mask = i != 2  # frame 2 is a dropout
        frames.append(
            SAM2FrameMask(
                frame_index=i,
                local_frame_index=i,
                timestamp_sec=i / 30.0,
                has_mask=has_mask,
                mask_area_px=500 + i * 10 if has_mask else None,
                mask_centroid_xy=[100.0 + i * 5.0, 200.0 + i * 2.0] if has_mask else None,
                mask_bbox=(
                    SAM2BoundingBox(
                        x_min=90 + i,
                        y_min=190 + i,
                        x_max=110 + i,
                        y_max=210 + i,
                        width=20,
                        height=20,
                    )
                    if has_mask
                    else None
                ),
                temporal_source="prompted" if i == 0 else ("propagated_forward" if has_mask else "none"),
            )
        )
    return SAM2DetectionsDocument(
        run_id=run_id,
        source_video_path=source_video_path,
        fps=30.0,
        frame_count=5,
        width=640,
        height=480,
        target_object_id=1,
        init_prompt=_fake_run_meta(run_id, source_video_path).init_prompt,
        frames=frames,
    )


def _fake_features(
    run_id: str,
    source_video_path: str,
    detections: SAM2DetectionsDocument,
) -> SAM2FeaturesDocument:
    feature_frames = [
        SAM2FrameFeatures(
            frame_index=fr.frame_index,
            timestamp_sec=fr.timestamp_sec,
            has_mask=fr.has_mask,
            mask_area_px=fr.mask_area_px,
            mask_centroid_xy=fr.mask_centroid_xy,
            mask_bbox=fr.mask_bbox,
        )
        for fr in detections.frames
    ]
    return SAM2FeaturesDocument(
        run_id=run_id,
        source_video_path=source_video_path,
        fps=30.0,
        frame_count=5,
        width=640,
        height=480,
        target_object_id=1,
        working_region=SAM2WorkingRegion(
            x_min=80,
            y_min=180,
            x_max=130,
            y_max=230,
            width=50,
            height=50,
            source="quantile",
            quantile=0.95,
            padding_px=20,
            frame_count_used=4,
        ),
        area_stats=SAM2AreaStats(min_px=500, max_px=540, mean_px=520.0, std_px=14.0),
        frames=feature_frames,
    )


def _write_contract_artifacts(
    run_dir: Path,
    *,
    detections: SAM2DetectionsDocument,
    features: SAM2FeaturesDocument,
    metadata: SAM2RunMeta,
    include_annotated: bool,
) -> SAM2ContractArtifacts:
    """Build + persist contract artifacts the way the pipeline would."""
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = build_sam2_raw_document(detections=detections, metadata=metadata)
    summary = build_sam2_summary_document(raw=raw, features=features, metadata=metadata)

    raw_path = run_dir / SAM2_RAW_FILENAME
    summary_path = run_dir / SAM2_SUMMARY_FILENAME
    metadata_path = run_dir / SAM2_METADATA_FILENAME
    annotated_path = run_dir / SAM2_ANNOTATED_FILENAME

    raw_path.write_text(raw.model_dump_json(indent=2))
    summary_path.write_text(summary.model_dump_json(indent=2))
    metadata_path.write_text(metadata.model_dump_json(indent=2))
    if include_annotated:
        annotated_path.write_bytes(b"fake-annotated-video-bytes")

    underlying = SAM2RunArtifacts(
        run_id=metadata.run_id,
        run_dir=run_dir,
        detections_path=run_dir / "detections.json",
        features_path=run_dir / "features.json",
        metadata_path=metadata_path,
        annotated_video_path=annotated_path if include_annotated else None,
        source_video_path=Path(metadata.source_video_path),
        detections=detections,
        metadata=metadata,
        features=features,
    )

    return SAM2ContractArtifacts(
        run_id=metadata.run_id,
        run_dir=run_dir,
        raw_path=raw_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path if include_annotated else None,
        source_video_path=Path(metadata.source_video_path),
        raw=raw,
        summary=summary,
        metadata=metadata,
        underlying=underlying,
    )


@dataclass
class _FakeSam2Call:
    video_path: Path
    mediapipe_features_path: Path
    run_id: Optional[str]
    render_annotation: bool


class _Sam2Recorder:
    """Test double for ``run_sam2_from_mediapipe_prompt``."""

    def __init__(self, storage_root: Path, *, should_fail: bool = False):
        self.storage_root = storage_root
        self.calls: list[_FakeSam2Call] = []
        self.should_fail = should_fail

    def __call__(
        self,
        video_path: str | Path,
        mediapipe_features_path: str | Path,
        *,
        run_id: Optional[str] = None,
        render_annotation: bool = True,
        **_kwargs: Any,
    ) -> SAM2ContractArtifacts:
        self.calls.append(
            _FakeSam2Call(
                video_path=Path(video_path),
                mediapipe_features_path=Path(mediapipe_features_path),
                run_id=run_id,
                render_annotation=render_annotation,
            )
        )

        if self.should_fail:
            from app.services.sam2.pipeline_service import SAM2PipelineError

            raise SAM2PipelineError("synthetic SAM 2 pipeline failure")

        effective_run_id = run_id or "fake_sam2_run"
        run_dir = self.storage_root / "sam2" / "runs" / effective_run_id
        metadata = _fake_run_meta(effective_run_id, str(video_path))
        detections = _fake_detections(effective_run_id, str(video_path))
        features = _fake_features(effective_run_id, str(video_path), detections)

        return _write_contract_artifacts(
            run_dir,
            detections=detections,
            features=features,
            metadata=metadata,
            include_annotated=render_annotation,
        )


# ---------------------------------------------------------------------------
# 1) Pure contract helpers
# ---------------------------------------------------------------------------

def test_build_sam2_raw_document_flattens_detections() -> None:
    run_id = "run_contract"
    source = "/tmp/source.mp4"
    metadata = _fake_run_meta(run_id, source)
    detections = _fake_detections(run_id, source)

    raw = build_sam2_raw_document(detections=detections, metadata=metadata)

    assert isinstance(raw, SAM2RawDocument)
    assert raw.run_id == run_id
    assert raw.pipeline_name == metadata.pipeline_name
    assert raw.model_name == metadata.model_name
    assert len(raw.frames) == 5

    frame_with_mask = raw.frames[0]
    assert frame_with_mask.has_mask is True
    assert frame_with_mask.mask_bbox_xyxy == [90, 190, 110, 210]
    assert frame_with_mask.mask_area_px == 500

    frame_dropout = raw.frames[2]
    assert frame_dropout.has_mask is False
    assert frame_dropout.mask_bbox_xyxy is None
    assert frame_dropout.mask_centroid_xy is None
    assert frame_dropout.mask_area_px is None


def test_build_sam2_summary_document_aggregates_expected_fields() -> None:
    run_id = "run_summary"
    source = "/tmp/source.mp4"
    metadata = _fake_run_meta(run_id, source)
    detections = _fake_detections(run_id, source)
    features = _fake_features(run_id, source, detections)

    raw = build_sam2_raw_document(detections=detections, metadata=metadata)
    summary = build_sam2_summary_document(raw=raw, features=features, metadata=metadata)

    assert isinstance(summary, SAM2SummaryDocument)
    assert summary.total_frames == 5
    assert summary.frames_with_mask == 4
    assert summary.detection_rate == pytest.approx(4 / 5)
    assert summary.mean_mask_area_px == pytest.approx(520.0)
    assert summary.min_mask_area_px == 500
    assert summary.max_mask_area_px == 540
    # The single dropout (frame 2) creates one has_mask True->False transition.
    assert summary.track_fragmentation_count == 1
    # Working region propagated through.
    assert summary.working_region is not None
    assert summary.working_region.x_min == 80
    assert summary.working_region.x_max == 130
    # A mean centroid speed must be reported from the three valid transitions.
    assert summary.mean_centroid_speed_px_per_frame is not None
    assert summary.mean_centroid_speed_px_per_frame > 0


# ---------------------------------------------------------------------------
# 2) Automatic prompt creation from MediaPipe
# ---------------------------------------------------------------------------

def _write_fake_mediapipe_features_with_metadata(
    folder: Path,
    *,
    width: int = 640,
    height: int = 480,
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    index_rel_x: float = 0.1,
    index_rel_y: float = -0.1,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    features_path = folder / "features.json"
    metadata_path = folder / "metadata.json"

    features = {
        "run_id": "fake_mp",
        "source_video_path": str(folder / "source.mp4"),
        "fps": 30.0,
        "frame_count": 2,
        "trajectory_history_size": 30,
        "frames": [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "has_detection": False,
                "handedness": None,
                "wrist": None,
                "hand_center": None,
                "wrist_relative_landmarks": None,
                "joint_angles": None,
                "trajectory_history": [],
            },
            {
                "frame_index": 1,
                "timestamp_sec": 1 / 30.0,
                "has_detection": True,
                "handedness": "Right",
                "wrist": [wrist_x, wrist_y, 0.0],
                "hand_center": [wrist_x, wrist_y],
                "wrist_relative_landmarks": {
                    "index_finger_tip": [index_rel_x, index_rel_y, 0.0],
                    "middle_finger_tip": [0.08, -0.05, 0.0],
                    "thumb_tip": [-0.04, 0.02, 0.0],
                },
                "joint_angles": None,
                "trajectory_history": [],
            },
        ],
    }

    metadata = {
        "run_id": "fake_mp",
        "source_video_path": str(folder / "source.mp4"),
        "fps": 30.0,
        "frame_count": 2,
        "width": width,
        "height": height,
        "created_at": "2026-04-21T00:00:00+00:00",
        "selected_hand_policy": "prefer_right_else_first",
        "total_frames": 2,
        "frames_with_detection": 1,
        "detection_rate": 0.5,
        "right_hand_selected_count": 1,
        "left_hand_selected_count": 0,
    }

    features_path.write_text(json.dumps(features))
    metadata_path.write_text(json.dumps(metadata))
    return features_path


def test_build_sam2_auto_prompt_selects_index_tip_and_converts_to_pixels(
    tmp_path: Path,
) -> None:
    features_path = _write_fake_mediapipe_features_with_metadata(
        tmp_path / "mp",
        width=640,
        height=480,
        wrist_x=0.5,
        wrist_y=0.5,
        index_rel_x=0.1,
        index_rel_y=-0.1,
    )

    prompt = build_sam2_auto_prompt(features_path)

    assert prompt["frame_index"] == 1
    assert prompt["source"] == "index_tip"
    assert prompt["used_fallback"] is False
    # (0.5 + 0.1) * 640 = 384, (0.5 + -0.1) * 480 = 192
    assert prompt["point_xy"][0] == pytest.approx(384.0)
    assert prompt["point_xy"][1] == pytest.approx(192.0)
    assert prompt["image_width"] == 640
    assert prompt["image_height"] == 480


def test_build_sam2_auto_prompt_raises_when_no_frame_has_detection(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "empty_mp"
    folder.mkdir()
    (folder / "features.json").write_text(
        json.dumps(
            {
                "run_id": "no_hands",
                "source_video_path": str(folder / "source.mp4"),
                "fps": 30.0,
                "frame_count": 1,
                "trajectory_history_size": 30,
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp_sec": 0.0,
                        "has_detection": False,
                        "handedness": None,
                        "wrist": None,
                        "hand_center": None,
                        "wrist_relative_landmarks": None,
                        "joint_angles": None,
                        "trajectory_history": [],
                    }
                ],
            }
        )
    )
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "no_hands",
                "source_video_path": str(folder / "source.mp4"),
                "fps": 30.0,
                "frame_count": 1,
                "width": 320,
                "height": 240,
                "created_at": "2026-04-21T00:00:00+00:00",
                "selected_hand_policy": "prefer_right_else_first",
                "total_frames": 1,
                "frames_with_detection": 0,
                "detection_rate": 0.0,
                "right_hand_selected_count": 0,
                "left_hand_selected_count": 0,
            }
        )
    )

    with pytest.raises(SAM2InitError):
        build_sam2_auto_prompt(folder / "features.json")


# ---------------------------------------------------------------------------
# 3) Expert folder helpers
# ---------------------------------------------------------------------------

def test_ensure_expert_sam2_dir_creates_folder(tmp_storage_root: Path) -> None:
    folder = svc.ensure_expert_sam2_dir("expert_001")
    assert folder.is_dir()
    assert folder == tmp_storage_root / "expert" / "sam2" / "expert_001"


@pytest.mark.parametrize("bad_code", ["", "..", "a/b", "a\\b", "../escape"])
def test_ensure_expert_sam2_dir_rejects_unsafe_codes(
    tmp_storage_root: Path, bad_code: str
) -> None:
    with pytest.raises(svc.ExpertSam2Error):
        svc.ensure_expert_sam2_dir(bad_code)


def test_cleanup_on_failure_removes_partials_and_can_keep_source(
    tmp_storage_root: Path,
) -> None:
    folder = svc.ensure_expert_sam2_dir("expert_cleanup")
    for filename in (
        SAM2_SOURCE_FILENAME,
        SAM2_RAW_FILENAME,
        SAM2_SUMMARY_FILENAME,
        SAM2_METADATA_FILENAME,
        SAM2_ANNOTATED_FILENAME,
    ):
        (folder / filename).write_text("partial")

    svc.cleanup_on_failure(folder, keep_source=True)

    assert (folder / SAM2_SOURCE_FILENAME).exists()
    assert not (folder / SAM2_RAW_FILENAME).exists()
    assert not (folder / SAM2_SUMMARY_FILENAME).exists()
    assert not (folder / SAM2_METADATA_FILENAME).exists()
    assert not (folder / SAM2_ANNOTATED_FILENAME).exists()


# ---------------------------------------------------------------------------
# 4) Expert orchestration (mocked pipeline)
# ---------------------------------------------------------------------------

def _seed_mediapipe_expert_reference(
    tmp_storage_root: Path, video: Video
) -> Path:
    """Write a MediaPipe features.json + metadata.json under the expected expert folder.

    Returns the storage-relative features.json path, which is what the
    service expects on ``video.mediapipe_features_path``.
    """
    mp_folder = tmp_storage_root / "expert" / "mediapipe" / str(video.id)
    features_path = _write_fake_mediapipe_features_with_metadata(mp_folder)
    # Also drop a pretend MediaPipe source.mp4 so save_expert_source_video
    # prefers it over video.file_path (consistent with real behavior).
    (mp_folder / "source.mp4").write_bytes(b"mp-source-bytes")

    relative = features_path.relative_to(tmp_storage_root).as_posix()
    video.mediapipe_features_path = relative
    video.mediapipe_status = "completed"
    return features_path


def test_register_expert_sam2_reference_happy_path(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapter = SimpleNamespace(id=str(uuid4()), title="Suture Tying")
    video = _make_expert_video(
        file_path=str(sample_source_video),
        chapter_id=chapter.id,
    )
    _seed_mediapipe_expert_reference(tmp_storage_root, video)

    session = _StubSession(video=video, chapter=chapter)
    recorder = _Sam2Recorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_sam2_from_mediapipe_prompt", recorder)

    reference = svc.register_expert_sam2_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
    )

    expert_folder = tmp_storage_root / "expert" / "sam2" / str(video.id)
    assert expert_folder.is_dir()
    for filename in (
        SAM2_SOURCE_FILENAME,
        SAM2_RAW_FILENAME,
        SAM2_SUMMARY_FILENAME,
        SAM2_METADATA_FILENAME,
        SAM2_ANNOTATED_FILENAME,
    ):
        assert (expert_folder / filename).is_file(), f"expected {filename} on disk"

    # The recorder was called once, with the MediaPipe features path.
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call.render_annotation is True
    assert call.mediapipe_features_path.is_file()

    # The response carries the canonical paths and a populated summary.
    assert reference.reused_existing is False
    assert reference.expert_code == str(video.id)
    assert reference.sam2_status == svc.STATUS_COMPLETED
    assert reference.summary.fps == 30.0
    assert reference.summary.model_name == "sam2_hiera_tiny"
    assert reference.summary.detection_rate == pytest.approx(0.8)
    assert reference.title == "Suture Tying"
    assert reference.annotated_path is not None
    assert reference.annotated_path.endswith(SAM2_ANNOTATED_FILENAME)

    # DB row was updated with SAM 2 paths + summary.
    assert video.sam2_status == svc.STATUS_COMPLETED
    assert video.sam2_fps is not None and float(video.sam2_fps) == 30.0
    assert video.sam2_frame_count == 60
    assert video.sam2_detection_rate is not None
    assert video.sam2_processed_at is not None

    # Temporary runs folder was cleaned up.
    temp_run_dir = tmp_storage_root / "sam2" / "runs" / call.run_id
    assert not temp_run_dir.exists()


def test_register_expert_sam2_reference_reuses_completed_outputs(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(
        file_path=str(sample_source_video),
        sam2_status=svc.STATUS_COMPLETED,
    )
    _seed_mediapipe_expert_reference(tmp_storage_root, video)

    # Pre-populate the stable expert folder so the reuse path kicks in.
    expert_folder = svc.ensure_expert_sam2_dir(str(video.id))
    run_id = "prebuilt_run"
    metadata = _fake_run_meta(run_id, str(expert_folder / SAM2_SOURCE_FILENAME))
    detections = _fake_detections(run_id, metadata.source_video_path)
    features = _fake_features(run_id, metadata.source_video_path, detections)
    _write_contract_artifacts(
        expert_folder,
        detections=detections,
        features=features,
        metadata=metadata,
        include_annotated=True,
    )
    (expert_folder / SAM2_SOURCE_FILENAME).write_bytes(b"pre-existing")

    session = _StubSession(video=video)
    recorder = _Sam2Recorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_sam2_from_mediapipe_prompt", recorder)

    reference = svc.register_expert_sam2_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
        overwrite=False,
    )

    assert reference.reused_existing is True
    assert not recorder.calls


def test_register_expert_sam2_reference_overwrite_reruns_pipeline(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(
        file_path=str(sample_source_video),
        sam2_status=svc.STATUS_COMPLETED,
    )
    _seed_mediapipe_expert_reference(tmp_storage_root, video)

    expert_folder = svc.ensure_expert_sam2_dir(str(video.id))
    metadata = _fake_run_meta("stale", str(expert_folder / SAM2_SOURCE_FILENAME))
    detections = _fake_detections("stale", metadata.source_video_path)
    features = _fake_features("stale", metadata.source_video_path, detections)
    _write_contract_artifacts(
        expert_folder,
        detections=detections,
        features=features,
        metadata=metadata,
        include_annotated=True,
    )
    (expert_folder / SAM2_SOURCE_FILENAME).write_bytes(b"stale")

    session = _StubSession(video=video)
    recorder = _Sam2Recorder(tmp_storage_root)
    monkeypatch.setattr(svc, "run_sam2_from_mediapipe_prompt", recorder)

    reference = svc.register_expert_sam2_reference(
        session,  # type: ignore[arg-type]
        expert_video_id=video.id,
        overwrite=True,
    )

    assert reference.reused_existing is False
    assert len(recorder.calls) == 1


def test_register_expert_sam2_reference_failure_marks_status_failed(
    tmp_storage_root: Path,
    sample_source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_expert_video(file_path=str(sample_source_video))
    _seed_mediapipe_expert_reference(tmp_storage_root, video)

    session = _StubSession(video=video)
    recorder = _Sam2Recorder(tmp_storage_root, should_fail=True)
    monkeypatch.setattr(svc, "run_sam2_from_mediapipe_prompt", recorder)

    with pytest.raises(svc.ExpertSam2Error):
        svc.register_expert_sam2_reference(
            session,  # type: ignore[arg-type]
            expert_video_id=video.id,
        )

    assert video.sam2_status == svc.STATUS_FAILED

    expert_folder = tmp_storage_root / "expert" / "sam2" / str(video.id)
    assert expert_folder.is_dir()
    for filename in (
        SAM2_RAW_FILENAME,
        SAM2_SUMMARY_FILENAME,
        SAM2_METADATA_FILENAME,
        SAM2_ANNOTATED_FILENAME,
    ):
        assert not (expert_folder / filename).exists()
    # Source is retained so retries don't need to re-copy.
    assert (expert_folder / SAM2_SOURCE_FILENAME).exists()


def test_register_expert_sam2_reference_rejects_non_expert_role(
    tmp_storage_root: Path,
    sample_source_video: Path,
) -> None:
    video = _make_expert_video(file_path=str(sample_source_video))
    video.video_role = "learner_upload"
    session = _StubSession(video=video)

    with pytest.raises(svc.ExpertSam2Error):
        svc.register_expert_sam2_reference(
            session,  # type: ignore[arg-type]
            expert_video_id=video.id,
        )


def test_register_expert_sam2_reference_requires_mediapipe_first(
    tmp_storage_root: Path,
    sample_source_video: Path,
) -> None:
    # No MediaPipe features set on the video -> ExpertMediaPipeMissingError.
    video = _make_expert_video(file_path=str(sample_source_video))
    session = _StubSession(video=video)

    with pytest.raises(svc.ExpertMediaPipeMissingError):
        svc.register_expert_sam2_reference(
            session,  # type: ignore[arg-type]
            expert_video_id=video.id,
        )


def test_register_expert_sam2_reference_requires_an_identifier(
    tmp_storage_root: Path,
) -> None:
    session = _StubSession(video=_make_expert_video(file_path="dummy"))
    with pytest.raises(svc.ExpertSam2Error):
        svc.register_expert_sam2_reference(session)  # type: ignore[arg-type]
