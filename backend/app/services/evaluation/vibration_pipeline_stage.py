"""Vibration pipeline stage — thin async wrapper for the evaluation orchestrator.

Runs the three-step vibration analysis:

    1. run_vibration_detection       → vibration_raw.json, vibration_summary.json
    2. detect_vibration_errors       → list of unified-error-compatible dicts
    3. generate_vibration_errors_preview → vibration_errors_preview.png

The stage is deliberately isolated:

- It never raises.  Any failure is caught, logged, and returns empty data so
  the rest of the evaluation pipeline is never affected.
- It does not modify the orchestrator.  The orchestrator can call this with
  ``asyncio.run(run_vibration_stage(...))`` or ``await run_vibration_stage(...)``
  from its own async context.

Typical call from the orchestrator (after the YOLO pre-pass):

    from app.services.evaluation.vibration_pipeline_stage import run_vibration_stage

    vibration_errors, vibration_summary = asyncio.run(
        run_vibration_stage(
            video_path=learner_video_path,
            yolo_roi_data=yolo_result["all_detections"],
            run_id=run_id,
            base_storage_path=str(_BACKEND_ROOT / "storage"),
        )
    )
"""
from __future__ import annotations

import os
import traceback
from typing import Any


async def run_vibration_stage(
    video_path: str,
    yolo_roi_data: list[dict],
    run_id: str,
    base_storage_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run vibration detection and convert results to unified error dicts.

    Parameters
    ----------
    video_path:
        Absolute path to the learner video file.
    yolo_roi_data:
        The ``all_detections`` list from ``run_shared_yolo()`` — already
        computed by the YOLO pre-pass.  YOLO is NOT called again.
    run_id:
        Evaluation run ID string (UUID).  Used to build the output path.
    base_storage_path:
        Root of the storage tree, e.g. ``/app/storage``.
        Output files are written to
        ``{base_storage_path}/evaluation/{run_id}/vibration/``.

    Returns
    -------
    tuple[list[dict], dict]
        ``(vibration_errors, vibration_summary)`` where:

        - ``vibration_errors`` is the list returned by
          ``detect_vibration_errors()`` — one dict per event, matching the
          ``unified_errors.json`` schema.  Empty list on failure or when no
          vibration is detected.
        - ``vibration_summary`` is the dict returned by
          ``run_vibration_detection()``.  Empty dict on failure.

        The tuple is always returned — it never raises.
    """
    _EMPTY: tuple[list, dict] = ([], {})

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = os.path.join(base_storage_path, "evaluation", run_id, "vibration")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        print(f"[VIBRATION] ERROR: could not create output dir {output_dir!r}")
        traceback.print_exc()
        return _EMPTY

    # ── Step 1: run optical-flow vibration detection ──────────────────────────
    try:
        from app.services.evaluation.vibration_detection import (  # noqa: PLC0415
            run_vibration_detection,
        )

        print(f"[VIBRATION DEBUG] yolo_roi_data[0] passed to run_vibration_detection = {yolo_roi_data[0] if yolo_roi_data else 'EMPTY'}")
        vibration_summary: dict[str, Any] = await run_vibration_detection(
            video_path=video_path,
            yolo_roi_data=yolo_roi_data,
            run_id=run_id,
            output_dir=output_dir,
        )
    except Exception:
        print("[VIBRATION] ERROR: run_vibration_detection failed — skipping vibration stage")
        traceback.print_exc()
        return _EMPTY

    # ── Step 2: read video metadata for bbox computation ──────────────────────
    frame_width   = 1920
    frame_height  = 1080
    fps           = 30.0
    total_frames: int | None = None

    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   or frame_width
            frame_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  or frame_height
            fps           = cap.get(cv2.CAP_PROP_FPS)                or fps
            _tf           = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames  = _tf if _tf > 0 else None
            cap.release()
        else:
            print(f"[VIBRATION] WARNING: could not open video {video_path!r} — "
                  "using default frame dimensions 1920×1080 @ 30 fps")
    except Exception:
        print("[VIBRATION] WARNING: cv2 frame-dimension read failed — "
              "using defaults 1920×1080 @ 30 fps")
        traceback.print_exc()

    # ── Step 3: convert vibration events to unified error dicts ──────────────
    try:
        from app.services.evaluation.vibration_error_detector import (  # noqa: PLC0415
            detect_vibration_errors,
        )

        print(f"[BBOX DEBUG] yolo_roi_data[0] passed to detect_vibration_errors = {yolo_roi_data[0] if yolo_roi_data else 'EMPTY'}")
        vibration_errors: list[dict[str, Any]] = detect_vibration_errors(
            vibration_summary=vibration_summary,
            yolo_roi_data=yolo_roi_data,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    except Exception:
        print("[VIBRATION] ERROR: detect_vibration_errors failed — returning empty error list")
        traceback.print_exc()
        return ([], vibration_summary)

    # ── Step 4: generate preview PNG ──────────────────────────────────────────
    preview_path = os.path.join(output_dir, "vibration_errors_preview.png")
    try:
        from app.services.evaluation.vibration_errors_preview import (  # noqa: PLC0415
            generate_vibration_errors_preview,
        )

        generate_vibration_errors_preview(
            preview_path    =preview_path,
            vibration_errors=vibration_errors,
            yolo_roi_data   =yolo_roi_data,
            fps             =fps,
            total_frames    =total_frames,
        )
        vibration_summary["preview_path"] = preview_path
    except Exception:
        print("[VIBRATION] WARNING: preview generation failed — continuing without preview")
        traceback.print_exc()
        vibration_summary["preview_path"] = None

    n = len(vibration_errors)
    detected = vibration_summary.get("vibration_detected", False)
    print(
        f"[VIBRATION] Stage complete — detected={detected}, "
        f"events={n}, output_dir={output_dir!r}"
    )

    return (vibration_errors, vibration_summary)
