"""Vibration detection for the AugMentor evaluation pipeline.

Accepts pre-computed YOLO detections from the shared YOLO pre-pass, builds
a per-frame ROI using the same expansion logic as the optical flow service,
computes dense optical flow (RAFT on GPU; Farneback fallback on CPU) inside
those ROIs, then runs FFT-based vibration analysis and a streak classifier
to find vibration events.

Inputs
------
video_path    : absolute path to the learner video
yolo_roi_data : per-frame YOLO detection list produced by run_shared_yolo()
                Each item has:
                    frame_index : int
                    detected    : bool
                    bbox        : [x1, y1, x2, y2] or None
                    confidence  : float or None
                    bbox_center : [cx, cy] or None
run_id        : evaluation run ID string
output_dir    : directory for vibration_raw.json / vibration_summary.json

Outputs
-------
vibration_raw.json     — per-frame flow features + all FFT windows
vibration_summary.json — vibration_detected bool, events, avg/peak magnitude

YOLO is NOT called again.  All logic is self-contained — no imports from
other optical_flow submodules so this module can run independently.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
# ROI hold  — same as YoloScissorsROIConfig.max_roi_hold_frames
_MAX_ROI_HOLD_FRAMES = 5

# Farneback parameters — same as FarnebackConfig defaults in farneback_service.py
_FB_PYR_SCALE  = 0.5
_FB_LEVELS     = 3
_FB_WINSIZE    = 15
_FB_ITERATIONS = 3
_FB_POLY_N     = 5
_FB_POLY_SIGMA = 1.2
_FB_FLAGS      = 0
_FB_BLUR_K     = 5     # gaussian_blur_kernel before Farneback
_MOTION_THRESHOLD = 2.0   # pixels — same as FarnebackConfig.motion_threshold

# FFT analysis — same as feature_extractor._compute_fft_vibration
_FFT_WINDOW_SEC = 1.0
_FFT_FREQ_MIN   = 2.0
_FFT_FREQ_MAX   = 20.0

# Event classification — same as vibration_classifier.py constants
_MIN_CONSECUTIVE = 3
_MIN_FREQ_HZ     = 4.0
_FREQ_TOLERANCE  = 2.0
_CONF_THRESHOLD  = 0.13


# ── RAFT singleton — mirrors raft_flow_service._get_raft_model() ──────────────
_RAFT_INSTANCE: tuple | None = None
_RAFT_LOCK = threading.Lock()


def _check_raft_available() -> bool:
    """Return True only when CUDA is available and torchvision can be imported.

    Copied from raft_flow_service._check_raft_available.
    """
    try:
        import torch  # type: ignore  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        import torchvision.models.optical_flow  # type: ignore  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


def _get_raft_model() -> tuple:
    """Load raft_large onto CUDA exactly once (thread-safe).

    Returns (model, preprocess, device_str).
    Pattern copied from raft_flow_service._get_raft_model.
    """
    global _RAFT_INSTANCE  # noqa: PLW0603

    if _RAFT_INSTANCE is not None:
        return _RAFT_INSTANCE

    with _RAFT_LOCK:
        if _RAFT_INSTANCE is not None:
            return _RAFT_INSTANCE

        import torch  # type: ignore  # noqa: PLC0415
        from torchvision.models.optical_flow import (  # type: ignore  # noqa: PLC0415
            Raft_Large_Weights,
            raft_large,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VIBRATION] Loading raft_large on device={device} …", flush=True)
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        _RAFT_INSTANCE = (model, preprocess, device)
        print("[VIBRATION] raft_large ready.", flush=True)
        return _RAFT_INSTANCE


# ── ROI expansion — exact logic from yolo_scissors_roi.expand_scissors_bbox_to_hand_roi

def _expand_roi(
    bbox: list[float],
    frame_width: int,
    frame_height: int,
) -> list[int] | None:
    """Expand a scissors YOLO bbox into a hand-region ROI.

    Logic copied verbatim from
    yolo_scissors_roi.expand_scissors_bbox_to_hand_roi (the proven expansion
    used by the standalone optical flow pipeline):

        left/right : ±0.2 × bbox_width
        top        : −0.1 × bbox_height
        bottom     : +0.4 × bbox_height

    This keeps the crop tight around the scissors while including enough of
    the hand below for the vibration signal.  Returns None on degenerate input.
    """
    if not bbox or len(bbox) != 4:
        return None
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    ex1 = max(0,             int(x1 - 0.2 * bw))
    ex2 = min(int(frame_width),  int(x2 + 0.2 * bw))
    ey1 = max(0,             int(y1 - 0.1 * bh))
    ey2 = min(int(frame_height), int(y2 + 0.4 * bh))
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return [ex1, ey1, ex2, ey2]


def _build_roi_lookup(
    yolo_roi_data: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict[int, list[int]]:
    """Build frame_index → expanded-ROI mapping from the YOLO pre-pass list.

    Items are sorted by frame_index first so the hold-frame logic is correct
    regardless of the order they arrive in yolo_roi_data.  Frames that miss a
    detection inherit the last valid expanded ROI for up to _MAX_ROI_HOLD_FRAMES
    consecutive frames — same behaviour as the optical flow pipeline.
    """
    roi_lookup: dict[int, list[int]] = {}
    last_valid: list[int] | None = None
    hold_remaining = 0

    for entry in sorted(yolo_roi_data, key=lambda d: int(d.get("frame_index", 0))):
        fi = int(entry.get("frame_index", -1))
        if fi < 0:
            continue

        if entry.get("detected", False) and entry.get("bbox"):
            expanded = _expand_roi(entry["bbox"], frame_width, frame_height)
            if expanded is not None:
                roi_lookup[fi] = expanded
                last_valid = expanded
                hold_remaining = _MAX_ROI_HOLD_FRAMES
                continue

        # Detection missing or invalid — inherit the previous ROI if possible
        if last_valid is not None and hold_remaining > 0:
            roi_lookup[fi] = last_valid
            hold_remaining -= 1
        else:
            hold_remaining = 0

    return roi_lookup


# ── Optical flow helpers ──────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to single-channel grayscale if needed."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _blur_for_flow(gray: np.ndarray, kernel: int = _FB_BLUR_K) -> np.ndarray:
    """Light Gaussian blur before Farneback to reduce sensor noise.

    Copied from farneback_service._blur_gray_for_flow.
    """
    if kernel <= 0:
        return gray
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    if k < 3:
        return gray
    return cv2.GaussianBlur(gray, (k, k), 0)


def _pad_to_8(tensor: Any) -> tuple[Any, tuple[int, int]]:
    """Pad H and W dimensions to multiples of 8 (RAFT requirement).

    Copied from raft_flow_service._pad_to_multiple_of_8.
    """
    import torch.nn.functional as F  # type: ignore  # noqa: PLC0415

    _, _, H, W = tensor.shape
    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph == 0 and pw == 0:
        return tensor, (0, 0)
    return F.pad(tensor, (0, pw, 0, ph), mode="replicate"), (ph, pw)


def _flow_magnitude_features(flow: np.ndarray) -> dict[str, float]:
    """Derive mean/peak magnitude and motion-area ratio from a flow field."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    if mag.size == 0:
        return {"mean_magnitude": 0.0, "peak_magnitude": 0.0, "motion_area_ratio": 0.0}
    return {
        "mean_magnitude":    round(float(mag.mean()), 6),
        "peak_magnitude":    round(float(mag.max()),  6),
        "motion_area_ratio": round(float((mag > _MOTION_THRESHOLD).mean()), 6),
    }


def _compute_flow_in_roi(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    roi: list[int],
    frame_width: int,
    frame_height: int,
    raft_ok: bool,
    raft_model_tuple: tuple | None,
) -> dict[str, float]:
    """Compute dense optical flow inside *roi* and return magnitude features.

    Tries RAFT first (same flow as farneback_service._compute_flow), falls
    back to Farneback on any failure.  Returns dict with mean_magnitude,
    peak_magnitude, motion_area_ratio.
    """
    x1, y1, x2, y2 = (max(0, int(v)) for v in roi)
    x2 = min(frame_width,  x2)
    y2 = min(frame_height, y2)
    if x2 <= x1 or y2 <= y1:
        return {"mean_magnitude": 0.0, "peak_magnitude": 0.0, "motion_area_ratio": 0.0}

    crop_prev = prev_bgr[y1:y2, x1:x2]
    crop_curr = curr_bgr[y1:y2, x1:x2]
    if crop_prev.size == 0 or crop_curr.size == 0:
        return {"mean_magnitude": 0.0, "peak_magnitude": 0.0, "motion_area_ratio": 0.0}

    gray_prev = _to_gray(crop_prev)
    gray_curr = _to_gray(crop_curr)
    flow: np.ndarray | None = None

    # ── RAFT path ─────────────────────────────────────────────────────────────
    if raft_ok and raft_model_tuple is not None:
        try:
            import torch  # type: ignore  # noqa: PLC0415

            model, preprocess, device = raft_model_tuple
            H, W = gray_prev.shape

            # Proportional downscale to fit within 320×320 (same cap as raft_flow_service)
            scale = min(320.0 / H, 320.0 / W, 1.0)
            if scale < 1.0:
                nh = max(128, int(round(H * scale)))
                nw = max(128, int(round(W * scale)))
                gp = cv2.resize(gray_prev, (nw, nh), interpolation=cv2.INTER_AREA)
                gc = cv2.resize(gray_curr, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                gp, gc = gray_prev, gray_curr
                nh, nw = H, W

            def _g2t(g: np.ndarray) -> Any:
                rgb = np.stack([g, g, g], axis=0)
                return torch.from_numpy(rgb).to(device)

            t1, t2 = preprocess(_g2t(gp), _g2t(gc))
            t1, t2 = t1.unsqueeze(0), t2.unsqueeze(0)
            t1, _ = _pad_to_8(t1)
            t2, _ = _pad_to_8(t2)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
                if device == "cuda"
                else torch.no_grad()
            )
            with torch.no_grad(), amp_ctx:
                preds = model(t1, t2)

            flow_t = preds[-1][:, :, :nh, :nw].float()
            flow_np = flow_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

            if scale < 1.0:
                flow_np[..., 0] /= (nw / W)
                flow_np[..., 1] /= (nh / H)
                flow_np = cv2.resize(flow_np, (W, H), interpolation=cv2.INTER_LINEAR)

            flow = flow_np
        except Exception as exc:  # noqa: BLE001
            print(
                f"[VIBRATION] RAFT frame failed, using Farneback: {exc}",
                flush=True,
            )
            flow = None

    # ── Farneback fallback ────────────────────────────────────────────────────
    if flow is None:
        fb_prev = gray_prev
        fb_curr = gray_curr
        fb_H, fb_W = fb_prev.shape
        # Proportional downscale to ≤320×320 before Farneback (same cap as RAFT
        # path above) so CPU inference on large crops stays fast.
        fb_scale = min(320.0 / fb_H, 320.0 / fb_W, 1.0)
        if fb_scale < 1.0:
            fb_nh = max(32, int(round(fb_H * fb_scale)))
            fb_nw = max(32, int(round(fb_W * fb_scale)))
            fb_prev = cv2.resize(fb_prev, (fb_nw, fb_nh), interpolation=cv2.INTER_AREA)
            fb_curr = cv2.resize(fb_curr, (fb_nw, fb_nh), interpolation=cv2.INTER_AREA)
        prev_b = _blur_for_flow(fb_prev)
        curr_b = _blur_for_flow(fb_curr)
        flow_small = cv2.calcOpticalFlowFarneback(
            prev=prev_b,
            next=curr_b,
            flow=None,
            pyr_scale=_FB_PYR_SCALE,
            levels=_FB_LEVELS,
            winsize=_FB_WINSIZE,
            iterations=_FB_ITERATIONS,
            poly_n=_FB_POLY_N,
            poly_sigma=_FB_POLY_SIGMA,
            flags=_FB_FLAGS,
        )
        if fb_scale < 1.0:
            # Scale flow vectors back and resize to original ROI dims.
            flow_small[..., 0] /= (fb_nw / fb_W)
            flow_small[..., 1] /= (fb_nh / fb_H)
            flow = cv2.resize(flow_small, (fb_W, fb_H), interpolation=cv2.INTER_LINEAR)
        else:
            flow = flow_small

    return _flow_magnitude_features(flow)


# ── FFT vibration analysis — copied from feature_extractor._compute_fft_vibration ──

def _compute_fft_windows(raw_magnitudes: list[float], fps: float) -> list[dict]:
    """Sliding-window FFT vibration analysis.

    Algorithm copied verbatim from
    feature_extractor._compute_fft_vibration(window_sec=1.0):

    For each 1-second window (step = window // 2):
      1. First-difference the window to remove DC / slow trends.
      2. rfft, isolate 2–20 Hz band.
      3. Record dominant frequency and confidence = peak_amp / total_amp.

    Returns list of window dicts.
    """
    if len(raw_magnitudes) < 8:
        return []

    window_size = max(8, int(_FFT_WINDOW_SEC * fps))
    step = max(1, window_size // 2)
    arr = np.asarray(raw_magnitudes, dtype=np.float32)
    results: list[dict] = []

    for start in range(0, len(arr) - window_size + 1, step):
        window = arr[start : start + window_size]
        signal = np.diff(window)
        if len(signal) < 4:
            continue
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
        mask = (freqs >= _FFT_FREQ_MIN) & (freqs <= _FFT_FREQ_MAX)
        if not np.any(mask):
            continue
        vib_freqs = freqs[mask]
        vib_amps  = fft_vals[mask]
        peak_idx  = int(np.argmax(vib_amps))
        dom_freq  = float(vib_freqs[peak_idx])
        dom_amp   = float(vib_amps[peak_idx])
        total_amp = float(np.sum(fft_vals)) + 1e-8
        confidence = dom_amp / total_amp
        results.append({
            "start_frame":      start,
            "end_frame":        start + window_size,
            "timestamp_sec":    round(start / fps, 2),
            "dominant_freq_hz": round(dom_freq, 4),
            "confidence":       round(confidence, 4),
        })

    return results


# ── Event classifier — adapted from vibration_classifier.classify_vibration ──

def _get_severity(consecutive: int, peak_confidence: float) -> str:
    """Severity mapping — copied from vibration_classifier._get_severity."""
    if consecutive >= 4 or peak_confidence > 0.30:
        return "severe"
    if peak_confidence >= 0.20:
        return "moderate"
    return "mild"


def _classify_vibration_events(
    windows: list[dict],
    window_size: int,
    fps: float,
) -> list[dict]:
    """Find all vibration events, keeping the strongest signal when streaks overlap.

    Algorithm (two phases):

    Phase 1 — collect ALL candidate streaks.
        For every qualifying window position (conf ≥ threshold, freq ≥ min_freq)
        try to build the longest streak starting there.  Every candidate with
        length ≥ _MIN_CONSECUTIVE is kept regardless of whether it overlaps an
        earlier candidate.  This ensures a strong streak at 14.48 Hz is not
        missed because a weaker 10.34 Hz streak was found first.

    Phase 2 — deduplicate overlapping events.
        Sort candidates by peak_confidence descending.  Greedily accept each
        candidate only when it does not overlap (in time) any already-accepted
        event.  Then re-sort the accepted events by start time for output.
    """
    # ── Phase 1: find every valid streak ─────────────────────────────────────
    candidates: list[dict] = []
    for i in range(len(windows)):
        w = windows[i]
        if w["confidence"] < _CONF_THRESHOLD or w["dominant_freq_hz"] < _MIN_FREQ_HZ:
            continue

        locked_freq = w["dominant_freq_hz"]
        streak = [w]
        j = i + 1
        while j < len(windows):
            nw = windows[j]
            if (
                nw["confidence"] >= _CONF_THRESHOLD
                and abs(nw["dominant_freq_hz"] - locked_freq) <= _FREQ_TOLERANCE
            ):
                streak.append(nw)
                j += 1
            else:
                break

        if len(streak) < _MIN_CONSECUTIVE:
            continue

        peak_conf = max(s["confidence"] for s in streak)
        severity  = _get_severity(len(streak), peak_conf)
        ts_start  = streak[0]["timestamp_sec"]
        ts_end    = round(streak[-1]["timestamp_sec"] + window_size / fps, 3)
        candidates.append({
            "start_frame":         streak[0]["start_frame"],
            "end_frame":           streak[-1]["end_frame"],
            "timestamp_start_sec": ts_start,
            "timestamp_end_sec":   ts_end,
            "duration_sec":        round(ts_end - ts_start, 3),
            "dominant_freq_hz":    round(locked_freq, 4),
            "peak_confidence":     round(peak_conf, 4),
            "severity":            severity,
            "consecutive_windows": len(streak),
        })

    # ── Phase 2: keep highest-confidence event when two overlap in time ───────
    # Sort strongest-first so greedy acceptance always picks the best signal.
    candidates.sort(key=lambda e: e["peak_confidence"], reverse=True)

    accepted: list[dict] = []
    for cand in candidates:
        overlaps = any(
            cand["timestamp_start_sec"] < acc["timestamp_end_sec"]
            and acc["timestamp_start_sec"] < cand["timestamp_end_sec"]
            for acc in accepted
        )
        if not overlaps:
            accepted.append(cand)

    # Re-sort by start time for consistent chronological output.
    accepted.sort(key=lambda e: e["timestamp_start_sec"])

    for ev in accepted:
        print(
            f"[VIBRATION]  event: {ev['timestamp_start_sec']}s – "
            f"{ev['timestamp_end_sec']}s | "
            f"{ev['dominant_freq_hz']:.2f} Hz | {ev['severity']} | "
            f"streak={ev['consecutive_windows']}",
            flush=True,
        )

    return accepted


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_result(
    run_id: str,
    fps: float,
    total_frames: int,
    output_dir: str,
) -> dict[str, Any]:
    """Write an empty summary and return it when there is nothing to analyse."""
    result: dict[str, Any] = {
        "run_id":             run_id,
        "fps":                fps,
        "total_frames":       total_frames,
        "vibration_detected": False,
        "vibration_events":   [],
        "avg_magnitude":      0.0,
        "peak_magnitude":     0.0,
    }
    summary_path = os.path.join(output_dir, "vibration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    result["raw_json_path"]     = os.path.join(output_dir, "vibration_raw.json")
    result["summary_json_path"] = summary_path
    return result


# ── Public entry point ────────────────────────────────────────────────────────

async def run_vibration_detection(
    video_path: str,
    yolo_roi_data: list[dict],
    run_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Run vibration detection using pre-computed YOLO bounding boxes.

    This is the sole public entry point for the vibration pipeline stage.
    It does NOT call YOLO; yolo_roi_data must be the ``all_detections`` list
    returned by ``run_shared_yolo()`` (from yolo_shared.py).

    Each item in yolo_roi_data must contain:
        frame_index : int   — video frame number (0-based)
        detected    : bool  — True when scissors were found
        bbox        : list[float] | None  — [x1, y1, x2, y2] in pixel coords
        confidence  : float | None
        bbox_center : list[float] | None  — [cx, cy]

    The function:
      1. Opens the video and reads every frame into memory.
      2. Builds a frame_index → expanded-ROI mapping (expand_scissors_bbox_to_hand_roi
         logic, hold up to 5 frames on miss).
      3. Computes dense optical flow frame-by-frame inside each ROI:
            • RAFT (raft_large, GPU, AMP half-precision) when available.
            • Farneback (CPU) when RAFT is unavailable or throws.
      4. Runs FFT sliding-window analysis on the per-frame mean-magnitude
         signal (1 s window, 0.5 s step, 2–20 Hz band).
      5. Classifies consecutive high-confidence windows into events
         (≥3 consecutive, ≥4 Hz, confidence ≥0.15).
      6. Saves vibration_raw.json and vibration_summary.json to output_dir.
      7. Returns the summary dict.

    Parameters
    ----------
    video_path    : Absolute path to the learner video file.
    yolo_roi_data : YOLO pre-pass detection list (see above).
    run_id        : Evaluation run ID string.
    output_dir    : Directory for output files
                    (typically storage/evaluation/{run_id}/vibration/).

    Returns
    -------
    dict with keys:
        run_id, fps, total_frames, vibration_detected, vibration_events,
        avg_magnitude, peak_magnitude, frame_width, frame_height,
        processing_time_sec, raw_json_path, summary_json_path
    """
    t_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[VIBRATION] run_id={run_id}", flush=True)
    print(f"[VIBRATION] video_path={video_path}", flush=True)

    # ── Open video and read all frames ────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[VIBRATION] Could not open video: {video_path}")

    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr: list[tuple[int, np.ndarray]] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append((fi, frame))
        fi += 1
    cap.release()

    total_video_frames = len(frames_bgr)
    print(
        f"[VIBRATION] Loaded {total_video_frames} frames  "
        f"fps={fps}  size={frame_width}×{frame_height}",
        flush=True,
    )

    if total_video_frames < 2:
        print("[VIBRATION] Too few frames — skipping.", flush=True)
        return _empty_result(run_id, fps, 0, output_dir)

    # ── Build ROI lookup ──────────────────────────────────────────────────────
    roi_lookup = _build_roi_lookup(yolo_roi_data, frame_width, frame_height)
    yolo_roi_count = sum(
        1 for fi_, _ in frames_bgr if fi_ in roi_lookup
    )
    print(
        f"[VIBRATION] ROI coverage: {yolo_roi_count} / {total_video_frames} frames",
        flush=True,
    )

    # Confirm crop size matches the expected ~150×250px from the ±0.2/0.1/0.4 expansion.
    if roi_lookup:
        roi_widths  = [r[2] - r[0] for r in roi_lookup.values()]
        roi_heights = [r[3] - r[1] for r in roi_lookup.values()]
        avg_w = sum(roi_widths)  / len(roi_widths)
        avg_h = sum(roi_heights) / len(roi_heights)
        print(
            f"[VIBRATION] Avg crop size: {avg_w:.0f}x{avg_h:.0f}px",
            flush=True,
        )

    # ── Initialise RAFT model (once) ──────────────────────────────────────────
    raft_ok = _check_raft_available()
    raft_tuple: tuple | None = None
    if raft_ok:
        try:
            raft_tuple = _get_raft_model()
        except Exception as exc:  # noqa: BLE001
            print(f"[VIBRATION] RAFT init failed ({exc}) — using Farneback", flush=True)
            raft_ok = False
    else:
        print("[VIBRATION] RAFT unavailable — using Farneback", flush=True)

    # Unpack the RAFT model tuple once so the batch helper can close over it.
    _raft_model, _raft_preprocess, _raft_device = (
        raft_tuple if (raft_ok and raft_tuple is not None) else (None, None, "cpu")
    )

    # ── Batched RAFT inference — 32 pairs per forward pass ───────────────────
    # Strategy
    # ─────────
    # 1. Scan all pairs in a single loop.
    #    • Zero-ROI frames: record 0.0 features instantly (no GPU work).
    #    • YOLO-ROI frames: crop → grayscale → enqueue.
    # 2. Every _RAFT_BATCH_SIZE enqueued pairs, call _flush_raft_batch which
    #    stacks all crops into one (B, 3, H, W) tensor and runs a single RAFT
    #    forward pass, returning (B, 2, H, W) flow for the whole batch.
    # 3. Flush the last partial batch (<32 pairs) after the loop ends.
    # 4. All results land in results_map[loop_i]; per_frame is assembled in
    #    frame order once everything is processed.
    # 5. If RAFT is unavailable, fall back to Farneback one-at-a-time.
    #    If a batch throws at runtime, Farneback handles that batch's pairs.

    _RAFT_BATCH_SIZE = 64       # max pairs per shape-bucket flush; halves on OOM
    _RAFT_MAX_SIDE   = 320      # cap for proportional resize before RAFT

    # loop_i → raw feature dict  (frame_index / timestamp / roi attached later)
    results_map: dict[int, dict] = {}

    # Shape-bucket dict: (tgt_h, tgt_w) → list of
    #   (loop_i, fidx, scaled_gp, scaled_gc, scaled_h, scaled_w)
    # Each pair is proportionally resized INDEPENDENTLY first; the bucket key
    # is the padded-to-×8 shape that results.  Pairs that coincidentally land
    # on the same padded shape are batched together; all others run alone.
    # Flow values are therefore identical to one-at-a-time processing.
    shape_buckets: dict[tuple[int, int], list[tuple]] = {}

    _first_batch_done = [False]
    _first_crop_done  = [False]

    def _proportional_resize(gray: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Resize *gray* so its longest side ≤ _RAFT_MAX_SIDE (aspect-preserving).

        Returns (resized, new_h, new_w).
        """
        orig_h, orig_w = gray.shape
        scale = min(_RAFT_MAX_SIDE / orig_h, _RAFT_MAX_SIDE / orig_w, 1.0)
        if scale < 1.0:
            new_h = max(8, int(round(orig_h * scale)))
            new_w = max(8, int(round(orig_w * scale)))
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), new_h, new_w
        return gray, orig_h, orig_w

    def _pad8(gray: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
        """Pad *gray* to (tgt_h, tgt_w) using edge values."""
        h, w = gray.shape
        ph, pw = tgt_h - h, tgt_w - w
        if ph == 0 and pw == 0:
            return gray
        return np.pad(gray, ((0, ph), (0, pw)), mode="edge")

    def _farneback_pairs(pairs: list[tuple]) -> None:
        """Farneback fallback: proportional-resize then run one at a time."""
        for item in pairs:
            loop_i, _, gp, gc = item[0], item[1], item[2], item[3]
            gp_r, _, _ = _proportional_resize(gp)
            gc_r, _, _ = _proportional_resize(gc)
            flow_fb = cv2.calcOpticalFlowFarneback(
                prev=_blur_for_flow(gp_r), next=_blur_for_flow(gc_r),
                flow=None,
                pyr_scale=_FB_PYR_SCALE, levels=_FB_LEVELS,
                winsize=_FB_WINSIZE,     iterations=_FB_ITERATIONS,
                poly_n=_FB_POLY_N,       poly_sigma=_FB_POLY_SIGMA,
                flags=_FB_FLAGS,
            )
            results_map[loop_i] = _flow_magnitude_features(flow_fb)

    def _flush_bucket(
        bucket: list[tuple],
        tgt_h: int,
        tgt_w: int,
        batch_size: int,
    ) -> None:
        """Run RAFT on one shape bucket in sub-batches of *batch_size*.

        Every item in *bucket* was already proportionally resized independently
        before being enqueued.  Here we only pad to (tgt_h, tgt_w) — the shared
        multiple-of-8 target — so every pair is processed at the same spatial
        resolution it would have at if run one at a time.

        OOM halves batch_size recursively; other errors fall back to Farneback.

        Bucket item tuple: (loop_i, fidx, scaled_gp, scaled_gc, scaled_h, scaled_w)
        """
        import torch as _t  # noqa: PLC0415

        if batch_size < 1:
            _farneback_pairs(bucket)
            return

        # tgt_h/tgt_w are already multiples of 8 (computed at enqueue time).
        if tgt_h < 128 or tgt_w < 128:
            _farneback_pairs(bucket)
            return

        if not _first_crop_done[0]:
            _, _, _, _, sh0, sw0 = bucket[0]
            print(
                f"[VIBRATION] Frame 0 crop: proportionally resized to {sw0}x{sh0} "
                f"→ padded to {tgt_w}x{tgt_h} for RAFT",
                flush=True,
            )
            _first_crop_done[0] = True

        # Preprocess: pad each pre-resized crop to (tgt_h, tgt_w).
        imgs1: list = []
        imgs2: list = []
        for loop_i, fidx, scaled_gp, scaled_gc, sh, sw in bucket:
            gp_p = _pad8(scaled_gp, tgt_h, tgt_w)
            gc_p = _pad8(scaled_gc, tgt_h, tgt_w)
            t1 = _t.from_numpy(np.stack([gp_p, gp_p, gp_p], axis=0)).to(_raft_device)
            t2 = _t.from_numpy(np.stack([gc_p, gc_p, gc_p], axis=0)).to(_raft_device)
            p1, p2 = _raft_preprocess(t1, t2)
            imgs1.append(p1)
            imgs2.append(p2)

        amp_ctx = (
            _t.autocast(device_type="cuda", dtype=_t.float16)  # type: ignore[attr-defined]
            if _raft_device == "cuda"
            else _t.no_grad()
        )

        for start in range(0, len(bucket), batch_size):
            sub_items = bucket[start : start + batch_size]
            sub_t1    = _t.stack(imgs1[start : start + batch_size])
            sub_t2    = _t.stack(imgs2[start : start + batch_size])

            if _raft_device == "cuda" and not _first_batch_done[0]:
                used  = _t.cuda.memory_allocated() / 1024 ** 2
                total = _t.cuda.get_device_properties(0).total_memory / 1024 ** 2
                print(
                    f"[VIBRATION] VRAM before inference: {used:.0f}MB / {total:.0f}MB",
                    flush=True,
                )

            try:
                with _t.no_grad(), amp_ctx:
                    preds = _raft_model(sub_t1, sub_t2)

                if _raft_device == "cuda" and not _first_batch_done[0]:
                    used_after = _t.cuda.memory_allocated() / 1024 ** 2
                    print(
                        f"[VIBRATION] First batch done — VRAM used: {used_after:.0f}MB",
                        flush=True,
                    )
                    _first_batch_done[0] = True

                flow_full = preds[-1].float()   # (B, 2, tgt_h, tgt_w)
                for b_idx, (loop_i, _, _, _, sh, sw) in enumerate(sub_items):
                    # Unpad: take only the proportionally-resized region.
                    flow_np = (
                        flow_full[b_idx, :, :sh, :sw]
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    results_map[loop_i] = _flow_magnitude_features(flow_np)

            except _t.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
                half = batch_size // 2
                print(
                    f"[VIBRATION] OOM at batch {batch_size} "
                    f"(shape {tgt_w}x{tgt_h}), retrying with {half}",
                    flush=True,
                )
                _t.cuda.empty_cache()
                _flush_bucket(sub_items, tgt_h, tgt_w, half)

            except Exception as exc:  # noqa: BLE001
                print(
                    f"[VIBRATION] RAFT batch failed ({exc}), "
                    f"Farneback fallback for {len(sub_items)} pairs",
                    flush=True,
                )
                _farneback_pairs(sub_items)

    # ── Pair scan — proportional resize each crop independently, then bucket ──
    _zero_feats = {"mean_magnitude": 0.0, "peak_magnitude": 0.0, "motion_area_ratio": 0.0}

    for i in range(len(frames_bgr) - 1):
        fidx, prev_bgr_f = frames_bgr[i]
        _,    curr_bgr_f = frames_bgr[i + 1]
        roi              = roi_lookup.get(fidx)

        if roi is None:
            results_map[i] = dict(_zero_feats)

        elif _raft_model is not None:
            x1, y1, x2, y2 = [max(0, int(v)) for v in roi]
            x2 = min(frame_width,  x2)
            y2 = min(frame_height, y2)
            gp = _to_gray(prev_bgr_f[y1:y2, x1:x2])
            gc = _to_gray(curr_bgr_f[y1:y2, x1:x2])
            if gp.size == 0 or gc.size == 0:
                results_map[i] = dict(_zero_feats)
            else:
                # 1. Proportionally resize each crop independently.
                gp_s, sh, sw = _proportional_resize(gp)
                gc_s, _,  _  = _proportional_resize(gc)
                # 2. Compute padded-to-×8 target — this is the bucket key.
                tgt_h = sh + (8 - sh % 8) % 8
                tgt_w = sw + (8 - sw % 8) % 8
                # 3. Reject crops too small for RAFT.
                if tgt_h < 128 or tgt_w < 128:
                    _farneback_pairs([(i, fidx, gp_s, gc_s, sh, sw)])
                else:
                    key = (tgt_h, tgt_w)
                    shape_buckets.setdefault(key, []).append((i, fidx, gp_s, gc_s, sh, sw))
                    if len(shape_buckets[key]) >= _RAFT_BATCH_SIZE:
                        _flush_bucket(shape_buckets.pop(key), tgt_h, tgt_w, _RAFT_BATCH_SIZE)

        else:
            # RAFT unavailable — Farneback one-at-a-time.
            results_map[i] = _compute_flow_in_roi(
                prev_bgr_f, curr_bgr_f, roi,
                frame_width, frame_height,
                False, None,
            )

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"[VIBRATION] {i + 1}/{total_video_frames - 1} pairs  ({elapsed:.1f}s)",
                flush=True,
            )

    # Flush all remaining partial buckets (each gets its own RAFT forward pass).
    for (tgt_h, tgt_w), bucket in shape_buckets.items():
        _flush_bucket(bucket, tgt_h, tgt_w, _RAFT_BATCH_SIZE)
    shape_buckets.clear()

    # ── Assemble per_frame in frame order ─────────────────────────────────────
    per_frame: list[dict] = []
    for i in range(len(frames_bgr) - 1):
        fidx  = frames_bgr[i][0]
        feats = dict(results_map.get(i, _zero_feats))
        feats["frame_index"]   = fidx
        feats["timestamp_sec"] = round(fidx / fps, 6)
        feats["roi_used"]      = fidx in roi_lookup
        feats["roi"]           = roi_lookup.get(fidx)
        per_frame.append(feats)

    print(
        f"[VIBRATION] Flow extraction done: {len(per_frame)} frame pairs",
        flush=True,
    )

    if not per_frame:
        return _empty_result(run_id, fps, 0, output_dir)

    # ── FFT vibration analysis ────────────────────────────────────────────────
    raw_mags    = [f["mean_magnitude"] for f in per_frame]
    fft_windows = _compute_fft_windows(raw_mags, fps)
    print(f"[VIBRATION] FFT windows computed: {len(fft_windows)}", flush=True)
    print("[VIBRATION DEBUG] All FFT windows:", flush=True)
    for _w in fft_windows:
        print(
            f"  t={_w['timestamp_sec']:.1f}s   "
            f"freq={_w['dominant_freq_hz']:.2f}Hz  "
            f"conf={_w['confidence']:.3f}",
            flush=True,
        )

    window_size = max(8, int(_FFT_WINDOW_SEC * fps))
    vibration_events = _classify_vibration_events(fft_windows, window_size, fps)
    print(
        f"[VIBRATION] Vibration events detected: {len(vibration_events)}",
        flush=True,
    )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    mags_arr       = np.asarray(raw_mags, dtype=np.float32)
    avg_magnitude  = round(float(mags_arr.mean()), 6)
    peak_magnitude = round(float(mags_arr.max()),  6)

    processing_time = round(time.perf_counter() - t_start, 3)

    # ── Build JSON payloads ───────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "run_id":              run_id,
        "fps":                 fps,
        "total_frames":        len(per_frame),
        "vibration_detected":  len(vibration_events) > 0,
        "vibration_events":    vibration_events,
        "avg_magnitude":       avg_magnitude,
        "peak_magnitude":      peak_magnitude,
        "frame_width":         frame_width,
        "frame_height":        frame_height,
        "processing_time_sec": processing_time,
    }

    raw_payload: dict[str, Any] = {
        "run_id":             run_id,
        "fps":                fps,
        "total_frames":       len(per_frame),
        "per_frame_features": per_frame,
        "vibration_windows":  fft_windows,
    }

    # ── Write files ───────────────────────────────────────────────────────────
    raw_path     = os.path.join(output_dir, "vibration_raw.json")
    summary_path = os.path.join(output_dir, "vibration_summary.json")

    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw_payload, fh, indent=2)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary["raw_json_path"]     = raw_path
    summary["summary_json_path"] = summary_path

    print(
        f"[VIBRATION] Done in {processing_time:.1f}s — "
        f"events={len(vibration_events)}  "
        f"avg_mag={avg_magnitude:.4f}  "
        f"peak_mag={peak_magnitude:.4f}",
        flush=True,
    )
    print(f"[VIBRATION] Saved → {summary_path}", flush=True)
    return summary
