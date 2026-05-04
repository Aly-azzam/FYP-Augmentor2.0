"""
GPU-aware Farneback optical flow helper (optional CUDA path + safe CPU fallback).

This module is **standalone**: it does not modify ``farneback_service.py``. It provides:

- ``compute_farneback_flow_auto`` — tries OpenCV CUDA Farneback when allowed, otherwise
  falls back to ``cv2.calcOpticalFlowFarneback`` with the same ``FarnebackConfig`` fields
  used by the CPU pipeline.

Output contract (matches CPU Farneback):

- ``numpy.ndarray``, shape ``(height, width, 2)``, ``dtype`` ``float32``.

OpenCV builds vary: some ship CUDA modules, some use different Python bind names
(``cuda_FarnebackOpticalFlow.create`` vs legacy helpers). This file uses defensive
detection and never raises from availability checks — failures degrade to CPU.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Module-level cache: avoid repeated CUDA probes and repeated fallback logs during video loops.
_CUDA_FARNEBACK_AVAILABLE_CACHE: bool | None = None
_CUDA_FALLBACK_LOGGED: bool = False


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def _ensure_grayscale_uint8(image: np.ndarray, *, name: str) -> np.ndarray:
    """Convert a frame to single-channel ``uint8`` for Farneback."""
    if image is None:
        raise ValueError(f"{name} must not be None.")

    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            gray = arr[:, :, 0]
        elif c == 3:
            # Match OpenCV default colour convention used across this codebase (BGR).
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                f"{name}: unsupported channel count {c}; expected 1, 3, or 4."
            )
    else:
        raise ValueError(f"{name}: expected 2D or 3D array, got shape {arr.shape}.")

    if gray.dtype != np.uint8:
        g = gray.astype(np.float64, copy=False)
        max_val = float(np.max(g)) if g.size else 0.0
        if max_val <= 1.0:
            g = g * 255.0
        g = np.clip(g, 0.0, 255.0)
        gray = np.round(g).astype(np.uint8)

    return gray


def _prepare_frame_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and return matching ``uint8`` grayscale frames."""
    if prev_gray is None or curr_gray is None:
        raise ValueError("prev_gray and curr_gray must not be None.")

    prev_u8 = _ensure_grayscale_uint8(prev_gray, name="prev_gray")
    curr_u8 = _ensure_grayscale_uint8(curr_gray, name="curr_gray")

    if prev_u8.shape != curr_u8.shape:
        raise ValueError(
            "prev_gray and curr_gray must have the same shape; "
            f"got {prev_u8.shape} vs {curr_u8.shape}."
        )
    return prev_u8, curr_u8


# -----------------------------------------------------------------------------
# CPU fallback (must mirror ``farneback_service`` Farneback calls)
# -----------------------------------------------------------------------------


def _compute_farneback_flow_cpu(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Run dense Farneback on CPU — same API as the main optical-flow pipeline."""
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=config.pyr_scale,
        levels=config.levels,
        winsize=config.winsize,
        iterations=config.iterations,
        poly_n=config.poly_n,
        poly_sigma=config.poly_sigma,
        flags=config.flags,
    )
    return np.asarray(flow, dtype=np.float32)


# -----------------------------------------------------------------------------
# CUDA Farneback factory (defensive across OpenCV Python builds)
# -----------------------------------------------------------------------------


def _get_cuda_device_count() -> int:
    """Return CUDA device count from OpenCV, or 0 if unavailable."""
    if not hasattr(cv2, "cuda"):
        return 0
    try:
        fn = getattr(cv2.cuda, "getCudaEnabledDeviceCount", None)
        if callable(fn):
            return int(fn())
    except Exception:
        return 0
    return 0


def _create_cuda_farneback_algorithm(config: Any) -> Any:
    """
    Instantiate ``cv::cuda::FarnebackOpticalFlow`` via Python bindings.

    Tries, in order:

    1. ``cv2.cuda_FarnebackOpticalFlow.create(...)``
    2. ``cv2.cuda_FarnebackOpticalFlow_create(...)`` (legacy / alternate bind)

    Parameter mapping matches ``cv2.calcOpticalFlowFarneback`` / ``FarnebackConfig``:

    - ``levels``  -> ``numLevels``
    - ``pyr_scale`` -> ``pyrScale``
    - ``winsize`` -> ``winSize``
    - ``iterations`` -> ``numIters``
    - ``poly_n`` -> ``polyN``
    - ``poly_sigma`` -> ``polySigma``
    - ``flags`` -> ``flags``

    ``fastPyramids`` is set to ``False`` for parity with the classic CPU API.
    """
    num_levels = int(config.levels)
    pyr_scale = float(config.pyr_scale)
    fast_pyramids = False
    win_size = int(config.winsize)
    num_iters = int(config.iterations)
    poly_n = int(config.poly_n)
    poly_sigma = float(config.poly_sigma)
    flags = int(config.flags)

    # --- Try 1: class-style ``cuda_FarnebackOpticalFlow.create`` ----------------
    cls = getattr(cv2, "cuda_FarnebackOpticalFlow", None)
    if cls is not None:
        create_fn = getattr(cls, "create", None)
        if callable(create_fn):
            try:
                return create_fn(
                    numLevels=num_levels,
                    pyrScale=pyr_scale,
                    fastPyramids=fast_pyramids,
                    winSize=win_size,
                    numIters=num_iters,
                    polyN=poly_n,
                    polySigma=poly_sigma,
                    flags=flags,
                )
            except TypeError:
                # Older bindings may require positional arguments only.
                return create_fn(
                    num_levels,
                    pyr_scale,
                    fast_pyramids,
                    win_size,
                    num_iters,
                    poly_n,
                    poly_sigma,
                    flags,
                )

    # --- Try 2: module-level legacy name --------------------------------------
    legacy_create = getattr(cv2, "cuda_FarnebackOpticalFlow_create", None)
    if callable(legacy_create):
        try:
            return legacy_create(
                numLevels=num_levels,
                pyrScale=pyr_scale,
                fastPyramids=fast_pyramids,
                winSize=win_size,
                numIters=num_iters,
                polyN=poly_n,
                polySigma=poly_sigma,
                flags=flags,
            )
        except TypeError:
            return legacy_create(
                num_levels,
                pyr_scale,
                fast_pyramids,
                win_size,
                num_iters,
                poly_n,
                poly_sigma,
                flags,
            )

    raise RuntimeError(
        "CUDA Farneback creator not found "
        "(no cuda_FarnebackOpticalFlow.create / cuda_FarnebackOpticalFlow_create)."
    )


def _new_gpu_mat() -> Any:
    """Construct an empty ``GpuMat`` — supports multiple OpenCV Python layouts."""
    ctor = getattr(cv2, "cuda_GpuMat", None)
    if callable(ctor):
        return ctor()
    cuda_mod = getattr(cv2, "cuda", None)
    gmat = getattr(cuda_mod, "GpuMat", None) if cuda_mod is not None else None
    if callable(gmat):
        return gmat()
    raise RuntimeError("OpenCV CUDA GpuMat constructor not available in this build.")


def _flow_to_float32_hw2(flow: np.ndarray, *, expected_hw: tuple[int, int]) -> np.ndarray:
    """Normalize downloaded flow to ``(H, W, 2)`` ``float32``."""
    out = np.asarray(flow, dtype=np.float32)
    h, w = expected_hw
    if out.ndim == 3 and out.shape[2] == 2:
        if out.shape[0] != h or out.shape[1] != w:
            raise ValueError(
                f"Flow shape {out.shape} does not match frame size {(h, w)}."
            )
        return out

    raise ValueError(
        f"Unexpected CUDA flow array shape {out.shape}; expected (H, W, 2) float flow."
    )


def _compute_farneback_flow_gpu(
    prev_u8: np.ndarray,
    curr_u8: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Run CUDA Farneback and return CPU ``float32`` flow ``(H, W, 2)``."""
    algorithm = _create_cuda_farneback_algorithm(config)

    gpu_prev = _new_gpu_mat()
    gpu_curr = _new_gpu_mat()
    gpu_flow = _new_gpu_mat()

    gpu_prev.upload(prev_u8)
    gpu_curr.upload(curr_u8)

    # Standard Farneback CUDA API: calc(prev, next, flow)
    algorithm.calc(gpu_prev, gpu_curr, gpu_flow)

    flow_cpu = gpu_flow.download()
    h, w = prev_u8.shape[:2]
    return _flow_to_float32_hw2(flow_cpu, expected_hw=(h, w))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def is_cuda_farneback_available() -> bool:
    """
    Return ``True`` only if this OpenCV build can run CUDA Farneback.

    Conditions:

    - ``cv2.cuda`` exists
    - ``cv2.cuda.getCudaEnabledDeviceCount() > 0``
    - A CUDA Farneback creator symbol exists (see ``_create_cuda_farneback_algorithm``)

    Result is computed once per process and reused (see ``_CUDA_FARNEBACK_AVAILABLE_CACHE``).
    """
    global _CUDA_FARNEBACK_AVAILABLE_CACHE

    if _CUDA_FARNEBACK_AVAILABLE_CACHE is not None:
        return _CUDA_FARNEBACK_AVAILABLE_CACHE

    available = False
    if hasattr(cv2, "cuda") and _get_cuda_device_count() > 0:
        cls = getattr(cv2, "cuda_FarnebackOpticalFlow", None)
        if cls is not None and hasattr(cls, "create"):
            available = True
        elif callable(getattr(cv2, "cuda_FarnebackOpticalFlow_create", None)):
            available = True

    _CUDA_FARNEBACK_AVAILABLE_CACHE = available
    return available


def compute_farneback_flow_auto(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: Any,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """
    Compute dense optical flow — CUDA Farneback when possible, else CPU Farneback.

    Parameters match ``cv2.calcOpticalFlowFarneback`` via ``FarnebackConfig``.

    Returns:
        ``np.ndarray`` of shape ``(H, W, 2)``, ``dtype`` ``float32``.
    """
    prev_u8, curr_u8 = _prepare_frame_pair(prev_gray, curr_gray)

    if not prefer_gpu:
        return _compute_farneback_flow_cpu(prev_u8, curr_u8, config)

    # GPU path: all prerequisites must hold; any failure -> CPU with warning.
    try:
        if not hasattr(cv2, "cuda"):
            raise RuntimeError("cv2.cuda is missing (OpenCV built without CUDA modules).")
        if _get_cuda_device_count() <= 0:
            raise RuntimeError("No CUDA-enabled OpenCV device (getCudaEnabledDeviceCount() == 0).")
        if not is_cuda_farneback_available():
            raise RuntimeError(
                "CUDA Farneback creator or device check failed (see is_cuda_farneback_available)."
            )

        flow_gpu = _compute_farneback_flow_gpu(prev_u8, curr_u8, config)
        print("[OF-GPU] Using CUDA Farneback Optical Flow", flush=True)
        return flow_gpu
    except Exception as exc:
        global _CUDA_FALLBACK_LOGGED
        if not _CUDA_FALLBACK_LOGGED:
            print(
                f"[OF-GPU] CUDA Farneback unavailable or failed, falling back to CPU: {exc}",
                flush=True,
            )
            _CUDA_FALLBACK_LOGGED = True
        return _compute_farneback_flow_cpu(prev_u8, curr_u8, config)
