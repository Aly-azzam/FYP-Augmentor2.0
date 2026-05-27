"""RAFT small optical flow service.

Wraps torchvision's RAFT small model as a CUDA-backed singleton whose public
interface matches ``cv2.calcOpticalFlowFarneback``:

    flow = compute_raft_flow(prev_gray, curr_gray)  # (H, W, 2) float32

Use ``_check_raft_available()`` to gate callers without importing torch at
module scope so the rest of the app works even when GPU/torchvision are absent.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np


# ── Availability ────────────────────────────────────────────────────────────

def _check_raft_available() -> bool:
    """Return True only if CUDA is available AND torchvision can be imported."""
    try:
        import torch as _torch  # type: ignore  # noqa: PLC0415

        if not _torch.cuda.is_available():
            return False

        import torchvision.models.optical_flow  # type: ignore  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


# ── Module-level graceful fallback notice ───────────────────────────────────

def _print_unavailable_reason() -> None:
    try:
        import torch as _torch  # type: ignore  # noqa: PLC0415

        if not _torch.cuda.is_available():
            print(
                "[RAFT] CUDA is not available on this machine. "
                "RAFT flow will not be used.",
                flush=True,
            )
            return
    except ImportError:
        print(
            "[RAFT] torch is not installed. RAFT flow will not be used.",
            flush=True,
        )
        return

    try:
        import torchvision.models.optical_flow  # type: ignore  # noqa: PLC0415, F401
    except ImportError:
        print(
            "[RAFT] torchvision is not installed or does not expose "
            "torchvision.models.optical_flow. RAFT flow will not be used.",
            flush=True,
        )


if not _check_raft_available():
    _print_unavailable_reason()


# ── Singleton model ──────────────────────────────────────────────────────────

# (_RaftModel, _preprocess_transform, device_str) or None
_RAFT_INSTANCE: tuple | None = None
_RAFT_LOCK = threading.Lock()


def _get_raft_model() -> tuple:
    """Load RAFT small onto CUDA exactly once (thread-safe) and return (model, preprocess, device)."""
    global _RAFT_INSTANCE  # noqa: PLW0603

    if _RAFT_INSTANCE is not None:
        return _RAFT_INSTANCE

    with _RAFT_LOCK:
        if _RAFT_INSTANCE is not None:
            return _RAFT_INSTANCE

        import torch  # type: ignore  # noqa: PLC0415
        from torchvision.models.optical_flow import (  # type: ignore  # noqa: PLC0415
            Raft_Small_Weights,
            raft_small,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RAFT] Loading RAFT small weights on device={device} …", flush=True)

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
        model = model.to(device).eval()
        preprocess = weights.transforms()

        _RAFT_INSTANCE = (model, preprocess, device)
        print("[RAFT] RAFT small loaded and ready (AMP autocast for half precision).", flush=True)
        return _RAFT_INSTANCE


# ── Padding helpers ──────────────────────────────────────────────────────────

def _pad_to_multiple_of_8(tensor: Any) -> tuple[Any, tuple[int, int]]:
    """Pad H and W up to the nearest multiple of 8.

    Args:
        tensor: float32 tensor of shape (B, C, H, W).

    Returns:
        padded tensor and (pad_h, pad_w) so callers can unpad later.
    """
    import torch.nn.functional as F  # type: ignore  # noqa: PLC0415

    _, _, H, W = tensor.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)

    # F.pad order is (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (pad_h, pad_w)


# ── Public API ───────────────────────────────────────────────────────────────

def compute_raft_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    max_height: int = 320,
    max_width: int = 320,
) -> np.ndarray:
    """Compute dense optical flow using RAFT small (CUDA, half precision).

    Accepts two single-channel grayscale uint8 frames of the same spatial size.
    Returns a dense flow array in the same (H, W, 2) format as
    ``cv2.calcOpticalFlowFarneback``: ``flow[..., 0]`` is horizontal
    displacement (dx) and ``flow[..., 1]`` is vertical displacement (dy),
    both in pixels at the **original** frame resolution.

    If either input dimension exceeds its cap both frames are resized
    proportionally (using the tighter of the two caps) before inference.
    Neither resized dimension is allowed to fall below 128 px (RAFT minimum).
    The resulting flow vectors are scaled back up to the original frame's
    coordinate space before returning.

    Args:
        prev_gray:  (H, W) uint8 numpy array — the previous frame.
        curr_gray:  (H, W) uint8 numpy array — the current frame.
        max_height: cap on the height passed into RAFT.  Default 320.
        max_width:  cap on the width passed into RAFT.   Default 320.

    Returns:
        (H, W, 2) float32 numpy array.

    Raises:
        RuntimeError: if RAFT is unavailable (CUDA or torchvision missing).
        ValueError:   if the input arrays have incompatible shapes or dtype.
    """
    if not _check_raft_available():
        raise RuntimeError(
            "[RAFT] RAFT is not available: CUDA is missing or torchvision "
            "is not installed. Use cv2.calcOpticalFlowFarneback instead."
        )

    if prev_gray.ndim != 2 or curr_gray.ndim != 2:
        raise ValueError(
            f"[RAFT] Both frames must be 2-D (H, W) grayscale arrays. "
            f"Got shapes {prev_gray.shape} and {curr_gray.shape}."
        )
    if prev_gray.shape != curr_gray.shape:
        raise ValueError(
            f"[RAFT] Frame shapes must match. "
            f"Got {prev_gray.shape} vs {curr_gray.shape}."
        )
    if prev_gray.dtype != np.uint8 or curr_gray.dtype != np.uint8:
        raise ValueError(
            f"[RAFT] Frames must be uint8. "
            f"Got {prev_gray.dtype} and {curr_gray.dtype}."
        )

    import cv2 as _cv2  # noqa: PLC0415
    import torch  # type: ignore  # noqa: PLC0415

    H, W = prev_gray.shape

    # ── Optional proportional downscale ────────────────────────────────────
    # Use the tighter of the two caps so neither dimension exceeds its limit.
    scale: float = min(max_height / H, max_width / W, 1.0)
    if scale < 1.0:
        new_h = max(128, int(round(H * scale)))
        new_w = max(128, int(round(W * scale)))
        prev_gray = _cv2.resize(prev_gray, (new_w, new_h), interpolation=_cv2.INTER_AREA)
        curr_gray = _cv2.resize(curr_gray, (new_w, new_h), interpolation=_cv2.INTER_AREA)
        print(
            f"[RAFT] Resized frames {H}x{W} -> {new_h}x{new_w} "
            f"(scale={scale:.3f}) to stay within max_height={max_height}, max_width={max_width}",
            flush=True,
        )

    model, preprocess, device = _get_raft_model()

    # Grayscale (H', W') → uint8 RGB tensor (3, H', W') by replicating channel.
    def _to_rgb_tensor(gray: np.ndarray) -> Any:
        rgb = np.stack([gray, gray, gray], axis=0)  # (3, H', W')
        return torch.from_numpy(rgb).to(device)

    t1 = _to_rgb_tensor(prev_gray)
    t2 = _to_rgb_tensor(curr_gray)

    # Preprocess: normalise uint8 → float32 in [-1, 1].
    t1_pre, t2_pre = preprocess(t1, t2)
    t1_pre = t1_pre.unsqueeze(0)  # (1, 3, H', W') float32
    t2_pre = t2_pre.unsqueeze(0)

    # Pad so H' and W' are multiples of 8 (RAFT requirement).
    t1_pad, _ = _pad_to_multiple_of_8(t1_pre)
    t2_pad, _ = _pad_to_multiple_of_8(t2_pre)

    Hs, Ws = prev_gray.shape  # scaled dims (may equal originals)

    # Free any previously cached CUDA allocations before the forward pass.
    if device == "cuda":
        torch.cuda.empty_cache()

    # Use AMP autocast for half-precision activations where cuDNN allows it
    # (e.g. conv layers).  Operations that require float32 (e.g. grid_sample)
    # are kept at full precision automatically, avoiding the cuDNN dtype error
    # that would occur with a fully static model.half() approach.
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
        if device == "cuda"
        else torch.no_grad()
    )
    with torch.no_grad(), amp_ctx:
        t_start = time.perf_counter()
        flow_predictions = model(t1_pad, t2_pad)
        print("[RAFT] inference_ms={:.1f}".format((time.perf_counter() - t_start) * 1000), flush=True)

    # RAFT returns a list of iterative predictions; the last is the finest.
    flow_padded: Any = flow_predictions[-1]  # (1, 2, H_pad, W_pad)

    # Unpad back to the scaled spatial size, then convert to float32.
    flow_tensor = flow_padded[:, :, :Hs, :Ws].float()  # (1, 2, H', W')

    # Permute to (H', W', 2) and move to CPU as float32.
    flow_np: np.ndarray = (
        flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    )

    # ── Scale flow vectors and resize back to original resolution ──────────
    if scale < 1.0:
        # The clamped dims may differ slightly from scale*H / scale*W, so
        # derive the per-axis ratios from the actual resized dimensions.
        scale_h = Hs / H
        scale_w = Ws / W
        flow_np[..., 0] /= scale_w  # dx component
        flow_np[..., 1] /= scale_h  # dy component
        flow_np = _cv2.resize(flow_np, (W, H), interpolation=_cv2.INTER_LINEAR)

    return flow_np
