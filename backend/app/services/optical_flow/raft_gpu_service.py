"""
RAFT optical flow backend (Torchvision ``raft_small`` on GPU when available).

This module is **standalone**: it does not replace the Farneback CPU pipeline.
It exposes helpers to run pretrained RAFT-small between two frames and return a
dense flow field compatible with the rest of the optical-flow stack:

- ``numpy.ndarray``, shape ``(H, W, 2)``, ``dtype`` ``float32``
- channel 0 = horizontal displacement ``dx``, channel 1 = ``dy``

References:
- Torchvision RAFT: https://pytorch.org/vision/stable/models/optical_flow.html

Minimum input size:
- Torchvision's RAFT correlation pyramid expects sufficiently large feature maps;
  effectively inputs should be at least **128×128** pixels after preprocessing.
  Smaller crops raise ``RuntimeError`` with an explicit message.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F  # noqa: N812  — required alias for FYP spec
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

# -----------------------------------------------------------------------------
# Cached model (load once per process — RAFT weights are ~4MB but init still costly)
# -----------------------------------------------------------------------------

_RAFT_MODEL: torch.nn.Module | None = None
_RAFT_DEVICE: torch.device | None = None

# RAFT expects spatial sizes above a minimum (internal downsampling × correlation grid).
_MIN_RAFT_SIDE = 128
_PAD_MULTIPLE = 8


def is_torch_cuda_available() -> bool:
    """Return ``True`` if PyTorch can run kernels on a CUDA device."""
    return bool(torch.cuda.is_available())


def get_raft_device() -> torch.device:
    """Pick ``cuda`` when available, otherwise ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raft_model() -> torch.nn.Module:
    """
    Load Torchvision ``raft_small`` with ``Raft_Small_Weights.DEFAULT`` once and cache it.

    Returns:
        The cached RAFT-small module in eval mode on ``_RAFT_DEVICE``.
    """
    global _RAFT_MODEL, _RAFT_DEVICE

    if _RAFT_MODEL is not None:
        return _RAFT_MODEL

    try:
        device = get_raft_device()
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
        model = model.to(device)
        model.eval()

        _RAFT_MODEL = model
        _RAFT_DEVICE = device

        if device.type == "cuda":
            print("[OF-RAFT] Loaded RAFT small on cuda", flush=True)
        else:
            print("[OF-RAFT] Loaded RAFT small on cpu", flush=True)

        return model
    except Exception as exc:
        raise RuntimeError(f"[OF-RAFT] Failed to load RAFT small model: {exc}") from exc


def _numpy_frame_to_rgb_uint8(frame: np.ndarray, *, name: str) -> np.ndarray:
    """Convert BGR or grayscale ``uint8`` / ``float`` image to RGB ``uint8`` ``HxWx3``."""
    if frame is None:
        raise ValueError(f"{name} must not be None.")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        raise RuntimeError(
            f"[OF-RAFT] {name}: unsupported shape {arr.shape}; expected HxW or HxWx3/4."
        )

    if rgb.dtype != np.uint8:
        x = rgb.astype(np.float64)
        if np.nanmax(x) <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0.0, 255.0)
        rgb = np.round(x).astype(np.uint8)

    return rgb


def _torchvision_preprocess_pair(
    tensor_chw_uint8_a: torch.Tensor,
    tensor_chw_uint8_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``Raft_Small_Weights.DEFAULT.transforms()`` to a CHW pair."""
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms(tensor_chw_uint8_a, tensor_chw_uint8_b)


def _pad_chw_to_multiple(
    tensor_chw: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Pad a ``[3, H, W]`` tensor with replicate padding on bottom/right so ``H`` and ``W``
    are multiples of ``multiple``.

    Returns:
        (padded_tensor, (orig_h, orig_w)) spatial sizes **before** padding.
    """
    _, h, w = tensor_chw.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor_chw, (h, w)

    # torch.nn.functional.pad on 4D NCHW: pad = (left, right, top, bottom) for last dims.
    t4 = tensor_chw.unsqueeze(0)
    t4 = torch.nn.functional.pad(t4, (0, pad_w, 0, pad_h), mode="replicate")
    return t4.squeeze(0), (h, w)


def _crop_flow_to_original(
    flow_bchw: torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> torch.Tensor:
    """Crop ``[1, 2, H_pad, W_pad]`` flow to the unpadded ``orig_h × orig_w`` region."""
    return flow_bchw[:, :, :orig_h, :orig_w]


def compute_raft_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """
    Compute optical flow with RAFT-small between ``prev_frame`` and ``curr_frame``.

    Args:
        prev_frame: Previous frame (BGR, RGB, or grayscale numpy array).
        curr_frame: Current frame; must match ``prev_frame`` spatial shape.

    Returns:
        Flow field ``float32`` of shape ``(H, W, 2)`` — ``[..., 0]`` = ``dx``, ``[..., 1]`` = ``dy``.

    Raises:
        RuntimeError: On invalid inputs, unsupported sizes, or inference failures.
    """
    if prev_frame.shape[:2] != curr_frame.shape[:2]:
        raise RuntimeError(
            "[OF-RAFT] prev_frame and curr_frame must have the same height and width."
        )

    prev_rgb = _numpy_frame_to_rgb_uint8(prev_frame, name="prev_frame")
    curr_rgb = _numpy_frame_to_rgb_uint8(curr_frame, name="curr_frame")

    h0, w0 = prev_rgb.shape[:2]
    if min(h0, w0) < _MIN_RAFT_SIDE:
        raise RuntimeError(
            f"[OF-RAFT] RAFT requires frames at least {_MIN_RAFT_SIDE} px per side after RGB "
            f"conversion; got {h0}×{w0}. Use larger crops or full-resolution frames."
        )

    try:
        # uint8 HWC -> CHW tensor on CPU first (weights.transforms handles scaling).
        t_prev = torch.from_numpy(prev_rgb).permute(2, 0, 1).contiguous()
        t_curr = torch.from_numpy(curr_rgb).permute(2, 0, 1).contiguous()

        t_prev, t_curr = _torchvision_preprocess_pair(t_prev, t_curr)

        if t_prev.shape != t_curr.shape:
            raise RuntimeError("[OF-RAFT] Internal preprocess produced mismatched tensor shapes.")

        _, hp, wp = t_prev.shape
        if min(hp, wp) < _MIN_RAFT_SIDE:
            raise RuntimeError(
                f"[OF-RAFT] After preprocessing, spatial size {hp}×{wp} is below RAFT minimum "
                f"{_MIN_RAFT_SIDE}."
            )

        t_prev, orig_hw = _pad_chw_to_multiple(t_prev, _PAD_MULTIPLE)
        t_curr, _ = _pad_chw_to_multiple(t_curr, _PAD_MULTIPLE)
        orig_h, orig_w = orig_hw

        device = get_raft_device()
        model = load_raft_model()

        # Batch dimension expected by RAFT: [1, 3, H, W]
        batch_prev = t_prev.unsqueeze(0).to(device)
        batch_curr = t_curr.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(batch_prev, batch_curr)

        if not isinstance(predictions, list) or len(predictions) == 0:
            raise RuntimeError("[OF-RAFT] RAFT returned an empty prediction list.")

        flow_tensor = predictions[-1]
        # Finest resolution is last; shape [1, 2, H_pad, W_pad]
        flow_tensor = _crop_flow_to_original(flow_tensor, orig_h, orig_w)

        flow_np = flow_tensor.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        return np.asarray(flow_np, dtype=np.float32)

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"[OF-RAFT] RAFT inference failed: {exc}") from exc


def raft_backend_info() -> dict:
    """
    Diagnostic bundle for logging / quick health checks (no inference).

    Returns:
        Dict with CUDA availability, chosen device string, and library versions.
    """
    cuda_ok = is_torch_cuda_available()
    return {
        "torch_cuda_available": cuda_ok,
        "device": "cuda" if cuda_ok else "cpu",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
