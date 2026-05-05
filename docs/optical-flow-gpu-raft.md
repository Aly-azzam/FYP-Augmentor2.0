# GPU Optical Flow Integration using RAFT

## 1. Overview

The AugMentor optical-flow pipeline originally relied on **classical Farneback dense optical flow** on the **CPU**, implemented through **OpenCV** (`cv2.calcOpticalFlowFarneback`). To modernise motion analysis and make better use of GPU hardware, we extended the codebase with a **GPU-ready optical-flow architecture** and integrated **RAFT** (Recurrent All-Pairs Field Transforms), a **deep-learning optical flow** model, using **PyTorch** and **Torchvision**. OpenCV **CUDA Farneback** is also available as an optional path inside a dedicated helper module, with automatic fallback to CPU Farneback when CUDA is not available or fails.

Together, these changes preserve the same downstream contract (dense flow as a NumPy array) while allowing experiments with GPU-accelerated and learning-based flow estimation.

---

## 2. Motivation

GPU support and RAFT integration were added for several practical and research-oriented reasons:

- **Reduce CPU dependency** — Large videos and dense per-frame flow are expensive on CPU; GPUs can offload heavy compute when available.
- **Experiment with deep learning optical flow** — RAFT is a strong baseline for learned dense flow and enables comparison with classical methods.
- **Improve motion estimation quality** — In many scenes, learned methods can capture finer or more coherent motion than classical Farneback (trade-offs apply; see Performance Notes).
- **Prepare for scalable GPU-based analysis** — Aligns the project with future pipelines (batch processing, larger models, other vision stages on GPU).

---

## 3. Files Added

### `backend/app/services/optical_flow/gpu_farneback_service.py`

Optional **OpenCV CUDA Farneback** path with **safe CPU fallback**:

- Tries **CUDA Farneback** when an OpenCV build with **CUDA** is present (CUDA device available, Farneback CUDA API present).
- On failure or unavailable GPU, falls back to **`cv2.calcOpticalFlowFarneback`** with the same configuration fields as the main pipeline.
- Returns flow as **`numpy.ndarray`**, shape **`(H, W, 2)`**, **`dtype float32`**, matching the CPU Farneback contract.

### `backend/app/services/optical_flow/raft_gpu_service.py`

**RAFT-small** via **Torchvision** on **GPU when CUDA is available**:

- Loads **`raft_small`** with **`Raft_Small_Weights.DEFAULT`** (cached after first load).
- Runs inference with **`torch.no_grad()`**; inputs are preprocessed using the weights’ transforms, with padding aligned to network expectations where applicable.
- Outputs dense flow in the same format: **`(H, W, 2)`**, **`float32`** (`dx`, `dy`).
- If inference or environment setup fails, raises clear **`RuntimeError`** messages (callers may catch and fall back).

---

## 4. Files Modified

### `backend/app/services/optical_flow/farneback_service.py`

The main video optical-flow pipeline no longer calls Farneback **only** through raw OpenCV at every site. Instead it uses a **backend selection layer**:

- When RAFT is enabled and **PyTorch CUDA** is available, the pipeline attempts **`compute_raft_flow`** for both **ROI crops** and **full-frame** pairs.
- On RAFT failure (or when CUDA is unavailable / RAFT disabled), it falls back to **`compute_farneback_flow_auto`** (OpenCV CUDA Farneback when possible, else CPU Farneback), preserving previous Farneback-oriented behaviour.

**ROI logic, feature extraction, embedding ROI flow into the full canvas, and downstream schemas are unchanged in intent** — only the source of the dense flow tensor is selected dynamically.

---

## 5. Backend Logic

Current behaviour in the integrated pipeline:

1. **If** the RAFT path is enabled **and** **`torch.cuda.is_available()`** is **`True`**, the code **tries RAFT** (`compute_raft_flow`) on the same blurred grayscale / ROI tensors used for Farneback.
2. **If** RAFT raises an exception (e.g. ROI too small for RAFT minimum size, OOM, missing weights), the code **logs** and **falls back** to **`compute_farneback_flow_auto`** (Farneback with optional OpenCV CUDA, then CPU).
3. **If** CUDA is not available or RAFT is disabled, the pipeline uses **`compute_farneback_flow_auto`** directly.

**Output format is unchanged for the rest of the stack:**

- **`numpy.ndarray`**
- **Shape:** **`(H, W, 2)`**
- **`dtype:`** **`float32`**
- Channel **0** = horizontal component, channel **1** = vertical component (consistent with the existing Farneback-based pipeline).

---

## 6. GPU Environment Setup

Install a **CUDA-enabled** PyTorch stack (example uses **CUDA 11.8** wheels; adjust the index URL if your driver/CUDA version differs — see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).

```bash
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Sanity check:**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see `CUDA available: True` and your GPU name when drivers and the wheel match.

---

## 7. Test Command

From the **`backend`** directory (so `app` and `scripts` resolve correctly):

```bash
python scripts\run_optical_flow_standalone.py --expert-video "storage\uploads\test_video.mp4" --learner-video "storage\uploads\test_video.mp4" --save-visualizations
```

Adjust paths if your test videos live elsewhere.

---

## 8. Expected Logs

When RAFT loads successfully on **CUDA**, you should see (once per process after first load):

```text
[OF-RAFT] Loaded RAFT small on cuda
```

Pipeline-level messages may also include:

```text
[OF] Backend: RAFT GPU if available, else Farneback CPU
```

If RAFT fails for a frame, a fallback line similar to:

```text
[OF] RAFT failed, fallback to Farneback: ...
```

---

## 9. Performance Notes

Be transparent about trade-offs:

- **RAFT** produces **learned, dense optical flow** and can yield **higher-quality motion estimates** in challenging textures and motion patterns than classical Farneback in some cases.
- On modest GPUs (e.g. **GTX 1650**), **RAFT inference is often slower per frame** than **classical CPU Farneback** for typical resolutions — deep networks dominate cost even when CUDA is used.
- **Farneback** remains **faster** for quick demos and long videos where throughput matters.
- **RAFT** is kept as an **optional** backend when CUDA is available, for experiments and higher-quality offline analysis rather than guaranteed real-time performance on all hardware.

---

## 10. Final Decision

This branch **keeps both** backends:

| Backend | Role |
|--------|------|
| **Farneback (CPU / optional OpenCV CUDA via `compute_farneback_flow_auto`)** | Default path when RAFT is skipped or unavailable; **speed** and **stability**. |
| **RAFT-small (PyTorch / Torchvision, CUDA)** | **Optional** deep-learning flow when GPU and dependencies are available; **quality** and **research** experiments. |

No single backend is forced; the integration layer chooses based on flags and runtime capability.

---

## 11. Notes for Team Members

1. **Pull** the latest branch that contains `gpu_farneback_service.py`, `raft_gpu_service.py`, and the updated `farneback_service.py`.
2. **If you have an NVIDIA GPU**, install **CUDA-enabled PyTorch** (see [GPU Environment Setup](#6-gpu-environment-setup)) and verify with the one-line CUDA check.
3. **If CUDA is unavailable** (CPU-only PyTorch, no GPU, or driver issues), the project **still runs**: RAFT is skipped and **Farneback** (**`compute_farneback_flow_auto`**) provides flow — **no separate CPU-only fork** is required for day-to-day development.
4. **OpenCV CUDA Farneback** requires an OpenCV build with CUDA modules; many standard `pip install opencv-python` installs are CPU-only — the helper falls back to CPU Farneback automatically.

---

*Document version: aligned with the optical-flow GPU / RAFT integration branch for FYP reporting.*
