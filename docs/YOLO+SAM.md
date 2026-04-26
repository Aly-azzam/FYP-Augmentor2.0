# YOLO + SAM2 Scissors Tracking in AugMentor 2.0

## 1. Overview

The YOLO + SAM2 module in AugMentor 2.0 tracks scissors/tool movement in expert and learner videos.

The main purpose of this module is to extract a reusable tool trajectory and region over time. These outputs are designed for later evaluation, where the learner movement can be compared against a preprocessed expert reference.

This module is not only a visual overlay generator. It produces reusable files that support:

- tool trajectory comparison
- region comparison
- mistake localization
- circle-the-mistake validation
- visual feedback in Compare Studio
- future expert-vs-learner evaluation

The implemented pipeline follows this flow:

```text
YOLO detects the scissors
-> YOLO bbox center becomes a SAM2 positive point prompt
-> YOLO bbox is shrunk and used as a SAM2 box prompt
-> SAM2 tracks the scissors/tool across extracted stride frames
-> trajectory and region data are extracted
-> raw.json, metrics.json, summary.json, and overlay video are saved
-> Compare Studio displays the result
```

The current backend implementation is isolated under:

```text
backend/app/services/sam2_yolo/
```

The FastAPI router is:

```text
backend/app/api/sam2_yolo.py
```

The Compare Studio integration is in:

```text
frontend/src/pages/CompareStudio.tsx
```

## 2. Why YOLO and SAM2 Are Used Together

YOLO and SAM2 have different roles.

YOLO is used for object detection. It finds the scissors and returns a bounding box around the object.

SAM2 is used for video object segmentation and tracking. It receives the YOLO detection as an initial prompt and tracks the same object across the processed frames.

This combination was chosen because SAM2 needs a good initial prompt. Earlier approaches using MediaPipe as the SAM2 prompt source were not reliable for scissors tracking because MediaPipe tracks the hand, not the tool. Since the goal is to track scissors trajectory, YOLO is a better prompt source.

The system therefore uses:

- YOLO for initialization
- SAM2 for tracking

## 3. YOLO Training and Dataset Preparation

A custom scissors detection dataset was created for the project. The dataset was built from egocentric task videos where the scissors appear from the learner/expert point of view.

The dataset includes frames where the scissors may appear:

- held by the user
- close to the hand
- partially occluded
- at different angles
- during cutting movement
- in different positions inside the frame

Frames were extracted from videos using OpenCV, then uploaded to Roboflow for annotation and training.

Example frame extraction logic:

```python
import cv2
from pathlib import Path

video_path = "expert_cutting.mp4"
frames_dir = Path("frames_expert")
frames_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)

i = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if i % 5 == 0:
        frame_path = frames_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved += 1

    i += 1

cap.release()

print(f"Saved {saved} frames inside '{frames_dir}' folder")
```

The annotation class used for the object was:

```text
scissors
```

The bounding boxes were drawn around the visible scissors. The goal at this stage was not pixel-perfect segmentation. The detector only needs to be accurate enough to provide a stable prompt for SAM2.

## 4. Roboflow and Environment Configuration

The backend settings are defined in:

```text
backend/app/core/config.py
```

Relevant environment variables are:

```env
ROBOFLOW_API_KEY=YOUR_PRIVATE_API_KEY
ROBOFLOW_MODEL_ID=scissors_ego_real/1
ROBOFLOW_CONFIDENCE=0.5
FRAME_STRIDE=5
SAM2_MAX_PROCESSED_FRAMES=
SAM2_YOLO_BBOX_SHRINK=0.65
SAM2_RUN_MODE=subprocess
```

The API key must remain private and must not be committed to GitHub.

Current defaults in `Settings`:

- `ROBOFLOW_API_KEY: str | None = None`
- `ROBOFLOW_MODEL_ID: str = "scissors_ego_real/1"`
- `ROBOFLOW_CONFIDENCE: float = 0.5`
- `FRAME_STRIDE: int = 5`
- `SAM2_MAX_PROCESSED_FRAMES: int | None = None`
- `SAM2_YOLO_BBOX_SHRINK: float = 0.65`
- `SAM2_RUN_MODE: str = "subprocess"`

Important implementation note: the service code currently uses hardcoded/default stride `5` in `backend/app/services/sam2_yolo/runner.py` and `backend/app/services/sam2_yolo/stable_runner.py`. The API request also defaults to `DEFAULT_FRAME_STRIDE`, which is `5`.

## 5. Backend File Map

The active YOLO+SAM2 implementation files are:

```text
backend/app/services/sam2_yolo/__init__.py
backend/app/services/sam2_yolo/yolo_scissors_detector.py
backend/app/services/sam2_yolo/sam2_video_tracker.py
backend/app/services/sam2_yolo/stable_runner.py
backend/app/services/sam2_yolo/runner.py
backend/app/services/sam2_yolo/schemas.py
backend/app/services/sam2_yolo/trajectory_metrics.py
backend/app/services/sam2_yolo/region_metrics.py
backend/app/services/sam2_yolo/visualization.py
```

Supporting backend files:

```text
backend/app/api/sam2_yolo.py
backend/app/scripts/process_expert_sam2_yolo.py
backend/app/main.py
backend/app/core/config.py
backend/requirements.txt
backend/vendor/sam2/
```

Frontend file:

```text
frontend/src/pages/CompareStudio.tsx
```

Model/config support:

```text
backend/models/sam2/sam2.1_hiera_t.yaml
backend/vendor/sam2/
```

The SAM2 code is made importable by `_ensure_local_sam2_on_path()` in `backend/app/services/sam2_yolo/stable_runner.py`, which inserts:

```text
backend/vendor/sam2
```

into `sys.path` if that folder exists.

## 6. YOLO Detection Logic

YOLO is used to find the scissors prompt at the beginning of tracking. It is not used for full-frame tracking across the entire video.

Main detection functions:

```text
detect_scissors_prompt()
find_initial_yolo_prompt_frame()
detect_best_scissors_roboflow()
shrink_bbox()
save_debug_first_frame()
```

These are implemented in:

```text
backend/app/services/sam2_yolo/yolo_scissors_detector.py
```

The system scans stride-aligned frames until a valid scissors detection is found. This is important because the scissors may not be clearly visible in the first frame.

Current scan behavior:

- `detect_scissors_prompt()` calls `find_initial_yolo_prompt_frame()`
- `frame_stride` defaults to `5`
- `max_scanned_frames` defaults to `30`
- only frames where `frame_index % frame_stride == 0` are checked
- Roboflow inference uses `InferenceHTTPClient`
- API URL is `https://serverless.roboflow.com`
- model id comes from `ROBOFLOW_MODEL_ID`
- confidence threshold comes from `ROBOFLOW_CONFIDENCE` or request/CLI argument

YOLO predictions are converted from Roboflow `x, y, width, height` format into:

```text
[x1, y1, x2, y2]
```

using `_roboflow_xywh_to_xyxy()`.

The bbox center is calculated by `YoloScissorsDetection.point` in `schemas.py`:

```python
point_x = (x1 + x2) / 2
point_y = (y1 + y2) / 2
```

This point becomes the SAM2 positive point prompt.

## 7. Detection Filtering

The intended design is to reject detections that are obviously unsuitable for SAM2 prompting. The current code implements a simpler filter than the original design text.

Current implemented filtering in `detect_best_scissors_roboflow()`:

- rejects predictions below `confidence_threshold`
- prefers detections whose class name contains `"scissor"`
- falls back to any confident detection if no class name contains `"scissor"`
- selects the detection with maximum confidence

Current code does not yet implement explicit box-size, aspect-ratio, too-large, too-small, or scissors-shape scoring. If those rules are needed later, they should be added to `detect_best_scissors_roboflow()`.

This is an important difference from the design notes: detection filtering is currently confidence/class-name based, not full geometry-based validation.

## 8. SAM2 Prompt Construction

SAM2 receives both:

- a positive point at the YOLO bbox center
- a shrunken version of the YOLO bbox as a box prompt

The SAM2 prompt is created in:

```text
backend/app/services/sam2_yolo/yolo_scissors_detector.py
```

and passed to SAM2 in:

```text
backend/app/services/sam2_yolo/stable_runner.py
```

The prompt insertion call is:

```python
predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=prompt_processed_frame,
    obj_id=1,
    points=points,
    labels=labels,
    box=box,
    normalize_coords=True,
)
```

The implementation uses:

- `points = np.array([initial_prompt.point], dtype=np.float32)`
- `labels = np.array([1], dtype=np.int32)`
- `box = np.array(initial_prompt.sam2_prompt_bbox or initial_prompt.bbox, dtype=np.float32)`
- `obj_id = 1`
- `normalize_coords = True`

The prompt frame is mapped from the original video frame index to the extracted processed frame index using `_processed_frame_for_original()`.

## 9. Bounding Box Shrinking

The raw YOLO bbox may include hand or background. To reduce that noise, the bbox is shrunk before being passed to SAM2.

Default shrink factor:

```text
0.65
```

Implemented function:

```text
shrink_bbox()
```

Location:

```text
backend/app/services/sam2_yolo/yolo_scissors_detector.py
```

Implementation behavior:

```python
def shrink_bbox(bbox: list[float], shrink_factor: float = DEFAULT_BBOX_SHRINK) -> list[float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(1.0, (x2 - x1) * shrink_factor)
    height = max(1.0, (y2 - y1) * shrink_factor)
    return [
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    ]
```

The point remains the original YOLO bbox center. The box passed to SAM2 is the shrunken bbox.

The output stores both boxes in `initial_prompt`:

```json
{
  "raw_yolo_bbox": [480, 300, 560, 380],
  "sam2_prompt_bbox": [494, 314, 546, 366],
  "bbox_shrink": 0.65
}
```

The debug image shows both boxes:

- raw YOLO bbox in red
- shrunken SAM2 prompt bbox in yellow
- SAM2 positive point in blue

The debug image filename is:

```text
debug_first_frame.jpg
```

## 10. Frame Extraction Strategy

SAM2 tracks over extracted frames, not directly over every original video frame.

Frame extraction is implemented by:

```text
extract_strided_frames_stable()
```

Location:

```text
backend/app/services/sam2_yolo/stable_runner.py
```

Frames are extracted into:

```text
backend/storage/outputs/sam2_yolo/runs/{run_id}/frames/
```

For expert preprocessing:

```text
backend/storage/outputs/sam2_yolo/experts/{expert_code}/frames/
```

Current extraction behavior:

- `frame_stride` must be `>= 1`
- the frames folder is deleted with `shutil.rmtree()` if it already exists
- the folder is recreated empty
- extracted frames are named `000000.jpg`, `000001.jpg`, etc.
- the YOLO prompt frame is always included
- `source_frame_count` is read from `cv2.CAP_PROP_FRAME_COUNT`
- `output_fps` is `source_fps / frame_stride`, with a minimum of `1.0`

With `FRAME_STRIDE = 5`, a 10-second video at 30 FPS has around 300 original frames and about 60 processed frames.

The default `max_processed_frames` is currently:

```text
None
```

That means normal runs are not capped unless the CLI/API explicitly passes a cap.

## 11. SAM2 Tracking Workflow

The stable tracking implementation is in:

```text
backend/app/services/sam2_yolo/stable_runner.py
```

Primary functions:

```text
run_stable_yolo_sam2_tracking()
run_stable_yolo_sam2_run()
_run_stable_sam2_tracking()
build_video_predictor_stable()
extract_strided_frames_stable()
cleanup_sam2_runtime()
```

The workflow is:

1. Create a unique run folder.
2. Run YOLO on stride-aligned frames until scissors are detected.
3. Compute YOLO bbox center.
4. Shrink YOLO bbox.
5. Extract strided frames into a fresh run-specific `frames/` folder.
6. Build a SAM2 video predictor inside the run.
7. Initialize SAM2 state with `async_loading_frames=False`.
8. Add the YOLO point and shrunken box prompt.
9. Propagate masks across extracted frames.
10. Convert masks into bbox, centroid, area, trajectory point, and region payloads.
11. Save `raw.json`.
12. Save `metrics.json`.
13. Generate `sam2_yolo_overlay.mp4`.
14. Save `summary.json`.
15. Return the summary to the CLI/API.

The currently active API path uses `runner.py`, which defaults to subprocess mode and calls:

```text
python -m app.services.sam2_yolo.stable_runner
```

This is controlled by:

```env
SAM2_RUN_MODE=subprocess
```

## 12. SAM2 Model and Device Usage

The default SAM2 model id is:

```text
facebook/sam2.1-hiera-tiny
```

This default is used in `build_video_predictor_stable()` if no local checkpoint/config is provided.

The predictor is built with either:

```python
build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=str(device))
```

or:

```python
build_sam2_video_predictor_hf(model_id, device=str(device))
```

GPU is used when:

- `use_gpu=True`
- `torch.cuda.is_available()` is true

Device selection is implemented by:

```text
choose_stable_device()
```

If CUDA is unavailable or GPU is disabled, CPU is used and `device_warning` is set.

CUDA autocast is implemented by:

```text
autocast_context_stable()
```

For CUDA, it uses:

```python
torch.autocast("cuda", dtype=torch.bfloat16)
```

SAM2 state initialization uses:

```python
predictor.init_state(
    video_path=str(extracted.frame_dir),
    async_loading_frames=False,
)
```

For GPU testing, run FastAPI without reload:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Avoid `--reload` when testing GPU inference because reload creates extra processes and can destabilize CUDA memory.

## 13. GPU Stability Notes

SAM2 can be sensitive to GPU memory and process lifetime. The current implementation keeps the system more stable by:

- building the predictor inside each run
- avoiding a global SAM2 predictor
- clearing runtime objects after each run
- supporting subprocess mode for FastAPI calls
- retrying on CPU if CUDA raises an out-of-memory error

Cleanup is implemented in:

```text
cleanup_sam2_runtime()
```

It deletes `inference_state` and `predictor`, runs `gc.collect()`, and calls CUDA cleanup functions when available:

```python
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.reset_peak_memory_stats()
```

GPU diagnostics are available in the stable runner CLI:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.services.sam2_yolo.stable_runner --video_path "path\to\video.mp4" --stride 5 --debug_gpu --save_debug
```

Subprocess mode is implemented in:

```text
backend/app/services/sam2_yolo/runner.py
```

It calls `stable_runner` using `subprocess.run()` with a 900-second timeout.

## 14. CLI Commands

Run the high-level wrapper:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.services.sam2_yolo.runner --video_path "path\to\video.mp4" --stride 5 --save_debug
```

Force CPU:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.services.sam2_yolo.runner --video_path "path\to\video.mp4" --stride 5 --no_gpu --save_debug
```

Run the stable runner directly with GPU diagnostics:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.services.sam2_yolo.stable_runner --video_path "path\to\video.mp4" --stride 5 --debug_gpu --save_debug
```

Run the stable runner with a frame cap:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.services.sam2_yolo.stable_runner --video_path "path\to\video.mp4" --stride 5 --max_processed_frames 60 --save_debug
```

Important CLI options:

- `--video_path`
- `--stride`
- `--output_root`
- `--output_dir` in `stable_runner.py`
- `--run_id`
- `--tracking_point_type bbox_center`
- `--save_debug`
- `--max_processed_frames`
- `--roboflow_confidence`
- `--sam2_checkpoint`
- `--sam2_config`
- `--sam2_model_id`
- `--no_gpu` / `--no-gpu`
- `--debug_gpu` in `stable_runner.py`

## 15. Output Folder Structure

Each learner run creates:

```text
backend/storage/outputs/sam2_yolo/runs/{run_id}/
```

Expected contents:

```text
frames/
raw.json
metrics.json
summary.json
sam2_yolo_overlay.mp4
debug_first_frame.jpg
```

`debug_first_frame.jpg` is only created when `save_debug` / `--save_debug` is enabled.

Expert preprocessing creates:

```text
backend/storage/outputs/sam2_yolo/experts/{expert_code}/
```

Expected contents:

```text
frames/
raw.json
metrics.json
summary.json
sam2_yolo_overlay.mp4
metadata.json
debug_first_frame.jpg
```

## 16. Raw JSON

`raw.json` contains frame-level reusable data. It is the most important file for future expert-vs-learner comparison.

It is built by `_build_raw_payload()` in:

```text
backend/app/services/sam2_yolo/stable_runner.py
```

Top-level fields include:

- `run_id`
- `video_path`
- `model`
- `device`
- `frame_stride`
- `original_frame_count`
- `expected_strided_frames`
- `max_processed_frames`
- `prompt_source`
- `initial_prompt`
- `tracking_point_type`
- `frames`
- `trajectory`
- `regions`

Example:

```json
{
  "run_id": "abc123",
  "video_path": "backend/storage/uploads/sam2_yolo/learner/.../original.mp4",
  "model": "yolo+sam2",
  "device": "cuda",
  "frame_stride": 5,
  "original_frame_count": 300,
  "expected_strided_frames": 60,
  "max_processed_frames": null,
  "prompt_source": "yolo_scissors_bbox_center",
  "initial_prompt": {
    "frame_index": 0,
    "timestamp_sec": 0.0,
    "point": [520, 340],
    "bbox": [494, 314, 546, 366],
    "raw_yolo_bbox": [480, 300, 560, 380],
    "sam2_prompt_bbox": [494, 314, 546, 366],
    "bbox_shrink": 0.65,
    "confidence": 0.87,
    "class_name": "scissors"
  },
  "tracking_point_type": "bbox_center",
  "frames": [
    {
      "frame_index": 0,
      "processed_frame_index": 0,
      "timestamp_sec": 0.0,
      "mask_bbox": [480, 300, 560, 380],
      "bbox_center": [520, 340],
      "mask_centroid": [518, 342],
      "blade_point": null,
      "chosen_tracking_point": [520, 340],
      "region": {
        "shape": "circle",
        "cx": 520,
        "cy": 340,
        "r": 45,
        "source": "sam2_yolo_tracking_point"
      },
      "mask_area": 2300,
      "tracking_valid": true,
      "tracking_warning": null
    }
  ],
  "trajectory": {
    "point_type": "bbox_center",
    "points": [
      {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "x": 520,
        "y": 340,
        "valid": true
      }
    ]
  },
  "regions": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "shape": "circle",
      "cx": 520,
      "cy": 340,
      "r": 45,
      "source": "sam2_yolo_tracking_point",
      "valid": true
    }
  ]
}
```

Current implementation note: `blade_point` is currently stored as `null`. The chosen tracking point is `bbox_center`.

## 17. Metrics JSON

`metrics.json` contains per-video metrics. These are not final expert-vs-learner scores. They describe one tracked video.

It is built by `_build_metrics_payload()` in:

```text
backend/app/services/sam2_yolo/stable_runner.py
```

Trajectory metrics are computed in:

```text
backend/app/services/sam2_yolo/trajectory_metrics.py
```

Region metrics are computed in:

```text
backend/app/services/sam2_yolo/region_metrics.py
```

Top-level fields:

- `run_id`
- `model`
- `frame_stride`
- `original_frame_count`
- `expected_strided_frames`
- `max_processed_frames`
- `tracking_point_type`
- `trajectory_metrics`
- `region_metrics`
- `quality_flags`

`trajectory_metrics` includes:

- `average_perpendicular_distance`
- `max_left_deviation`
- `max_right_deviation`
- `horizontal_drift_range`
- `vertical_drift_range`
- `smoothness_score`
- `trajectory_stability_score`
- `jump_count`
- `lost_tracking_count`
- `jump_threshold`
- `mean_step_distance`
- `step_distance_std`
- `fitted_line`

`region_metrics` includes:

- `average_region_area`
- `average_mask_area`
- `region_stability_score`
- `mask_area_variation`
- `region_jump_count`
- `region_jump_threshold`
- `region_step_distance_std`

`quality_flags` includes:

- `has_sudden_jumps`
- `has_lost_masks`
- `has_unstable_region`
- `usable_for_trajectory_comparison`
- `usable_for_region_comparison`

## 18. Summary JSON

`summary.json` is a frontend-friendly quick display file.

Successful summary fields include:

- `run_id`
- `status`
- `model`
- `device`
- `frame_stride`
- `original_frame_count`
- `expected_strided_frames`
- `max_processed_frames`
- `video_path`
- `raw_json_path`
- `metrics_json_path`
- `summary_json_path`
- `overlay_video_path`
- `overlay_video_url`
- `run_dir`
- `processed_frames`
- `successful_masks`
- `failure_frames`
- `prompt_source`
- `main_tracked_feature`
- `trajectory_stability_score`
- `horizontal_drift_range`
- `region_stability_score`
- `tracking_quality_note`
- `device_warning` when applicable

Example:

```json
{
  "run_id": "abc123",
  "status": "success",
  "model": "yolo+sam2",
  "device": "cuda",
  "frame_stride": 5,
  "original_frame_count": 300,
  "expected_strided_frames": 60,
  "max_processed_frames": null,
  "video_path": "backend/storage/uploads/sam2_yolo/learner/.../original.mp4",
  "raw_json_path": "backend/storage/outputs/sam2_yolo/runs/abc123/raw.json",
  "metrics_json_path": "backend/storage/outputs/sam2_yolo/runs/abc123/metrics.json",
  "summary_json_path": "backend/storage/outputs/sam2_yolo/runs/abc123/summary.json",
  "overlay_video_path": "backend/storage/outputs/sam2_yolo/runs/abc123/sam2_yolo_overlay.mp4",
  "overlay_video_url": "/storage/outputs/sam2_yolo/runs/abc123/sam2_yolo_overlay.mp4",
  "processed_frames": 60,
  "successful_masks": 60,
  "failure_frames": [],
  "prompt_source": "yolo_scissors_bbox_center",
  "main_tracked_feature": "bbox_center_trajectory",
  "trajectory_stability_score": 0.91,
  "horizontal_drift_range": 15.3,
  "region_stability_score": 0.88,
  "tracking_quality_note": "Mask may include hand, but trajectory remains usable because the tracked region follows the scissors movement."
}
```

Failed summaries include:

- `status: "failed"`
- `failure_reason`
- `stage`
- `device_attempted`
- `fallback_attempted`
- paths for expected output files

## 19. Tracking Point Selection

The module stores several tracking-related fields:

- `bbox_center`
- `mask_centroid`
- `blade_point`
- `chosen_tracking_point`

The current default tracking point is:

```text
bbox_center
```

This is defined by:

```text
TRACKING_POINT_TYPE = "bbox_center"
```

in:

```text
backend/app/services/sam2_yolo/schemas.py
```

This was chosen because the SAM2 mask may sometimes include part of the hand. For the current project goal, the exact mask shape is less important than the stability and direction of the tracked trajectory.

The main question is:

```text
Is the scissors/tool trajectory straight, or does it drift left/right?
```

A bad mask shape may still be acceptable if the tracked trajectory follows the scissors movement. A bad trajectory is not acceptable.

## 20. Trajectory Metrics

The trajectory metrics describe how the tracked tool point moves over time.

A line is fitted through valid trajectory points using SVD. Then the system calculates how far each tracked point deviates from that fitted line.

Implemented function:

```text
compute_trajectory_metrics()
```

Location:

```text
backend/app/services/sam2_yolo/trajectory_metrics.py
```

Important calculations:

- valid points are read from `raw_payload["trajectory"]["points"]`
- lost frames are counted from `tracking_valid == false`
- a fitted line is represented by `point`, `direction`, `start_point`, and `end_point`
- jump threshold is `max(median_step * 3.0, 50.0)`
- stability combines line score and smoothness score, then subtracts loss and jump penalties

These metrics help determine whether the scissors moved in a stable path or drifted during the task.

## 21. Region Metrics

The system also stores and analyzes circular regions around the tracked point.

Each frame has a region object:

```json
{
  "shape": "circle",
  "cx": 520,
  "cy": 340,
  "r": 45,
  "source": "sam2_yolo_tracking_point"
}
```

The radius is:

```text
REGION_RADIUS_PX = 45
```

Implemented function:

```text
compute_region_metrics()
```

Location:

```text
backend/app/services/sam2_yolo/region_metrics.py
```

Region metrics include:

- average region area
- average mask area
- region stability score
- mask area variation
- region jump count
- region jump threshold
- region step distance standard deviation

These regions are useful for later circle-the-mistake validation and mistake localization.

## 22. Overlay Video

The overlay video is generated by:

```text
write_sam2_yolo_overlay_video()
```

Location:

```text
backend/app/services/sam2_yolo/visualization.py
```

The output filename is:

```text
sam2_yolo_overlay.mp4
```

The visualization draws:

- original extracted frame
- semi-transparent SAM2 mask
- mask bbox
- circular region
- chosen tracking point
- trajectory line
- fitted straight reference line
- deviation line from current point to reference line

The OpenCV intermediate video is written with `mp4v`, then the code attempts to convert it to browser-friendly H.264 using `imageio_ffmpeg`:

```text
libx264
yuv420p
+faststart
```

If conversion succeeds, the final file remains:

```text
sam2_yolo_overlay.mp4
```

The browser-playable URL is:

```text
/storage/outputs/sam2_yolo/runs/{run_id}/sam2_yolo_overlay.mp4
```

Static file serving is mounted in `backend/app/main.py`:

```python
app.mount("/storage", StaticFiles(directory=settings.STORAGE_ROOT), name="storage")
```

## 23. Backend API

The YOLO+SAM2 router is:

```text
backend/app/api/sam2_yolo.py
```

It is included in the app in:

```text
backend/app/main.py
```

with:

```python
app.include_router(sam2_yolo.router)
```

### POST /api/sam2-yolo/run-learner

Endpoint:

```text
POST /api/sam2-yolo/run-learner
```

Function:

```text
run_learner_sam2_yolo()
```

Accepted JSON fields are defined by `SAM2YoloLearnerRunRequest`:

- `learner_video_id`
- `learner_video_path`
- `attempt_id`
- `compare_session_id`
- `chapter_id`
- `expert_code`
- `stride`
- `max_processed_frames`
- `tracking_point_type`
- `save_debug`

The backend requires exactly one of:

- `learner_video_path`
- `learner_video_id`
- `attempt_id`

Multipart upload is also supported. The uploaded file field must be named:

```text
file
```

Multipart form fields used by Compare Studio:

- `file`
- `expert_code`
- `stride`
- `tracking_point_type`
- optional `max_processed_frames`
- optional `save_debug`

Uploaded learner videos are saved under:

```text
backend/storage/uploads/sam2_yolo/learner/{upload_id}/original{extension}
```

The API response includes summary fields plus:

- `trajectory_metrics`
- `region_metrics`
- `quality_flags`
- `raw_json_url`
- `metrics_json_url`
- `summary_json_url`
- `overlay_video_url`
- `expert_reference_available`
- `expert_code`
- `expert_raw_json_path`
- `expert_metrics_json_path`
- `expert_raw_json_url`
- `expert_metrics_json_url`
- `expert_warning` when an expert code is supplied but preprocessed files are missing

### GET /api/sam2-yolo/runs/{run_id}

Endpoint:

```text
GET /api/sam2-yolo/runs/{run_id}
```

Function:

```text
get_sam2_yolo_run()
```

This endpoint loads an existing run's `summary.json` and returns the same frontend-friendly response structure.

## 24. Expert Preprocessing

The expert video is fixed for a task. To avoid rerunning YOLO+SAM2 on the expert every time, the expert video is processed once from the terminal.

Script:

```text
backend/app/scripts/process_expert_sam2_yolo.py
```

Main function:

```text
process_expert_sam2_yolo()
```

Command:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.scripts.process_expert_sam2_yolo --expert_video_path "path\to\expert.mp4" --expert_code "expert_cutting_01" --stride 5 --save_debug --overwrite
```

Force CPU:

```powershell
cd "AugMentor 2.0/backend"
.\.venv\Scripts\python.exe -m app.scripts.process_expert_sam2_yolo --expert_video_path "path\to\expert.mp4" --expert_code "expert_cutting_01" --stride 5 --no_gpu --overwrite
```

Expert outputs are saved under:

```text
backend/storage/outputs/sam2_yolo/experts/{expert_code}/
```

The script writes:

- `raw.json`
- `metrics.json`
- `summary.json`
- `sam2_yolo_overlay.mp4`
- `metadata.json`
- optional `debug_first_frame.jpg`

`metadata.json` includes:

- `expert_code`
- `created_at`
- `source_video_path`
- `frame_stride`
- `max_processed_frames`
- `tracking_point_type`
- `model`
- `prompt_source`
- `raw_json_path`
- `metrics_json_path`
- `summary_json_path`
- `overlay_video_path`

Overwrite behavior:

- if output exists and `--overwrite` is not passed, existing output is reused
- if `--overwrite` is passed, the expert folder is deleted and regenerated

The learner API does not rerun expert SAM2. It only checks whether expert `raw.json` and `metrics.json` exist.

## 25. Frontend Integration

The Compare Studio implementation is in:

```text
frontend/src/pages/CompareStudio.tsx
```

Relevant frontend items:

- `Sam2LearnerResult`
- `toPlayableStorageUrl()`
- `sam2LearnerRun`
- `sam2LearnerError`
- `sam2OverlayVideoError`
- `sam2LearnerVideoVersion`
- `learnerOverlay`
- `runSam2Learner()`

`runSam2Learner()` posts the uploaded learner video directly to:

```text
/api/sam2-yolo/run-learner
```

It sends multipart form data:

```text
file = userVideo
expert_code = DEFAULT_SAM2_YOLO_EXPERT_CODE
stride = 5
tracking_point_type = bbox_center
```

After success, Compare Studio switches:

```text
learnerOverlay = "sam2"
```

The video player source is selected through:

```text
learnerVideoSource
```

For YOLO+SAM2 it uses:

```text
sam2AnnotatedSource
```

which prefers:

```text
sam2OverlayBaseUrl
```

`sam2OverlayBaseUrl` is built by:

```text
toPlayableStorageUrl(sam2LearnerRun?.overlay_video_url, sam2LearnerRun?.overlay_video_path)
```

This is important because the browser must receive a `/storage/...` URL, not a Windows filesystem path.

The visible UI section is labeled:

```text
Scissors Tracking (YOLO+SAM2)
```

The run button displays:

```text
Run YOLO+SAM2
```

The overlay switch label is:

```text
YOLO+SAM2 Scissors
```

Current implementation note: the `learnerOverlay` state currently supports:

```typescript
'none' | 'mediapipe' | 'sam2'
```

So the visible video switching path is Original, MediaPipe, and YOLO+SAM2. Any Optical Flow UI is separate from this specific `learnerOverlay` state.

## 26. Failure Handling

YOLO failures are caught in `run_stable_yolo_sam2_tracking()` and saved as failed summaries with:

- `status: "failed"`
- `stage: "yolo_prompt"`
- `failure_reason`
- `device_attempted`
- `fallback_attempted`

Current YOLO failure messages may include:

```text
YOLO did not detect scissors in the scanned prompt frames
YOLO did not detect scissors on the first usable frame
Missing ROBOFLOW_API_KEY in .env
Missing ROBOFLOW_MODEL_ID in .env
```

SAM2 runtime failures are saved with stages such as:

- `sam2_initialization`
- `sam2_propagation`
- `visualization`
- `sam2_subprocess`

If CUDA raises an out-of-memory error, the stable runner retries once on CPU and records:

```text
CUDA out of memory during SAM2, retried on CPU for SAM2
```

If SAM2 returns empty masks for some frames, those frames are represented with:

```json
{
  "tracking_valid": false,
  "tracking_warning": "mask_missing"
}
```

Failed frame indexes are listed in `summary.json` under:

```text
failure_frames
```

## 27. Known Differences Between Design Text and Current Code

This section records differences between the original design description and the current implementation.

1. Detection filtering is simpler than the design text.

   Current code filters by confidence, prefers class names containing `"scissor"`, and selects the highest confidence candidate. It does not yet reject boxes by explicit size, aspect ratio, or scissors-like geometry.

2. `blade_point` is not implemented yet.

   The field exists in each frame payload but is currently `null`.

3. `bbox_center` is the only supported tracking point.

   `tracking_point_type` is accepted by the API/CLI, but only `"bbox_center"` is supported.

4. The frontend overlay state is not a four-option enum.

   In `CompareStudio.tsx`, `learnerOverlay` currently supports `'none'`, `'mediapipe'`, and `'sam2'`. YOLO+SAM2 appears as the SAM2 overlay option in that state.

5. `SAM2_MAX_PROCESSED_FRAMES` defaults to unlimited.

   The current default is `None`, and CLI `--max_processed_frames 0` means no cap.

6. FastAPI uses subprocess mode by default.

   `SAM2_RUN_MODE=subprocess` causes `runner.py` to call `stable_runner.py` in a child process. This is intentional for stability.

7. Browser video compatibility is handled by re-encoding.

   `visualization.py` writes a temporary OpenCV MP4 and then attempts H.264 conversion using `imageio_ffmpeg`.

## 28. Relation to the Evaluation Engine

YOLO+SAM2 is an extraction module, not the final scoring module.

It produces:

```text
raw.json      = frame-level tracking data
metrics.json  = per-video calculations
summary.json  = frontend summary
overlay.mp4   = visual proof
```

The evaluation engine later compares expert and learner outputs using saved trajectory and region data.

This supports the larger AugMentor 2.0 mistake detection pipeline:

```text
feature extraction
-> temporal alignment
-> expert-vs-learner comparison
-> mistake event generation
-> visualization
-> game validation
```

## 29. Final Summary

The YOLO+SAM2 module tracks the scissors/tool trajectory in learner and expert videos.

The most important design decisions are:

- YOLO is used as the scissors detector.
- SAM2 is used as the video tracker.
- YOLO provides both a point prompt and a box prompt.
- The YOLO bbox is shrunk before being passed to SAM2.
- Frames are extracted using stride.
- Each run uses a unique output folder.
- The predictor is built inside the run.
- FastAPI uses subprocess mode by default for stability.
- The system saves raw data, metrics, summary, and overlay video.
- The frontend displays the overlay video in Compare Studio.
- The outputs are reusable for expert-vs-learner comparison.

This design makes the tracking module useful not only for visualization, but also for future evaluation and mistake localization stages of AugMentor 2.0.
