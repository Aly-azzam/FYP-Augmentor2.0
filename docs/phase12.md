# AugMentor 2.0 Backend Documentation
## Phase 1 and Phase 2 Computer Vision Foundation

---

## 1. Project Overview

AugMentor 2.0 is a deterministic, modular learning system that helps a learner compare their performance against an expert demonstration.

At the current stage, the backend can already do the following:

1. store and serve expert videos
2. accept and persist learner uploads
3. standardize both videos into a comparable frame sequence
4. detect hand landmarks over time
5. clean and stabilize the landmark sequence
6. convert the cleaned landmarks into comparison-ready motion features

The project is designed so that every stage of the pipeline produces a structured output contract that can be reused by the next stage.

This is important because the later comparison phase will depend on:

- stable timing
- stable handedness
- stable coordinate conventions
- fixed-size per-frame feature vectors

---

## 2. Core Design Philosophy

AugMentor 2.0 is intentionally built as an interpretable system rather than a black-box end-to-end trained model.

The current backend foundation is:

- deterministic
- explainable
- modular
- testable
- suitable for debugging step by step

The main approach is:

1. extract useful hand structure with computer vision
2. clean and stabilize that structure over time
3. transform it into explicit motion features
4. later compare those features mathematically

This means the system relies on:

- video preprocessing
- MediaPipe-based hand landmark extraction
- signal cleaning
- geometric feature engineering
- temporal kinematics

It does not currently depend on training a custom model for the core comparison pipeline.

---

## 3. Current End-to-End Backend Pipeline

The completed foundation at this point is:

```text
Expert Video + Learner Video
-> Phase 1: Upload, storage, linking, delivery
-> Step 2.1: Video preprocessing / standardization
-> Step 2.2: Hand detection
-> Step 2.3: Landmark cleaning and smoothing
-> Step 2.4: Motion representation
-> Phase 3: Comparison and evaluation (next)
```

The current output of Phase 2 is no longer "raw video". It is a structured, frame-aligned, fixed-dimension motion representation that is ready for later alignment and comparison.

---

## 4. Scope of What Was Completed

This document covers the work completed in:

- Phase 1: upload and storage foundation
- Phase 2 Step 2.1: video preprocessing
- Phase 2 Step 2.2: hand detection
- Phase 2 Step 2.3: landmark cleaning and stabilization
- Phase 2 Step 2.4: motion representation

This document does not cover:

- DTW
- scoring
- evaluation logic
- VLM explanation
- frontend UI design details
- long-term persistence of landmark or motion outputs

---

## 5. Phase 1 - Upload and Storage Foundation

### 5.1 Goal

The goal of Phase 1 was to make the system usable with real videos before any computer vision work:

- expert video must exist in the system
- expert video must be browser-accessible
- learner video upload must be real, not mocked
- upload must create a database record
- upload must persist the file to storage
- backend must return clean responses and clean errors

This phase established the entire media foundation required by the rest of the pipeline.

---

### 5.2 Backend Application Structure

The backend is a FastAPI application with:

- router-based API modules
- SQLAlchemy async database access
- local file storage for the MVP
- static media serving

Relevant backend routes include:

- `GET /api/chapters/default/expert-video`
- `GET /api/chapters/{chapter_id}`
- `GET /api/chapters/{chapter_id}/expert-video`
- `POST /api/uploads/practice-video`
- `GET /storage/...`
- `GET /health`

The backend mounts local storage as static content so that the browser receives a proper HTTP media URL rather than a local filesystem path.

---

### 5.3 Expert Video Storage and Delivery

Expert videos are stored under:

```text
backend/storage/expert/
```

Examples:

- `backend/storage/expert/pottery.mp4`
- `backend/storage/expert/chapter 1/pottery.mp4`

The backend normalizes file storage keys into browser-safe URLs. For example:

```text
/storage/expert/pottery.mp4
```

This is critical because a browser cannot play a raw local path such as:

```text
C:\Users\aliaz\Desktop\AugMentor\video.mp4
```

Instead, it must receive a route the browser can request over HTTP.

The backend now:

1. discovers a usable expert video file
2. normalizes its storage key
3. returns a URL such as `/storage/expert/pottery.mp4`

---

### 5.4 Database Linking for Expert Context

Phase 1 also established expert-video linkage through database context.

Relevant entities:

- `Course`
- `Chapter`
- `ExpertVideo`
- `LearnerAttempt`

The seed script creates a minimal context for testing:

- demo course
- demo chapter
- expert video row pointing to the stored expert file

This makes it possible to attach learner uploads to a real chapter and know which expert reference they belong to.

---

### 5.5 Learner Video Upload Flow

Learner uploads are handled through:

```text
POST /api/uploads/practice-video
```

Expected form fields:

- `file`
- `chapter_id`
- `user_id` optional

The backend validates:

- a file was provided
- the file has a name
- extension is supported
- content type is acceptable
- the referenced chapter exists
- optional user exists if provided

Supported formats for MVP:

- mp4
- mov
- avi
- webm

Storage layout:

```text
storage/uploads/learner_videos/{attempt_id}/original.ext
```

This gives each learner attempt its own isolated directory and preserves the original uploaded media as the source artifact.

---

### 5.6 LearnerAttempt Persistence

Every valid learner upload creates a `LearnerAttempt` record.

Important stored fields include:

- attempt id
- chapter id
- optional user id
- upload status
- original filename
- content type
- file size
- stored relative path
- timestamps

This record is the future anchor point for:

- preprocessing outputs
- landmarks
- motion representation
- evaluation results

---

### 5.7 Phase 1 Response and Error Handling

Successful upload returns a structured response like:

```json
{
  "attempt_id": "uuid",
  "chapter_id": "uuid",
  "expert_video_id": "uuid or null",
  "upload_status": "uploaded",
  "original_filename": "practice.mp4",
  "stored_path": "uploads/learner_videos/<attempt_id>/original.mp4",
  "video_url": "/storage/uploads/learner_videos/<attempt_id>/original.mp4",
  "message": "Video uploaded successfully. Learner attempt is ready for evaluation."
}
```

Handled error cases include:

- missing file
- empty file
- unsupported format
- invalid chapter
- invalid user
- file save failure
- DB persistence failure
- missing expert video

No raw Python traceback is intentionally returned as the API response.

---

### 5.8 Frontend Integration Foundation

Although this document is backend-focused, one important Phase 1 outcome was ensuring the frontend could actually use the backend media:

- backend moved to `8001` to avoid conflict with AugMentor 1.0 on `8000`
- Vite proxy routes `/api` and `/storage` to AugMentor 2.0 backend
- the expert video shown in the frontend now comes from the backend storage URL
- the pottery expert clip can be selected and opened in Compare Studio

This matters because Phase 1 was not just file storage. It was media availability from the actual application flow.

---

### 5.9 Phase 1 Verification Assets

Important verification assets created in this phase:

- `backend/STEP1_VERIFICATION.md`
- `backend/tests/test_step1_upload_routes.py`
- `backend/app/scripts/seed_upload_demo_context.py`

These give a reproducible way to verify:

- expert media access
- upload success
- upload failure handling
- DB row creation
- file presence on disk

---

## 6. Phase 2 - Computer Vision Foundation

Phase 2 transforms raw uploaded videos into structured motion data.

The key design decision is that each phase produces a clean intermediate output rather than collapsing everything into one large opaque function.

That produced this sequence:

1. standardized frames
2. raw hand landmarks
3. cleaned hand landmarks
4. comparison-ready motion features

---

## 7. Step 2.1 - Video Preprocessing and Standardization

### 7.1 Goal

The purpose of preprocessing was to ensure that both expert and learner videos are transformed into a clean, shared temporal and spatial format before any landmark extraction begins.

Without this step, comparison later would be fundamentally inconsistent.

---

### 7.2 Inputs and Outputs

Inputs:

- expert video path
- learner video path

Main outputs:

- `VideoFrame`
- `VideoFramesOutput`
- `PreparedVideosOutput`

Each normalized frame contains:

- `frame_index`
- `source_frame_index`
- `timestamp_sec`
- `frame_rgb`

---

### 7.3 Resolution Normalization

Every kept frame is resized to:

```text
640 x 480
```

This ensures:

- same image dimensions for both videos
- consistent detector input
- consistent coordinate interpretation

---

### 7.4 FPS Normalization

All videos are normalized to:

```text
30 FPS
```

The final implementation performs true target-time resampling.

For each target frame index:

```text
target_time = frame_index / target_fps
source_frame_index = round(target_time * source_fps)
```

This means:

- higher FPS sources are downsampled
- lower FPS sources are upsampled by duplicate sampling
- normalized timestamps always follow the 30 FPS timeline

Result:

- `normalized_fps` is always 30
- timestamps follow:

```text
0.000
0.033
0.067
0.100
...
```

---

### 7.5 Step 2.1 Validation

Validation was added through:

- `backend/tests/test_video_preprocessing_service.py`
- `backend/app/scripts/test_video_preprocessing.py`

The demo script prints:

- source fps
- normalized fps
- total frames
- first timestamps
- sample frame shape

This verified that:

- frame extraction worked
- resolution normalization worked
- timestamps were correct
- normalized fps stayed fixed at 30

---

## 8. Step 2.2 - Hand Detection

### 8.1 Goal

The goal of Step 2.2 was to move from standardized frames to actual structured hand motion observations.

This was the first real computer vision stage of the backend.

---

### 8.2 What Was Reused from AugMentor 1.0

Useful ideas reused from AugMentor 1.0:

- MediaPipe-based hand detection
- one detector instance reused across frames
- handedness extraction for left vs right hands
- per-frame structured storage of hand outputs

What was not copied from 1.0:

- frame skipping
- pixel-coordinate-only storage
- inline metrics inside detection
- monolithic analyzer design

So 2.0 keeps the useful detection ideas while using cleaner dataclasses and cleaner pipeline boundaries.

---

### 8.3 Runtime Adaptation: MediaPipe Tasks

An important implementation detail from today:

The installed environment did not expose the old `mp.solutions.hands` runtime directly.

Instead, the working runtime was:

- `mediapipe.tasks.vision.HandLandmarker`

To support that, the backend now uses:

- official `hand_landmarker.task` model file
- MediaPipe Tasks API
- a model path configured in backend settings

This improved runtime compatibility while preserving the same conceptual output.

---

### 8.4 Hand Detection Output Contract

Step 2.2 outputs:

- `VideoLandmarksOutput`
- `FrameLandmarks`
- `HandLandmarks`
- `LandmarkPoint`

For each frame:

- `frame_index`
- `timestamp_sec`
- `left_hand`
- `right_hand`

For each detected hand:

- 21 landmarks
- each landmark has:
  - `x`
  - `y`
  - `z`

Coordinate convention:

- normalized MediaPipe coordinates

So the backend now has a consistent normalized hand representation instead of raw image pixels.

---

### 8.5 Missing Detections

Not all frames contain visible hands.

Step 2.2 handles this safely:

- no crash
- no fake landmarks
- missing hand is stored as `None`

This was necessary before any real temporal processing.

---

### 8.6 Step 2.2 Validation

Validation assets:

- `backend/tests/test_hand_detection_service.py`
- `backend/app/scripts/test_hand_detection.py`

Checks included:

- detector runs successfully
- at least some frames contain hands
- detected hands have 21 landmarks
- frame count matches preprocessing
- timestamps match preprocessing
- blank frames produce `None` without crashing

This established a stable raw hand landmark stream over time.

---

## 9. Step 2.3 - Landmark Cleaning and Stabilization

### 9.1 Goal

Raw MediaPipe landmarks are useful, but too noisy to compare directly.

Problems addressed:

- frame-to-frame jitter
- short missing detections
- occasional left/right instability
- unstable boundary smoothing

This step converts raw detections into a cleaner signal suitable for motion features.

---

### 9.2 Cleaning Pipeline

The cleaning stage now performs:

1. handedness stabilization
2. short-gap interpolation
3. edge-aware smoothing

All while preserving:

- same frame count
- same timestamps
- same frame indices
- same 21-landmark structure

---

### 9.3 Handedness Stabilization

The backend uses a lightweight temporal continuity rule:

- compare current left/right wrist continuity against the previous left/right wrists
- if swapping current hands reduces continuity cost, swap them

This is intentionally simple and deterministic.

It is not yet full tracking, but it already reduces label flicker significantly.

---

### 9.4 Short-Gap Interpolation

Short gaps are filled only when safe.

Current behavior:

- gaps up to 2 frames can be interpolated
- interpolation is linear for all 21 landmarks
- long gaps remain missing

This avoids inventing long fake motion while still repairing brief detector dropouts.

---

### 9.5 Edge-Aware Smoothing

This was refined later because the initial smoothing distorted boundary frames too much.

Old issue:

- the first frame was being pulled too strongly toward later frames

Fix:

- compute the standard moving average
- blend it with the raw signal near the edges
- preserve the exact raw values at the first and last frame
- gradually transition to full smoothing toward the middle

This preserves:

- honest motion at the boundaries
- effective smoothing in the middle

This refinement is important because later velocity and acceleration are sensitive to edge artifacts.

---

### 9.6 Cleaning Summary Output

The cleaning layer can report useful summary stats, including:

- raw detected frames
- cleaned detected frames
- left gaps filled
- right gaps filled
- still missing left frames
- still missing right frames
- handedness swaps applied

This makes it easier to audit the quality of the cleaned sequence.

---

### 9.7 Step 2.3 Validation

Validation assets:

- `backend/tests/test_landmark_cleaning_service.py`
- `backend/app/scripts/test_landmark_cleaning.py`

Validation confirms:

- frame count preserved
- timestamps preserved
- jitter reduced
- short gaps filled
- long gaps preserved
- handedness stabilization works
- boundaries preserved better after the edge-aware smoothing fix

---

## 10. Step 2.4 - Motion Representation

### 10.1 Goal

The goal of Step 2.4 was to move from cleaned landmarks to a structured motion representation suitable for later comparison.

This is the point where the system stops working with "landmarks only" and starts building explicit comparison-ready features.

These features are intended to support future:

- DTW alignment
- similarity analysis
- metric computation
- scoring
- explainable feedback

---

### 10.2 Design Principles

The motion representation had to be:

- frame-aligned
- interpretable
- normalized where possible
- stable across videos
- flattenable into a fixed-length numeric vector

This means the output needed both:

- structured named features
- a deterministic flat vector for machine-style comparison later

---

### 10.3 Motion Feature Set

For each hand, per frame, the current MVP representation includes:

#### Presence

- whether the hand exists in this frame

#### Core positions

- wrist position
- palm center

#### Relative position features

- wrist relative to palm center
- fingertip positions
- fingertip positions relative to palm center

Tracked fingertips:

- thumb tip
- index tip
- middle tip
- ring tip
- pinky tip

#### Relative distances

- thumb tip to index tip
- thumb tip to middle tip
- index tip to wrist
- middle tip to wrist
- thumb tip to palm center
- index tip to palm center
- middle tip to palm center
- ring tip to palm center
- pinky tip to palm center
- index tip to pinky tip

#### Joint and shape angles

- wrist-index_mcp-index_tip
- wrist-middle_mcp-middle_tip
- wrist-ring_mcp-ring_tip
- wrist-index_mcp-pinky_mcp

#### Hand shape proxies

- hand openness
- pinch distance
- finger spread

#### Temporal features

- wrist velocity
- palm velocity
- wrist acceleration
- palm acceleration

---

### 10.4 Palm Center and Hand Scale

Palm center is computed from:

- wrist
- index_mcp
- middle_mcp
- ring_mcp
- pinky_mcp

Hand scale is computed from:

- distance between wrist and middle_mcp

That scale is then used to normalize distances and relative coordinates.

This is important for:

- cross-user robustness
- different hand sizes
- different camera framing

---

### 10.5 Temporal Feature Computation

Temporal features are computed using timestamps, not hardcoded assumptions.

For each frame:

```text
dt = current_timestamp - previous_timestamp
```

Then:

```text
velocity = (current_position - previous_position) / dt
acceleration = (current_velocity - previous_velocity) / dt
```

This is currently computed for:

- wrist
- palm center

The first valid frame has zero velocity and zero acceleration because no previous frame exists for differentiation.

---

### 10.6 Missing-Hand Strategy

Missing hands are represented deterministically.

If a hand is not present in a frame:

- `present = false`
- structured vectors are zero-filled
- flattened vector is zero-filled for that hand

This preserves:

- same vector dimension in every frame
- same number of frames
- same timestamps
- same downstream comparison shape

This is essential for DTW and later numerical comparison.

---

### 10.7 Flattened Feature Vector

Each hand produces a deterministic flattened numeric vector.

Current dimensions:

- per hand: `54`
- per frame: `108`

Frame vector layout:

```text
left_hand_vector + right_hand_vector
```

The ordering is fixed and deterministic, so later comparison algorithms can rely on a stable meaning for each dimension.

---

### 10.8 Step 2.4 Output Contract

Important outputs:

- `HandMotionFeatures`
- `MotionFrameFeatures`
- `MotionRepresentationOutput`

Per frame:

- `frame_index`
- `timestamp_sec`
- `left_hand_features`
- `right_hand_features`
- `flattened_feature_vector`

Per video:

- `video_id`
- `video_path`
- `fps`
- `total_frames`
- `sequence_length`
- `coordinate_system`
- `hand_feature_vector_dim`
- `frame_feature_vector_dim`
- `frames`

This now gives the project a real comparison representation instead of only raw landmarks.

---

### 10.9 Step 2.4 Validation

Validation assets:

- `backend/tests/test_motion_representation.py`
- `backend/app/scripts/test_motion_representation.py`

Validation confirms:

- timestamps match Step 2.3 input
- frame count is preserved
- structured features exist
- feature values are finite
- hand vector length is constant
- frame vector length is constant
- missing-hand handling keeps vector dimensions stable

This verifies the representation is ready for later alignment logic.

---

## 11. Practical Commands Used for Validation

The backend now includes reproducible smoke tests for the completed steps.

### Phase 1

```powershell
python -m app.scripts.seed_upload_demo_context
pytest tests/test_step1_upload_routes.py
```

### Step 2.1

```powershell
pytest tests/test_video_preprocessing_service.py
python -m app.scripts.test_video_preprocessing
```

### Step 2.2

```powershell
pytest tests/test_hand_detection_service.py
python -m app.scripts.test_hand_detection
```

### Step 2.3

```powershell
pytest tests/test_landmark_cleaning_service.py
python -m app.scripts.test_landmark_cleaning
```

### Step 2.4

```powershell
pytest tests/test_motion_representation.py
python -m app.scripts.test_motion_representation
```

---

## 12. Important Technical Decisions Taken Today

Several important engineering decisions were made during this work:

### 12.1 Use backend port 8001 for AugMentor 2.0

Reason:

- avoid collision with AugMentor 1.0 backend running on 8000

### 12.2 Serve media through `/storage/...`

Reason:

- browsers need HTTP media URLs, not local filesystem paths

### 12.3 Keep coordinates normalized

Reason:

- more robust than raw pixels for comparison across videos

### 12.4 Use MediaPipe Tasks HandLandmarker runtime

Reason:

- matches the installed environment
- official supported path in the current package layout

### 12.5 Keep outputs typed and staged

Reason:

- each phase can be validated independently
- easier debugging
- easier replacement of one stage without rewriting the others

### 12.6 Keep missing-hand handling deterministic

Reason:

- later vector-based comparison requires fixed dimensionality

---

## 13. What Is Now Complete

At the end of today’s work, the backend can transform real videos through the following chain:

```text
Video
-> stored and linked in backend
-> standardized frame sequence
-> raw hand landmarks
-> cleaned hand landmarks
-> stable motion feature representation
```

This means the backend now has a true computer vision foundation rather than only upload and placeholder logic.

The system now supports:

- real media ingestion
- real hand landmark detection
- signal cleaning
- normalized geometric motion representation

This is the complete foundation required before moving to comparison logic.

---

## 14. What Is Not Yet Done

The following phases are intentionally not part of this document yet:

- DTW alignment
- expert/learner temporal pairing
- final metric scoring
- evaluation result generation
- VLM explanation layer
- persistence of landmark and motion outputs
- advanced multi-hand interaction modeling
- frontend visualization of landmarks or motion vectors

These belong to later phases.

---

## 15. Final Status

### Phase 1

Complete.

### Phase 2

Completed through Step 2.4.

That means the current backend already has:

- upload and storage foundation
- standardized frame generation
- hand landmark detection
- landmark cleaning and stabilization
- comparison-ready motion representation

---

## 16. Final Statement

AugMentor 2.0 now has a solid backend foundation for the skill-comparison pipeline.

The system can already take real expert and learner videos and transform them into:

- synchronized, normalized frame sequences
- structured hand landmark timelines
- cleaned and stabilized landmark signals
- fixed-dimension interpretable motion features

This is the bridge between raw video perception and future comparison logic.

The next backend phase should focus on:

- temporal alignment
- similarity comparison
- metric aggregation
- explainable evaluation outputs

At this point, the project is no longer at the "mock CV" stage. It has a real, working, testable computer vision foundation.
