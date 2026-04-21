# AugMentor 2.0 MediaPipe Documentation

This document explains how MediaPipe is currently used in AugMentor 2.0 for:

- **Expert references** (offline, process once, reuse later)
- **Learner videos** (runtime, per-upload/per-evaluation processing)

It also records the architecture rules and files we added/updated for the expert reusable flow.

---

## 1) Architecture Rule (Expert vs Learner)

### Expert MediaPipe flow

- Expert preprocessing is a **one-time offline operation**.
- It is triggered manually by:
  - CLI script (primary)
  - Admin/debug endpoint (optional)
- Outputs are saved in a **stable reusable folder**.
- Database stores stable paths + summary metadata on the existing expert `videos` row.

### Learner MediaPipe flow

- Learner processing is **runtime** and can happen per learner upload/evaluation.
- Learner uploads and evaluation continue to use the existing learner pipeline.
- Learner flow must **not trigger expert preprocessing**.

---

## 2) Storage Layout

### Expert stable storage (new reusable reference location)

Each expert gets a dedicated folder:

`backend/storage/expert/mediapipe/{expert_code}/`

`expert_code` is the expert `Video.id` (UUID string).

Expected files:

- `source.mp4`
- `detections.json`
- `features.json`
- `metadata.json`
- `annotated.mp4` (best effort; present when visualization succeeds)

### MediaPipe runtime storage (existing transient location)

- `backend/storage/mediapipe/runs/{run_id}/`
- `backend/storage/mediapipe/sources/{run_id}/`

These run folders are temporary/run-scoped artifacts. For expert references, final reusable files are copied to the stable expert folder above.

---

## 3) Expert Flow We Implemented

Main service:

- `backend/app/services/expert_mediapipe_service.py`
- entry function: `register_expert_mediapipe_reference(...)`

### Service behavior

1. Resolve existing expert row (by `expert_video_id` or `chapter_id`).
2. Validate role is `video_role == "expert"`.
3. Build stable folder under `storage/expert/mediapipe/{expert_code}`.
4. Save/copy source video as `source.mp4`.
5. Run existing MediaPipe pipeline (`run_pipeline`) on `source.mp4`.
6. Copy outputs from run folder into stable expert folder:
   - `detections.json`
   - `features.json`
   - `metadata.json`
   - `annotated.mp4` (if available)
7. Parse `metadata.json`.
8. Update same expert `videos` row with paths + summary metadata.
9. Return a response object for script/API callers.

### Reprocessing policy

- If expert already completed and `overwrite=false`:
  - return existing saved reference, no rerun.
- If `overwrite=true`:
  - rerun and replace saved files.
- On failure:
  - set status to `failed`
  - cleanup partial artifacts
  - keep source file optionally for retry
  - avoid leaving inconsistent DB/file state

---

## 4) Expert Triggers

### A) CLI (Primary, recommended)

Script:

- `backend/app/scripts/process_expert_mediapipe.py`

Examples:

```bash
python -m app.scripts.process_expert_mediapipe --expert_code <expert_video_uuid>
python -m app.scripts.process_expert_mediapipe --chapter_id <chapter_uuid> --overwrite
python -m app.scripts.process_expert_mediapipe --expert_code <expert_video_uuid> --video_path C:/path/to/video.mp4
```

### B) Admin/debug API (optional)

Route file:

- `backend/app/api/routes/expert_mediapipe.py`

Endpoint:

- `POST /api/admin/experts/{expert_video_id}/mediapipe/process`

This endpoint is explicitly for admin/debug use and is not part of frontend compare runtime flow.

---

## 5) Existing MediaPipe Pipeline Reused

We intentionally reused existing MediaPipe modules instead of rewriting extraction:

- `backend/app/services/mediapipe/run_service.py` (`run_pipeline`)
- `backend/app/services/mediapipe/extraction_service.py`
- `backend/app/services/mediapipe/feature_service.py`
- `backend/app/services/mediapipe/visualization_service.py`
- `backend/app/schemas/mediapipe/mediapipe_schema.py` (`MediaPipeRunMeta`, etc.)

Expert service wraps this existing pipeline and redirects final artifacts into stable expert storage.

---

## 6) Database Integration (No Parallel Expert Table)

We integrated on the existing `videos` table (expert rows), without creating a second expert reference architecture.

### Added columns for expert MediaPipe references

- `mediapipe_source_path`
- `mediapipe_detections_path`
- `mediapipe_features_path`
- `mediapipe_metadata_path`
- `mediapipe_annotated_path`
- `mediapipe_status` (`pending`, `processing`, `completed`, `failed`)
- `mediapipe_processed_at`
- `mediapipe_pipeline_version`
- `mediapipe_fps`
- `mediapipe_frame_count`
- `mediapipe_detection_rate`
- `mediapipe_selected_hand_policy`

### Alembic migrations

- `backend/alembic/versions/20260421_0001_add_mediapipe_fields_to_videos.py`
- `backend/alembic/versions/20260421_0002_add_mediapipe_annotated_path.py`

Alembic environment wiring was fixed to use app settings:

- `backend/alembic/env.py` now uses `settings.DATABASE_URL` and `Base.metadata`.

---

## 7) Learner Side (Current State)

Learner upload/evaluation remains separate and runtime-oriented:

- Upload learner video: `POST /api/uploads/practice-video`
  - route: `backend/app/api/routes/uploads.py`
- Evaluation start: `POST /api/evaluations/start`
  - route: `backend/app/api/routes/evaluations.py`
- Generic MediaPipe routes:
  - `POST /api/mediapipe/process`
  - `POST /api/mediapipe/process-upload`
  - `GET /api/mediapipe/result/{run_id}`
  - route: `backend/app/api/mediapipe/routes.py`

Important: expert preprocessing is designed to be offline/manual and separate from learner runtime operations.

---

## 8) How to View Expert Annotated Output

After successful expert preprocessing, annotated output is stored at:

`backend/storage/expert/mediapipe/{expert_code}/annotated.mp4`

Because backend mounts storage statically, the same file is accessible from:

`/storage/expert/mediapipe/{expert_code}/annotated.mp4`

This makes it easy to inspect the generated expert visualization directly.

---

## 9) Files Added/Updated for This Work

### Added

- `backend/app/services/expert_mediapipe_service.py`
- `backend/app/schemas/expert_mediapipe_schema.py`
- `backend/app/api/routes/expert_mediapipe.py`
- `backend/app/scripts/process_expert_mediapipe.py`
- `backend/app/tests/test_expert_mediapipe_service.py`
- `backend/alembic/versions/20260421_0001_add_mediapipe_fields_to_videos.py`
- `backend/alembic/versions/20260421_0002_add_mediapipe_annotated_path.py`
- `docs/Mediapipe.md` (this file)

### Updated

- `backend/app/models/video.py`
- `backend/app/main.py`
- `backend/alembic/env.py`

---

## 10) Operational Notes

- Use CLI as the default way to preprocess experts.
- Use `--overwrite` when you want to regenerate references (including `annotated.mp4`).
- Keep expert references stable in `storage/expert/mediapipe/`, not in `storage/mediapipe/runs/`.
- Runtime compare/evaluation should consume saved expert references and not retrigger expert preprocessing.

