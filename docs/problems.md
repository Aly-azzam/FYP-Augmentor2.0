# AugMentor 2.0 - Frontend/Backend Integration + Scoring Problems Report

This document records, in detail, what happened during the recent integration/debugging cycle, what failed, how each failure was diagnosed, and exactly what was changed.

Use this as a handoff report for future work and as a source for formal reporting.

---

## 1) Scope of this report

This report covers the sequence from:

1. Connecting `CompareStudio` frontend to real backend APIs
2. Resolving request/response and DB runtime errors
3. Fixing processing UX (stuck at step 5)
4. Investigating bad scoring behavior (including the observed symptom: high score for out-of-context videos)
5. Replacing synthetic evaluation inputs with real perception/motion pipeline
6. Adding scoring sanity benchmark tooling
7. Tightening scoring sensitivity
8. Adding context-gate hard rejection for out-of-context videos
9. Updating frontend behavior for out-of-context responses

---

## 2) Product context and target pipeline

Target flow in AugMentor 2.0:

`Frontend -> Backend API -> Upload/Job creation -> Perception Engine -> Motion Representation -> Temporal Alignment (DTW) -> Metrics -> Score -> Explanation -> Frontend display`

Important project constraints that guided decisions:

- Keep AugMentor 2.0 as main architecture
- Keep existing frontend
- Keep modular backend service boundaries
- Avoid broad refactors while stabilizing behavior
- Prefer deterministic, explainable logic (no black-box model dependency)

---

## 3) Main observed problems (symptoms)

### Symptom A - Integration/runtime breakages

- Backend error: `UnicodeDecodeError: 'utf-8' codec can't decode byte ...`
- Backend error: async/sync SQLAlchemy mismatch (`await` on sync result)
- Frontend flow gets to "Processing... Step 5 of 5" then does not finalize

### Symptom B - Scoring did not reflect reality

- Similar videos scoring too high/too compressed
- Out-of-context video still receiving non-zero/high score (example observed in this cycle)
- Early reports showed behavior like high 70s/80s in cases that should be near-reject

### Symptom C - UX friction in Compare Studio

- Could not reliably switch learner video after first upload
- Out-of-context response still showed explanation-like details in learner UI

---

## 4) Root causes and fixes (chronological)

---

### 4.1 Frontend -> Backend request format mismatch

#### Problem
`POST /api/evaluations/start` initially expected JSON body, but frontend sends `multipart/form-data` (video file upload).

#### Failure mode
FastAPI attempted JSON parse on binary payload and raised UTF-8 decode errors.

#### Fix
`backend/app/api/routes/evaluations.py`

- Route updated to accept `Request`
- Added content-type branching:
  - parse `await request.form()` for multipart
  - parse JSON only when content type is JSON
- Added robust extraction of `file`, `course_id`, `clip_id`, `chapter_id`, `attempt_id`

#### Outcome
File upload request no longer crashes during decode/parsing.

---

### 4.2 Async/Sync database mismatch

#### Problem
Some routes/services still used async DB patterns (`AsyncSession`, `await db.execute`) after DB layer had moved to sync SQLite session.

#### Failure mode
Runtime errors (`TypeError` and follow-up failures).

#### Fixes
- `backend/app/api/routes/chapters.py`
- `backend/app/services/media_service.py`

Changes:
- Converted session type to sync `Session`
- Removed `await` from sync DB calls

#### Outcome
Routes/services became compatible with current SQLite sync DB setup.

---

### 4.3 Compare Studio stuck at "Step 5 of 5"

#### Problem
Frontend did not fully exit evaluating state after successful result fetch.

#### Root cause
Store state (`isEvaluating`) wasn’t reset at the right point in success path.

#### Fix
`frontend/src/pages/CompareStudio.tsx`

- Added `resetEvaluation()` after successful result retrieval
- Confirmed result object is set in local state and displayed

#### Outcome
UI transitions from "processing" to final results state.

---

### 4.4 Core scoring bug: synthetic motion used in live endpoint

#### Problem
Live `/evaluations/start` path was using synthetic hash-derived motion payloads instead of real video-derived motion.

#### Why this is severe
Hash-based synthetic series do not represent actual hand movement. DTW/metrics had little semantic relation to video content, compressing score range and producing unrealistic similarity.

#### Fix
`backend/app/api/routes/evaluations.py`

- Removed hash/synthetic motion generator path from live endpoint
- Introduced real processing path:
  - save uploaded learner file to temp storage
  - run `process_video_to_motion_representation()` on expert and learner
  - validate minimum valid landmark frames
  - pass real motion outputs into evaluation engine

#### Added diagnostics
- Input IDs and resolved paths
- Valid landmark counts
- Sequence lengths
- Final metrics + score logs

#### Outcome
Scoring now based on actual MediaPipe + motion representation output.

---

### 4.5 Scoring sanity tooling added

#### Problem
No fast way to compare scoring behavior across known categories.

#### Fix
Created benchmark harness:
- `backend/app/scripts/run_scoring_sanity_benchmark.py`

Capabilities added:
- Synthetic benchmark cases:
  - identical
  - same task but different performance
  - clearly different/unrelated
  - low-signal/invalid-like
- Real-video mode (`--real`)
- Structured output prints:
  - valid frame counts
  - sequence/alignment lengths
  - DTW distances
  - metric values
  - final score
  - confidence flags

#### Outcome
Enabled iterative tuning and objective before/after checks without touching UI.

---

### 4.6 Score compression and insufficient discrimination

#### Problem
Even with real pipeline active, score still too forgiving in certain mismatch cases.

#### Root causes
1. Metric normalization constants were too loose (theoretical maxima, not practical ranges)
2. DTW cost signal was computed but not strongly represented in final decision behavior
3. Some quality metrics can contribute "free points" unless balanced carefully

#### Fixes
##### (a) Normalization tuning
- `backend/app/services/angle_metrics_service.py`
- `backend/app/services/trajectory_metrics_service.py`
- `backend/app/services/velocity_metrics_service.py`

Updated constants to increase sensitivity.

##### (b) Add DTW-derived similarity to metrics/scoring
- `backend/app/core/evaluation_constants.py`
- `backend/app/services/evaluation_engine_service.py`
- `backend/app/services/scoring_service.py`
- `backend/app/schemas/evaluation_schema.py`
- `backend/app/services/feedback_structuring_service.py`
- `backend/app/scripts/run_scoring_sanity_benchmark.py`

Changes included:
- `dtw_similarity` metric added to schema and scoring breakdown
- DTW normalized cost carried into metric build path
- New deterministic conversion from cost -> similarity
- Weights rebalanced to improve discrimination

#### Outcome
Identical cases preserved high score while mismatch cases separated more consistently.

---

### 4.7 Learner video "Change video" not reliably applying

#### Problem
User changed input video but playback/state appeared unchanged.

#### Root causes
- Blob URL lifecycle handling issues (revocation timing)
- Need for stronger remount behavior on video src change
- MIME-only validation can fail for some local files

#### Fix
`frontend/src/pages/CompareStudio.tsx`

- Added explicit "Change video" control
- Unified file handling logic for drop + button paths
- Improved blob URL lifecycle management (replace/unmount)
- Added `key={userVideoUrl}` on learner video element
- Relaxed file validation to support extension fallback where needed
- Reset stale evaluation state when a new learner video is chosen

#### Outcome
User can reliably replace learner video input in Compare Studio.

---

### 4.8 Hard rejection layer for out-of-context inputs

#### Problem
Need deterministic early-stop when learner clip is clearly unrelated/unusable.

#### Requirement
If rejected:
- stop before normal comparison path
- return `status = "out_of_context"`
- return `score = 0`
- return clear user-facing message

#### Initial implementation
Created:
- `backend/app/services/context_gate_service.py`

Integrated in:
- `backend/app/api/routes/evaluations.py`
- `backend/app/schemas/upload_schema.py`

Gate inserted after motion extraction but before evaluation pipeline call.

#### Signals used
- valid landmark coverage
- hand presence ratio
- usable frame count
- motion energy and energy ratio
- workspace center drift
- (optional) dtw similarity signal

#### Response shape (out-of-context)
`EvaluationStartResponse` extended with:
- `status`
- `score`
- `gate_status`
- `gate_reasons`
- `message`

#### Frontend handling
- `frontend/src/api/evaluationApi.js`
- `frontend/src/pages/CompareStudio.tsx`

If start response is `out_of_context`, frontend:
- stops normal result polling path
- shows rejection state with score 0
- allows retry

---

### 4.9 Gate too permissive in real scenario -> second tightening pass

#### Problem
Real out-of-context example still passed in one observed case.

#### Why this happened
Some unrelated clips can still have sufficient hand detection + moderate motion, so basic thresholds are not enough.

#### Tightening updates
`backend/app/services/context_gate_service.py`

Added stronger deterministic checks:
- **relative hand coverage vs expert**
- **activity-pattern similarity** (motion intensity timeline similarity)
- **hand-scale ratio vs expert**

Thresholds tightened for mismatch detection while preserving same-video path.

#### Outcome
Out-of-context rejection behavior became stricter and more aligned with user expectation.

---

### 4.10 "No VLM explanation when out_of_context"

#### Problem
UI still showed detailed reason bullets that looked like explanation output.

#### Fix
`frontend/src/pages/CompareStudio.tsx`

- Removed reason-list display block from learner-facing out-of-context panel
- Kept concise rejection message and retry action only

#### Outcome
No explanation-style content shown when hard rejection triggers.

---

## 5) Impasses encountered and how they were resolved

1. **Decode errors blocked evaluation start**
   - Resolved by form parsing branch in route

2. **DB session mode inconsistency**
   - Resolved by synchronizing route/service usage with sync SQLite session

3. **State-machine UI dead-end at processing step 5**
   - Resolved by explicit evaluation store reset after result

4. **Score did not reflect video semantics**
   - Root cause discovered: synthetic motion path in live endpoint
   - Resolved by replacing with real motion representation path

5. **Need confidence while tuning scoring**
   - Resolved by adding benchmark harness + real mode

6. **Out-of-context still slipping through initial gate**
   - Resolved by adding stronger relative/contextual signals and tighter thresholds

---

## 6) File-level change log (this cycle)

### Backend - created

- `backend/app/scripts/run_scoring_sanity_benchmark.py`
- `backend/app/services/context_gate_service.py`

### Backend - modified

- `backend/app/api/routes/evaluations.py`
  - real upload parsing
  - real motion extraction integration
  - context gate short-circuit
  - richer debug logging
- `backend/app/api/routes/chapters.py`
  - async->sync DB compatibility
- `backend/app/services/media_service.py`
  - async->sync DB compatibility
- `backend/app/services/evaluation_engine_service.py`
  - DTW similarity integration into metric construction
  - debug logging of DTW and metric inputs
- `backend/app/services/scoring_service.py`
  - breakdown support for expanded metric set
- `backend/app/core/evaluation_constants.py`
  - updated scoring constants/weights and DTW-related settings
- `backend/app/services/angle_metrics_service.py`
  - normalization sensitivity tuning
- `backend/app/services/trajectory_metrics_service.py`
  - normalization sensitivity tuning
- `backend/app/services/velocity_metrics_service.py`
  - normalization sensitivity tuning
- `backend/app/services/feedback_structuring_service.py`
  - mapping support for new metric labels
- `backend/app/schemas/evaluation_schema.py`
  - additional metric field(s) in result contracts
- `backend/app/schemas/upload_schema.py`
  - start response extended for gate status payload

### Frontend - created

- `frontend/src/api/evaluationApi.js`

### Frontend - modified

- `frontend/src/pages/CompareStudio.tsx`
  - real API flow
  - processing->done transition fix
  - change-video control and robust file replacement
  - out-of-context handling branch
  - out-of-context UI cleanup (no explanation-like details)
- `frontend/src/services/api/endpoints.ts`
  - endpoint integration updates

### Documentation generated in this period

- `docs/phase3.md`
- `docs/phase4.md`
- `docs/phase5.md`
- `docs/problems.md` (this file)

---

## 7) Current contracts relevant to this incident

### Start evaluation response (`POST /api/evaluations/start`)

#### Normal success
- `status = "completed"`
- returns `evaluation_id`

#### Out-of-context rejection
- `status = "out_of_context"`
- `score = 0`
- `message` with user guidance
- `gate_status = "rejected"`
- `gate_reasons` for diagnostics/internal visibility

### Frontend behavior

- If status is `out_of_context`, frontend does not proceed with normal result polling/rendering path
- Shows rejection panel and retry action

---

## 8) Verification evidence used

1. Backend tests:
- `app/tests/test_evaluation_engine.py`
- `app/tests/test_evaluation_persistence.py`

2. Benchmark runs:
- `python -m app.scripts.run_scoring_sanity_benchmark`
- `python -m app.scripts.run_scoring_sanity_benchmark --real`

3. Runtime logs:
- evaluation input/path logs
- DTW and metric debug logs
- context-gate logs (`gate_entered`, `gate_passed`, reasons, signal values)

---

## 9) Remaining limitations and risk notes

1. **Threshold tuning remains data-dependent**
   - Context gate thresholds may still need calibration as dataset variety grows.

2. **Potential false rejects/accepts**
   - Deterministic gates are transparent and controllable, but no fixed threshold set is perfect for all tasks.

3. **Current out-of-context path returns early from start route**
   - This is intentional for hard rejection, but should remain documented for client teams.

4. **Legacy placeholder endpoints still exist**
   - `/api/evaluations/{id}/result` fallback exists for compatibility; primary read path is `/api/evaluations/{id}`.

---

## 10) Practical handoff summary for teammates

If you are continuing development from here:

1. Treat `context_gate_service.py` as the deterministic out-of-context authority.
2. For tuning, use benchmark + real clips and inspect gate signal logs.
3. Keep frontend out-of-context UX minimal and explicit (no explanation block).
4. Keep scoring changes tied to benchmark evidence to avoid regression-by-opinion.
5. Keep real motion path in `evaluations.py`; do not reintroduce synthetic inputs into live endpoint.

---

## 11) Suggested next actions

1. Add a small offline calibration script for gate-threshold tuning on labeled clips.
2. Add automated tests specifically for out-of-context branch (`status=out_of_context`).
3. Add a dashboard/log export for gate signal distributions across accepted/rejected uploads.
4. Move long-term DB/schema changes to explicit migrations when ready.

---

## 12) One-line conclusion

The major failures were caused by request-format mismatch, DB session mismatch, UI state transition issues, and most critically a synthetic-motion path in live evaluation; all were corrected, scoring was made more sensitive, and a deterministic hard rejection layer was added and tightened for out-of-context videos.
