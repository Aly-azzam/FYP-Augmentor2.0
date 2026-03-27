# AugMentor 2.0 - Phase 3 Technical Handoff (3.1 to 3.4)

This document explains exactly what was implemented in Phase 3, why it was done, and where the code lives so the next developer can continue without guessing.

---

## Phase 3 Scope

Phase 3 focused on replacing naive frame-by-frame comparison with temporal alignment and modern deterministic evaluation:

- **3.1** Temporal alignment (DTW)
- **3.2** Real aligned comparison metrics
- **3.3** Deterministic weighted final scoring with per-metric breakdown
- **3.4** Localized key error moments (segments)

Design constraints respected:

- Deterministic, explainable, no ML model usage
- Backward compatibility with legacy motion input where possible
- Modern path uses aligned Step 2.4 motion output (`frames[*].flattened_feature_vector`)

---

## Before Phase 3

Evaluation was previously driven by legacy motion structures (`angle_series`, `joint_trajectories`, `velocity_profiles`, `tool_motion`) and direct index-to-index pairing (or min-length clipping behavior inside legacy metric comparisons).

This caused timing/speed mismatch problems: if the learner performed the same motion at a different speed, the comparison quality degraded unfairly.

---

## 3.1 - Temporal Alignment (DTW)

### What was added

**File:** `backend/app/services/temporal_alignment_service.py`

Core function:

- `align_sequences(expert_motion, learner_motion)`

It:

1. Extracts sequence vectors from `frames[*].flattened_feature_vector`
2. Uses Euclidean frame distance
3. Builds full DTW cost matrix
4. Backtracks optimal path
5. Returns:
   - `alignment_path`
   - `aligned_pairs` (`expert_index`, `learner_index`)
   - `dtw_total_cost`
   - `dtw_normalized_cost`
   - `path_length`

### Integration

**File:** `backend/app/services/evaluation_engine_service.py`

Inside `evaluate_motion_pair()`:

- Loads expert/learner motion
- Detects modern Step 2.4 input
- Runs DTW
- Uses `aligned_pairs` to build aligned metric input before metric computation
- Keeps legacy fallback path for non-modern inputs

### Tests

DTW tests were added/validated in the backend test suite and include:

- identical sequence near-zero cost
- same motion at different speed still aligns
- monotonic path checks
- output shape checks

---

## 3.2 - Real Metrics on DTW-Aligned Data

### Goal

Compute meaningful metrics from aligned expert/learner pairs (not raw frame index matching).

### Channel extraction on aligned pairs

**File:** `backend/app/services/comparison_service.py`

`pair_motion_data_with_alignment(...)` was extended to:

- Build aligned expert/learner frame lists using DTW pair order
- Convert aligned frames into legacy-compatible metric structures
- Build modern aligned channel packs under:
  - `modern_aligned_channels`

Included channel groups:

- `trajectory_channels` (left/right wrist, palm center, index tip XY)
- `velocity_channels` (left/right wrist and palm speed magnitudes)
- `angle_channels` (4 joint-angle channels per hand)
- `hand_shape_channels` (`hand_openness`, `pinch_distance`, `finger_spread`)
- `alignment_path`, `expert_timestamps`, `learner_timestamps`

### Metric services updated to use modern aligned channels first

- `backend/app/services/angle_metrics_service.py`
  - `compute_angle_deviation(...)`
  - `compute_hand_openness_deviation(...)`
- `backend/app/services/trajectory_metrics_service.py`
  - `compute_trajectory_deviation(...)`
  - `compute_timing_score(...)`
- `backend/app/services/velocity_metrics_service.py`
  - `compute_velocity_difference(...)`
  - `compute_smoothness_score(...)`

### Metric set now includes

**Schema file:** `backend/app/schemas/evaluation_schema.py`

`MetricSet` now contains:

- `angle_deviation` (lower better)
- `trajectory_deviation` (lower better)
- `velocity_difference` (lower better)
- `smoothness_score` (higher better)
- `timing_score` (higher better)
- `hand_openness_deviation` (lower better)
- `tool_alignment_deviation` (legacy-compatible optional)

---

## 3.3 - Deterministic Final Score + Breakdown

### Goal

Make final score explicit and auditable from raw metrics.

### Scoring constants

**File:** `backend/app/core/evaluation_constants.py`

Defined:

- `DEVIATION_METRICS`
- `QUALITY_METRICS`
- `ACTIVE_SCORING_METRICS`
- `DEFAULT_METRIC_WEIGHTS`

Active scoring weights used:

- `trajectory_deviation`: 0.30
- `angle_deviation`: 0.25
- `velocity_difference`: 0.20
- `smoothness_score`: 0.15
- `timing_score`: 0.10

`hand_openness_deviation` and `tool_alignment_deviation` are still reported but currently weight `0.0` in active scoring to avoid destabilizing behavior.

### Scoring formula

**File:** `backend/app/services/scoring_service.py`

Quality conversion:

- Deviation metric -> `1 - value`
- Quality metric -> `value`

Then:

`score_quality = sum(weight * normalized_quality)`

`final_score = round(100 * score_quality)` clamped to `[0, 100]`

### Per-metric breakdown

Added explicit breakdown generation with:

- `raw_value`
- `normalized_quality`
- `weight`
- `contribution` (score points out of 100)

Returned by `compute_final_score(...)` as:

- `score`
- `score_quality`
- `label`
- `per_metric_breakdown`
- `contribution_sum`

### Schema support

**File:** `backend/app/schemas/evaluation_schema.py`

Added:

- `MetricBreakdownItem`
- `per_metric_breakdown` in:
  - `EvaluationResult`
  - `VLMInputPayload`

### Engine integration

**File:** `backend/app/services/evaluation_engine_service.py`

`evaluate_motion_pair()` now:

- Computes metrics
- Computes final score + breakdown
- Returns both in result
- Passes breakdown into summary/VLM payload creation

### Feedback directionality fix

**File:** `backend/app/services/feedback_structuring_service.py`

Best/worst metric logic updated to support mixed-direction metrics.

---

## 3.4 - Key Error Moments (Localized Segments)

### Goal

Move from only global metrics to localized “where it went wrong” moments for explainability and future VLM grounding.

### What was added

**File:** `backend/app/services/comparison_service.py`

New helpers:

- `build_angle_error_series(...)`
- `build_speed_error_series(...)`
- `build_trajectory_error_series(...)`
- `detect_top_error_segment(...)`
- `detect_key_error_moments(...)`

### Error signal definitions (per aligned pair)

- **angle_error:** mean absolute angle diff / 180
- **speed_error:** mean absolute scalar speed diff across velocity channels
- **trajectory_error:** mean normalized XY point distance across key tracked points

All normalized/clamped to `[0,1]`.

### Segment detection algorithm

For each error series:

1. Light smoothing (`window=3` moving average)
2. Threshold = `max(mean + 0.5 * std, 0.1)`
3. Group consecutive indices above threshold as segments
4. Rank by mean segment error
5. Pick top segment
6. If no segment crosses threshold, fallback to the highest local point

### Output object

Each moment includes:

- `error_type`
- `label`
- `start_aligned_index`, `end_aligned_index`
- `start_expert_frame`, `end_expert_frame`
- `start_learner_frame`, `end_learner_frame`
- `severity` in `[0,1]`

Default labels:

- `angle_error` -> `wrist angle issue`
- `trajectory_error` -> `path drift`
- `speed_error` -> `inconsistent speed`

### Schema support

**File:** `backend/app/schemas/evaluation_schema.py`

Added:

- `KeyErrorMoment`
- `key_error_moments` in:
  - `EvaluationResult`
  - `VLMInputPayload`

### Engine + feedback integration

**File:** `backend/app/services/evaluation_engine_service.py`

Modern path now computes:

- `key_error_moments = detect_key_error_moments(modern_aligned_channels)`

and returns/forwards them.

**File:** `backend/app/services/feedback_structuring_service.py`

`build_vlm_payload(...)` and `structure_feedback(...)` now accept and include `key_error_moments`.

---

## Demo Visibility

**File:** `backend/app/scripts/run_evaluation_demo.py`

Demo now prints:

- modern aligned path flag
- DTW path details
- metrics
- per-metric breakdown
- contribution sum
- key error moments with aligned + expert + learner ranges and severity

This makes each phase behavior visible during manual verification.

---

## Test Coverage Updated

**File:** `backend/app/tests/test_evaluation_engine.py`

Includes checks for:

- modern metric fields
- per-metric breakdown presence
- VLM payload mirroring breakdown
- key error moments present on modern path
- required error types (`angle_error`, `trajectory_error`, `speed_error`)
- valid ordering/ranges and severity bounds

**Scoring tests (already in project):**

- deterministic score
- score bounds
- contribution consistency
- directionality handling

---

## Current End-to-End Flow (Modern Path)

1. Step 2.4 motion outputs are loaded
2. DTW alignment computes aligned pairs
3. Aligned frames are converted to channel structures
4. Phase 3.2 metrics are computed from aligned channels
5. Phase 3.3 weighted deterministic score + breakdown is computed
6. Phase 3.4 key localized error moments are detected
7. Evaluation result + VLM payload include:
   - score
   - metrics
   - per_metric_breakdown
   - key_error_moments

---

## Known Limitations (Intentional MVP)

- `tool_alignment_deviation` is retained for compatibility but modern tool channels are not fully modeled yet.
- `hand_openness_deviation` is reported but weight is currently `0.0` in active score to preserve stability.
- Segment detection is intentionally simple threshold-based logic (deterministic, explainable baseline).

---

## Where to Continue Next

If continuing into next phase:

1. Add richer tool/object motion channels if needed and decide whether to score them.
2. Tune segment thresholds/window per domain after collecting real learner videos.
3. Use `key_error_moments` to generate clip-level explanation prompts for VLM.
4. Optionally export per-frame error traces for frontend overlays.

---

## File-Level Summary (Phase 3)

- `backend/app/services/temporal_alignment_service.py`  
  DTW algorithm + alignment path output.

- `backend/app/services/comparison_service.py`  
  Aligned frame pairing, modern channel extraction, frame-level error signals, segment detection.

- `backend/app/services/evaluation_engine_service.py`  
  Orchestration of modern aligned path, metric computation, scoring, error moments, payload assembly.

- `backend/app/services/angle_metrics_service.py`  
  Aligned angle + hand openness deviation.

- `backend/app/services/trajectory_metrics_service.py`  
  Aligned trajectory deviation + timing score.

- `backend/app/services/velocity_metrics_service.py`  
  Aligned velocity difference + smoothness score.

- `backend/app/services/scoring_service.py`  
  Deterministic weighted scoring and per-metric contribution breakdown.

- `backend/app/core/evaluation_constants.py`  
  Metric categories + active scoring weights.

- `backend/app/schemas/evaluation_schema.py`  
  Modern metric schema, breakdown schema, key error moments schema.

- `backend/app/services/feedback_structuring_service.py`  
  Mixed-direction best/worst logic, payload propagation for breakdown and key error moments.

- `backend/app/scripts/run_evaluation_demo.py`  
  Human-readable debug output for all Phase 3 artifacts.

- `backend/app/tests/test_evaluation_engine.py`  
  Validation for modern outputs including key error moments.

