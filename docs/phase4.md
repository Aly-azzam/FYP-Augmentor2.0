# AugMentor 2.0 - Phase 4 Technical Handoff (4.1 to 4.4)

This document explains exactly what was implemented in Phase 4, why it was added, where the code lives, and how to continue safely.

It is written as a teammate handoff so a new developer can continue without reverse-engineering the pipeline.

---

## Phase 4 Scope

Phase 4 adds semantic understanding and grounded explanation on top of the Phase 3 aligned metrics pipeline:

- **4.1** Semantic phase segmentation (V-JEPA-inspired, deterministic)
- **4.2** Error-to-semantic-phase fusion
- **4.3** Grounded explanation generation (rule-based)
- **4.4** Explanation architecture refactor (payload + multi-mode dispatcher)

Design constraints respected:

- deterministic and explainable behavior
- no external LLM/VLM API calls in current implementation
- no changes to DTW alignment algorithm
- no changes to metric formulas or scoring formulas
- backward compatibility retained in evaluation output contract

---

## Where Phase 4 Fits in the Pipeline

Current modern pipeline (high level):

1. Load expert and learner motion outputs
2. Run DTW alignment (Phase 3.1)
3. Build aligned channels and compute metrics (Phase 3.2)
4. Compute deterministic score + metric breakdown (Phase 3.3)
5. Detect key error moments (Phase 3.4)
6. **Analyze semantic phases (Phase 4.1)**
7. **Map error moments to semantic phases (Phase 4.2)**
8. **Generate grounded explanation (Phase 4.3)**
9. **Route explanation through mode-based architecture (Phase 4.4)**

---

## 4.1 - Semantic Understanding Layer (V-JEPA-Inspired)

### Goal

Add a lightweight semantic interpretation layer to split each video into action phases, without replacing CV metrics.

### File Created

- `backend/app/services/vjepa_service.py`

### Main Function

- `analyze_video_semantics(motion_data)`

### Input

- `MotionRepresentationOutput`-like payload or dict with `frames`
- Uses hand velocity channels from each frame:
  - `left_hand_features.wrist_velocity`
  - `right_hand_features.wrist_velocity`
  - `left_hand_features.palm_velocity`
  - `right_hand_features.palm_velocity`

### Core Logic

1. Build `motion_intensity_series` per frame using average velocity magnitude.
2. Compute global intensity stats (mean/std) and local slope thresholds.
3. Classify each frame into one label:
   - `preparation phase`
   - `execution phase`
   - `steady phase`
   - `finishing phase`
4. Merge consecutive same-label frames into semantic segments.

### Output

Returns:

- `phases`: list of `{label, start_frame, end_frame}`
- `motion_intensity_series`: rounded float sequence

If frames are missing, returns a deterministic fallback phase:

- `preparation phase` from frame `0` to `0`

---

## 4.2 - Fuse Error Moments with Semantic Phases

### Goal

Turn generic localized errors into contextualized errors (for example: "path drift during execution phase").

### File Modified

- `backend/app/services/comparison_service.py`

### Main Function

- `map_errors_to_phases(error_moments, semantic_phases)`

### Mapping Rule

For each error moment:

1. Use expert error range:
   - `start_expert_frame`
   - `end_expert_frame`
2. Compare against expert semantic phases.
3. Select phase with **maximum overlap**.
4. If no overlap exists, fallback to **closest phase by boundary distance**.

### Output Enrichment

Each error moment gains:

- `semantic_phase`: `{label, start_frame, end_frame}`
- `semantic_label`: `"<error label> during <phase label>"`

Examples:

- `path drift during preparation phase`
- `inconsistent speed during execution phase`

### Engine Integration

In `backend/app/services/evaluation_engine_service.py`:

- semantic phases are computed for both expert and learner
- key error moments are computed
- key error moments are fused with semantic phase info via `map_errors_to_phases(...)`

---

## 4.3 - Grounded Explanation Generation (Rule-Based)

### Goal

Produce deterministic human-readable feedback grounded only in computed evaluation outputs.

### Files Modified

- `backend/app/services/feedback_structuring_service.py`
- `backend/app/services/evaluation_engine_service.py`
- `backend/app/schemas/evaluation_schema.py`
- `backend/app/scripts/run_evaluation_demo.py`
- `backend/app/tests/test_evaluation_engine.py`

### Rule-Based Explanation Function

Originally introduced as:

- `generate_explanation(evaluation_result)`

It uses:

- `score`
- `per_metric_breakdown` (best/worst metric quality)
- `key_error_moments` with `semantic_label`

### Explanation Structure

Generated output contains:

- overall performance sentence (score-bucket based)
- strengths list (best metric based)
- weaknesses list (top error moments with semantic labels)
- advice sentence (worst metric based)

### Grounding Principle

No hallucination or external calls. Every phrase is derived from:

- metric values and breakdown
- detected error moments
- semantic labels from 4.2

---

## 4.4 - Explanation Modes Architecture (Rule + Future VLM)

### Goal

Refactor explanation generation into a model-agnostic architecture:

`analysis -> structured payload -> explanation generator`

This prepares future VLM integration without changing metric/scoring logic.

### Files Modified

- `backend/app/services/feedback_structuring_service.py`
- `backend/app/services/evaluation_engine_service.py`
- `backend/app/schemas/evaluation_schema.py`
- `backend/app/core/config.py`
- `backend/app/scripts/run_evaluation_demo.py`
- `backend/app/tests/test_evaluation_engine.py`

### New/Updated Functions

In `backend/app/services/feedback_structuring_service.py`:

1. `build_explanation_payload(evaluation_result)`
   - Extracts model-agnostic factual payload:
     - `score`
     - `metrics`
     - `per_metric_breakdown`
     - `key_error_moments`
     - `semantic_phases`
     - `strength`
     - `weakness`

2. `generate_rule_based_explanation(payload)`
   - Keeps the existing deterministic explanation behavior
   - Now consumes the payload instead of full result object

3. `generate_vlm_explanation(payload)`
   - Placeholder only (no external API calls)
   - Returns:
     - explanation text `"VLM explanation not implemented yet"`
     - mode marker `vlm_placeholder`

4. `generate_explanation(evaluation_result, mode="rule")`
   - Dispatcher:
     - builds payload
     - routes to rule-based or VLM placeholder mode
     - defaults to `rule`

### Configuration

In `backend/app/core/config.py`:

- Added `EXPLANATION_MODE: str = "rule"`

In `backend/app/services/evaluation_engine_service.py`:

- Explanation generation uses `settings.EXPLANATION_MODE`
- Current default behavior remains rule-based

### Schema Contract Updates

In `backend/app/schemas/evaluation_schema.py`:

- `ExplanationOutput` now includes:
  - `mode: str`
  - `explanation: str`
  - `strengths: list[str]`
  - `weaknesses: list[str]`
  - `advice: str`

- `VLMInputPayload` now includes:
  - `explanation_payload: dict[str, Any]`
  - `explanation: ExplanationOutput`

This allows downstream systems to consume both:

- factual payload
- generated text

### Demo Visibility

In `backend/app/scripts/run_evaluation_demo.py`:

- explanation mode is printed explicitly:
  - `mode: rule_based`

---

## Evaluation Result and VLM Payload (Phase 4 State)

`EvaluationResult` now carries:

- `score`
- `metrics`
- `per_metric_breakdown`
- `key_error_moments` (with semantic enrichment)
- `semantic_phases` (expert + learner)
- `explanation` (with `mode`)
- `summary`
- `vlm_payload`

`VLMInputPayload` now carries:

- core score/metrics/summary fields
- semantic phases
- key error moments
- `explanation_payload` (model-agnostic facts)
- `explanation` (current generated output)

---

## File-Level Change Log for Phase 4

### Created

- `backend/app/services/vjepa_service.py`
- `docs/phase4.md`

### Modified

- `backend/app/services/evaluation_engine_service.py`
  - integrate semantic analysis
  - map errors to phases
  - compute and attach explanation output
  - integrate explanation mode/config and payload forwarding

- `backend/app/services/comparison_service.py`
  - add `map_errors_to_phases(...)`
  - enrich key error moments with `semantic_phase` and `semantic_label`

- `backend/app/services/feedback_structuring_service.py`
  - add explanation payload builder
  - split rule-based generator
  - add VLM placeholder generator
  - add mode dispatcher
  - propagate explanation payload into VLM payload

- `backend/app/schemas/evaluation_schema.py`
  - add semantic phase schema outputs
  - add explanation output schema
  - add semantic and explanation fields in result payloads
  - add explanation mode field
  - add explanation payload field in VLM payload

- `backend/app/core/config.py`
  - add `EXPLANATION_MODE`

- `backend/app/scripts/run_evaluation_demo.py`
  - print semantic phases
  - print semantic-labeled error moments
  - print explanation text and explanation mode

- `backend/app/tests/test_evaluation_engine.py`
  - semantic phase existence/validity tests
  - error-to-phase mapping stability tests
  - grounded explanation existence + semantic label usage tests
  - deterministic explanation tests
  - Phase 4.4 payload/mode dispatcher tests

---

## Determinism and Safety Notes

- No random or probabilistic explanation logic added.
- No network dependency in explanation generation path.
- `mode="vlm"` is intentionally a non-calling placeholder to keep architecture ready while safe.
- Rule-based outputs remain stable for same inputs.

---

## Known Limitations (Current Intended State)

1. `generate_vlm_explanation(...)` is placeholder only.
2. Explanation content quality depends on error detection quality from Phase 3.4.
3. Current semantic segmentation is heuristic; it is deterministic but not learned.
4. `EXPLANATION_MODE` is global config-based; request-level mode override is not yet exposed in API.

---

## How to Continue (Recommended Next Steps)

### 1) Real VLM integration (next logical phase)

- Replace `generate_vlm_explanation(payload)` placeholder with real adapter.
- Keep payload contract unchanged.
- Add strict guardrails:
  - no metric invention
  - must reference factual payload fields

### 2) Add API-level explanation mode override

- Keep config default (`rule`), but allow optional request override per evaluation call.

### 3) Add richer explanation artifacts for frontend overlays

- include optional references from weaknesses to error IDs/indices for clickable timeline highlights.

### 4) Prompt/template versioning

- if VLM mode is added, include `prompt_version` and `explanation_version` in payload/output.

---

## Quick Verification Commands

From `backend` directory:

- `pytest -q app/tests/test_evaluation_engine.py`
- `python -m app.scripts.run_evaluation_demo`

Expected:

- evaluation runs successfully
- semantic phases are printed
- key error moments include semantic labels
- explanation is printed with mode (`rule_based` by default)
- `mode="vlm"` path is available in code and does not crash

---

## Practical Teammate Summary

If you are continuing from this point:

1. Do not change metric/scoring contracts for VLM work.
2. Treat `build_explanation_payload(...)` as the single source of truth for explanation input.
3. Keep rule mode as deterministic fallback even after VLM is integrated.
4. Keep `evaluation_schema.py` payload compatibility to avoid breaking frontend/API clients.

