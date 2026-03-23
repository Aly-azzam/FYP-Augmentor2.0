# Evaluation Engine Progress

## 1. Purpose

This document explains what was implemented for the **Evaluation Engine** in the backend of **AugMentor 2.0** so the rest of the team can quickly understand the current progress.

This work was done by **Ahmad Dia** for the part of the pipeline responsible for:

- receiving expert motion JSON
- receiving learner motion JSON
- comparing both inputs
- computing evaluation metrics
- computing a final score out of 100
- generating a structured summary
- preparing a payload for the future AI explanation layer

---

## 2. What Was Built

The Evaluation Engine now has a basic but working modular structure.

It currently supports:

- loading motion JSON files
- validating the required structure
- pairing expert and learner motion data
- computing:
  - angle deviation
  - trajectory deviation
  - velocity difference
  - tool alignment deviation
- combining metrics into a final score
- generating a structured summary
- preparing a VLM payload
- running the full pipeline manually with mock JSON data
- testing the pipeline with a simple pytest file

---

## 3. Main Files Created

### `backend/app/schemas/evaluation_schema.py`

Contains the schemas used by the Evaluation Engine:

- `MetricSet`
- `EvaluationSummary`
- `VLMInputPayload`
- `EvaluationResult`

It also keeps compatibility models needed by the existing backend:

- `EvaluationMetrics`
- `EvaluationResultOut`
- `HistoryEntry`
- `ProgressOut`

This was important because some old routes and services were still importing these names.

---

### `backend/app/core/evaluation_constants.py`

Contains constants used for scoring:

- required metric names
- metric weights
- min/max score
- qualitative thresholds such as:
  - excellent
  - good
  - fair
  - needs improvement

The goal was to make the scoring logic easy to adjust later.

---

### `backend/app/utils/evaluation_utils.py`

Contains small reusable helpers:

- `round_metric()`
- `safe_average()`
- `clamp()`

These are used by the metric services and scoring service.

---

### `backend/app/services/comparison_service.py`

Handles input preparation for the Evaluation Engine.

It currently provides:

- `load_motion_data()`
- `validate_motion_data()`
- `pair_motion_data()`

It validates the required top-level fields:

- `video_id`
- `angle_series`
- `joint_trajectories`
- `velocity_profiles`
- `tool_motion`

It also now validates nested values:

- `angle_series` must contain lists of numbers
- `velocity_profiles` must contain lists of numbers
- `joint_trajectories` must contain lists of 2D points
- `tool_motion` must contain lists of 2D points

This helps catch bad input earlier with clearer errors.

---

### `backend/app/services/angle_metrics_service.py`

Computes a normalized angle deviation score between expert and learner using:

- shared angle keys only
- shared length only if series lengths differ

Main functions:

- `compute_series_difference()`
- `compute_angle_deviation()`

Output is normalized between `0.0` and `1.0`.

---

### `backend/app/services/trajectory_metrics_service.py`

Computes trajectory deviation using 2D point-by-point distance.

Main functions:

- `compute_point_distance()`
- `compute_trajectory_difference()`
- `compute_trajectory_deviation()`

Only shared trajectory keys are compared.
Different lengths are trimmed to the minimum shared length.

Output is normalized between `0.0` and `1.0`.

---

### `backend/app/services/velocity_metrics_service.py`

Computes normalized velocity difference between expert and learner.

Main functions:

- `compute_velocity_series_difference()`
- `compute_velocity_difference()`

Only shared velocity keys are compared.
Different lengths are trimmed to the minimum shared length.

Output is normalized between `0.0` and `1.0`.

---

### `backend/app/services/tool_metrics_service.py`

Computes normalized tool alignment deviation from 2D tool motion paths.

Main functions:

- `compute_tool_path_difference()`
- `compute_tool_alignment_deviation()`

Only shared tool keys are compared.
Different lengths are trimmed to the minimum shared length.

Output is normalized between `0.0` and `1.0`.

---

### `backend/app/services/scoring_service.py`

Combines the metric outputs into a final score.

Main functions:

- `compute_total_deviation()`
- `compute_final_score()`
- `classify_score()`

Current logic:

- lower deviation = better performance
- weighted total deviation is computed
- score is converted to a value from `0` to `100`
- a simple label is also returned:
  - `excellent`
  - `good`
  - `fair`
  - `needs_improvement`
  - `poor`

---

### `backend/app/services/feedback_structuring_service.py`

Builds the structured summary and VLM payload.

Main functions:

- `get_best_metric()`
- `get_worst_metric()`
- `build_summary()`
- `build_vlm_payload()`
- `structure_feedback()`

Current summary rules:

- lowest deviation metric = `main_strength`
- highest deviation metric = `main_weakness`
- highest deviation metric also defines `focus_area`

Metric names are converted into simple human-readable phrases.

---

### `backend/app/services/evaluation_engine_service.py`

This is the main orchestrator of the Evaluation Engine.

Main functions:

- `generate_evaluation_id()`
- `build_metric_set()`
- `evaluate_motion_pair()`
- `run_evaluation_pipeline()`

Current pipeline:

1. load expert motion JSON
2. load learner motion JSON
3. validate and pair both inputs
4. compute all four metrics
5. compute final score
6. build summary
7. build VLM payload
8. return final `EvaluationResult`

A small cleanup was also applied here to remove duplicate validation.

---

## 4. Mock Data Added

For local testing, these mock files were created:

- `backend/storage/processed/motion/expert_mock.json`
- `backend/storage/processed/motion/learner_mock.json`

They contain realistic small sample data with:

- `video_id`
- `angle_series`
- `joint_trajectories`
- `velocity_profiles`
- `tool_motion`

The learner file has small differences compared to the expert file so the metrics produce non-zero results.

---

## 5. Testing and Demo Files

### `backend/app/tests/test_evaluation_engine.py`

Simple pytest file that checks:

- the pipeline returns an `EvaluationResult`
- score is between `0` and `100`
- all metric fields exist
- summary fields exist
- VLM payload exists

### `backend/app/scripts/run_evaluation_demo.py`

Simple manual demo script that:

- loads the mock JSON files
- runs the evaluation engine
- prints:
  - `evaluation_id`
  - `score`
  - metrics
  - summary
  - VLM payload

This is useful before integrating the engine into API routes.

---

## 6. Minimal Fixes Applied After Review

After reviewing the module, a few small fixes were applied:

### Schema compatibility fix

The first version of `evaluation_schema.py` only contained the new Evaluation Engine models.
This caused compatibility issues because the existing backend still expected:

- `EvaluationMetrics`
- `EvaluationResultOut`
- `HistoryEntry`
- `ProgressOut`

These were added back to keep the rest of the backend working.

### Validation improvement

`comparison_service.py` was improved so invalid nested data structures fail early with clearer errors.

### Cleanup in orchestration

`evaluation_engine_service.py` had duplicate validation.
This was removed because `pair_motion_data()` already performs validation.

### Cleanup in feedback service

`feedback_structuring_service.py` had an unused `score` parameter in `build_summary()`.
This was cleaned up by using the score in a very small way to adjust wording.

---

## 7. Current Limitations

This module is intentionally simple for the current FYP phase.

Current limitations:

- metric logic is simple and rule-based
- no advanced temporal alignment yet
- no database persistence yet
- no API route integration yet
- no async job queue yet
- no real VLM call yet
- no side-by-side media generation yet
- test execution still depends on installing project dependencies and pytest locally

---

## 8. How To Run Locally

From the `backend` folder:

### Run the demo script

```bash
python -m app.scripts.run_evaluation_demo
```

### Run the test

```bash
python -m pytest app/tests/test_evaluation_engine.py
```

If `pytest` or backend packages are missing, install them first:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

---

## 9. Final Note For The Team

The Evaluation Engine now has a clean base architecture and can already run end to end using mock motion JSON.

It is not the final production version yet, but it gives the team:

- a clear input/output contract
- modular metric services
- a scoring layer
- a summary layer
- a VLM-ready payload
- local test/demo support

This should make future integration with the Motion Representation Engine, API routes, and AI explanation layer easier.
