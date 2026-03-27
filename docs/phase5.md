# AugMentor 2.0 - Phase 5 Technical Handoff (5.1 to 5.4)

This document explains what was implemented in Phase 5, why it was implemented, and exactly which files were created/modified so another developer can continue quickly and safely.

---

## Phase 5 Scope

Phase 5 moved the evaluation system from compute-only outputs to persisted outputs with retrieval and progress APIs:

- **5.1** SQLite persistence foundation
- **5.2** Save evaluation results
- **5.3** Retrieve evaluation history
- **5.4** Compute and expose progress

Design constraints respected:

- Existing metric/scoring/evaluation logic was not changed.
- Persistence was added after evaluation computation.
- Existing tests for evaluation pipeline remained green after changes.

---

## Architecture Before vs After

### Before Phase 5

- Evaluation pipeline produced in-memory `EvaluationResult` objects.
- No guaranteed DB persistence for full modern outputs.
- No direct API to fetch persisted evaluation by id.
- No history endpoint for repeated attempts.
- No progress aggregation endpoint for attempt-level trends.

### After Phase 5

- Every evaluation run is persisted into `evaluation_results`.
- Evaluation id returned by the engine is now the DB row id.
- API supports:
  - fetch by id
  - fetch history by attempt id
  - fetch progress summary by attempt id
- Demo script validates DB roundtrip (save + fetch).

---

## 5.1 - Real Database Setup (SQLite Dev)

### Goal

Use SQLite as development persistence backend with working engine/session/table creation.

### Files Modified

- `backend/app/core/config.py`
- `backend/app/core/database.py`
- `backend/app/main.py`
- `backend/app/models/*.py` (SQLite compatibility pass)

### Key Changes

1. **Database URL switched to SQLite**
   - `DATABASE_URL = "sqlite:///./augmentor.db"`

2. **SQLAlchemy sync engine/session setup**
   - `engine = create_engine(..., connect_args={"check_same_thread": False})`
   - `SessionLocal = sessionmaker(...)`
   - `Base = declarative_base()`
   - `get_db()` dependency yields and closes session

3. **Startup table creation**
   - `Base.metadata.create_all(bind=engine)` is called at app startup after model imports.

4. **SQLite compatibility updates in models**
   - PostgreSQL-specific types replaced with SQLite-safe forms:
     - UUID columns moved to `String(36)` ids/fks
     - JSONB moved to `JSON`

### Why this was needed

SQLite is file-based and used for local dev velocity. The project previously used PostgreSQL-specific patterns that are not portable without conversion.

---

## 5.2 - Store Evaluation Results

### Goal

Persist full modern evaluation outputs (score + analysis artifacts) immediately after evaluation.

### Files Modified

- `backend/app/models/evaluation_result.py`
- `backend/app/services/evaluation_engine_service.py`
- `backend/app/schemas/evaluation_schema.py`

### Data Model Contract Implemented

`evaluation_results` table now stores:

- `id` (PK)
- `attempt_id` (indexed FK)
- `score` (float)
- `metrics` (JSON)
- `per_metric_breakdown` (JSON)
- `key_error_moments` (JSON)
- `semantic_phases` (JSON)
- `explanation` (JSON)
- `created_at` (timestamp)

Important:

- `attempt_id` is **not unique** to allow multiple evaluations (history) for the same attempt context.

### Engine Persistence Flow

In `evaluation_engine_service.py`:

1. Pipeline computes final structured result as before.
2. `_persist_evaluation_result(...)` writes to DB.
3. Stored row id is returned as `evaluation_id` in output schema.

### Compatibility Guard

Because some local DB files may exist with an older table schema, a defensive schema guard was added:

- `_ensure_evaluation_results_schema()`
  - checks table columns via `PRAGMA table_info(evaluation_results)`
  - adds missing JSON columns through `ALTER TABLE ... ADD COLUMN ...`

This avoids immediate failure when running against an old `augmentor.db` during dev.

---

## 5.3 - Retrieve Evaluation History

### Goal

Allow querying persisted evaluation records for an attempt over time.

### Files Modified

- `backend/app/api/routes/evaluations.py`
- `backend/app/schemas/evaluation_schema.py`

### Endpoints Added

1. **Get evaluation by id**
   - `GET /api/evaluations/{evaluation_id}`
   - Returns one persisted record or 404.

2. **Get history by attempt**
   - `GET /api/evaluations/history/{attempt_id}`
   - Returns list sorted by `created_at DESC`.

### Response Schemas Added

- `PersistedEvaluationOut`
- `EvaluationHistoryOut`

These schemas expose stored JSON payloads directly in structured API responses.

---

## 5.4 - Compute Progress

### Goal

Provide attempt-level progress summary from persisted scores.

### Files Created/Modified

- `backend/app/services/progress_service.py` (created)
- `backend/app/api/routes/evaluations.py` (modified)
- `backend/app/schemas/evaluation_schema.py` (modified)

### Service Function

- `compute_progress(evaluations)`

Returns:

- `best_score`
- `average_score`
- `attempt_count`
- `latest_score`

### Progress Endpoint Added

- `GET /api/evaluations/progress/{attempt_id}`

Flow:

1. Query all evaluations for attempt ordered by latest first.
2. Convert rows to dict payloads.
3. Compute stats via `compute_progress`.
4. Return `AttemptProgressOut`.

---

## Demo Update (Persistence Roundtrip)

### File Modified

- `backend/app/scripts/run_evaluation_demo.py`

### What changed

- Demo now ensures a demo attempt exists.
- Passes `attempt_id` into `run_evaluation_pipeline(...)`.
- Prints saved DB evaluation id.
- Queries DB for that id and prints fetched record.

This gives a visible end-to-end proof:

compute -> save -> fetch -> print.

---

## Test Coverage Added/Updated

### New Tests

- `backend/app/tests/test_evaluation_persistence.py`
  - evaluation is saved
  - retrieval by id works
  - history returns multiple rows sorted desc
  - progress computation is correct
  - progress endpoint responds with valid summary

### Updated Tests

- `backend/app/tests/test_database_setup.py`
  - reset schema for deterministic local test runs
  - validates insert/query behavior with updated model

### Regression Validation

Existing evaluation tests remained passing after persistence integration:

- `backend/app/tests/test_evaluation_engine.py`

---

## File-Level Change Log (Phase 5)

### Created

- `backend/app/services/progress_service.py`
- `backend/app/tests/test_evaluation_persistence.py`
- `docs/phase5.md`

### Modified

- `backend/app/core/config.py`
  - SQLite database URL

- `backend/app/core/database.py`
  - sync engine/session/base/get_db setup for SQLite

- `backend/app/main.py`
  - startup table creation

- `backend/app/models/evaluation_result.py`
  - modern persisted JSON fields for evaluation artifacts

- `backend/app/models/course.py`
- `backend/app/models/chapter.py`
- `backend/app/models/expert_video.py`
- `backend/app/models/learner_attempt.py`
- `backend/app/models/progress.py`
- `backend/app/models/user.py`
  - SQLite compatibility updates (id/fk/json portability)

- `backend/app/services/evaluation_engine_service.py`
  - DB persistence after evaluation computation
  - schema guard for existing SQLite table compatibility

- `backend/app/api/routes/evaluations.py`
  - get-by-id endpoint
  - history endpoint
  - progress endpoint

- `backend/app/schemas/evaluation_schema.py`
  - persisted/history/progress response schemas

- `backend/app/scripts/run_evaluation_demo.py`
  - persistence roundtrip print

- `backend/app/tests/test_database_setup.py`
  - schema reset + insert/query validation for current table shape

---

## Known Limitations (Current Intended State)

1. Persistence currently stores only evaluation-side payloads (by design for 5.2).
2. No migration framework (Alembic) yet; schema evolution is still lightweight/dev-focused.
3. Compatibility guard in engine is pragmatic for local DBs, not a replacement for formal migrations.
4. Some route handlers remain placeholder from earlier phases (`/start`, `/{id}/status`, `/{id}/result` legacy placeholder path).

---

## Why these decisions were made

- **SQLite first**: fastest iteration for dev and local testing.
- **JSON storage for complex outputs**: avoids premature relational decomposition of nested evaluation artifacts.
- **Persist after compute**: zero impact on scoring/metrics correctness.
- **History and progress by attempt**: directly supports frontend analytics and learner tracking needs.

---

## How to Continue (Recommended Next Steps)

### 1) Add proper DB migrations

- Introduce Alembic.
- Convert schema guard behavior into migration scripts.

### 2) Persist richer context

- Add optional fields for chapter/course/user linkage directly in stored evaluation rows if needed by dashboards.

### 3) Harden APIs

- Add pagination for history endpoint.
- Add auth/ownership checks for attempt-based queries.

### 4) Phase 5.5+ candidate

- Persist derived timeline artifacts for frontend visual overlays (frame ranges, semantic phase references, clip pointers).

---

## Quick Verification Commands

Run from `backend`:

- `pytest -q app/tests/test_database_setup.py`
- `pytest -q app/tests/test_evaluation_persistence.py`
- `pytest -q app/tests/test_evaluation_engine.py`
- `python -m app.scripts.run_evaluation_demo`

Expected:

- DB tests pass
- persistence/history/progress tests pass
- evaluation regression tests pass
- demo prints saved evaluation id and fetched DB record

---

## Practical Teammate Summary

If you are continuing from this point:

1. Keep evaluation compute logic untouched; persistence is already layered after compute.
2. Use `evaluation_results` JSON columns as source of truth for stored evaluation artifacts.
3. Extend retrieval/progress endpoints, not core metrics/scoring, for product-facing analytics work.
4. Plan migration tooling soon before schema complexity increases.

