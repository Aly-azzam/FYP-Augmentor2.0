# Database Status - AugMentor 2.0

We finished the backend DB migration phase for AugMentor 2.0 without changing the fixed plan.

## Completed

- PostgreSQL schema is in use.
- Canonical SQLAlchemy models are active.
- Parity and smoke checks passed.
- Legacy ORM duplication was cleaned.
- Routes were audited.
- Courses, chapters, uploads, evaluations, history, and progress are DB-backed.
- Upload creates `videos` + `attempts` correctly.
- Evaluation can start from `attempt_id`.
- `out_of_context` evaluations are persisted.
- Attempt status is updated consistently.
- `evaluation_feedback` is persisted.
- History and progress reflect DB state.

## Phase Result

The DB connection/migration phase is complete.

## Direction From Now On

Do not reopen database redesign.

Next work should focus on:

- Business logic
- Evaluation quality
- Frontend integration
- Cleanup only

---

## Full Database Documentation

### 1) Scope and Constraints Used During Migration

This database phase was completed under strict constraints:

- Keep the fixed implementation plan.
- Keep PostgreSQL as the single database.
- Keep sync SQLAlchemy only (no async DB layer).
- Do not redesign architecture.
- Do not change existing DB schema/table names.
- Do not re-enable `Base.metadata.create_all()`.
- Use canonical ORM models only.

### 2) Final Database Stack

- Backend: FastAPI
- ORM: SQLAlchemy (sync session pattern)
- DB: PostgreSQL
- Driver: `psycopg` (sync usage)
- Persistence mode: existing real tables, ORM mapped to them

### 3) Canonical Tables in Active Use

The backend now uses these PostgreSQL tables as the source of truth:

- `users`
- `courses`
- `chapters`
- `videos`
- `attempts`
- `evaluations`
- `evaluation_feedback`

### 4) Canonical ORM Model Set

Canonical models are aligned with real table names and used by routes/services:

- `User` -> `users`
- `Course` -> `courses`
- `Chapter` -> `chapters`
- `Video` -> `videos`
- `Attempt` -> `attempts`
- `Evaluation` -> `evaluations`
- `EvaluationFeedback` -> `evaluation_feedback`

Legacy duplicate mappings were removed/neutralized so these canonical models are the active mapping layer.

### 5) Relationship and Linking Rules

Core links now enforced through the active flow:

- `videos.chapter_id` links uploaded/ expert videos to a chapter.
- `attempts.learner_video_id` links an attempt to the learner upload in `videos`.
- `attempts.chapter_id` links attempt context to `chapters`.
- `evaluations.attempt_id` links evaluation to exactly one attempt.
- `evaluations.expert_video_id` and `evaluations.learner_video_id` link compared videos.
- `evaluation_feedback.evaluation_id` links one feedback payload to one evaluation.

### 6) Route Migration Outcome (DB-backed State)

All target route groups are now DB-backed:

- `courses` routes: read from `courses`/`chapters`.
- `chapters` routes: read from `chapters`/`videos`.
- `uploads` route: persists learner upload + attempt.
- `evaluations` routes: persist/read evaluation state and feedback.
- `history` route: reads joined attempt/evaluation/chapter/course data.
- `progress` route: computes progress from persisted attempts/chapters/courses.

### 7) End-to-End Persistence Flow

#### Upload flow

`POST /api/uploads/practice-video`:

1. Validate input (`chapter_id`, `file`, `user_id`).
2. Save file to storage.
3. Insert learner upload row in `videos` (`video_role=learner_upload`).
4. Insert attempt row in `attempts` linked by `learner_video_id`.
5. Return persisted IDs and storage path info.

#### Evaluation start flow (attempt-driven)

`POST /api/evaluations/start`:

1. Accept `attempt_id` (Swagger-visible request schema).
2. Resolve learner video path from persisted attempt linkage when no fresh file is provided.
3. Resolve expert video for chapter context.
4. Run gate + evaluation pipeline.

#### Successful evaluation persistence

On success:

- Persist/update `evaluations` row (`status=completed`, metrics, breakdown, summary).
- Update `attempts.status` consistently (`evaluated`).
- Persist/update `evaluation_feedback` row (explanation, strengths/weaknesses/advice if available).

#### Out-of-context persistence

On gate rejection:

- Persist/update `evaluations` row (`status=out_of_context`, `gate_passed=false`, `gate_reasons`).
- Update `attempts.status` to `out_of_context`.
- Persist/update linked `evaluation_feedback` with context-gate explanation and reasons.

### 8) Status Consistency Rules in Practice

Current flow now keeps DB state aligned with API outcome:

- Upload accepted -> attempt is `uploaded`.
- Evaluation completed -> attempt is `evaluated`.
- Gate rejected -> attempt is `out_of_context`.

Evaluation rows are persisted for both completed and out-of-context outcomes so history/progress/reporting read consistent DB state.

### 9) Verification Performed

This phase included:

- Runtime DB connectivity checks (`SELECT 1`).
- ORM smoke checks against all canonical models.
- Model/metadata parity checks against live schema.
- Route-level audit and conversion from mock/partial to DB-backed behavior.
- Manual end-to-end validation:
  - upload -> videos/attempts persisted
  - evaluation from `attempt_id`
  - out-of-context persistence
  - feedback persistence
  - history/progress reflecting DB state

### 10) Current API Contract Notes

- `POST /api/uploads/practice-video` expects explicit `user_id`.
- `POST /api/evaluations/start` supports request body with `attempt_id`.
- Evaluation read endpoints now return persisted data rather than placeholders.

### 11) What Is Explicitly Out of Scope Now

The database phase is closed. The following are out of scope unless a critical bug appears:

- Schema redesign
- Table renaming
- New DB architecture
- Async migration of DB layer

### 12) Recommended Focus for Next Phases

- Improve evaluation/business logic quality.
- Improve context-gate reliability and scoring behavior.
- Frontend integration polish and UX.
- Technical cleanup and testing hardening.
