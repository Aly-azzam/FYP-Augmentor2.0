# AugMentor 2.0 — Backend Architecture (Phase 0)

## 1. Product Flow

User flow:
- Browse courses
- Enter a course
- View chapters
- Each chapter has one expert video
- Watch expert video
- Upload learner video for that chapter
- Receive evaluation:
  - Score /100
  - Angle, trajectory, velocity metrics
  - AI explanation
  - Side-by-side video comparison

---

## 2. Frontend Sections

- Courses
- My Learning (current courses)
- Compare Studio (expert vs learner)
- History (all attempts)
- Achievements (frontend only for now)

---

## 3. Core Assumptions

- One chapter = one expert video
- One learner upload = one chapter
- Results are stored (history)
- Achievements are not part of backend V1

---

## 4. Backend Stack

- Python 3.10+
- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Docker
- Async processing
- Monorepo

---

## 5. System Pipeline

Frontend  
→ Backend API  
→ Upload / Job creation  
→ Perception Engine  
→ Motion Representation Engine  
→ Evaluation Engine  
→ VLM Explanation  
→ Response  

---

## 6. Module Responsibilities

### Perception (Rafic)
- video → landmarks + tools + (future V-JEPA)

### Motion Representation (Ali)
- landmarks → trajectories / velocity / kinematics

### Evaluation (Dia)
- motion → metrics + score + summary

---

## 7. Required Metrics (V1)

- Angle deviation
- Trajectory deviation
- Velocity difference
- Tool alignment deviation

---

## 8. Database Entities

- User
- Course
- Chapter
- ExpertVideo
- LearnerAttempt
- EvaluationResult
- HistoryEntry
- Progress

---

## 9. API (MVP)

- GET /api/courses
- GET /api/courses/{course_id}
- GET /api/chapters/{chapter_id}
- GET /api/chapters/{chapter_id}/expert-video

- POST /api/uploads/practice-video
- POST /api/evaluations/start

- GET /api/evaluations/{evaluation_id}/status
- GET /api/evaluations/{evaluation_id}/result

- GET /api/history
- GET /api/progress

---

## 10. Storage

storage/
- uploads/
- processed/landmarks/
- processed/motion/
- processed/tools/
- outputs/evaluations/

---

## 11. Key Decisions

- Async processing
- Tool detection required
- V-JEPA integration planned
- Modular architecture (no coupling)
- JSON contracts between modules