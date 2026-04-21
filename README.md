# AugMentor 2.0

### Vision–Language–Action Framework for Craft Skill Learning

AugMentor 2.0 is an AI-powered system designed to help learners improve **manual craft skills** by analyzing and comparing their movements with those of an expert.

The system processes **egocentric craft videos** and uses computer vision and AI models to extract motion information, analyze gesture quality, and generate meaningful feedback.

This project is developed as a **Final Year Project (FYP)** in Computer & Communication Engineering in collaboration with **Mines Paris – PSL University**.

---

## Project Concept

Learning complex manual skills such as carving, sculpting, or other crafting techniques often relies on observing experts and practicing repeatedly. However, many important details of expert movements are **subtle and difficult to notice** in normal videos.

AugMentor 2.0 aims to transform instructional videos into **interactive learning tools** by automatically extracting motion features and highlighting the key elements of expert performance.

By comparing a learner’s attempt with an expert’s demonstration, the system can identify differences in motion and provide explanations that help improve technique.

---

## Core Idea

The system analyzes two videos:

- **Expert video** – demonstration of the correct technique  
- **Learner video** – attempt performed by the student  

Using AI and computer vision, the system:

1. Detects hand and object movements  
2. Extracts motion features such as trajectories and gesture patterns  
3. Compares the learner's motion with the expert's motion  
4. Generates explanations and visual feedback to guide improvement  

This approach turns simple craft videos into **augmented learning material**.

---

## Technologies

The project explores the integration of several AI technologies:

- **Computer Vision** for hand and object detection
- **Motion Analysis** to measure gesture quality
- **Vision-Language Models (VLMs)** to generate explanations
- **Video Augmentation** for pedagogical visualization

These components work together to convert raw videos into **structured motion data and instructional feedback**.

---

## Project Status

🚧 **Work in progress**

This repository currently contains the early development of the AugMentor 2.0 system.  
The architecture and components are under active development.

---

## Development Note (Expert Upload)

If the **Expert Upload / Expert Video Manager** page shows:

`Could not load chapters. Make sure the backend is running.`

the frontend cannot reach the backend API.

- Frontend dev server proxies `/api` and `/storage` to `http://localhost:8001` by default.
- Start the backend on port `8001` before using Expert Upload:

`uvicorn app.main:app --reload --port 8001`

- If you use a different backend port, set `AUGMENTOR_API_TARGET` in the frontend environment to match it.

---

## Database Recovery & Test Safety

If DB rows are lost but files still exist under `backend/storage`, run:

- Audit only (read-only):
  - `python -m app.scripts.recover_storage_registry`
- Re-register recoverable expert rows:
  - `python -m app.scripts.recover_storage_registry --apply --create-missing-chapters`

Notes:
- Recovery only re-registers expert records when a source video file still exists on disk.
- Learner runtime run folders (`storage/mediapipe/runs`, `storage/sam2/runs`) are audit-only in this helper.

### Pytest DB isolation (important)

Pytest now requires `TEST_DATABASE_URL` and refuses to run if it matches the app `DATABASE_URL`.

Example:
- `set TEST_DATABASE_URL=postgresql+psycopg://augmentor_user:augmentor_password@localhost:5432/augmentor_test_db`
- `pytest`

This prevents destructive test setup (`drop_all/create_all`) from touching the real app database.

---

## Contributors

Final Year Project – Computer & Communication Engineering

**Students**
- Ali Azzam
- Ahmad Dia
- Rafic Dergham


**Supervisors**
- Dr. Alina Glushkova – Mines Paris PSL  

---

## Research Context

AugMentor 2.0 extends an earlier prototype by introducing new AI models and analytics capable of extracting motion indicators from craft videos and producing annotated learning material.
