# Phase 1 Step 1 Verification

This verifies the real upload and expert-video foundation for AugMentor 2.0.

## 1. Start the correct backend

Use AugMentor 2.0 on port `8001` so it does not conflict with AugMentor 1.0 on `8000`.

```powershell
cd "C:\Users\aliaz\Desktop\AugMentor\AugMentor 2.0\backend"
uvicorn app.main:app --reload --port 8001
```

## 2. Seed one valid expert video + chapter

Place at least one supported video under `backend/storage/expert/`. Nested folders are allowed.

Example:

- `backend/storage/expert/pottery.mp4`
- `backend/storage/expert/chapter 1/pottery.mp4`

Then seed the demo course/chapter/expert-video row:

```powershell
cd "C:\Users\aliaz\Desktop\AugMentor\AugMentor 2.0\backend"
python -m app.scripts.seed_upload_demo_context
```

Expected output includes:

- `course_id=...`
- `chapter_id=...`
- `expert_video_id=...`
- `expert_video_storage_key=expert/...`

Keep the printed `chapter_id`. You need it for the learner upload test.

## 3. Prove the expert video URL works

Open this in the browser:

```text
http://localhost:8001/api/chapters/default/expert-video
```

Expected JSON shape:

```json
{
  "filename": "pottery.mp4",
  "storage_key": "expert/pottery.mp4",
  "file_path": "expert/pottery.mp4",
  "url": "/storage/expert/pottery.mp4",
  "chapter_id": "UUID or null",
  "expert_video_id": "UUID or null"
}
```

Then open the `url` value directly against the backend host:

```text
http://localhost:8001/storage/expert/pottery.mp4
```

Expected result:

- browser starts downloading or playing the video
- response status is `200`
- response `Content-Type` starts with `video/`

## 4. Prove the frontend uses the correct expert video URL

Start the frontend after updating the proxy:

```powershell
cd "C:\Users\aliaz\Desktop\AugMentor\AugMentor 2.0\frontend"
npm run dev
```

Open:

- `http://localhost:5173/courses`
- `http://localhost:5173/compare`

If port `5173` is already taken, Vite will print the next port in the terminal, for example `5174`.

In browser DevTools > Network:

- filter by `expert-video`
- confirm `GET /api/chapters/default/expert-video` succeeds
- filter by `pottery.mp4` or `storage`
- confirm the browser requests `/storage/...` and gets `200`
- confirm the video duration is no longer stuck at `0:00`

## 5. Success test for learner upload

Endpoint:

```text
POST http://localhost:8001/api/uploads/practice-video
```

Content type:

```text
multipart/form-data
```

Form fields:

- `chapter_id`: `<seeded chapter UUID>`
- `file`: `<mp4|mov|avi|webm video>`
- `user_id`: optional

Swagger test:

```text
http://localhost:8001/docs
```

Expected success response:

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

## 6. Invalid test for learner upload

Use either of these:

- upload `notes.txt` instead of a video
- use a random `chapter_id` that does not exist

Expected failures:

```json
{
  "detail": "Unsupported video format. Allowed formats: mp4, mov, avi, webm."
}
```

or:

```json
{
  "detail": "Chapter not found."
}
```

## 7. Prove the DB row exists

Open Postgres and run:

```sql
select id, chapter_id, user_id, status, original_filename, file_size_bytes, video_path, created_at
from learner_attempts
order by created_at desc;
```

Expected:

- new row with the returned `attempt_id`
- `status = 'uploaded'`
- `original_filename` matches the uploaded file
- `video_path` matches `stored_path`

## 8. Prove the file exists on disk

Check:

```text
backend/storage/uploads/learner_videos/<attempt_id>/original.mp4
```

Expected:

- the file exists
- file size is greater than zero

## 9. Automated route checks

These route-level tests verify the upload API response contract and clean invalid-format errors:

```powershell
cd "C:\Users\aliaz\Desktop\AugMentor\AugMentor 2.0\backend"
pytest tests/test_step1_upload_routes.py
```
