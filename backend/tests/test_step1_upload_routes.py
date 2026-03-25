from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes import uploads
from app.main import app
from app.services.upload_service import UploadResult, UploadValidationError


@pytest.fixture()
def client():
    original_startup = list(app.router.on_startup)
    app.router.on_startup.clear()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.router.on_startup[:] = original_startup


def test_practice_video_upload_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    attempt_id = uuid.uuid4()
    chapter_id = uuid.uuid4()
    expert_video_id = uuid.uuid4()
    stored_path = f"uploads/learner_videos/{attempt_id}/original.mp4"

    async def fake_create_upload(db, *, chapter_id, file, user_id=None):
        return UploadResult(
            attempt=SimpleNamespace(
                id=attempt_id,
                chapter_id=chapter_id,
                status="uploaded",
                original_filename=file.filename,
                video_path=stored_path,
            ),
            expert_video_id=expert_video_id,
        )

    monkeypatch.setattr(uploads, "create_learner_attempt_upload", fake_create_upload)

    response = client.post(
        "/api/uploads/practice-video",
        data={"chapter_id": str(chapter_id)},
        files={"file": ("practice.mp4", b"fake-video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt_id"] == str(attempt_id)
    assert payload["chapter_id"] == str(chapter_id)
    assert payload["expert_video_id"] == str(expert_video_id)
    assert payload["upload_status"] == "uploaded"
    assert payload["original_filename"] == "practice.mp4"
    assert payload["stored_path"] == stored_path
    assert payload["video_url"] == f"/storage/{stored_path}"


def test_practice_video_upload_invalid_format(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_upload(db, *, chapter_id, file, user_id=None):
        raise UploadValidationError("Unsupported video format. Allowed formats: mp4, mov, avi, webm.")

    monkeypatch.setattr(uploads, "create_learner_attempt_upload", fake_create_upload)

    response = client.post(
        "/api/uploads/practice-video",
        data={"chapter_id": str(uuid.uuid4())},
        files={"file": ("notes.txt", b"not-a-video", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unsupported video format. Allowed formats: mp4, mov, avi, webm."
    }
