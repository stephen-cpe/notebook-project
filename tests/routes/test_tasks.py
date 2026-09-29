"""Route tests for the background-task bell feed."""

from __future__ import annotations

import pytest

from src.extensions import db
from src.models import User
from src.services.auth_service import hash_password
from src.services.progress_tracker import (
    claim_job,
    finish_job,
    reset_tracker,
    unread_count,
)


def _login(client: object, app: object, username: str = "tasksuser") -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password("pw")))
            db.session.commit()
    client.post("/login", data={"username": username, "password": "pw"})  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _clean_tracker() -> None:
    reset_tracker()
    yield
    reset_tracker()


class TestTasksFeed:
    def test_lists_jobs_with_status_fields(self, client: object, app: object) -> None:
        claim_job("audio:1", label="Audio Overview", notebook_id=1)
        finish_job("audio:1", label="Audio Overview ready")
        _login(client, app)
        res = client.get("/tasks")  # type: ignore[attr-defined]
        assert res.status_code == 200
        data = res.get_json()
        assert data["unread"] == 1
        assert len(data["tasks"]) == 1
        task = data["tasks"][0]
        assert task["status"] == "ready"
        assert task["read"] is False
        assert "updated_at" in task
        assert task["notebook_id"] == 1

    def test_empty_feed(self, client: object, app: object) -> None:
        _login(client, app, "tasksempty")
        res = client.get("/tasks")  # type: ignore[attr-defined]
        assert res.status_code == 200
        data = res.get_json()
        assert data["tasks"] == []
        assert data["unread"] == 0

    def test_login_required(self, client: object, app: object) -> None:
        res = client.get("/tasks")  # type: ignore[attr-defined]
        assert res.status_code in (302, 401)


class TestMarkRead:
    def test_marks_all_read(self, client: object, app: object) -> None:
        claim_job("audio:1", label="Audio", notebook_id=1)
        finish_job("audio:1")
        claim_job("video:1", label="Video", notebook_id=1)
        finish_job("video:1")
        _login(client, app, "tasksread")
        res = client.post("/tasks/read", json={})  # type: ignore[attr-defined]
        assert res.status_code == 200
        assert unread_count() == 0
        res = client.get("/tasks")  # type: ignore[attr-defined]
        assert res.get_json()["unread"] == 0

    def test_marks_single_key(self, client: object, app: object) -> None:
        claim_job("audio:1", label="Audio", notebook_id=1)
        finish_job("audio:1")
        claim_job("audio:2", label="Audio 2", notebook_id=2)
        finish_job("audio:2")
        _login(client, app, "tasksread2")
        res = client.post("/tasks/read", json={"key": "audio:1"})  # type: ignore[attr-defined]
        assert res.status_code == 200
        assert unread_count() == 1

    def test_login_required(self, client: object, app: object) -> None:
        res = client.post("/tasks/read", json={})  # type: ignore[attr-defined]
        assert res.status_code in (302, 401)
