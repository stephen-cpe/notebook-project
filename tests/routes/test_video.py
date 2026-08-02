"""Route tests for video overview endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, User
from src.services.auth_service import hash_password


def _login(client: object, app: object, username: str, password: str = "pw") -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password(password)))
            db.session.commit()
    client.post("/login", data={"username": username, "password": password})


def _create_notebook(client: object, app: object, name: str = "Video NB") -> int:
    client.post("/notebooks", data={"name": name})
    with app.app_context():
        nb = db.session.query(Notebook).filter_by(name=name).first()
        assert nb is not None
        return nb.id


def _set_video(app: object, notebook_id: int, path: str | None, status: str) -> None:
    with app.app_context():
        nb = db.session.get(Notebook, notebook_id)
        assert nb is not None
        nb.video_path = path
        nb.video_status = status
        db.session.commit()


class TestRequestVideo:
    def test_queues_video(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.routes.video.launch_video_job", lambda *a, **k: None)
        _login(client, app, "vidroute1")
        nb_id = _create_notebook(client, app)
        res = client.post(f"/notebooks/{nb_id}/video", json={"topic": "  ML  "})
        assert res.status_code == 202
        data = res.get_json()
        assert data["status"] == "queued"
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            assert nb.video_status == "queued"

    def test_login_required(self, client: object, app: object) -> None:
        _login(client, app, "vidroute2")
        nb_id = _create_notebook(client, app)
        client.get("/logout")
        res = client.post(f"/notebooks/{nb_id}/video")
        assert res.status_code == 302

    def test_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "vidroute3")
        nb_id = _create_notebook(client, app)
        _login(client, app, "vidroute4")
        res = client.post(f"/notebooks/{nb_id}/video")
        assert res.status_code == 404


class TestVideoStatus:
    def test_returns_status(self, client: object, app: object) -> None:
        _login(client, app, "vidroute4")
        nb_id = _create_notebook(client, app)
        _set_video(app, nb_id, None, "generating")
        res = client.get(f"/notebooks/{nb_id}/video/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "generating"
        assert data["has_video"] is False

    def test_ready_has_video(self, client: object, app: object, tmp_path: Path) -> None:
        _login(client, app, "vidroute5")
        nb_id = _create_notebook(client, app)
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        _set_video(app, nb_id, str(fake), "ready")
        res = client.get(f"/notebooks/{nb_id}/video/status")
        assert res.status_code == 200
        assert res.get_json()["has_video"] is True

    def test_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "vidroute6")
        nb_id = _create_notebook(client, app)
        _login(client, app, "vidroute7")
        res = client.get(f"/notebooks/{nb_id}/video/status")
        assert res.status_code == 404


class TestVideoFile:
    def test_serves_file(self, client: object, app: object, tmp_path: Path) -> None:
        _login(client, app, "vidroute8")
        nb_id = _create_notebook(client, app)
        fake = tmp_path / "out.mp4"
        fake.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        _set_video(app, nb_id, str(fake), "ready")
        res = client.get(f"/notebooks/{nb_id}/video/file")
        assert res.status_code == 200
        assert res.mimetype == "video/mp4"
        assert res.data == fake.read_bytes()

    def test_serves_relative_path(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _login(client, app, "vidroute9")
        nb_id = _create_notebook(client, app)
        fake = tmp_path / "rel.mp4"
        fake.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        _set_video(app, nb_id, "rel.mp4", "ready")
        res = client.get(f"/notebooks/{nb_id}/video/file")
        assert res.status_code == 200
        assert res.data == fake.read_bytes()

    def test_404_when_no_path(self, client: object, app: object) -> None:
        _login(client, app, "vidroute10")
        nb_id = _create_notebook(client, app)
        _set_video(app, nb_id, None, "none")
        res = client.get(f"/notebooks/{nb_id}/video/file")
        assert res.status_code == 404

    def test_404_when_missing_file(self, client: object, app: object) -> None:
        _login(client, app, "vidroute11")
        nb_id = _create_notebook(client, app)
        _set_video(app, nb_id, "C:/does/not/exist.mp4", "ready")
        res = client.get(f"/notebooks/{nb_id}/video/file")
        assert res.status_code == 404


class TestDeleteVideo:
    def test_deletes_video(self, client: object, app: object, tmp_path: Path) -> None:
        _login(client, app, "vidroute12")
        nb_id = _create_notebook(client, app)
        fake = tmp_path / "del.mp4"
        fake.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        _set_video(app, nb_id, str(fake), "ready")
        res = client.delete(f"/notebooks/{nb_id}/video")
        assert res.status_code == 200
        assert not fake.exists()
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            assert nb.video_path is None
            assert nb.video_status == "none"

    def test_delete_without_video(self, client: object, app: object) -> None:
        _login(client, app, "vidroute13")
        nb_id = _create_notebook(client, app)
        _set_video(app, nb_id, None, "none")
        res = client.delete(f"/notebooks/{nb_id}/video")
        assert res.status_code == 200

    def test_delete_relative_path(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _login(client, app, "vidroute14")
        nb_id = _create_notebook(client, app)
        fake = tmp_path / "rel_del.mp4"
        fake.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        _set_video(app, nb_id, "rel_del.mp4", "ready")
        res = client.delete(f"/notebooks/{nb_id}/video")
        assert res.status_code == 200
        assert not fake.exists()
