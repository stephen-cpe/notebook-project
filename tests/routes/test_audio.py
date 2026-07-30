"""Route tests for audio endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.auth_service import hash_password


def _login(client: object, app: object, username: str) -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password("pw")))
            db.session.commit()
    client.post("/login", data={"username": username, "password": "pw"})


def _create_notebook_with_source(
    client: object, app: object, monkeypatch: pytest.MonkeyPatch, username: str
) -> int:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("AI_MOCK", "true")
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
    _login(client, app, username)
    client.post("/notebooks", data={"name": "Audio NB"})
    with app.app_context():
        nb = db.session.query(Notebook).filter_by(name="Audio NB").first()
        assert nb is not None
        nb_id = nb.id
        h = "a" * 64
        db.session.add(
            Source(
                notebook_id=nb_id,
                filename="doc.txt",
                content_hash=h,
                content_type="txt",
                status="ready",
            )
        )
        db.session.commit()
        content_registry_repo.get_or_create(
            content_hash=h,
            chroma_collection="doc_a",
            extracted_text="This document discusses machine learning and neural networks.",
            char_count=60,
        )
    return nb_id


class TestRequestAudio:
    def test_queues_generation(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "audroute1")
        res = client.post(f"/notebooks/{nb_id}/audio")
        assert res.status_code in (200, 202)
        data = res.get_json()
        assert data["status"] == "queued"

    def test_non_owner_404(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "audroute2")
        _login(client, app, "audroute3")
        res = client.post(f"/notebooks/{nb_id}/audio")
        assert res.status_code == 404


class TestAudioStatus:
    def test_returns_status(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "audroute4")
        res = client.get(f"/notebooks/{nb_id}/audio/status")
        assert res.status_code == 200
        data = res.get_json()
        assert "status" in data
        assert "has_audio" in data


class TestDeleteAudio:
    def test_deletes_audio(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "audroute5")
        # Generate audio first.
        client.post(f"/notebooks/{nb_id}/audio")
        time.sleep(2)
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            if nb.audio_status == "ready":
                res = client.delete(f"/notebooks/{nb_id}/audio")
                assert res.status_code == 200
                db.session.refresh(nb)
                assert nb.audio_status == "none"
                assert nb.audio_path is None
