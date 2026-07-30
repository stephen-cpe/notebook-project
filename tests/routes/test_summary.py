"""Route tests for summary endpoints."""

from __future__ import annotations

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
    client.post("/notebooks", data={"name": "Summary NB"})
    with app.app_context():
        nb = db.session.query(Notebook).filter_by(name="Summary NB").first()
        assert nb is not None
        nb_id = nb.id
        h = "s" * 64
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
            chroma_collection="doc_s",
            extracted_text="This document discusses Python programming and data science.",
            char_count=60,
        )
    return nb_id


class TestGetSummary:
    def test_returns_summary(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "sumroute1")
        res = client.get(f"/notebooks/{nb_id}/summary")
        assert res.status_code == 200
        data = res.get_json()
        assert "summary" in data
        assert "suggested_questions" in data

    def test_non_owner_404(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "sumroute2")
        _login(client, app, "sumroute3")
        res = client.get(f"/notebooks/{nb_id}/summary")
        assert res.status_code == 404


class TestRegenerateSummary:
    def test_regenerates(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb_id = _create_notebook_with_source(client, app, monkeypatch, "sumroute4")
        res = client.post(f"/notebooks/{nb_id}/summary/regenerate")
        assert res.status_code == 200
        data = res.get_json()
        assert "summary" in data
        assert "suggested_questions" in data
