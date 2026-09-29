"""Route tests for notebook PDF export."""

from __future__ import annotations

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import chat_repo, content_registry_repo
from src.services.auth_service import hash_password


def _login(client: object, app: object, username: str) -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password("pw")))
            db.session.commit()
    client.post("/login", data={"username": username, "password": "pw"})  # type: ignore[attr-defined]


def _seed_notebook(client: object, app: object, username: str) -> int:
    _login(client, app, username)
    with app.app_context():
        user = db.session.query(User).filter_by(username=username).one()
        nb = Notebook(
            user_id=user.id,
            name="Export NB",
            description="A notebook to export.",
            summary="Summary with unicode: café naïve.",
            suggested_questions='["What is this?"]',
        )
        db.session.add(nb)
        db.session.commit()
        nb_id = nb.id
        db.session.add(
            Source(
                notebook_id=nb_id,
                filename="doc.txt",
                content_hash="e" * 64,
                content_type="txt",
                status="ready",
                char_count=10,
            )
        )
        db.session.commit()
        content_registry_repo.get_or_create("e" * 64, "doc_e", "export text", 11)
        chat_repo.create_message(nb_id, "user", "What is this?")
        chat_repo.create_message(
            nb_id,
            "assistant",
            "An answer.",
            sources_json='[{"filename": "doc.txt", "page": null}]',
        )
    return nb_id


class TestExportNotebook:
    def test_returns_pdf(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        nb_id = _seed_notebook(client, app, "export1")
        res = client.get(f"/notebooks/{nb_id}/export")  # type: ignore[attr-defined]
        assert res.status_code == 200
        assert res.content_type == "application/pdf"
        assert res.data.startswith(b"%PDF")
        assert f"notebook-{nb_id}.pdf" in res.headers.get("Content-Disposition", "")

    def test_empty_notebook_exports(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        _login(client, app, "export2")
        with app.app_context():
            user = db.session.query(User).filter_by(username="export2").one()
            nb = Notebook(user_id=user.id, name="Empty NB")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id
        res = client.get(f"/notebooks/{nb_id}/export")  # type: ignore[attr-defined]
        assert res.status_code == 200
        assert res.data.startswith(b"%PDF")

    def test_non_owner_404(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        nb_id = _seed_notebook(client, app, "export3")
        _login(client, app, "export4")
        res = client.get(f"/notebooks/{nb_id}/export")  # type: ignore[attr-defined]
        assert res.status_code == 404

    def test_login_required(self, client: object, app: object) -> None:
        res = client.get("/notebooks/1/export")  # type: ignore[attr-defined]
        assert res.status_code in (302, 401)
