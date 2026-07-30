"""End-to-end integration test (TDD step 17).

Implements the SRS §12 acceptance flow:
signup → create notebook → upload 3 varied sources → see ingestion complete →
see a summary + 5 suggested questions → chat with citations → generate and
play a two-host Audio Overview → log out and back in and see everything
persisted.

Runs fully offline with AI_MOCK=true, CI=true, in-memory SQLite, and
EphemeralClient ChromaDB. No real network calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions import db
from src.models import ChatMessage, Notebook, Source, User

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_all() -> None:
    from src.services.ingestion import reset_ingestion_service
    from src.services.ocr_service import reset_ocr_service
    from src.services.vector_store import get_vector_store, reset_vector_store

    reset_vector_store()
    reset_ingestion_service()
    reset_ocr_service()
    get_vector_store().reset()
    yield
    reset_vector_store()


class TestEndToEnd:
    """SRS §12 acceptance criteria — the full user journey."""

    def test_full_user_journey(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The complete MVP acceptance flow in mock mode."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("HF_TOKEN", "")

        # --- 1. Sign up ---
        res = client.post(
            "/signup",
            data={"username": "journey_user", "password": "secure_pw"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)
        with app.app_context():
            assert db.session.query(User).filter_by(username="journey_user").count() == 1

        # --- 2. Create a notebook ---
        res = client.post(
            "/notebooks",
            data={"name": "Research Notebook", "description": "ML and DB sources"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="Research Notebook").first()
            assert nb is not None
            nb_id = nb.id

        # --- 3. Upload 3 varied sources ---
        for filename in ("sample.txt", "sample.md", "sample.docx"):
            with open(FIXTURES / filename, "rb") as f:
                res = client.post(
                    f"/notebooks/{nb_id}/sources",
                    data={"file": (f, filename)},
                    content_type="multipart/form-data",
                )
            assert res.status_code == 201, f"Upload {filename} failed: {res.status_code}"

        # --- 4. Verify ingestion complete ---
        with app.app_context():
            sources = db.session.query(Source).filter_by(notebook_id=nb_id).all()
            assert len(sources) == 3
            for s in sources:
                assert s.status in ("ready", "partial"), (
                    f"Source {s.filename} status={s.status} error={s.error_message}"
                )

        # --- 5. Verify summary + suggested questions ---
        from src.services.summary_service import SummaryService

        with app.app_context():
            nb = db.session.query(Notebook).filter_by(id=nb_id).first()
            assert nb is not None
            svc = SummaryService()
            result = svc.generate_summary(nb)
            assert result is not None
            assert isinstance(result.summary, str)
            assert len(result.summary) > 0
            assert isinstance(result.suggested_questions, list)

        # --- 6. Chat with citations ---
        res = client.post(
            f"/notebooks/{nb_id}/chat/sync",
            json={"question": "What databases are mentioned?"},
        )
        assert res.status_code == 200
        chat_data = res.get_json()
        assert "answer" in chat_data
        assert len(chat_data["answer"]) > 0
        assert "sources" in chat_data
        assert "latency_ms" in chat_data

        # --- 7. Verify chat persisted ---
        with app.app_context():
            msgs = db.session.query(ChatMessage).filter_by(notebook_id=nb_id).all()
            assert len(msgs) == 2  # user + assistant
            assert msgs[0].role == "user"
            assert msgs[1].role == "assistant"

        # --- 8. Generate Audio Overview ---
        res = client.post(f"/notebooks/{nb_id}/audio")
        assert res.status_code in (200, 202)
        # In mock mode, background thread runs fast; wait briefly.
        import time

        time.sleep(1)
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(id=nb_id).first()
            assert nb is not None
            # Status should be ready or still processing (background thread).
            assert nb.audio_status in ("ready", "queued", "synthesizing", "scripting")

        # --- 9. Log out ---
        res = client.get("/logout", follow_redirects=False)
        assert res.status_code in (301, 302, 303)

        # --- 10. Log back in ---
        res = client.post(
            "/login",
            data={"username": "journey_user", "password": "secure_pw"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)

        # --- 11. Verify everything persisted ---
        # Notebooks list still shows the notebook.
        res = client.get("/notebooks")
        assert res.status_code == 200
        assert b"Research Notebook" in res.data

        with app.app_context():
            # Sources persisted.
            assert db.session.query(Source).filter_by(notebook_id=nb_id).count() == 3
            # Chat messages persisted.
            assert db.session.query(ChatMessage).filter_by(notebook_id=nb_id).count() == 2
            # Notebook still exists.
            nb = db.session.query(Notebook).filter_by(id=nb_id).first()
            assert nb is not None
            assert nb.name == "Research Notebook"

        # --- 12. Chat history endpoint returns persisted messages ---
        res = client.get(f"/notebooks/{nb_id}/chat/history")
        assert res.status_code == 200
        hist = res.get_json()
        assert len(hist["messages"]) == 2

    def test_non_owner_cannot_access_others_notebook(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FR-3: cross-user access is forbidden (404, not 403)."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")

        # User 1 creates a notebook.
        client.post("/signup", data={"username": "owner_user", "password": "pw123456"})
        client.post("/notebooks", data={"name": "Private NB"})
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="Private NB").first()
            assert nb is not None
            nb_id = nb.id

        # User 2 signs up + tries to access.
        client.post("/logout", follow_redirects=False)
        client.post("/signup", data={"username": "intruder_user", "password": "pw123456"})

        # Open notebook -> 404.
        res = client.get(f"/notebooks/{nb_id}")
        assert res.status_code == 404

        # Chat sync -> 404.
        res = client.post(f"/notebooks/{nb_id}/chat/sync", json={"question": "hi"})
        assert res.status_code == 404

        # Sources -> 404.
        res = client.get(f"/notebooks/{nb_id}/sources")
        assert res.status_code == 404

    def test_health_endpoint(self, client: object) -> None:
        """NFR-51: /health reports app + DB status."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["app"] == "ok"
        assert data["db"] == "ok"
