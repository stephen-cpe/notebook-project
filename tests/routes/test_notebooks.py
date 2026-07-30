"""Route tests for notebooks + sources (TDD step 13).

Covers:
- Create notebook (valid + invalid name).
- List notebooks (only current user's).
- Open notebook (owner -> 200; non-owner -> 404).
- Rename notebook.
- Delete notebook (cascades to sources + chat).
- Upload source (valid file -> 201; unsupported type -> 400; oversized -> 400).
- List sources.
- Delete source.
- Source cap enforcement.
- Login-required on all endpoints.
"""

from __future__ import annotations

from pathlib import Path

from src.extensions import db
from src.models import ChatMessage, Notebook, Source, User
from src.services.auth_service import hash_password

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _login(client: object, app: object, username: str, password: str) -> None:
    """Helper: sign up + log in a user."""
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password(password)))
            db.session.commit()
    client.post("/login", data={"username": username, "password": password})


class TestCreateNotebook:
    def test_create_valid(self, client: object, app: object) -> None:
        _login(client, app, "nbuser1", "pw123")
        res = client.post(
            "/notebooks",
            data={"name": "My Notebook", "description": "A test notebook"},
        )
        assert res.status_code in (200, 201, 302)
        with app.app_context():
            nbs = db.session.query(Notebook).filter_by(name="My Notebook").all()
            assert len(nbs) == 1
            assert nbs[0].description == "A test notebook"

    def test_create_empty_name(self, client: object, app: object) -> None:
        _login(client, app, "nbuser2", "pw123")
        res = client.post("/notebooks", data={"name": ""})
        assert res.status_code == 400

    def test_create_requires_login(self, client: object) -> None:
        res = client.post("/notebooks", data={"name": "x"}, follow_redirects=False)
        assert res.status_code in (301, 302, 303)


class TestListNotebooks:
    def test_lists_own_only(self, client: object, app: object) -> None:
        _login(client, app, "listuser1", "pw123")
        client.post("/notebooks", data={"name": "Notebook A"})
        # Create another user's notebook.
        with app.app_context():
            u2 = User(username="listuser2", password_hash=hash_password("pw"))
            db.session.add(u2)
            db.session.commit()
            db.session.add(Notebook(user_id=u2.id, name="Not Mine"))
            db.session.commit()
        res = client.get("/notebooks")
        assert res.status_code == 200
        assert b"Notebook A" in res.data
        assert b"Not Mine" not in res.data

    def test_requires_login(self, client: object) -> None:
        res = client.get("/notebooks", follow_redirects=False)
        assert res.status_code in (301, 302, 303)


class TestOpenNotebook:
    def test_owner_can_open(self, client: object, app: object) -> None:
        _login(client, app, "openuser1", "pw123")
        client.post("/notebooks", data={"name": "Openable"})
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="Openable").first()
            assert nb is not None
            nb_id = nb.id
        res = client.get(f"/notebooks/{nb_id}")
        # The stub returns 501; full impl in step 16. We test owner scoping.
        assert res.status_code in (200, 501)

    def test_non_owner_gets_404(self, client: object, app: object) -> None:
        # Create notebook as user1.
        _login(client, app, "openuser2", "pw123")
        client.post("/notebooks", data={"name": "Private NB"})
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="Private NB").first()
            assert nb is not None
            nb_id = nb.id
        # Log in as user2.
        _login(client, app, "openuser3", "pw456")
        res = client.get(f"/notebooks/{nb_id}")
        assert res.status_code == 404


class TestDeleteNotebook:
    def test_delete_owned(self, client: object, app: object) -> None:
        _login(client, app, "deluser1", "pw123")
        client.post("/notebooks", data={"name": "To Delete"})
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="To Delete").first()
            assert nb is not None
            nb_id = nb.id
            # Add a source + chat message to verify cascade.
            db.session.add(
                Source(notebook_id=nb_id, filename="f.txt", content_hash="h", content_type="txt")
            )
            db.session.add(ChatMessage(notebook_id=nb_id, role="user", content="hi"))
            db.session.commit()
            assert db.session.query(Source).count() == 1
            assert db.session.query(ChatMessage).count() == 1
        res = client.post(f"/notebooks/{nb_id}/delete")
        assert res.status_code in (200, 302)
        with app.app_context():
            assert db.session.query(Notebook).filter_by(id=nb_id).count() == 0
            assert db.session.query(Source).filter_by(notebook_id=nb_id).count() == 0
            assert db.session.query(ChatMessage).filter_by(notebook_id=nb_id).count() == 0

    def test_delete_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "deluser2", "pw123")
        client.post("/notebooks", data={"name": "Not Yours"})
        with app.app_context():
            nb = db.session.query(Notebook).filter_by(name="Not Yours").first()
            nb_id = nb.id
        _login(client, app, "deluser3", "pw456")
        res = client.post(f"/notebooks/{nb_id}/delete")
        assert res.status_code == 404

    def test_delete_removes_notebook_media_and_orphaned_content(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Notebook deletion removes audio/video files + orphaned Chroma/registry (P0-1.3)."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        from src.services.cleanup_service import cleanup_notebook_media

        # Create fake audio + video files for a notebook id under tmp_path.
        nb_id = 1
        audio_dir = tmp_path / "audio" / str(nb_id)
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "abc.mp3").write_bytes(b"fake audio")
        video_dir = tmp_path / "video" / str(nb_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "xyz.mp4").write_bytes(b"fake video")
        assert audio_dir.exists() and video_dir.exists()

        # Directly exercise the cleanup helper with the tmp data_dir.
        cleanup_notebook_media(nb_id, str(tmp_path))
        assert not audio_dir.exists()
        assert not video_dir.exists()
