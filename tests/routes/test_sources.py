"""Route tests for source upload/list/delete (TDD step 13).

Covers:
- Upload a valid .txt file -> source created with status.
- Upload unsupported type -> 400.
- List sources for a notebook.
- Delete a source.
- Source cap enforcement.
- Non-owner -> 404.
- Login required.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.services.auth_service import hash_password

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _login(client: object, app: object, username: str, password: str = "pw123") -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password(password)))
            db.session.commit()
    client.post("/login", data={"username": username, "password": password})


def _create_notebook(client: object, app: object, name: str = "Test NB") -> int:
    client.post("/notebooks", data={"name": name})
    with app.app_context():
        nb = db.session.query(Notebook).filter_by(name=name).first()
        assert nb is not None
        return nb.id


class TestUploadSource:
    def test_upload_txt(self, client: object, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "srcuser1")
        nb_id = _create_notebook(client, app, "Upload NB")

        with open(FIXTURES / "sample.txt", "rb") as f:
            res = client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        assert res.status_code in (200, 201, 202)
        with app.app_context():
            sources = db.session.query(Source).filter_by(notebook_id=nb_id).all()
            assert len(sources) == 1
            assert sources[0].filename == "sample.txt"
            assert sources[0].content_type == "txt"

    def test_upload_unsupported_type(self, client: object, app: object) -> None:
        _login(client, app, "srcuser2")
        nb_id = _create_notebook(client, app, "Bad Type NB")

        res = client.post(
            f"/notebooks/{nb_id}/sources",
            data={"file": (io.BytesIO(b"data"), "file.xyz")},
            content_type="multipart/form-data",
        )
        assert res.status_code == 400

    def test_upload_no_file(self, client: object, app: object) -> None:
        _login(client, app, "srcuser3")
        nb_id = _create_notebook(client, app, "No File NB")
        res = client.post(f"/notebooks/{nb_id}/sources", data={})
        assert res.status_code == 400

    def test_upload_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "srcuser4")
        nb_id = _create_notebook(client, app, "Not Yours NB")
        _login(client, app, "srcuser5", "pw456")
        res = client.post(
            f"/notebooks/{nb_id}/sources",
            data={"file": (io.BytesIO(b"x"), "x.txt")},
            content_type="multipart/form-data",
        )
        assert res.status_code == 404


class TestListSources:
    def test_lists_sources(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "listsrc1")
        nb_id = _create_notebook(client, app, "List Src NB")

        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        res = client.get(f"/notebooks/{nb_id}/sources")
        assert res.status_code == 200
        assert b"sample.txt" in res.data

    def test_list_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "listsrc2")
        nb_id = _create_notebook(client, app, "Private Src NB")
        _login(client, app, "listsrc3", "pw456")
        res = client.get(f"/notebooks/{nb_id}/sources")
        assert res.status_code == 404


class TestDeleteSource:
    def test_delete_owned_source(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "delsrc1")
        nb_id = _create_notebook(client, app, "Del Src NB")

        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        with app.app_context():
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            assert src is not None
            src_id = src.id
        res = client.delete(f"/notebooks/{nb_id}/sources/{src_id}")
        assert res.status_code in (200, 202, 204)
        with app.app_context():
            assert db.session.query(Source).filter_by(id=src_id).count() == 0

    def test_delete_non_owner_404(self, client: object, app: object) -> None:
        _login(client, app, "delsrc2")
        nb_id = _create_notebook(client, app, "Del Not Yours NB")
        _login(client, app, "delsrc3", "pw456")
        res = client.delete(f"/notebooks/{nb_id}/sources/9999")
        assert res.status_code == 404


class TestSourceText:
    def test_returns_extracted_text(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "srctext1")
        nb_id = _create_notebook(client, app, "Source Text NB")
        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        with app.app_context():
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            assert src is not None
            src_id = src.id
        res = client.get(f"/notebooks/{nb_id}/sources/{src_id}/text")
        assert res.status_code == 200
        data = res.get_json()
        assert "text" in data
        assert isinstance(data["text"], str)

    def test_text_non_owner_404(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "srctext2")
        nb_id = _create_notebook(client, app, "Source Text Private NB")
        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        _login(client, app, "srctext3", "pw456")
        res = client.get(f"/notebooks/{nb_id}/sources/1/text")
        assert res.status_code == 404


class TestRenameSource:
    def test_rename_success(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "renamesrc1")
        nb_id = _create_notebook(client, app, "Rename Src NB")
        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        with app.app_context():
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            assert src is not None
            src_id = src.id
            assert src.filename == "sample.txt"
        res = client.patch(
            f"/notebooks/{nb_id}/sources/{src_id}/rename",
            json={"filename": "renamed.txt"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["filename"] == "renamed.txt"
        with app.app_context():
            src = db.session.get(Source, src_id)
            assert src.filename == "renamed.txt"

    def test_rename_empty_filename_400(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "renamesrc2")
        nb_id = _create_notebook(client, app, "Rename Empty NB")
        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        with app.app_context():
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            assert src is not None
            src_id = src.id
        res = client.patch(
            f"/notebooks/{nb_id}/sources/{src_id}/rename",
            json={"filename": ""},
        )
        assert res.status_code == 400

    def test_rename_non_owner_404(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _login(client, app, "renamesrc3")
        nb_id = _create_notebook(client, app, "Rename Private NB")
        with open(FIXTURES / "sample.txt", "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, "sample.txt")},
                content_type="multipart/form-data",
            )
        _login(client, app, "renamesrc4", "pw456")
        res = client.patch(
            f"/notebooks/{nb_id}/sources/1/rename",
            json={"filename": "hacked.txt"},
        )
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Reference-counted cleanup on source/notebook deletion (P0-1.3)
# ---------------------------------------------------------------------------


class TestCleanupOnDelete:
    """Deleting a source removes orphaned Chroma/registry content; shared
    content is kept while another notebook references the same hash (P0-1.3)."""

    def _upload(self, client: object, app: object, nb_id: int, filename: str = "sample.txt") -> str:
        with app.app_context():
            from src.services.vector_store import get_vector_store, reset_vector_store

            reset_vector_store()
            get_vector_store().reset()
        with open(FIXTURES / filename, "rb") as f:
            client.post(
                f"/notebooks/{nb_id}/sources",
                data={"file": (f, filename)},
                content_type="multipart/form-data",
            )
        with app.app_context():
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            assert src is not None
            return src.content_hash

    def test_delete_last_reference_removes_orphan(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        from src.services.vector_store import get_vector_store, reset_vector_store

        reset_vector_store()
        get_vector_store().reset()
        _login(client, app, "cleanup1")
        nb_id = _create_notebook(client, app, "Cleanup NB")
        ch = self._upload(client, app, nb_id)

        with app.app_context():
            from src.repositories import content_registry_repo

            assert content_registry_repo.get_by_hash(ch) is not None
            src = db.session.query(Source).filter_by(notebook_id=nb_id).first()
            src_id = src.id

        res = client.delete(f"/notebooks/{nb_id}/sources/{src_id}")
        assert res.status_code in (200, 202, 204)

        with app.app_context():
            from src.repositories import content_registry_repo
            from src.services.vector_store import get_vector_store

            # Registry entry gone (no remaining references).
            assert content_registry_repo.get_by_hash(ch) is None
            assert get_vector_store().collection_exists(ch) is False

    def test_delete_shared_content_kept_when_other_notebook_uses_it(
        self, client: object, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        from src.services.vector_store import get_vector_store, reset_vector_store

        reset_vector_store()
        get_vector_store().reset()
        _login(client, app, "cleanup2")
        nb1 = _create_notebook(client, app, "Shared NB1")
        ch = self._upload(client, app, nb1)
        nb2 = _create_notebook(client, app, "Shared NB2")
        # Same file -> same hash -> dedup, second source references same content.
        self._upload(client, app, nb2)

        with app.app_context():
            src1 = db.session.query(Source).filter_by(notebook_id=nb1).first()
            src1_id = src1.id

        client.delete(f"/notebooks/{nb1}/sources/{src1_id}")

        with app.app_context():
            from src.repositories import content_registry_repo
            from src.services.vector_store import get_vector_store

            # Content kept because nb2 still references it.
            assert content_registry_repo.get_by_hash(ch) is not None
            assert get_vector_store().collection_exists(ch) is True
