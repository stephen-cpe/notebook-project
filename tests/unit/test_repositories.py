"""Unit tests for src.repositories (TDD step 4).

Covers CRUD operations and owner scoping for all repository modules.
"""

from __future__ import annotations

from src.repositories import (
    chat_repo,
    content_registry_repo,
    notebook_repo,
    source_repo,
    user_repo,
)


class TestUserRepo:
    def test_create_and_get(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("repouser1", "hash", "user")
            assert u.id is not None
            fetched = user_repo.get_by_id(u.id)
            assert fetched is not None
            assert fetched.username == "repouser1"

    def test_get_by_username(self, app: object) -> None:
        with app.app_context():
            user_repo.create_user("repouser2", "hash")
            u = user_repo.get_by_username("repouser2")
            assert u is not None
            assert u.username == "repouser2"

    def test_get_by_username_missing(self, app: object) -> None:
        with app.app_context():
            assert user_repo.get_by_username("nobody") is None

    def test_list_all(self, app: object) -> None:
        with app.app_context():
            user_repo.create_user("repouser3", "hash")
            user_repo.create_user("repouser4", "hash")
            users = user_repo.list_all()
            assert len(users) >= 2


class TestNotebookRepo:
    def test_create_and_get(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("nbrepo1", "hash")
            nb = notebook_repo.create_notebook(u.id, "Test NB", "desc")
            assert nb.id is not None
            fetched = notebook_repo.get_by_id(nb.id)
            assert fetched is not None
            assert fetched.name == "Test NB"

    def test_list_by_user(self, app: object) -> None:
        with app.app_context():
            u1 = user_repo.create_user("nbrepo2", "hash")
            u2 = user_repo.create_user("nbrepo3", "hash")
            notebook_repo.create_notebook(u1.id, "NB1")
            notebook_repo.create_notebook(u1.id, "NB2")
            notebook_repo.create_notebook(u2.id, "NB3")
            nbs = notebook_repo.list_by_user(u1.id)
            assert len(nbs) == 2
            assert all(nb.user_id == u1.id for nb in nbs)

    def test_update(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("nbrepo4", "hash")
            nb = notebook_repo.create_notebook(u.id, "Old")
            updated = notebook_repo.update_notebook(nb, name="New", description="Updated")
            assert updated.name == "New"
            assert updated.description == "Updated"

    def test_delete(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("nbrepo5", "hash")
            nb = notebook_repo.create_notebook(u.id, "ToDelete")
            nb_id = nb.id
            notebook_repo.delete_notebook(nb)
            assert notebook_repo.get_by_id(nb_id) is None


class TestSourceRepo:
    def test_create_and_get(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo1", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            s = source_repo.create_source(nb.id, "f.txt", "abc123", "txt")
            assert s.id is not None
            assert s.status == "queued"

    def test_list_by_notebook(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo2", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            source_repo.create_source(nb.id, "a.txt", "h1", "txt")
            source_repo.create_source(nb.id, "b.txt", "h2", "txt")
            sources = source_repo.list_by_notebook(nb.id)
            assert len(sources) == 2

    def test_update_status(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo3", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            s = source_repo.create_source(nb.id, "f.txt", "h", "txt")
            updated = source_repo.update_status(s, "ready", char_count=100, page_count=2)
            assert updated.status == "ready"
            assert updated.char_count == 100
            assert updated.page_count == 2

    def test_delete(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo4", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            s = source_repo.create_source(nb.id, "f.txt", "h", "txt")
            sid = s.id
            source_repo.delete_source(s)
            assert source_repo.get_by_id(sid) is None

    def test_count_by_notebook(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo5", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            assert source_repo.count_by_notebook(nb.id) == 0
            source_repo.create_source(nb.id, "a.txt", "h1", "txt")
            assert source_repo.count_by_notebook(nb.id) == 1

    def test_get_by_notebook_and_hash(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("srcrepo6", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            source_repo.create_source(nb.id, "f.txt", "myhash", "txt")
            s = source_repo.get_by_notebook_and_hash(nb.id, "myhash")
            assert s is not None
            assert s.filename == "f.txt"


class TestChatRepo:
    def test_create_and_list(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("chatrepo1", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            chat_repo.create_message(nb.id, "user", "hello")
            chat_repo.create_message(
                nb.id, "assistant", "hi there", sources_json='[{"a":1}]', latency_ms=500
            )
            msgs = chat_repo.list_by_notebook(nb.id)
            assert len(msgs) == 2
            assert msgs[0].role == "user"
            assert msgs[1].role == "assistant"
            assert msgs[1].latency_ms == 500

    def test_delete_by_notebook(self, app: object) -> None:
        with app.app_context():
            u = user_repo.create_user("chatrepo2", "hash")
            nb = notebook_repo.create_notebook(u.id, "NB")
            chat_repo.create_message(nb.id, "user", "msg")
            count = chat_repo.delete_by_notebook(nb.id)
            assert count == 1
            assert chat_repo.list_by_notebook(nb.id) == []


class TestContentRegistryRepo:
    def test_create_and_get(self, app: object) -> None:
        with app.app_context():
            entry = content_registry_repo.create_entry("hash1", "doc_hash1", "text", 4)
            fetched = content_registry_repo.get_by_hash("hash1")
            assert fetched is not None
            assert fetched.extracted_text == "text"

    def test_get_or_create_new(self, app: object) -> None:
        with app.app_context():
            entry = content_registry_repo.get_or_create("hash2", "doc_hash2", "new text", 8)
            assert entry.content_hash == "hash2"
            assert entry.extracted_text == "new text"

    def test_get_or_create_existing(self, app: object) -> None:
        with app.app_context():
            content_registry_repo.create_entry("hash3", "doc_hash3", "original", 8)
            entry = content_registry_repo.get_or_create("hash3", "doc_hash3", "ignored", 0)
            assert entry.extracted_text == "original"

    def test_delete_entry(self, app: object) -> None:
        with app.app_context():
            content_registry_repo.create_entry("hash4", "doc_hash4", "text", 4)
            content_registry_repo.delete_entry("hash4")
            assert content_registry_repo.get_by_hash("hash4") is None

    def test_delete_missing_no_error(self, app: object) -> None:
        with app.app_context():
            content_registry_repo.delete_entry("nonexistent")
