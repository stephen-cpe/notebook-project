"""Unit tests for src.services.cleanup_service (P0-1.3)."""

from __future__ import annotations

from pathlib import Path

from src.services.cleanup_service import (
    cleanup_notebook_media,
    cleanup_notebook_orphaned_content,
    cleanup_orphaned_content,
)


class TestCleanupNotebookMedia:
    def test_removes_audio_video_voice(self, tmp_path: Path) -> None:
        nb_id = 7
        for sub in ("audio", "video", "voice"):
            d = tmp_path / sub / str(nb_id)
            d.mkdir(parents=True)
            (d / "file.bin").write_bytes(b"x")
        cleanup_notebook_media(nb_id, str(tmp_path))
        assert not (tmp_path / "audio" / str(nb_id)).exists()
        assert not (tmp_path / "video" / str(nb_id)).exists()
        assert not (tmp_path / "voice" / str(nb_id)).exists()

    def test_missing_dir_is_noop(self, tmp_path: Path) -> None:
        cleanup_notebook_media(999, str(tmp_path))  # should not raise


class TestCleanupOrphanedContent:
    def test_returns_false_when_shared(self, app: object) -> None:
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, Source, User

            u = User(username="cleanu", password_hash="h")
            db.session.add(u)
            db.session.commit()
            nb1 = Notebook(user_id=u.id, name="nb1")
            nb2 = Notebook(user_id=u.id, name="nb2")
            db.session.add_all([nb1, nb2])
            db.session.commit()
            db.session.add_all(
                [
                    Source(
                        notebook_id=nb1.id,
                        filename="a",
                        content_hash="shared_h",
                        content_type="txt",
                    ),
                    Source(
                        notebook_id=nb2.id,
                        filename="b",
                        content_hash="shared_h",
                        content_type="txt",
                    ),
                ]
            )
            db.session.commit()
            # Two references -> not orphaned.
            removed = cleanup_orphaned_content("shared_h")
        assert removed is False

    def test_returns_true_when_no_references(self, app: object) -> None:
        with app.app_context():
            from src.repositories import content_registry_repo

            content_registry_repo.create_entry("solo_h", "doc_solo", "text", 4)
            removed = cleanup_orphaned_content("solo_h")
        assert removed is True

    def test_cleanup_notebook_orphaned_content_iterates(self, app: object) -> None:
        with app.app_context():
            from src.repositories import content_registry_repo

            content_registry_repo.create_entry("list_h", "doc_list", "text", 4)
            cleanup_notebook_orphaned_content(["list_h"])
            assert content_registry_repo.get_by_hash("list_h") is None
