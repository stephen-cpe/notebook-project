"""Unit tests for src.services.context_builder.

Verifies the shared source-selection helper used by the summary, audio, and
video overview generators: deterministic upload-order selection, character
budget enforcement, oversized-first-source inclusion, status filtering, and
graceful handling of missing ContentRegistry text.
"""

from __future__ import annotations

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.auth_service import hash_password
from src.services.context_builder import SourceSelection, select_sources_within_budget


def _make_user_and_notebook(app: object, username: str) -> tuple[int, int]:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="CB NB")
        db.session.add(nb)
        db.session.commit()
        return u.id, nb.id


def _add_source(
    app: object,
    nb_id: int,
    filename: str,
    content_hash: str,
    text: str,
    status: str = "ready",
) -> None:
    with app.app_context():
        db.session.add(
            Source(
                notebook_id=nb_id,
                filename=filename,
                content_hash=content_hash,
                content_type="txt",
                status=status,
            )
        )
        db.session.commit()
        content_registry_repo.get_or_create(
            content_hash=content_hash,
            chroma_collection=f"doc_{content_hash[:6]}",
            extracted_text=text,
            char_count=len(text),
        )


class TestSelectSourcesWithinBudget:
    def test_empty_notebook(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb1")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == []
        assert sel.total_chars == 0
        assert sel.used_count == 0
        assert sel.total_count == 0

    def test_single_source_under_budget(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb2")
        _add_source(app, nb_id, "a.txt", "a" * 64, "hello world")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == ["hello world"]
        assert sel.total_chars == 11
        assert sel.used_count == 1
        assert sel.total_count == 1

    def test_includes_all_when_under_budget(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb3")
        _add_source(app, nb_id, "a.txt", "a" * 64, "aaaa")
        _add_source(app, nb_id, "b.txt", "b" * 64, "bbbb")
        _add_source(app, nb_id, "c.txt", "c" * 64, "cccc")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == ["aaaa", "bbbb", "cccc"]
        assert sel.used_count == 3
        assert sel.total_count == 3

    def test_truncates_when_over_budget(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb4")
        _add_source(app, nb_id, "a.txt", "a" * 64, "aaaa")  # 4
        _add_source(app, nb_id, "b.txt", "b" * 64, "bbbb")  # 4
        _add_source(app, nb_id, "c.txt", "c" * 64, "cccc")  # 4
        _add_source(app, nb_id, "d.txt", "d" * 64, "dddd")  # 4
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10)
        # 4+4=8 fits, 4+4+4=12 > 10 -> stop at 2 sources.
        assert sel.texts == ["aaaa", "bbbb"]
        assert sel.total_chars == 8
        assert sel.used_count == 2
        assert sel.total_count == 4

    def test_first_source_always_included_even_if_oversized(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb5")
        big = "x" * 1000
        _add_source(app, nb_id, "big.txt", "z" * 64, big)
        _add_source(app, nb_id, "small.txt", "y" * 64, "yy")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=100)
        assert sel.texts == [big]
        assert sel.total_chars == 1000
        assert sel.used_count == 1
        assert sel.total_count == 2

    def test_deterministic_upload_order(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb6")
        _add_source(app, nb_id, "first.txt", "f" * 64, "first")
        _add_source(app, nb_id, "second.txt", "s" * 64, "second")
        _add_source(app, nb_id, "third.txt", "t" * 64, "third")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == ["first", "second", "third"]

    def test_filters_failed_status(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb7")
        _add_source(app, nb_id, "ok.txt", "a" * 64, "ready", status="ready")
        _add_source(app, nb_id, "part.txt", "b" * 64, "partial", status="partial")
        _add_source(app, nb_id, "fail.txt", "c" * 64, "failed", status="failed")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == ["ready", "partial"]
        assert sel.used_count == 2
        assert sel.total_count == 2

    def test_skips_sources_without_registry_text(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb8")
        _add_source(app, nb_id, "with.txt", "a" * 64, "has text")
        with app.app_context():
            db.session.add(
                Source(
                    notebook_id=nb_id,
                    filename="without.txt",
                    content_hash="q" * 64,
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert sel.texts == ["has text"]
        assert sel.used_count == 1
        assert sel.total_count == 1

    def test_returns_source_selection_dataclass(self, app: object) -> None:
        _, nb_id = _make_user_and_notebook(app, "cb9")
        _add_source(app, nb_id, "a.txt", "a" * 64, "hi")
        with app.app_context():
            sel = select_sources_within_budget(nb_id, max_chars=10000)
        assert isinstance(sel, SourceSelection)
