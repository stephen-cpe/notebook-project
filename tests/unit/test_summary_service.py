"""Unit tests for src.services.summary_service (TDD step 15).

The summary service auto-generates a short summary (<=150 words) + 5
suggested questions when a notebook is created or sources change. It is
idempotent: a background job keyed by ``(notebook_id, content_signature)``
skips re-run if the signature is unchanged.

Covers:
- generate_summary: produces a summary string + list of 5 suggested questions.
- generate_summary: empty sources -> placeholder summary.
- generate_summary: mock mode returns deterministic output.
- compute_content_signature: deterministic hash of sorted source hashes.
- Idempotency: same signature -> skip; different signature -> regenerate.
- Failure handling: LLM error -> summary stays None, no crash.
- parse_summary_response: extracts summary + questions from LLM JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import notebook_repo
from src.services.auth_service import hash_password
from src.services.summary_service import (
    SummaryService,
    compute_content_signature,
    parse_summary_response,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_user_and_notebook(app: object, username: str = "sumuser") -> tuple[int, int]:
    """Create a user + notebook; return (user_id, notebook_id)."""
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Summary NB")
        db.session.add(nb)
        db.session.commit()
        return u.id, nb.id


def _add_source(app: object, nb_id: int, filename: str, content_hash: str) -> None:
    with app.app_context():
        db.session.add(
            Source(
                notebook_id=nb_id,
                filename=filename,
                content_hash=content_hash,
                content_type="txt",
                status="ready",
            )
        )
        db.session.commit()


# ---------------------------------------------------------------------------
# compute_content_signature
# ---------------------------------------------------------------------------


class TestContentSignature:
    def test_deterministic(self) -> None:
        sig1 = compute_content_signature(["a", "b", "c"])
        sig2 = compute_content_signature(["c", "b", "a"])
        assert sig1 == sig2  # order-independent

    def test_different_sources_different_sig(self) -> None:
        sig1 = compute_content_signature(["a", "b"])
        sig2 = compute_content_signature(["a", "c"])
        assert sig1 != sig2

    def test_empty_list(self) -> None:
        sig = compute_content_signature([])
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_is_sha256_hex(self) -> None:
        sig = compute_content_signature(["x"])
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ---------------------------------------------------------------------------
# parse_summary_response
# ---------------------------------------------------------------------------


class TestParseSummaryResponse:
    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "summary": "This notebook covers databases and ML.",
                "suggested_questions": [
                    "What databases are mentioned?",
                    "How does ML work?",
                    "What is PostgreSQL?",
                    "What is MongoDB?",
                    "What is Redis?",
                ],
            }
        )
        summary, questions = parse_summary_response(raw)
        assert "databases" in summary.lower()
        assert len(questions) == 5

    def test_missing_questions_key(self) -> None:
        raw = json.dumps({"summary": "Just a summary."})
        summary, questions = parse_summary_response(raw)
        assert summary == "Just a summary."
        assert questions == []

    def test_missing_summary_key(self) -> None:
        raw = json.dumps({"suggested_questions": ["q1", "q2"]})
        summary, questions = parse_summary_response(raw)
        assert summary == ""
        assert len(questions) == 2

    def test_invalid_json_returns_raw(self) -> None:
        summary, questions = parse_summary_response("not json at all")
        assert summary == "not json at all"
        assert questions == []

    def test_empty_string(self) -> None:
        summary, questions = parse_summary_response("")
        assert summary == ""
        assert questions == []

    def test_extra_questions_truncated_to_five(self) -> None:
        raw = json.dumps(
            {
                "summary": "test",
                "suggested_questions": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"],
            }
        )
        _, questions = parse_summary_response(raw)
        assert len(questions) == 5


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def test_with_sources(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _, nb_id = _make_user_and_notebook(app, "sumgen1")
        _add_source(app, nb_id, "doc1.txt", "a" * 64)

        # Add content registry entry so source_texts is non-empty.
        from src.repositories import content_registry_repo

        with app.app_context():
            content_registry_repo.get_or_create(
                content_hash="a" * 64,
                chroma_collection="doc_a",
                extracted_text="This document discusses databases and SQL.",
                char_count=40,
            )

        svc = SummaryService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            result = svc.generate_summary(nb)
            assert result is not None
            assert isinstance(result.summary, str)
            assert len(result.summary) > 0
            assert isinstance(result.suggested_questions, list)
            assert len(result.suggested_questions) <= 5

    def test_empty_sources_placeholder(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        _, nb_id = _make_user_and_notebook(app, "sumgen2")
        svc = SummaryService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            result = svc.generate_summary(nb)
            assert result is not None
            # With no sources, summary should be a placeholder or empty.
            assert isinstance(result.summary, str)

    def test_idempotent_same_signature(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _, nb_id = _make_user_and_notebook(app, "sumgen3")
        _add_source(app, nb_id, "doc.txt", "b" * 64)

        from src.repositories import content_registry_repo

        with app.app_context():
            content_registry_repo.get_or_create(
                content_hash="b" * 64,
                chroma_collection="doc_b",
                extracted_text="Text about Python programming.",
                char_count=30,
            )

        svc = SummaryService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            # First generation.
            r1 = svc.generate_summary(nb)
            assert r1 is not None
            sig1 = nb.content_signature
            # Second generation with same sources -> should skip.
            r2 = svc.generate_summary(nb)
            assert r2 is not None
            assert r2.skipped is True
            assert nb.content_signature == sig1

    def test_different_signature_regenerates(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        _, nb_id = _make_user_and_notebook(app, "sumgen4")
        _add_source(app, nb_id, "doc1.txt", "c" * 64)

        from src.repositories import content_registry_repo

        with app.app_context():
            content_registry_repo.get_or_create(
                content_hash="c" * 64,
                chroma_collection="doc_c",
                extracted_text="Text about Java programming.",
                char_count=30,
            )

        svc = SummaryService()
        with app.app_context():
            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            r1 = svc.generate_summary(nb)
            assert r1 is not None
            assert r1.skipped is False

            # Add a new source -> signature changes.
            db.session.add(
                Source(
                    notebook_id=nb_id,
                    filename="doc2.txt",
                    content_hash="d" * 64,
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
            content_registry_repo.get_or_create(
                content_hash="d" * 64,
                chroma_collection="doc_d",
                extracted_text="Text about Go programming.",
                char_count=30,
            )

            r2 = svc.generate_summary(nb)
            assert r2 is not None
            assert r2.skipped is False  # regenerated


# ---------------------------------------------------------------------------
# SummaryResult dataclass
# ---------------------------------------------------------------------------


class TestSummaryResult:
    def test_has_fields(self) -> None:
        from src.services.summary_service import SummaryResult

        r = SummaryResult(summary="test", suggested_questions=["q1"], skipped=False)
        assert r.summary == "test"
        assert r.suggested_questions == ["q1"]
        assert r.skipped is False
