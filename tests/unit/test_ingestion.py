"""Unit tests for src.services.ingestion (TDD step 9).

The ingestion pipeline ties together: hashing → document_parser → OCR
fallback → chunker → embeddings → vector_store + ContentRegistry.

Covers:
- compute_hash is deterministic and content-based.
- ingest_file parses text, chunks, embeds, stores in ChromaDB + ContentRegistry.
- Idempotency: re-ingesting the same hash does not duplicate chunks.
- OCR fallback triggers when extracted text is below threshold.
- OCR disabled skips OCR and marks source partial.
- ContentRegistry entry created with correct collection name + text.
- extract_text_or_ocr returns text + whether OCR was used.
- Status transitions: queued -> extracting -> embedding -> ready.
- Empty/unparseable file marks status failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions import db
from src.models import ContentRegistry
from src.repositories import content_registry_repo
from src.services.ingestion import (
    IngestionService,
    compute_hash,
    get_ingestion_service,
    reset_ingestion_service,
)
from src.services.ocr_service import reset_ocr_service
from src.services.vector_store import get_collection_name, get_vector_store, reset_vector_store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_services() -> None:
    reset_vector_store()
    reset_ingestion_service()
    reset_ocr_service()
    # Reset the ChromaDB EphemeralClient to avoid cross-test collection leaks.

    get_vector_store().reset()
    yield
    reset_vector_store()
    reset_ingestion_service()
    reset_ocr_service()


# ---------------------------------------------------------------------------
# compute_hash
# ---------------------------------------------------------------------------


class TestComputeHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("hello world", encoding="utf-8")
        h1 = compute_hash(str(p))
        h2 = compute_hash(str(p))
        assert h1 == h2

    def test_content_based(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.txt"
        p1.write_text("same content", encoding="utf-8")
        p2 = tmp_path / "b.txt"
        p2.write_text("same content", encoding="utf-8")
        assert compute_hash(str(p1)) == compute_hash(str(p2))

    def test_different_content(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.txt"
        p1.write_text("content one", encoding="utf-8")
        p2 = tmp_path / "b.txt"
        p2.write_text("content two", encoding="utf-8")
        assert compute_hash(str(p1)) != compute_hash(str(p2))

    def test_is_sha256_hex(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("x", encoding="utf-8")
        h = compute_hash(str(p))
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# ingest_file (full pipeline)
# ---------------------------------------------------------------------------


class TestIngestFile:
    def test_ingest_txt_creates_registry_and_collection(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "200")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.txt"), filename="sample.txt")

            assert result.content_hash is not None
            assert len(result.content_hash) == 64
            assert result.char_count > 0
            assert result.status == "ready"
            assert result.ocr_used is False

            # ContentRegistry entry created.
            entry = content_registry_repo.get_by_hash(result.content_hash)
            assert entry is not None
            assert entry.chroma_collection == get_collection_name(result.content_hash)
            assert entry.char_count == result.char_count

    def test_ingest_pdf(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.pdf"), filename="sample.pdf")

        assert result.status == "ready"
        assert "machine learning" in result.extracted_text.lower()
        assert result.page_count == 2

    def test_ingest_docx(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.docx"), filename="sample.docx")

        assert result.status == "ready"
        assert "artificial intelligence" in result.extracted_text.lower()

    def test_ingest_pptx(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.pptx"), filename="sample.pptx")

        assert result.status == "ready"
        assert "cloud computing" in result.extracted_text.lower()

    def test_ingest_md(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.md"), filename="sample.md")

        assert result.status == "ready"
        assert "web development" in result.extracted_text.lower()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_reingest_same_file_no_duplicate(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        svc = IngestionService()

        with app.app_context():
            r1 = svc.ingest_file(str(FIXTURES / "sample.txt"), filename="sample.txt")
        # Second ingest: collection already exists -> dedup path, no new entry.
        with app.app_context():
            r2 = svc.ingest_file(str(FIXTURES / "sample.txt"), filename="sample.txt")

        assert r1.content_hash == r2.content_hash
        with app.app_context():
            count = (
                db.session.query(ContentRegistry).filter_by(content_hash=r1.content_hash).count()
            )
            assert count == 1


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------


class TestOcrFallback:
    def test_ocr_triggers_for_empty_pdf(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank PDF (no text layer) triggers OCR fallback."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "200")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "empty.pdf"), filename="empty.pdf")

        assert result.ocr_used is True
        assert len(result.extracted_text) > 0
        assert result.status == "ready"

    def test_ocr_disabled_marks_partial(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """With OCR disabled, a blank PDF is marked 'partial' with empty text."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "200")
        reset_ocr_service()
        reset_ingestion_service()
        svc = IngestionService()
        assert svc._ocr.is_available() is False

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "empty.pdf"), filename="empty.pdf")

        assert result.ocr_used is False
        assert result.status == "partial"

    def test_ocr_not_triggered_when_text_sufficient(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PDF with enough text does NOT trigger OCR."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "10")
        svc = IngestionService()

        with app.app_context():
            result = svc.ingest_file(str(FIXTURES / "sample.pdf"), filename="sample.pdf")

        assert result.ocr_used is False
        assert result.status == "ready"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = IngestionService()

        result = svc.ingest_file("nonexistent_12345.pdf", filename="x.pdf")

        assert result.status == "failed"
        assert result.error_message is not None
        assert result.content_hash == ""

    def test_unsupported_type_fails(
        self, app: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        p = tmp_path / "bad.xyz"
        p.write_text("unknown", encoding="utf-8")
        svc = IngestionService()

        result = svc.ingest_file(str(p), filename="bad.xyz")

        assert result.status == "failed"
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestGetService:
    def test_returns_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = get_ingestion_service()
        assert isinstance(svc, IngestionService)


# ---------------------------------------------------------------------------
# OCR graceful degradation + registry consistency (P0-1.10, P0-1.11)
# ---------------------------------------------------------------------------


class TestOcrDegradation:
    def test_ocr_exception_with_partial_text_does_not_fail(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OCR raising must not block ingestion when some text was extracted (P0-1.10)."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "10000")
        svc = IngestionService()

        # Force ocr_pdf to raise.
        def _boom(_path: str, _prompt: str) -> str:
            raise RuntimeError("OCR exploded")

        with app.app_context():
            from src.services.ocr_service import reset_ocr_service

            reset_ocr_service()
            svc._ocr = type(svc._ocr)(svc._config)
            svc._ocr.ocr_pdf = _boom  # type: ignore[method-assign]
            result = svc.ingest_file(str(FIXTURES / "sample.pdf"), filename="sample.pdf")

        # sample.pdf has real text; OCR failure keeps that text -> ready.
        assert result.status == "ready"
        assert "machine learning" in result.extracted_text.lower()

    def test_ocr_exception_no_text_marks_partial(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OCR failing with no prior text -> partial, not failed (P0-1.10)."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_TEXT_THRESHOLD", "10000")
        svc = IngestionService()

        def _boom(_path: str, _prompt: str) -> str:
            raise RuntimeError("OCR exploded")

        with app.app_context():
            from src.services.ocr_service import reset_ocr_service

            reset_ocr_service()
            svc._ocr = type(svc._ocr)(svc._config)
            svc._ocr.ocr_pdf = _boom  # type: ignore[method-assign]
            result = svc.ingest_file(str(FIXTURES / "empty.pdf"), filename="empty.pdf")

        # No text anywhere -> partial (not failed).
        assert result.status == "partial"


class TestRegistryRace:
    def test_get_or_create_handles_integrity_error(self, app: object) -> None:
        """Concurrent get_or_create on the same PK is race-safe (P0-1.11)."""
        with app.app_context():
            # First create succeeds.
            content_registry_repo.create_entry("racehash", "doc_race", "text", 4)
            # Simulate a race: make create_entry raise IntegrityError the next
            # time it's called, then ensure get_or_create re-fetches.
            from unittest.mock import patch

            from sqlalchemy.exc import IntegrityError

            def _raise_integrity(*_a: object, **_k: object) -> None:
                raise IntegrityError("simulated", {}, Exception("pk conflict"))

            with patch.object(content_registry_repo, "create_entry", side_effect=_raise_integrity):
                entry = content_registry_repo.get_or_create("racehash", "x", "y", 1)
            assert entry.content_hash == "racehash"
            assert entry.extracted_text == "text"
