"""Unit tests for src.services.rag_retriever (TDD step 10).

The RAG retriever orchestrates multi-collection retrieval with corruption
recovery + source provenance. It builds on ``VectorStore`` and the
``ContentRegistry``.

Covers:
- retrieve_with_sources: queries all of a notebook's source hashes, merges
  by score, returns top_k with {text, filename, page, chunk_index, score}.
- retrieve: same but returns joined context text (for prompt building).
- Empty source list -> empty results / "".
- Single source -> works like basic retrieval.
- Corruption recovery: a broken collection is deleted + rebuilt from
  ContentRegistry cached text, then re-queried.
- Source hash -> filename resolution via Source rows.
- build_context_string: formats retrieved chunks with [Source: filename, p. N].
- format_sources: deduplicates sources by filename+page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.rag_retriever import (
    RAGRetriever,
    build_context_string,
    format_sources,
    get_rag_retriever,
    reset_rag_retriever,
)
from src.services.vector_store import (
    get_collection_name,
    get_vector_store,
    reset_vector_store,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_services() -> None:
    reset_vector_store()
    reset_rag_retriever()
    from src.services.ingestion import reset_ingestion_service
    from src.services.ocr_service import reset_ocr_service

    reset_ingestion_service()
    reset_ocr_service()
    # Reset ChromaDB EphemeralClient.
    get_vector_store().reset()
    yield
    reset_vector_store()
    reset_rag_retriever()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_two_sources(app: object) -> tuple[list[str], list[str]]:
    """Ingest two fixture files; return (content_hashes, filenames)."""
    from src.services.ingestion import IngestionService

    with app.app_context():
        svc = IngestionService()
        r1 = svc.ingest_file(str(FIXTURES / "sample.txt"), filename="sample.txt")
        r2 = svc.ingest_file(str(FIXTURES / "sample.md"), filename="sample.md")
    return [r1.content_hash, r2.content_hash], [r1.extracted_text, r2.extracted_text]


# ---------------------------------------------------------------------------
# retrieve_with_sources
# ---------------------------------------------------------------------------


class TestRetrieveWithSources:
    def test_returns_results_for_single_source(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        hashes, _ = _ingest_two_sources(app)
        retriever = RAGRetriever()

        results = retriever.retrieve_with_sources([hashes[0]], "databases", top_k=3)

        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "filename" in r
            assert "page" in r
            assert "chunk_index" in r
            assert "score" in r

    def test_multi_source_merges_by_score(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        hashes, _ = _ingest_two_sources(app)
        retriever = RAGRetriever()

        results = retriever.retrieve_with_sources(hashes, "databases web", top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_source_list_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        retriever = RAGRetriever()
        assert retriever.retrieve_with_sources([], "query", top_k=5) == []

    def test_nonexistent_hash_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        retriever = RAGRetriever()
        results = retriever.retrieve_with_sources(["a" * 64], "query", top_k=5)
        assert results == []

    def test_truncates_to_top_k(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        hashes, _ = _ingest_two_sources(app)
        retriever = RAGRetriever()

        results = retriever.retrieve_with_sources(hashes, "content", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# retrieve (joined text)
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_returns_joined_text(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        hashes, _ = _ingest_two_sources(app)
        retriever = RAGRetriever()

        text = retriever.retrieve(hashes, "databases", top_k=3)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_sources_returns_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        retriever = RAGRetriever()
        assert retriever.retrieve([], "query", top_k=5) == ""


# ---------------------------------------------------------------------------
# Corruption recovery
# ---------------------------------------------------------------------------


class TestCorruptionRecovery:
    def test_broken_collection_rebuilt_from_registry(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        hashes, texts = _ingest_two_sources(app)
        target_hash = hashes[0]
        cached_text = texts[0]

        # Simulate corruption: delete the collection directly.
        retriever = RAGRetriever()
        retriever._vector_store.delete_collection(target_hash)
        assert retriever._vector_store.collection_exists(target_hash) is False

        # Create a ContentRegistry entry with cached text (simulating prior ingest).
        from src.repositories import content_registry_repo

        with app.app_context():
            content_registry_repo.get_or_create(
                content_hash=target_hash,
                chroma_collection=get_collection_name(target_hash),
                extracted_text=cached_text,
                char_count=len(cached_text),
            )

        # retrieve_with_sources should detect the missing collection,
        # rebuild from registry text, and return results.
        with app.app_context():
            results = retriever.retrieve_with_sources([target_hash], "databases", top_k=3)

        assert len(results) >= 1
        assert retriever._vector_store.collection_exists(target_hash) is True


# ---------------------------------------------------------------------------
# build_context_string
# ---------------------------------------------------------------------------


class TestBuildContextString:
    def test_formats_chunks_with_source_labels(self) -> None:
        results = [
            {
                "text": "The sky is blue.",
                "filename": "doc1.pdf",
                "page": 1,
                "chunk_index": 0,
                "score": 0.9,
            },
            {
                "text": "Grass is green.",
                "filename": "doc2.pdf",
                "page": 3,
                "chunk_index": 2,
                "score": 0.8,
            },
        ]
        context = build_context_string(results)
        assert "doc1.pdf" in context
        assert "The sky is blue." in context
        assert "doc2.pdf" in context
        assert "Grass is green." in context
        assert "Page" in context or "p." in context

    def test_empty_results_returns_empty_string(self) -> None:
        assert build_context_string([]) == ""

    def test_handles_missing_page(self) -> None:
        results = [
            {
                "text": "no page info",
                "filename": "f.txt",
                "page": None,
                "chunk_index": 0,
                "score": 0.5,
            }
        ]
        context = build_context_string(results)
        assert "no page info" in context
        assert "f.txt" in context


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------


class TestFormatSources:
    def test_deduplicates_by_filename_page(self) -> None:
        results = [
            {"filename": "a.pdf", "page": 1, "chunk_index": 0, "score": 0.9, "text": "x"},
            {"filename": "a.pdf", "page": 1, "chunk_index": 1, "score": 0.8, "text": "y"},
            {"filename": "b.pdf", "page": 2, "chunk_index": 0, "score": 0.7, "text": "z"},
        ]
        sources = format_sources(results)
        # a.pdf p.1 appears twice but should be deduplicated.
        assert len(sources) == 2
        filenames_pages = {s["filename"] for s in sources}
        assert "a.pdf" in filenames_pages
        assert "b.pdf" in filenames_pages

    def test_empty_results_returns_empty(self) -> None:
        assert format_sources([]) == []

    def test_handles_missing_page(self) -> None:
        results = [
            {"filename": "f.txt", "page": None, "chunk_index": 0, "score": 0.5, "text": "x"},
        ]
        sources = format_sources(results)
        assert len(sources) == 1
        assert sources[0]["filename"] == "f.txt"


# ---------------------------------------------------------------------------
# get_rag_retriever singleton
# ---------------------------------------------------------------------------


class TestGetRetriever:
    def test_returns_retriever(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        r = get_rag_retriever()
        assert isinstance(r, RAGRetriever)
