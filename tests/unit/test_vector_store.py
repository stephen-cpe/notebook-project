"""Unit tests for src.services.vector_store (TDD step 6).

Covers:
- Backend selection: CI -> EphemeralClient, CHROMA_DB=local -> PersistentClient.
- Collection naming: content-keyed `doc_<hash[:59]>`.
- store_chunks with metadata (source_hash, filename, page, chunk_index).
- retrieve (basic similarity search) returns joined text.
- retrieve_with_scores returns score = 1 - distance.
- retrieve_from_multiple_collections merges + sorts by score + truncates.
- retrieve_from_multiple_collections_with_sources preserves provenance.
- Corruption recovery: broken collection is deleted + rebuilt.
- Mock embeddings (offline, deterministic).
"""

from __future__ import annotations

import pytest

from src.services.vector_store import (
    VectorStore,
    get_collection_name,
    get_vector_store,
    reset_vector_store,
)


@pytest.fixture(autouse=True)
def _reset_chroma() -> None:
    reset_vector_store()
    # Reset the ChromaDB EphemeralClient to avoid cross-test collection leaks.
    get_vector_store().reset()
    yield
    reset_vector_store()


# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------


class TestCollectionNaming:
    def test_naming_format(self) -> None:
        h = "a" * 64
        name = get_collection_name(h)
        assert name.startswith("doc_")
        assert len(name) == 4 + 59  # "doc_" + 59 chars

    def test_different_hashes_different_collections(self) -> None:
        h1 = "a" * 64
        h2 = "b" * 64
        assert get_collection_name(h1) != get_collection_name(h2)

    def test_short_hash_padded(self) -> None:
        h = "abc123"
        name = get_collection_name(h)
        assert name.startswith("doc_")
        # Should still produce a valid collection name.
        assert len(name) >= 5


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_ci_forces_ephemeral(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("CI", "true")
        vs = VectorStore()
        assert vs.backend == "ephemeral"

    def test_local_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("CHROMA_DB", "local")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        vs = VectorStore()
        assert vs.backend == "local"

    def test_cloud_used_when_credentials_valid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CHROMA_DB=cloud + valid creds + successful heartbeat -> cloud backend."""
        import logging

        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("CHROMA_DB", "cloud")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "ck-valid-key")
        monkeypatch.setenv("CHROMA_CLOUD_CONNECTION_STRING", "tenant-uuid")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test-db")

        import chromadb

        class _FakeCloudClient:
            def heartbeat(self) -> int:
                return 1

        captured: dict[str, str] = {}

        def fake_cloud_client(tenant, database, api_key):
            captured["tenant"] = tenant
            captured["database"] = database
            captured["api_key"] = api_key
            return _FakeCloudClient()

        monkeypatch.setattr(chromadb, "CloudClient", fake_cloud_client)
        caplog.set_level(logging.INFO, logger="src.services.vector_store")
        vs = VectorStore()
        assert vs.backend == "cloud"
        assert captured["tenant"] == "tenant-uuid"
        assert captured["database"] == "test-db"
        assert captured["api_key"] == "ck-valid-key"
        assert any("Chroma Cloud connection established" in r.message for r in caplog.records)

    def test_cloud_falls_back_when_creds_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CHROMA_DB=cloud but one cred empty -> ERROR log + local fallback."""
        import logging

        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("CHROMA_DB", "cloud")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "")  # missing
        monkeypatch.setenv("CHROMA_CLOUD_CONNECTION_STRING", "tenant")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "db")
        caplog.set_level(logging.ERROR, logger="src.services.vector_store")
        vs = VectorStore()
        assert vs.backend == "local"
        assert any("CHROMA_CLOUD_API_KEY" in r.message for r in caplog.records)
        assert any("Reverting to local" in r.message for r in caplog.records)

    def test_cloud_falls_back_when_heartbeat_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CHROMA_DB=cloud + creds set but heartbeat raises -> ERROR + local fallback."""
        import logging

        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("CHROMA_DB", "cloud")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "ck-bad-key")
        monkeypatch.setenv("CHROMA_CLOUD_CONNECTION_STRING", "tenant")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "db")

        import chromadb

        class _BrokenCloudClient:
            def heartbeat(self) -> int:
                raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(chromadb, "CloudClient", lambda **kw: _BrokenCloudClient())
        caplog.set_level(logging.ERROR, logger="src.services.vector_store")
        vs = VectorStore()
        assert vs.backend == "local"
        assert any("Chroma Cloud connection failed" in r.message for r in caplog.records)
        assert any("401 Unauthorized" in r.message for r in caplog.records)

    def test_cloud_fallback_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When cloud fails, a WARNING is logged explaining the fallback."""
        import logging

        monkeypatch.setenv("CI", "false")
        monkeypatch.setenv("CHROMA_DB", "cloud")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "")
        caplog.set_level(logging.WARNING, logger="src.services.vector_store")
        VectorStore()
        assert any("Chroma Cloud unavailable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Store + retrieve
# ---------------------------------------------------------------------------


class TestStoreAndRetrieve:
    def test_store_and_retrieve_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        content_hash = "h" * 64
        chunks = ["The sky is blue.", "Grass is green.", "Fire is hot."]
        metadatas = [
            {"source_hash": content_hash, "filename": "doc.pdf", "page": 1, "chunk_index": i}
            for i in range(3)
        ]
        vs.store_chunks(content_hash, chunks, metadatas)
        results = vs.retrieve(content_hash, "What color is the sky?", top_k=2)
        assert isinstance(results, str)
        assert len(results) > 0

    def test_store_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        content_hash = "a" * 64
        vs.store_chunks(
            content_hash,
            ["chunk content here"],
            [{"source_hash": content_hash, "filename": "f.txt", "page": 5, "chunk_index": 0}],
        )
        results = vs.retrieve_with_scores(content_hash, "chunk", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "document" in r
        assert "score" in r
        assert "metadata" in r
        assert r["metadata"]["filename"] == "f.txt"
        assert r["metadata"]["page"] == 5
        assert r["metadata"]["chunk_index"] == 0

    def test_retrieve_with_scores_sorted_desc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        content_hash = "b" * 64
        chunks = ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel"]
        metadatas = [
            {"source_hash": content_hash, "filename": "d.pdf", "page": 1, "chunk_index": i}
            for i in range(4)
        ]
        vs.store_chunks(content_hash, chunks, metadatas)
        results = vs.retrieve_with_scores(content_hash, "alpha bravo", top_k=4)
        assert len(results) == 4
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_in_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        content_hash = "c" * 64
        vs.store_chunks(
            content_hash,
            ["some text"],
            [{"source_hash": content_hash, "filename": "f", "page": 1, "chunk_index": 0}],
        )
        results = vs.retrieve_with_scores(content_hash, "some text", top_k=1)
        for r in results:
            assert -2.0 <= r["score"] <= 2.0

    def test_retrieve_empty_collection_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        results = vs.retrieve(
            "nonexistent_hash_1234567890123456789012345678901234567890", "q", top_k=3
        )
        assert results == ""


# ---------------------------------------------------------------------------
# Multi-collection retrieval + source provenance
# ---------------------------------------------------------------------------


class TestMultiCollectionRetrieval:
    def test_retrieve_from_multiple_collections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h1 = "1" * 64
        h2 = "2" * 64
        vs.store_chunks(
            h1,
            ["alpha content about cats"],
            [{"source_hash": h1, "filename": "cats.pdf", "page": 1, "chunk_index": 0}],
        )
        vs.store_chunks(
            h2,
            ["beta content about dogs"],
            [{"source_hash": h2, "filename": "dogs.pdf", "page": 1, "chunk_index": 0}],
        )
        results = vs.retrieve_from_multiple_collections([h1, h2], "cats dogs", top_k=5)
        assert isinstance(results, str)
        assert "cats" in results or "dogs" in results

    def test_retrieve_with_sources_preserves_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h1 = "x" * 64
        h2 = "y" * 64
        vs.store_chunks(
            h1,
            ["content about apples"],
            [{"source_hash": h1, "filename": "apples.pdf", "page": 3, "chunk_index": 0}],
        )
        vs.store_chunks(
            h2,
            ["content about oranges"],
            [{"source_hash": h2, "filename": "oranges.pdf", "page": 7, "chunk_index": 1}],
        )
        results = vs.retrieve_from_multiple_collections_with_sources(
            [h1, h2], "apples oranges", top_k=5
        )
        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "filename" in r
            assert "page" in r
            assert "chunk_index" in r
            assert "score" in r
            assert r["filename"] in ("apples.pdf", "oranges.pdf")

    def test_multi_collection_truncates_to_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h1 = "p" * 64
        h2 = "q" * 64
        vs.store_chunks(
            h1,
            ["a", "b", "c"],
            [{"source_hash": h1, "filename": "f1", "page": 1, "chunk_index": i} for i in range(3)],
        )
        vs.store_chunks(
            h2,
            ["d", "e", "f"],
            [{"source_hash": h2, "filename": "f2", "page": 1, "chunk_index": i} for i in range(3)],
        )
        results = vs.retrieve_from_multiple_collections_with_sources(
            [h1, h2], "a b c d e f", top_k=3
        )
        assert len(results) <= 3

    def test_multi_collection_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        results = vs.retrieve_from_multiple_collections_with_sources([], "q", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Collection existence + corruption recovery
# ---------------------------------------------------------------------------


class TestCollectionExistence:
    def test_collection_exists_after_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h = "e" * 64
        vs.store_chunks(
            h, ["text"], [{"source_hash": h, "filename": "f", "page": 1, "chunk_index": 0}]
        )
        assert vs.collection_exists(h) is True

    def test_collection_does_not_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        assert vs.collection_exists("nonexistent" + "z" * 55) is False

    def test_delete_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h = "f" * 64
        vs.store_chunks(
            h, ["text"], [{"source_hash": h, "filename": "f", "page": 1, "chunk_index": 0}]
        )
        assert vs.collection_exists(h) is True
        vs.delete_collection(h)
        assert vs.collection_exists(h) is False


class TestCorruptionRecovery:
    def test_rebuild_from_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken collection can be rebuilt from cached extracted text."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        vs = VectorStore()
        h = "g" * 64
        # Store then delete to simulate corruption.
        vs.store_chunks(
            h, ["original text"], [{"source_hash": h, "filename": "f", "page": 1, "chunk_index": 0}]
        )
        vs.delete_collection(h)
        assert vs.collection_exists(h) is False
        # Rebuild from cached text (chunked internally).
        vs.rebuild_collection(h, "rebuilt text content here", filename="rebuilt.pdf", page_count=1)
        assert vs.collection_exists(h) is True
        results = vs.retrieve_with_scores(h, "rebuilt", top_k=1)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# get_vector_store singleton
# ---------------------------------------------------------------------------


class TestGetVectorStore:
    def test_returns_vector_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        from src.services.vector_store import reset_vector_store

        reset_vector_store()
        vs = get_vector_store()
        assert isinstance(vs, VectorStore)
        reset_vector_store()
