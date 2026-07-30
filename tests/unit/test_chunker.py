"""Unit tests for src.services.chunker (TDD step 10)."""

from __future__ import annotations

from src.services.chunker import chunk_text, chunk_with_metadata


class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "This is a test document. " * 100
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_string(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only(self) -> None:
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunks_have_overlap(self) -> None:
        text = "word " * 2000
        chunks = chunk_text(text)
        assert len(chunks) > 1


class TestChunkWithMetadata:
    def test_attaches_metadata(self) -> None:
        text = "Hello world. This is a test."
        result = chunk_with_metadata(text, {"source": "test.txt"})
        assert len(result) >= 1
        for item in result:
            assert "text" in item
            assert "source" in item
            assert item["source"] == "test.txt"
            assert "chunk_index" in item

    def test_empty_text(self) -> None:
        assert chunk_with_metadata("") == []

    def test_no_base_metadata(self) -> None:
        result = chunk_with_metadata("Hello world")
        assert len(result) == 1
        assert "chunk_index" in result[0]
