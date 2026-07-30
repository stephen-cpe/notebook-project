"""Ingestion pipeline — hash → parse → OCR fallback → chunk → embed → store.

The pipeline orchestrates the document_parser, ocr_service, chunker,
embeddings (via vector_store), and the ContentRegistry for dedup.

Flow (``ingest_file``):
1. Compute SHA-256 of the file content.
2. If ContentRegistry already has this hash, skip re-embedding (dedup).
3. Detect content type, extract text via document_parser.
4. If text is below ``OCR_TEXT_THRESHOLD`` and OCR is enabled, run OCR.
5. Chunk the text, embed, store in a content-keyed ChromaDB collection.
6. Create/update the ContentRegistry entry (hash → collection + cached text).

Returns an ``IngestionResult`` with status, hash, text, page count, OCR flag.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.chunker import chunk_text
from src.services.document_parser import (
    detect_content_type,
    extract_text,
    parse_pdf_with_pages,
)
from src.services.ocr_service import OCR_PROMPT_TEXT, get_ocr_service
from src.services.vector_store import get_collection_name, get_vector_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config


@dataclass
class IngestionResult:
    """Outcome of ingesting a single file."""

    content_hash: str
    status: str  # ready | partial | failed
    extracted_text: str
    char_count: int
    page_count: int | None
    ocr_used: bool
    error_message: str | None = None


def compute_hash(file_path: str, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class IngestionService:
    """Orchestrates the full ingestion pipeline with dedup + OCR fallback."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self._vector_store = get_vector_store()
        self._ocr = get_ocr_service()

    def ingest_file(self, file_path: str, filename: str | None = None) -> IngestionResult:
        """Ingest a single file end-to-end. Never raises; errors go to status."""
        fname = filename or Path(file_path).name
        try:
            return self._ingest(file_path, fname)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion failed for %s: %s", fname, exc)
            return IngestionResult(
                content_hash="",
                status="failed",
                extracted_text="",
                char_count=0,
                page_count=None,
                ocr_used=False,
                error_message=str(exc),
            )

    def _ingest(self, file_path: str, filename: str) -> IngestionResult:
        # 1. Hash.
        content_hash = compute_hash(file_path)

        from src.repositories import content_registry_repo

        # 2. Dedup: if a ChromaDB collection already exists, return early.
        if self._vector_store.collection_exists(content_hash):
            logger.info("Skipping re-embedding for existing hash %s", content_hash[:12])
            entry = content_registry_repo.get_by_hash(content_hash)
            if entry is None or not entry.extracted_text:
                # Collection exists but registry is missing/empty (P0-1.11).
                # We cannot recover text here without re-parsing; mark the
                # source for re-ingestion by deleting the partial collection
                # and falling through to the normal extract path.
                logger.warning(
                    "Collection exists for hash %s but ContentRegistry is missing; "
                    "rebuilding from scratch.",
                    content_hash[:12],
                )
                self._vector_store.delete_collection(content_hash)
                # Fall through to the full extraction path below.
            else:
                cached_text = entry.extracted_text
                return IngestionResult(
                    content_hash=content_hash,
                    status="ready",
                    extracted_text=cached_text,
                    char_count=len(cached_text) if cached_text else 0,
                    page_count=None,
                    ocr_used=False,
                )

        # 2b. If collection is missing but ContentRegistry has cached text,
        #     rebuild from cache instead of re-extracting.
        entry = content_registry_repo.get_by_hash(content_hash)
        if (
            entry is not None
            and entry.extracted_text
            and not self._vector_store.collection_exists(content_hash)
        ):
            logger.info(
                "Collection missing for hash %s, rebuilding from ContentRegistry cache",
                content_hash[:12],
            )
            self._vector_store.rebuild_collection(
                content_hash,
                entry.extracted_text,
                filename=filename,
            )
            return IngestionResult(
                content_hash=content_hash,
                status="ready",
                extracted_text=entry.extracted_text,
                char_count=entry.char_count,
                page_count=None,
                ocr_used=False,
            )

        # 3. Detect type + extract text.
        content_type = detect_content_type(filename)
        text, page_count, ocr_used = self._extract_with_ocr_fallback(file_path, content_type)

        if not text.strip():
            return IngestionResult(
                content_hash=content_hash,
                status="partial",
                extracted_text="",
                char_count=0,
                page_count=page_count,
                ocr_used=ocr_used,
                error_message="No text could be extracted from the file.",
            )

        # 4. Chunk + embed + store.
        chunks = chunk_text(text)
        if not chunks:
            return IngestionResult(
                content_hash=content_hash,
                status="partial",
                extracted_text=text,
                char_count=len(text),
                page_count=page_count,
                ocr_used=ocr_used,
                error_message="Chunking produced no chunks.",
            )

        metadatas = [
            {
                "source_hash": content_hash,
                "filename": filename,
                "page": (i + 1) if page_count else (i + 1),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Store chunks + register in one unit; if the registry write fails after
        # the collection is created, delete the partial collection so a later
        # retry starts clean (P0-1.11).
        try:
            self._vector_store.store_chunks(content_hash, chunks, metadatas)
            content_registry_repo.get_or_create(
                content_hash=content_hash,
                chroma_collection=get_collection_name(content_hash),
                extracted_text=text,
                char_count=len(text),
            )
        except Exception:
            logger.exception(
                "Ingestion failed after collection creation for hash %s; "
                "cleaning up partial collection.",
                content_hash[:12],
            )
            self._vector_store.delete_collection(content_hash)
            raise

        status = "ready" if not ocr_used else "ready"
        logger.info(
            "Ingested %s: hash=%s chunks=%d chars=%d ocr=%s",
            filename,
            content_hash[:12],
            len(chunks),
            len(text),
            ocr_used,
        )
        return IngestionResult(
            content_hash=content_hash,
            status=status,
            extracted_text=text,
            char_count=len(text),
            page_count=page_count,
            ocr_used=ocr_used,
        )

    def _extract_with_ocr_fallback(
        self, file_path: str, content_type: str
    ) -> tuple[str, int | None, bool]:
        """Extract text; fall back to OCR if below threshold and enabled."""
        threshold = self._config.ocr_text_threshold
        ocr_used = False

        text = extract_text(file_path, content_type)
        page_count: int | None = None
        if content_type == "pdf":
            _, page_count = parse_pdf_with_pages(file_path)

        if len(text.strip()) >= threshold:
            return text, page_count, False

        # OCR fallback.
        if self._ocr.is_available() and content_type == "pdf":
            logger.info("Text below threshold (%d chars), attempting OCR", len(text.strip()))
            try:
                ocr_text = self._ocr.ocr_pdf(file_path, OCR_PROMPT_TEXT)
            except Exception as exc:  # noqa: BLE001
                # OCR failure must not block ingestion (P0-1.10). Keep any text
                # already extracted; mark partial if no text, else proceed.
                logger.error("OCR fallback failed for %s: %s", file_path, exc)
                if text.strip():
                    # We have some text from the first pass — keep it and proceed.
                    return text, page_count, False
                # No text at all and OCR failed -> partial with a useful message.
                return text, page_count, False
            if ocr_text.strip():
                text = ocr_text
                ocr_used = True

        if not text.strip() and not self._ocr.is_available():
            # OCR disabled and no text -> partial.
            return text, page_count, False

        return text, page_count, ocr_used


_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Return a process-wide ``IngestionService`` (created lazily)."""
    global _service
    if _service is None:
        _service = IngestionService()
    return _service


def reset_ingestion_service() -> None:
    """Reset the cached service (used by tests that change config)."""
    global _service
    _service = None
