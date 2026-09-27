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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.chunker import chunk_units
from src.services.document_parser import (
    detect_content_type,
    extract_docx_images,
    extract_pptx_images,
    extract_text,
    extract_units,
    parse_pdf_document,
)
from src.services.ocr_service import OCR_PROMPT_TEXT, get_ocr_service
from src.services.vector_store import get_vector_store

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


_OCR_PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]\n?")


def _split_ocr_pages(text: str) -> list[tuple[str, int | None]]:
    """Split OCR output on its ``[Page N]`` markers into ``(text, page)`` units.

    Returns ``[]`` when the text carries no markers (caller falls back to a
    single pageless unit). A trailing skip-note without a marker stays with
    the last page.
    """
    if "[Page " not in text:
        return []
    parts = _OCR_PAGE_MARKER_RE.split(text)
    # parts: [pre, n1, t1, n2, t2, ...]
    units: list[tuple[str, int | None]] = []
    pre = parts[0].strip()
    rest = parts[1:]
    first_page: int | None = None
    for i in range(0, len(rest) - 1, 2):
        try:
            page = int(rest[i])
        except ValueError:
            continue
        if first_page is None:
            first_page = page
        body = rest[i + 1].strip() if i + 1 < len(rest) else ""
        if body:
            units.append((body, page))
    if pre and first_page is not None:
        units.insert(0, (pre, first_page))
    elif pre:
        units.insert(0, (pre, None))
    return units


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

        fingerprint = self._vector_store.embedding_fingerprint

        # 2. Dedup: if a collection for the current embedding backend already
        # exists and the registry holds matching cached text, skip re-embedding.
        if self._vector_store.collection_exists(content_hash):
            logger.info("Skipping re-embedding for existing hash %s", content_hash[:12])
            entry = content_registry_repo.get_by_hash(content_hash)
            if (
                entry is not None
                and entry.extracted_text
                and entry.embedding_fingerprint == fingerprint
            ):
                cached_text = entry.extracted_text
                return IngestionResult(
                    content_hash=content_hash,
                    status="ready",
                    extracted_text=cached_text,
                    char_count=len(cached_text) if cached_text else 0,
                    page_count=None,
                    ocr_used=False,
                )
            # Collection exists but the registry is missing/empty, or was
            # embedded with a different backend version. Delete the partial or
            # stale collection and fall through to the full extraction path,
            # which re-embeds and refreshes the registry fingerprint.
            logger.warning(
                "Collection exists for hash %s but ContentRegistry is missing, "
                "empty, or stale (backend changed); rebuilding from scratch.",
                content_hash[:12],
            )
            self._vector_store.delete_collection(content_hash)
            # Fall through to the full extraction path below.

        # 2b. If the versioned collection is missing but ContentRegistry has
        # cached text embedded with the current backend, rebuild from cache
        # instead of re-extracting. Stale-fingerprint caches fall through to
        # full extraction so vectors are never mixed across backend versions.
        entry = content_registry_repo.get_by_hash(content_hash)
        if (
            entry is not None
            and entry.extracted_text
            and entry.embedding_fingerprint == fingerprint
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
        text, page_count, ocr_used, prebuilt_units = self._extract_with_ocr_fallback(
            file_path, content_type
        )

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

        # 4. Chunk per location-aware unit + embed + store. Page numbers
        # come from real document locations (PDF pages / PPTX slides) or
        # None for pageless content — never the chunk index.
        chunked = chunk_units(
            self._build_units(file_path, content_type, text, ocr_used, prebuilt_units)
        )
        if not chunked:
            return IngestionResult(
                content_hash=content_hash,
                status="partial",
                extracted_text=text,
                char_count=len(text),
                page_count=page_count,
                ocr_used=ocr_used,
                error_message="Chunking produced no chunks.",
            )
        chunks = [chunk for chunk, _ in chunked]

        metadatas = []
        for i, (_, page) in enumerate(chunked):
            # ChromaDB metadata rejects None values: pageless content omits
            # the key entirely (readers treat a missing page as None).
            metadata: dict[str, object] = {
                "source_hash": content_hash,
                "filename": filename,
                "chunk_index": i,
            }
            if page is not None:
                metadata["page"] = page
            metadatas.append(metadata)

        # Store chunks + register in one unit; if the registry write fails after
        # the collection is created, delete the partial collection so a later
        # retry starts clean.
        try:
            self._vector_store.store_chunks(content_hash, chunks, metadatas)
            content_registry_repo.get_or_create(
                content_hash=content_hash,
                chroma_collection=self._vector_store.collection_name(content_hash),
                extracted_text=text,
                char_count=len(text),
                embedding_fingerprint=fingerprint,
            )
        except Exception:
            logger.exception(
                "Ingestion failed after collection creation for hash %s; "
                "cleaning up partial collection.",
                content_hash[:12],
            )
            self._vector_store.delete_collection(content_hash)
            raise

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
            status="ready",
            extracted_text=text,
            char_count=len(text),
            page_count=page_count,
            ocr_used=ocr_used,
        )

    def _extract_with_ocr_fallback(
        self, file_path: str, content_type: str
    ) -> tuple[str, int | None, bool, list[tuple[str, int | None]] | None]:
        """Extract text; fall back to OCR if below threshold and enabled.

        Returns ``(text, page_count, ocr_used, prebuilt_units)`` where
        ``prebuilt_units`` carries location-aware ``(text, page)`` units from
        the same single read whenever the returned text is the natively
        parsed text (PDF/DOCX/TXT/MD) — ``None`` when OCR replaced the text
        or the type needs a fresh unit pass (PPTX). Reusing the read avoids
        parsing the file a second and third time in ``_build_units``.

        OCR is dispatched by content type:
        - PDF: rendered to images via Poppler (``ocr_pdf``).
        - DOCX/PPTX: embedded images extracted from the ZIP archive (``ocr_images``);
          OCR is only attempted when images exist, per the "only OCR when images
          exist" rule.
        - TXT/MD: no OCR (no images to OCR).

        OCR failure never blocks ingestion (FR-24): on failure, any text already
        extracted is kept and the source is later marked ``partial`` if empty.
        """
        threshold = self._config.ocr_text_threshold
        ocr_used = False

        prebuilt_units: list[tuple[str, int | None]] | None = None
        if content_type == "pdf":
            # One PdfReader traversal for text + units + page count.
            text, pdf_units, page_count = parse_pdf_document(file_path)
            prebuilt_units = [(u.text, u.page) for u in pdf_units]
        else:
            text = extract_text(file_path, content_type)
            page_count = None
            if content_type in ("docx", "txt", "md"):
                # extract_units() for these types trivially wraps the same
                # parsed text in one pageless unit — reuse it directly.
                prebuilt_units = [(text, None)]

        if len(text.strip()) >= threshold:
            return text, page_count, False, prebuilt_units

        # OCR fallback — only when enabled and the type has images to OCR.
        if not self._ocr.is_available():
            return text, page_count, False, prebuilt_units

        ocr_text = self._run_ocr_fallback(file_path, content_type)
        if ocr_text is None:
            # OCR was not applicable for this type (e.g. TXT/MD, or DOCX/PPTX
            # with no embedded images). Keep any extracted text as-is.
            return text, page_count, False, prebuilt_units
        if ocr_text.strip():
            text = ocr_text
            ocr_used = True
            prebuilt_units = None

        return text, page_count, ocr_used, prebuilt_units

    def _build_units(
        self,
        file_path: str,
        content_type: str,
        text: str,
        ocr_used: bool,
        prebuilt_units: list[tuple[str, int | None]] | None = None,
    ) -> list[tuple[str, int | None]]:
        """Build ``(text, page)`` units for chunking with true locations.

        PDF pages and PPTX slides keep their real 1-based numbers; other
        content yields ``page=None``. When PDF OCR replaced near-empty
        native text, units are re-split from the OCR output's ``[Page N]``
        markers (see ``ocr_service.ocr_pdf``). ``prebuilt_units`` (from the
        extraction read) skips a redundant re-parse when provided.
        """
        if content_type == "pdf" and ocr_used:
            return _split_ocr_pages(text) or ([(text, None)] if text.strip() else [])
        if prebuilt_units is not None:
            nonempty = [
                (unit_text, page) for unit_text, page in prebuilt_units if unit_text.strip()
            ]
            return nonempty or ([(text, None)] if text.strip() else [])
        try:
            units, _ = extract_units(file_path, content_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unit extraction failed for %s: %s", file_path, exc)
            return [(text, None)] if text.strip() else []
        nonempty = [(unit.text, unit.page) for unit in units if unit.text.strip()]
        return nonempty or ([(text, None)] if text.strip() else [])

    def _run_ocr_fallback(self, file_path: str, content_type: str) -> str | None:
        """Run the type-appropriate OCR fallback.

        Returns the OCR'd text (possibly empty), or ``None`` if OCR is not
        applicable for this content type (no images to OCR). Logs and swallows
        OCR errors so they never block ingestion (FR-24).
        """
        try:
            if content_type == "pdf":
                logger.info("Text below threshold, attempting PDF OCR")
                return self._ocr.ocr_pdf(file_path, OCR_PROMPT_TEXT)
            if content_type == "docx":
                images = extract_docx_images(file_path)
                if not images:
                    return None
                logger.info("Text below threshold, attempting DOCX OCR (%d images)", len(images))
                return self._ocr.ocr_images(images, OCR_PROMPT_TEXT)
            if content_type == "pptx":
                images = extract_pptx_images(file_path)
                if not images:
                    return None
                logger.info("Text below threshold, attempting PPTX OCR (%d images)", len(images))
                return self._ocr.ocr_images(images, OCR_PROMPT_TEXT)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR fallback failed for %s: %s", file_path, exc)
            return ""
        return None


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
