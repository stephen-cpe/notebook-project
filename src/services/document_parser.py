"""Document parser — text extraction from PDF/DOCX/PPTX/TXT/MD.

Each parser reads a file path and returns extracted text. The dispatcher
``extract_text(path, content_type)`` routes by type. When text extraction
yields little/no text (e.g. scanned PDFs), the ingestion pipeline falls back
to GLM-OCR (see ``ocr_service.py``).

Supported types and libraries:
- pdf:  ``pypdf``
- docx: ``python-docx``
- pptx: ``python-pptx``
- txt/md: plain read
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.services.exceptions import IngestionError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
TYPE_TO_EXTENSIONS = {
    "pdf": ".pdf",
    "docx": ".docx",
    "pptx": ".pptx",
    "txt": ".txt",
    "md": ".md",
}

# Resource bounds for ZIP-based Office media extraction (zip-bomb protection).
# Checked against ZIP metadata *before* member data is read into memory.
_MAX_ARCHIVE_MEMBERS = 100
_MAX_ARCHIVE_MEMBER_BYTES = 25 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 100 * 1024 * 1024


def detect_content_type(filename: str) -> str:
    """Return the content type string from a filename's extension.

    Raises:
        IngestionError: if the extension is unsupported or missing.
    """
    ext = Path(filename).suffix.lower()
    for ctype, mapped_ext in TYPE_TO_EXTENSIONS.items():
        if ext == mapped_ext:
            return ctype
    if not ext:
        raise IngestionError(f"Cannot detect content type: {filename!r} has no extension")
    raise IngestionError(f"Unsupported file type: {ext!r} (allowed: {SUPPORTED_EXTENSIONS})")


def extract_text(path: str, content_type: str) -> str:
    """Dispatch to the right parser by ``content_type``.

    Raises:
        IngestionError: if the file is missing or the type is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if content_type == "pdf":
        return parse_pdf(path)
    if content_type == "docx":
        return parse_docx(path)
    if content_type == "pptx":
        return parse_pptx(path)
    if content_type in ("txt", "md"):
        return parse_text_file(path)
    raise IngestionError(f"Unsupported content type: {content_type!r}")


def parse_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf (multi-page)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n\n".join(parts)


def parse_pdf_with_pages(path: str) -> tuple[str, int]:
    """Like ``parse_pdf`` but also returns the page count."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n\n".join(parts), len(reader.pages)


def parse_pdf_pages(path: str) -> list[tuple[int, str]]:
    """Extract per-page ``(page_number, text)`` from a PDF.

    Page numbers are 1-based and always reflect the real document page,
    so chunks built from these units cite correct pages. Empty pages
    are skipped (their numbers are simply absent from the result).
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    for n, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((n, text))
    return pages


@dataclass
class TextUnit:
    """A chunkable text span with its real location.

    ``page`` is the 1-based PDF page or PPTX slide number, or ``None``
    for formats/locations without pages (TXT/MD/DOCX, OCR images).
    """

    text: str
    page: int | None


def extract_units(path: str, content_type: str) -> tuple[list[TextUnit], int | None]:
    """Extract location-aware text units plus a page/slide count.

    Returns ``(units, page_count)`` where ``page_count`` is the total PDF
    page or PPTX slide count (``None`` for other types). Raises
    ``IngestionError`` for missing files or unsupported types, mirroring
    ``extract_text``.
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if content_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        units: list[TextUnit] = []
        for n, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                units.append(TextUnit(text=text, page=n))
        return units, len(reader.pages)
    if content_type == "pptx":
        from pptx import Presentation

        prs = Presentation(path)
        units = [
            TextUnit(text=text, page=n)
            for n, text in _slides_from_presentation(prs)
            if text.strip()
        ]
        return units, len(prs.slides)
    if content_type == "docx":
        return [TextUnit(text=parse_docx(path), page=None)], None
    if content_type in ("txt", "md"):
        return [TextUnit(text=parse_text_file(path), page=None)], None
    raise IngestionError(f"Unsupported content type: {content_type!r}")


def _slides_from_presentation(prs: Any) -> list[tuple[int, str]]:  # noqa: ANN401
    """Collect ``(slide_number, text)`` from an open PPTX presentation."""
    slides: list[tuple[int, str]] = []
    for n, slide in enumerate(prs.slides, start=1):
        slide_texts: list[str] = []
        for shape in slide.shapes:
            slide_texts.extend(_iter_shape_texts(shape))
        notes = _extract_slide_notes(slide)
        if notes:
            slide_texts.append(notes)
        if slide_texts:
            slides.append((n, "\n".join(slide_texts)))
    return slides


def parse_docx(path: str) -> str:
    """Extract text from a DOCX using python-docx.

    Includes body paragraphs, tables (row cells joined with ``|``),
    and headers/footers so tabular and peripheral content is not silently
    dropped from retrieval and summaries.
    """
    from docx import Document

    doc = Document(path)
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text and para.text.strip():
            parts.append(para.text)
    for i, table in enumerate(doc.tables):
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[Table {i + 1}]\n" + "\n".join(rows))
    for section in doc.sections:
        for container in (section.header, section.footer):
            for para in container.paragraphs:
                if para.text and para.text.strip():
                    parts.append(para.text)
    return "\n\n".join(parts)


def parse_pptx(path: str) -> str:
    """Extract text from a PPTX using python-pptx (all slides joined)."""
    return "\n\n".join(text for _, text in parse_pptx_slides(path))


def parse_pptx_slides(path: str) -> list[tuple[int, str]]:
    """Extract per-slide ``(slide_number, text)`` from a PPTX.

    Covers text frames, tables, grouped shapes (recursively), and speaker
    notes so slide content beyond plain text boxes reaches retrieval.
    Slide numbers are 1-based. Slides without text are skipped.
    """
    from pptx import Presentation

    return _slides_from_presentation(Presentation(path))


def _iter_shape_texts(shape: Any) -> list[str]:  # noqa: ANN401
    """Recursively collect text from a PPTX shape (groups/tables/frames)."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    try:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            texts: list[str] = []
            for sub in shape.shapes:
                texts.extend(_iter_shape_texts(sub))
            return texts
        if shape.has_table:
            cells: list[str] = []
            for row in shape.table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                if any(row_cells):
                    cells.append(" | ".join(row_cells))
            return cells
        if shape.has_text_frame:
            return [
                para.text for para in shape.text_frame.paragraphs if para.text and para.text.strip()
            ]
    except Exception:  # noqa: BLE001
        logger.debug("Skipping unreadable PPTX shape", exc_info=True)
    return []


def _extract_slide_notes(slide: Any) -> str:  # noqa: ANN401
    """Return a slide's speaker-notes text, or "" when absent."""
    try:
        notes_slide = slide.notes_slide
        parts = [
            shape.text
            for shape in notes_slide.placeholders
            if shape.has_text_frame and shape.text and shape.text.strip()
        ]
        return "\n".join(parts)
    except Exception:  # noqa: BLE001
        return ""


def parse_text_file(path: str) -> str:
    """Read a plain text or markdown file as UTF-8."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def extract_docx_images(path: str) -> list[Any]:
    """Extract all embedded images from a DOCX as PIL Images.

    DOCX files are ZIP archives; embedded images live under ``word/media/``.
    Returns a list of PIL Images. Returns an empty list if the file contains
    no images or extraction fails for any reason (graceful degradation, FR-24).
    """
    return _extract_zip_media(path, "word/media/")


def extract_pptx_images(path: str) -> list[Any]:
    """Extract all embedded images from a PPTX as PIL Images.

    PPTX files are ZIP archives; embedded images live under ``ppt/media/``.
    Returns a list of PIL Images. Returns an empty list if the file contains
    no images or extraction fails for any reason (graceful degradation, FR-24).
    """
    return _extract_zip_media(path, "ppt/media/")


def _extract_zip_media(path: str, media_prefix: str) -> list[Any]:
    """Extract images from a ZIP-based Office file under ``media_prefix``.

    Used by both DOCX (``word/media/``) and PPTX (``ppt/media/``) extraction.

    Resource bounds (zip-bomb protection): entry count, per-member sizes, and
    total uncompressed size are capped *before* reading member data. A file
    exceeding the caps yields no images (with a warning) rather than risking
    memory exhaustion.
    """
    import io
    import zipfile

    from PIL import Image

    images: list[Any] = []
    try:
        with zipfile.ZipFile(path) as z:
            members = [info for info in z.infolist() if info.filename.startswith(media_prefix)]
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                logger.warning(
                    "Skipping media extraction for %s: %d members exceeds limit %d",
                    path,
                    len(members),
                    _MAX_ARCHIVE_MEMBERS,
                )
                return []
            total_uncompressed = 0
            for info in members:
                if info.is_dir():
                    continue
                if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
                    logger.warning(
                        "Skipping oversized archive member %s (%d bytes) in %s",
                        info.filename,
                        info.file_size,
                        path,
                    )
                    continue
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_ARCHIVE_TOTAL_BYTES:
                    logger.warning(
                        "Skipping media extraction for %s: total uncompressed size "
                        "exceeds limit %d bytes",
                        path,
                        _MAX_ARCHIVE_TOTAL_BYTES,
                    )
                    return []
            for info in members:
                if info.is_dir() or info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
                    continue
                with z.open(info) as fh:
                    images.append(Image.open(io.BytesIO(fh.read())).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract images from %s: %s", path, exc)
        return []
    return images


# Magic bytes for supported file types (NFR-23).
_MAGIC_BYTES: dict[str, bytes] = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",
    "pptx": b"PK\x03\x04",
    # txt/md have no magic bytes — validated as plain text (no null bytes).
}


def validate_magic_bytes(file_path: str, content_type: str) -> bool:
    """Check that a file's first bytes match the expected magic bytes for its type.

    For txt/md, verifies the file contains no null bytes (plain text check).
    Returns True if the file passes validation, False otherwise.
    """
    p = Path(file_path)
    if not p.exists():
        return False

    if content_type in ("txt", "md"):
        # Plain text: read first 4 KB and check for null bytes.
        with p.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" not in chunk

    expected = _MAGIC_BYTES.get(content_type)
    if expected is None:
        return False

    with p.open("rb") as f:
        header = f.read(len(expected))
    return header == expected
