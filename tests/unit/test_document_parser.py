"""Unit tests for src.services.document_parser (TDD step 7).

Covers PDF/DOCX/PPTX/TXT/MD extraction against real fixture files:
- Each parser returns non-empty text for a valid fixture.
- TXT/MD parsers return exact content.
- PDF parser extracts multi-page text.
- DOCX parser extracts paragraph text.
- PPTX parser extracts slide text.
- Unknown/unsupported type raises a clear error.
- Missing file raises a clear error.
- Empty-text detection (for OCR fallback threshold).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.document_parser import (
    detect_content_type,
    extract_text,
    parse_docx,
    parse_pdf,
    parse_pptx,
    parse_text_file,
)
from src.services.exceptions import IngestionError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Per-type parsing
# ---------------------------------------------------------------------------


class TestParsePdf:
    def test_extracts_text(self) -> None:
        text = parse_pdf(str(FIXTURES / "sample.pdf"))
        assert isinstance(text, str)
        assert "machine learning" in text.lower()
        assert "neural networks" in text.lower()

    def test_multi_page_content(self) -> None:
        text = parse_pdf(str(FIXTURES / "sample.pdf"))
        assert "page one" in text.lower()
        assert "page two" in text.lower()

    def test_returns_pages_count(self) -> None:
        from src.services.document_parser import parse_pdf_with_pages

        text, pages = parse_pdf_with_pages(str(FIXTURES / "sample.pdf"))
        assert pages == 2


class TestParseDocx:
    def test_extracts_text(self) -> None:
        text = parse_docx(str(FIXTURES / "sample.docx"))
        assert isinstance(text, str)
        assert "artificial intelligence" in text.lower()
        assert "transformers" in text.lower()

    def test_includes_headings(self) -> None:
        text = parse_docx(str(FIXTURES / "sample.docx"))
        assert "DOCX Fixture Document" in text


class TestParsePptx:
    def test_extracts_text(self) -> None:
        text = parse_pptx(str(FIXTURES / "sample.pptx"))
        assert isinstance(text, str)
        assert "cloud computing" in text.lower()
        assert "kubernetes" in text.lower()

    def test_includes_slide_titles(self) -> None:
        text = parse_pptx(str(FIXTURES / "sample.pptx"))
        assert "Slide One" in text or "Slide Two" in text


class TestParseTextFile:
    def test_txt_exact(self) -> None:
        text = parse_text_file(str(FIXTURES / "sample.txt"))
        assert "databases" in text.lower()
        assert "PostgreSQL" in text

    def test_md_exact(self) -> None:
        text = parse_text_file(str(FIXTURES / "sample.md"))
        assert "web development" in text.lower()
        assert "Flask" in text

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert parse_text_file(str(p)) == ""

    def test_whitespace_only_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "ws.txt"
        p.write_text("   \n\n  \t  ", encoding="utf-8")
        assert parse_text_file(str(p)).strip() == ""


# ---------------------------------------------------------------------------
# detect_content_type
# ---------------------------------------------------------------------------


class TestDetectContentType:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("doc.pdf", "pdf"),
            ("report.PDF", "pdf"),
            ("file.docx", "docx"),
            ("slides.pptx", "pptx"),
            ("notes.txt", "txt"),
            ("readme.md", "md"),
            ("README.MD", "md"),
        ],
    )
    def test_extension_detection(self, filename: str, expected: str) -> None:
        assert detect_content_type(filename) == expected

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(IngestionError):
            detect_content_type("file.xyz")

    def test_no_extension_raises(self) -> None:
        with pytest.raises(IngestionError):
            detect_content_type("noextension")


# ---------------------------------------------------------------------------
# extract_text (dispatch by type)
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_pdf(self) -> None:
        text = extract_text(str(FIXTURES / "sample.pdf"), "pdf")
        assert "machine learning" in text.lower()

    def test_docx(self) -> None:
        text = extract_text(str(FIXTURES / "sample.docx"), "docx")
        assert "artificial intelligence" in text.lower()

    def test_pptx(self) -> None:
        text = extract_text(str(FIXTURES / "sample.pptx"), "pptx")
        assert "cloud computing" in text.lower()

    def test_txt(self) -> None:
        text = extract_text(str(FIXTURES / "sample.txt"), "txt")
        assert "databases" in text.lower()

    def test_md(self) -> None:
        text = extract_text(str(FIXTURES / "sample.md"), "md")
        assert "web development" in text.lower()

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(IngestionError):
            extract_text(str(FIXTURES / "sample.txt"), "xyz")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(IngestionError):
            extract_text("nonexistent_file_12345.pdf", "pdf")

    def test_empty_pdf_returns_empty(self) -> None:
        """A blank PDF (no text layer) returns empty -> triggers OCR fallback."""
        text = extract_text(str(FIXTURES / "empty.pdf"), "pdf")
        assert text.strip() == ""


# ---------------------------------------------------------------------------
# Image extraction (DOCX/PPTX) for OCR fallback
# ---------------------------------------------------------------------------


class TestExtractDocxImages:
    def test_extracts_image_from_docx_with_image(self) -> None:
        from src.services.document_parser import extract_docx_images

        images = extract_docx_images(str(FIXTURES / "_ocr_with_image.docx"))
        assert len(images) == 1
        # Each image is a PIL Image.
        from PIL import Image

        assert isinstance(images[0], Image.Image)

    def test_no_images_in_text_only_docx(self) -> None:
        from src.services.document_parser import extract_docx_images

        images = extract_docx_images(str(FIXTURES / "sample.docx"))
        assert images == []

    def test_missing_file_returns_empty(self) -> None:
        from src.services.document_parser import extract_docx_images

        assert extract_docx_images("nonexistent_12345.docx") == []


class TestExtractPptxImages:
    def test_extracts_image_from_pptx_with_image(self) -> None:
        from src.services.document_parser import extract_pptx_images

        images = extract_pptx_images(str(FIXTURES / "_ocr_with_image.pptx"))
        assert len(images) == 1
        from PIL import Image

        assert isinstance(images[0], Image.Image)

    def test_no_images_in_text_only_pptx(self) -> None:
        from src.services.document_parser import extract_pptx_images

        images = extract_pptx_images(str(FIXTURES / "sample.pptx"))
        assert images == []

    def test_missing_file_returns_empty(self) -> None:
        from src.services.document_parser import extract_pptx_images

        assert extract_pptx_images("nonexistent_12345.pptx") == []
