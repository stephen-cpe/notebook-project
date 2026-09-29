"""Notebook export — summary, sources, and chat history as a PDF file.

Purpose: give the user one portable document they can read or share offline.
Everything the notebook contains (description, summary, suggested questions,
source list with status, and the full chat transcript with citation labels)
is rendered with fpdf2 core fonts, so only latin-1-encodable text is used.
"""

from __future__ import annotations

import json
import logging

from flask import Blueprint, Response
from flask_login import login_required

from src.repositories import chat_repo, source_repo
from src.routes._helpers import require_owner
from src.services.summary_service import parse_suggested_questions

export_bp = Blueprint("export", __name__)

logger = logging.getLogger(__name__)

_PAGE_WIDTH = 0.0  # fpdf2 sentinel: full width minus both margins


class _NotebookPDF:  # thin builder wrapper; keeps fpdf details in one place
    """Small wrapper that centralizes fpdf2 cursor handling.

    fpdf2's ``multi_cell`` leaves the cursor at the right edge; the next
    width-0 cell then has zero available width and raises
    "Not enough horizontal space to render a single character". Every
    paragraph here flows to the next line at the left margin so calls are
    independent of each other.
    """

    def __init__(self, title: str) -> None:
        from fpdf import FPDF

        self._pdf = FPDF()
        self._pdf.set_auto_page_break(auto=True, margin=18)
        self._pdf.set_title(_safe(title))
        self._pdf.add_page()

    def heading(self, text: str, size: int = 13) -> None:
        self._pdf.ln(3)
        self._pdf.set_font("Helvetica", "B", size)
        self._pdf.cell(_PAGE_WIDTH, 8, _safe(text), new_x="LMARGIN", new_y="NEXT")

    def label(self, text: str, size: int = 11) -> None:
        """Bold single-line label (e.g. the chat role)."""
        self._pdf.set_font("Helvetica", "B", size)
        self._pdf.cell(_PAGE_WIDTH, 6, _safe(text), new_x="LMARGIN", new_y="NEXT")

    def paragraph(self, text: str, size: int = 11, italic: bool = False) -> None:
        self._pdf.set_font("Helvetica", "I" if italic else "", size)
        self._pdf.multi_cell(_PAGE_WIDTH, 6, _safe(text), new_x="LMARGIN", new_y="NEXT")

    def gap(self, lines: int = 1) -> None:
        self._pdf.ln(3 * lines)

    def footer(self, text: str) -> None:
        self._pdf.set_y(-14)
        self._pdf.set_font("Helvetica", "I", 8)
        self._pdf.set_text_color(130, 130, 130)
        self._pdf.cell(_PAGE_WIDTH, 6, _safe(text), align="C")
        self._pdf.set_text_color(0, 0, 0)

    def output(self) -> bytes:
        return bytes(self._pdf.output())


@export_bp.get("/notebooks/<int:notebook_id>/export")
@login_required
def export_notebook(notebook_id: int) -> Response:
    """Download the notebook as a PDF (summary, sources, chat)."""
    try:
        from fpdf import FPDF  # noqa: F401
    except ImportError:
        return Response("PDF export requires fpdf2 (pip install fpdf2).", status=501)
    notebook = require_owner(notebook_id)
    sources = source_repo.list_by_notebook(notebook_id)
    messages = chat_repo.list_by_notebook(notebook_id)
    questions = parse_suggested_questions(notebook.suggested_questions)

    try:
        pdf = _NotebookPDF(notebook.name)
        pdf.heading(notebook.name, size=16)
        if notebook.description:
            pdf.paragraph(notebook.description)

        pdf.heading("Summary")
        pdf.paragraph(notebook.summary or "No summary yet.")

        if questions:
            pdf.heading("Suggested questions", size=12)
            for q in questions:
                pdf.paragraph(f"- {q}")

        pdf.heading(f"Sources ({len(sources)})")
        if not sources:
            pdf.paragraph("No sources uploaded yet.")
        for s in sources:
            pdf.paragraph(f"- {s.filename} [{s.status}]")

        pdf.heading("Chat history")
        if not messages:
            pdf.paragraph("No messages yet.")
        for m in messages:
            pdf.label("You" if m.role == "user" else "Assistant")
            pdf.paragraph(m.content or "")
            labels = _source_labels(m.sources_json)
            if labels:
                pdf.paragraph(f"Sources: {labels}", size=10, italic=True)

        pdf.footer(f"notebook-project — {notebook.name}")
        payload = pdf.output()
    except Exception:  # noqa: BLE001
        logger.exception("PDF export failed for notebook %d", notebook_id)
        return Response("PDF export failed. Please try again.", status=500)

    return Response(
        payload,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="notebook-{notebook_id}.pdf"'},
    )


def _source_labels(sources_json: str | None) -> str:
    """Group a message's cited sources as ``file (pp. 1, 2)`` labels."""
    try:
        entries = json.loads(sources_json) if sources_json else []
    except (ValueError, TypeError):
        return ""
    if not isinstance(entries, list):
        return ""
    pages: dict[str, list[int]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("filename", "")).strip()
        if not name:
            continue
        page = entry.get("page")
        bucket = pages.setdefault(name, [])
        if isinstance(page, int) and page not in bucket:
            bucket.append(page)
    parts: list[str] = []
    for name, nums in pages.items():
        nums.sort()
        if len(nums) == 1:
            parts.append(f"{name} (p. {nums[0]})")
        elif nums:
            parts.append(f"{name} (pp. {', '.join(str(n) for n in nums)})")
        else:
            parts.append(name)
    return ", ".join(parts)


def _safe(text: str) -> str:
    """Strip characters fpdf2 core fonts cannot encode (latin-1)."""
    return (text or "").encode("latin-1", errors="replace").decode("latin-1")
