"""Summary routes: get + regenerate."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify
from flask_login import login_required

from src.extensions import db
from src.routes._helpers import require_owner
from src.services.summary_service import SummaryService

summary_bp = Blueprint("summary", __name__)


@summary_bp.get("/notebooks/<int:notebook_id>/summary")
@login_required
def get_summary(notebook_id: int) -> tuple[Response, int]:
    """Return the notebook's current summary + suggested questions."""
    notebook = require_owner(notebook_id)
    return (
        jsonify(
            summary=notebook.summary or "",
            suggested_questions=json.loads(notebook.suggested_questions)
            if notebook.suggested_questions
            else [],
        ),
        200,
    )


@summary_bp.post("/notebooks/<int:notebook_id>/summary/regenerate")
@login_required
def regenerate_summary(notebook_id: int) -> tuple[Response, int]:
    """Force-regenerate the summary (clears content_signature to bypass idempotency)."""
    notebook = require_owner(notebook_id)
    notebook.content_signature = None
    db.session.commit()

    svc = SummaryService()
    result = svc.generate_summary(notebook)
    if result is None:
        return jsonify(error="Summary generation failed."), 500
    return (
        jsonify(
            summary=result.summary,
            suggested_questions=result.suggested_questions,
            skipped=result.skipped,
        ),
        200,
    )
