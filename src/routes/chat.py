"""Chat routes: SSE streaming + non-streaming JSON twin."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    stream_with_context,
)
from flask_login import login_required

from src.repositories import chat_repo
from src.routes._helpers import require_owner
from src.services.chat_service import ChatService

chat_bp = Blueprint("chat", __name__)

ViewReturn = Response | tuple[Any, int]


@chat_bp.post("/notebooks/<int:notebook_id>/chat")
@login_required
def chat_stream(notebook_id: int) -> ViewReturn:
    """SSE streaming chat endpoint.

    Accepts JSON body ``{"question": "..."}``. Returns a streaming response
    with ``text/event-stream`` content type. Each frame is ``data: {...}\\n\\n``.
    """
    notebook = require_owner(notebook_id)
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Question is required."), 400

    svc = ChatService()

    def generate() -> Generator[str]:
        try:
            yield from svc.chat_stream(notebook, question)
        except Exception as exc:
            import json
            import logging

            logger = logging.getLogger(__name__)
            logger.error("Chat stream error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@chat_bp.post("/notebooks/<int:notebook_id>/chat/sync")
@login_required
def chat_sync(notebook_id: int) -> tuple[Response, int]:
    """Non-streaming JSON twin for tests/curl (FR-45)."""
    notebook = require_owner(notebook_id)
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Question is required."), 400

    svc = ChatService()
    result = svc.chat_sync(notebook, question)
    return jsonify(result), 200


@chat_bp.post("/notebooks/<int:notebook_id>/chat/clear")
@login_required
def clear_history(notebook_id: int) -> tuple[Response, int]:
    """Clear all chat history for a notebook (FR-42)."""
    require_owner(notebook_id)
    count = chat_repo.delete_by_notebook(notebook_id)
    return jsonify(deleted=count), 200


@chat_bp.get("/notebooks/<int:notebook_id>/chat/history")
@login_required
def chat_history(notebook_id: int) -> tuple[Response, int]:
    """Return chat history for a notebook (JSON)."""
    require_owner(notebook_id)
    messages = chat_repo.list_by_notebook(notebook_id)
    return (
        jsonify(
            messages=[
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "sources": json.loads(m.sources_json) if m.sources_json else [],
                    "latency_ms": m.latency_ms,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ]
        ),
        200,
    )
