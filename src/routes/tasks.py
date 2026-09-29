"""Background-task bell feed (singleflight tracker surface)."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request
from flask_login import login_required

tasks_bp = Blueprint("tasks", __name__)


@tasks_bp.get("/tasks")
@login_required
def list_tasks() -> tuple[Response, int]:
    """Return recent background jobs + unread count (bell feed)."""
    from src.services.progress_tracker import list_jobs, unread_count

    jobs = list_jobs()
    names = _notebook_names([j.get("notebook_id", 0) for j in jobs])
    # Never leak error internals: strip to label/status/progress/notebook.
    # ``read`` and ``updated_at`` drive the bell dropdown (unread highlight,
    # relative time); neither is sensitive.
    safe = [
        {
            "key": j.get("key", ""),
            "label": j.get("label", ""),
            "notebook_id": j.get("notebook_id", 0),
            "notebook_name": names.get(j.get("notebook_id", 0), ""),
            "status": j.get("status", ""),
            "progress": j.get("progress", 0),
            "read": bool(j.get("read", False)),
            "updated_at": j.get("updated_at", 0),
        }
        for j in jobs
    ]
    return jsonify(tasks=safe, unread=unread_count()), 200


def _notebook_names(notebook_ids: list[int]) -> dict[int, str]:
    """Look up notebook names for the bell (single query, never raises)."""
    ids = sorted({int(nb_id) for nb_id in notebook_ids if nb_id})
    if not ids:
        return {}
    try:
        from src.models import Notebook

        rows = (
            Notebook.query.filter(Notebook.id.in_(ids)).with_entities(Notebook.id, Notebook.name)
        ).all()
        return {int(row[0]): str(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


@tasks_bp.post("/tasks/read")
@login_required
def mark_tasks_read() -> tuple[Response, int]:
    """Mark bell jobs read (all, or one ``key``)."""
    from src.services.progress_tracker import mark_all_read, mark_read

    data = request.get_json(silent=True) or {}
    key = (data.get("key") or "").strip()
    if key:
        mark_read(key)
    else:
        mark_all_read()
    return jsonify(ok=True), 200
