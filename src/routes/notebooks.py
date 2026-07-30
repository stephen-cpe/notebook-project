"""Notebook CRUD routes."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.wrappers import Response as WerkzeugResponse

from src.repositories import notebook_repo
from src.routes._helpers import require_owner

notebooks_bp = Blueprint("notebooks", __name__)

ViewReturn = WerkzeugResponse | str | tuple[Any, int]


@notebooks_bp.get("/notebooks")
@login_required
def list_notebooks() -> ViewReturn:
    """List the current user's notebooks."""
    uid = int(current_user.get_id())
    notebooks = notebook_repo.list_by_user(uid)
    return render_template("notebooks/list.html", notebooks=notebooks)


@notebooks_bp.post("/notebooks")
@login_required
def create_notebook() -> ViewReturn:
    """Create a new notebook."""
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip() or None
    if not name:
        flash("Notebook name is required.", category="error")
        return render_template("notebooks/list.html"), 400
    if len(name) > 120:
        flash("Notebook name must be 120 characters or fewer.", category="error")
        return render_template("notebooks/list.html"), 400
    uid = int(current_user.get_id())
    notebook_repo.create_notebook(uid, name, description)
    flash(f"Notebook '{name}' created.", category="success")
    return redirect(url_for("notebooks.list_notebooks"))


@notebooks_bp.get("/notebooks/<int:notebook_id>")
@login_required
def open_notebook(notebook_id: int) -> ViewReturn:
    """Open the 3-panel notebook view."""
    from flask import current_app

    notebook = require_owner(notebook_id)
    from src.repositories import source_repo

    sources = source_repo.list_by_notebook(notebook_id)
    suggested_questions: list[str] = []
    if notebook.suggested_questions:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            suggested_questions = json.loads(notebook.suggested_questions)
    cfg = current_app.config["NOTEBOOK_CONFIG"]
    return render_template(
        "notebook.html",
        notebook=notebook,
        sources=sources,
        suggested_questions=suggested_questions,
        voice_enabled=cfg.voice_enabled,
        voice_max_recording_seconds=cfg.voice_max_recording_seconds,
    )


@notebooks_bp.post("/notebooks/<int:notebook_id>/rename")
@login_required
def rename_notebook(notebook_id: int) -> ViewReturn:
    """Rename a notebook."""
    notebook = require_owner(notebook_id)
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify(error="Name is required"), 400
    notebook_repo.update_notebook(notebook, name=name)
    return jsonify(ok=True)


@notebooks_bp.post("/notebooks/<int:notebook_id>/delete")
@login_required
def delete_notebook(notebook_id: int) -> ViewReturn:
    """Delete a notebook (cascades to sources + chat).

    Reference-counted cleanup (P0-1.3): the notebook's audio/video/voice files
    are deleted (notebook-specific), and each source's shared ChromaDB
    collection + ContentRegistry entry is removed if no other notebook
    references the same content hash.
    """
    from flask import current_app

    from src.repositories import source_repo
    from src.services.cleanup_service import (
        cleanup_notebook_media,
        cleanup_notebook_orphaned_content,
    )

    notebook = require_owner(notebook_id)
    # Snapshot hashes BEFORE cascade delete removes the sources.
    hashes = source_repo.list_hashes_by_notebook(notebook_id)
    cfg = current_app.config["NOTEBOOK_CONFIG"]
    notebook_repo.delete_notebook(notebook)
    # Best-effort cleanup of orphaned shared content + notebook media files.
    cleanup_notebook_orphaned_content(hashes)
    cleanup_notebook_media(notebook_id, cfg.data_dir)
    flash("Notebook deleted.", category="success")
    return redirect(url_for("notebooks.list_notebooks"))
