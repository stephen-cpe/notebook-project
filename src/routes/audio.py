"""Audio Overview routes: request / status / file / delete."""

from __future__ import annotations

from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    jsonify,
    request,
    send_file,
)
from flask_login import current_user, login_required
from werkzeug.wrappers import Response as WerkzeugResponse

from src.extensions import db
from src.routes._helpers import require_owner
from src.services.audio_service import AUDIO_STATUS_QUEUED
from src.services.jobs import launch_audio_job

audio_bp = Blueprint("audio", __name__)


@audio_bp.post("/notebooks/<int:notebook_id>/audio")
@login_required
def request_audio(notebook_id: int) -> tuple[Response, int]:
    """Request Audio Overview generation (async background job)."""
    notebook = require_owner(notebook_id)

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    speaker_a = getattr(current_user, "audio_speaker_a", "Ava")
    speaker_b = getattr(current_user, "audio_speaker_b", "Andrew")

    notebook.audio_status = AUDIO_STATUS_QUEUED
    db.session.commit()

    from flask import current_app

    launch_audio_job(
        notebook_id,
        current_app._get_current_object(),  # type: ignore[attr-defined]
        topic=topic,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )

    return jsonify(status="queued", message="Audio generation started."), 202


@audio_bp.get("/notebooks/<int:notebook_id>/audio/status")
@login_required
def audio_status(notebook_id: int) -> tuple[Response, int]:
    """Return the current audio generation status."""
    notebook = require_owner(notebook_id)
    return (
        jsonify(
            status=notebook.audio_status,
            has_audio=notebook.audio_path is not None,
        ),
        200,
    )


@audio_bp.get("/notebooks/<int:notebook_id>/audio/file")
@login_required
def audio_file(notebook_id: int) -> WerkzeugResponse:
    """Serve the generated audio file (owner-only)."""
    notebook = require_owner(notebook_id)
    if not notebook.audio_path:
        abort(404)
    audio_path = Path(notebook.audio_path)
    if not audio_path.is_absolute():
        audio_path = Path.cwd() / audio_path
    if not audio_path.exists():
        abort(404)
    return send_file(str(audio_path), mimetype="audio/mpeg")


@audio_bp.delete("/notebooks/<int:notebook_id>/audio")
@login_required
def delete_audio(notebook_id: int) -> tuple[Response, int]:
    """Delete the generated audio file and reset status."""
    notebook = require_owner(notebook_id)
    if notebook.audio_path:
        audio_path = Path(notebook.audio_path)
        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path
        audio_path.unlink(missing_ok=True)
    notebook.audio_path = None
    notebook.audio_status = "none"
    db.session.commit()
    return jsonify(ok=True), 200
