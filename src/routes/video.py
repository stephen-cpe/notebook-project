"""Video Overview routes: request / status / file / delete."""

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
from src.services.jobs import launch_video_job
from src.services.video_service import VIDEO_STATUS_QUEUED

video_bp = Blueprint("video", __name__)


@video_bp.post("/notebooks/<int:notebook_id>/video")
@login_required
def request_video(notebook_id: int) -> tuple[Response, int]:
    """Request Video Overview generation (async background job)."""
    notebook = require_owner(notebook_id)

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    speaker = getattr(current_user, "video_speaker", "Ava")

    notebook.video_status = VIDEO_STATUS_QUEUED
    db.session.commit()

    from flask import current_app as flask_app

    launch_video_job(notebook_id, flask_app._get_current_object(), topic=topic, speaker=speaker)  # type: ignore[attr-defined]

    return jsonify(status="queued", message="Video generation started."), 202


@video_bp.get("/notebooks/<int:notebook_id>/video/status")
@login_required
def video_status(notebook_id: int) -> tuple[Response, int]:
    """Return the current video generation status."""
    notebook = require_owner(notebook_id)
    return (
        jsonify(
            status=notebook.video_status,
            has_video=notebook.video_path is not None,
        ),
        200,
    )


@video_bp.get("/notebooks/<int:notebook_id>/video/file")
@login_required
def video_file(notebook_id: int) -> WerkzeugResponse:
    """Serve the generated video file (owner-only)."""
    notebook = require_owner(notebook_id)
    if not notebook.video_path:
        abort(404)
    video_path = Path(notebook.video_path)
    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path
    if not video_path.exists():
        abort(404)
    return send_file(str(video_path), mimetype="video/mp4")


@video_bp.delete("/notebooks/<int:notebook_id>/video")
@login_required
def delete_video(notebook_id: int) -> tuple[Response, int]:
    """Delete the generated video file and reset status."""
    notebook = require_owner(notebook_id)
    if notebook.video_path:
        video_path = Path(notebook.video_path)
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path
        video_path.unlink(missing_ok=True)
    notebook.video_path = None
    notebook.video_status = "none"
    db.session.commit()
    return jsonify(ok=True), 200
