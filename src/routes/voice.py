"""Voice conversation routes.

HTTP push-to-talk endpoints. SocketIO is optional and registered separately
when ``VOICE_ENABLED`` and Flask-SocketIO are available (see ``src/app.py``
wiring). These HTTP routes handle audio upload and reply serving.
"""

from __future__ import annotations

import os
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

from src.routes._helpers import require_owner

voice_bp = Blueprint("voice", __name__)


def _voice_enabled() -> bool:
    """Return True when voice is enabled in the app config."""
    from flask import current_app

    return bool(current_app.config["NOTEBOOK_CONFIG"].voice_enabled)


@voice_bp.post("/notebooks/<int:notebook_id>/voice/turn")
@login_required
def voice_turn(notebook_id: int) -> tuple[Response, int]:
    """Run one voice turn: STT -> chat -> TTS. Returns JSON.

    Multipart form: field ``audio`` (the recorded blob), optional ``topic``.
    """
    if not _voice_enabled():
        return jsonify(error="voice_disabled"), 404

    notebook = require_owner(notebook_id)
    from flask import current_app

    cfg = current_app.config["NOTEBOOK_CONFIG"]

    if "audio" not in request.files:
        return jsonify(error="No audio file provided."), 400
    file = request.files["audio"]
    if not file or not file.filename:
        return jsonify(error="No audio file selected."), 400

    # Size guard (server-side, in addition to MAX_CONTENT_LENGTH).
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > cfg.voice_max_upload_mb * 1024 * 1024:
        return jsonify(error="Audio file too large."), 413

    # Save to a temp file for the STT pipeline.
    import tempfile

    suffix = Path(file.filename).suffix or ".webm"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file.save(tmp_name)
        tmp_path = tmp_name

        speaker = getattr(current_user, "voice_speaker", cfg.voice_tts_fallback_speaker)
        from src.services.voice_service import get_voice_service

        result = get_voice_service().run_voice_turn(notebook, tmp_path, speaker)

        if result.error == "no_speech":
            return jsonify(error="no_speech"), 422
        if result.error:
            return jsonify(error=result.error), 422

        return (
            jsonify(
                transcript=result.transcript,
                answer=result.answer,
                sources=result.sources,
                latency_ms=result.latency_ms,
                reply_audio_url=result.reply_audio_url,
            ),
            200,
        )
    finally:
        Path(tmp_name).unlink(missing_ok=True)


@voice_bp.get("/notebooks/<int:notebook_id>/voice/reply/<path:filename>")
@login_required
def voice_reply(notebook_id: int, filename: str) -> Response:
    """Serve a generated reply audio file (owner-only, sanitized path)."""
    if not _voice_enabled():
        abort(404)
    require_owner(notebook_id)
    # Sanitize filename: reject traversal, allow only a basename.
    safe = Path(filename).name
    if not safe or safe in (".", "..") or "/" in filename or "\\" in filename:
        abort(400)
    from flask import current_app

    cfg = current_app.config["NOTEBOOK_CONFIG"]
    reply_path = Path(cfg.data_dir).resolve() / "voice" / str(notebook_id) / safe
    if not reply_path.is_file():
        abort(404)
    return send_file(str(reply_path), mimetype="audio/mpeg")
