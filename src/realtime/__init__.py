"""Realtime (SocketIO) namespace for voice conversation.

Implements the streaming voice status notifications over a ``/voice`` SocketIO
namespace:

Client -> server events:
- ``voice:start``    {notebook_id, topic?}  — authorize + open a session
- ``voice:audio_chunk`` (binary bytes)      — append recorded audio
- ``voice:stop``     — finalize + run the pipeline
- ``voice:cancel``   — abort the in-flight turn

Server -> client events:
- ``voice:status``      {state}  — ready | transcribing | thinking | speaking | done
- ``voice:transcript``  {text, final}
- ``voice:answer``      {text, sources}
- ``voice:audio_chunk`` (binary bytes) — streamed reply audio
- ``voice:sources``     {sources}
- ``voice:error``       {error}
- ``voice:done``        {state}

The background worker writes the accumulated audio to a temp file, calls
``VoiceService.run_voice_turn`` inside a Flask app context, and streams the
reply MP3 back in chunks. The HTTP ``/voice/turn`` endpoint handles audio
upload and the full pipeline (STT -> chat -> TTS); this namespace carries
status notifications only.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-sid voice session state: {notebook_id, user_id, buffer, cancelled}.
_SESSIONS: dict[str, dict[str, Any]] = {}

# One active voice session per sid (enforced by the session map itself).

_AUDIO_CHUNK_BYTES = 16 * 1024  # 16 KB per streamed reply chunk


class VoiceNamespace:
    """Flask-SocketIO namespace ``/voice``.

    Subclassed at runtime from ``flask_socketio.Namespace`` in
    ``register_voice_namespace`` so this module imports cleanly even when
    flask_socketio is not installed.
    """

    sid: str = ""

    def on_connect(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        from flask import request
        from flask_login import current_user

        if not current_user.is_authenticated:
            return False  # reject unauthenticated connections
        sid = getattr(request, "sid", "")
        logger.info("voice namespace connected: sid=%s user=%s", sid, current_user.username)
        return True

    def on_disconnect(self) -> None:
        from flask import request

        sid = getattr(request, "sid", "")
        logger.info("voice namespace disconnected: sid=%s", sid)
        _SESSIONS.pop(sid, None)

    def on_voice_start(self, data: dict[str, Any]) -> None:
        from flask import request
        from flask_login import current_user
        from flask_socketio import emit

        logger.info("voice:start received: data=%s", data)
        notebook_id = (data or {}).get("notebook_id")
        if notebook_id is None:
            emit("voice:error", {"error": "notebook_id required"})
            return
        from src.routes._helpers import require_owner

        try:
            require_owner(int(notebook_id))
        except Exception:  # noqa: BLE001
            emit("voice:error", {"error": "not owner"})
            return
        _SESSIONS[getattr(request, "sid", "")] = {
            "notebook_id": int(notebook_id),
            "user_id": int(current_user.get_id() or 0),
            "buffer": bytearray(),
            "cancelled": False,
        }
        emit("voice:status", {"state": "ready"})

    def on_voice_audio_chunk(self, data: Any) -> None:  # noqa: ANN401
        # Audio is sent via the HTTP /voice/turn endpoint (multipart form),
        # not via SocketIO binary events. The HTTP path is robust and runs in
        # a proper Flask request context. This handler is kept for protocol
        # compatibility but binary audio chunks are not accepted here.
        from flask_socketio import emit

        emit("voice:error", {"error": "audio must be sent via HTTP POST /voice/turn"})

    def on_voice_stop(self, data: dict[str, Any] | None = None) -> None:
        from flask_socketio import emit

        # Audio processing happens via the HTTP /voice/turn endpoint, which
        # runs in a proper Flask request context. SocketIO only carries status.
        emit("voice:status", {"state": "sending"})

    def on_voice_cancel(self, data: dict[str, Any] | None = None) -> None:
        from flask import request
        from flask_socketio import emit

        sid = getattr(request, "sid", "")
        sess = _SESSIONS.get(sid)
        if sess is not None:
            sess["cancelled"] = True
        _SESSIONS.pop(sid, None)
        emit("voice:done", {"state": "cancelled"})


def _run_turn(app: Any, sid: str, sess: dict[str, Any]) -> None:  # noqa: ANN401
    """Background worker: write buffer to temp file, run the voice pipeline,
    and emit transcript/answer/sources/audio_chunk/done events to ``sid``.

    Runs inside a pushed Flask app context (``app``) so DB + config access
    work. Checks the cancellation flag before each heavy step; if cancelled,
    emits ``voice:done`` with ``state=cancelled`` and cleans up.
    """
    from flask_socketio import emit

    cfg = app.config["NOTEBOOK_CONFIG"]
    ns = "/voice"

    def _emit(event: str, data: Any) -> None:  # noqa: ANN401
        emit(event, data, namespace=ns, to=sid)

    tmp_path: str | None = None
    try:
        with app.app_context():
            if sess.get("cancelled"):
                _emit("voice:done", {"state": "cancelled"})
                return

            buffer: bytearray = sess["buffer"]
            if not buffer:
                _emit("voice:error", {"error": "no audio received"})
                return

            # Write the accumulated audio to a temp file for the STT pipeline.
            fd, tmp_path = tempfile.mkstemp(suffix="_voice.webm")
            with os.fdopen(fd, "wb") as fh:
                fh.write(bytes(buffer))

            # Resolve the notebook + speaker.
            from src.repositories import notebook_repo

            notebook = notebook_repo.get_by_id(int(sess["notebook_id"]))
            if notebook is None:
                _emit("voice:error", {"error": "notebook not found"})
                return

            from src.repositories import user_repo

            user = user_repo.get_by_id(int(sess["user_id"]))
            speaker = (
                getattr(user, "voice_speaker", cfg.voice_tts_fallback_speaker)
                if user
                else cfg.voice_tts_fallback_speaker
            )

            # Transcribe + chat + TTS.
            _emit("voice:status", {"state": "transcribing"})
            from src.services.voice_service import get_voice_service

            result = get_voice_service().run_voice_turn(notebook, tmp_path, speaker)

            if result.error == "no_speech":
                _emit("voice:error", {"error": "no_speech"})
                return
            if result.error:
                _emit("voice:error", {"error": result.error})
                return

            _emit("voice:transcript", {"text": result.transcript, "final": True})

            if sess.get("cancelled"):
                _emit("voice:done", {"state": "cancelled"})
                return

            # Answer + sources.
            _emit("voice:answer", {"text": result.answer, "sources": result.sources})
            _emit("voice:sources", {"sources": result.sources})

            # Stream the reply audio back in chunks, if present.
            if result.reply_audio_path and Path(result.reply_audio_path).is_file():
                _emit("voice:status", {"state": "speaking"})
                with open(result.reply_audio_path, "rb") as af:
                    while True:
                        if sess.get("cancelled"):
                            _emit("voice:done", {"state": "cancelled"})
                            return
                        chunk = af.read(_AUDIO_CHUNK_BYTES)
                        if not chunk:
                            break
                        emit("voice:audio_chunk", chunk, namespace=ns, to=sid)

            _emit("voice:status", {"state": "done"})
            _emit("voice:done", {"state": "done"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("voice namespace turn failed: %s", exc)
        with contextlib.suppress(Exception):
            _emit("voice:error", {"error": str(exc)})
    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink(missing_ok=True)


def register_voice_namespace(socketio: Any) -> None:  # noqa: ANN401
    """Bind the ``/voice`` namespace to a SocketIO server.

    flask_socketio dispatches event ``foo:bar`` to a handler literally named
    ``on_foo:bar`` (it does not convert ``:`` to ``_``). Since ``:`` is not a
    valid Python identifier character, we register the colon-named handlers
    via ``setattr`` on a dynamically created subclass, mapping each to the
    underscore-named method defined on ``VoiceNamespace``.
    """
    try:
        from flask_socketio import Namespace

        # Map client event names (with colons) -> handler methods (with underscores).
        event_to_method = {
            "voice:start": "on_voice_start",
            "voice:audio_chunk": "on_voice_audio_chunk",
            "voice:stop": "on_voice_stop",
            "voice:cancel": "on_voice_cancel",
        }

        class _Bound(VoiceNamespace, Namespace):  # type: ignore[misc]
            pass

        for event, method_name in event_to_method.items():
            handler = getattr(_Bound, method_name)
            setattr(_Bound, "on_" + event, handler)

        socketio.on_namespace(_Bound("/voice"))
        logger.info("Registered /voice SocketIO namespace.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not register /voice namespace: %s", exc)
