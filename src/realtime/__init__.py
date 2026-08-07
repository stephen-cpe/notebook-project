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

The HTTP ``/voice/turn`` endpoint handles audio upload and the full pipeline
(STT -> chat -> TTS); this namespace carries status notifications only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-sid voice session state: {notebook_id, user_id}.
_SESSIONS: dict[str, dict[str, Any]] = {}


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
        _SESSIONS.pop(sid, None)
        emit("voice:done", {"state": "cancelled"})


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
