"""Thin application entry point.

Run with: ``python app.py``
"""

from __future__ import annotations

from src.app import create_app
from src.config import get_config

app = create_app(get_config())


if __name__ == "__main__":
    cfg = app.config["NOTEBOOK_CONFIG"]
    socketio = app.extensions.get("socketio")
    if socketio is not None and cfg.voice_enabled:
        socketio.run(app, host="127.0.0.1", port=cfg.flask_port, debug=cfg.is_test())
    else:
        app.run(host="127.0.0.1", port=cfg.flask_port, debug=cfg.is_test())