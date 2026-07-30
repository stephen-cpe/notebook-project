"""Route blueprints registration + app-wide error handlers."""

from __future__ import annotations

from flask import Flask, Response, jsonify, render_template, request

ViewReturn = tuple[Response | str, int]


def register_blueprints(app: Flask) -> None:
    """Register all route blueprints and error handlers on ``app``."""
    from .admin import admin_bp
    from .audio import audio_bp
    from .auth import auth_bp
    from .chat import chat_bp
    from .index import index_bp
    from .notebooks import notebooks_bp
    from .sources import sources_bp
    from .summary import summary_bp
    from .video import video_bp

    app.register_blueprint(index_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(notebooks_bp)
    app.register_blueprint(sources_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(summary_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(video_bp)

    # Voice blueprint is registered only when voice is enabled.
    cfg = app.config.get("NOTEBOOK_CONFIG")
    if cfg is not None and getattr(cfg, "voice_enabled", False):
        from .voice import voice_bp

        app.register_blueprint(voice_bp)

    @app.errorhandler(413)
    def payload_too_large(_err: object) -> ViewReturn:
        """JSON/HTML-friendly 413 for oversized uploads (P0-1.5)."""
        best = request.accept_mimetypes.best_match(["application/json", "text/html"])
        if best == "application/json":
            return jsonify(error="File too large."), 413
        return render_template("error.html", code=413, message="File too large."), 413

    @app.errorhandler(404)
    def not_found(_err: object) -> tuple[Response, int]:
        return jsonify(error="not found"), 404

    @app.errorhandler(403)
    def forbidden(_err: object) -> tuple[Response, int]:
        return jsonify(error="forbidden"), 403

    @app.errorhandler(500)
    def server_error(_err: object) -> tuple[Response, int]:
        return jsonify(error="internal server error"), 500

    @app.after_request
    def add_security_headers(resp: Response) -> Response:
        """Add basic Content-Security-Policy + hardening headers (P0-1.8).

        CSP allows the Bootstrap CDN (jsdelivr) plus the app's own origin and
        inline styles (needed for Bootstrap utility classes rendered inline).
        """
        if not resp.headers.get("Content-Security-Policy"):
            resp.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; "
                "font-src 'self' https://cdn.jsdelivr.net data:; "
                "media-src 'self' data: blob:; "
                "connect-src 'self' ws: wss: https://cdn.jsdelivr.net; "
                "frame-ancestors 'self'",
            )
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return resp
