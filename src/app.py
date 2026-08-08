"""Flask application factory.

Creates and configures the app, binds extensions, registers blueprints, and
wires error handlers. Business logic lives in ``src.services`` and
``src.repositories`` — routes stay thin (NFR-36).
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify
from flask.logging import default_handler
from flask_wtf import CSRFProtect

from . import models  # noqa: F401  (register models on db for migrations)
from .config import Config, get_config
from .extensions import db, login_manager, migrate

logger = logging.getLogger(__name__)
csrf = CSRFProtect()


def create_app(config: Config | None = None) -> Flask:
    """Build a Flask application from the given (or env-derived) config."""
    cfg = config or get_config()
    app = Flask(__name__)

    # --- Core config ---
    app.config["SECRET_KEY"] = cfg.secret_key
    if not cfg.is_test() and cfg.secret_key in ("change-me", "change-me-to-a-random-string"):
        raise RuntimeError(
            "SECRET_KEY is set to a placeholder value. "
            "Set a real SECRET_KEY in your .env file before running in production."
        )
    # Tests use SQLite in-memory; production requires PostgreSQL.
    db_uri = cfg.test_database_url if cfg.is_test() else cfg.database_url
    # For in-memory SQLite, use a StaticPool + check_same_thread=False so all
    # threads/contexts share ONE in-memory database. Without this, each
    # connection from the pool gets its own empty :memory: DB and writes in one
    # app_context (e.g. disabling a user in a test) are invisible to the
    # request context, breaking session-invalidation tests and background jobs.
    if db_uri == "sqlite:///:memory:":
        from sqlalchemy.pool import StaticPool

        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False},
        }
        app.logger.info("Using StaticPool for in-memory SQLite tests")
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["NOTEBOOK_CONFIG"] = cfg
    app.config["WTF_CSRF_ENABLED"] = not cfg.is_test()

    # --- Request size limit ---
    # max_file_size_mb plus a 2 MB margin for multipart overhead.
    app.config["MAX_CONTENT_LENGTH"] = (cfg.max_file_size_mb + 2) * 1024 * 1024

    # --- Session cookie hardening ---
    app.config["SESSION_COOKIE_HTTPONLY"] = cfg.session_cookie_httponly
    app.config["SESSION_COOKIE_SAMESITE"] = cfg.session_cookie_samesite
    # SECURE defaults False for local HTTP; set True in production over HTTPS.
    app.config["SESSION_COOKIE_SECURE"] = cfg.session_cookie_secure

    # --- Logging (structured-ish; never log secrets) ---
    app.logger.removeHandler(default_handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG if cfg.is_test() else logging.INFO)

    # Configure the root logger so all module loggers (src.services.* etc.)
    # also emit to the console at INFO level.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Quiet down noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # --- Extensions ---
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login_form"
    csrf.init_app(app)

    # --- Optional SocketIO (voice conversation) ---
    # Only initialized when VOICE_ENABLED and flask_socketio are both available.
    # The voice HTTP endpoints are registered via the voice blueprint regardless.
    socketio = None
    if cfg.voice_enabled:
        try:
            from flask_socketio import SocketIO

            cors = cfg.voice_cors_origins.split(",") if cfg.voice_cors_origins else None
            socketio = SocketIO(
                app,
                cors_allowed_origins=cors or "*",
                async_mode="threading",
                manage_session=False,
            )
            app.logger.info("Flask-SocketIO initialized for voice streaming.")
        except ImportError:
            app.logger.info(
                "VOICE_ENABLED but flask_socketio is not installed; "
                "voice routes work over HTTP only. "
                "Install flask-socketio for streaming."
            )
    app.extensions["socketio"] = socketio
    if socketio is not None:
        try:
            from .realtime import register_voice_namespace

            register_voice_namespace(socketio)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Could not register voice namespace: %s", exc)

    # --- Register blueprints (imported lazily to avoid circular imports) ---
    from .routes import register_blueprints

    register_blueprints(app)

    # --- Health endpoint ---
    @app.get("/health")
    def health() -> tuple[Any, int]:
        """Best-effort liveness probe (NFR-51)."""
        status: dict[str, str] = {"app": "ok"}
        http = 200

        # DB check.
        try:
            db.session.execute(db.text("SELECT 1"))
            status["db"] = "ok"
        except Exception as exc:  # noqa: BLE001
            status["db"] = "down"
            status["db_error"] = str(exc)
            http = 503

        # ChromaDB check (best-effort).
        try:
            from src.services.vector_store import get_vector_store

            vs = get_vector_store()
            status["chroma"] = "ok" if vs.backend else "degraded"
        except Exception as exc:  # noqa: BLE001
            status["chroma"] = "down"
            status["chroma_error"] = str(exc)

        # Ollama Cloud check (best-effort, short timeout).
        try:
            import requests

            r = requests.get(
                f"{cfg.ollama_cloud_base_url}/api/tags",
                headers={"Authorization": f"Bearer {cfg.ollama_cloud_api_key}"},
                timeout=5,
            )
            status["ollama_cloud"] = "ok" if r.status_code == 200 else "degraded"
        except Exception:  # noqa: BLE001
            status["ollama_cloud"] = "degraded"

        # Voice conversation status (observability).
        if cfg.voice_enabled:
            status["voice"] = "ok"
            try:
                from src.services.stt_service import get_stt_service

                get_stt_service()
                status["stt"] = "ok"
            except Exception:  # noqa: BLE001
                status["stt"] = "down"
        else:
            status["voice"] = "disabled"
            status["stt"] = "disabled"

        return jsonify(status), http

    app.logger.info("notebook-project app created (chat_model=%s)", cfg.chat_model)

    # --- Admin seeding command ---
    _register_seed_admin_command(app)

    # --- Non-test start guard for default admin password ---
    if not cfg.is_test() and cfg.admin_password == "change-me":  # noqa: S105
        app.logger.warning(
            "ADMIN_PASSWORD is still the default 'change-me'. "
            "Run `flask seed-admin` to set a real admin password before serving real traffic."
        )

    return app


def _register_seed_admin_command(app: Flask) -> None:
    """Register an idempotent ``flask seed-admin`` command."""

    @app.cli.command("seed-admin")
    def seed_admin() -> None:  # noqa: ANN202
        """Create or update the admin account from ADMIN_USERNAME/ADMIN_PASSWORD.

        Idempotent: if the admin already exists, the password is refreshed.
        Uses config values so seeding is consistent with documentation.
        """
        from src.extensions import db
        from src.models import User
        from src.services.auth_service import hash_password

        cfg = app.config["NOTEBOOK_CONFIG"]
        if cfg.admin_password == "change-me":  # noqa: S105
            app.logger.warning(
                "ADMIN_PASSWORD is 'change-me'. Set ADMIN_PASSWORD in .env "
                "or run `flask seed-admin`."
            )
        with app.app_context():
            existing = db.session.query(User).filter_by(username=cfg.admin_username).first()
            if existing is None:
                db.session.add(
                    User(
                        username=cfg.admin_username,
                        password_hash=hash_password(cfg.admin_password),
                        role="admin",
                    )
                )
                db.session.commit()
                app.logger.info("Seeded admin user %r.", cfg.admin_username)
            else:
                existing.password_hash = hash_password(cfg.admin_password)
                existing.role = "admin"
                db.session.commit()
                app.logger.info("Refreshed admin user %r password.", cfg.admin_username)
