"""Authentication routes: /signup, /login, /logout, /settings, /reset-password."""

from __future__ import annotations

from typing import Any

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.wrappers import Response as WerkzeugResponse

from src.extensions import db
from src.services.auth_service import (
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    authenticate,
    hash_password,
    to_auth_user,
    verify_password,
)
from src.services.auth_service import signup as signup_service
from src.services.exceptions import (
    AuthError,
    DuplicateUsernameError,
    InvalidCredentialsError,
)

auth_bp = Blueprint("auth", __name__)

ALLOWED_AVATARS = [
    "avatar-0.png",
    "avatar-1.png",
    "avatar-2.png",
    "avatar-3.png",
    "avatar-4.png",
    "avatar-5.png",
    "avatar-6.png",
    "avatar-7.png",
    "avatar-8.png",
]

ALLOWED_SPEAKERS = ["Ava", "Andrew", "Emma", "Ryan"]

# Type alias for Flask view return values (werkzeug base covers both flask + redirects).
ViewReturn = WerkzeugResponse | str | tuple[Any, int]


@auth_bp.get("/signup")
def signup_form() -> ViewReturn:
    if current_user.is_authenticated:
        return redirect(url_for("notebooks.list_notebooks"))
    return render_template("auth/signup.html")


@auth_bp.post("/signup")
def signup() -> ViewReturn:
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if not username or not password:
        return _render_signup_error("Username and password are required"), 400
    try:
        user = signup_service(username, password)
    except InvalidCredentialsError as exc:
        # Includes password-policy violations (P0-1.6).
        return _render_signup_error(str(exc)), 400
    except DuplicateUsernameError:
        return _render_signup_error("That username is already taken"), 409
    except AuthError:
        return _render_signup_error("Could not create account"), 400
    login_user(to_auth_user(user))
    return redirect(url_for("notebooks.list_notebooks"))


@auth_bp.get("/login")
def login_form() -> ViewReturn:
    if current_user.is_authenticated:
        return redirect(url_for("notebooks.list_notebooks"))
    return render_template("auth/login.html")


@auth_bp.post("/login")
def login() -> ViewReturn:
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if not username or not password:
        return _render_login_error("Username and password are required"), 400
    try:
        user = authenticate(username, password)
    except InvalidCredentialsError:
        return _render_login_error("Invalid username or password"), 401
    except AuthError:
        return _render_login_error("Authentication failed"), 400
    login_user(to_auth_user(user))
    return redirect(url_for("notebooks.list_notebooks"))


@auth_bp.get("/logout")
def logout() -> WerkzeugResponse:
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login_form"))


@auth_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings() -> ViewReturn:
    if request.method == "POST":
        user = current_user.get_underlying()
        changed = False

        avatar = request.form.get("avatar", "").strip()
        if avatar in ALLOWED_AVATARS and avatar != user.avatar:
            user.avatar = avatar
            changed = True

        speaker_a = request.form.get("audio_speaker_a", "").strip()
        if speaker_a in ALLOWED_SPEAKERS and speaker_a != user.audio_speaker_a:
            user.audio_speaker_a = speaker_a
            changed = True

        speaker_b = request.form.get("audio_speaker_b", "").strip()
        if speaker_b in ALLOWED_SPEAKERS and speaker_b != user.audio_speaker_b:
            user.audio_speaker_b = speaker_b
            changed = True

        video_spk = request.form.get("video_speaker", "").strip()
        if video_spk in ALLOWED_SPEAKERS and video_spk != user.video_speaker:
            user.video_speaker = video_spk
            changed = True

        voice_spk = request.form.get("voice_speaker", "").strip()
        if voice_spk in ALLOWED_SPEAKERS and voice_spk != user.voice_speaker:
            user.voice_speaker = voice_spk
            changed = True

        if changed:
            db.session.commit()
            flash("Settings updated.", "success")
        else:
            flash("Settings are up to date.", "info")
        return redirect(url_for("auth.settings"))

    from flask import current_app

    cfg = current_app.config["NOTEBOOK_CONFIG"]
    return render_template(
        "settings.html",
        avatars=ALLOWED_AVATARS,
        speakers=ALLOWED_SPEAKERS,
        voice_enabled=cfg.voice_enabled,
    )


@auth_bp.route("/reset-password", methods=["GET", "POST"])
@login_required
def reset_password() -> ViewReturn:
    if request.method == "POST":
        current_password = request.form.get("current_password") or ""
        new_password = (request.form.get("new_password") or "").strip()
        if not new_password:
            flash("New password is required.", "error")
            return redirect(url_for("auth.reset_password"))
        if len(new_password) < PASSWORD_MIN_LENGTH:
            flash(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", "error")
            return redirect(url_for("auth.reset_password"))
        if len(new_password) > PASSWORD_MAX_LENGTH:
            flash(f"Password must be at most {PASSWORD_MAX_LENGTH} characters.", "error")
            return redirect(url_for("auth.reset_password"))

        # Require the current password to authorize the change (P0-1.6).
        user = current_user.get_underlying()
        if not verify_password(current_password, user.password_hash):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("auth.reset_password"))

        user.password_hash = hash_password(new_password)
        db.session.commit()
        flash("Your password has been updated.", "success")
        return redirect(url_for("notebooks.list_notebooks"))

    return render_template("reset_password.html")


def _render_signup_error(message: str) -> str:
    flash(message, category="error")
    return render_template("auth/signup.html")


def _render_login_error(message: str) -> str:
    flash(message, category="error")
    return render_template("auth/login.html")


# Re-export for type checkers / tests that import the mixin.
__all__ = ["auth_bp", "ViewReturn"]
