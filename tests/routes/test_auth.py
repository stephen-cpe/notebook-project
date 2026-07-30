"""Route tests for src.routes.auth (TDD step 4).

Covers: signup success, duplicate username, login success, bad password,
logout, login-required redirect on protected routes, and admin role.
Tests run with CSRF disabled (``WTF_CSRF_ENABLED=false`` in test config).
"""

from __future__ import annotations

import pytest

from src.extensions import db
from src.models import User
from src.services.auth_service import (
    authenticate,
    hash_password,
    verify_password,
)
from src.services.auth_service import (
    signup as signup_service,
)
from src.services.exceptions import (
    DuplicateUsernameError,
    InvalidCredentialsError,
)

# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self) -> None:
        h = hash_password("s3cret")
        assert h != "s3cret"
        assert verify_password("s3cret", h) is True

    def test_wrong_password_fails(self) -> None:
        h = hash_password("s3cret")
        assert verify_password("wrong", h) is False

    def test_different_hashes_for_same_password(self) -> None:
        assert hash_password("pw") != hash_password("pw")


class TestSignupService:
    def test_signup_creates_user(self, app: object) -> None:
        with app.app_context():
            u = signup_service("alice", "pw123456")
            assert u.id is not None
            assert u.username == "alice"
            assert u.role == "user"
            assert u.password_hash != "pw123456"

    def test_signup_admin_role(self, app: object) -> None:
        with app.app_context():
            u = signup_service("boss", "pw123456", role="admin")
            assert u.role == "admin"

    def test_signup_duplicate_raises(self, app: object) -> None:
        with app.app_context():
            signup_service("bob", "pw123456")
            with pytest.raises(DuplicateUsernameError):
                signup_service("bob", "other123")

    def test_signup_empty_username_raises(self, app: object) -> None:
        with app.app_context(), pytest.raises(InvalidCredentialsError):
            signup_service("", "pw123456")

    def test_signup_empty_password_raises(self, app: object) -> None:
        with app.app_context(), pytest.raises(InvalidCredentialsError):
            signup_service("carol", "")


class TestAuthenticateService:
    def test_authenticate_success(self, app: object) -> None:
        with app.app_context():
            signup_service("dave", "pw123456")
            u = authenticate("dave", "pw123456")
            assert u.username == "dave"

    def test_authenticate_unknown_user(self, app: object) -> None:
        with app.app_context(), pytest.raises(InvalidCredentialsError):
            authenticate("nobody", "pw123456")

    def test_authenticate_wrong_password(self, app: object) -> None:
        with app.app_context():
            signup_service("eve", "pw123456")
            with pytest.raises(InvalidCredentialsError):
                authenticate("eve", "wrong")

    def test_authenticate_disabled_user(self, app: object) -> None:
        with app.app_context():
            u = signup_service("disauth", "pw123456")
            u.role = "disabled"
            db.session.commit()
            with pytest.raises(InvalidCredentialsError):
                authenticate("disauth", "pw123456")


# ---------------------------------------------------------------------------
# Route / HTTP tests
# ---------------------------------------------------------------------------


class TestSignupRoute:
    def test_signup_form_renders(self, client: object) -> None:
        res = client.get("/signup")
        assert res.status_code == 200
        assert b"Create your account" in res.data

    def test_successful_signup_redirects(self, client: object, app: object) -> None:
        res = client.post(
            "/signup",
            data={"username": "frank", "password": "pw123456"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)
        with app.app_context():
            assert db.session.query(User).filter_by(username="frank").count() == 1

    def test_duplicate_username_shows_error(self, client: object, app: object) -> None:
        with app.app_context():
            db.session.add(User(username="grace", password_hash=hash_password("pw123456")))
            db.session.commit()
        res = client.post(
            "/signup",
            data={"username": "grace", "password": "pw123456"},
        )
        assert res.status_code == 409
        assert b"already taken" in res.data

    def test_missing_fields(self, client: object) -> None:
        res = client.post("/signup", data={"username": "", "password": ""})
        assert res.status_code == 400
        assert b"required" in res.data


class TestLoginRoute:
    def test_login_form_renders(self, client: object) -> None:
        res = client.get("/login")
        assert res.status_code == 200
        assert b"Log in" in res.data

    def test_successful_login_redirects(self, client: object, app: object) -> None:
        with app.app_context():
            signup_service("heidi", "pw123456")
        res = client.post(
            "/login",
            data={"username": "heidi", "password": "pw123456"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)

    def test_bad_password(self, client: object, app: object) -> None:
        with app.app_context():
            signup_service("ivan", "pw123456")
        res = client.post(
            "/login",
            data={"username": "ivan", "password": "wrong"},
        )
        assert res.status_code == 401
        assert b"Invalid username or password" in res.data

    def test_unknown_user(self, client: object) -> None:
        res = client.post(
            "/login",
            data={"username": "ghost", "password": "pw123456"},
        )
        assert res.status_code == 401


class TestLogoutRoute:
    def test_logout_redirects_to_login(self, client: object, app: object) -> None:
        with app.app_context():
            signup_service("judy", "pw123456")
        client.post("/login", data={"username": "judy", "password": "pw123456"})
        res = client.get("/logout", follow_redirects=False)
        assert res.status_code in (301, 302, 303)


class TestLoginRequired:
    def test_notebooks_requires_login(self, client: object) -> None:
        res = client.get("/notebooks", follow_redirects=False)
        # Flask-Login redirects to login view when @login_required fails.
        assert res.status_code in (301, 302, 303)
        assert "/login" in res.headers.get("Location", "")

    def test_authenticated_user_can_access_notebooks(self, client: object, app: object) -> None:
        with app.app_context():
            signup_service("karl", "pw123456")
        client.post("/login", data={"username": "karl", "password": "pw123456"})
        res = client.get("/notebooks")
        assert res.status_code == 200
        assert b"My notebooks" in res.data


# ---------------------------------------------------------------------------
# Disabled-user behavior (P0-1.1)
# ---------------------------------------------------------------------------


class TestDisabledUser:
    """A user with role='disabled' cannot authenticate or keep a session."""

    def test_disabled_user_cannot_log_in(self, client: object, app: object) -> None:
        with app.app_context():
            u = signup_service("disuser", "pw123456")
            u.role = "disabled"
            db.session.commit()
        res = client.post(
            "/login",
            data={"username": "disuser", "password": "pw123456"},
            follow_redirects=False,
        )
        # Same error as bad credentials; no account-state leak.
        assert res.status_code == 401
        assert b"Invalid username or password" in res.data

    def test_existing_disabled_session_rejected(self, client: object, app: object) -> None:
        with app.app_context():
            u = signup_service("dissess", "pw123456")
            uid = u.id
        client.post("/login", data={"username": "dissess", "password": "pw123456"})
        with app.app_context():
            u = db.session.get(User, uid)
            assert u is not None
            u.role = "disabled"
            db.session.commit()
        # The existing session cookie must be rejected on next request.
        res = client.get("/notebooks", follow_redirects=False)
        assert res.status_code in (301, 302, 303), f"got {res.status_code}"
        assert "/login" in res.headers.get("Location", "")


# ---------------------------------------------------------------------------
# Password policy + reset-password (P0-1.6)
# ---------------------------------------------------------------------------


class TestPasswordPolicy:
    def test_short_password_rejected(self, app: object) -> None:
        with app.app_context(), pytest.raises(InvalidCredentialsError):
            signup_service("shortpw", "short")

    def test_long_password_rejected(self, app: object) -> None:
        with app.app_context(), pytest.raises(InvalidCredentialsError):
            signup_service("longpw", "x" * 257)

    def test_valid_length_accepted(self, app: object) -> None:
        with app.app_context():
            u = signup_service("okpw", "pw123456")
            assert u.id is not None


class TestResetPasswordRoute:
    def _login(self, client: object, app: object, username: str, password: str) -> None:
        with app.app_context():
            if db.session.query(User).filter_by(username=username).count() == 0:
                db.session.add(User(username=username, password_hash=hash_password(password)))
                db.session.commit()
        client.post("/login", data={"username": username, "password": password})

    def test_reset_requires_current_password(self, client: object, app: object) -> None:
        self._login(client, app, "reset1", "pw123456")
        res = client.post(
            "/reset-password",
            data={"current_password": "WRONG", "new_password": "newpass12"},
            follow_redirects=False,
        )
        # Stays on the reset page (302 to /reset-password) with a flash error.
        assert res.status_code in (301, 302, 303)
        # Confirm password unchanged.
        with app.app_context():
            u = db.session.query(User).filter_by(username="reset1").one()
            assert verify_password("pw123456", u.password_hash)

    def test_reset_succeeds_with_correct_current(self, client: object, app: object) -> None:
        self._login(client, app, "reset2", "pw123456")
        res = client.post(
            "/reset-password",
            data={"current_password": "pw123456", "new_password": "newpass12"},
            follow_redirects=False,
        )
        assert res.status_code in (301, 302, 303)
        with app.app_context():
            u = db.session.query(User).filter_by(username="reset2").one()
            assert verify_password("newpass12", u.password_hash)
            assert not verify_password("pw123456", u.password_hash)

    def test_reset_rejects_short_new_password(self, client: object, app: object) -> None:
        self._login(client, app, "reset3", "pw123456")
        client.post(
            "/reset-password",
            data={"current_password": "pw123456", "new_password": "short"},
            follow_redirects=False,
        )
        with app.app_context():
            u = db.session.query(User).filter_by(username="reset3").one()
            assert verify_password("pw123456", u.password_hash)
