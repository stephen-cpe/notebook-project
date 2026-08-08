"""Authentication service — password hashing, login validation, user lookup.

Business logic only (no HTTP). Routes call these functions and translate
``AuthError`` subclasses into HTTP responses.
"""

from __future__ import annotations

import logging

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from src.extensions import login_manager
from src.models import ROLE_DISABLED, User
from src.repositories import user_repo
from src.services.exceptions import (
    DuplicateUsernameError,
    InvalidCredentialsError,
)

logger = logging.getLogger(__name__)

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 256


def hash_password(password: str) -> str:
    """Return a scrypt-based password hash (werkzeug)."""
    return generate_password_hash(password, method="scrypt")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify ``password`` against the stored hash (constant-time internally)."""
    return bool(check_password_hash(password_hash, password))


def signup(username: str, password: str, role: str = "user") -> User:
    """Create a new user.

    Raises:
        DuplicateUsernameError: if ``username`` is already taken or empty.
        InvalidCredentialsError: if username/password are missing or the
            password violates the length policy.
    """
    if not username or not password:
        raise InvalidCredentialsError("Username and password are required")
    _validate_password_policy(password)
    if user_repo.get_by_username(username) is not None:
        raise DuplicateUsernameError(f"Username {username!r} is already taken")
    user = user_repo.create_user(username, hash_password(password), role=role)
    logger.info("User signed up: id=%s username=%s role=%s", user.id, user.username, user.role)
    return user


def _validate_password_policy(password: str) -> None:
    """Enforce minimum/maximum password length.

    Raises:
        InvalidCredentialsError: if the password is shorter than
            ``PASSWORD_MIN_LENGTH`` or longer than ``PASSWORD_MAX_LENGTH``.
    """
    if len(password) < PASSWORD_MIN_LENGTH:
        raise InvalidCredentialsError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
        )
    if len(password) > PASSWORD_MAX_LENGTH:
        raise InvalidCredentialsError(f"Password must be at most {PASSWORD_MAX_LENGTH} characters.")


def authenticate(username: str, password: str) -> User:
    """Validate credentials and return the user.

    Disabled users (``role == "disabled"``) are rejected with the same
    ``InvalidCredentialsError`` used for bad credentials so that account
    state is not leaked through the login flow.

    Raises:
        InvalidCredentialsError: if username is unknown, the password is
            wrong, or the account is disabled.
    """
    user = user_repo.get_by_username(username)
    if user is None:
        # Run a dummy hash check to keep timing roughly constant.
        verify_password(password, hash_password("dummy"))
        raise InvalidCredentialsError("Invalid username or password")
    if not verify_password(password, user.password_hash):
        raise InvalidCredentialsError("Invalid username or password")
    if user.role == ROLE_DISABLED:
        # Same error as bad credentials to avoid leaking account state.
        raise InvalidCredentialsError("Invalid username or password")
    return user


class AuthUser(UserMixin):  # type: ignore[misc]
    """Flask-Login wrapper around ``User``.

    Flask-Login requires a UserMixin with ``id`` and ``is_authenticated``. We
    proxy to the underlying ``User`` row so the DB stays the source of truth.
    """

    def __init__(self, user: User) -> None:
        self._user = user

    def get_id(self) -> str:
        return str(self._user.id)

    @property
    def is_active(self) -> bool:
        """False when the account is disabled."""
        return self._user.role != ROLE_DISABLED

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def username(self) -> str:
        return self._user.username

    @property
    def role(self) -> str:
        return self._user.role

    @property
    def is_admin(self) -> bool:
        return self._user.role == "admin"

    @property
    def avatar(self) -> str:
        return self._user.avatar

    @property
    def audio_speaker_a(self) -> str:
        return self._user.audio_speaker_a

    @property
    def audio_speaker_b(self) -> str:
        return self._user.audio_speaker_b

    @property
    def video_speaker(self) -> str:
        return self._user.video_speaker

    @property
    def voice_speaker(self) -> str:
        return self._user.voice_speaker

    def get_underlying(self) -> User:
        """Return the underlying ``User`` ORM object."""
        return self._user


def to_auth_user(user: User) -> AuthUser:
    """Wrap an ORM ``User`` in an ``AuthUser`` for Flask-Login."""
    return AuthUser(user)


@login_manager.user_loader
def load_user(user_id: str) -> AuthUser | None:
    """Flask-Login callback: load the ``AuthUser`` for the session.

    Returns ``None`` for disabled users so existing sessions are invalidated
    on the next request. Returning None makes Flask-Login treat the user as
    anonymous, so ``@login_required`` redirects to the login view.
    """
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    user = user_repo.get_by_id(uid)
    if user is None:
        return None
    if user.role == ROLE_DISABLED:
        return None
    return to_auth_user(user)
