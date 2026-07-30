"""Shared route helpers — owner scoping + current-user notebook resolution."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import abort
from flask_login import current_user

from src.models import Notebook
from src.repositories import notebook_repo


def require_owner(notebook_id: int) -> Notebook:
    """Return the notebook if owned by the current user, else 404.

    We return 404 (not 403) to avoid leaking the existence of other users'
    notebooks (FR-3, NFR-22).
    """
    notebook = notebook_repo.get_by_id(notebook_id)
    if notebook is None:
        abort(404)
    if notebook.user_id != _current_user_id():
        abort(404)
    return notebook


def require_admin(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: abort 403 if the current user is not an admin."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)

    return wrapper


def _current_user_id() -> int:
    """Return the current user's ID as an int."""
    uid = current_user.get_id()
    return int(uid) if uid else 0
