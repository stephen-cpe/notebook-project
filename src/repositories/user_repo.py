"""User repository — DB access for User (no business logic)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from src.extensions import db
from src.models import User


def create_user(username: str, password_hash: str, role: str = "user") -> User:
    """Insert a new user and return it."""
    user = User(username=username, password_hash=password_hash, role=role)
    db.session.add(user)
    db.session.commit()
    return user


def get_by_id(user_id: int) -> User | None:
    """Fetch a user by primary key."""
    return db.session.get(User, user_id)


def get_by_username(username: str) -> User | None:
    """Fetch a user by username (case-sensitive)."""
    return db.session.scalar(select(User).where(User.username == username))


def list_all() -> Sequence[User]:
    """Return all users (admin view)."""
    return db.session.scalars(select(User).order_by(User.created_at)).all()
