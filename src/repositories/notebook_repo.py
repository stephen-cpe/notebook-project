"""Notebook repository — DB access for Notebook (no business logic)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from src.extensions import db
from src.models import Notebook


def create_notebook(user_id: int, name: str, description: str | None = None) -> Notebook:
    """Insert a new notebook and return it."""
    nb = Notebook(user_id=user_id, name=name, description=description)
    db.session.add(nb)
    db.session.commit()
    return nb


def get_by_id(notebook_id: int) -> Notebook | None:
    """Fetch a notebook by primary key."""
    return db.session.get(Notebook, notebook_id)


def list_by_user(user_id: int) -> Sequence[Notebook]:
    """Return all notebooks owned by ``user_id``, newest first."""
    return db.session.scalars(
        select(Notebook).where(Notebook.user_id == user_id).order_by(Notebook.created_at.desc())
    ).all()


def update_notebook(
    notebook: Notebook, name: str | None = None, description: str | None = None
) -> Notebook:
    """Update name and/or description. Only non-None fields are changed."""
    if name is not None:
        notebook.name = name
    if description is not None:
        notebook.description = description
    db.session.commit()
    return notebook


def delete_notebook(notebook: Notebook) -> None:
    """Delete a notebook (cascades to sources + chat messages)."""
    db.session.delete(notebook)
    db.session.commit()
