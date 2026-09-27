"""Admin routes — user management dashboard (FR-4, admin-only)."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, flash, jsonify, redirect, render_template, url_for
from flask_login import login_required
from sqlalchemy import func

from src.extensions import db
from src.models import Notebook, Source
from src.repositories import user_repo
from src.routes._helpers import require_admin

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.get("/")
@login_required
@require_admin
def dashboard() -> str:
    """Admin dashboard — list users with notebook/source counts."""
    users = user_repo.list_all()
    # Aggregate counts in two queries instead of two per user (N+1).
    nb_counts: dict[int, int] = {
        row[0]: row[1]
        for row in db.session.query(Notebook.user_id, func.count(Notebook.id))
        .group_by(Notebook.user_id)
        .all()
    }
    src_counts: dict[int, int] = {
        row[0]: row[1]
        for row in db.session.query(Notebook.user_id, func.count(Source.id))
        .join(Source, Source.notebook_id == Notebook.id)
        .group_by(Notebook.user_id)
        .all()
    }
    user_data: list[dict[str, Any]] = []
    for u in users:
        nb_count = nb_counts.get(u.id, 0)
        src_count = src_counts.get(u.id, 0)
        user_data.append(
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "notebook_count": nb_count,
                "source_count": src_count,
                "created_at": u.created_at,
            }
        )
    return render_template("admin/dashboard.html", users=user_data)


@admin_bp.get("/users")
@login_required
@require_admin
def list_users() -> tuple[Response, int]:
    """List all users as JSON (API endpoint)."""
    users = user_repo.list_all()
    return (
        jsonify(
            users=[
                {
                    "id": u.id,
                    "username": u.username,
                    "role": u.role,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                }
                for u in users
            ]
        ),
        200,
    )


@admin_bp.post("/users/<int:user_id>/disable")
@login_required
@require_admin
def disable_user(user_id: int) -> Any:  # noqa: ANN401
    """Disable a user by changing their role to 'disabled' (admin only)."""
    user = user_repo.get_by_id(user_id)
    if user is None:
        flash("User not found.", "error")
        return redirect(url_for("admin.dashboard"))
    if user.role == "admin":
        flash("Cannot disable an admin user.", "error")
        return redirect(url_for("admin.dashboard"))
    user.role = "disabled"
    db.session.commit()
    flash(f"User '{user.username}' has been disabled.", "success")
    return redirect(url_for("admin.dashboard"))


@admin_bp.post("/users/<int:user_id>/enable")
@login_required
@require_admin
def enable_user(user_id: int) -> Any:  # noqa: ANN401
    """Re-enable a disabled user (admin only)."""
    user = user_repo.get_by_id(user_id)
    if user is None:
        flash("User not found.", "error")
        return redirect(url_for("admin.dashboard"))
    if user.role != "disabled":
        flash("User is not disabled.", "error")
        return redirect(url_for("admin.dashboard"))
    user.role = "user"
    db.session.commit()
    flash(f"User '{user.username}' has been enabled.", "success")
    return redirect(url_for("admin.dashboard"))
