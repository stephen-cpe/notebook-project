"""Index route — landing page."""

from __future__ import annotations

from flask import Blueprint, redirect, url_for
from werkzeug.wrappers import Response

index_bp = Blueprint("index", __name__)


@index_bp.get("/")
def index() -> Response:
    """Redirect to the notebooks list (login required downstream)."""
    return redirect(url_for("notebooks.list_notebooks"))
