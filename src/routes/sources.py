"""Source upload/list/delete/text routes, scoped to a notebook."""

from __future__ import annotations

from flask import (
    Blueprint,
    Response,
    abort,
    jsonify,
    request,
)
from flask_login import login_required

from src.config import Config
from src.repositories import source_repo
from src.routes._helpers import require_owner
from src.services.document_parser import detect_content_type
from src.services.exceptions import IngestionError

sources_bp = Blueprint("sources", __name__)


def _get_config() -> Config:
    """Return the app's Config from the Flask app context."""
    from flask import current_app

    cfg: Config = current_app.config["NOTEBOOK_CONFIG"]
    return cfg


@sources_bp.get("/notebooks/<int:notebook_id>/sources")
@login_required
def list_sources(notebook_id: int) -> tuple[Response, int]:
    """List all sources for a notebook (JSON)."""
    require_owner(notebook_id)
    sources = source_repo.list_by_notebook(notebook_id)
    return (
        jsonify(
            sources=[
                {
                    "id": s.id,
                    "filename": s.filename,
                    "content_type": s.content_type,
                    "status": s.status,
                    "char_count": s.char_count,
                    "page_count": s.page_count,
                    "error_message": s.error_message,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in sources
            ]
        ),
        200,
    )


@sources_bp.post("/notebooks/<int:notebook_id>/sources")
@login_required
def upload_source(notebook_id: int) -> tuple[Response, int]:
    """Upload a file to a notebook and trigger ingestion."""
    require_owner(notebook_id)
    cfg = _get_config()

    # Check source cap.
    existing_count = source_repo.count_by_notebook(notebook_id)
    if existing_count >= cfg.max_sources_per_notebook:
        return (
            jsonify(error=f"Source limit reached ({cfg.max_sources_per_notebook})."),
            400,
        )

    if "file" not in request.files:
        return jsonify(error="No file provided."), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify(error="No file selected."), 400

    # Sanitize filename (NFR-24): basename only, no path traversal.
    from pathlib import Path as PathLib

    safe_filename = PathLib(file.filename).name
    if not safe_filename or safe_filename in (".", ".."):
        return jsonify(error="Invalid filename."), 400

    # Validate type by extension.
    try:
        content_type = detect_content_type(safe_filename)
    except IngestionError:
        return jsonify(error=f"Unsupported file type: {safe_filename}"), 400

    # Validate size.
    file.seek(0, 2)  # seek to end
    size = file.tell()
    file.seek(0)
    max_bytes = cfg.max_file_size_mb * 1024 * 1024
    if size > max_bytes:
        return jsonify(error=f"File too large (max {cfg.max_file_size_mb} MB)."), 400

    # Save to temp file and ingest.
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_filename}") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Validate file content by magic bytes (NFR-23).
    from src.services.document_parser import validate_magic_bytes

    if not validate_magic_bytes(tmp_path, content_type):
        Path(tmp_path).unlink(missing_ok=True)
        return jsonify(error=f"File content does not match its extension: {safe_filename}"), 400

    # Compute hash + check for duplicate source in this notebook.
    from src.services.ingestion import compute_hash

    content_hash = compute_hash(tmp_path)
    existing = source_repo.get_by_notebook_and_hash(notebook_id, content_hash)
    if existing is not None:
        # Re-ingest the same file (ChromaDB dedup will skip re-embedding).
        from src.services.ingestion import get_ingestion_service

        ingest_svc = get_ingestion_service()
        result = ingest_svc.ingest_file(tmp_path, filename=safe_filename)
        source_repo.update_status(
            existing,
            status=result.status,
            char_count=result.char_count,
            page_count=result.page_count,
            error_message=result.error_message,
        )
        Path(tmp_path).unlink(missing_ok=True)
        return (
            jsonify(
                source_id=existing.id,
                filename=existing.filename,
                content_hash=result.content_hash,
                status=result.status,
                char_count=result.char_count,
                page_count=result.page_count,
                ocr_used=result.ocr_used,
                error=result.error_message,
                message="File already uploaded — re-ingested.",
            ),
            200,
        )

    source = source_repo.create_source(
        notebook_id=notebook_id,
        filename=safe_filename,
        content_hash=content_hash,
        content_type=content_type,
    )

    # Run ingestion synchronously.
    from src.services.ingestion import get_ingestion_service

    ingest_svc = get_ingestion_service()
    result = ingest_svc.ingest_file(tmp_path, filename=safe_filename)

    # Update source row with results.
    source_repo.update_status(
        source,
        status=result.status,
        char_count=result.char_count,
        page_count=result.page_count,
        error_message=result.error_message,
    )

    # Clean up temp file.
    Path(tmp_path).unlink(missing_ok=True)

    return (
        jsonify(
            source_id=source.id,
            filename=source.filename,
            content_hash=result.content_hash,
            status=result.status,
            char_count=result.char_count,
            page_count=result.page_count,
            ocr_used=result.ocr_used,
            error=result.error_message,
        ),
        201,
    )


@sources_bp.delete("/notebooks/<int:notebook_id>/sources/<int:source_id>")
@login_required
def delete_source(notebook_id: int, source_id: int) -> tuple[Response, int]:
    """Delete a source from a notebook.

    Reference-counted cleanup: if no other Source row references the same
    content_hash, the underlying ChromaDB collection and ContentRegistry entry
    are also removed so content/embeddings don't leak.
    """
    require_owner(notebook_id)
    source = source_repo.get_by_id(source_id)
    if source is None or source.notebook_id != notebook_id:
        abort(404)
    content_hash = source.content_hash
    source_repo.delete_source(source)
    # Best-effort cleanup of orphaned shared content.
    from src.services.cleanup_service import cleanup_orphaned_content

    cleanup_orphaned_content(content_hash, exclude_source_id=source_id)
    return jsonify(ok=True), 200


@sources_bp.get("/notebooks/<int:notebook_id>/sources/<int:source_id>/text")
@login_required
def source_text(notebook_id: int, source_id: int) -> tuple[Response, int]:
    """Return the extracted text for a source (from ContentRegistry)."""
    require_owner(notebook_id)
    source = source_repo.get_by_id(source_id)
    if source is None or source.notebook_id != notebook_id:
        abort(404)

    from src.repositories import content_registry_repo

    entry = content_registry_repo.get_by_hash(source.content_hash)
    if entry is None:
        return jsonify(text="", error="No cached text available."), 404
    return jsonify(text=entry.extracted_text, char_count=entry.char_count), 200


@sources_bp.patch("/notebooks/<int:notebook_id>/sources/<int:source_id>/rename")
@login_required
def rename_source(notebook_id: int, source_id: int) -> tuple[Response, int]:
    """Rename a source's display filename (does not re-ingest)."""
    require_owner(notebook_id)
    source = source_repo.get_by_id(source_id)
    if source is None or source.notebook_id != notebook_id:
        abort(404)

    data = request.get_json(silent=True) or {}
    new_filename = (data.get("filename") or "").strip()
    if not new_filename:
        return jsonify(error="Filename is required."), 400

    # Sanitize the new filename (NFR-24): basename only, no path traversal.
    from pathlib import Path as PathLib

    safe_name = PathLib(new_filename).name
    if not safe_name or safe_name in (".", ".."):
        return jsonify(error="Invalid filename."), 400

    source_repo.rename_source(source, safe_name)
    return jsonify(ok=True, filename=safe_name), 200
