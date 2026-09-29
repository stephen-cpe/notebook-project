"""Figure persistence — source-image thumbnails grounded to content hashes.

Ported from study-and-learn ``figure_store.py``: when vision describes a
figure (or an image is uploaded directly), the pixels are kept under
``DATA_DIR/figures/<content_hash>/`` with a ``manifest.json`` so chat can
show thumbnails next to citations. Never raises; all failures degrade to
``None`` / empty lists.

Layout per content hash::
    figures/<hash>/fig-01.png
    figures/<hash>/fig-01.json   (caption/label/mermaid/diagram_type/confidence)
    figures/<hash>/manifest.json (list of entries)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FIGURE_MAX_DIMENSION = 1024
FIGURES_PER_SOURCE = 2
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _figures_root(data_dir: str = "./data") -> Path:
    return Path(data_dir) / "figures"


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or f"fig-{uuid.uuid4().hex[:8]}.png"


def save_figure(
    content_hash: str,
    image: Any,  # noqa: ANN401 — PIL Image, path, or bytes
    caption: str = "",
    label: str = "",
    data_dir: str = "./data",
    mermaid: str = "",
    diagram_type: str = "",
    confidence: int = 0,
) -> str | None:
    """Persist a figure image + sidecar metadata. Returns filename or None."""
    try:
        if not content_hash:
            return None
        root = _figures_root(data_dir) / content_hash
        root.mkdir(parents=True, exist_ok=True)
        existing = _read_manifest(root)
        if len(existing) >= 50:
            logger.warning("Figure store full for hash %s; skipping", content_hash[:12])
            return None
        fname = f"fig-{len(existing) + 1:02d}.png"
        fpath = root / fname
        _write_image(image, fpath)
        entry = {
            "filename": fname,
            "caption": caption or "",
            "label": label or "",
            "mermaid": mermaid or "",
            "diagram_type": diagram_type or "",
            "confidence": int(confidence or 0),
        }
        (root / f"{fname}.json").write_text(json.dumps(entry), encoding="utf-8")
        existing.append(entry)
        (root / "manifest.json").write_text(json.dumps(existing), encoding="utf-8")
        return fname
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_figure failed for %s: %s", content_hash[:12], exc)
        return None


def _write_image(image: Any, dest: Path) -> None:  # noqa: ANN401
    """Write a PIL Image / path / bytes to ``dest`` as PNG (downscaled)."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)) and Path(image).exists():
        img = Image.open(image).convert("RGB")
    elif isinstance(image, (bytes, bytearray)):
        import io

        img = Image.open(io.BytesIO(bytes(image))).convert("RGB")
    elif hasattr(image, "save"):
        img = image if image.mode == "RGB" else image.convert("RGB")
    else:
        raise ValueError("Unsupported figure image type")
    w, h = img.size
    if max(w, h) > FIGURE_MAX_DIMENSION:
        ratio = FIGURE_MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)))
    img.save(dest, format="PNG")


def _read_manifest(root: Path) -> list[dict[str, Any]]:
    try:
        raw = (root / "manifest.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def get_figures(content_hash: str, data_dir: str = "./data") -> list[dict[str, Any]]:
    """Return manifest entries for a content hash (traversal-guarded)."""
    if not content_hash or "/" in content_hash or "\\" in content_hash or ".." in content_hash:
        return []
    return _read_manifest(_figures_root(data_dir) / content_hash)


def figure_url(notebook_id: int, content_hash: str, filename: str) -> str:
    """Return the owner-scoped URL for a stored figure."""
    safe = _safe_filename(filename)
    return f"/notebooks/{notebook_id}/figures/{content_hash}/{safe}"


def attach_figures(
    sources: list[dict[str, Any]],
    content_hashes: dict[str, str] | None = None,
    notebook_id: int = 0,
    data_dir: str = "./data",
    max_per_source: int = FIGURES_PER_SOURCE,
) -> list[dict[str, Any]]:
    """Attach up to ``max_per_source`` figure thumbnails per source.

    Args:
        sources: citation dicts with at least ``filename``.
        content_hashes: optional ``filename -> content_hash`` map. When
            omitted, figures cannot be resolved and sources pass through.
        notebook_id: used to build owner-scoped URLs.
    """
    if not sources or not content_hashes:
        return sources
    try:
        hash_by_filename = dict(content_hashes)
        out: list[dict[str, Any]] = []
        for src in sources:
            entry = dict(src)
            fname = str(src.get("filename", ""))
            chash = hash_by_filename.get(fname, "")
            figs = get_figures(chash, data_dir)[:max_per_source] if chash else []
            attached = []
            for f in figs:
                attached.append(
                    {
                        "url": figure_url(notebook_id, chash, str(f.get("filename", ""))),
                        "caption": str(f.get("caption", "")),
                        "label": str(f.get("label", "")),
                        "diagram_type": str(f.get("diagram_type", "")),
                    }
                )
            if attached:
                entry["figures"] = attached
            out.append(entry)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("attach_figures failed: %s", exc)
        return sources
