"""Diagram → Mermaid reinterpretation (ported from pdf2md, simplified).

For image figures, ask the vision model whether the figure can be faithfully
re-expressed as Mermaid source. Convertible types (flowcharts, sequences,
class/ER, state, mindmap, timeline, gantt, pie, etc.) return validated
Mermaid; photos/artwork/maps/logos return ``convertible=False`` and the
caller keeps the original image. Never invents nodes/labels.

Validation is structural (non-empty, allowlisted type, no envelope markers,
non-empty body) plus an optional vision re-verify pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.services.llm_json import extract_json

logger = logging.getLogger(__name__)

ALLOWED_DIAGRAM_TYPES = frozenset(
    {
        "flowchart",
        "sequenceDiagram",
        "classDiagram",
        "stateDiagram-v2",
        "erDiagram",
        "gantt",
        "mindmap",
        "timeline",
        "journey",
        "pie",
        "gitGraph",
        "quadrantChart",
        "xychart-beta",
        "sankey-beta",
        "architecture-beta",
        "radar-beta",
        "kanban",
    }
)

_TYPE_ALIASES = {"graph": "flowchart"}

DIAGRAM_SYSTEM_PROMPT = (
    "You are a diagram-to-Mermaid reinterpreter. You are shown ONE cropped "
    "figure region. Decide whether it can be faithfully re-expressed as "
    "Mermaid source. Convertible: flowcharts, sequence, class/ER, state, "
    "mindmap, timeline, gantt, quadrant, pie/xychart/sankey/radar. NOT "
    "convertible: photo, artwork, map, logo, chemical structure, hand-drawn "
    "schematic, pixel-dependent image. NEVER invent nodes, labels, numbers, "
    "or relationships. Respond as JSON: "
    '{"convertible": true|false, "type": "<mermaid type or empty>", '
    '"confidence": 0-100, "description": "<one sentence>", '
    '"mermaid": "<complete Mermaid source, no fences, empty when false>"}'
)


@dataclass
class DiagramResult:
    """Outcome of attempting a figure → Mermaid conversion."""

    convertible: bool
    diagram_type: str = ""
    confidence: int = 0
    description: str = ""
    mermaid: str = ""
    reason: str = ""


def detect_mermaid_type(source: str) -> str:
    """Return the Mermaid diagram type (first non-comment line, aliased)."""
    for line in (source or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        first = stripped.split()[0].rstrip("{[(")
        return _TYPE_ALIASES.get(first, first)
    return ""


def validate_mermaid(source: str) -> bool:
    """Structural validation: non-empty, allowlisted type, no markers."""
    if not source or not source.strip():
        return False
    if "<<<" in source or "<!--FIG:" in source:
        return False
    dtype = detect_mermaid_type(source)
    if dtype not in ALLOWED_DIAGRAM_TYPES:
        return False
    body = "\n".join(
        ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("%%")
    )
    return len(body.strip().splitlines()) >= 2


def convert_figure(
    image: Any,  # noqa: ANN401 — PIL Image
    min_confidence: int = 80,
    verify: bool = True,
) -> DiagramResult:
    """Attempt figure → Mermaid conversion via the vision model.

    Never raises: any failure yields ``convertible=False`` with a reason so
    callers keep the original image.
    """
    try:
        from src.config import Config
        from src.services.ocr_service import _pil_to_b64
        from src.services.ollama_client import get_ollama_client

        cfg = Config()
        if cfg.ai_mock:
            return DiagramResult(convertible=False, reason="mock mode")
        if not cfg.diagram_to_mermaid:
            return DiagramResult(convertible=False, reason="disabled")
        b64 = _pil_to_b64(image)
        if not b64:
            return DiagramResult(convertible=False, reason="unencodable image")
        client = get_ollama_client()
        raw = client.chat_with_images(DIAGRAM_SYSTEM_PROMPT, [b64], model=cfg.vision_model)
        data = extract_json(raw)
        if not isinstance(data, dict):
            return DiagramResult(convertible=False, reason="unparseable response")
        convertible = bool(data.get("convertible", False))
        dtype = str(data.get("type", "") or "")
        try:
            confidence = int(data.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0
        description = str(data.get("description", "") or "")
        mermaid = str(data.get("mermaid", "") or "")
        mermaid = _strip_fences(mermaid)
        if not convertible:
            return DiagramResult(
                convertible=False,
                diagram_type=dtype,
                confidence=confidence,
                description=description,
                reason="model marked non-convertible",
            )
        if confidence < min_confidence:
            return DiagramResult(
                convertible=False,
                diagram_type=dtype,
                confidence=confidence,
                description=description,
                reason=f"confidence {confidence} < {min_confidence}",
            )
        if not validate_mermaid(mermaid):
            return DiagramResult(
                convertible=False,
                diagram_type=dtype,
                confidence=confidence,
                description=description,
                reason="structural validation failed",
            )
        if verify and cfg.diagram_verify and not _verify_mermaid(image, mermaid):
            return DiagramResult(
                convertible=False,
                diagram_type=dtype,
                confidence=confidence,
                description=description,
                reason="vision re-verify failed",
            )
        return DiagramResult(
            convertible=True,
            diagram_type=dtype,
            confidence=confidence,
            description=description,
            mermaid=mermaid,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diagram conversion failed: %s", exc)
        return DiagramResult(convertible=False, reason=str(exc)[:200])


def _verify_mermaid(image: Any, mermaid: str) -> bool:  # noqa: ANN401
    """Best-effort vision re-verify: does the Mermaid match the figure?"""
    try:
        from src.config import Config
        from src.services.ocr_service import _pil_to_b64
        from src.services.ollama_client import get_ollama_client

        cfg = Config()
        b64 = _pil_to_b64(image)
        if not b64:
            return False
        client = get_ollama_client()
        prompt = (
            "You are a strict Mermaid fidelity judge. Compare the candidate "
            "Mermaid source against the figure image. Reply with exactly one "
            f"word: pass or retry.\n\nMermaid:\n{mermaid}"
        )
        verdict = client.chat_with_images(prompt, [b64], model=cfg.vision_model)
        text = verdict.strip().lower()
        if "pass" in text and "retry" not in text.split("pass")[0][-20:]:
            return "pass" in text
        return text.startswith("pass")
    except Exception:  # noqa: BLE001
        return False


def _strip_fences(source: str) -> str:
    """Remove ```mermaid fences if the model added them."""
    text = (source or "").strip()
    m = re.match(r"^```(?:mermaid)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text
