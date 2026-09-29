"""Opt-in web search for suggested questions (fail-closed, ported).

Web search fires ONLY for public topics when internal doc questions run
out — proprietary docs never touch the network. Reuses the existing
``OLLAMA_CLOUD_*`` credentials (no new secret). All failures degrade to
``[]`` (never raise).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_BLOCKLIST = (
    "salary",
    "payroll",
    "hr",
    "confidential",
    "proprietary",
    "internal",
    "password",
    "ssn",
    "medical record",
)

_ALLOWLIST = (
    "physics",
    "engineering",
    "math",
    "computer science",
    "biology",
    "chemistry",
    "history",
    "economics",
    "machine learning",
    "python",
    "tutorial",
)


def sanitize_query(query: str, max_len: int = 200) -> str:
    """Strip emails/org-like tokens + truncate (never raises)."""
    text = (query or "").strip()
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "", text)
    text = re.sub(r"\b[A-Z]{2,}-\d+\b", "", text)
    text = " ".join(text.split())
    return text[:max_len]


def classify_topic_source(topic: str) -> str:
    """Return 'proprietary', 'public', or 'unknown' (fail-closed)."""
    lowered = (topic or "").lower()
    if any(b in lowered for b in _BLOCKLIST):
        return "proprietary"
    if any(a in lowered for a in _ALLOWLIST):
        return "public"
    return "unknown"


def web_search(query: str, max_results: int = 5, timeout: int = 30) -> list[dict[str, Any]]:
    """POST to Ollama Cloud ``/api/web_search`` (``[]`` on any failure)."""
    try:
        import requests

        from src.config import Config

        cfg = Config()
        if not cfg.web_search_enabled or cfg.ai_mock:
            return []
        if classify_topic_source(query) == "proprietary":
            return []
        clean = sanitize_query(query)
        if not clean:
            return []
        url = f"{cfg.ollama_cloud_base_url.rstrip('/')}/api/web_search"
        headers = {"Content-Type": "application/json"}
        if cfg.ollama_cloud_api_key:
            headers["Authorization"] = f"Bearer {cfg.ollama_cloud_api_key}"
        resp = requests.post(
            url,
            json={"query": clean, "max_results": max_results},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data if isinstance(data, list) else [])
        out = []
        for r in results[:max_results]:
            if not isinstance(r, dict):
                continue
            r_url = str(r.get("url", ""))
            if not r_url.startswith(("http://", "https://")):
                continue
            out.append(
                {
                    "title": str(r.get("title", ""))[:200],
                    "url": r_url,
                    "content": str(r.get("content", ""))[:2000],
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return []


def synthesize_external_questions(
    topic: str, snippets: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Synthesize ≤3 suggested questions from web snippets (verbatim URLs)."""
    try:
        from src.config import Config
        from src.services.llm_json import extract_json
        from src.services.ollama_client import get_ollama_client

        cfg = Config()
        if not snippets or cfg.ai_mock:
            return []
        context = "\n\n".join(
            f"- {s.get('title', '')}: {s.get('content', '')} ({s.get('url', '')})"
            for s in snippets[:3]
        )
        prompt = (
            "Suggest up to 3 follow-up study questions based on these web "
            "snippets. Reply as JSON with questions, each having title, "
            "reason, and verbatim source_urls. Use verbatim URLs only, no PII. "
            f"Topic: {topic}\n\nSnippets:\n{context}"
        )
        client = get_ollama_client()
        raw = client.chat([{"role": "user", "content": prompt}])
        data = extract_json(raw)
        if not isinstance(data, dict):
            return []
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            return []
        allowed_urls = {s.get("url", "") for s in snippets}
        out: list[dict[str, Any]] = []
        for q in questions[:3]:
            if not isinstance(q, dict):
                continue
            title = str(q.get("title", "")).strip()
            if not title:
                continue
            urls = [u for u in (q.get("source_urls", []) or []) if u in allowed_urls]
            out.append(
                {
                    "title": title,
                    "reason": str(q.get("reason", ""))[:300],
                    "source_urls": urls,
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("External synthesis failed: %s", exc)
        return []
