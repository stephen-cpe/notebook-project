"""RAG retriever — multi-collection retrieval with corruption recovery + sources.

Orchestrates ``VectorStore`` to query all of a notebook's source collections,
merge results by score, and return top_k chunks with source provenance. If a
collection probe fails (corruption), it rebuilds from the ``ContentRegistry``
cached text and re-queries.

Functions:
- ``retrieve_with_sources(hashes, query, top_k)`` -> list of provenance dicts.
- ``retrieve(hashes, query, top_k)`` -> joined context text.
- ``build_context_string(results)`` -> formatted context for the LLM prompt.
- ``format_sources(results)`` -> deduplicated source list for citations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config


class RAGRetriever:
    """Retrieve + merge across content-keyed ChromaDB collections."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()
        self._config = config
        self._vector_store: VectorStore = get_vector_store()

    def retrieve_with_sources(
        self,
        content_hashes: list[str],
        query: str,
        top_k: int = 5,
        filenames: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query all collections, merge by score, return top_k with provenance.

        Each result: ``{text, filename, page, chunk_index, score}``.
        Broken collections are auto-rebuilt from ContentRegistry text.
        ``filenames`` optionally maps content hash -> the requesting
        notebook's display name for that hash. Vector metadata carries the
        *first uploader's* filename (shared dedup storage), so citations must
        resolve through this map — otherwise one user can see another user's
        filename, and renames never take effect.
        """
        if not content_hashes:
            return []
        # Embed once and reuse the vector across collections: a notebook at
        # the source cap would otherwise pay one embedding call per source
        # (latency, rate limits, and cost with hosted providers).
        query_embedding = self._vector_store.embed_query(query)
        all_results: list[dict[str, Any]] = []
        for h in content_hashes:
            partial = self._safe_retrieve(
                h, query, filenames=filenames, query_embedding=query_embedding
            )
            all_results.extend(partial)
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return all_results[:top_k]

    def retrieve(
        self,
        content_hashes: list[str],
        query: str,
        top_k: int = 5,
        filenames: dict[str, str] | None = None,
    ) -> str:
        """Like ``retrieve_with_sources`` but returns joined context text."""
        results = self.retrieve_with_sources(content_hashes, query, top_k, filenames=filenames)
        return build_context_string(results)

    # ------------------------------------------------------------------
    # Corruption recovery
    # ------------------------------------------------------------------

    def _safe_retrieve(
        self,
        content_hash: str,
        query: str,
        per_collection_k: int = 3,
        filenames: dict[str, str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve from one collection; rebuild from registry if corrupted.

        Returns results in provenance format: ``{text, filename, page,
        chunk_index, score}``.
        """
        if not self._vector_store.collection_exists(content_hash):
            logger.warning(
                "Collection for hash %s... not found; attempting rebuild",
                content_hash[:12],
            )
            self._try_rebuild(content_hash, filenames)
        raw = self._safe_query(content_hash, query, per_collection_k, filenames, query_embedding)
        return [_to_provenance(r, filenames) for r in raw]

    def _safe_query(
        self,
        content_hash: str,
        query: str,
        top_k: int,
        filenames: dict[str, str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Query with one retry after rebuild on failure."""
        try:
            return self._vector_store.retrieve_with_scores(
                content_hash, query, top_k=top_k, query_embedding=query_embedding
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed for %s...: %s; rebuilding", content_hash[:12], exc)
            self._try_rebuild(content_hash, filenames)
            return self._vector_store.retrieve_with_scores(
                content_hash, query, top_k=top_k, query_embedding=query_embedding
            )

    def _try_rebuild(self, content_hash: str, filenames: dict[str, str] | None = None) -> None:
        """Rebuild a collection from the ContentRegistry cached text."""
        try:
            from src.repositories import content_registry_repo

            entry = content_registry_repo.get_by_hash(content_hash)
            if entry is None:
                logger.error(
                    "No ContentRegistry entry for hash %s..., cannot rebuild", content_hash[:12]
                )
                return
            filename = self._resolve_filename(content_hash, filenames)
            self._vector_store.rebuild_collection(
                content_hash,
                entry.extracted_text,
                filename=filename,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rebuild failed for hash %s...: %s", content_hash[:12], exc)

    @staticmethod
    def _resolve_filename(content_hash: str, filenames: dict[str, str] | None = None) -> str:
        """Find the display filename for a content hash.

        Prefers the requesting notebook's alias map (never leaks another
        user's filename); falls back to an arbitrary Source row for legacy
        callers without notebook context.
        """
        if filenames is not None and content_hash in filenames:
            return filenames[content_hash]
        try:
            from src.extensions import db
            from src.models import Source

            src = db.session.query(Source).filter_by(content_hash=content_hash).first()
            if src is not None:
                return src.filename
        except Exception:  # noqa: BLE001, S110
            pass
        return "unknown"


# ---------------------------------------------------------------------------
# Formatting helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _to_provenance(raw: dict[str, Any], filenames: dict[str, str] | None = None) -> dict[str, Any]:
    """Convert a raw VectorStore result to provenance format.

    Raw: ``{document, score, metadata}``
    Provenance: ``{text, filename, page, chunk_index, score}``

    ``filenames`` (hash -> requesting notebook's display name) overrides the
    stored vector metadata filename, which reflects the first uploader.
    """
    md = raw.get("metadata", {}) or {}
    source_hash = md.get("source_hash", "")
    filename = md.get("filename", "")
    if filenames is not None and source_hash in filenames:
        filename = filenames[source_hash]
    return {
        "text": raw.get("document", ""),
        "filename": filename,
        "page": md.get("page"),
        "chunk_index": md.get("chunk_index"),
        "score": raw.get("score", 0.0),
    }


def build_context_string(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt.

    Format: ``[Source N] (filename, Page X)\\n{text}`` repeated, separated by
    blank lines. Handles missing page gracefully.
    """
    if not results:
        return ""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        filename = r.get("filename", "unknown")
        page = r.get("page")
        text = r.get("text", r.get("document", ""))
        if page is not None:
            header = f"[Source {i}] ({filename}, Page {page})"
        else:
            header = f"[Source {i}] ({filename})"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def format_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sources by (filename, page) and return a clean list.

    Returns ``[{filename, page}]`` with duplicates removed.
    """
    seen: set[tuple[str, int | None]] = set()
    sources: list[dict[str, Any]] = []
    for r in results:
        filename = r.get("filename", "unknown")
        page = r.get("page")
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"filename": filename, "page": page})
    return sources


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_retriever: RAGRetriever | None = None


def get_rag_retriever() -> RAGRetriever:
    """Return a process-wide ``RAGRetriever`` (created lazily)."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def reset_rag_retriever() -> None:
    """Reset the cached retriever (used by tests that change config)."""
    global _retriever
    _retriever = None
