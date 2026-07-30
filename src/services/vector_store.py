"""Vector store — ChromaDB local PersistentClient with content-keyed collections.

Architecture:
- Backend selection: ``CI=true`` -> ``EphemeralClient`` (in-memory, test
  isolation); otherwise -> ``PersistentClient`` at ``DATA_DIR/chroma_db``.
- Collections are content-keyed: ``doc_<sha256[:59]>`` so identical file
  content reuses one collection (dedup across users).
- Embeddings come from the shared ``EmbeddingService`` (mock in CI).
- Retrieval returns ``score = 1.0 - distance`` (ChromaDB uses cosine distance,
  so higher score = more similar).
- Multi-collection retrieval merges results from all a notebook's source
  collections, sorts by score descending, and truncates to ``top_k``.
- Corruption recovery: ``rebuild_collection`` deletes a broken collection and
  re-chunks + re-embeds from cached text (used by ``rag_retriever``).
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from src.services.chunker import chunk_text
from src.services.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config


def get_collection_name(content_hash: str) -> str:
    """Return the content-keyed collection name ``doc_<hash[:59]>``."""
    return f"doc_{content_hash[:59]}"


class VectorStore:
    """ChromaDB vector store with content-keyed collections + mock embeddings."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self._embedder = get_embedding_service()
        # _create_client returns (client, backend_label) so the label can
        # reflect the actual outcome (e.g. "cloud" vs "local" when cloud
        # was requested but fell back).
        self._client: Any
        self.backend: str
        self._client, self.backend = self._create_client_and_backend()

    # ------------------------------------------------------------------
    # Client creation
    # ------------------------------------------------------------------

    def _create_client_and_backend(self) -> tuple[Any, str]:
        """Create the ChromaDB client + a backend label.

        Precedence (highest first):
          1. CI=true              -> EphemeralClient (in-memory, never network)
          2. CHROMA_DB=cloud      -> CloudClient (validates creds + heartbeat;
                                     falls back to local on any failure)
          3. otherwise / fallback -> PersistentClient (local disk, the default)

        Returns:
            (client, backend_label) where backend_label is one of
            "ephemeral", "cloud", or "local". When cloud is requested but
            fails, the label is "local" (since that's what was actually
            constructed) and a WARNING is logged.
        """
        # 1. CI always wins: keep tests isolated and offline.
        if self._config.is_test():
            import chromadb
            from chromadb.config import Settings

            return chromadb.EphemeralClient(Settings(allow_reset=True)), "ephemeral"

        # 2. Cloud backend (opt-in, with graceful fallback).
        if self._config.chroma_db == "cloud":
            cloud_client = self._try_cloud_client()
            if cloud_client is not None:
                return cloud_client, "cloud"
            # Fall through to local on any cloud failure.
            logger.warning("Chroma Cloud unavailable; using local PersistentClient.")

        # 3. Default: local persistent client.
        return self._get_local_client(), "local"

    def _get_local_client(self) -> Any:  # noqa: ANN401
        """Build the default local PersistentClient at ``DATA_DIR/chroma_db``."""
        import chromadb

        path = f"{self._config.data_dir}/chroma_db"
        return chromadb.PersistentClient(path=path)

    def _try_cloud_client(self) -> Any:  # noqa: ANN401
        """Attempt to build a Chroma Cloud client from config credentials.

        Returns a chromadb CloudClient on success, or None on any failure
        (missing/empty credentials, client construction error, heartbeat
        probe failure). Every failure path logs a clear, actionable message
        so developers can diagnose misconfiguration.
        """
        import chromadb

        api_key = (self._config.chroma_cloud_api_key or "").strip()
        connection_string = (self._config.chroma_cloud_connection_string or "").strip()
        database = (self._config.chroma_collection_name or "").strip()

        # Validate required credentials.
        missing = []
        if not api_key:
            missing.append("CHROMA_CLOUD_API_KEY")
        if not connection_string:
            missing.append("CHROMA_CLOUD_CONNECTION_STRING")
        if not database:
            missing.append("CHROMA_COLLECTION_NAME")
        if missing:
            logger.error(
                "CHROMA_DB=cloud requested but required credential(s) are "
                "empty/unset: %s. Reverting to local PersistentClient. "
                "Set these in your .env to use Chroma Cloud.",
                ", ".join(missing),
            )
            return None

        # Attempt client construction + connectivity probe.
        try:
            logger.info(
                "Connecting to Chroma Cloud (tenant=%s, database=%s)",
                connection_string[:8] + "...",
                database,
            )
            client = chromadb.CloudClient(
                tenant=connection_string,
                database=database,
                api_key=api_key,
            )
            # Lightweight connectivity/auth probe. heartbeat() hits the
            # server and raises on bad credentials or unreachable host.
            client.heartbeat()
            logger.info("Chroma Cloud connection established.")
            return client
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Chroma Cloud connection failed (CHROMA_DB=cloud): %s. "
                "Reverting to local PersistentClient. Verify "
                "CHROMA_CLOUD_API_KEY, CHROMA_CLOUD_CONNECTION_STRING, "
                "and CHROMA_COLLECTION_NAME are valid.",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, content_hash: str) -> Any:  # noqa: ANN401
        name = get_collection_name(content_hash)
        return self._client.get_or_create_collection(name=name)

    def collection_exists(self, content_hash: str) -> bool:
        """Return True if a collection for ``content_hash`` exists."""
        name = get_collection_name(content_hash)
        with contextlib.suppress(Exception):
            self._client.get_collection(name=name)
            return True
        return False

    def delete_collection(self, content_hash: str) -> None:
        """Delete the collection for ``content_hash`` (no error if missing)."""
        name = get_collection_name(content_hash)
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=name)

    def reset(self) -> None:
        """Reset the underlying ChromaDB client (clears all collections).

        Only works with ``allow_reset=True`` (test/EphemeralClient only).
        """
        with contextlib.suppress(Exception):
            self._client.reset()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        content_hash: str,
        chunks: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Embed + store ``chunks`` with their ``metadatas`` in a collection."""
        if not chunks:
            return
        collection = self._get_or_create_collection(content_hash)
        embeddings = self._embedder.embed_documents(chunks)
        ids = [f"{content_hash[:16]}_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Stored %d chunks in %s", len(chunks), get_collection_name(content_hash))

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, content_hash: str, query: str, top_k: int = 5) -> str:
        """Basic similarity search; return joined text of top results."""
        results = self.retrieve_with_scores(content_hash, query, top_k)
        return "\n\n".join(r["document"] for r in results)

    def retrieve_with_scores(
        self,
        content_hash: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return ``[{document, score, metadata}]`` sorted by score desc."""
        if not self.collection_exists(content_hash):
            return []
        collection = self._get_or_create_collection(content_hash)
        t0 = time.time()
        query_embedding = self._embedder.embed_query(query)
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        results = self._parse_query_result(raw)
        logger.debug(
            "retrieve_with_scores: %s → %d results (%.0fms)",
            get_collection_name(content_hash),
            len(results),
            (time.time() - t0) * 1000,
        )
        return results

    def retrieve_from_multiple_collections(
        self,
        content_hashes: list[str],
        query: str,
        top_k: int = 5,
    ) -> str:
        """Query multiple collections, merge, return joined text of top results."""
        results = self.retrieve_from_multiple_collections_with_sources(content_hashes, query, top_k)
        return "\n\n".join(r["text"] for r in results)

    def retrieve_from_multiple_collections_with_sources(
        self,
        content_hashes: list[str],
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query multiple collections, merge by score, return top_k with provenance.

        Each result: ``{text, filename, page, chunk_index, score}``.
        """
        if not content_hashes:
            return []
        all_results: list[dict[str, Any]] = []
        per_collection_k = 3
        for h in content_hashes:
            partial = self.retrieve_with_scores(h, query, top_k=per_collection_k)
            for r in partial:
                md = r.get("metadata", {}) or {}
                all_results.append(
                    {
                        "text": r["document"],
                        "filename": md.get("filename", ""),
                        "page": md.get("page"),
                        "chunk_index": md.get("chunk_index"),
                        "score": r["score"],
                    }
                )
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Corruption recovery
    # ------------------------------------------------------------------

    def rebuild_collection(
        self,
        content_hash: str,
        extracted_text: str,
        filename: str,
        page_count: int | None = None,
    ) -> None:
        """Delete a broken collection and rebuild from cached text."""
        self.delete_collection(content_hash)
        chunks = chunk_text(extracted_text)
        if not chunks:
            return
        metadatas = [
            {
                "source_hash": content_hash,
                "filename": filename,
                "page": i + 1 if page_count else (i + 1),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.store_chunks(content_hash, chunks, metadatas)
        logger.info(
            "Rebuilt collection %s with %d chunks", get_collection_name(content_hash), len(chunks)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_query_result(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a ChromaDB query response into a flat list of results."""
        docs = raw.get("documents", [[]])
        dists = raw.get("distances", [[]])
        mds = raw.get("metadatas", [[]])
        if not docs or not docs[0]:
            return []
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(docs[0]):
            dist = dists[0][i] if dists and i < len(dists[0]) else 0.0
            md = mds[0][i] if mds and i < len(mds[0]) else {}
            out.append(
                {
                    "document": doc,
                    "score": 1.0 - float(dist),
                    "metadata": md,
                }
            )
        out.sort(key=lambda r: r["score"], reverse=True)
        return out


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return a process-wide ``VectorStore`` (created lazily)."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Reset the cached store (used by tests that change config)."""
    global _store
    _store = None
