"""Embeddings service — Qwen3-Embedding via sentence-transformers OR HF Inference API.

In production the service loads ``Qwen/Qwen3-Embedding-0.6B`` (0.6B, 1024-dim,
instruction-aware, MRL) — by default locally via ``sentence_transformers.SentenceTransformer``.
Set ``EMBEDDING_PROVIDER=hf_inference`` to instead use Hugging Face's hosted
Inference API (no local weights, per-call network latency, requires ``HF_TOKEN``).
When ``AI_MOCK=true`` (tests/CI) a deterministic hash-based mock is used so no
model download or network is required.

HF token handling (NFR-26):
- Absent: a one-time ``WARNING`` is logged explaining rate limits may appear.
- Invalid (HF returns 401/403): an ``ERROR`` is logged with the HF status; the
  service proceeds unauthenticated (degraded, not fatal).
- Valid: used silently; the token value is never logged.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Protocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config

# Module-level flag ensuring the "HF_TOKEN not set" warning is logged once.
_token_warning_emitted = False


class _EmbedderBackend(Protocol):
    """Internal contract for the real (non-mock) embedding backends."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingService:
    """Wrap a sentence-transformers OR HF Inference API model with mock support.

    Call ``embed_query(text)`` for search queries (uses the model's query
    prompt) and ``embed_documents(texts)`` for documents to be indexed.
    """

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self.model_name: str = config.embedding_model
        self.dimension: int = config.embedding_dim
        self.provider: str = config.embedding_provider
        self._mock: bool = bool(config.ai_mock)
        self._hf_token: str = config.hf_token
        self._backend: _EmbedderBackend | None = None
        self._device: str = self._detect_device()

        self._handle_hf_token()
        if not self._mock:
            self._backend = self._make_backend(config)

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (uses the model's "query" prompt)."""
        if self._mock:
            return self._mock_vector(text, is_query=True)
        assert self._backend is not None
        t0 = time.time()
        vec = self._backend.embed_query(text)
        elapsed_ms = (time.time() - t0) * 1000
        logger.info("embed_query: %d chars → %d dims (%.0fms)", len(text), len(vec), elapsed_ms)
        return self._to_float_list(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (no query prompt)."""
        if not texts:
            return []
        if self._mock:
            return [self._mock_vector(t, is_query=False) for t in texts]
        assert self._backend is not None
        t0 = time.time()
        vecs = self._backend.embed_documents(texts)
        logger.info(
            "embed_documents: %d texts → %d vectors (%.0fms)",
            len(texts),
            len(vecs),
            (time.time() - t0) * 1000,
        )
        return [self._to_float_list(v) for v in vecs]

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _make_backend(self, config: Config) -> _EmbedderBackend:
        """Select the real embedding backend based on ``embedding_provider``."""
        provider = self.provider
        if provider == "hf_inference":
            return _HfInferenceBackend(
                model=self.model_name,
                token=self._hf_token,
                endpoint=config.hf_inference_endpoint,
                timeout=config.hf_timeout_seconds,
            )
        if provider == "local":
            return _LocalSentenceTransformerBackend(
                model_name=self.model_name,
                token=self._hf_token,
                device=self._device,
            )
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. "
            "Expected 'local' (sentence-transformers) or 'hf_inference' (HF Inference API)."
        )

    # ------------------------------------------------------------------
    # HF token handling (NFR-26)
    # ------------------------------------------------------------------

    def _handle_hf_token(self) -> None:
        global _token_warning_emitted
        if not self._hf_token:
            if not _token_warning_emitted:
                logger.warning(
                    "HF_TOKEN not set; HuggingFace calls will be unauthenticated. "
                    "You may see rate-limit warnings. Set HF_TOKEN (READ scope) "
                    "in .env to suppress them."
                )
                _token_warning_emitted = True
            return

    # ------------------------------------------------------------------
    # Mock embedder (deterministic, L2-normalized)
    # ------------------------------------------------------------------

    def _mock_vector(self, text: str, is_query: bool) -> list[float]:
        """Produce a deterministic, L2-normalized vector from ``text``.

        Query vectors are salted differently from document vectors so the two
        paths yield distinct embeddings (mirroring Qwen3's instruction-awareness).
        """
        salt = b"query" if is_query else b"document"
        digest = hashlib.sha256(salt + text.encode("utf-8")).digest()
        # Expand the 32-byte digest to the configured dimension.
        raw: list[float] = []
        i = 0
        while len(raw) < self.dimension:
            chunk = hashlib.sha256(digest + i.to_bytes(4, "big")).digest()
            for b in chunk:
                raw.append(float(b) / 255.0 - 0.5)
                if len(raw) >= self.dimension:
                    break
            i += 1
        # L2-normalize.
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float_list(vec: Any) -> list[float]:  # noqa: ANN401
        """Coerce a numpy/vector output to a plain ``list[float]``."""
        try:
            return [float(x) for x in vec.tolist()]
        except AttributeError:
            return [float(x) for x in vec]


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


class _LocalSentenceTransformerBackend:
    """Local backend: ``sentence-transformers.SentenceTransformer`` on CPU/GPU.

    Model weights are downloaded from Hugging Face once, then all inference runs
    on the local machine — no network traffic per embedding call.
    """

    def __init__(self, model_name: str, token: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(
                model_name,
                token=token or None,
                trust_remote_code=True,
                device=device,
            )
            logger.info("Loaded local embedding model %s (device=%s)", model_name, device)
        except Exception as exc:  # noqa: BLE001
            status = _extract_hf_status(exc)
            if status in (401, 403):
                logger.error(
                    "HuggingFace rejected HF_TOKEN (status=%s). "
                    "Proceeding unauthenticated. Check that the token is valid "
                    "and has READ scope.",
                    status,
                )
                # Retry without the token (degraded, not fatal).
                self._model = SentenceTransformer(
                    model_name, token=None, trust_remote_code=True, device=device
                )
            else:
                logger.error("Failed to load embedding model %s: %s", model_name, exc)
                raise

    def embed_query(self, text: str) -> list[float]:
        return list(
            self._model.encode(
                text,
                prompt_name="query",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(v) for v in vecs]


class _HfInferenceBackend:
    """Hosted backend: Hugging Face Inference API (``feature_extraction`` task).

    No local weights are required. Each call is an HTTPS request to the HF
    Inference router (or a dedicated TEI endpoint if ``endpoint`` is set).
    Requires ``HF_TOKEN`` with READ scope; unauthenticated calls are rate-limited.
    """

    def __init__(self, model: str, token: str, endpoint: str = "", timeout: int = 60) -> None:
        from huggingface_hub import InferenceClient

        # Pass base_url=None (not "") so the client uses the default router.
        self._client = InferenceClient(
            model=model, token=token or None, base_url=endpoint or None, timeout=timeout
        )
        self._model = model
        self._token = token
        logger.info(
            "Configured HF Inference API backend (model=%s, endpoint=%s, timeout=%ss)",
            model,
            endpoint or "default-router",
            timeout,
        )
        if not token:
            logger.warning(
                "EMBEDDING_PROVIDER=hf_inference but HF_TOKEN is not set. "
                "Calls will be unauthenticated and rate-limited."
            )

    def embed_query(self, text: str) -> list[float]:
        # prompt_name="query" mirrors the local backend — Qwen3 prepends its
        # query instruction server-side. normalize=True yields L2-normalized
        # vectors consistent with the local path.
        vec = self._call_with_retry(
            lambda: self._client.feature_extraction(text, normalize=True, prompt_name="query")
        )
        return self._flatten(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = self._call_with_retry(
                lambda t=t: self._client.feature_extraction(t, normalize=True)
            )
            out.append(self._flatten(vec))
        return out

    @staticmethod
    def _call_with_retry(fn: Any) -> Any:  # noqa: ANN401
        """Call ``fn`` once, retry once on a transient network error."""
        import time

        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "429" in msg:
                logger.error("HF Inference API rate-limited (429). Backing off and retrying once.")
            elif any(s in msg for s in ("timeout", "timed out", "connection")):
                logger.warning("HF Inference API transient error (%s). Retrying once.", exc)
            else:
                raise
            time.sleep(2.0)
            return fn()

    @staticmethod
    def _flatten(vec: Any) -> list[float]:  # noqa: ANN401
        """Coerce the HF ndarray (possibly shape [1, seq, dim] or [dim]) to [dim]."""
        try:
            arr = vec.squeeze()
        except AttributeError:
            return list(vec)
        # Take the last token's embedding (CLS-style) when sequence dim present.
        if arr.ndim > 1:
            arr = arr[-1]
        return [float(x) for x in arr]


# ----------------------------------------------------------------------


def _extract_hf_status(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status from a HF exception."""
    msg = str(exc).lower()
    if "401" in msg:
        return 401
    if "403" in msg:
        return 403
    return None


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return a process-wide ``EmbeddingService`` (created lazily)."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


def reset_embedding_service() -> None:
    """Reset the cached service (used by tests that change config)."""
    global _service, _token_warning_emitted
    _service = None
    _token_warning_emitted = False
