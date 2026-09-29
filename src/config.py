"""Centralized configuration loaded from environment variables.

All config values are read lazily at instance creation. ``Config.summary()``
returns a redacted, human-readable snapshot for diagnostics/logging — secrets
are masked so they never appear in logs (NFR-20, NFR-26).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: bool = False) -> bool:
    """Parse a boolean env var (``true``/``1``/``yes`` -> True)."""
    return str(os.getenv(key, str(default))).strip().lower() in {"true", "1", "yes"}


def _int(key: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on parse error."""
    try:
        return int(str(os.getenv(key, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    """Application configuration resolved from the environment."""

    # Flask
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me"))
    flask_port: int = field(default_factory=lambda: _int("FLASK_PORT", 5000))

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://notebook_user:notebook_pass@localhost:5432/notebook_project",
        )
    )
    test_database_url: str = field(
        default_factory=lambda: os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")
    )

    # AI / Ollama Cloud
    ai_mock: bool = field(default_factory=lambda: _bool("AI_MOCK", False))
    ollama_cloud_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CLOUD_BASE_URL", "")
    )
    ollama_cloud_api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_CLOUD_API_KEY", ""))
    ollama_timeout: int = field(default_factory=lambda: _int("OLLAMA_TIMEOUT", 120))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gemma4:31b-cloud"))
    enable_thinking: bool = field(default_factory=lambda: _bool("ENABLE_THINKING", True))

    # Embeddings (HuggingFace)
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 1024))
    # Provider: "local" (default, sentence-transformers on CPU/GPU) or "hf_inference"
    # (hosted HF Inference API — no local weights, per-call network latency).
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    )
    # Optional HF Inference API endpoint override (e.g. a dedicated TEI endpoint).
    # Empty -> use the default router (https://api-inference.huggingface.co).
    hf_inference_endpoint: str = field(
        default_factory=lambda: os.getenv("HF_INFERENCE_ENDPOINT", "").strip()
    )
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # OCR / vision (Ollama glm-5.3-flash, consolidated vision understanding)
    # The old HuggingFace GLM-OCR path has been replaced: all image OCR,
    # table extraction, and figure description now use a single natively
    # multimodal vision model via the Ollama Cloud chat API. ``ocr_provider``
    # / ``ocr_inference_endpoint`` are kept as deprecated aliases so old
    # imports/env files keep working; new code uses ``vision_model``.
    ocr_fallback_enabled: bool = field(default_factory=lambda: _bool("OCR_FALLBACK_ENABLED", True))
    ocr_text_threshold: int = field(default_factory=lambda: _int("OCR_TEXT_THRESHOLD", 200))
    ocr_max_image_dimension: int = field(
        default_factory=lambda: _int("OCR_MAX_IMAGE_DIMENSION", 2048)
    )
    ocr_max_pages: int = field(default_factory=lambda: _int("OCR_MAX_PAGES", 30))
    ocr_dpi: int = field(default_factory=lambda: _int("OCR_DPI", 150))
    poppler_path: str = field(default_factory=lambda: os.getenv("POPPLER_PATH", ""))
    vision_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "glm-5.3-flash:cloud")
    )
    vision_timeout: int = field(default_factory=lambda: _int("OLLAMA_VISION_TIMEOUT", 300))
    pdf_needs_ocr_min_total_chars: int = field(
        default_factory=lambda: _int("PDF_NEEDS_OCR_MIN_TOTAL_CHARS", 1000)
    )
    pdf_needs_ocr_min_chars_per_page: int = field(
        default_factory=lambda: _int("PDF_NEEDS_OCR_MIN_CHARS_PER_PAGE", 300)
    )
    ocr_figure_description: bool = field(
        default_factory=lambda: _bool("OCR_FIGURE_DESCRIPTION", True)
    )
    # Deprecated: kept so old .env files/tests keep working. The vision
    # model above is used for all OCR regardless of this value.
    # Provider: "local" (default, transformers on CPU/GPU) or "hf_inference"
    # (hosted HF Inference API — no local weights, per-call network latency).
    ocr_provider: str = field(
        default_factory=lambda: os.getenv("OCR_PROVIDER", "local").strip().lower()
    )
    # Optional: dedicated inference endpoint for GLM-OCR.
    # Empty -> default HF Inference router.
    ocr_inference_endpoint: str = field(
        default_factory=lambda: os.getenv("OCR_INFERENCE_ENDPOINT", "").strip()
    )

    # RAG budget (context-window-derived retrieval, ported from study-and-learn).
    # Sized for a 256K-window chat model (gemma4:31b-cloud): ~839K usable
    # chars after the 20% reserve, so the ceilings below — not hardcoded
    # top_k values — are what bound each prompt.
    ollama_num_ctx: int = field(default_factory=lambda: _int("OLLAMA_NUM_CTX", 262144))
    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 20))
    rag_max_top_k: int = field(default_factory=lambda: _int("RAG_MAX_TOP_K", 200))
    rag_max_context_chars: int = field(
        default_factory=lambda: _int("RAG_MAX_CONTEXT_CHARS", 700000)
    )
    rag_max_citations: int = field(default_factory=lambda: _int("RAG_MAX_CITATIONS", 12))
    rag_summary_map: bool = field(default_factory=lambda: _bool("RAG_SUMMARY_MAP", True))
    # Full-coverage digest sizing (map-reduce over source sections).
    rag_digest_max_chars: int = field(default_factory=lambda: _int("RAG_DIGEST_MAX_CHARS", 80000))
    rag_map_section_chars: int = field(default_factory=lambda: _int("RAG_MAP_SECTION_CHARS", 6000))
    rag_map_max_sections: int = field(default_factory=lambda: _int("RAG_MAP_MAX_SECTIONS", 80))

    # Suggested questions: dedup + opt-in web augmentation (fail-closed)
    web_search_enabled: bool = field(default_factory=lambda: _bool("WEB_SEARCH_ENABLED", False))
    web_search_max_results: int = field(default_factory=lambda: _int("WEB_SEARCH_MAX_RESULTS", 5))
    web_search_timeout: int = field(default_factory=lambda: _int("WEB_SEARCH_TIMEOUT", 30))
    web_search_synth_model: str = field(
        default_factory=lambda: os.getenv("WEB_SEARCH_SYNTH_MODEL", "gpt-oss:20b-cloud")
    )

    # Diagrams → Mermaid (optional figure reinterpretation via vision model)
    diagram_to_mermaid: bool = field(default_factory=lambda: _bool("DIAGRAM_TO_MERMAID", True))
    diagram_min_confidence: int = field(default_factory=lambda: _int("DIAGRAM_MIN_CONFIDENCE", 80))
    diagram_verify: bool = field(default_factory=lambda: _bool("DIAGRAM_VERIFY", True))

    # Vector store
    chroma_db: str = field(default_factory=lambda: os.getenv("CHROMA_DB", "local"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
    # Chroma Cloud credentials (only used when CHROMA_DB=cloud). If any are
    # missing/empty or the connection probe fails, the app logs an error and
    # falls back to the local PersistentClient.
    chroma_cloud_api_key: str = field(default_factory=lambda: os.getenv("CHROMA_CLOUD_API_KEY", ""))
    # CHROMA_CLOUD_CONNECTION_STRING is the Chroma tenant ID.
    chroma_cloud_connection_string: str = field(
        default_factory=lambda: os.getenv("CHROMA_CLOUD_CONNECTION_STRING", "")
    )
    # CHROMA_COLLECTION_NAME becomes the Chroma Cloud database name; the app's
    # per-file collections (doc_<filehash>) live inside it.
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "notebook-project-chromadb")
    )

    # Sources
    max_sources_per_notebook: int = field(
        default_factory=lambda: _int("MAX_SOURCES_PER_NOTEBOOK", 50)
    )
    max_file_size_mb: int = field(default_factory=lambda: _int("MAX_FILE_SIZE_MB", 25))

    # Session cookie hardening
    session_cookie_secure: bool = field(
        default_factory=lambda: _bool("SESSION_COOKIE_SECURE", False)
    )
    session_cookie_httponly: bool = field(
        default_factory=lambda: _bool("SESSION_COOKIE_HTTPONLY", True)
    )
    session_cookie_samesite: str = field(
        default_factory=lambda: os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    )

    # HuggingFace Inference API timeouts/retries
    hf_timeout_seconds: int = field(default_factory=lambda: _int("HF_TIMEOUT_SECONDS", 60))

    # Audio
    overview_min_duration_seconds: int = field(
        default_factory=lambda: _int("OVERVIEW_MIN_DURATION_SECONDS", 60)
    )
    overview_max_duration_seconds: int = field(
        default_factory=lambda: _int("OVERVIEW_MAX_DURATION_SECONDS", 480)
    )
    overview_max_context_chars: int = field(
        default_factory=lambda: _int("OVERVIEW_MAX_CONTEXT_CHARS", 150000)
    )

    # Admin seed (first run only)
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "change-me"))

    # Voice conversation
    voice_enabled: bool = field(default_factory=lambda: _bool("VOICE_ENABLED", False))
    voice_stt_provider: str = field(
        default_factory=lambda: os.getenv("VOICE_STT_PROVIDER", "local").strip().lower()
    )
    voice_stt_model: str = field(default_factory=lambda: os.getenv("VOICE_STT_MODEL", "base.en"))
    voice_stt_device: str = field(
        default_factory=lambda: os.getenv("VOICE_STT_DEVICE", "cpu").strip().lower()
    )
    voice_stt_compute_type: str = field(
        default_factory=lambda: os.getenv("VOICE_STT_COMPUTE_TYPE", "int8").strip().lower()
    )
    voice_stt_language: str = field(
        default_factory=lambda: os.getenv("VOICE_STT_LANGUAGE", "").strip()
    )
    voice_max_recording_seconds: int = field(
        default_factory=lambda: _int("VOICE_MAX_RECORDING_SECONDS", 60)
    )
    voice_max_upload_mb: int = field(default_factory=lambda: _int("VOICE_MAX_UPLOAD_MB", 10))
    voice_tts_fallback_speaker: str = field(
        default_factory=lambda: os.getenv("VOICE_TTS_FALLBACK_SPEAKER", "Ava")
    )
    voice_cors_origins: str = field(default_factory=lambda: os.getenv("VOICE_CORS_ORIGINS", ""))

    # CI / test flags
    ci: bool = field(default_factory=lambda: _bool("CI", False))

    def is_test(self) -> bool:
        """True when running under pytest / CI (use in-memory backends, mock AI)."""
        return self.ci

    def summary(self) -> dict[str, Any]:
        """Return a redacted, log-safe snapshot of the config (no secrets)."""
        return {
            "flask_port": self.flask_port,
            "database_url": _redact_url(self.database_url),
            "test_database_url": _redact_url(self.test_database_url),
            "ai_mock": self.ai_mock,
            "ollama_cloud_base_url": self.ollama_cloud_base_url,
            "ollama_cloud_api_key": _redact_secret(self.ollama_cloud_api_key),
            "ollama_timeout": self.ollama_timeout,
            "chat_model": self.chat_model,
            "enable_thinking": self.enable_thinking,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_provider": self.embedding_provider,
            "hf_inference_endpoint": _redact_secret(self.hf_inference_endpoint),
            "hf_token": _redact_secret(self.hf_token),
            "ocr_fallback_enabled": self.ocr_fallback_enabled,
            "ocr_text_threshold": self.ocr_text_threshold,
            "ocr_max_image_dimension": self.ocr_max_image_dimension,
            "ocr_max_pages": self.ocr_max_pages,
            "ocr_dpi": self.ocr_dpi,
            "poppler_path": self.poppler_path,
            "ocr_provider": self.ocr_provider,
            "ocr_inference_endpoint": _redact_secret(self.ocr_inference_endpoint),
            "vision_model": self.vision_model,
            "vision_timeout": self.vision_timeout,
            "pdf_needs_ocr_min_total_chars": self.pdf_needs_ocr_min_total_chars,
            "pdf_needs_ocr_min_chars_per_page": self.pdf_needs_ocr_min_chars_per_page,
            "ocr_figure_description": self.ocr_figure_description,
            "ollama_num_ctx": self.ollama_num_ctx,
            "rag_top_k": self.rag_top_k,
            "rag_max_top_k": self.rag_max_top_k,
            "rag_max_context_chars": self.rag_max_context_chars,
            "rag_max_citations": self.rag_max_citations,
            "rag_summary_map": self.rag_summary_map,
            "rag_digest_max_chars": self.rag_digest_max_chars,
            "rag_map_section_chars": self.rag_map_section_chars,
            "rag_map_max_sections": self.rag_map_max_sections,
            "web_search_enabled": self.web_search_enabled,
            "diagram_to_mermaid": self.diagram_to_mermaid,
            "diagram_min_confidence": self.diagram_min_confidence,
            "diagram_verify": self.diagram_verify,
            "chroma_db": self.chroma_db,
            "data_dir": self.data_dir,
            "chroma_cloud_api_key": _redact_secret(self.chroma_cloud_api_key),
            "chroma_cloud_connection_string": _redact_secret(self.chroma_cloud_connection_string),
            "chroma_collection_name": self.chroma_collection_name,
            "max_sources_per_notebook": self.max_sources_per_notebook,
            "max_file_size_mb": self.max_file_size_mb,
            "session_cookie_secure": self.session_cookie_secure,
            "session_cookie_httponly": self.session_cookie_httponly,
            "session_cookie_samesite": self.session_cookie_samesite,
            "hf_timeout_seconds": self.hf_timeout_seconds,
            "voice_enabled": self.voice_enabled,
            "voice_stt_provider": self.voice_stt_provider,
            "voice_stt_model": self.voice_stt_model,
            "voice_stt_device": self.voice_stt_device,
            "voice_stt_compute_type": self.voice_stt_compute_type,
            "voice_max_recording_seconds": self.voice_max_recording_seconds,
            "voice_max_upload_mb": self.voice_max_upload_mb,
            "voice_tts_fallback_speaker": self.voice_tts_fallback_speaker,
            "overview_min_duration_seconds": self.overview_min_duration_seconds,
            "overview_max_duration_seconds": self.overview_max_duration_seconds,
            "overview_max_context_chars": self.overview_max_context_chars,
            "ci": self.ci,
        }


def _redact_secret(value: str) -> str:
    """Mask a secret, showing only whether it is set."""
    return "<set>" if value else "<unset>"


def _redact_url(url: str) -> str:
    """Mask the password in a database URL for safe logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme_rest = url.split("://", 1)
    if len(scheme_rest) != 2:
        return url
    scheme, rest = scheme_rest
    if "@" not in rest:
        return url
    creds, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}"


def get_config() -> Config:
    """Build a ``Config`` from the current environment."""
    return Config()
