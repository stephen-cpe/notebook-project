"""Pipeline fingerprint — config/prompt versioning (ported from pdf2md).

Any change to prompts, models, DPI, thresholds, or chunking must produce a
new fingerprint so a resume/rebuild never silently mixes artifacts produced
under different settings. The fingerprint is logged at ingestion start and
included in the ingestion log line; embedding collections already carry
their own version via ``vector_store.embedding_fingerprint``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

PIPELINE_VERSION = "nb-pipeline-v1"


def compute_pipeline_version(config: Config | None = None) -> str:
    """Return a stable hex fingerprint of the ingestion pipeline settings."""
    if config is None:
        from src.config import Config

        config = Config()
    parts = [
        PIPELINE_VERSION,
        f"vision={config.vision_model}",
        f"dpi={config.ocr_dpi}",
        f"max_pages={config.ocr_max_pages}",
        f"max_dim={config.ocr_max_image_dimension}",
        f"threshold={config.ocr_text_threshold}",
        f"pdf_gate={config.pdf_needs_ocr_min_total_chars}/{config.pdf_needs_ocr_min_chars_per_page}",
        f"figure={config.ocr_figure_description}",
        f"emb={config.embedding_provider}|{config.embedding_model}|{config.embedding_dim}",
        f"ctx={config.ollama_num_ctx}|{config.rag_max_context_chars}|{config.rag_top_k}",
        f"chat={config.chat_model}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
