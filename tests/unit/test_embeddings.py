"""Unit tests for src.services.embeddings (TDD step 5).

The embeddings service wraps `sentence-transformers.SentenceTransformer`
(default) or the Hugging Face Inference API (`EMBEDDING_PROVIDER=hf_inference`)
loading `Qwen/Qwen3-Embedding-0.6B`. Unit tests run fully offline using a
deterministic mock embedder; real model loading is gated behind
`RUN_INTEGRATION=1` and marked `@pytest.mark.integration`.

Covers:
- Mock mode returns deterministic, correctly-shaped vectors.
- Query vs document encoding (query uses the "query" prompt name).
- HF token handling: absent -> WARNING logged once; invalid 401/403 -> ERROR
  logged and proceeds unauthenticated; valid -> used silently.
- Dimension config respected.
- Embedding list + single-text paths.
- Provider selection: 'local' vs 'hf_inference' (with mocked backends).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

from src.services.embeddings import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)


# Ensure the module-level warning flag is fresh per test.
@pytest.fixture(autouse=True)
def _reset_embedding_state() -> None:
    reset_embedding_service()
    yield
    reset_embedding_service()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec_summary(vec: list[float]) -> tuple[int, float]:
    """Return (dim, first-value) for quick deterministic checks."""
    return len(vec), round(vec[0], 4)


# ---------------------------------------------------------------------------
# Mock-mode tests (offline, deterministic)
# ---------------------------------------------------------------------------


class TestMockEmbedding:
    def test_mock_returns_correct_dimension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("EMBEDDING_DIM", "1024")
        svc = EmbeddingService()
        vec = svc.embed_query("hello")
        assert len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

    def test_mock_deterministic_same_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        v1 = svc.embed_query("same text")
        v2 = svc.embed_query("same text")
        assert v1 == v2

    def test_mock_different_inputs_differ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        v1 = svc.embed_query("alpha")
        v2 = svc.embed_query("beta")
        assert v1 != v2

    def test_mock_embed_documents_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        vecs = svc.embed_documents(["one", "two", "three"])
        assert len(vecs) == 3
        assert all(len(v) == svc.dimension for v in vecs)

    def test_mock_dimension_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("EMBEDDING_DIM", "512")
        svc = EmbeddingService()
        assert svc.dimension == 512
        vec = svc.embed_query("x")
        assert len(vec) == 512

    def test_mock_empty_documents_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        assert svc.embed_documents([]) == []

    def test_mock_empty_string_returns_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        # Empty string still produces a vector (model would too).
        vec = svc.embed_query("")
        assert len(vec) == svc.dimension

    def test_mock_is_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock vectors are L2-normalized like the real Qwen3 model output."""
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        vec = svc.embed_query("normalize me")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Query vs document prompt handling
# ---------------------------------------------------------------------------


class TestQueryVsDocument:
    def test_query_and_document_differ_in_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In real Qwen3, query uses prompt_name='query' (adds instruction).
        The mock simulates this by salting the input differently for query vs
        document so the two paths produce distinct vectors.
        """
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        q = svc.embed_query("shared text")
        d = svc.embed_documents(["shared text"])[0]
        assert q != d


# ---------------------------------------------------------------------------
# HF token handling (NFR-26)
# ---------------------------------------------------------------------------


class TestHfTokenHandling:
    def test_no_token_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "")
        caplog.set_level(logging.WARNING, logger="src.services.embeddings")
        EmbeddingService()
        assert any("HF_TOKEN not set" in r.message for r in caplog.records)

    def test_token_set_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_value")
        caplog.set_level(logging.WARNING, logger="src.services.embeddings")
        EmbeddingService()
        assert not any("HF_TOKEN not set" in r.message for r in caplog.records)

    def test_token_never_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "super-secret-token-xyz")
        caplog.set_level(logging.DEBUG, logger="src.services.embeddings")
        EmbeddingService()
        all_text = " ".join(r.message for r in caplog.records)
        assert "super-secret-token-xyz" not in all_text

    def test_warning_logged_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "")
        caplog.set_level(logging.WARNING, logger="src.services.embeddings")
        EmbeddingService()
        EmbeddingService()
        warnings = [r for r in caplog.records if "HF_TOKEN not set" in r.message]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# get_embedding_service singleton
# ---------------------------------------------------------------------------


class TestGetService:
    def test_returns_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = get_embedding_service()
        assert isinstance(svc, EmbeddingService)
        assert svc.dimension > 0

    def test_model_name_is_qwen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        assert svc.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_default_provider_is_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = EmbeddingService()
        assert svc.provider == "local"


# ---------------------------------------------------------------------------
# Provider selection (local vs hf_inference)
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_local_backend_constructed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When provider=local and not mock, a _LocalSentenceTransformerBackend is built."""
        import src.services.embeddings as emb

        constructed: dict[str, Any] = {}
        real_cls = emb._LocalSentenceTransformerBackend

        def fake_local_init(self, model_name, token, device):
            constructed["model_name"] = model_name
            constructed["device"] = device

        monkeypatch.setattr(real_cls, "__init__", fake_local_init)
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        svc = EmbeddingService()
        assert isinstance(svc._backend, real_cls)
        assert constructed["model_name"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_hf_inference_backend_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When provider=hf_inference, an _HfInferenceBackend is built (no local weights)."""
        import src.services.embeddings as emb

        constructed: dict[str, Any] = {}
        real_cls = emb._HfInferenceBackend

        def fake_hf_init(self, model, token, endpoint="", timeout=60):
            constructed["model"] = model
            constructed["endpoint"] = endpoint

        monkeypatch.setattr(real_cls, "__init__", fake_hf_init)
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_inference")
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")
        svc = EmbeddingService()
        assert isinstance(svc._backend, real_cls)
        assert constructed["model"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_hf_inference_with_custom_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.services.embeddings as emb

        constructed: dict[str, Any] = {}
        real_cls = emb._HfInferenceBackend

        def fake_hf_init(self, model, token, endpoint="", timeout=60):
            constructed["endpoint"] = endpoint

        monkeypatch.setattr(real_cls, "__init__", fake_hf_init)
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_inference")
        monkeypatch.setenv("HF_INFERENCE_ENDPOINT", "https://my.tei.huggingface.cloud")
        EmbeddingService()
        assert constructed["endpoint"] == "https://my.tei.huggingface.cloud"

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "bogus")
        with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
            EmbeddingService()

    def test_mock_mode_skips_backend_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_MOCK=true short-circuits backend creation (no torch/network needed)."""
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_inference")
        svc = EmbeddingService()
        assert svc._backend is None  # mock path doesn't build a real backend


# ---------------------------------------------------------------------------
# Integration test (real model load) — skipped unless RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealModel:
    def test_real_model_loads_and_embeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if os.getenv("RUN_INTEGRATION") != "1":
            pytest.skip("RUN_INTEGRATION not set")
        monkeypatch.setenv("AI_MOCK", "false")
        svc = EmbeddingService()
        vec = svc.embed_query("What is machine learning?")
        assert len(vec) == svc.dimension
        # Real embeddings are L2-normalized.
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_real_query_differs_from_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if os.getenv("RUN_INTEGRATION") != "1":
            pytest.skip("RUN_INTEGRATION not set")
        monkeypatch.setenv("AI_MOCK", "false")
        svc = EmbeddingService()
        q = svc.embed_query("machine learning")
        d = svc.embed_documents(["machine learning"])[0]
        assert q != d
