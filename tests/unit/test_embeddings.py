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


# ---------------------------------------------------------------------------
# Real backend paths (heavy deps / network mocked)
# ---------------------------------------------------------------------------


class TestRealBackendDispatch:
    def test_embed_query_uses_backend(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.embeddings import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._mock = False
        svc._backend = _FakeBackend()
        caplog.set_level(logging.INFO, logger="src.services.embeddings")
        vec = svc.embed_query("hello world")
        assert vec == [0.5, 1.5]
        assert any("embed_query" in r.message for r in caplog.records)

    def test_embed_documents_uses_backend(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.embeddings import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._mock = False
        svc._backend = _FakeBackend()
        caplog.set_level(logging.INFO, logger="src.services.embeddings")
        vecs = svc.embed_documents(["a", "b"])
        assert vecs == [[1.0], [2.0]]
        assert any("embed_documents" in r.message for r in caplog.records)

    def test_detect_device_cpu_on_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        assert EmbeddingService._detect_device() == "cpu"

    def test_detect_device_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert EmbeddingService._detect_device() == "cuda"


class _FakeBackend:
    """Minimal in-memory backend for dispatch tests."""

    def embed_query(self, text: str) -> list[float]:
        return [0.5, 1.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] for i in range(len(texts))]


class _ArrLike:
    """Array-like stub: supports squeeze(), ndim, indexing and iteration."""

    def __init__(self, values: list[float], ndim: int = 1) -> None:
        self._values = values
        self.ndim = ndim

    def squeeze(self) -> _ArrLike:
        return self

    def __getitem__(self, index: object) -> list[float]:
        return self._values

    def __iter__(self):
        return iter(self._values)


class TestToFloatList:
    def test_with_tolist(self) -> None:
        from src.services.embeddings import EmbeddingService

        class _NumpyLike:
            def tolist(self) -> list[float]:
                return [1.0, 2.0]

        assert EmbeddingService._to_float_list(_NumpyLike()) == [1.0, 2.0]

    def test_without_tolist(self) -> None:
        from src.services.embeddings import EmbeddingService

        assert EmbeddingService._to_float_list([3, 4]) == [3.0, 4.0]


# ---------------------------------------------------------------------------
# _LocalSentenceTransformerBackend (sentence_transformers mocked)
# ---------------------------------------------------------------------------


class TestLocalSentenceTransformerBackend:
    def test_load_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types
        from unittest.mock import MagicMock

        from src.services.embeddings import _LocalSentenceTransformerBackend

        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = MagicMock(return_value=MagicMock())
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        backend = _LocalSentenceTransformerBackend("m", "tok", "cpu")
        assert backend._model is not None
        fake.SentenceTransformer.assert_called_once()

    def test_load_retries_without_token_on_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types
        from unittest.mock import MagicMock

        from src.services.embeddings import _LocalSentenceTransformerBackend

        fake = types.ModuleType("sentence_transformers")
        real = MagicMock()
        real.side_effect = [Exception("HTTPError 401 Unauthorized"), MagicMock()]
        fake.SentenceTransformer = real
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        backend = _LocalSentenceTransformerBackend("m", "tok", "cpu")
        assert real.call_count == 2
        assert backend._model is not None

    def test_load_raises_on_other_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types
        from unittest.mock import MagicMock

        from src.services.embeddings import _LocalSentenceTransformerBackend

        fake = types.ModuleType("sentence_transformers")
        real = MagicMock()
        real.side_effect = Exception("disk full")
        fake.SentenceTransformer = real
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        with pytest.raises(Exception, match="disk full"):
            _LocalSentenceTransformerBackend("m", "tok", "cpu")

    def test_embed_query_and_documents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types
        from unittest.mock import MagicMock

        from src.services.embeddings import _LocalSentenceTransformerBackend

        fake = types.ModuleType("sentence_transformers")
        model = MagicMock()
        model.encode.side_effect = [[0.1, 0.2], [[0.3], [0.4]]]
        fake.SentenceTransformer = MagicMock(return_value=model)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        backend = _LocalSentenceTransformerBackend("m", "tok", "cpu")
        assert backend.embed_query("q") == [0.1, 0.2]
        assert backend.embed_documents(["a", "b"]) == [[0.3], [0.4]]
        model.encode.assert_called()


# ---------------------------------------------------------------------------
# _HfInferenceBackend (network mocked)
# ---------------------------------------------------------------------------


class TestHfInferenceBackend:
    def _fake_hub(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
        import sys
        import types
        from unittest.mock import MagicMock

        fake = types.ModuleType("huggingface_hub")
        constructed: list[dict[str, Any]] = []

        class FakeInferenceClient:
            def __init__(self, model: str, token: str | None = None, base_url: str | None = None, timeout: int = 60) -> None:
                constructed.append(
                    {"model": model, "token": token, "base_url": base_url, "timeout": timeout}
                )
                self.feature_extraction = MagicMock()

        fake.InferenceClient = FakeInferenceClient
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        return fake, constructed

    def test_constructs_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        _, constructed = self._fake_hub(monkeypatch)
        backend = _HfInferenceBackend(model="m", token="tok", endpoint="https://e", timeout=9)
        assert constructed[0]["model"] == "m"
        assert constructed[0]["token"] == "tok"
        assert constructed[0]["base_url"] == "https://e"
        assert constructed[0]["timeout"] == 9
        assert backend._token == "tok"

    def test_warns_when_no_token(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.embeddings import _HfInferenceBackend

        self._fake_hub(monkeypatch)
        caplog.set_level(logging.WARNING, logger="src.services.embeddings")
        _HfInferenceBackend(model="m", token="")
        assert any("HF_TOKEN is not set" in r.message for r in caplog.records)

    def test_embed_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        _, constructed = self._fake_hub(monkeypatch)
        backend = _HfInferenceBackend(model="m", token="tok")
        client = backend._client
        client.feature_extraction.return_value = _ArrLike([0.7, 0.8])
        vec = backend.embed_query("question")
        assert vec == [0.7, 0.8]
        client.feature_extraction.assert_called_once()

    def test_embed_documents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        _, constructed = self._fake_hub(monkeypatch)
        backend = _HfInferenceBackend(model="m", token="tok")
        client = backend._client
        client.feature_extraction.return_value = _ArrLike([1.0])
        assert backend.embed_documents(["a", "b"]) == [[1.0], [1.0]]
        assert client.feature_extraction.call_count == 2

    def test_call_with_retry_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        self._fake_hub(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda s: None)
        calls = 0

        def _flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("429 rate limit")  # noqa: TRY002
            return "ok"

        assert _HfInferenceBackend._call_with_retry(_flaky) == "ok"
        assert calls == 2

    def test_call_with_retry_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        self._fake_hub(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda s: None)
        calls = 0

        def _flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("connection reset")
            return "ok"

        assert _HfInferenceBackend._call_with_retry(_flaky) == "ok"
        assert calls == 2

    def test_call_with_retry_reraises_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.embeddings import _HfInferenceBackend

        self._fake_hub(monkeypatch)

        def _boom() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _HfInferenceBackend._call_with_retry(_boom)

    def test_flatten_multidim(self) -> None:
        from src.services.embeddings import _HfInferenceBackend

        vec = _HfInferenceBackend._flatten(_ArrLike([0.1, 0.2], ndim=2))
        assert vec == [0.1, 0.2]

    def test_flatten_no_squeeze(self) -> None:
        from src.services.embeddings import _HfInferenceBackend

        assert _HfInferenceBackend._flatten([5.0, 6.0]) == [5.0, 6.0]


class TestExtractHfStatusEmbeddings:
    def test_statuses(self) -> None:
        from src.services.embeddings import _extract_hf_status

        assert _extract_hf_status(Exception("HTTP 401 Unauthorized")) == 401
        assert _extract_hf_status(Exception("Forbidden 403")) == 403
        assert _extract_hf_status(Exception("boom")) is None
