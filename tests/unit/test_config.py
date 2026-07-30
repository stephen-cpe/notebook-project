"""Unit tests for src.config (TDD step 2)."""

from __future__ import annotations

import pytest

from src.config import Config, _bool, _int, _redact_secret, _redact_url


class TestEnvParsers:
    def test_bool_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("true", "1", "yes", "True", "YES"):
            monkeypatch.setenv("X", val)
            assert _bool("X", default=False) is True

    def test_bool_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("false", "0", "no", "", "anything-else"):
            monkeypatch.setenv("X", val)
            assert _bool("X", default=True) is False

    def test_bool_missing_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("X", raising=False)
        assert _bool("X", default=True) is True
        assert _bool("X", default=False) is False

    def test_int_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "42")
        assert _int("X", default=10) == 42

    def test_int_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "not-a-number")
        assert _int("X", default=99) == 99

    def test_int_missing_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("X", raising=False)
        assert _int("X", default=7) == 7


class TestConfigDefaults:
    def test_defaults_applied(self) -> None:
        cfg = Config()
        assert cfg.chat_model == "gemma4:31b-cloud"
        assert cfg.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
        assert cfg.embedding_dim == 1024
        assert cfg.embedding_provider == "local"
        assert cfg.hf_inference_endpoint == ""
        assert cfg.enable_thinking is True
        assert cfg.audio_voice_a == "en-US-AvaNeural"
        assert cfg.audio_voice_b == "en-US-AndrewNeural"
        assert cfg.audio_format == "mp3"
        assert cfg.max_sources_per_notebook == 50
        assert cfg.max_file_size_mb == 25
        assert cfg.chroma_db == "local"
        assert cfg.data_dir == "./data"
        assert cfg.chroma_cloud_api_key == ""
        assert cfg.chroma_cloud_connection_string == ""
        assert cfg.chroma_collection_name == "notebook-project-chromadb"
        assert cfg.hf_token == ""
        assert cfg.ocr_fallback_enabled is True
        assert cfg.ocr_provider == "local"
        assert cfg.ocr_inference_endpoint == ""

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAT_MODEL", "other-model")
        monkeypatch.setenv("EMBEDDING_DIM", "512")
        monkeypatch.setenv("MAX_SOURCES_PER_NOTEBOOK", "10")
        monkeypatch.setenv("ENABLE_THINKING", "false")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hf_inference")
        monkeypatch.setenv("HF_INFERENCE_ENDPOINT", "https://example.tei.huggingface.cloud")
        monkeypatch.setenv("OCR_PROVIDER", "hf_inference")
        monkeypatch.setenv("OCR_INFERENCE_ENDPOINT", "https://example.vlm.huggingface.cloud")
        monkeypatch.setenv("CHROMA_DB", "cloud")
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "ck-test-key")
        monkeypatch.setenv("CHROMA_CLOUD_CONNECTION_STRING", "tenant-123")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "my-chroma-db")
        cfg = Config()
        assert cfg.chat_model == "other-model"
        assert cfg.embedding_dim == 512
        assert cfg.max_sources_per_notebook == 10
        assert cfg.enable_thinking is False
        assert cfg.embedding_provider == "hf_inference"
        assert cfg.hf_inference_endpoint == "https://example.tei.huggingface.cloud"
        assert cfg.ocr_provider == "hf_inference"
        assert cfg.ocr_inference_endpoint == "https://example.vlm.huggingface.cloud"
        assert cfg.chroma_db == "cloud"
        assert cfg.chroma_cloud_api_key == "ck-test-key"
        assert cfg.chroma_cloud_connection_string == "tenant-123"
        assert cfg.chroma_collection_name == "my-chroma-db"

    def test_is_test_reflects_ci_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        assert Config().is_test() is True
        monkeypatch.setenv("CI", "false")
        assert Config().is_test() is False


class TestConfigRedaction:
    def test_redact_secret_set(self) -> None:
        assert _redact_secret("super-secret-key") == "<set>"

    def test_redact_secret_unset(self) -> None:
        assert _redact_secret("") == "<unset>"

    def test_redact_url_masks_password(self) -> None:
        url = "postgresql+psycopg2://user:p4ss@localhost:5432/db"
        redacted = _redact_url(url)
        assert "p4ss" not in redacted
        assert "***" in redacted
        assert "localhost:5432/db" in redacted

    def test_redact_url_without_password(self) -> None:
        url = "sqlite:///:memory:"
        assert _redact_url(url) == url

    def test_summary_never_leaks_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "ollama-secret")
        monkeypatch.setenv("HF_TOKEN", "hf-secret")
        monkeypatch.setenv("HF_INFERENCE_ENDPOINT", "https://secret.tei.huggingface.cloud")
        monkeypatch.setenv("OCR_INFERENCE_ENDPOINT", "https://secret.vlm.huggingface.cloud")
        monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "ck-secret-key")
        monkeypatch.setenv("CHROMA_CLOUD_CONNECTION_STRING", "secret-tenant")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:secretpw@host/db")
        s = Config().summary()
        joined = " ".join(str(v) for v in s.values())
        assert "ollama-secret" not in joined
        assert "hf-secret" not in joined
        assert "secretpw" not in joined
        assert "secret.tei.huggingface.cloud" not in joined
        assert "secret.vlm.huggingface.cloud" not in joined
        assert "ck-secret-key" not in joined
        assert "secret-tenant" not in joined
        assert s["ollama_cloud_api_key"] == "<set>"
        assert s["hf_token"] == "<set>"
        assert s["hf_inference_endpoint"] == "<set>"
        assert s["ocr_inference_endpoint"] == "<set>"
        assert s["chroma_cloud_api_key"] == "<set>"
        assert s["chroma_cloud_connection_string"] == "<set>"
