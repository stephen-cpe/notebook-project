"""Unit tests for src.services.ocr_service (TDD step 8).

GLM-OCR via HuggingFace transformers. Unit tests run fully offline using a
mock OCR that returns canned text; real model loading + image OCR is gated
behind ``RUN_INTEGRATION=1`` and marked ``@pytest.mark.integration``.

Covers:
- Mock mode returns deterministic text per prompt type.
- is_available() reflects config (OCR_FALLBACK_ENABLED).
- render_pdf_pages converts a PDF to PIL images (Poppler required).
- ocr_image extracts text from a single image (mock returns canned).
- ocr_pdf orchestrates render + ocr per page.
- Image resizing caps large images at OCR_MAX_IMAGE_DIMENSION.
- HF token handling mirrors embeddings (absent -> WARNING, never logged).
- Disabled mode short-circuits (returns empty, no model load).
- Prompt selection (Text/Formula/Table recognition).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from src.services.ocr_service import (
    OCR_PROMPT_FORMULA,
    OCR_PROMPT_TABLE,
    OCR_PROMPT_TEXT,
    OCRService,
    get_ocr_service,
    reset_ocr_service,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_ocr_state() -> None:
    reset_ocr_service()
    yield
    reset_ocr_service()


# ---------------------------------------------------------------------------
# Availability + config
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_disabled_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        assert svc.is_available() is False

    def test_enabled_mock_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        assert svc.is_available() is True

    def test_disabled_ocr_image_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        # Even with a real image, disabled OCR returns "".
        result = svc.ocr_image(FIXTURES / "empty.pdf", OCR_PROMPT_TEXT)
        assert result == ""


# ---------------------------------------------------------------------------
# Mock OCR
# ---------------------------------------------------------------------------


class TestMockOcr:
    def test_mock_ocr_image_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        # Pass a fake "image path" — mock doesn't actually read it.
        result = svc.ocr_image("fake_image.png", OCR_PROMPT_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_prompt_type_affects_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        text_result = svc.ocr_image("fake.png", OCR_PROMPT_TEXT)
        formula_result = svc.ocr_image("fake.png", OCR_PROMPT_FORMULA)
        table_result = svc.ocr_image("fake.png", OCR_PROMPT_TABLE)
        # Different prompts should produce different mock outputs.
        assert text_result != formula_result or text_result != table_result

    def test_mock_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        r1 = svc.ocr_image("same.png", OCR_PROMPT_TEXT)
        r2 = svc.ocr_image("same.png", OCR_PROMPT_TEXT)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PDF rendering + orchestration
# ---------------------------------------------------------------------------


class TestRenderPdfPages:
    def test_render_sample_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        images = svc.render_pdf_pages(str(FIXTURES / "sample.pdf"))
        assert len(images) == 2  # sample.pdf has 2 pages
        # Each should be a PIL Image-like object with a size.
        for img in images:
            assert hasattr(img, "size")
            assert img.size[0] > 0 and img.size[1] > 0

    def test_render_empty_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        images = svc.render_pdf_pages(str(FIXTURES / "empty.pdf"))
        assert len(images) == 1

    def test_render_passes_poppler_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys
        import types

        fake_pdf2image = types.ModuleType("pdf2image")
        calls: list[dict[str, Any]] = []

        def _fake_convert_from_path(path: str, **kwargs: Any) -> list[Any]:
            calls.append({"path": path, "kwargs": kwargs})
            img = types.SimpleNamespace(size=(10, 10))
            return [img]

        fake_pdf2image.convert_from_path = _fake_convert_from_path
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
        monkeypatch.setenv("POPPLER_PATH", str(tmp_path / "poppler-bin"))
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        images = svc.render_pdf_pages("whatever.pdf")
        assert len(images) == 1
        assert calls[0]["path"] == "whatever.pdf"
        assert calls[0]["kwargs"]["poppler_path"] == str(tmp_path / "poppler-bin")


class TestOcrPdf:
    def test_ocr_pdf_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        text = svc.ocr_pdf(str(FIXTURES / "sample.pdf"), OCR_PROMPT_TEXT)
        assert isinstance(text, str)
        assert len(text) > 0
        # Mock should have "OCRed" both pages.
        assert "page" in text.lower() or "ocr" in text.lower()

    def test_ocr_pdf_disabled_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        text = svc.ocr_pdf(str(FIXTURES / "sample.pdf"), OCR_PROMPT_TEXT)
        assert text == ""

    def test_ocr_pdf_no_pages_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        monkeypatch.setattr(svc, "render_pdf_pages", lambda pdf_path: [])
        assert svc.ocr_pdf("empty.pdf") == ""


# ---------------------------------------------------------------------------
# Mock OCR fallback prompt
# ---------------------------------------------------------------------------


class TestMockOcrFallbackPrompt:
    def test_unknown_prompt_returns_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        svc = OCRService()
        result = svc.ocr_image("fake.png", "Some Custom Prompt")
        assert result.startswith("[mock OCR ")
        assert "recognition" not in result.lower()


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------


class TestImageResizing:
    def test_resize_large_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_MAX_IMAGE_DIMENSION", "100")
        svc = OCRService()
        from PIL import Image

        large = Image.new("RGB", (2000, 1000), "white")
        resized = svc.resize_if_needed(large)
        max_dim = max(resized.size)
        assert max_dim <= 100

    def test_small_image_not_resized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_MAX_IMAGE_DIMENSION", "2048")
        svc = OCRService()
        from PIL import Image

        small = Image.new("RGB", (50, 50), "white")
        resized = svc.resize_if_needed(small)
        assert resized.size == (50, 50)


# ---------------------------------------------------------------------------
# HF token handling (mirrors embeddings)
# ---------------------------------------------------------------------------


class TestHfTokenHandling:
    def test_no_token_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "")
        caplog.set_level(logging.WARNING, logger="src.services.ocr_service")
        OCRService()
        assert any("HF_TOKEN not set" in r.message for r in caplog.records)

    def test_token_never_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("HF_TOKEN", "ocr-secret-token-abc")
        caplog.set_level(logging.DEBUG, logger="src.services.ocr_service")
        OCRService()
        all_text = " ".join(r.message for r in caplog.records)
        assert "ocr-secret-token-abc" not in all_text


# ---------------------------------------------------------------------------
# get_ocr_service singleton
# ---------------------------------------------------------------------------


class TestGetService:
    def test_returns_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        svc = get_ocr_service()
        assert isinstance(svc, OCRService)
        assert svc.is_available() is True

    def test_default_provider_is_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        svc = OCRService()
        assert svc.provider == "local"


# ---------------------------------------------------------------------------
# Provider selection (local vs hf_inference)
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_local_backend_constructed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When provider=local and not mock, a _LocalTransformersOcrBackend is built."""
        import src.services.ocr_service as ocr

        constructed: dict[str, Any] = {}
        real_cls = ocr._LocalTransformersOcrBackend

        def fake_local_init(self, token):
            constructed["token"] = token

        monkeypatch.setattr(real_cls, "__init__", fake_local_init)
        monkeypatch.setattr(real_cls, "ocr_image", lambda self, image, prompt: "mock local ocr")
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_PROVIDER", "local")
        svc = OCRService()
        # Trigger lazy backend construction by calling ocr_image.
        result = svc.ocr_image("fake.png", OCR_PROMPT_TEXT)
        assert isinstance(svc._backend, real_cls)
        assert result == "mock local ocr"

    def test_hf_inference_backend_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When provider=hf_inference, an _HfInferenceOcrBackend is built (no local weights)."""
        import src.services.ocr_service as ocr

        constructed: dict[str, Any] = {}
        real_cls = ocr._HfInferenceOcrBackend

        def fake_hf_init(self, model, token, endpoint="", timeout=60):
            constructed["model"] = model
            constructed["endpoint"] = endpoint

        monkeypatch.setattr(real_cls, "__init__", fake_hf_init)
        monkeypatch.setattr(real_cls, "ocr_image", lambda self, image, prompt: "mock hf ocr")
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_PROVIDER", "hf_inference")
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")
        svc = OCRService()
        result = svc.ocr_image("fake.png", OCR_PROMPT_TEXT)
        assert isinstance(svc._backend, real_cls)
        assert constructed["model"] == "zai-org/GLM-OCR"
        assert result == "mock hf ocr"

    def test_hf_inference_with_custom_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.services.ocr_service as ocr

        constructed: dict[str, Any] = {}
        real_cls = ocr._HfInferenceOcrBackend

        def fake_hf_init(self, model, token, endpoint="", timeout=60):
            constructed["endpoint"] = endpoint

        monkeypatch.setattr(real_cls, "__init__", fake_hf_init)
        monkeypatch.setattr(real_cls, "ocr_image", lambda self, image, prompt: "mock hf ocr")
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_PROVIDER", "hf_inference")
        monkeypatch.setenv("OCR_INFERENCE_ENDPOINT", "https://my.vlm.huggingface.cloud")
        svc = OCRService()
        svc.ocr_image("fake.png", OCR_PROMPT_TEXT)
        assert constructed["endpoint"] == "https://my.vlm.huggingface.cloud"

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_PROVIDER", "bogus")
        svc = OCRService()
        with pytest.raises(ValueError, match="Unknown OCR_PROVIDER"):
            svc.ocr_image("fake.png", OCR_PROMPT_TEXT)

    def test_mock_mode_skips_backend_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_MOCK=true short-circuits backend creation (no torch/network needed)."""
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        monkeypatch.setenv("OCR_PROVIDER", "hf_inference")
        svc = OCRService()
        assert svc._backend is None  # mock path doesn't build a real backend
        # But OCR still works via the mock.
        assert svc.ocr_image("fake.png", OCR_PROMPT_TEXT) != ""


# ---------------------------------------------------------------------------
# _LocalTransformersOcrBackend (heavy deps mocked)
# ---------------------------------------------------------------------------


class TestLocalTransformersBackend:
    def _fake_transformers(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
        import sys
        import types
        from unittest.mock import MagicMock

        fake = types.ModuleType("transformers")
        fake.AutoProcessor = MagicMock()
        fake.AutoModelForImageTextToText = MagicMock()
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return fake.AutoProcessor, fake.AutoModelForImageTextToText

    def test_load_model_success(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.ocr_service import _LocalTransformersOcrBackend

        processor, model = self._fake_transformers(monkeypatch)
        caplog.set_level(logging.INFO, logger="src.services.ocr_service")
        backend = _LocalTransformersOcrBackend(token="tok")
        assert backend._loaded is False
        backend._load_model()
        assert backend._loaded is True
        processor.from_pretrained.assert_called_once()
        model.from_pretrained.assert_called_once()
        assert any("Loaded GLM-OCR model" in r.message for r in caplog.records)

    def test_load_model_retries_without_token_on_401(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.ocr_service import _LocalTransformersOcrBackend

        processor, model = self._fake_transformers(monkeypatch)
        processor.from_pretrained.side_effect = [
            Exception("HTTPError 401 Unauthorized"),
            "processor-without-token",
        ]
        caplog.set_level(logging.ERROR, logger="src.services.ocr_service")
        backend = _LocalTransformersOcrBackend(token="bad-token")
        backend._load_model()
        assert backend._loaded is True
        assert processor.from_pretrained.call_count == 2
        assert model.from_pretrained.call_count == 1
        assert any("status=401" in r.message for r in caplog.records)

    def test_load_model_raises_on_other_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.ocr_service import _LocalTransformersOcrBackend

        processor, _ = self._fake_transformers(monkeypatch)
        processor.from_pretrained.side_effect = Exception("disk full")
        backend = _LocalTransformersOcrBackend(token="tok")
        with pytest.raises(Exception, match="disk full"):
            backend._load_model()
        assert backend._loaded is False

    def test_ocr_image_runs_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types
        from unittest.mock import MagicMock

        from src.services.ocr_service import _LocalTransformersOcrBackend

        self._fake_transformers(monkeypatch)
        fake_torch = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self) -> _NoGrad:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

        fake_torch.no_grad = _NoGrad
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        backend = _LocalTransformersOcrBackend(token="tok")
        backend._loaded = True
        backend._model = MagicMock()
        backend._model.device = "cpu"
        backend._processor = MagicMock()
        inputs = {"input_ids": MagicMock()}
        backend._processor.apply_chat_template.return_value.to.return_value = inputs
        backend._model.generate.return_value = MagicMock()
        backend._processor.decode.return_value = "OCR text output"

        result = backend.ocr_image("image-placeholder", OCR_PROMPT_TEXT)
        assert result == "OCR text output"
        backend._model.generate.assert_called_once()
        backend._processor.apply_chat_template.assert_called_once()


# ---------------------------------------------------------------------------
# _HfInferenceOcrBackend (network mocked)
# ---------------------------------------------------------------------------


class TestHfInferenceBackend:
    def _fake_hub(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
        import sys
        import types
        from unittest.mock import MagicMock

        fake = types.ModuleType("huggingface_hub")
        constructed: list[dict[str, Any]] = []

        class FakeInferenceClient:
            def __init__(
                self,
                model: str,
                token: str | None = None,
                base_url: str | None = None,
                timeout: int = 60,
            ) -> None:
                constructed.append(
                    {"model": model, "token": token, "base_url": base_url, "timeout": timeout}
                )
                self.chat_completion = MagicMock()

        fake.InferenceClient = FakeInferenceClient
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        return fake, constructed

    def test_constructs_client(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.ocr_service import _HfInferenceOcrBackend

        _, constructed = self._fake_hub(monkeypatch)
        caplog.set_level(logging.INFO, logger="src.services.ocr_service")
        backend = _HfInferenceOcrBackend(model="m", token="tok", endpoint="https://e", timeout=42)
        assert backend._model == "m"
        assert constructed[0]["model"] == "m"
        assert constructed[0]["token"] == "tok"
        assert constructed[0]["base_url"] == "https://e"
        assert constructed[0]["timeout"] == 42
        assert any("Configured HF Inference API" in r.message for r in caplog.records)

    def test_warns_when_no_token(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.services.ocr_service import _HfInferenceOcrBackend

        self._fake_hub(monkeypatch)
        caplog.set_level(logging.WARNING, logger="src.services.ocr_service")
        _HfInferenceOcrBackend(model="m", token="")
        assert any("HF_TOKEN is not set" in r.message for r in caplog.records)

    def test_ocr_image_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        from src.services.ocr_service import _HfInferenceOcrBackend

        _, constructed = self._fake_hub(monkeypatch)
        backend = _HfInferenceOcrBackend(model="m", token="tok")
        reply = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="OCR result"))]
        )
        constructed[0]["client"] = backend._client
        backend._client.chat_completion.return_value = reply

        result = backend.ocr_image(b"\x89PNG\r\nbinary", OCR_PROMPT_TEXT)
        assert result == "OCR result"
        backend._client.chat_completion.assert_called_once()

    def test_ocr_image_malformed_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        from src.services.ocr_service import _HfInferenceOcrBackend

        self._fake_hub(monkeypatch)
        backend = _HfInferenceOcrBackend(model="m", token="tok")
        backend._client.chat_completion.return_value = types.SimpleNamespace()
        assert backend.ocr_image(b"raw", OCR_PROMPT_TEXT) == ""

    def test_call_with_retry_rate_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:

        from src.services.ocr_service import _HfInferenceOcrBackend

        self._fake_hub(monkeypatch)
        backend = _HfInferenceOcrBackend(model="m", token="tok")
        monkeypatch.setattr("time.sleep", lambda s: None)

        calls = 0

        def _flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("429 rate limit hit")  # noqa: TRY002
            return "ok"

        result = backend._call_with_retry(_flaky)
        assert result == "ok"
        assert calls == 2

    def test_call_with_retry_transient_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.ocr_service import _HfInferenceOcrBackend

        self._fake_hub(monkeypatch)
        backend = _HfInferenceOcrBackend(model="m", token="tok")
        monkeypatch.setattr("time.sleep", lambda s: None)

        calls = 0

        def _flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("request timed out")
            return "ok"

        assert backend._call_with_retry(_flaky) == "ok"
        assert calls == 2

    def test_call_with_retry_reraises_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.ocr_service import _HfInferenceOcrBackend

        self._fake_hub(monkeypatch)
        backend = _HfInferenceOcrBackend(model="m", token="tok")

        def _boom() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            backend._call_with_retry(_boom)


# ---------------------------------------------------------------------------
# _pil_to_data_url + _extract_hf_status
# ---------------------------------------------------------------------------


class TestPilToDataUrl:
    def test_pil_image_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from PIL import Image

        from src.services.ocr_service import _pil_to_data_url

        img = Image.new("RGB", (4, 4), "white")
        url = _pil_to_data_url(img)
        assert url.startswith("data:image/png;base64,")

    def test_bytes_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.ocr_service import _pil_to_data_url

        url = _pil_to_data_url(b"raw-bytes")
        assert url == "data:application/octet-stream;base64," + "cmF3LWJ5dGVz"

    def test_file_like_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        from src.services.ocr_service import _pil_to_data_url

        buf = io.BytesIO(b"file-like-data")
        url = _pil_to_data_url(buf)
        assert url == "data:application/octet-stream;base64," + "ZmlsZS1saWtlLWRhdGE="

    def test_path_input(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from src.services.ocr_service import _pil_to_data_url

        f = tmp_path / "img.bin"
        f.write_bytes(b"path-data")
        url = _pil_to_data_url(str(f))
        assert url == "data:application/octet-stream;base64," + "cGF0aC1kYXRh"


class TestExtractHfStatus:
    def test_401_and_403(self) -> None:
        from src.services.ocr_service import _extract_hf_status

        assert _extract_hf_status(Exception("HTTP 401 Unauthorized")) == 401
        assert _extract_hf_status(Exception("Forbidden 403")) == 403
        assert _extract_hf_status(Exception("disk full")) is None


# ---------------------------------------------------------------------------
# Integration test (real GLM-OCR model) — skipped unless RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealOcr:
    def test_real_ocr_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        if os.getenv("RUN_INTEGRATION") != "1":
            pytest.skip("RUN_INTEGRATION not set")
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
        # Render a page from sample.pdf and OCR it.
        svc = OCRService()
        images = svc.render_pdf_pages(str(FIXTURES / "sample.pdf"))
        assert len(images) > 0
        text = svc.ocr_image(images[0], OCR_PROMPT_TEXT)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain some recognizable words from the fixture.
        assert "learning" in text.lower() or "fox" in text.lower()
