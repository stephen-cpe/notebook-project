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
