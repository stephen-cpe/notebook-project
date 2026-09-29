"""OCR / vision service — ``glm-5.3-flash`` via Ollama Cloud.

The old HuggingFace ``GLM-OCR`` (``zai-org/GLM-OCR``) path has been replaced:
all image OCR, table extraction, and figure description now use a single
natively multimodal vision model (``OLLAMA_VISION_MODEL``, default
``glm-5.3-flash:cloud``) through the existing Ollama Cloud chat API
(``OllamaClient.chat_with_images``).

Behavior (public API unchanged):
- ``OCR_FALLBACK_ENABLED=false`` (or ``AI_MOCK=true`` with no real call):
  ``is_available()`` returns False; ``ocr_image``/``ocr_pdf`` return "".
- ``AI_MOCK=true`` + enabled: a deterministic mock returns canned text per
  prompt type (offline, no network).
- Real mode: renders PDFs via Poppler, encodes images as base64, and calls
  the vision model. Failures degrade to "" (never raise from public API
  except for unknown ``OCR_PROVIDER``).

``OCR_PROVIDER`` / ``OCR_INFERENCE_ENDPOINT`` / ``HF_TOKEN`` are kept as
deprecated aliases so old ``.env`` files and tests keep importing; they no
longer select the backend. The old ``_LocalTransformersOcrBackend`` and
``_HfInferenceOcrBackend`` classes are kept as deprecated shims so their
isolated unit tests keep passing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from typing import TYPE_CHECKING, Any, Protocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config

OCR_PROMPT_TEXT = "Text Recognition:"
OCR_PROMPT_FORMULA = "Formula Recognition:"
OCR_PROMPT_TABLE = "Table Recognition:"

# Deprecated alias: the HF GLM-OCR model id is no longer used for inference.
# Kept so old imports keep working.
GLM_OCR_MODEL = "zai-org/GLM-OCR"

# Default vision model (mirrors Config default).
VISION_MODEL_DEFAULT = "glm-5.3-flash:cloud"

# Natural-language prompts sent to the vision model per OCR prompt type.
_VISION_PROMPTS = {
    OCR_PROMPT_TEXT: (
        "Extract all text visible in this image. Return only the extracted "
        "text, no commentary. Preserve reading order."
    ),
    OCR_PROMPT_TABLE: (
        "Extract any tables visible in this image. Preserve row/column "
        "structure using GitHub-Flavored Markdown pipe tables. Return only "
        "the table content."
    ),
    OCR_PROMPT_FORMULA: (
        "Extract any mathematical formulas visible in this image. Return "
        "them using LaTeX ($...$ / $$...$$). Return only the formulas."
    ),
}

_VISION_FIGURE_PROMPT = (
    "Describe the figure, diagram, or chart visible in this image in 2-3 "
    "sentences, focusing on visual elements, labels, axes, and structure."
)

_token_warning_emitted = False


class _OcrBackend(Protocol):
    """Internal contract for the real (non-mock) OCR backends."""

    def ocr_image(self, image: Any, prompt: str) -> str: ...  # noqa: ANN401


def _vision_prompt_for(prompt: str) -> str:
    """Map a legacy OCR prompt constant to a vision-model prompt."""
    return _VISION_PROMPTS.get(prompt, _VISION_PROMPTS[OCR_PROMPT_TEXT])


class OCRService:
    """Vision OCR wrapper with mock support + lazy backend selection."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self._enabled: bool = bool(config.ocr_fallback_enabled)
        self._mock: bool = bool(config.ai_mock)
        self._max_dim: int = config.ocr_max_image_dimension
        self._max_pages: int = max(1, config.ocr_max_pages)
        self._dpi: int = max(72, config.ocr_dpi)
        self._poppler_path: str = config.poppler_path
        self._hf_token: str = config.hf_token
        self.provider: str = config.ocr_provider
        self.vision_model: str = config.vision_model
        self._backend: _OcrBackend | None = None

        self._handle_hf_token()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if OCR is enabled (mock or real)."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ocr_image(self, image: Any, prompt: str = OCR_PROMPT_TEXT) -> str:  # noqa: ANN401
        """OCR a single PIL image with the given prompt. Returns "" if disabled."""
        if not self._enabled:
            return ""
        if self._mock:
            return self._mock_ocr(image, prompt)
        if self._backend is None:
            self._backend = self._make_backend()
        try:
            return self._backend.ocr_image(image, prompt)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision OCR failed: %s", exc)
            return ""

    def ocr_pdf(self, pdf_path: str, prompt: str = OCR_PROMPT_TEXT) -> str:
        """Render a PDF to images and OCR each page. Returns "" if disabled.

        At most ``OCR_MAX_PAGES`` pages are rendered; any remainder is
        reported in a trailing note so callers know content was skipped.
        """
        if not self._enabled:
            return ""
        images = self.render_pdf_pages(pdf_path)
        if not images:
            return ""
        parts: list[str] = []
        for i, img in enumerate(images):
            text = self.ocr_image(img, prompt)
            if text:
                parts.append(f"[Page {i + 1}]\n{text}")
        total = self.pdf_page_count(pdf_path)
        skipped = total - len(images) if total > len(images) else 0
        if skipped > 0:
            parts.append(
                f"[OCR processed {len(images)} of {total} pages; "
                f"{skipped} skipped (OCR_MAX_PAGES={self._max_pages})]"
            )
        return "\n\n".join(parts)

    def ocr_images(self, images: list[Any], prompt: str = OCR_PROMPT_TEXT) -> str:  # noqa: ANN401
        """OCR a list of PIL images (e.g. embedded DOCX/PPTX images).

        Returns concatenated text with per-image ``[Image N]`` headers, or "" if
        disabled or the image list is empty.
        """
        if not self._enabled or not images:
            return ""
        parts: list[str] = []
        for i, img in enumerate(images):
            text = self.ocr_image(img, prompt)
            if text:
                parts.append(f"[Image {i + 1}]\n{text}")
        return "\n\n".join(parts)

    def describe_figure(self, image: Any) -> str:  # noqa: ANN401
        """Describe a figure/diagram image in 2-3 sentences ("" if disabled)."""
        if not self._enabled:
            return ""
        if self._mock:
            return self._mock_ocr(image, _VISION_FIGURE_PROMPT)
        if self._backend is None:
            self._backend = self._make_backend()
        backend = self._backend
        if isinstance(backend, _OllamaVisionBackend):
            try:
                return backend.ocr_image_with_prompt(image, _VISION_FIGURE_PROMPT)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Figure description failed: %s", exc)
                return ""
        return self.ocr_image(image, OCR_PROMPT_TEXT)

    def render_pdf_pages(self, pdf_path: str, max_pages: int | None = None) -> list[Any]:
        """Convert a PDF to a list of PIL images using pdf2image + Poppler.

        Renders at ``OCR_DPI`` and at most ``max_pages`` pages (default
        ``OCR_MAX_PAGES``) so a large scanned PDF cannot exhaust memory by
        rasterizing every page at once.
        """
        from pdf2image import convert_from_path

        limit = self._max_pages if max_pages is None else max(1, max_pages)
        kwargs: dict[str, Any] = {"dpi": self._dpi}
        if self._poppler_path:
            kwargs["poppler_path"] = self._poppler_path
        images = convert_from_path(pdf_path, first_page=1, last_page=limit, **kwargs)
        return [self.resize_if_needed(img) for img in images]

    @staticmethod
    def pdf_page_count(pdf_path: str) -> int:
        """Return the number of pages in a PDF (0 when unreadable)."""
        try:
            from pypdf import PdfReader

            return len(PdfReader(pdf_path).pages)
        except Exception:  # noqa: BLE001
            logger.warning("Could not read page count for %s", pdf_path)
            return 0

    def resize_if_needed(self, image: Any) -> Any:  # noqa: ANN401
        """Resize an image so its largest dimension <= OCR_MAX_IMAGE_DIMENSION."""
        w, h = image.size
        max_dim = max(w, h)
        if max_dim <= self._max_dim:
            return image
        scale = self._max_dim / max_dim
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _make_backend(self) -> _OcrBackend:
        """Select the vision backend.

        ``OCR_PROVIDER`` is deprecated: ``local`` / ``hf_inference`` /
        ``vision`` / ``ollama`` / ``cloud`` all resolve to the consolidated
        Ollama vision backend. Anything else raises ``ValueError`` so a typo
        is never silently ignored.
        """
        provider = (self.provider or "").strip().lower()
        if provider in ("local", "hf_inference", "vision", "ollama", "cloud", ""):
            if provider in ("local", "hf_inference"):
                logger.info(
                    "OCR_PROVIDER=%r is deprecated; using consolidated vision "
                    "model %r instead of HuggingFace GLM-OCR.",
                    self.provider,
                    self.vision_model,
                )
            return _OllamaVisionBackend(
                model=self.vision_model,
                timeout=getattr(self._config, "vision_timeout", 300),
            )
        raise ValueError(
            f"Unknown OCR_PROVIDER={self.provider!r}. "
            "Expected 'local', 'hf_inference' (both deprecated, use vision), "
            "or 'vision' (glm-5.3-flash via Ollama Cloud)."
        )

    # ------------------------------------------------------------------
    # Mock OCR (deterministic, offline)
    # ------------------------------------------------------------------

    def _mock_ocr(self, image: Any, prompt: str) -> str:  # noqa: ANN401
        """Produce deterministic canned text per prompt type."""
        # Hash stable image bytes (not repr(), which embeds a memory address
        # for PIL Images and would differ on every call).
        try:
            image_bytes = bytes(image.tobytes())
        except Exception:  # noqa: BLE001
            image_bytes = repr(image).encode("utf-8")
        digest = hashlib.sha256(image_bytes + prompt.encode("utf-8")).hexdigest()[:8]
        if prompt == OCR_PROMPT_TEXT:
            return (
                f"[mock OCR text recognition page {digest}] "
                "The document contains searchable content extracted via mock OCR."
            )
        if prompt == OCR_PROMPT_FORMULA:
            return f"[mock OCR formula recognition {digest}] E = mc^2 (mock formula)"
        if prompt == OCR_PROMPT_TABLE:
            return f"[mock OCR table recognition {digest}] | Col A | Col B |\n| 1 | 2 |"
        return f"[mock OCR {digest}]"

    # ------------------------------------------------------------------
    # HF token handling (deprecated; kept so the absence warning still fires
    # for old .env files and the token is never logged).
    # ------------------------------------------------------------------

    def _handle_hf_token(self) -> None:
        global _token_warning_emitted
        if not self._hf_token and not _token_warning_emitted:
            logger.warning(
                "HF_TOKEN not set; HuggingFace calls will be unauthenticated. "
                "You may see rate-limit warnings. Set HF_TOKEN (READ scope) "
                "in .env to suppress them."
            )
            _token_warning_emitted = True


# ----------------------------------------------------------------------
# Vision backend (consolidated glm-5.3-flash via Ollama Cloud)
# ----------------------------------------------------------------------


class _OllamaVisionBackend:
    """Vision backend: Ollama Cloud ``/api/chat`` with images.

    No local weights are required. Each call is an HTTPS request to the
    configured Ollama Cloud host. Requires ``OLLAMA_CLOUD_API_KEY``; without
    it the API returns 4xx and the error is surfaced (not silently empty).
    """

    def __init__(self, model: str = VISION_MODEL_DEFAULT, timeout: int = 300) -> None:
        from src.services.ollama_client import get_ollama_client

        self._model = model or VISION_MODEL_DEFAULT
        self._timeout = timeout
        self._client = get_ollama_client()
        logger.info(
            "Configured Ollama vision OCR backend (model=%s, timeout=%ss)",
            self._model,
            timeout,
        )

    def ocr_image(self, image: Any, prompt: str = OCR_PROMPT_TEXT) -> str:  # noqa: ANN401
        """OCR a PIL image with a legacy prompt constant."""
        return self.ocr_image_with_prompt(image, _vision_prompt_for(prompt))

    def ocr_image_with_prompt(self, image: Any, prompt: str) -> str:  # noqa: ANN401
        """OCR a PIL image with a full natural-language prompt."""
        b64 = _pil_to_b64(image)
        if not b64:
            return ""
        out = self._call_with_retry(prompt, b64)
        return out.strip() if out else ""

    def _call_with_retry(self, prompt: str, b64: str) -> str:
        """Call the vision API with backoff; pause (not fail) on long backoff."""
        from src.services.resilience import PauseJob, resilient

        def _call() -> str:
            return self._client.chat_with_images(prompt, [b64], model=self._model)

        try:
            result: str = resilient(_call, max_attempts=3, base_delay=2.0)
            return result
        except PauseJob as exc:
            logger.error("Vision OCR paused (rate-limited): %s", exc)
            return ""


def _pil_to_b64(image: Any) -> str:  # noqa: ANN401
    """Encode a PIL Image (or bytes-like) as a base64 PNG string (no prefix)."""
    buf = io.BytesIO()
    try:
        image.save(buf, format="PNG")
    except AttributeError:
        if isinstance(image, (bytes, bytearray)):
            return base64.b64encode(bytes(image)).decode()
        if hasattr(image, "read"):
            raw = image.read()
        else:
            with open(image, "rb") as fh:  # noqa: SIM115
                raw = fh.read()
        return base64.b64encode(raw).decode()
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------
# Deprecated HuggingFace backends (kept so old unit tests keep passing)
# ----------------------------------------------------------------------


class _LocalTransformersOcrBackend:
    """Deprecated: local ``transformers`` GLM-OCR backend (kept for tests)."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._model: Any = None
        self._processor: Any = None
        self._loaded: bool = False

    def _load_model(self) -> None:
        """Lazy-load the GLM-OCR model + processor (heavy)."""
        if self._loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        token = self._token or None
        try:
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                GLM_OCR_MODEL, token=token
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                GLM_OCR_MODEL, torch_dtype="auto", device_map="auto"
            )
            self._loaded = True
            logger.info("Loaded GLM-OCR model (local)")
        except Exception as exc:  # noqa: BLE001
            status = _extract_hf_status(exc)
            if status in (401, 403):
                logger.error(
                    "HuggingFace rejected HF_TOKEN for GLM-OCR (status=%s). "
                    "Proceeding unauthenticated.",
                    status,
                )
                self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                    GLM_OCR_MODEL, token=None
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    GLM_OCR_MODEL, torch_dtype="auto", device_map="auto"
                )
                self._loaded = True
            else:
                logger.error("Failed to load GLM-OCR: %s", exc)
                raise

    def ocr_image(self, image: Any, prompt: str) -> str:  # noqa: ANN401
        """Run the loaded GLM-OCR model on a single image."""
        import torch

        self._load_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=8192)
        output_text: str = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )
        return output_text


class _HfInferenceOcrBackend:
    """Deprecated: HF Inference API GLM-OCR backend (kept for tests)."""

    def __init__(self, model: str, token: str, endpoint: str = "", timeout: int = 60) -> None:
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(
            model=model, token=token or None, base_url=endpoint or None, timeout=timeout
        )
        self._model = model
        self._token = token
        logger.info(
            "Configured HF Inference API OCR backend (model=%s, endpoint=%s, timeout=%ss)",
            model,
            endpoint or "default-router",
            timeout,
        )
        if not token:
            logger.warning(
                "OCR_PROVIDER=hf_inference but HF_TOKEN is not set. "
                "Calls will be unauthenticated and rate-limited."
            )

    def ocr_image(self, image: Any, prompt: str) -> str:  # noqa: ANN401
        data_url = _pil_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        def _call() -> Any:  # noqa: ANN401
            return self._client.chat_completion(messages=messages, max_tokens=8192)

        out = self._call_with_retry(_call)
        try:
            return str(out.choices[0].message.content)
        except (AttributeError, IndexError, TypeError) as exc:
            logger.error("Unexpected GLM-OCR inference response shape: %s", exc)
            return ""

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


# ----------------------------------------------------------------------


def _pil_to_data_url(image: Any) -> str:  # noqa: ANN401
    """Encode a PIL Image (or bytes-like) as a base64 PNG data URL."""
    buf = io.BytesIO()
    # PIL Image.save; fall back to raw bytes if `image` is already bytes/path.
    try:
        image.save(buf, format="PNG")
    except AttributeError:
        # Already bytes or a file-like object.
        if isinstance(image, (bytes, bytearray)):
            return "data:application/octet-stream;base64," + base64.b64encode(image).decode()
        # Path-like or file-like: read raw bytes.
        if hasattr(image, "read"):
            raw = image.read()
        else:
            with open(image, "rb") as fh:  # noqa: SIM115
                raw = fh.read()
        return "data:application/octet-stream;base64," + base64.b64encode(raw).decode()
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def _extract_hf_status(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status from a HF exception."""
    msg = str(exc).lower()
    if "401" in msg:
        return 401
    if "403" in msg:
        return 403
    return None


_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    """Return a process-wide ``OCRService`` (created lazily)."""
    global _service
    if _service is None:
        _service = OCRService()
    return _service


def reset_ocr_service() -> None:
    """Reset the cached service + warning flag (used by tests)."""
    global _service, _token_warning_emitted
    _service = None
    _token_warning_emitted = False


def pdf_needs_vision_ocr(
    file_path: str,
    basic_text: str = "",
    min_total_chars: int = 1000,
    min_chars_per_page: int = 300,
) -> bool:
    """Decide whether a PDF actually needs vision OCR (smart gate).

    Text-layer PDFs skip expensive rendering + LLM calls entirely; vision
    runs only when the PDF looks scanned/image-heavy: almost no extractable
    text, sparse text per page, or embedded raster images outnumbering pages.
    Never raises — on any inspection error returns True (run vision rather
    than silently dropping content).
    """
    try:
        total_chars = len((basic_text or "").strip())
        if total_chars < min_total_chars:
            return True
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        num_pages = len(reader.pages) or 1
        if (total_chars / num_pages) < min_chars_per_page:
            return True
        try:
            image_count = _count_embedded_images(reader)
            if image_count >= num_pages and num_pages > 0:
                logger.info(
                    "PDF %s has %d embedded images across %d pages — "
                    "enabling vision OCR for figures/diagrams",
                    file_path,
                    image_count,
                    num_pages,
                )
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("PDF image-count probe failed for %s: %s", file_path, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF vision-OCR gate failed for %s: %s", file_path, str(exc))
        return True


def _count_embedded_images(reader: Any) -> int:  # noqa: ANN401
    """Count embedded raster images across all PDF pages (best-effort)."""
    import contextlib

    image_count = 0
    for page in reader.pages:
        resources = page.get("/Resources")
        if not resources:
            continue
        xobjects = resources.get("/XObject")
        if not xobjects:
            continue
        with contextlib.suppress(Exception):
            xobjects = xobjects.get_object()
        if not hasattr(xobjects, "keys"):
            continue
        for key in xobjects:
            try:
                obj = xobjects[key]
                with contextlib.suppress(Exception):
                    obj = obj.get_object()
                subtype = obj.get("/Subtype") if hasattr(obj, "get") else None
                if str(subtype) == "/Image":
                    image_count += 1
            except Exception:  # noqa: BLE001, S112
                logger.debug("Skipping unreadable XObject %s", key)
                continue
    return image_count
