"""OCR service — GLM-OCR via HuggingFace transformers OR HF Inference API.

GLM-OCR (``zai-org/GLM-OCR``) is a multimodal OCR model for documents/images.
By default it runs locally via ``transformers.AutoProcessor`` +
``AutoModelForImageTextToText``. Set ``OCR_PROVIDER=hf_inference`` to instead
use Hugging Face's hosted Inference API (chat-completion with image content) —
no local weights, per-call network latency, requires ``HF_TOKEN``.

Behavior:
- ``OCR_FALLBACK_ENABLED=false`` (or ``AI_MOCK=true`` with no real call):
  ``is_available()`` returns False; ``ocr_image``/``ocr_pdf`` return "".
- ``AI_MOCK=true`` + enabled: a deterministic mock returns canned text per
  prompt type (offline, no model download).
- Real mode: loads the model once (local) or calls the HF router per request
  (hf_inference), processes rendered PIL images.

HF token handling mirrors ``embeddings`` (NFR-26): absent -> one-time
WARNING; never logged.
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

# Default model id (used by both backends).
GLM_OCR_MODEL = "zai-org/GLM-OCR"

_token_warning_emitted = False


class _OcrBackend(Protocol):
    """Internal contract for the real (non-mock) OCR backends."""

    def ocr_image(self, image: Any, prompt: str) -> str: ...  # noqa: ANN401


class OCRService:
    """GLM-OCR wrapper with mock support + lazy backend selection."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self._enabled: bool = bool(config.ocr_fallback_enabled)
        self._mock: bool = bool(config.ai_mock)
        self._max_dim: int = config.ocr_max_image_dimension
        self._poppler_path: str = config.poppler_path
        self._hf_token: str = config.hf_token
        self.provider: str = config.ocr_provider
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
        return self._backend.ocr_image(image, prompt)

    def ocr_pdf(self, pdf_path: str, prompt: str = OCR_PROMPT_TEXT) -> str:
        """Render a PDF to images and OCR each page. Returns "" if disabled."""
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

    def render_pdf_pages(self, pdf_path: str) -> list[Any]:
        """Convert a PDF to a list of PIL images using pdf2image + Poppler."""
        from pdf2image import convert_from_path

        kwargs: dict[str, Any] = {}
        if self._poppler_path:
            kwargs["poppler_path"] = self._poppler_path
        images = convert_from_path(pdf_path, **kwargs)
        return [self.resize_if_needed(img) for img in images]

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
        """Select the real OCR backend based on ``ocr_provider``."""
        provider = self.provider
        if provider == "hf_inference":
            return _HfInferenceOcrBackend(
                model=GLM_OCR_MODEL,
                token=self._hf_token,
                endpoint=self._config.ocr_inference_endpoint,
                timeout=self._config.hf_timeout_seconds,
            )
        if provider == "local":
            return _LocalTransformersOcrBackend(token=self._hf_token)
        raise ValueError(
            f"Unknown OCR_PROVIDER={provider!r}. "
            "Expected 'local' (transformers) or 'hf_inference' (HF Inference API)."
        )

    # ------------------------------------------------------------------
    # Mock OCR (deterministic, offline)
    # ------------------------------------------------------------------

    def _mock_ocr(self, image: Any, prompt: str) -> str:  # noqa: ANN401
        """Produce deterministic canned text per prompt type."""
        # Hash the image's string repr + prompt for determinism.
        key = repr(image) + prompt
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
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
    # HF token handling
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
# Backends
# ----------------------------------------------------------------------


class _LocalTransformersOcrBackend:
    """Local backend: ``transformers`` AutoProcessor + AutoModelForImageTextToText.

    Model weights are downloaded from Hugging Face once, then all inference runs
    on the local machine — no network traffic per OCR call.
    """

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
    """Hosted backend: Hugging Face Inference API (chat-completion with image).

    No local weights are required. Each call is an HTTPS request to the HF
    Inference router (or a dedicated endpoint if ``endpoint`` is set).
    Requires ``HF_TOKEN`` with READ scope; unauthenticated calls are rate-limited.

    GLM-OCR is an ``image-text-to-text`` model — the HF router serves it via the
    chat-completion endpoint. The PIL image is encoded as a base64 data URL and
    passed as the ``image_url`` content part of a chat message.
    """

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
        """Call ``fn`` once, retry once on a transient network error (P0-1.9)."""
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
