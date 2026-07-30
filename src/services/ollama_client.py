"""Ollama Cloud chat client — single model ``gemma4:31b-cloud``.

Wraps Ollama Cloud's API for the in-app chat model. The model is hidden from
the end user (no selector in the UI). Thinking is enabled server-side via the
``<|think|>`` system-prompt token (Gemma 4 built-in) when ``ENABLE_THINKING=true``.

Behavior:
- ``AI_MOCK=true``: deterministic mock returns canned text / word chunks.
  No network calls (used in tests/CI).
- Real mode: POSTs to ``{OLLAMA_CLOUD_BASE_URL}/api/chat`` (Ollama-native) or
  ``/v1/chat/completions`` (OpenAI-compatible). Retries once on failure.

Gemma 4 output format when thinking is enabled:
  ``<|channel|thought\\n [internal reasoning] <channel|> [final answer]``
The ``extract_final_answer`` helper strips the thinking block.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import requests

from src.services.exceptions import AIModelUnavailableError, AITimeoutError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config

# Regex to strip Gemma 4 thinking blocks from model output.
# Format: <|channel|thought\n [reasoning] <channel|> [final answer]
_THINK_BLOCK_RE = re.compile(r"<\|channel\|thought\s*\n?.*?<\|?channel\|>", re.DOTALL)


def extract_final_answer(raw: str) -> str:
    """Strip Gemma 4 thinking blocks, returning only the final answer text."""
    if not raw:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", raw).strip()
    return cleaned if cleaned else raw.strip()


def build_prompt(
    system: str,
    context: str,
    question: str,
    enable_thinking: bool = True,
) -> list[dict[str, str]]:
    """Assemble the message list for the LLM.

    When ``enable_thinking`` is True, the ``<|think|>`` token is prepended to
    the system prompt (Gemma 4's mechanism for enabling internal reasoning).
    """
    system_content = system
    if enable_thinking:
        system_content = "<|think|>" + system_content

    user_content = question
    if context:
        user_content = (
            f"Use the following context to answer the question. "
            f"If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


class OllamaClient:
    """Ollama Cloud chat client with mock support + retry."""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            from src.config import Config

            config = Config()

        self._config = config
        self.model: str = config.chat_model
        self._mock: bool = bool(config.ai_mock)
        self.enable_thinking: bool = bool(config.enable_thinking)
        self._base_url: str = config.ollama_cloud_base_url.rstrip("/")
        self._api_key: str = config.ollama_cloud_api_key
        self._timeout: int = config.ollama_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages and return the full response text.

        Retries once on failure before raising.
        """
        if self._mock:
            return self._mock_chat(messages)
        return self._chat_with_retry(messages)

    def stream(self, messages: list[dict[str, str]]) -> Generator[str]:
        """Stream response tokens as a generator of string chunks.

        In mock mode, yields word-level chunks of the mock response.
        Retries once on connection failure before raising.
        """
        if self._mock:
            yield from self._mock_stream(messages)
            return
        yield from self._stream_with_retry(messages)

    # ------------------------------------------------------------------
    # Retry wrappers
    # ------------------------------------------------------------------

    def _chat_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Call ``_real_chat`` with one retry on failure."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return self._real_chat(messages)
            except (ConnectionError, TimeoutError, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("Chat attempt %d failed: %s", attempt + 1, exc)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Chat attempt %d failed: %s", attempt + 1, exc)
        raise AIModelUnavailableError(f"Chat failed after retry: {last_exc}")

    def _stream_with_retry(self, messages: list[dict[str, str]]) -> Generator[str]:
        """Call ``_real_stream`` with one retry on connection failure."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                yield from self._real_stream(messages)
                return
            except (ConnectionError, TimeoutError, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("Stream attempt %d failed: %s", attempt + 1, exc)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Stream attempt %d failed: %s", attempt + 1, exc)
        raise AIModelUnavailableError(f"Stream failed after retry: {last_exc}")

    # ------------------------------------------------------------------
    # Real HTTP calls
    # ------------------------------------------------------------------

    def _real_chat(self, messages: list[dict[str, str]]) -> str:
        """POST to Ollama Cloud and return the response text."""
        url = f"{self._base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        t0 = time.time()
        logger.info(
            "Ollama chat request → %s  model=%s  timeout=%ds  thinking=%s",
            url,
            self.model,
            self._timeout,
            self.enable_thinking,
        )
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            elapsed = time.time() - t0
            logger.info(
                "Ollama chat response ← HTTP %d  elapsed=%.1fs  content-length=%s",
                resp.status_code,
                elapsed,
                resp.headers.get("content-length", "?"),
            )
            resp.raise_for_status()
        except requests.Timeout as exc:
            elapsed = time.time() - t0
            logger.error("Ollama chat timed out after %.1fs (limit=%ds)", elapsed, self._timeout)
            raise AITimeoutError(f"Ollama Cloud timed out after {self._timeout}s") from exc
        except requests.ConnectionError as exc:
            elapsed = time.time() - t0
            logger.error("Ollama chat connection failed after %.1fs: %s", elapsed, exc)
            raise ConnectionError(f"Ollama Cloud unreachable: {exc}") from exc
        except requests.HTTPError as exc:
            elapsed = time.time() - t0
            body = resp.text[:500] if resp.text else ""
            logger.error(
                "Ollama chat HTTP %d after %.1fs: %s  body=%s",
                resp.status_code,
                elapsed,
                exc,
                body,
            )
            raise ConnectionError(f"Ollama Cloud HTTP {resp.status_code}: {exc}") from exc

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        logger.info(
            "Ollama chat answer: %d chars  first=%.80s...",
            len(content),
            content[:80],
        )
        return extract_final_answer(content)

    def _real_stream(self, messages: list[dict[str, str]]) -> Generator[str]:
        """POST to Ollama Cloud with stream=True, yielding token chunks."""
        url = f"{self._base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        t0 = time.time()
        logger.info(
            "Ollama stream request → %s  model=%s  timeout=%ds  thinking=%s",
            url,
            self.model,
            self._timeout,
            self.enable_thinking,
        )
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(10, self._timeout),
                stream=True,
            )
            logger.info(
                "Ollama stream connected ← HTTP %d  elapsed=%.1fs",
                resp.status_code,
                time.time() - t0,
            )
            resp.raise_for_status()
        except requests.Timeout as exc:
            elapsed = time.time() - t0
            logger.error("Ollama stream timed out after %.1fs (limit=%ds)", elapsed, self._timeout)
            raise AITimeoutError(f"Ollama Cloud timed out after {self._timeout}s") from exc
        except requests.ConnectionError as exc:
            elapsed = time.time() - t0
            logger.error("Ollama stream connection failed after %.1fs: %s", elapsed, exc)
            raise ConnectionError(f"Ollama Cloud unreachable: {exc}") from exc

        token_count = 0
        first_token_at: float | None = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                import json

                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if first_token_at is None:
                        first_token_at = time.time()
                        logger.info(
                            "Ollama stream first token at %.1fs (TTFT)",
                            first_token_at - t0,
                        )
                    token_count += 1
                    yield token
            except (ValueError, KeyError):
                continue

        elapsed = time.time() - t0
        logger.info(
            "Ollama stream done: %d tokens  total=%.1fs  TTFT=%.1fs",
            token_count,
            elapsed,
            (first_token_at - t0) if first_token_at else 0,
        )

    # ------------------------------------------------------------------
    # Mock implementations (deterministic, offline)
    # ------------------------------------------------------------------

    def _mock_chat(self, messages: list[dict[str, str]]) -> str:
        """Deterministic mock chat response based on message content."""
        key = self._messages_key(messages)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        # Produce a plausible answer referencing the question.
        user_content = self._extract_user_content(messages)
        return (
            f"[mock answer {digest}] Based on the provided sources, "
            f"here is a response to: {user_content[:60]}."
        )

    def _mock_stream(self, messages: list[dict[str, str]]) -> Generator[str]:
        """Yield word-level chunks of the mock response."""
        full = self._mock_chat(messages)
        words = full.split()
        for word in words:
            yield word + " "

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_key(messages: list[dict[str, str]]) -> str:
        """Produce a stable hash key from the message list."""
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    @staticmethod
    def _extract_user_content(messages: list[dict[str, str]]) -> str:
        """Return the last user message content (or empty)."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Return a process-wide ``OllamaClient`` (created lazily)."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def reset_ollama_client() -> None:
    """Reset the cached client (used by tests that change config)."""
    global _client
    _client = None
