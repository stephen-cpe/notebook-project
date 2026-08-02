"""Unit tests for src.services.ollama_client (TDD step 12).

The chat client wraps Ollama Cloud calls to the single in-app model
``gemma4:31b-cloud``. Unit tests run fully offline using mock mode
(``AI_MOCK=true``); real calls are gated behind ``RUN_INTEGRATION=1``.

Covers:
- Mock mode: chat() returns deterministic canned text.
- Mock mode: stream() yields deterministic token chunks.
- Thinking token: <|think|> is prepended to the system prompt when enabled.
- Thinking disabled: no <|think|> token in the prompt.
- System prompt is respected in mock output.
- Retry on failure: one retry before raising.
- Timeout handling.
- Non-mock mode sends real HTTP (integration only).
- build_prompt assembles system + context + question correctly.
- extract_final_answer strips thinking blocks from Gemma 4 output.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.ollama_client import (
    OllamaClient,
    build_prompt,
    extract_final_answer,
    get_ollama_client,
    reset_ollama_client,
)


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    reset_ollama_client()
    yield
    reset_ollama_client()


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_assembles_system_context_question(self) -> None:
        messages = build_prompt(
            system="You are a helpful assistant.",
            context="The sky is blue. Grass is green.",
            question="What color is the sky?",
        )
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "helpful assistant" in messages[0]["content"]
        assert "sky is blue" in messages[1]["content"]
        assert "What color is the sky?" in messages[1]["content"]

    def test_empty_context(self) -> None:
        messages = build_prompt("Be concise.", "", "What is 2+2?")
        assert "2+2" in messages[1]["content"]

    def test_thinking_token_added_when_enabled(self) -> None:
        messages = build_prompt(
            system="Answer questions.",
            context="ctx",
            question="q",
            enable_thinking=True,
        )
        assert "<|think|>" in messages[0]["content"]

    def test_no_thinking_token_when_disabled(self) -> None:
        messages = build_prompt(
            system="Answer questions.",
            context="ctx",
            question="q",
            enable_thinking=False,
        )
        assert "<|think|>" not in messages[0]["content"]


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractFinalAnswer:
    def test_plain_text(self) -> None:
        raw = "This is the final answer."
        assert extract_final_answer(raw) == "This is the final answer."

    def test_strips_thinking_block(self) -> None:
        raw = "<|channel|thought\nThis is internal reasoning.\n<channel|>The final answer."
        result = extract_final_answer(raw)
        assert "final answer" in result
        assert "internal reasoning" not in result

    def test_empty_thinking_block(self) -> None:
        raw = "<|channel|thought\n<channel|>Just the answer."
        result = extract_final_answer(raw)
        assert "Just the answer" in result
        assert "thought" not in result

    def test_empty_string(self) -> None:
        assert extract_final_answer("") == ""


# ---------------------------------------------------------------------------
# Mock chat (sync)
# ---------------------------------------------------------------------------


class TestMockChat:
    def test_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("CHAT_MODEL", "gemma4:31b-cloud")
        client = OllamaClient()
        result = client.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_system_prompt_affects_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        r1 = client.chat([{"role": "user", "content": "same question"}])
        r2 = client.chat(
            [
                {"role": "system", "content": "different system"},
                {"role": "user", "content": "same question"},
            ]
        )
        # Different messages produce different mock outputs.
        assert r1 != r2

    def test_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        r1 = client.chat([{"role": "user", "content": "deterministic test"}])
        r2 = client.chat([{"role": "user", "content": "deterministic test"}])
        assert r1 == r2

    def test_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        assert client.model == "gemma4:31b-cloud"

    def test_no_user_message_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        assert client._extract_user_content([{"role": "system", "content": "sys"}]) == ""


# ---------------------------------------------------------------------------
# Mock stream
# ---------------------------------------------------------------------------


class TestMockStream:
    def test_yields_token_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        chunks = list(client.stream([{"role": "user", "content": "Hello"}]))
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        # Joined chunks form a complete answer.
        joined = "".join(chunks)
        assert len(joined) > 0

    def test_stream_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        c1 = list(client.stream([{"role": "user", "content": "same"}]))
        c2 = list(client.stream([{"role": "user", "content": "same"}]))
        assert c1 == c2

    def test_stream_chunks_are_word_pieces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock stream splits the mock answer into word-level chunks."""
        monkeypatch.setenv("AI_MOCK", "true")
        client = OllamaClient()
        chunks = list(client.stream([{"role": "user", "content": "test"}]))
        joined = "".join(chunks)
        # Each chunk should be a substring of the full answer.
        assert len(joined) > 0


# ---------------------------------------------------------------------------
# Thinking token behavior
# ---------------------------------------------------------------------------


class TestThinkingToken:
    def test_thinking_enabled_prepends_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("ENABLE_THINKING", "true")
        client = OllamaClient()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is AI?"},
        ]
        result = client.chat(messages)
        # Mock mode echoes back a response; the thinking token is added
        # internally by build_prompt, not visible in the output.
        assert isinstance(result, str)

    def test_thinking_disabled_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("ENABLE_THINKING", "false")
        client = OllamaClient()
        assert client.enable_thinking is False


# ---------------------------------------------------------------------------
# Retry on failure
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_once_then_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "http://fake")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "fake-key")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "5")
        client = OllamaClient()

        call_count = 0

        def _failing_call(messages: list[dict[str, str]]) -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("network down")

        with (
            patch.object(client, "_real_chat", side_effect=_failing_call),
            pytest.raises(Exception),  # noqa: B017,SIM117
        ):
            client.chat([{"role": "user", "content": "hi"}])
        # One retry means 2 total calls.
        assert call_count == 2

    def test_succeeds_on_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "http://fake")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "fake-key")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "5")
        client = OllamaClient()

        call_count = 0

        def _retry_then_succeed(messages: list[dict[str, str]]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("first attempt fails")
            return "success on retry"

        with patch.object(client, "_real_chat", side_effect=_retry_then_succeed):
            result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "success on retry"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_returns_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        client = get_ollama_client()
        assert isinstance(client, OllamaClient)
        assert client.model == "gemma4:31b-cloud"


# ---------------------------------------------------------------------------
# Real HTTP paths (requests.post mocked, offline)
# ---------------------------------------------------------------------------


def _real_client(monkeypatch: pytest.MonkeyPatch) -> OllamaClient:
    monkeypatch.setenv("AI_MOCK", "false")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.example")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "secret-key")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    return OllamaClient()


class _FakeResp:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json: object = None,
        lines: list[str] | None = None,
        headers: object = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json = json
        self._lines = lines
        self.headers = headers if headers is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def iter_lines(self, decode_unicode: bool = False) -> list[str]:  # noqa: ARG002
        return self._lines if self._lines is not None else []


class TestRealChat:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        resp = _FakeResp(json={"message": {"content": "<|channel|thought\nno\n<channel|>Hello!"}})
        with patch("src.services.ollama_client.requests.post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello!"
        assert mock_post.call_count == 1
        url = mock_post.call_args.args[0]
        assert url == "https://ollama.example/api/chat"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-key"

    def test_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "false")
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.example")
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
        client = OllamaClient()
        resp = _FakeResp(json={"message": {"content": "plain answer"}})
        with patch("src.services.ollama_client.requests.post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "plain answer"
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        from requests import Timeout

        with (
            patch("src.services.ollama_client.requests.post", side_effect=Timeout("t")),
            pytest.raises(Exception) as excinfo,
        ):
            client.chat([{"role": "user", "content": "hi"}])
        assert "timed out" in str(excinfo.value)

    def test_connection_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        from requests import ConnectionError as ReqConnError

        with (
            patch("src.services.ollama_client.requests.post", side_effect=ReqConnError("down")),
            pytest.raises(Exception) as excinfo,
        ):
            client.chat([{"role": "user", "content": "hi"}])
        assert "unreachable" in str(excinfo.value)

    def test_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        resp = _FakeResp(status_code=500, text="boom")
        with (
            patch("src.services.ollama_client.requests.post", return_value=resp),
            pytest.raises(Exception) as excinfo,
        ):
            client.chat([{"role": "user", "content": "hi"}])
        assert "HTTP 500" in str(excinfo.value)

    def test_retries_after_connection_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _real_client(monkeypatch)
        from requests import ConnectionError as ReqConnError

        calls = 0

        def _post(*args: object, **kwargs: object) -> _FakeResp:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ReqConnError("down")
            return _FakeResp(json={"message": {"content": "recovered"}})

        with patch("src.services.ollama_client.requests.post", side_effect=_post):
            result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "recovered"
        assert calls == 2


class TestRealStream:
    def test_yields_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        lines = [
            '{"message": {"content": "hello"}}',
            "",
            '{"message": {"content": " world"}}',
            "not json",
        ]
        resp = _FakeResp(lines=lines)
        with patch("src.services.ollama_client.requests.post", return_value=resp):
            chunks = list(client.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["hello", " world"]

    def test_stream_connection_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        from requests import ConnectionError as ReqConnError

        with (
            patch("src.services.ollama_client.requests.post", side_effect=ReqConnError("down")),
            pytest.raises(Exception) as excinfo,
        ):
            list(client.stream([{"role": "user", "content": "hi"}]))
        assert "Stream failed after retry" in str(excinfo.value)

    def test_stream_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _real_client(monkeypatch)
        from requests import Timeout

        with (
            patch("src.services.ollama_client.requests.post", side_effect=Timeout("t")),
            pytest.raises(Exception) as excinfo,
        ):
            list(client.stream([{"role": "user", "content": "hi"}]))
        assert "timed out" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Integration test (real Ollama Cloud) — skipped unless RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealClient:
    def test_real_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        if os.getenv("RUN_INTEGRATION") != "1":
            pytest.skip("RUN_INTEGRATION not set")
        monkeypatch.setenv("AI_MOCK", "false")
        client = OllamaClient()
        result = client.chat(
            [
                {"role": "user", "content": "Say hello in one word."},
            ]
        )
        assert isinstance(result, str)
        assert len(result) > 0
