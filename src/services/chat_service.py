"""Chat service — orchestrates the full chat flow.

Flow (FR-40 through FR-46):
1. Scope guardrail: if the question is off-topic, return a refusal (no LLM call).
2. RAG retrieval: query all the notebook's source collections, get context + sources.
3. Build prompt (system + context + question, with <|think|> if enabled).
4. Call Ollama Cloud (sync or stream).
5. Groundedness check: append disclaimer if answer isn't grounded.
6. Persist user + assistant ChatMessages.
7. Return {answer, sources, latency_ms}.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Any

from src.extensions import db
from src.models import Notebook, Source
from src.repositories import chat_repo, content_registry_repo
from src.services.guardrails import check_groundedness, is_in_scope
from src.services.ollama_client import build_prompt, get_ollama_client
from src.services.rag_retriever import (
    build_context_string,
    format_sources,
    get_rag_retriever,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided "
    "sources. Always cite sources when possible. If the answer is not in the "
    "sources, say you don't have enough information. Keep answers concise."
)

OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions related to the sources in this notebook. "
    "Please ask a question about the uploaded documents."
)


class ChatService:
    """Orchestrates the chat flow: guardrails → retrieve → generate → persist."""

    def __init__(self) -> None:
        from src.config import Config

        self._config = Config()
        self._retriever = get_rag_retriever()
        self._client = get_ollama_client()

    def chat_sync(self, notebook: Notebook, question: str) -> dict[str, Any]:
        """Non-streaming chat. Returns the full result dict."""
        start = time.time()
        logger.info("=== Chat sync start: notebook=%d question=%.100s...", notebook.id, question)

        # 1. Get source texts for scope check.
        t0 = time.time()
        source_texts = self._get_source_texts(notebook.id)
        logger.info(
            "  [1/5] Loaded %d source texts for scope check (%.0fms)",
            len(source_texts),
            (time.time() - t0) * 1000,
        )

        # 2. Scope guardrail.
        t0 = time.time()
        in_scope = is_in_scope(question, source_texts)
        logger.info(
            "  [2/5] Scope check: %s (%.0fms)",
            "in-scope" if in_scope else "OUT-OF-SCOPE",
            (time.time() - t0) * 1000,
        )
        if not in_scope:
            answer = OUT_OF_SCOPE_RESPONSE
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, [], latency)
            logger.info("=== Chat sync done (out-of-scope): %dms", latency)
            return {"answer": answer, "sources": [], "latency_ms": latency}

        # 3. RAG retrieval.
        t0 = time.time()
        content_hashes = self._get_source_hashes(notebook.id)
        logger.info("  [3/5] Retrieving from %d source collections...", len(content_hashes))
        results = self._retriever.retrieve_with_sources(content_hashes, question, top_k=5)
        context = build_context_string(results)
        sources = format_sources(results)
        logger.info(
            "  [3/5] Retrieved %d chunks, %d unique sources, %d chars context (%.0fms)",
            len(results),
            len(sources),
            len(context),
            (time.time() - t0) * 1000,
        )

        if not context:
            answer = "I don't have enough information from the sources to answer that question."
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, sources, latency)
            logger.info("=== Chat sync done (no context): %dms", latency)
            return {"answer": answer, "sources": sources, "latency_ms": latency}

        # 4. Build prompt + call LLM.
        t0 = time.time()
        messages = build_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
            enable_thinking=self._config.enable_thinking,
        )
        logger.info(
            "  [4/5] Calling Ollama Cloud (model=%s, thinking=%s)...",
            self._config.chat_model,
            self._config.enable_thinking,
        )
        raw_answer = self._client.chat(messages)
        logger.info(
            "  [4/5] LLM response: %d chars (%.0fms)", len(raw_answer), (time.time() - t0) * 1000
        )

        # 5. Groundedness check.
        t0 = time.time()
        is_grounded, answer = check_groundedness(raw_answer, context)
        logger.info(
            "  [5/5] Groundedness: %s (%.0fms)",
            "grounded" if is_grounded else "UNGROUNDED",
            (time.time() - t0) * 1000,
        )

        # 6. Persist.
        latency = int((time.time() - start) * 1000)
        self._persist(notebook.id, question, answer, sources, latency)

        logger.info(
            "=== Chat sync done: notebook=%d latency=%dms grounded=%s sources=%d answer=%dchars",
            notebook.id,
            latency,
            is_grounded,
            len(sources),
            len(answer),
        )
        return {"answer": answer, "sources": sources, "latency_ms": latency}

    def chat_stream(self, notebook: Notebook, question: str) -> Generator[str]:
        """Streaming chat via SSE. Yields SSE-format frames.

        Token frames: ``data: {"token": "..."}\\n\\n``
        Final frame: ``data: {"sources": [...], "latency_ms": N, "done": true}\\n\\n``
        """
        start = time.time()
        logger.info("=== Chat stream start: notebook=%d question=%.100s...", notebook.id, question)

        # 1. Scope guardrail.
        t0 = time.time()
        source_texts = self._get_source_texts(notebook.id)
        in_scope = is_in_scope(question, source_texts)
        logger.info(
            "  [1/4] Scope check: %s (%.0fms)",
            "in-scope" if in_scope else "OUT-OF-SCOPE",
            (time.time() - t0) * 1000,
        )
        if not in_scope:
            answer = OUT_OF_SCOPE_RESPONSE
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, [], latency)
            yield self._sse_frame({"token": answer})
            yield self._sse_frame({"sources": [], "latency_ms": latency, "done": True})
            logger.info("=== Chat stream done (out-of-scope): %dms", latency)
            return

        # 2. RAG retrieval.
        t0 = time.time()
        content_hashes = self._get_source_hashes(notebook.id)
        logger.info("  [2/4] Retrieving from %d source collections...", len(content_hashes))
        results = self._retriever.retrieve_with_sources(content_hashes, question, top_k=5)
        context = build_context_string(results)
        sources = format_sources(results)
        logger.info(
            "  [2/4] Retrieved %d chunks, %d unique sources, %d chars context (%.0fms)",
            len(results),
            len(sources),
            len(context),
            (time.time() - t0) * 1000,
        )

        if not context:
            answer = "I don't have enough information from the sources to answer that question."
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, sources, latency)
            yield self._sse_frame({"token": answer})
            yield self._sse_frame({"sources": sources, "latency_ms": latency, "done": True})
            logger.info("=== Chat stream done (no context): %dms", latency)
            return

        # 3. Build prompt + stream tokens.
        t0 = time.time()
        messages = build_prompt(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
            enable_thinking=self._config.enable_thinking,
        )
        logger.info(
            "  [3/4] Streaming from Ollama Cloud (model=%s, thinking=%s)...",
            self._config.chat_model,
            self._config.enable_thinking,
        )

        full_answer_parts: list[str] = []
        for token in self._client.stream(messages):
            full_answer_parts.append(token)
            yield self._sse_frame({"token": token})

        raw_answer = "".join(full_answer_parts)
        logger.info(
            "  [3/4] Stream complete: %d tokens, %d chars (%.0fms)",
            len(full_answer_parts),
            len(raw_answer),
            (time.time() - t0) * 1000,
        )

        # 4. Groundedness check.
        t0 = time.time()
        _, answer = check_groundedness(raw_answer, context)
        logger.info(
            "  [4/4] Groundedness: %s (%.0fms)",
            "grounded" if answer == raw_answer else "UNGROUNDED",
            (time.time() - t0) * 1000,
        )
        # If disclaimer was appended, send it as a final token.
        if answer != raw_answer:
            disclaimer = answer[len(raw_answer) :]
            yield self._sse_frame({"token": disclaimer})

        # 5. Persist.
        latency = int((time.time() - start) * 1000)
        self._persist(notebook.id, question, answer, sources, latency)

        # 6. Final frame.
        yield self._sse_frame({"sources": sources, "latency_ms": latency, "done": True})
        logger.info(
            "=== Chat stream done: notebook=%d latency=%dms sources=%d answer=%dchars",
            notebook.id,
            latency,
            len(sources),
            len(answer),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_source_hashes(self, notebook_id: int) -> list[str]:
        """Return content hashes for all ready sources in a notebook."""
        sources = (
            db.session.query(Source)
            .filter(
                Source.notebook_id == notebook_id,
                Source.status.in_(["ready", "partial"]),
            )
            .all()
        )
        return [s.content_hash for s in sources]

    def _get_source_texts(self, notebook_id: int) -> list[str]:
        """Return cached extracted texts for scope checking."""
        hashes = self._get_source_hashes(notebook_id)
        texts: list[str] = []
        for h in hashes:
            entry = content_registry_repo.get_by_hash(h)
            if entry and entry.extracted_text:
                texts.append(entry.extracted_text)
        return texts

    def _persist(
        self,
        notebook_id: int,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        latency_ms: int,
    ) -> None:
        """Persist the user question + assistant answer as ChatMessages."""
        chat_repo.create_message(notebook_id, "user", question)
        chat_repo.create_message(
            notebook_id,
            "assistant",
            answer,
            sources_json=json.dumps(sources) if sources else None,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _sse_frame(data: dict[str, Any]) -> str:
        """Format a dict as an SSE ``data:`` frame."""
        return f"data: {json.dumps(data)}\n\n"
