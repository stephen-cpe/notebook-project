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
from src.services.rag_budget import (
    estimate_coverage_ratio,
    get_context_budget_chars,
    get_top_k_for_budget,
    per_collection_k_for,
    truncate_results_to_budget,
)
from src.services.rag_retriever import (
    build_context_string,
    build_coverage_digest,
    format_sources,
    get_rag_retriever,
)
from src.services.section_digest import notebook_digest_text

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
            return {
                "answer": answer,
                "sources": [],
                "source_total": 0,
                "latency_ms": latency,
                "coverage_ratio": 0.0,
            }

        # 3. RAG retrieval (budget-derived top_k + coverage digest).
        t0 = time.time()
        aliases = self._get_source_aliases(notebook.id)
        content_hashes = list(aliases.keys())
        logger.info("  [3/5] Retrieving from %d source collections...", len(content_hashes))
        results, top_k, per_collection_k = self._retrieve_budgeted(
            content_hashes, question, aliases
        )
        context = build_context_string(results)
        sources, source_total = self._citation_sets(results)
        sources = self._attach_figures(notebook.id, sources, aliases)
        coverage_ratio = self._coverage_ratio(notebook.id, context)
        digest = self._coverage_layer(notebook.id, source_texts)
        if digest:
            context = f"{digest}\n\n{context}" if context else digest
        logger.info(
            "  [3/5] Retrieved %d chunks (top_k=%d, per_source=%d), %d/%d cited sources,"
            " %d chars context (%.0fms)",
            len(results),
            top_k,
            per_collection_k,
            len(sources),
            source_total,
            len(context),
            (time.time() - t0) * 1000,
        )

        if not context:
            answer = "I don't have enough information from the sources to answer that question."
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, sources, latency)
            logger.info("=== Chat sync done (no context): %dms", latency)
            return {
                "answer": answer,
                "sources": sources,
                "source_total": source_total,
                "latency_ms": latency,
                "coverage_ratio": coverage_ratio,
            }

        # 4. Build prompt + call LLM.
        t0 = time.time()
        difficulty = self._user_difficulty(notebook)
        from src.services.difficulty import difficulty_instruction

        system = f"{SYSTEM_PROMPT} {difficulty_instruction(difficulty)}"
        messages = build_prompt(
            system=system,
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
            "=== Chat sync done: notebook=%d latency=%dms grounded=%s cited=%d/%d answer=%dchars",
            notebook.id,
            latency,
            is_grounded,
            len(sources),
            source_total,
            len(answer),
        )
        return {
            "answer": answer,
            "sources": sources,
            "source_total": source_total,
            "latency_ms": latency,
            "coverage_ratio": coverage_ratio,
        }

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
            yield self._sse_frame(
                {
                    "sources": [],
                    "source_total": 0,
                    "latency_ms": latency,
                    "coverage_ratio": 0.0,
                    "done": True,
                }
            )
            logger.info("=== Chat stream done (out-of-scope): %dms", latency)
            return

        # 2. RAG retrieval (budget-derived top_k + coverage digest).
        t0 = time.time()
        aliases = self._get_source_aliases(notebook.id)
        content_hashes = list(aliases.keys())
        logger.info("  [2/4] Retrieving from %d source collections...", len(content_hashes))
        results, top_k, per_collection_k = self._retrieve_budgeted(
            content_hashes, question, aliases
        )
        context = build_context_string(results)
        sources, source_total = self._citation_sets(results)
        sources = self._attach_figures(notebook.id, sources, aliases)
        coverage_ratio = self._coverage_ratio(notebook.id, context)
        digest = self._coverage_layer(notebook.id, source_texts)
        if digest:
            context = f"{digest}\n\n{context}" if context else digest
        logger.info(
            "  [2/4] Retrieved %d chunks (top_k=%d, per_source=%d), %d/%d cited sources,"
            " %d chars context (%.0fms)",
            len(results),
            top_k,
            per_collection_k,
            len(sources),
            source_total,
            len(context),
            (time.time() - t0) * 1000,
        )

        if not context:
            answer = "I don't have enough information from the sources to answer that question."
            latency = int((time.time() - start) * 1000)
            self._persist(notebook.id, question, answer, sources, latency)
            yield self._sse_frame({"token": answer})
            yield self._sse_frame(
                {
                    "sources": sources,
                    "source_total": source_total,
                    "latency_ms": latency,
                    "coverage_ratio": coverage_ratio,
                    "done": True,
                }
            )
            logger.info("=== Chat stream done (no context): %dms", latency)
            return

        # 3. Build prompt + stream tokens.
        t0 = time.time()
        from src.services.difficulty import difficulty_instruction as _diff_instr

        stream_system = f"{SYSTEM_PROMPT} {_diff_instr(self._user_difficulty(notebook))}"
        messages = build_prompt(
            system=stream_system,
            context=context,
            question=question,
            enable_thinking=self._config.enable_thinking,
        )
        logger.info(
            "  [3/4] Streaming from Ollama Cloud (model=%s, thinking=%s)...",
            self._config.chat_model,
            self._config.enable_thinking,
        )

        # Persist the user turn up-front: if the client disconnects
        # mid-stream, the question itself is never lost from history.
        chat_repo.create_message(notebook.id, "user", question)

        full_answer_parts: list[str] = []
        assistant_persisted = False
        try:
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
            chat_repo.create_message(
                notebook.id,
                "assistant",
                answer,
                sources_json=json.dumps(sources) if sources else None,
                latency_ms=latency,
            )
            assistant_persisted = True

            # 6. Final frame.
            yield self._sse_frame(
                {
                    "sources": sources,
                    "source_total": source_total,
                    "latency_ms": latency,
                    "coverage_ratio": coverage_ratio,
                    "done": True,
                }
            )
            logger.info(
                "=== Chat stream done: notebook=%d latency=%dms cited=%d/%d answer=%dchars",
                notebook.id,
                latency,
                len(sources),
                source_total,
                len(answer),
            )
        finally:
            if not assistant_persisted:
                # The client disconnected (or streaming failed) after the
                # user turn was saved: keep whatever partial answer had
                # streamed so the turn is not silently lost from history.
                try:
                    latency = int((time.time() - start) * 1000)
                    chat_repo.create_message(
                        notebook.id,
                        "assistant",
                        "".join(full_answer_parts),
                        sources_json=json.dumps(sources) if sources else None,
                        latency_ms=latency,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Could not persist partial chat turn")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_source_hashes(self, notebook_id: int) -> list[str]:
        """Return content hashes for all ready sources in a notebook."""
        return list(self._get_source_aliases(notebook_id).keys())

    def _get_source_aliases(self, notebook_id: int) -> dict[str, str]:
        """Map content hash -> this notebook's display filename.

        Citations resolve through this map so shared (deduped) vectors never
        surface another user's original filename, and renames take effect
        immediately without re-embedding.
        """
        sources = (
            db.session.query(Source)
            .filter(
                Source.notebook_id == notebook_id,
                Source.status.in_(["ready", "partial"]),
            )
            .all()
        )
        return {s.content_hash: s.filename for s in sources}

    def _get_source_texts(self, notebook_id: int) -> list[str]:
        """Return cached extracted texts for scope checking (one query)."""
        return content_registry_repo.get_texts_by_hashes(self._get_source_hashes(notebook_id))

    def _retrieve_budgeted(
        self,
        content_hashes: list[str],
        question: str,
        aliases: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Retrieve chunks bounded by the real context-window budget.

        ``top_k`` comes from the window-derived budget capped by the visible
        ``RAG_MAX_TOP_K`` ceiling (no hidden constants); per-source depth
        scales so one giant document can fill its share. Retrieved chunks are
        truncated to the char budget minus reserved digest room, whole chunks
        only. Returns ``(results, top_k, per_collection_k)``.
        """
        top_k = min(get_top_k_for_budget(config=self._config), self._config.rag_max_top_k)
        per_collection_k = per_collection_k_for(top_k, len(content_hashes))
        results = self._retriever.retrieve_with_sources(
            content_hashes,
            question,
            top_k=top_k,
            filenames=aliases,
            per_collection_k=per_collection_k,
        )
        char_budget = get_context_budget_chars(config=self._config)
        digest_room = min(self._config.rag_digest_max_chars, char_budget // 4)
        kept, _kept_chars = truncate_results_to_budget(results, char_budget - digest_room)
        return kept, top_k, per_collection_k

    def _citation_sets(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Split retrieved results into (shown citations, total unique count).

        The UI renders the capped top-N list behind a "Sources (N of M)"
        toggle; the full count keeps the "+N more" label honest. Retrieval
        breadth (what the model read) is unaffected — only display is capped.
        """
        all_sources = format_sources(results)
        total = len(all_sources)
        limit = max(1, int(self._config.rag_max_citations))
        return all_sources[:limit], total

    def _coverage_layer(self, notebook_id: int, source_texts: list[str]) -> str:
        """Full-coverage digest layer for the prompt.

        Prefers cached section digests (every section of every source was
        read during the background build); falls back to the cheap
        extractive digest when the cache is cold, e.g. the user asks before
        the summary job finishes. Honors ``RAG_SUMMARY_MAP``.
        """
        if not self._config.rag_summary_map:
            return ""
        stitched = notebook_digest_text(
            notebook_id, self._config.rag_digest_max_chars, self._config
        )
        if stitched:
            return stitched
        return build_coverage_digest(source_texts, max_chars=6000)

    def _coverage_ratio(self, notebook_id: int, context: str) -> float:
        """Return retrieved-context chars / total source chars (0..1)."""
        try:
            total = sum(len(t) for t in self._get_source_texts(notebook_id))
            return estimate_coverage_ratio(len(context), total)
        except Exception:  # noqa: BLE001
            return 1.0

    def _user_difficulty(self, notebook: Notebook) -> str:
        """Return the notebook owner's difficulty preference (safe default)."""
        try:
            from src.models import User

            user = db.session.get(User, notebook.user_id)
            level = getattr(user, "difficulty", "Normal") if user else "Normal"
            from src.services.difficulty import normalize_difficulty

            return normalize_difficulty(str(level or "Normal"))
        except Exception:  # noqa: BLE001
            return "Normal"

    def _attach_figures(
        self,
        notebook_id: int,
        sources: list[dict[str, Any]],
        aliases: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Attach figure thumbnails to citation dicts (best-effort)."""
        try:
            from src.services.figure_store import attach_figures

            filename_to_hash = {fname: h for h, fname in aliases.items()}
            return attach_figures(
                sources,
                content_hashes=filename_to_hash,
                notebook_id=notebook_id,
                data_dir=self._config.data_dir,
            )
        except Exception:  # noqa: BLE001
            return sources

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
