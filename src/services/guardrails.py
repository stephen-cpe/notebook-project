"""Guardrails -- scope validation + groundedness check.

Notebook-aware heuristic approach: scope is determined by the actual source
content (not hardcoded keyword lists), and groundedness uses word-overlap
against retrieved context.

1. ``is_in_scope(question, source_texts)`` — checks whether the question is
   related to the notebook's sources by extracting meaningful terms from both
   and measuring overlap. Returns False -> the chat returns a polite refusal
   without calling the LLM (FR-43).

2. ``check_groundedness(answer, context)`` — extracts substantive words (>= 3
   chars, excluding ~100 stopwords) from the answer and checks what fraction
   appear in the context. Threshold: >= 50% = grounded. If ungrounded, a
   verification disclaimer is appended (FR-44). The answer is still returned.
"""

from __future__ import annotations

import re

# Common English stopwords excluded from word-overlap scoring.
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "over",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "not",
    "no",
    "nor",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "if",
    "because",
    "while",
    "about",
    "against",
    "up",
    "down",
    "out",
    "off",
    "away",
}

# Off-topic indicators — questions containing these phrases are almost
# certainly unrelated to any notebook source.
OFF_TOPIC_INDICATORS: set[str] = {
    "weather",
    "sports",
    "cook",
    "recipe",
    "joke",
    "movie",
    "song",
    "video game",
    "tv show",
    "celebrity",
    "gossip",
    "horoscope",
    "lottery",
    "stock market",
    "crypto price",
}

# Common greetings / social phrases that are never source-relevant.
GREETING_PATTERNS: list[str] = [
    r"^\s*hello\b",
    r"^\s*hi\b",
    r"^\s*hey\b",
    r"^\s*how are you\b",
    r"^\s*good morning\b",
    r"^\s*good evening\b",
    r"^\s*what(?:'s| is) up\b",
]


def _extract_meaningful_words(text: str) -> set[str]:
    """Extract lowercase words >= 3 chars, excluding stopwords."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def is_in_scope(question: str, source_texts: list[str]) -> bool:
    """Determine if a question is related to the notebook's sources.

    Uses a keyword-overlap heuristic: extract meaningful words from both the
    question and the combined source text, then check if any question words
    appear in the sources. Greetings and off-topic indicators short-circuit
    to False.

    Returns:
        True if the question appears related to the source content.
    """
    if not question or not question.strip():
        return False
    if not source_texts:
        return False

    q_lower = question.lower().strip()

    # Greetings are never in scope.
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, q_lower):
            return False

    # Hard off-topic indicators.
    for indicator in OFF_TOPIC_INDICATORS:
        if indicator in q_lower:
            return False

    # Extract meaningful words from the question.
    q_meaningful = _extract_meaningful_words(question)
    if not q_meaningful:
        return False

    # Extract meaningful words from all source texts combined.
    combined_sources = " ".join(source_texts)
    source_words = _extract_meaningful_words(combined_sources)
    if not source_words:
        return False

    # Overlap: any question word appearing in sources -> in scope.
    overlap = q_meaningful & source_words
    return len(overlap) > 0


def check_groundedness(answer: str, context: str, threshold: float = 0.5) -> tuple[bool, str]:
    """Check if an answer is grounded in the retrieved context.

    Extracts substantive words (>= 3 chars, excluding stopwords) from the
    answer and checks what fraction appear in the context. If the fraction is
    below ``threshold``, a verification disclaimer is appended.

    Returns:
        (is_grounded, result_text) where result_text may include a disclaimer.
    """
    if not answer:
        return True, answer
    if not context:
        disclaimer = (
            "\n\n*Note: This answer could not be verified against the "
            "provided sources. Please verify independently.*"
        )
        return False, answer + disclaimer

    answer_words = _extract_meaningful_words(answer)
    if not answer_words:
        return True, answer

    context_words = _extract_meaningful_words(context)
    if not context_words:
        disclaimer = (
            "\n\n*Note: This answer could not be verified against the "
            "provided sources. Please verify independently.*"
        )
        return False, answer + disclaimer

    overlap = answer_words & context_words
    fraction = len(overlap) / len(answer_words)

    if fraction >= threshold:
        return True, answer

    disclaimer = (
        "\n\n*Note: This answer could not be fully verified against the "
        "provided sources. Please verify independently.*"
    )
    return False, answer + disclaimer


def maybe_append_disclaimer(answer: str, context: str) -> str:
    """Convenience: run ``check_groundedness`` and return only the result text."""
    _, result = check_groundedness(answer, context)
    return result
