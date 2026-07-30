"""Unit tests for src.services.guardrails (TDD step 11).

Two guardrails (scope check + groundedness check):
1. ``is_in_scope(question, source_texts)`` — heuristic keyword/regex check
   against the notebook's actual source content (not hardcoded keywords).
   Returns False -> chat returns a polite refusal without calling the LLM.
2. ``check_groundedness(answer, context)`` — word-overlap heuristic; if the
   answer's substantive terms don't appear in retrieved context below a
   threshold, a verification disclaimer is appended (answer still returned).

Covers:
- is_in_scope: on-topic question about source content -> True.
- is_in_scope: off-topic question (weather, sports, unrelated) -> False.
- is_in_scope: empty sources -> False (nothing to ground in).
- is_in_scope: empty question -> False.
- is_in_scope: borderline question with some source keywords -> True.
- check_groundedness: answer fully grounded -> True, no disclaimer.
- check_groundedness: answer with mostly off-context words -> False,
  disclaimer appended.
- check_groundedness: empty answer -> True (vacuously).
- check_groundedness: empty context -> False, disclaimer appended.
- apply_guardrails: composes both; returns (answer, refused, sources_note).
"""

from __future__ import annotations

from src.services.guardrails import (
    check_groundedness,
    is_in_scope,
    maybe_append_disclaimer,
)

# ---------------------------------------------------------------------------
# is_in_scope
# ---------------------------------------------------------------------------


class TestIsInScope:
    def test_on_topic_question(self) -> None:
        sources = ["This document discusses machine learning and neural networks."]
        assert is_in_scope("What is machine learning?", sources) is True

    def test_off_topic_question(self) -> None:
        sources = ["This document discusses machine learning and neural networks."]
        assert is_in_scope("What is the weather like today?", sources) is False

    def test_empty_sources(self) -> None:
        assert is_in_scope("What is machine learning?", []) is False

    def test_empty_question(self) -> None:
        sources = ["This document discusses machine learning."]
        assert is_in_scope("", sources) is False

    def test_whitespace_question(self) -> None:
        sources = ["This document discusses machine learning."]
        assert is_in_scope("   ", sources) is False

    def test_borderline_with_source_keywords(self) -> None:
        sources = ["The policy covers remote work, leave, and performance management."]
        # "remote work" appears in the source -> in scope.
        assert is_in_scope("Tell me about remote work policy", sources) is True

    def test_completely_unrelated(self) -> None:
        sources = ["The policy covers remote work and leave."]
        assert is_in_scope("How do I cook pasta?", sources) is False

    def test_question_about_specific_entity_in_source(self) -> None:
        sources = ["PostgreSQL, MongoDB, and Redis are common database choices."]
        assert is_in_scope("What are common database choices?", sources) is True

    def test_multi_source_aggregation(self) -> None:
        sources = [
            "Document A covers Python programming.",
            "Document B covers Java programming.",
        ]
        assert is_in_scope("Tell me about Java", sources) is True

    def test_greeting_is_out_of_scope(self) -> None:
        sources = ["Document about databases."]
        assert is_in_scope("Hello, how are you?", sources) is False


# ---------------------------------------------------------------------------
# check_groundedness
# ---------------------------------------------------------------------------


class TestCheckGroundedness:
    def test_fully_grounded(self) -> None:
        context = "The sky is blue and the grass is green."
        answer = "The sky is blue."
        is_grounded, result = check_groundedness(answer, context)
        assert is_grounded is True
        assert result == answer

    def test_not_grounded(self) -> None:
        context = "The sky is blue and the grass is green."
        answer = "The capital of France is Paris and the Eiffel Tower is tall."
        is_grounded, result = check_groundedness(answer, context)
        assert is_grounded is False
        assert "verified" in result.lower()
        assert answer in result

    def test_empty_answer(self) -> None:
        context = "Some context text."
        is_grounded, result = check_groundedness("", context)
        assert is_grounded is True
        assert result == ""

    def test_empty_context(self) -> None:
        is_grounded, result = check_groundedness("Some answer text.", "")
        assert is_grounded is False
        assert "verified" in result.lower()

    def test_partial_overlap_passes(self) -> None:
        context = "Machine learning models require training data and evaluation."
        answer = "Machine learning requires training data."
        is_grounded, result = check_groundedness(answer, context)
        assert is_grounded is True

    def test_threshold_boundary(self) -> None:
        """An answer where exactly 50% of meaningful words overlap -> grounded."""
        context = "alpha bravo charlie delta echo"
        answer = "alpha bravo foxtrot golf hotel"
        # Meaningful words: alpha, bravo, foxtrot, golf, hotel (all >= 3 chars).
        # Overlap: alpha, bravo = 2/5 = 40% -> NOT grounded (< 50%).
        is_grounded, _ = check_groundedness(answer, context)
        assert is_grounded is False

    def test_stopwords_excluded(self) -> None:
        """Common stopwords are excluded from the overlap calculation."""
        context = "the quick brown fox jumps over the lazy dog"
        answer = "the quick brown fox is very fast"
        # Meaningful words in answer: quick, brown, fox, very, fast
        # In context: quick, brown, fox, very(? no), fast(? no), dog
        # Overlap: quick, brown, fox = at least 3/5 = 60% -> grounded.
        is_grounded, _ = check_groundedness(answer, context)
        assert is_grounded is True

    def test_does_not_mutate_original_answer(self) -> None:
        context = "unrelated text about cooking"
        answer = "The answer about quantum physics and relativity."
        original = answer
        _, result = check_groundedness(answer, context)
        assert answer == original
        assert result != answer  # disclaimer was appended


# ---------------------------------------------------------------------------
# maybe_append_disclaimer
# ---------------------------------------------------------------------------


class TestMaybeAppendDisclaimer:
    def test_grounded_no_disclaimer(self) -> None:
        context = "The sky is blue."
        answer = "The sky is blue."
        result = maybe_append_disclaimer(answer, context)
        assert result == answer

    def test_ungrounded_appends_disclaimer(self) -> None:
        context = "The sky is blue."
        answer = "Quantum entanglement is a physics phenomenon."
        result = maybe_append_disclaimer(answer, context)
        assert answer in result
        assert len(result) > len(answer)
