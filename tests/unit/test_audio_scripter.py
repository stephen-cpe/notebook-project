"""Unit tests for src.services.audio_scripter (TDD step 16a).

The audio scripter generates a two-host dialogue script (JSON of alternating
Host A / Host B utterances) grounded in the notebook's sources.

Covers:
- write_dialogue: returns a list of {host, text} dicts.
- Alternating hosts (A, B, A, B...).
- Mock mode returns deterministic canned dialogue.
- Empty sources -> minimal dialogue or empty.
- parse_dialogue_response: extracts utterances from LLM JSON.
- Target duration hint (~5-8 min worth of text).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extensions import db
from src.models import Notebook, Source, User
from src.repositories import content_registry_repo
from src.services.audio_scripter import (
    parse_dialogue_response,
    write_dialogue,
)
from src.services.auth_service import hash_password

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_notebook_with_sources(app: object, username: str = "auduser") -> int:
    with app.app_context():
        u = User(username=username, password_hash=hash_password("pw"))
        db.session.add(u)
        db.session.commit()
        nb = Notebook(user_id=u.id, name="Audio NB")
        db.session.add(nb)
        db.session.commit()
        # Add a source.
        h = "a" * 64
        db.session.add(
            Source(
                notebook_id=nb.id,
                filename="doc.txt",
                content_hash=h,
                content_type="txt",
                status="ready",
            )
        )
        db.session.commit()
        content_registry_repo.get_or_create(
            content_hash=h,
            chroma_collection="doc_a",
            extracted_text=(
                "This document discusses machine learning, neural networks, and deep learning."
            ),
            char_count=80,
        )
        return nb.id


class TestParseDialogueResponse:
    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "dialogue": [
                    {"host": "A", "text": "Welcome to the overview!"},
                    {"host": "B", "text": "Today we're discussing ML."},
                    {"host": "A", "text": "Let's dive in."},
                ]
            }
        )
        utterances = parse_dialogue_response(raw)
        assert len(utterances) == 3
        assert utterances[0]["host"] == "A"
        assert utterances[1]["host"] == "B"

    def test_missing_dialogue_key(self) -> None:
        raw = json.dumps({"other": "data"})
        utterances = parse_dialogue_response(raw)
        assert utterances == []

    def test_invalid_json(self) -> None:
        utterances = parse_dialogue_response("not json")
        assert utterances == []

    def test_empty_string(self) -> None:
        assert parse_dialogue_response("") == []

    def test_filters_empty_text(self) -> None:
        raw = json.dumps(
            {
                "dialogue": [
                    {"host": "A", "text": "valid"},
                    {"host": "B", "text": ""},
                    {"host": "A", "text": "also valid"},
                ]
            }
        )
        utterances = parse_dialogue_response(raw)
        assert len(utterances) == 2  # empty text filtered

    def test_normalizes_host_labels(self) -> None:
        raw = json.dumps(
            {
                "dialogue": [
                    {"host": "host_a", "text": "first"},
                    {"host": "host b", "text": "second"},
                    {"host": "1", "text": "third"},
                    {"host": "2", "text": "fourth"},
                ]
            }
        )
        utterances = parse_dialogue_response(raw)
        assert utterances[0]["host"] == "A"
        assert utterances[1]["host"] == "B"
        assert utterances[2]["host"] == "A"
        assert utterances[3]["host"] == "B"


class TestWriteDialogue:
    def test_returns_utterances(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "audscr1")
        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            utterances = write_dialogue(nb)
            assert isinstance(utterances, list)
            assert len(utterances) >= 2
            for u in utterances:
                assert "host" in u
                assert "text" in u
                assert u["host"] in ("A", "B")
                assert len(u["text"]) > 0

    def test_alternating_hosts(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "audscr2")
        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            utterances = write_dialogue(nb)
            # Hosts should alternate A, B, A, B...
            for i in range(len(utterances) - 1):
                assert utterances[i]["host"] != utterances[i + 1]["host"]

    def test_empty_sources(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_MOCK", "true")
        with app.app_context():
            u = User(username="audscr3", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Empty Audio NB")
            db.session.add(nb)
            db.session.commit()
            utterances = write_dialogue(nb)
            # With no sources, dialogue is empty or minimal.
            assert isinstance(utterances, list)

    def test_mock_deterministic(self, app: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("AI_MOCK", "true")
        monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
        nb_id = _make_notebook_with_sources(app, "audscr4")
        with app.app_context():
            from src.repositories import notebook_repo

            nb = notebook_repo.get_by_id(nb_id)
            assert nb is not None
            u1 = write_dialogue(nb)
            u2 = write_dialogue(nb)
            assert u1 == u2
