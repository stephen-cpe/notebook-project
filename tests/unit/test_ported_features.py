"""Unit tests for ported features (vision swap + P1-P3 bundles).

Covers offline behavior only (AI_MOCK=true): llm_json, rag_budget,
furniture, pipeline_version, resilience, figure_store, diagram validation,
web_search gating, progress_tracker singleflight, difficulty, and the
vision smart gate. No network calls.
"""

from __future__ import annotations


class TestLlmJson:
    def test_extract_object_with_fence_and_trailing_prose(self) -> None:
        from src.services.llm_json import extract_json

        raw = '```json\n{"summary": "hi", "suggested_questions": ["a"]}```\nSome prose.'
        data = extract_json(raw)
        assert isinstance(data, dict)
        assert data["summary"] == "hi"

    def test_extract_none_on_garbage(self) -> None:
        from src.services.llm_json import extract_json

        assert extract_json("no braces here") is None
        assert extract_json("") is None

    def test_extract_array(self) -> None:
        from src.services.llm_json import extract_json_array

        assert extract_json_array('[{"a": 1}] trailing') == [{"a": 1}]
        assert extract_json_array("nothing") is None


class TestRagBudget:
    def test_budget_scales_with_ctx(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services import rag_budget

        small = rag_budget.get_context_budget_chars(num_ctx=32000, max_chars_cap=10**9)
        big = rag_budget.get_context_budget_chars(num_ctx=131072, max_chars_cap=10**9)
        assert big > small

    def test_top_k_never_lowered(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services import rag_budget

        assert rag_budget.get_top_k_for_budget(per_collection_top_k=50, num_ctx=8000) >= 50

    def test_256k_window_budget(self) -> None:
        from src.services import rag_budget

        budget = rag_budget.get_context_budget_chars(num_ctx=262144, max_chars_cap=10**9)
        assert budget == int(262144 * 0.8 * 4)
        assert rag_budget.get_top_k_for_budget(num_ctx=262144) >= 800

    def test_per_collection_scales(self) -> None:
        from src.services.rag_budget import per_collection_k_for

        assert per_collection_k_for(200, 1) == 200
        assert per_collection_k_for(200, 50) == 12
        assert per_collection_k_for(0, 3) == 12

    def test_truncate_keeps_whole_chunks(self) -> None:
        from src.services.rag_budget import truncate_results_to_budget

        results = [
            {"text": "a" * 1000},
            {"text": "b" * 1000},
            {"text": "c" * 1000},
        ]
        kept, chars = truncate_results_to_budget(results, 2500)
        assert len(kept) == 2
        assert chars == 2000
        assert truncate_results_to_budget(results, 0) == ([], 0)

    def test_coverage_ratio(self) -> None:
        from src.services.rag_budget import estimate_coverage_ratio

        assert estimate_coverage_ratio(50, 100) == 0.5
        assert estimate_coverage_ratio(0, 0) == 1.0
        assert estimate_coverage_ratio(200, 100) == 1.0


class TestFurniture:
    def test_strips_recurring_header_footer(self) -> None:
        from src.services.furniture import dedup_furniture

        pages = [
            "Acme Report\nBody page one.\nFooter 1",
            "Acme Report\nBody page two.\nFooter 1",
            "Acme Report\nBody page three.\nFooter 1",
        ]
        cleaned = dedup_furniture(pages)
        assert all("Acme Report" not in p for p in cleaned)
        assert "Body page two." in cleaned[1]

    def test_short_docs_untouched(self) -> None:
        from src.services.furniture import dedup_furniture

        pages = ["Header\nBody"]
        assert dedup_furniture(pages) == pages


class TestPipelineVersion:
    def test_stable_and_sensitive(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services.pipeline_version import compute_pipeline_version

        monkeypatch.setenv("OLLAMA_VISION_MODEL", "glm-5.3-flash:cloud")
        v1 = compute_pipeline_version()
        v2 = compute_pipeline_version()
        assert v1 == v2 and len(v1) == 16
        monkeypatch.setenv("OLLAMA_VISION_MODEL", "other-model")
        assert compute_pipeline_version() != v1


class TestResilience:
    def test_retries_transient(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services.resilience import resilient

        monkeypatch.setattr("time.sleep", lambda s: None)
        calls = 0

        def _flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("connection reset")
            return "ok"

        assert resilient(_flaky) == "ok"
        assert calls == 2

    def test_reraises_client_errors(self) -> None:
        from src.services.resilience import resilient

        def _boom() -> str:
            raise ValueError("bad key")

        try:
            resilient(_boom)
        except ValueError:
            pass
        else:
            raise AssertionError("should have raised")

    def test_pause_on_long_backoff(self) -> None:
        from src.services.resilience import PauseJob, resilient

        def _limited() -> str:
            raise Exception("429 rate limited, retry-after: 9999")

        try:
            resilient(_limited, max_attempts=1)
        except PauseJob:
            pass
        else:
            raise AssertionError("should have paused")


class TestFigureStore:
    def test_save_get_attach(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        from PIL import Image

        import src.services.figure_store as fs

        monkeypatch.setattr(fs, "_figures_root", lambda data_dir="./data": tmp_path)
        img = Image.new("RGB", (8, 8), "white")
        fname = fs.save_figure("a" * 64, img, caption="Cap", label="L", data_dir=str(tmp_path))
        assert fname is not None
        figs = fs.get_figures("a" * 64, str(tmp_path))
        assert len(figs) == 1
        sources = [{"filename": "doc.png"}]
        out = fs.attach_figures(
            sources, {"doc.png": "a" * 64}, notebook_id=1, data_dir=str(tmp_path)
        )
        assert out[0]["figures"][0]["url"].startswith("/notebooks/1/figures/")

    def test_traversal_guarded(self) -> None:
        from src.services.figure_store import get_figures

        assert get_figures("../evil") == []


class TestDiagram:
    def test_validate_mermaid(self) -> None:
        from src.services.diagram import detect_mermaid_type, validate_mermaid

        good = "flowchart TD\n    A-->B\n    B-->C"
        assert detect_mermaid_type(good) == "flowchart"
        assert validate_mermaid(good) is True
        assert validate_mermaid("just some text") is False
        assert validate_mermaid("") is False

    def test_convert_mock_never_raises(self, monkeypatch) -> None:  # noqa: ANN001
        from PIL import Image

        from src.services.diagram import convert_figure

        monkeypatch.setenv("AI_MOCK", "true")
        img = Image.new("RGB", (8, 8), "white")
        result = convert_figure(img)
        assert result.convertible is False


class TestWebSearch:
    def test_blocklist_and_sanitize(self) -> None:
        from src.services.web_search_service import classify_topic_source, sanitize_query

        assert classify_topic_source("my salary report") == "proprietary"
        assert classify_topic_source("machine learning tutorial") == "public"
        assert "a@b.com" not in sanitize_query("contact a@b.com now")

    def test_disabled_returns_empty(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services.web_search_service import web_search

        monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
        monkeypatch.setenv("AI_MOCK", "true")
        assert web_search("python tutorial") == []


class TestProgressTracker:
    def test_singleflight(self) -> None:
        from src.services.progress_tracker import claim_job, finish_job, reset_tracker

        reset_tracker()
        assert claim_job("audio:1", label="A", notebook_id=1) is True
        assert claim_job("audio:1", label="A", notebook_id=1) is False
        finish_job("audio:1")
        assert claim_job("audio:1", label="A", notebook_id=1) is True
        reset_tracker()


class TestDifficulty:
    def test_normalize(self) -> None:
        from src.services.difficulty import difficulty_instruction, normalize_difficulty

        assert normalize_difficulty("easy") == "Easy"
        assert normalize_difficulty("bogus") == "Normal"
        assert "analogy" in difficulty_instruction("Easy").lower()

    def test_summary_dedup(self) -> None:
        from src.services.summary_service import dedup_questions

        out = dedup_questions(["What is X?", "what is x? ", "  ", "Why Y?"])
        assert out == ["What is X?", "Why Y?"]


class TestVisionGate:
    def test_short_text_needs_vision(self, tmp_path) -> None:  # noqa: ANN001
        from src.services.ocr_service import pdf_needs_vision_ocr

        assert pdf_needs_vision_ocr(str(tmp_path / "nope.pdf"), "") is True

    def test_image_content_types_known(self) -> None:
        from src.services.document_parser import (
            IMAGE_CONTENT_TYPES,
            detect_content_type,
            validate_magic_bytes,
        )

        assert detect_content_type("photo.png") == "png"
        assert detect_content_type("scan.JPEG") == "jpeg"
        assert "png" in IMAGE_CONTENT_TYPES
        assert validate_magic_bytes.__name__ == "validate_magic_bytes"


class TestSectionDigest:
    def test_split_keeps_full_coverage(self) -> None:
        from src.services.section_digest import split_into_sections

        paras = [f"Paragraph {i} with some substance. " * 20 for i in range(10)]
        text = "\n\n".join(paras)
        sections = split_into_sections(text, 6000)
        assert sections
        assert all(len(s) <= 6000 for s in sections)
        for p in paras:
            assert any(p[:60] in s for s in sections)

    def test_split_empty(self) -> None:
        from src.services.section_digest import split_into_sections

        assert split_into_sections("") == []
        assert split_into_sections("   \n\n  ") == []

    def test_hard_splits_giant_paragraph(self) -> None:
        from src.services.section_digest import split_into_sections

        para = "Sentence one. Sentence two. " * 500
        sections = split_into_sections(para, 1000)
        assert len(sections) > 1
        assert all(len(s) <= 1000 for s in sections)

    def test_build_cache_and_rebuild(self, app: object) -> None:  # noqa: ANN001
        from src.repositories import content_registry_repo
        from src.services.section_digest import (
            ensure_source_digest,
            get_cached_digest,
            reset_digest_build_state,
        )

        reset_digest_build_state()
        with app.app_context():
            content_registry_repo.get_or_create("d" * 64, "doc_d", "Alpha text. " * 200, 2200)
            first = ensure_source_digest("d" * 64, "doc.txt")
            assert first
            assert "[Section 1/" in first
            assert ensure_source_digest("d" * 64, "doc.txt") == first
            content_registry_repo.save_digest("d" * 64, "stale digest", "stale-pipeline")
            assert get_cached_digest("d" * 64) == ""
            rebuilt = ensure_source_digest("d" * 64, "doc.txt")
            assert rebuilt and rebuilt != "stale digest"
        reset_digest_build_state()

    def test_save_digest_missing_entry(self, app: object) -> None:  # noqa: ANN001
        from src.repositories import content_registry_repo

        with app.app_context():
            assert content_registry_repo.save_digest("e" * 64, "x", "p") is None

    def test_ensure_notebook_digests(self, app: object) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook, Source, User
        from src.repositories import content_registry_repo
        from src.services.auth_service import hash_password
        from src.services.section_digest import (
            ensure_notebook_digests,
            reset_digest_build_state,
        )

        reset_digest_build_state()
        with app.app_context():
            u = User(username="digestnb", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Digest NB")
            db.session.add(nb)
            db.session.commit()
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="a.txt",
                    content_hash="f" * 64,
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
            content_registry_repo.get_or_create(
                "f" * 64, "doc_f", "Digestible content here. " * 100, 2500
            )
            assert ensure_notebook_digests(nb.id) == 1
        reset_digest_build_state()

    def test_notebook_digest_text_truncates(self, app: object) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook, Source, User
        from src.repositories import content_registry_repo
        from src.services.auth_service import hash_password
        from src.services.pipeline_version import compute_pipeline_version
        from src.services.section_digest import notebook_digest_text

        with app.app_context():
            u = User(username="truncnb", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Trunc NB")
            db.session.add(nb)
            db.session.commit()
            for i, h in enumerate(("t1" * 32, "t2" * 32)):
                db.session.add(
                    Source(
                        notebook_id=nb.id,
                        filename=f"{i}.txt",
                        content_hash=h,
                        content_type="txt",
                        status="ready",
                    )
                )
                content_registry_repo.get_or_create(h, f"doc_{i}", "body text", 9)
                content_registry_repo.save_digest(
                    h, f"DIGEST-{i}-" + "x" * 2000, compute_pipeline_version()
                )
            db.session.commit()
            assert notebook_digest_text(99999, 10**9) == ""
            small = notebook_digest_text(nb.id, 2500)
            assert "DIGEST-0" in small
            assert "DIGEST-1" not in small
            big = notebook_digest_text(nb.id, 10**9)
            assert "DIGEST-0" in big and "DIGEST-1" in big


class TestDigestWiring:
    def _seed_notebook(self, app: object, username: str) -> int:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook, Source, User
        from src.repositories import content_registry_repo
        from src.services.auth_service import hash_password
        from src.services.pipeline_version import compute_pipeline_version

        with app.app_context():
            u = User(username=username, password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Wired NB")
            db.session.add(nb)
            db.session.commit()
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="doc.txt",
                    content_hash="h" * 64,
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
            body = "This document discusses databases and indexing at length. " * 60
            content_registry_repo.get_or_create("h" * 64, "doc_h", body, len(body))
            content_registry_repo.save_digest(
                "h" * 64, "DIGEST-MARKER full summary of databases.", compute_pipeline_version()
            )
            return nb.id

    def test_chat_prompt_carries_cached_digest(self, app: object, monkeypatch) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook
        from src.services.chat_service import ChatService

        nb_id = self._seed_notebook(app, "wired1")
        svc = ChatService()
        seen: list[str] = []
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            orig = svc._client.chat

            def _recorder(messages: object) -> str:  # noqa: ANN001
                seen.append(str(messages))
                return orig(messages)  # type: ignore[operator]

            monkeypatch.setattr(svc._client, "chat", _recorder)
            result = svc.chat_sync(nb, "What databases are mentioned?")
            assert "DIGEST-MARKER" in seen[0]
            assert result["answer"]

    def test_chat_falls_back_to_extractive(self, app: object, monkeypatch) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook
        from src.services.chat_service import ChatService

        nb_id = self._seed_notebook(app, "wired2")
        from src.repositories import content_registry_repo

        with app.app_context():
            entry = content_registry_repo.get_by_hash("h" * 64)
            assert entry is not None
            entry.section_digest = None
            db.session.commit()
        svc = ChatService()
        seen: list[str] = []
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            orig = svc._client.chat

            def _recorder(messages: object) -> str:  # noqa: ANN001
                seen.append(str(messages))
                return orig(messages)  # type: ignore[operator]

            monkeypatch.setattr(svc._client, "chat", _recorder)
            svc.chat_sync(nb, "What databases are mentioned?")
            assert "DIGEST-MARKER" not in seen[0]
            assert "[Document 1 overview]" in seen[0]

    def test_summary_prompt_carries_digest(self, app: object, monkeypatch) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook
        from src.services.summary_service import SummaryService

        nb_id = self._seed_notebook(app, "wired3")
        svc = SummaryService()
        seen: list[str] = []
        with app.app_context():
            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            orig = svc._client.chat

            def _recorder(messages: object) -> str:  # noqa: ANN001
                seen.append(str(messages))
                return orig(messages)  # type: ignore[operator]

            monkeypatch.setattr(svc._client, "chat", _recorder)
            svc.generate_summary(nb)
            assert seen and "DIGEST-MARKER" in seen[0]


class TestCitationCap:
    def test_format_sources_limit(self) -> None:
        from src.services.rag_retriever import format_sources

        results = [
            {"text": f"t{i}", "filename": f"f{i}.pdf", "page": i, "score": 1.0} for i in range(10)
        ]
        capped = format_sources(results, limit=3)
        assert [s["filename"] for s in capped] == ["f0.pdf", "f1.pdf", "f2.pdf"]
        assert len(format_sources(results)) == 10
        assert format_sources(results, limit=0) == []

    def test_citation_sets_caps_display_keeps_total(self, monkeypatch) -> None:  # noqa: ANN001
        from src.services.chat_service import ChatService

        monkeypatch.setenv("RAG_MAX_CITATIONS", "3")
        svc = ChatService()
        results = [
            {
                "text": f"t{i}",
                "filename": f"f{i}.pdf",
                "page": i,
                "chunk_index": i,
                "score": 1.0 - i * 0.01,
            }
            for i in range(10)
        ]
        shown, total = svc._citation_sets(results)
        assert total == 10
        assert len(shown) == 3
        assert shown[0]["filename"] == "f0.pdf"

    def test_chat_result_carries_source_total(self, app: object) -> None:  # noqa: ANN001
        from src.extensions import db
        from src.models import Notebook, Source, User
        from src.repositories import content_registry_repo
        from src.services.auth_service import hash_password
        from src.services.chat_service import ChatService

        with app.app_context():
            u = User(username="cite1", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Cite NB")
            db.session.add(nb)
            db.session.commit()
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="doc.txt",
                    content_hash="c" * 64,
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
            content_registry_repo.get_or_create(
                "c" * 64,
                "doc_c",
                "This document discusses databases, SQL, and indexing.",
                55,
            )
            svc = ChatService()
            result = svc.chat_sync(nb, "What databases are mentioned?")
            assert result["source_total"] == len(result["sources"])
            assert len(result["sources"]) <= 12

    def test_stream_final_frame_carries_source_total(self, app: object) -> None:  # noqa: ANN001
        import json

        from src.extensions import db
        from src.models import Notebook, Source, User
        from src.repositories import content_registry_repo
        from src.services.auth_service import hash_password
        from src.services.chat_service import ChatService

        with app.app_context():
            u = User(username="cite2", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Cite Stream NB")
            db.session.add(nb)
            db.session.commit()
            db.session.add(
                Source(
                    notebook_id=nb.id,
                    filename="doc.txt",
                    content_hash="d" * 63 + "e",
                    content_type="txt",
                    status="ready",
                )
            )
            db.session.commit()
            content_registry_repo.get_or_create(
                "d" * 63 + "e",
                "doc_d",
                "This document discusses databases, SQL, and indexing.",
                55,
            )
            svc = ChatService()
            frames = list(svc.chat_stream(nb, "What databases are mentioned?"))
            done = [json.loads(f.removeprefix("data: ")) for f in frames if '"done": true' in f]
            assert done and "source_total" in done[-1]
