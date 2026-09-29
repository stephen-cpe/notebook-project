# Architecture Document -- notebook-project

**Version:** 0.3

This document defines the module layout, data model, sequence flows, and tech
choices for the implemented codebase.

---

## 1. Design decisions

| # | Question | Resolution |
|---|----------|------------|
| 1 | Chat delivery | SSE streaming. A Flask generator yields `data: {token}\n\n` frames; a final frame carries `{sources, source_total, latency_ms, coverage_ratio, done: true}`. A non-streaming `/chat/sync` endpoint is also exposed for tests. |
| 2 | Two-host voices | MP3, `en-US-AvaNeural` (Ava) + `en-US-AndrewNeural` (Andrew). Per-utterance audio synthesized via edge-TTS and concatenated via pydub. |
| 3 | Audio format | MP3 (one file per notebook version). |
| 4 | In-app model | Single model `gemma4:31b-cloud` via Ollama Cloud. Hidden from the user. Thinking enabled server-side via the `<|think|>` system-prompt token when `ENABLE_THINKING=true`. |
| 5 | Admin seeding | `init_db.sql` seeds a fallback admin (`admin` / `change-me`). Real deployments set `ADMIN_USERNAME` / `ADMIN_PASSWORD` and run `flask seed-admin` (idempotent). Disabled users cannot authenticate and existing sessions are invalidated. |
| 6 | Summary trigger | Auto on every notebook change. Idempotent via `content_signature`. User can force-regenerate. |
| 7 | Source panel | List + modal detail + inline actions (view text, rename, delete). |
| 8 | Embedding/vision provider | Embeddings support `local` (default, sentence-transformers) and `hf_inference` (opt-in). Vision understanding is consolidated on Ollama Cloud (`glm-5.3-flash`); the old `OCR_PROVIDER` local/HF setting is a deprecated alias that always resolves to the vision backend. |
| 9 | Vector store backend | `CHROMA_DB=local` (default) or `CHROMA_DB=cloud` with graceful fallback. |
| 10 | Voice mode | Toggle hands-free conversation over HTTP `/voice/turn`: the mic stays open, utterances auto-send on pause (client VAD), transcribe with faster-whisper, answer via `ChatService.chat_sync`, synthesize reply via edge-TTS with auto-play and barge-in. A `/voice` SocketIO namespace still exists server-side, but the V2 web client is HTTP-only and tracks status locally. The spoken reply has markdown and citation brackets stripped for natural narration. Disabled by default (`VOICE_ENABLED=false`). |
| 11 | Overview source selection | Summary, audio, and video generators share one `select_sources_within_budget()` helper (`context_builder.py`) that orders sources by upload time (`created_at`) and includes as many as fit within `OVERVIEW_MAX_CONTEXT_CHARS` (default 150000). The first source is always included even if oversized (truncated with a marker). On top of the raw texts, each generator prepends the cached section digests (decision 14), so material past the budget is still represented. Dropped sources are logged. |
| 12 | Vision + figures scope | `glm-5.3-flash` via Ollama Cloud handles text/table OCR, figure description, and diagram reinterpretation. Text-layer PDFs skip vision entirely (smart gate); vision runs for scanned/image-heavy PDFs (rendered via Poppler), DOCX/PPTX embedded images (`word/media/`, `ppt/media/`), and PNG/JPG/JPEG uploads (always vision, no text layer). TXT/MD never trigger vision. Image uploads and up to 2 embedded Office images per file persist as thumbnails under `DATA_DIR/figures/<hash>/` with vision captions; convertible diagrams additionally store validated Mermaid. Orphaned figure dirs are removed with the content they belong to. |
| 13 | Budgeted retrieval (256K window) | `top_k` is derived from `OLLAMA_NUM_CTX` (default 262144) capped by the visible `RAG_MAX_TOP_K` ceiling (default 200). Per-source depth scales (`per_collection_k_for`, floor 12) so one giant document can fill its share. Retrieved chunks truncate to `RAG_MAX_CONTEXT_CHARS` (default 700000) minus digest room, whole chunks only. |
| 14 | Full-coverage section digest | `section_digest.py` splits each source into ~6000-char sections (max 80; widened, never truncated, when over), summarizes every section with the chat model, and caches the stitched digest in `content_registry` keyed by content hash + pipeline fingerprint (migration `0006_source_digest`). Builds run in background threads (summary/audio/video jobs); chat only reads the cache and falls back to the cheap extractive digest when cold. |
| 15 | Citation display | Retrieval breadth is unchanged; only display is capped at `RAG_MAX_CITATIONS` (default 12, top-N by relevance). The UI renders one collapsed "Sources (N of M) · X% read" toggle per answer with the file/page-grouped list, coverage, and thumbnails inside (`chat-ui.js`). History renders the persisted capped list. |
| 16 | Background tasks bell | `progress_tracker.py` (in-memory, thread-safe, 2h expiry) provides per-notebook singleflight for media jobs plus a `GET /tasks` feed (recent jobs + unread count) and `POST /tasks/read`. The header bell polls it (`tasks.js`). |
| 17 | Notebook PDF export | `GET /notebooks/<id>/export` renders summary + sources + chat history to PDF via fpdf2 (latin-1 safe, 501 when fpdf2 is missing). |
| 18 | Answer difficulty | Per-user `difficulty` (Easy/Normal/Hard, default Normal) on `users`, set in Settings and appended to the chat system prompt. |

## 2. Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.13 |
| Web framework | Flask 3.1 + Flask-Login + Flask-Migrate + Flask-SQLAlchemy + Flask-WTF + Flask-SocketIO |
| DB | PostgreSQL via SQLAlchemy + Alembic |
| Vector store | ChromaDB (local PersistentClient or CloudClient with fallback) |
| Embeddings | Qwen3-Embedding-0.6B (local sentence-transformers or HF Inference API) |
| Vision | Ollama Cloud `glm-5.3-flash` (OCR, tables, figure description, diagram reinterpretation) |
| Chat LLM | Ollama Cloud, `gemma4:31b-cloud` |
| STT | faster-whisper (local, mock in test mode) |
| TTS | edge-TTS (neural voices) |
| Audio concat | pydub + audioop-lts (Python 3.13 compatibility) |
| Video | ffmpeg subprocess (slide images + TTS narration -> MP4) |
| PDF export | fpdf2 (notebook summary + sources + chat history) |
| Frontend | Jinja2 templates + vanilla JS + Bootstrap 5 (dark theme) |
| Lint/format | Ruff |
| Type check | mypy strict |
| Tests | pytest + pytest-cov |
| CI | GitHub Actions |

## 3. Module layout

```
notebook-project/
|-- .env.example
|-- .editorconfig
|-- .pre-commit-config.yaml
|-- .github/workflows/ci.yml
|-- pyproject.toml
|-- requirements.txt
|-- init_db.sql
|-- app.py                      # entry point
|-- docs/
|   |-- SRS.md
|   `-- ARCHITECTURE.md
|-- migrations/
|   |-- env.py
|   `-- versions/              # ...0004_media_generations, 0005_user_difficulty, 0006_source_digest
|-- src/
|   |-- app.py                  # create_app() factory + SocketIO init
|   |-- config.py               # Config (env-driven: vision, RAG budgets, digest, diagram, web, difficulty)
|   |-- extensions.py           # db, login_manager, migrate, socketio
|   |-- models.py               # User (+difficulty), Notebook (+audio/video_generation), Source, ChatMessage, ContentRegistry (+digest cols)
|   |-- repositories/           # user_repo, notebook_repo, source_repo, chat_repo, content_registry_repo (+save_digest)
|   |-- services/
|   |   |-- exceptions.py       # typed hierarchy
|   |   |-- auth_service.py     # password hashing, signup, authenticate, AuthUser (+difficulty)
|   |   |-- embeddings.py       # Qwen3-Embedding (local or HF Inference API)
|   |   |-- vector_store.py     # ChromaDB (local or Cloud with fallback); doc_<hash[:50]>_<fingerprint>
|   |   |-- document_parser.py  # PDF/DOCX/PPTX/TXT/MD/PNG/JPG/JPEG extraction + magic bytes + Office image extraction
|   |   |-- ocr_service.py      # glm-5.3-flash vision via Ollama Cloud; smart gate (pdf_needs_vision_ocr); describe_figure
|   |   |-- chunker.py          # RecursiveCharacterTextSplitter
|   |   |-- furniture.py        # header/footer dedup without vectors
|   |   |-- ingestion.py        # parse -> vision fallback (smart-gated) -> chunk -> embed -> store; figure persist; pipeline log
|   |   |-- pipeline_version.py # pipeline fingerprint for digest staleness
|   |   |-- resilience.py       # retry with backoff; PauseJob instead of failing long jobs
|   |   |-- context_builder.py  # select_sources_within_budget (shared by summary/audio/video)
|   |   |-- rag_retriever.py    # multi-collection retrieve + merge + recovery; format_sources(limit); extractive digest fallback
|   |   |-- rag_budget.py       # window-derived top_k, per_collection_k_for, truncate_results_to_budget, coverage ratio
|   |   |-- section_digest.py   # map-reduce section digests, cached per content hash; ensure_* (background) vs get_cached (chat)
|   |   |-- llm_json.py         # robust JSON extraction (fence-strip + raw_decode)
|   |   |-- guardrails.py       # scope + groundedness checks
|   |   |-- ollama_client.py    # Ollama Cloud chat (sync + stream) + chat_with_images (vision)
|   |   |-- chat_service.py     # scope -> budgeted retrieve -> digest layer -> prompt -> stream -> persist; citation cap
|   |   |-- summary_service.py  # summary + deduped questions + opt-in web augmentation; digest ensure
|   |   |-- difficulty.py       # Easy/Normal/Hard instructions
|   |   |-- web_search_service.py # opt-in, fail-closed web search + synthesis (reuses Ollama Cloud creds)
|   |   |-- figure_store.py     # DATA_DIR/figures/<hash>/ thumbnails + manifest; attach_figures
|   |   |-- diagram.py          # figure -> validated Mermaid (confidence gate + vision re-verify)
|   |   |-- audio_scripter.py   # two-host dialogue JSON (digest ensure + digest block)
|   |   |-- audio_service.py    # edge-TTS synth + concat MP3
|   |   |-- video_scripter.py   # narrated slide script (digest ensure + digest block)
|   |   |-- video_service.py    # ffmpeg slides + TTS -> MP4
|   |   |-- tts_utils.py        # shared speaker_to_voice + synthesize + clean_for_tts (+asyncgen FD fix)
|   |   |-- stt_service.py      # faster-whisper STT (mock + real backends)
|   |   |-- voice_service.py    # STT -> chat -> TTS voice turn pipeline (+source_total)
|   |   |-- progress_tracker.py # in-memory singleflight + bell feed (2h expiry)
|   |   |-- cleanup_service.py  # reference-counted content cleanup incl. figure dirs; notebook media dirs
|   |   `-- jobs.py             # background workers (audio, video, summary) + singleflight + crash recovery
|   |-- realtime/
|   |   `-- __init__.py         # /voice SocketIO namespace (server-side; V2 web client is HTTP-only)
|   |-- routes/
|   |   |-- __init__.py          # blueprint registration + error handlers + CSP
|   |   |-- _helpers.py          # require_owner, require_admin
|   |   |-- auth.py              # signup, login, logout, settings (+difficulty), reset-password
|   |   |-- admin.py             # admin user management
|   |   |-- notebooks.py         # notebook CRUD
|   |   |-- sources.py           # upload, list, delete, text, figure file, rename
|   |   |-- chat.py              # SSE streaming + sync + clear + history
|   |   |-- summary.py           # get + regenerate
|   |   |-- audio.py             # request, status, file, delete
|   |   |-- video.py             # request, status, file, delete
|   |   |-- voice.py             # voice turn (HTTP POST, +source_total) + reply file serving
|   |   |-- tasks.py             # bell feed (GET /tasks) + mark read
|   |   |-- export.py            # notebook PDF download
|   |   `-- index.py             # redirect
|   |-- static/
|   |   |-- css/app.css
|   |   |-- js/chat-markdown.js  # lightweight markdown-to-HTML renderer (no external libs)
|   |   |-- js/chat-ui.js        # shared chat helpers (messages, typing, collapsible grouped citations)
|   |   |-- js/app.js            # upload, chat, audio, video, source actions
|   |   |-- js/voice.js          # voice-mode toggle: VAD loop, auto-send, barge-in + voice turn
|   |   |-- js/tasks.js          # header bell polling
|   |   `-- js/settings.js
|   `-- templates/
|       |-- base.html            # layout + tasks bell
|       |-- notebook.html        # 3-panel app + export button
|       |-- settings.html        # speakers + difficulty
|       |-- reset_password.html
|       |-- error.html
|       |-- auth/ (login.html, signup.html)
|       |-- notebooks/ (list.html)
|       `-- admin/ (dashboard.html)
|-- tests/
|   |-- conftest.py              # offline flags, temp SQLite, vector-store + tracker reset
|   |-- fixtures/
|   |-- unit/                    # incl. test_ported_features.py (budgets, digests, citations, figures, tasks)
|   |-- routes/
|   `-- integration/
`-- data/                        # gitignored: chroma_db/, audio/, video/, voice/, figures/
```

## 4. Data model (SQLAlchemy)

```
User
  id, username (UNIQUE), password_hash, role (user|admin|disabled),
  avatar, audio_speaker_a, audio_speaker_b, video_speaker, voice_speaker,
  difficulty (Easy|Normal|Hard, default Normal),
  created_at

Notebook
  id, user_id (FK CASCADE), name (1..120), description,
  summary, suggested_questions (JSON), content_signature,
  audio_path, audio_status, audio_error, audio_generation,
  video_path, video_status, video_error, video_generation,
  created_at, updated_at

Source
  id, notebook_id (FK CASCADE), filename, content_hash (sha256),
  content_type (pdf|docx|pptx|txt|md|png|jpg|jpeg), char_count, page_count,
  status, error_message, created_at
  UNIQUE(notebook_id, content_hash), INDEX(content_hash)

ChatMessage
  id, notebook_id (FK CASCADE), role, content,
  sources_json (capped display list), metadata_json, latency_ms, created_at

ContentRegistry (global, not user-scoped)
  content_hash (PK), chroma_collection, embedding_fingerprint,
  extracted_text, char_count,
  section_digest (nullable), digest_pipeline (nullable),
  created_at
```

## 5. Key sequence flows

### 5.1 Source upload + ingestion

```
POST /notebooks/<id>/sources -> routes/sources.py
  validate owner, magic bytes, size cap
  compute sha256; check duplicate
  create Source row (status=queued)
  ingest: parse (+furniture strip on PDF) -> smart gate ->
    (vision fallback?) -> chunk -> embed -> store + registry
  persist figures (images always; <=2 embedded Office images)
  update status to ready/partial/failed
```

- Smart gate: text-layer PDFs skip vision entirely; vision runs for
  scanned/image-heavy PDFs, DOCX/PPTX embedded images, and PNG/JPG/JPEG
  (always). Image uploads also get a figure description + optional
  diagram-to-Mermaid conversion (skipped in mock mode).
- The pipeline fingerprint is logged at ingest start.
- Idempotent: if `ContentRegistry.content_hash` exists with a current
  embedding fingerprint, embedding is skipped (or rebuilt from cache).

### 5.2 Chat (SSE)

```
POST /notebooks/<id>/chat -> routes/chat.py
  validate owner
  guardrails.is_in_scope() -> refuse if off-topic
  budgeted retrieval: top_k from window budget (cap RAG_MAX_TOP_K),
    scaled per-source depth -> truncate to char budget
  digest layer: cached section digests (or extractive fallback when cold)
  build prompt (system + difficulty + digest + context + question + <|think|>)
  ollama_client.stream() -> yield tokens as SSE frames
  guardrails.check_groundedness() -> maybe append disclaimer
  persist user + assistant ChatMessages (capped citation list)
  yield final frame {sources, source_total, latency_ms, coverage_ratio, done}
```

The UI renders one collapsed "Sources (N of M) · X% read" toggle per answer
instead of the full chip list; retrieval breadth is unaffected.

### 5.3 Audio Overview

```
POST /notebooks/<id>/audio -> routes/audio.py
  enqueue background job:
    1. scripting: audio_scripter -> Ollama Cloud -> dialogue JSON
    2. synthesizing: edge-TTS per utterance -> concat MP3
    3. persist audio_path, status=ready
```

### 5.4 Video Overview

```
POST /notebooks/<id>/video -> routes/video.py
  enqueue background job:
    1. scripting: video_scripter -> slide JSON
    2. render slide images (PIL)
    3. edge-TTS narration per slide
    4. ffmpeg: combine slides + narration -> MP4
    5. persist video_path, status=ready
```

### 5.5 Voice mode

```
POST /notebooks/<id>/voice/turn -> routes/voice.py
  validate owner, save audio to temp file
  VoiceService.run_voice_turn():
    1. STTService.transcribe() -> faster-whisper (or mock)
    2. if empty transcript -> return error
    3. ChatService.chat_sync() -> RAG + LLM + persist (same as text chat)
    4. clean_for_tts(answer) -> strip markdown + citations
    5. synthesize_utterance() -> edge-TTS -> MP3 reply
    6. return {transcript, answer, sources, source_total, reply_audio_url}
  serve reply via GET /notebooks/<id>/voice/reply/<filename> (owner-only)
```

Client (`js/voice.js`, V2): the mic button toggles voice mode, swapping the
text input for a voice panel (status orb, live transcript, End button). One
`getUserMedia` stream serves the whole session; a per-utterance
`MediaRecorder` plus `AnalyserNode` voice-activity detection auto-sends after
~1.2 s of silence (max-length fallback). Replies auto-play, then listening
resumes; speech during playback barges in, speech while thinking aborts the
request. Turns render through the shared `ChatUI` helpers.

The SocketIO `/voice` namespace is still registered server-side, but the V2
web client does not connect to it -- status is tracked locally and audio has
always gone over the HTTP endpoint, never SocketIO binary events.

### 5.6 Content cleanup on delete

```
DELETE source -> routes/sources.py
  delete Source row
  cleanup_orphaned_content(hash, exclude_source_id):
    if no other Source references this hash:
      delete ChromaDB collection + ContentRegistry entry + figures/<hash>/
      (figures only after vector deletion is confirmed)

DELETE notebook -> routes/notebooks.py
  snapshot source hashes
  delete Notebook (cascades to sources + chat)
  cleanup_orphaned_content() per hash
  cleanup_notebook_media() -> delete audio/video/voice dirs
```

### 5.7 Section digest build (background)

```
summary/audio/video job -> section_digest.ensure_notebook_digests(notebook_id)
  per ready/partial source (in-progress guard dedups concurrent builds):
    get_cached_digest(hash): hit when digest_pipeline == current fingerprint
    else build_source_digest(hash):
      split_into_sections(text, ~6000 chars; widened, never truncated, if >80)
      summarize every section with the chat model (map)
      stitch with [Section i/n] headers; save_digest(hash, digest, pipeline)
```

Chat never builds: it reads `notebook_digest_text()` (bounded by
`RAG_DIGEST_MAX_CHARS`) and falls back to the extractive digest when cold.

### 5.8 Background tasks bell

```
launch_audio_job / launch_video_job -> progress_tracker.claim_job("audio|video:<id>")
  duplicate launch while running is skipped (singleflight)
  finish_job / fail_job on completion
GET /tasks -> {tasks, unread} for the header bell (tasks.js polls)
POST /tasks/read -> mark one key or all read
```

### 5.9 Notebook PDF export

```
GET /notebooks/<id>/export -> routes/export.py
  summary + source list + chat history -> fpdf2 -> attachment
  (latin-1 safe; 501 when fpdf2 is missing)
```

## 6. Mocking strategy (tests)

- `AI_MOCK=true` -> deterministic stubs for LLM, embeddings, vision
  (`chat_with_images`), OCR, digests, TTS, STT. No network calls, no model
  downloads.
- `CI=true` -> ChromaDB EphemeralClient (in-memory).
- Tests use a temp file-based SQLite DB (not `:memory:`) so background threads
  and multiple app contexts share one database. The vector store and the
  task tracker are reset per test.
- Integration tests gated behind `RUN_INTEGRATION=1`.

## 7. Security specifics

- Magic-bytes file type validation.
- Filename sanitization (basename only).
- Owner-scoped routes return 404 (not 403). Figure files additionally require
  the hash to belong to the notebook, with traversal guards + resolved-path
  containment checks.
- `SECRET_KEY` must not be placeholder in production.
- CSRF on state-changing routes via Flask-WTF.
- Session cookie hardening: HttpOnly, SameSite=Lax, Secure configurable.
- CSP header with SRI on CDN assets.
- `MAX_CONTENT_LENGTH` enforced with 413 handler.
- Password policy: 8-256 chars. Reset requires current password.
- `Config.summary()` redacts all secrets.

## 8. Observability

- Python logging with module-level loggers.
- `/health` returns `{app, db, chroma, ollama_cloud, voice, stt}`.
- Structured log lines for ingestion (incl. pipeline fingerprint), chat
  (top_k/per-source/cited counts), audio, video, voice with durations.
  No secrets or transcript content logged.

## 9. Configuration

See `.env.example` for all environment variables. Key groups: Flask, database,
Ollama Cloud (chat + vision), HuggingFace embeddings, vector store, sources,
RAG budgets, overview duration/context bounds, audio, voice, diagrams, web
search, admin seed, CI flags.

Notable controls:
- `OLLAMA_NUM_CTX` (default 262144) -- the window all retrieval math derives
  from; must match the deployed chat model.
- `RAG_MAX_TOP_K` (default 200) / `RAG_MAX_CONTEXT_CHARS` (default 700000) --
  real ceilings enforced on every chat prompt.
- `RAG_MAX_CITATIONS` (default 12) -- visible citation badges per answer.
- `RAG_DIGEST_MAX_CHARS` (default 80000), `RAG_MAP_SECTION_CHARS` (6000),
  `RAG_MAP_MAX_SECTIONS` (80) -- full-coverage digest sizing.
- `OLLAMA_VISION_MODEL` (default `glm-5.3-flash:cloud`) + smart-gate
  thresholds + `OCR_FIGURE_DESCRIPTION`.
- `DIAGRAM_TO_MERMAID` / `DIAGRAM_MIN_CONFIDENCE` / `DIAGRAM_VERIFY`.
- `WEB_SEARCH_*` (opt-in, default off).
- `OVERVIEW_MIN_DURATION_SECONDS` / `OVERVIEW_MAX_DURATION_SECONDS` -- target
  spoken-duration bounds for both Audio and Video Overview generation.
- `OVERVIEW_MAX_CONTEXT_CHARS` (default 150000) -- character budget for raw
  source texts fed to the LLM when generating summaries, audio, and video
  overviews. Sources are included in upload order until the budget is
  reached; the cached section digest covers whatever exceeds it.
