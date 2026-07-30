# Architecture Document -- notebook-project

**Version:** 0.1

This document defines the module layout, data model, sequence flows, and tech
choices for the implemented codebase.

---

## 1. Design decisions

| # | Question | Resolution |
|---|----------|------------|
| 1 | Chat delivery | SSE streaming. A Flask generator yields `data: {token}\n\n` frames; a final frame carries `{sources, latency_ms, done: true}`. A non-streaming `/chat/sync` endpoint is also exposed for tests. |
| 2 | Two-host voices | MP3, `en-US-AvaNeural` (Ava) + `en-US-AndrewNeural` (Andrew). Per-utterance audio synthesized via edge-TTS and concatenated via pydub. |
| 3 | Audio format | MP3 (one file per notebook version). |
| 4 | In-app model | Single model `gemma4:31b-cloud` via Ollama Cloud. Hidden from the user. Thinking enabled server-side via the `<|think|>` system-prompt token when `ENABLE_THINKING=true`. |
| 5 | Admin seeding | `init_db.sql` seeds a fallback admin (`admin` / `change-me`). Real deployments set `ADMIN_USERNAME` / `ADMIN_PASSWORD` and run `flask seed-admin` (idempotent). Disabled users cannot authenticate and existing sessions are invalidated. |
| 6 | Summary trigger | Auto on every notebook change. Idempotent via `content_signature`. User can force-regenerate. |
| 7 | Source panel | List + modal detail + inline actions (view text, rename, delete). |
| 8 | Embedding/OCR provider | Both support `local` (default) and `hf_inference` (opt-in). |
| 9 | Vector store backend | `CHROMA_DB=local` (default) or `CHROMA_DB=cloud` with graceful fallback. |
| 10 | Voice conversation | Push-to-talk via HTTP `/voice/turn` endpoint: record audio, transcribe with faster-whisper, answer via `ChatService.chat_sync`, synthesize reply via edge-TTS. A `/voice` SocketIO namespace provides real-time status notifications. The spoken reply has markdown and citation brackets stripped for natural narration. Disabled by default (`VOICE_ENABLED=false`). |

## 2. Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.13 |
| Web framework | Flask 3.1 + Flask-Login + Flask-Migrate + Flask-SQLAlchemy + Flask-WTF + Flask-SocketIO |
| DB | PostgreSQL via SQLAlchemy + Alembic |
| Vector store | ChromaDB (local PersistentClient or CloudClient with fallback) |
| Embeddings | Qwen3-Embedding-0.6B (local sentence-transformers or HF Inference API) |
| OCR | GLM-OCR (local transformers or HF Inference API) |
| Chat LLM | Ollama Cloud, `gemma4:31b-cloud` |
| STT | faster-whisper (local, mock in test mode) |
| TTS | edge-TTS (neural voices) |
| Audio concat | pydub + audioop-lts (Python 3.13 compatibility) |
| Video | ffmpeg subprocess (slide images + TTS narration -> MP4) |
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
|   `-- versions/
|-- src/
|   |-- app.py                  # create_app() factory + SocketIO init
|   |-- config.py               # Config (env-driven)
|   |-- extensions.py           # db, login_manager, migrate, socketio
|   |-- models.py               # User, Notebook, Source, ChatMessage, ContentRegistry
|   |-- repositories/           # user_repo, notebook_repo, source_repo, chat_repo, content_registry_repo
|   |-- services/
|   |   |-- exceptions.py       # typed hierarchy
|   |   |-- embeddings.py       # Qwen3-Embedding (local or HF Inference API)
|   |   |-- vector_store.py     # ChromaDB (local or Cloud with fallback)
|   |   |-- document_parser.py  # PDF/DOCX/PPTX/TXT/MD extraction + magic bytes
|   |   |-- ocr_service.py      # GLM-OCR (local or HF Inference API)
|   |   |-- chunker.py          # RecursiveCharacterTextSplitter
|   |   |-- ingestion.py        # parse -> OCR fallback -> chunk -> embed -> store
|   |   |-- rag_retriever.py    # multi-collection retrieve + merge + recovery
|   |   |-- guardrails.py       # scope + groundedness checks
|   |   |-- ollama_client.py    # Ollama Cloud chat (sync + stream)
|   |   |-- chat_service.py     # scope -> retrieve -> prompt -> stream -> persist
|   |   |-- summary_service.py  # summary + suggested questions
|   |   |-- audio_scripter.py   # two-host dialogue JSON
|   |   |-- audio_service.py    # edge-TTS synth + concat MP3
|   |   |-- video_scripter.py   # narrated slide script
|   |   |-- video_service.py    # ffmpeg slides + TTS -> MP4
|   |   |-- tts_utils.py        # shared speaker_to_voice + synthesize + clean_for_tts
|   |   |-- stt_service.py      # faster-whisper STT (mock + real backends)
|   |   |-- voice_service.py    # STT -> chat -> TTS voice turn pipeline
|   |   |-- cleanup_service.py  # reference-counted content cleanup on delete
|   |   `-- jobs.py             # background workers (audio, video, summary)
|   |-- realtime/
|   |   `-- __init__.py         # /voice SocketIO namespace (status notifications)
|   |-- routes/
|   |   |-- __init__.py          # blueprint registration + error handlers + CSP
|   |   |-- _helpers.py          # require_owner, require_admin
|   |   |-- auth.py              # signup, login, logout, settings, reset-password
|   |   |-- admin.py             # admin user management
|   |   |-- notebooks.py         # notebook CRUD
|   |   |-- sources.py           # upload, list, delete, text, rename
|   |   |-- chat.py              # SSE streaming + sync + clear + history
|   |   |-- summary.py           # get + regenerate
|   |   |-- audio.py             # request, status, file, delete
|   |   |-- video.py             # request, status, file, delete
|   |   |-- voice.py             # voice turn (HTTP POST) + reply file serving
|   |   `-- index.py             # redirect
|   |-- static/
|   |   |-- css/app.css
|   |   |-- js/app.js            # upload, chat, audio, video, source actions
|   |   |-- js/voice.js          # push-to-talk recording + voice turn
|   |   `-- js/settings.js
|   `-- templates/
|       |-- base.html, notebook.html, settings.html, error.html
|       |-- auth/, notebooks/, admin/, _partials/
|-- tests/
|   |-- conftest.py
|   |-- fixtures/
|   |-- unit/
|   |-- routes/
|   `-- integration/
`-- data/                        # gitignored: chroma_db/, audio/, video/, voice/
```

## 4. Data model (SQLAlchemy)

```
User
  id, username (UNIQUE), password_hash, role (user|admin|disabled),
  avatar, audio_speaker_a, audio_speaker_b, video_speaker, voice_speaker,
  created_at

Notebook
  id, user_id (FK CASCADE), name (1..120), description,
  summary, suggested_questions (JSON), content_signature,
  audio_path, audio_status, audio_error,
  video_path, video_status, video_error,
  created_at, updated_at

Source
  id, notebook_id (FK CASCADE), filename, content_hash (sha256),
  content_type, char_count, page_count, status, error_message, created_at
  UNIQUE(notebook_id, content_hash), INDEX(content_hash)

ChatMessage
  id, notebook_id (FK CASCADE), role, content,
  sources_json, metadata_json, latency_ms, created_at

ContentRegistry (global, not user-scoped)
  content_hash (PK), chroma_collection, extracted_text, char_count, created_at
```

## 5. Key sequence flows

### 5.1 Source upload + ingestion

```
POST /notebooks/<id>/sources -> routes/sources.py
  validate owner, magic bytes, size cap
  compute sha256; check duplicate
  create Source row (status=queued)
  ingest: parse -> (OCR fallback?) -> chunk -> embed -> store + registry
  update status to ready/partial/failed
```

Idempotent: if `ContentRegistry.content_hash` exists, embedding is skipped.

### 5.2 Chat (SSE)

```
POST /notebooks/<id>/chat -> routes/chat.py
  validate owner
  guardrails.is_in_scope() -> refuse if off-topic
  rag_retriever.retrieve_with_sources() -> top K chunks with provenance
  build prompt (system + context + question + <|think|>)
  ollama_client.stream() -> yield tokens as SSE frames
  guardrails.check_groundedness() -> maybe append disclaimer
  persist user + assistant ChatMessages
  yield final frame {sources, latency_ms, done}
```

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

### 5.5 Voice conversation

```
POST /notebooks/<id>/voice/turn -> routes/voice.py
  validate owner, save audio to temp file
  VoiceService.run_voice_turn():
    1. STTService.transcribe() -> faster-whisper (or mock)
    2. if empty transcript -> return error
    3. ChatService.chat_sync() -> RAG + LLM + persist (same as text chat)
    4. clean_for_tts(answer) -> strip markdown + citations
    5. synthesize_utterance() -> edge-TTS -> MP3 reply
    6. return {transcript, answer, sources, reply_audio_url}
  serve reply via GET /voice/reply/<filename>
```

SocketIO `/voice` namespace provides real-time status notifications
(connected, ready, transcribing, thinking, speaking, done). Audio is sent via
the HTTP endpoint, not via SocketIO binary events.

### 5.6 Content cleanup on delete

```
DELETE source -> routes/sources.py
  delete Source row
  cleanup_orphaned_content(hash, exclude_source_id):
    if no other Source references this hash:
      delete ChromaDB collection + ContentRegistry entry

DELETE notebook -> routes/notebooks.py
  snapshot source hashes
  delete Notebook (cascades to sources + chat)
  cleanup_orphaned_content() per hash
  cleanup_notebook_media() -> delete audio/video/voice files
```

## 6. Mocking strategy (tests)

- `AI_MOCK=true` -> deterministic stubs for LLM, embeddings, OCR, TTS, STT.
  No network calls, no model downloads.
- `CI=true` -> ChromaDB EphemeralClient (in-memory).
- Tests use a temp file-based SQLite DB (not `:memory:`) so background threads
  and multiple app contexts share one database.
- Integration tests gated behind `RUN_INTEGRATION=1`.

## 7. Security specifics

- Magic-bytes file type validation.
- Filename sanitization (basename only).
- Owner-scoped routes return 404 (not 403).
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
- Structured log lines for ingestion, chat, audio, video, voice with
  durations. No secrets or transcript content logged.

## 9. Configuration

See `.env.example` for all environment variables. Key groups: Flask, database,
Ollama Cloud, HuggingFace embeddings/OCR, vector store, sources, audio, voice,
admin seed, CI flags.