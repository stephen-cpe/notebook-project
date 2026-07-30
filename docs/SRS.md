# Software Requirements Specification -- notebook-project

**Version:** 0.1

---

## 1. Purpose

`notebook-project` is a self-hosted, Flask + PostgreSQL + ChromaDB RAG
application with source-grounded chat, audio/video overviews, and push-to-talk
voice conversation. Upload source documents, ask questions grounded in those
sources with inline citations, generate spoken and narrated-video summaries,
and converse with your notebook by voice.

## 2. Scope

### 2.1 In scope

1. **User accounts** -- signup, login, logout, per-user notebooks. Admin role
   for user management (list/disable/enable). Disabled users cannot
   authenticate and existing sessions are invalidated.
2. **Notebooks** -- a named collection of uploaded source documents owned by a
   user.
3. **Source ingestion** -- upload PDF / DOCX / PPTX / TXT / MD; extract text;
   fall back to OCR (GLM-OCR) when text extraction is sparse. Reference-counted
   cleanup: deleting a source removes its ChromaDB collection and content
   registry entry only when no other notebook references the same content.
4. **RAG pipeline** -- Qwen3-Embedding-0.6B embeddings (local or HF Inference
   API), ChromaDB store (local persistent or Chroma Cloud), content-keyed
   collections with dedup and corruption recovery, score-merged retrieval with
   source provenance.
5. **Chat** -- ask questions grounded in the active notebook's sources; answers
   carry inline citations as clickable source tags. Served by the single
   in-app model `gemma4:31b-cloud` via Ollama Cloud; the model is hidden from
   the user (no model selector in the UI). Streaming via SSE.
6. **Guardrails** -- scope validation (refuse off-topic questions) and
   groundedness check (flag answers not supported by retrieved context).
7. **Auto summary + suggested questions** -- generated when a notebook is
   created or sources are added. Suggested questions are clickable and send
   the question directly to chat.
8. **Audio Overview** -- a two-host dialogue script generated from the
   notebook's sources, rendered to a single MP3 via edge-TTS.
9. **Video Overview** -- a narrated slide presentation generated from the
   notebook's sources, rendered to MP4 via ffmpeg with TTS narration.
10. **Voice conversation** -- push-to-talk: record audio, transcribe via local
    faster-whisper, get a RAG-grounded answer, and hear the spoken reply via
    edge-TTS. Voice turns are persisted to the same chat history as text.
11. **Three-panel UI** -- sources (left), chat (center), config (right), built
    with Bootstrap 5 + custom dark theme.

### 2.2 Out of scope

- Study guides / briefing docs / glossaries.
- Quiz/lesson generation.
- Public sharing of notebooks.
- Multi-file drag-and-drop reordering, pinning, source-level notes.
- Mobile-native apps.
- Chat history pagination (all messages returned).

### 2.3 Non-goals

- No local Ollama for chat or embeddings. Chat is Ollama Cloud only; embeddings
  are HuggingFace only (local sentence-transformers or HF Inference API).
- GLM-OCR runs via HuggingFace transformers (local) or HF Inference API, not
  Ollama. Invoked only as an opt-in fallback.

## 3. Stakeholders

- **User (primary):** the developer building and using the application locally.
- **Future contributors:** anyone extending the project.

## 4. Glossary

| Term | Definition |
|------|------------|
| Notebook | A named, user-owned collection of source documents that grounds chat and generation. |
| Source | An uploaded file (PDF/DOCX/PPTX/TXT/MD) added to a notebook. |
| Content Registry | A DB table mapping `file_hash -> (chroma_collection_name, extracted_text)` for dedup and corruption recovery. |
| Content-keyed collection | A ChromaDB collection named `doc_<sha256[:59]>` so identical content reuses one collection. |
| Groundedness | A heuristic check that the answer's substantive terms appear in retrieved context. |
| Audio Overview | A two-host spoken dialogue summarizing the notebook's sources, produced via edge-TTS. |
| Video Overview | A narrated slide presentation summarizing the notebook's sources, produced via ffmpeg + edge-TTS. |
| Voice conversation | Push-to-talk: record audio, transcribe (faster-whisper), answer (RAG + LLM), speak reply (edge-TTS). |
| Thinking token | Gemma 4's `<|think|>` system-prompt token; server-side config flag, not a UI control. |

## 5. Functional Requirements

### 5.1 Authentication & accounts

- **FR-1** A visitor can sign up with username + password. Passwords are hashed
  (scrypt). Minimum 8 characters, maximum 256. Duplicate usernames rejected.
- **FR-2** A user can log in and log out. Sessions persist via Flask-Login.
  Disabled users (`role='disabled'`) cannot log in and existing sessions are
  invalidated on the next request.
- **FR-3** Notebooks, sources, chat history, audio, video, summaries are
  owner-scoped; cross-user access returns 404 (not 403).
- **FR-4** An admin role exists for user management. Seeded via `flask
  seed-admin` (idempotent, reads `ADMIN_USERNAME`/`ADMIN_PASSWORD` from
  config). Admin routes at `/admin`.

### 5.2 Notebooks

- **FR-10** Create a notebook with name (1-120 chars) and optional description.
- **FR-11** List, open, rename, delete notebooks. Deletion removes sources,
  ChromaDB collections (if no other notebook references the same hash), chat
  history, audio/video files, and summary.
- **FR-12** At most N sources per notebook (default 50).

### 5.3 Source ingestion

- **FR-20** Upload PDF/DOCX/PPTX/TXT/MD. Other types rejected.
- **FR-21** Max file size configurable (default 25 MB). Flask enforces
  `MAX_CONTENT_LENGTH` with a 413 error handler.
- **FR-22** SHA-256 hashing; content registry dedup.
- **FR-23** Text extraction per type: pypdf, python-docx, python-pptx, plain
  read.
- **FR-24** OCR fallback (GLM-OCR) when text is below threshold. OCR failure
  does not block ingestion; source marked partial.
- **FR-25** Chunking, embedding, storage in content-keyed ChromaDB collection.
- **FR-26** Idempotent: re-uploading same content does not duplicate chunks.
- **FR-27** Ingestion status surfaced to UI.

### 5.4 Source management

- **FR-28** View extracted text in a modal.
- **FR-29** Rename source (inline, sanitized).
- **FR-29a** Delete source with confirmation + reference-counted cleanup.

### 5.5 RAG retrieval

- **FR-30** Multi-collection retrieve, merge by score, top K (default 5).
- **FR-31** Provenance: filename, page, chunk index, score.
- **FR-32** Corruption recovery from content registry.
- **FR-33** Mock mode for tests.

### 5.6 Chat

- **FR-40** Grounded answers with inline citation tags.
- **FR-41** Returns `{answer, sources, latency_ms}`.
- **FR-42** Chat history persisted and shown on notebook open. Clear button.
- **FR-43** Scope guardrail: refuse off-topic questions.
- **FR-44** Groundedness guardrail: append disclaimer if ungrounded.
- **FR-45** Streaming via SSE; non-streaming `/chat/sync` for tests.
- **FR-46** Single model `gemma4:31b-cloud`; thinking via `<|think|>` when
  `ENABLE_THINKING=true`.

### 5.7 Auto summary & suggested questions

- **FR-60** Auto-regenerate on notebook change (create/source add/remove).
  Idempotent via `content_signature`.
- **FR-61** Summary + 5 suggested questions displayed in chat panel.
- **FR-62** Failures do not block notebook use; retry button available.

### 5.8 Audio Overview

- **FR-70** Two-host dialogue via edge-TTS (Ava + Andrew).
- **FR-71** Structured JSON dialogue with target duration bounds.
- **FR-72** Synthesized and concatenated to MP3.
- **FR-73** Progress shown in UI; re-generate and delete supported.
- **FR-74** Per-utterance failure isolation.
- **FR-75** Focus topic input steers discussion.

### 5.9 Video Overview

- **FR-90** Narrated slide presentation via ffmpeg + edge-TTS.
- **FR-91** Slide script with title, bullets, narration. Focus topic supported.
- **FR-92** MP4 stored on disk; progress shown in UI.
- **FR-93** Requires ffmpeg on PATH.
- **FR-94** Re-generate and delete supported.

### 5.10 Voice conversation

- **FR-100** Push-to-talk: press and hold the mic button to record, release to
  send. Audio is transcribed via faster-whisper (local, mock in test mode).
- **FR-101** The transcribed question is answered using the same RAG pipeline
  as text chat (guardrails, retrieval, LLM, groundedness, persistence).
- **FR-102** The answer is synthesized to speech via edge-TTS and played back.
  Markdown and citation brackets are stripped from the spoken version for
  natural narration.
- **FR-103** Voice turns are persisted to the same chat history as text turns.
- **FR-104** Disabled by default (`VOICE_ENABLED=false`); the mic button is
  hidden when disabled.

### 5.11 UI / UX

- **FR-80** Three-panel layout: Sources (left), Chat (center), Config (right).
  Bootstrap 5 dark theme.
- **FR-82** Sources panel: status badges, upload, rename, delete, view text.
- **FR-83** Chat panel: history, streaming with typing indicator, citation
  tags, clear button, suggested questions, mic button (when voice enabled).
- **FR-84** Config panel: audio/video controls, focus topic, metadata.
- **FR-85** Loading states for all long-running actions.

## 6. Non-functional Requirements

### 6.1 Performance

- **NFR-1** Chat first-token latency <= 5 s p95 (network-dependent).
- **NFR-2** 10-page PDF ingestion <= 30 s (embedding-bound).
- **NFR-3** Retrieval <= 500 ms p95 for 10-source notebook.
- **NFR-4** Audio Overview <= 5 min for 5-source notebook.
- **NFR-5** Video Overview <= 10 min for 5-source notebook.

### 6.2 Reliability & graceful degradation

- **NFR-10** Ollama Cloud failure: single retry before error.
- **NFR-11** ChromaDB corruption: auto-recover from content registry.
- **NFR-12** OCR failure: does not block ingestion.
- **NFR-13** Audio per-utterance failure isolated.
- **NFR-14** Chroma Cloud failure: fallback to local PersistentClient.

### 6.3 Security

- **NFR-20** No secrets in VCS. `.env` gitignored.
- **NFR-21** Passwords hashed (scrypt); never logged. Password policy: 8-256
  chars. Reset requires current password.
- **NFR-22** All data access owner-scoped.
- **NFR-23** Magic-bytes file type validation.
- **NFR-24** Path sanitization on upload + rename.
- **NFR-25** `SECRET_KEY` from env; refuses placeholder in production.
- **NFR-26** `HF_TOKEN` optional, never logged.
- **NFR-27** Session cookie hardening: HttpOnly, SameSite=Lax, Secure
  configurable.
- **NFR-28** CSP header with SRI on CDN assets.

### 6.4 Maintainability

- **NFR-30** Ruff lint + format.
- **NFR-31** pytest with coverage.
- **NFR-32** mypy strict.
- **NFR-33** Pre-commit hooks.
- **NFR-34** GitHub Actions CI.
- **NFR-36** Layered architecture: routes -> services -> repositories -> models.

### 6.5 Observability

- **NFR-50** Structured logging; no secrets logged.
- **NFR-51** `/health` reports app + DB + ChromaDB + Ollama Cloud + voice/STT
  status.

## 7. Data model (high-level)

- `User` (id, username, password_hash, role, avatar, audio_speaker_a/b,
  video_speaker, voice_speaker, created_at)
- `Notebook` (id, user_id, name, description, summary, suggested_questions,
  content_signature, audio_path, audio_status, audio_error, video_path,
  video_status, video_error, created_at, updated_at)
- `Source` (id, notebook_id, filename, content_hash, content_type, char_count,
  page_count, status, error_message, created_at)
- `ChatMessage` (id, notebook_id, role, content, sources_json, metadata_json,
  latency_ms, created_at)
- `ContentRegistry` (content_hash PK, chroma_collection, extracted_text,
  char_count, created_at)

## 8. External interfaces

- **Ollama Cloud** -- chat/reasoning via `gemma4:31b-cloud`.
- **HuggingFace** -- Qwen3-Embedding (local or HF Inference API), GLM-OCR
  (local or HF Inference API).
- **edge-TTS** -- neural voices for audio overview, video narration, and voice
  reply.
- **faster-whisper** -- local speech-to-text for voice conversation.
- **ffmpeg** -- video generation + audio decoding for STT normalization.
- **PostgreSQL** -- primary database via SQLAlchemy + Alembic.
- **ChromaDB** -- vector store (local or Cloud).

## 9. Constraints

- Python 3.13+.
- No local Ollama for chat or embeddings.
- ffmpeg required for Video Overview + STT audio normalization.
- Poppler required for OCR fallback (PDF-to-image).

## 10. Assumptions

- Valid Ollama Cloud API key and base URL.
- Python 3.13, PostgreSQL, Poppler, and ffmpeg available.
- HuggingFace model download (~1.2 GB embeddings, ~1.8 GB OCR) on first run
  when using local provider; cached afterward.
- faster-whisper model download on first voice turn when not in mock mode.

## 11. Acceptance criteria

- All FR-1 ... FR-104 implemented with passing tests.
- A user can: sign up -> create a notebook -> upload sources -> see ingestion
  complete -> see summary + suggested questions -> chat with citations ->
  generate audio overview -> generate video overview -> use push-to-talk voice
  conversation -> log out and back in and see everything persisted.
- No secrets committed; `.env.example` is the only env file in VCS.