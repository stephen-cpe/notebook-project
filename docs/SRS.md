# Software Requirements Specification -- notebook-project

**Version:** 0.3

---

## 1. Purpose

`notebook-project` is a self-hosted, Flask + PostgreSQL + ChromaDB RAG
application with source-grounded chat, audio/video overviews, and hands-free
voice mode. Upload source documents, ask questions grounded in those
sources with inline citations, generate spoken and narrated-video summaries,
and converse with your notebook by voice.

## 2. Scope

### 2.1 In scope

1. **User accounts** -- signup, login, logout, per-user notebooks. Admin role
   for user management (list/disable/enable). Disabled users cannot
   authenticate and existing sessions are invalidated.
2. **Notebooks** -- a named collection of uploaded source documents owned by a
   user.
3. **Source ingestion** -- upload PDF / DOCX / PPTX / TXT / MD / PNG / JPG /
   JPEG; extract text; fall back to vision understanding (`glm-5.3-flash`
   via Ollama Cloud) when text extraction is sparse, with a smart gate so
   text-layer PDFs skip vision entirely. Figure thumbnails are persisted for
   image sources and embedded Office images. Reference-counted cleanup:
   deleting a source removes its ChromaDB collection, content registry entry,
   and figure thumbnails only when no other notebook references the same
   content.
4. **RAG pipeline** -- Qwen3-Embedding-0.6B embeddings (local or HF Inference
   API), ChromaDB store (local persistent or Chroma Cloud), content-keyed
   collections with dedup and corruption recovery, window-budgeted retrieval
   (top-K derived from the 256K context window with per-source scaling and a
   hard character ceiling) with source provenance, plus cached full-coverage
   section digests so every part of every source is represented even when its
   chunks do not retrieve.
5. **Chat** -- ask questions grounded in the active notebook's sources; answers
   carry a collapsed "Sources (N of M)" toggle with the full grouped citation
   list and coverage inside. Served by the single
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
10. **Voice mode** -- toggle the mic to talk hands-free: utterances auto-send
    on pause, are transcribed via local faster-whisper, answered with the same
    RAG pipeline as text chat, and hear the spoken reply via edge-TTS with
    barge-in. Voice turns are persisted to the same chat history as text.
11. **Three-panel UI** -- sources (left), chat (center), config (right), built
    with Bootstrap 5 + custom dark theme. Header bell reports background-task
    status (running -> ready/failed).
12. **Answer difficulty** -- per-user reading level (Easy/Normal/Hard) applied
    to chat answers; set in Settings.
13. **Notebook PDF export** -- download summary, sources, and chat history as
    a PDF file.

### 2.2 Out of scope

- Study guides / briefing docs / glossaries.
- Quiz/lesson generation.
- Public sharing of notebooks.
- Multi-file drag-and-drop reordering, pinning, source-level notes.
- Mobile-native apps.
- Chat history pagination (all messages returned).

### 2.3 Non-goals

- No local Ollama for chat, vision, or embeddings. Chat and vision are Ollama
  Cloud only; embeddings are HuggingFace only (local sentence-transformers
  or HF Inference API).
- Vision understanding (`glm-5.3-flash` via Ollama Cloud) replaces the old
  local/HF GLM-OCR path entirely; the `OCR_PROVIDER` setting is a deprecated
  alias that no longer selects a backend.

## 3. Stakeholders

- **User (primary):** the developer building and using the application locally.
- **Future contributors:** anyone extending the project.

## 4. Glossary

| Term | Definition |
|------|------------|
| Notebook | A named, user-owned collection of source documents that grounds chat and generation. |
| Source | An uploaded file (PDF/DOCX/PPTX/TXT/MD/PNG/JPG/JPEG) added to a notebook. |
| Content Registry | A DB table mapping `file_hash -> (chroma_collection_name, extracted_text, section_digest)` for dedup, corruption recovery, and full-coverage digests. |
| Content-keyed collection | A ChromaDB collection named `doc_<sha256[:50]>_<fingerprint>` (63 chars) so identical content reuses one collection while embedding-backend changes produce a distinct collection instead of mixing vector spaces. |
| Section digest | A cached per-source summary built by summarizing every text section with the LLM (map-reduce), stored in the content registry. Guarantees every part of a source is represented even when its chunks do not retrieve. |
| Coverage ratio | Verbatim retrieved-context characters divided by total source characters (0..1); shown on the citation toggle. The section digest is the full-coverage layer behind it. |
| Groundedness | A heuristic check that the answer's substantive terms appear in retrieved context. |
| Audio Overview | A two-host spoken dialogue summarizing the notebook's sources, produced via edge-TTS. |
| Video Overview | A narrated slide presentation summarizing the notebook's sources, produced via ffmpeg + edge-TTS. |
| Voice mode | Toggle hands-free conversation: auto-send on pause, transcribe (faster-whisper), answer (RAG + LLM), speak reply (edge-TTS) with barge-in. |
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
- **FR-5** Per-user answer difficulty (Easy/Normal/Hard, default Normal) set
  in Settings and applied to chat answers.

### 5.2 Notebooks

- **FR-10** Create a notebook with name (1-120 chars) and optional description.
- **FR-11** List, open, rename, delete notebooks. Deletion removes sources,
  ChromaDB collections (if no other notebook references the same hash), chat
  history, audio/video files, and summary.
- **FR-12** At most N sources per notebook (default 50).
- **FR-13** Export a notebook (summary, sources, chat history) as a PDF
  download at `/notebooks/<id>/export` (fpdf2; latin-1 safe).

### 5.3 Source ingestion

- **FR-20** Upload PDF/DOCX/PPTX/TXT/MD/PNG/JPG/JPEG. Other types rejected
  (extension + magic-bytes validation).
- **FR-21** Max file size configurable (default 25 MB). Flask enforces
  `MAX_CONTENT_LENGTH` with a 413 error handler.
- **FR-22** SHA-256 hashing; content registry dedup.
- **FR-23** Text extraction per type: pypdf, python-docx, python-pptx, plain
  read. Image types (PNG/JPG/JPEG) have no text layer and always go through
  vision understanding.
- **FR-24** Vision fallback (`glm-5.3-flash` via Ollama Cloud) when text is
  below threshold. A smart gate skips vision entirely for text-layer PDFs;
  vision runs only for scanned/image-heavy PDFs, DOCX/PPTX embedded images,
  and image sources. Figure descriptions are produced during the vision pass
  when enabled. Vision failure does not block ingestion; source marked
  partial.
- **FR-24a** Figure persistence: image uploads are stored as figures; up to 2
  embedded images per DOCX/PPTX are stored with vision captions. Served
  owner-scoped at `/notebooks/<id>/figures/<hash>/<file>` with traversal
  guards; shown as thumbnails with chat citations.
- **FR-24b** Diagram-to-Mermaid: convertible figures are reinterpreted as
  validated Mermaid source (allowlisted types, confidence gate, optional
  vision re-verify); non-convertible figures keep the original image. Never
  raises; failures keep the image.
- **FR-25** Chunking, embedding, storage in content-keyed ChromaDB collection.
- **FR-26** Idempotent: re-uploading same content does not duplicate chunks.
- **FR-27** Ingestion status surfaced to UI.

### 5.4 Source management

- **FR-28** View extracted text in a modal.
- **FR-29** Rename source (inline, sanitized).
- **FR-29a** Delete source with confirmation + reference-counted cleanup.

### 5.5 RAG retrieval

- **FR-30** Multi-collection retrieve, merge by score. `top_k` is derived
  from the configured context window (`OLLAMA_NUM_CTX`, default 262144)
  capped by the visible `RAG_MAX_TOP_K` ceiling (default 200) -- no hidden
  constants.
- **FR-30a** Per-source depth scales so one giant document can fill its share
  of the window while many small sources each contribute their best chunks
  (floor 12 per collection).
- **FR-30b** Retrieved chunks are truncated to `RAG_MAX_CONTEXT_CHARS`
  (default 700000) minus reserved digest room, whole chunks only.
- **FR-31** Provenance: filename, page, chunk index, score.
- **FR-32** Corruption recovery from content registry.
- **FR-33** Mock mode for tests.
- **FR-34** Full-coverage section digest: every source text is split into
  sections (default 6000 chars, max 80; widened rather than truncated when
  over the cap), each section is summarized by the LLM, and the stitched
  digest is cached in the content registry keyed by content hash + pipeline
  fingerprint. Stale fingerprints rebuild; cold/missing digests degrade to
  "". Never raises.
- **FR-35** Digest wiring: chat and summary/audio/video prompts prepend the
  cached digests (bounded by `RAG_DIGEST_MAX_CHARS`, default 80000) on top of
  retrieved/budgeted raw texts. Digest builds run in background threads
  (summary/audio/video jobs); the chat request path only reads the cache and
  falls back to the cheap extractive digest when cold.

### 5.6 Chat

- **FR-40** Grounded answers with citations behind a collapsed toggle.
- **FR-41** Returns `{answer, sources, source_total, latency_ms,
  coverage_ratio}` where `sources` is the capped display list and
  `source_total` the full unique count.
- **FR-42** Chat history persisted and shown on notebook open. Clear button.
- **FR-43** Scope guardrail: refuse off-topic questions.
- **FR-44** Groundedness guardrail: append disclaimer if ungrounded.
- **FR-45** Streaming via SSE; non-streaming `/chat/sync` for tests. Final
  frames carry `{sources, source_total, latency_ms, coverage_ratio, done}`.
- **FR-46** Single model `gemma4:31b-cloud`; thinking via `<|think|>` when
  `ENABLE_THINKING=true`.
- **FR-47** Citation display: at most `RAG_MAX_CITATIONS` badges (default 12,
  top-N by relevance); the UI shows one collapsed "Sources (N of M) · X%
  read" toggle per answer with the grouped file/page list, coverage, and
  figure thumbnails inside. Chat history renders the persisted capped list.

### 5.7 Auto summary & suggested questions

- **FR-60** Auto-regenerate on notebook change (create/source add/remove).
  Idempotent via `content_signature`. The summary job best-effort builds
  full-coverage section digests first (FR-34), then prompts with digest +
  budgeted raw texts.
- **FR-61** Summary + 5 suggested questions displayed in chat panel.
  Questions are deduplicated (normalized-title + substring match).
- **FR-62** Failures do not block notebook use; retry button available.
- **FR-63** Opt-in web-augmented questions (`WEB_SEARCH_ENABLED`, default
  off, fail-closed): only for non-proprietary topics when internal questions
  run short; reuses Ollama Cloud credentials; verbatim URLs only.

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

### 5.10 Voice mode

- **FR-100** Toggle voice mode: click the mic button once to replace the text
  input with a voice panel (status orb, live transcript, End button); click
  the mic / End (or Esc) to return to text chat. The mic stays open while in
  voice mode -- no press-and-hold. Utterances auto-send after a short pause
  (client-side voice-activity detection) with a max-length fallback
  (`VOICE_MAX_RECORDING_SECONDS`). Audio is transcribed via faster-whisper
  (local, mock in test mode).
- **FR-101** The transcribed question is answered using the same RAG pipeline
  as text chat (guardrails, retrieval, LLM, groundedness, persistence).
  The turn response carries `{transcript, answer, sources, source_total,
  latency_ms, reply_audio_url}`.
- **FR-102** The answer is synthesized to speech via edge-TTS, played back
  automatically, and listening resumes when playback ends. Markdown and
  citation brackets are stripped from the spoken version for natural
  narration. Talking over the reply interrupts it (barge-in); talking while
  the answer is being prepared cancels the request and re-listens.
- **FR-103** Voice turns are persisted to the same chat history as text turns.
- **FR-104** Disabled by default (`VOICE_ENABLED=false`); the mic button is
  hidden when disabled.

### 5.11 UI / UX

- **FR-80** Three-panel layout: Sources (left), Chat (center), Config (right).
  Bootstrap 5 dark theme.
- **FR-82** Sources panel: status badges, upload, rename, delete, view text.
- **FR-83** Chat panel: history, streaming with typing indicator, collapsed
  citation toggle (FR-47), clear button, suggested questions, mic toggle
  button (when voice enabled) that swaps the text input for the voice panel.
- **FR-84** Config panel: audio/video controls, focus topic, metadata, export
  PDF button.
- **FR-85** Loading states for all long-running actions.
- **FR-86** Header tasks bell: polls `GET /tasks` for background-job status
  (running -> ready/failed) with an unread badge; clicking marks read.

### 5.12 Background tasks

- **FR-76** Media jobs (audio/video) carry a generation token so superseded
  results are discarded; launches are singleflight per notebook (duplicate
  launches while running are skipped). Crashed jobs are marked failed;
  interrupted (transient-status) jobs are recovered to failed on restart.
- **FR-77** In-memory task tracker feeds the header bell: `GET /tasks`
  returns recent jobs + unread count; `POST /tasks/read` marks read (single
  key or all). Entries expire after 2h; never raises.

## 6. Non-functional Requirements

### 6.1 Performance

- **NFR-1** Chat first-token latency <= 5 s p95 (network-dependent).
- **NFR-2** 10-page PDF ingestion <= 30 s (embedding-bound).
- **NFR-3** Retrieval <= 500 ms p95 for 10-source notebook.
- **NFR-4** Audio Overview <= 5 min for 5-source notebook.
- **NFR-5** Video Overview <= 10 min for 5-source notebook.

### 6.2 Reliability & graceful degradation

- **NFR-10** Ollama Cloud failure: single retry before error. Vision calls
  use backoff with resume-pause instead of failing long jobs.
- **NFR-11** ChromaDB corruption: auto-recover from content registry.
- **NFR-12** Vision OCR failure: does not block ingestion.
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
  video_speaker, voice_speaker, difficulty, created_at)
- `Notebook` (id, user_id, name, description, summary, suggested_questions,
  content_signature, audio_path, audio_status, audio_error, audio_generation,
  video_path, video_status, video_error, video_generation, created_at,
  updated_at)
- `Source` (id, notebook_id, filename, content_hash, content_type, char_count,
  page_count, status, error_message, created_at)
- `ChatMessage` (id, notebook_id, role, content, sources_json, metadata_json,
  latency_ms, created_at)
- `ContentRegistry` (content_hash PK, chroma_collection, embedding_fingerprint,
  extracted_text, char_count, section_digest, digest_pipeline, created_at)

## 8. External interfaces

- **Ollama Cloud** -- chat/reasoning via `gemma4:31b-cloud`; vision
  understanding (OCR, tables, figure description, diagram reinterpretation)
  via `glm-5.3-flash`.
- **HuggingFace** -- Qwen3-Embedding (local or HF Inference API). No model
  weights are downloaded for vision (cloud-served).
- **edge-TTS** -- neural voices for audio overview, video narration, and voice
  reply.
- **faster-whisper** -- local speech-to-text for voice conversation.
- **ffmpeg** -- video generation + audio decoding for STT normalization.
- **PostgreSQL** -- primary database via SQLAlchemy + Alembic.
- **ChromaDB** -- vector store (local or Cloud).

## 9. Constraints

- Python 3.13+.
- No local Ollama for chat, vision, or embeddings.
- ffmpeg required for Video Overview + STT audio normalization.
- Poppler required for PDF-to-image rendering for vision OCR. DOCX/PPTX
  OCR uses embedded images extracted directly from the Office ZIP archive (no
  Poppler dependency for those types).

## 10. Assumptions

- Valid Ollama Cloud API key and base URL.
- Python 3.13, PostgreSQL, Poppler, and ffmpeg available.
- HuggingFace embedding model download (~1.2 GB) on first run when using the
  local provider; cached afterward. Vision runs cloud-side (no local weights).
- faster-whisper model download on first voice turn when not in mock mode.

## 11. Acceptance criteria

- All functional requirements implemented with passing tests.
- A user can: sign up -> create a notebook -> upload sources -> see ingestion
  complete -> see summary + suggested questions -> chat with citations ->
  generate audio overview -> generate video overview -> use voice mode
  (toggle, auto-send, barge-in) -> log out and back in and see everything
  persisted.
- No secrets committed; `.env.example` is the only env file in VCS.
