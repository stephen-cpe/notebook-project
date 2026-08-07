# notebook-project

A self-hosted, Flask + PostgreSQL + ChromaDB RAG application with
source-grounded chat, audio/video overviews, and push-to-talk voice
conversation. Upload source documents, ask questions grounded in those sources
with inline citations, generate spoken and narrated-video summaries, and
converse with your notebook by voice.

See [`docs/SRS.md`](docs/SRS.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the full requirements and architecture.

## Features

- User accounts (signup/login), per-user notebooks, admin user management.
- Source ingestion: PDF / DOCX / PPTX / TXT / MD, with opt-in OCR fallback via
  HuggingFace GLM-OCR.
- RAG with HuggingFace Qwen3-Embedding-0.6B + ChromaDB (local persistent or
  Chroma Cloud, content-keyed collections, dedup, corruption recovery, source
  provenance).
- Chat grounded in notebook sources with inline citation tags, streamed via SSE.
- Single in-app chat model `gemma4:31b-cloud` via Ollama Cloud (hidden from user).
- Auto summary + 5 clickable suggested questions on every notebook change.
- Source management UI: view extracted text (modal), inline rename, delete
  with confirmation.
- Chat history with clear button.
- Two-host Audio Overview via edge-TTS with optional focus topic.
- Video Overview: narrated slide presentation rendered to MP4 via ffmpeg.
- Voice conversation: push-to-talk, transcribe via faster-whisper, RAG-grounded
  answer, spoken reply via edge-TTS. Voice turns persisted to chat history.
- Three-panel UI (Sources / Chat / Config) with Bootstrap 5 + dark theme.

## Prerequisites

1. **Python 3.13** from https://python.org
2. **PostgreSQL** from https://www.postgresql.org/download/windows/
3. **Poppler** (for OCR / PDF-to-image) from
   https://github.com/oschwartz10612/poppler-windows/releases
4. **ffmpeg** (for Video Overview and voice STT audio decoding) -- install and
   add to your PATH.
5. An **Ollama Cloud** API key + base URL (for chat).
6. (Optional) A **HuggingFace READ token** to suppress rate-limit warnings --
   https://huggingface.co/settings/tokens
7. (Optional) **Chroma Cloud** credentials if you want to use `CHROMA_DB=cloud`.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/stephen-cpe/notebook-project.git notebook-project
cd notebook-project
```

### 2. Create virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set up PostgreSQL database (Windows)

This project requires PostgreSQL. Follow these steps in **PowerShell**.

#### 4a. Open PowerShell and connect to PostgreSQL

```powershell
psql -U postgres
```

(If `psql` is not on your PATH, use the full path, e.g.
`& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres`.)

#### 4b. Create the database and user

Inside the `psql` prompt, run:

```sql
-- Only run these if you want to start from scratch
DROP DATABASE IF EXISTS notebook_project;
DROP USER IF EXISTS notebook_user;

CREATE USER notebook_user WITH PASSWORD 'notebook_pass';
CREATE DATABASE notebook_project;
ALTER DATABASE notebook_project OWNER TO notebook_user;
GRANT CREATE ON SCHEMA public TO notebook_user;
\q
```

#### 4c. Initialize the database schema + seed admin

```powershell
psql -U postgres -d notebook_project -f init_db.sql
```

This creates all tables, indexes, foreign keys, stamps the Alembic version, and
seeds one fallback admin account (`admin` / `change-me`).

For any real deployment, set `ADMIN_USERNAME` / `ADMIN_PASSWORD` in your `.env`
and run the idempotent app command to create or refresh the admin from config:

```powershell
flask seed-admin
```

The app warns on startup if `ADMIN_PASSWORD` is still the default `change-me`.

#### 4d. Verify the connection

```powershell
psql -U notebook_user -d notebook_project -c "SELECT 1;"
```

You should see `?column?` / `1`.

### 5. Install Poppler (Windows 11)

The OCR fallback requires Poppler to render PDF pages for GLM-OCR.

1. Download the latest Poppler for Windows from:
   https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the archive (e.g. `poppler-26.02.0`) to `C:\Program Files\poppler-26.02.0\`.
3. Add the `bin` directory to your system `PATH`:
   - Open **System Properties > Environment Variables**
   - Under **System variables**, edit `Path` and add:
     `C:\Program Files\poppler-26.02.0\Library\bin`
   - Alternatively, set `POPPLER_PATH=C:\Program Files\poppler-26.02.0\Library\bin`
     in your `.env` file.
4. Restart any open terminals for the change to take effect.

Verify Poppler is installed:

```powershell
pdftoppm -v
```

### 6. Create the `.env` file

```powershell
copy .env.example .env
```

Open `.env` and update at minimum:
- `SECRET_KEY` -> a random string
- `DATABASE_URL` -> match your PostgreSQL credentials (from step 4)
- `OLLAMA_CLOUD_BASE_URL` + `OLLAMA_CLOUD_API_KEY` -> your Ollama Cloud access
- `HF_TOKEN` (optional) -> your HuggingFace READ token to suppress rate-limit
  warnings on model download/inference.
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` -> desired admin seed credentials

Optional provider configuration:
- `EMBEDDING_PROVIDER=hf_inference` to use Hugging Face Inference API for
  embeddings instead of local sentence-transformers (requires `HF_TOKEN`).
- `OCR_PROVIDER=hf_inference` to use Hugging Face Inference API for OCR
  instead of local transformers (requires `HF_TOKEN`).
- `CHROMA_DB=cloud` to use Chroma Cloud instead of local storage (requires
  `CHROMA_CLOUD_API_KEY`, `CHROMA_CLOUD_CONNECTION_STRING`, and
  `CHROMA_COLLECTION_NAME`).

### 7. (Optional) Enable Voice Conversation

Voice conversation (push-to-talk) is disabled by default. The voice
dependencies (`flask-socketio`, `faster-whisper`) are already
listed in `requirements.txt`, so the standard install in step 3 covers them.
To enable the feature:

1. Set in `.env`:
   ```env
   VOICE_ENABLED=true
   ```
   Other `VOICE_*` variables tune the STT model, device, compute type, and
   recording/size limits (see `.env.example`). When `AI_MOCK=true` (tests/CI),
   STT and TTS use deterministic mocks -- no model download and no network.
   **ffmpeg must be on your PATH** (already required for the Video Overview /
   pydub audio).

The microphone button appears to the left of the chat Send button on the
notebook page. Press and hold to record (or Space/Enter when focused), release
to send. Voice turns are persisted to the same chat history as text turns.
When the SocketIO streaming layer is available it provides real-time status
notifications; the HTTP endpoint handles the audio upload and processing.

### 8. Run the application

```powershell
python app.py
```

Open http://localhost:5000 in your browser.

## Testing

```powershell
pytest -v tests/
```

Tests run fully offline:
- `AI_MOCK=true` returns deterministic stubs (no Ollama Cloud calls, no model
  downloads).
- `CI=true` forces in-memory ChromaDB.
- SQLAlchemy uses a temporary file-based SQLite database (per-test isolation);
  PostgreSQL-only validation is bypassed per-test.

CI enforces 80% coverage (`--cov-fail-under=80`); the current suite is at 89%.

Integration tests (real Ollama Cloud / real HuggingFace model loads) are
gated behind `RUN_INTEGRATION=1`:

```powershell
$env:RUN_INTEGRATION = "1"
pytest -v tests/ -m integration
```

## Code quality

```powershell
ruff check src tests
ruff format --check src tests
mypy src
```

Pre-commit hooks install with:

```powershell
pre-commit install
```

## Documentation

- [`docs/SRS.md`](docs/SRS.md) -- Software Requirements Specification
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- Architecture document

## Disclaimer

This project is a personal learning exercise. It is **not** production-ready
and has not undergone comprehensive testing.

- **Bugs are expected.** Edge cases in file parsing, OCR, streaming, and
  cloud API integration may not be handled gracefully.
- **Configuration may change.** Environment variables, defaults, and the
  overall setup process may change between commits without a changelog.
- **No warranty.** This software is provided "as is" without any warranty.
- **External dependencies incur costs.** Ollama Cloud, Hugging Face
  Inference API, and Chroma Cloud may charge for usage. Monitor your
  billing dashboards.

## License

MIT
