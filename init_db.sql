-- =============================================================================
-- notebook-project — database schema initialization
--
-- Run AFTER creating the database + user (see README.md step 4b):
--   psql -U postgres -d notebook_project -f init_db.sql
--
-- This script is idempotent: it drops existing tables before recreating them.
-- It creates all tables, indexes, foreign keys, stamps the Alembic version,
-- and seeds one admin account.
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- Drop existing tables (idempotent re-run)
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS sources CASCADE;
DROP TABLE IF EXISTS notebooks CASCADE;
DROP TABLE IF EXISTS content_registry CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------
CREATE TABLE users (
    id            SERIAL          PRIMARY KEY,
    username      VARCHAR(120)    UNIQUE NOT NULL,
    password_hash TEXT            NOT NULL,
    role          VARCHAR(20)     NOT NULL DEFAULT 'user',
    avatar        VARCHAR(32)     NOT NULL DEFAULT 'avatar-0.png',
    audio_speaker_a VARCHAR(32)   NOT NULL DEFAULT 'Ava',
    audio_speaker_b VARCHAR(32)   NOT NULL DEFAULT 'Andrew',
    video_speaker VARCHAR(32)     NOT NULL DEFAULT 'Ava',
    voice_speaker VARCHAR(32)     NOT NULL DEFAULT 'Ava',
    created_at    TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- notebooks
-- ---------------------------------------------------------------------------
CREATE TABLE notebooks (
    id                 SERIAL          PRIMARY KEY,
    user_id            INTEGER         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name               VARCHAR(120)    NOT NULL,
    description        TEXT,
    summary            TEXT,
    suggested_questions TEXT,
    content_signature  TEXT,
    audio_path         TEXT,
    audio_status       VARCHAR(20)     NOT NULL DEFAULT 'none',
    audio_error        TEXT,
    video_path         TEXT,
    video_status       VARCHAR(20)     NOT NULL DEFAULT 'none',
    video_error        TEXT,
    created_at         TIMESTAMP       NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMP       NOT NULL DEFAULT NOW(),
    CONSTRAINT notebook_name_nonempty CHECK (length(name) >= 1),
    CONSTRAINT notebook_name_length   CHECK (length(name) <= 120)
);

CREATE INDEX ix_notebooks_user_id ON notebooks(user_id);

-- ---------------------------------------------------------------------------
-- sources
-- ---------------------------------------------------------------------------
CREATE TABLE sources (
    id            SERIAL          PRIMARY KEY,
    notebook_id   INTEGER         NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    filename      VARCHAR(255)    NOT NULL,
    content_hash  VARCHAR(64)      NOT NULL,
    content_type  VARCHAR(10)      NOT NULL,
    char_count    INTEGER          NOT NULL DEFAULT 0,
    page_count    INTEGER,
    status        VARCHAR(20)      NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at    TIMESTAMP        NOT NULL DEFAULT NOW(),
    CONSTRAINT ix_sources_notebook_hash UNIQUE (notebook_id, content_hash)
);

CREATE INDEX ix_sources_notebook_id ON sources(notebook_id);
CREATE INDEX ix_sources_content_hash ON sources(content_hash);

-- ---------------------------------------------------------------------------
-- chat_messages
-- ---------------------------------------------------------------------------
CREATE TABLE chat_messages (
    id            SERIAL          PRIMARY KEY,
    notebook_id   INTEGER         NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    role          VARCHAR(10)     NOT NULL,
    content       TEXT            NOT NULL,
    sources_json  TEXT,
    metadata_json TEXT,
    latency_ms    INTEGER,
    created_at    TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_chat_notebook_created ON chat_messages(notebook_id, created_at);

-- ---------------------------------------------------------------------------
-- content_registry (global, not user-scoped — cross-user dedup)
-- ---------------------------------------------------------------------------
CREATE TABLE content_registry (
    content_hash      VARCHAR(64) PRIMARY KEY,
    chroma_collection  VARCHAR(80) NOT NULL,
    extracted_text     TEXT        NOT NULL,
    char_count         INTEGER     NOT NULL DEFAULT 0,
    created_at         TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Grant privileges to notebook_user (the app's DB user)
--
-- init_db.sql is run as `postgres` (the superuser), so tables are owned by
-- `postgres`. The app connects as `notebook_user`, which needs full DML
-- (SELECT/INSERT/UPDATE/DELETE) on all tables + USAGE on the sequences
-- backing SERIAL primary keys.
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO notebook_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO notebook_user;

-- Ensure future tables (e.g. created by Alembic) are also accessible.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO notebook_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO notebook_user;

-- ---------------------------------------------------------------------------
-- Seed admin account
--
-- IMPORTANT: The row below is a FALLBACK seed with a known placeholder password
-- ('change-me'). For any real deployment, set ADMIN_USERNAME / ADMIN_PASSWORD
-- in your .env and run the idempotent app command instead:
--
--     flask seed-admin
--
-- That command reads config values, creates the admin if missing, or refreshes
-- its password if it already exists. The placeholder hash below is only kept
-- so a fresh `psql -f init_db.sql` run produces a bootable admin you can sign
-- in with and immediately change. Do NOT rely on it in production.
--
-- To regenerate this hash for a different password:
--   python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your-password', method='scrypt'))"
-- ---------------------------------------------------------------------------
INSERT INTO users (username, password_hash, role)
VALUES (
    'admin',
    'scrypt:32768:8:1$R4y6jOZ3wHoPOal5$8cd3b3b209e236d26ada6d027ceaae55e242cc8453384297bb324be937778cd15f24cd16c780d0bdc2e4726e32671eda569e11342ae40c2c27fc5fcdca382457',
    'admin'
)
ON CONFLICT (username) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Stamp Alembic version so Flask-Migrate knows the DB is at the latest
-- migration (prevents auto-migration from re-running CREATE TABLE).
-- Replace the version id below with your latest migration revision.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

INSERT INTO alembic_version (version_num)
VALUES ('0002_voice_and_errors')
ON CONFLICT (version_num) DO NOTHING;

COMMIT;

-- =============================================================================
-- Verification (run manually to confirm):
--   psql -U notebook_user -d notebook_project -c "\dt"
--   psql -U notebook_user -d notebook_project -c "SELECT 1;"
-- =============================================================================
