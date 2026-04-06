-- PostgreSQL initialization script
-- Runs automatically when the postgres container starts for the first time.
-- pgvector/pgvector:pg16 image includes the pgvector extension.

CREATE EXTENSION IF NOT EXISTS vector;

-- Airflow 需要獨立的 database
CREATE DATABASE airflow;

-- ── Core tables ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS "user" (
    user_id     INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS brand (
    brand_id    INTEGER PRIMARY KEY,
    brand_name  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS category (
    category_id INTEGER PRIMARY KEY,
    category    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS item (
    item_id         INTEGER PRIMARY KEY,
    category_id1    INTEGER REFERENCES category(category_id),
    category_id2    INTEGER REFERENCES category(category_id),
    brand_id        INTEGER REFERENCES brand(brand_id),
    price           NUMERIC(10, 2)
);

CREATE TABLE IF NOT EXISTS interaction (
    interaction_id  SERIAL  PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES "user"(user_id),
    item_id         INTEGER NOT NULL REFERENCES item(item_id),
    timestamp       TIMESTAMP NOT NULL
);

-- ── Model versioning ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_version (
    id              SERIAL  PRIMARY KEY,
    model_version   TEXT    NOT NULL UNIQUE,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    note            TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ── Vector store ─────────────────────────────────────────────────────────────
-- embedding dimension = hidden_size * 3 = 64 * 3 = 192
-- (item_emb + category_emb + brand_emb concatenated)

CREATE TABLE IF NOT EXISTS item_embedding (
    item_id         INTEGER NOT NULL REFERENCES item(item_id),
    model_version   TEXT    NOT NULL REFERENCES model_version(model_version),
    embedding       vector(192) NOT NULL,
    PRIMARY KEY (item_id, model_version)
);

CREATE INDEX IF NOT EXISTS item_embedding_hnsw_idx
    ON item_embedding USING hnsw (embedding vector_cosine_ops);

-- ── Serving tables ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS recommendation_log (
    id                  SERIAL    PRIMARY KEY,
    user_id             INTEGER   NOT NULL,
    recommended_items   INTEGER[] NOT NULL,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ── User representation ───────────────────────────────────────────────────────
-- 每個 user 在特定 model version 下的 sequence representation（transformer 最後一位輸出）

CREATE TABLE IF NOT EXISTS user_representation (
    user_id         INTEGER NOT NULL REFERENCES "user"(user_id),
    model_version   TEXT    NOT NULL REFERENCES model_version(model_version),
    representation  vector(192) NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, model_version)
);

CREATE INDEX IF NOT EXISTS user_representation_hnsw_idx
    ON user_representation USING hnsw (representation vector_cosine_ops);

-- ── Feedback log ──────────────────────────────────────────────────────────────
-- /feedback 端點觸發時記錄，用於計算推薦命中率（Grafana dashboard）

CREATE TABLE IF NOT EXISTS recommendation_feedback_log (
    id          SERIAL    PRIMARY KEY,
    user_id     INTEGER   NOT NULL,
    item_id     INTEGER   NOT NULL,
    timestamp   TIMESTAMP NOT NULL,
    hit         BOOLEAN   NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_interaction_user_id  ON interaction(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_timestamp ON interaction(timestamp);
CREATE INDEX IF NOT EXISTS idx_rec_log_user_id       ON recommendation_log(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_user_id  ON recommendation_feedback_log(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_timestamp ON recommendation_feedback_log(timestamp);
