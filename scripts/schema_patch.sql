-- schema_patch.sql
-- 對現有 volume 補套 init.sql 中的 schema 變更。
-- 所有語句都是冪等的（IF NOT EXISTS），可安全重複執行。
--
-- 使用方式（每次修改 init.sql 後對現有 DB 執行一次）：
--   docker compose exec postgres psql -U mlops -d mlops -f /dev/stdin < scripts/schema_patch.sql
--
-- 注意：CREATE DATABASE airflow 不在此 script（只能在 init.sql 初始化時跑一次）。

CREATE EXTENSION IF NOT EXISTS vector;

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

-- ── Vector store ──────────────────────────────────────────────────────────────

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

CREATE TABLE IF NOT EXISTS recommendation_feedback_log (
    id          SERIAL    PRIMARY KEY,
    user_id     INTEGER   NOT NULL,
    item_id     INTEGER   NOT NULL,
    timestamp   TIMESTAMP NOT NULL,
    hit         BOOLEAN   NOT NULL
);

-- ── Sequences ────────────────────────────────────────────────────────────────

CREATE SEQUENCE IF NOT EXISTS user_id_seq;
CREATE SEQUENCE IF NOT EXISTS category_id_seq;
CREATE SEQUENCE IF NOT EXISTS brand_id_seq;
CREATE SEQUENCE IF NOT EXISTS item_id_seq;

-- ── Training snapshot ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS training_snapshot (
    id                  SERIAL  PRIMARY KEY,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    max_item_id         INTEGER NOT NULL,
    max_interaction_id  INTEGER NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_interaction_user_id    ON interaction(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_timestamp  ON interaction(timestamp);
CREATE INDEX IF NOT EXISTS idx_rec_log_user_id        ON recommendation_log(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_user_id   ON recommendation_feedback_log(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_timestamp ON recommendation_feedback_log(timestamp);
