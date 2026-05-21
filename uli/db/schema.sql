-- ─────────────────────────────────────────────────────────────────────────────
-- ULI — Universal Legal Intelligence
-- Phase 2B: PostgreSQL DDL — All tables with pgvector, composite indexes,
--           partial indexes, and immutable audit_log enforcement.
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector for 1536-dim embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram for text search

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: acts
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS acts (
    act_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name              TEXT NOT NULL CHECK (char_length(name) BETWEEN 3 AND 512),
    short_title       TEXT NOT NULL,
    year_enacted      SMALLINT NOT NULL CHECK (year_enacted BETWEEN 1860 AND 2026),
    status            TEXT NOT NULL DEFAULT 'in_force'
                          CHECK (status IN ('in_force','repealed','amended','suspended')),
    branch            CHAR(1) NOT NULL CHECK (branch IN ('A','B','C')),
    sub_branch        TEXT NOT NULL,
    gazette_number    TEXT,
    landmark_flag     BOOLEAN NOT NULL DEFAULT FALSE,
    overruled_by      TEXT,                        -- citation_key of overruling judgment
    citation_key      TEXT NOT NULL UNIQUE,
    citation_hash     TEXT NOT NULL UNIQUE,        -- SHA-256 of citation_key
    recency_weight    NUMERIC(5,4) NOT NULL DEFAULT 0.75
                          CHECK (recency_weight BETWEEN 0.0 AND 1.0),
    amendment_history JSONB NOT NULL DEFAULT '[]',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Composite index: branch taxonomy + status + year (primary filter pattern)
CREATE INDEX IF NOT EXISTS idx_acts_taxonomy
    ON acts (branch, sub_branch, status, year_enacted DESC);

-- Partial index: active landmark acts (highest-priority retrieval lane)
CREATE INDEX IF NOT EXISTS idx_acts_landmark_active
    ON acts (year_enacted DESC, citation_hash)
    WHERE status = 'in_force' AND landmark_flag = TRUE;

-- GIN index for JSONB amendment history queries
CREATE INDEX IF NOT EXISTS idx_acts_amendment_history
    ON acts USING GIN (amendment_history);

-- Trigger: auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER acts_updated_at
    BEFORE UPDATE ON acts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: sections
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sections (
    section_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    act_id       UUID NOT NULL REFERENCES acts(act_id) ON DELETE CASCADE,
    section_num  TEXT NOT NULL,
    title        TEXT,
    text         TEXT NOT NULL,
    sub_branch   TEXT NOT NULL,
    is_repealed  BOOLEAN NOT NULL DEFAULT FALSE,
    -- pgvector column: 1536 dims for text-embedding-3-small / ada-002
    embedding    VECTOR(1536),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for ANN similarity search (cosine metric)
CREATE INDEX IF NOT EXISTS idx_sections_embedding_hnsw
    ON sections USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- B-tree index for FK lookups
CREATE INDEX IF NOT EXISTS idx_sections_act_id ON sections (act_id);

-- Partial index: only active (non-repealed) sections in retrieval path
CREATE INDEX IF NOT EXISTS idx_sections_active
    ON sections (act_id, section_num)
    WHERE is_repealed = FALSE;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: citations
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS citations (
    citation_hash TEXT PRIMARY KEY,
    citation_key  TEXT NOT NULL,
    score         NUMERIC(5,4) NOT NULL CHECK (score BETWEEN 0.0 AND 1.0),
    status        TEXT NOT NULL
                      CHECK (status IN ('in_force','repealed','amended','suspended')),
    last_verified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    overruled_by  TEXT,
    ttl_seconds   INTEGER NOT NULL DEFAULT 3600 CHECK (ttl_seconds > 0),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citations_score ON citations (score DESC);
CREATE INDEX IF NOT EXISTS idx_citations_verified ON citations (last_verified DESC);

-- Partial index: citations that still pass the 0.98 threshold
CREATE INDEX IF NOT EXISTS idx_citations_passing
    ON citations (citation_hash, score)
    WHERE score >= 0.98 AND status = 'in_force';


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: audit_log  (APPEND-ONLY — no UPDATE/DELETE for app role)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    citation_hash     TEXT NOT NULL,
    score             NUMERIC(5,4) NOT NULL,
    agent_id          TEXT NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    phase_token_spend JSONB NOT NULL DEFAULT '{}'
);

-- Immutability: revoke UPDATE and DELETE from the application role
-- Run as superuser during schema setup:
-- REVOKE UPDATE, DELETE ON audit_log FROM uli_app;
-- GRANT INSERT, SELECT ON audit_log TO uli_app;

-- Index for audit queries by citation and time
CREATE INDEX IF NOT EXISTS idx_audit_citation ON audit_log (citation_hash, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_agent    ON audit_log (agent_id, timestamp DESC);

-- Partitioning hint (for high-volume deployments): partition by RANGE(timestamp)
-- ALTER TABLE audit_log PARTITION BY RANGE (timestamp);


-- ─────────────────────────────────────────────────────────────────────────────
-- ROLES & PRIVILEGES
-- ─────────────────────────────────────────────────────────────────────────────

-- Application role (non-superuser)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'uli_app') THEN
        CREATE ROLE uli_app LOGIN PASSWORD 'CHANGE_IN_PRODUCTION';
    END IF;
END
$$;

GRANT SELECT, INSERT, UPDATE, DELETE ON acts, sections, citations TO uli_app;
GRANT SELECT, INSERT ON audit_log TO uli_app;  -- No UPDATE / DELETE
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO uli_app;
