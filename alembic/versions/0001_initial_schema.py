"""
ULI — Initial schema migration
Revision: 0001
Creates all tables: acts, sections, citations, audit_log
Enables pgvector and pg_trgm extensions.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ── acts ──────────────────────────────────────────────────────────────────
    op.create_table(
        "acts",
        sa.Column("act_id",            UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name",              sa.Text,   nullable=False),
        sa.Column("short_title",       sa.Text,   nullable=False),
        sa.Column("year_enacted",      sa.SmallInteger, nullable=False),
        sa.Column("status",            sa.Text,   nullable=False, server_default="in_force"),
        sa.Column("branch",            sa.CHAR(1), nullable=False),
        sa.Column("sub_branch",        sa.Text,   nullable=False),
        sa.Column("gazette_number",    sa.Text,   nullable=True),
        sa.Column("landmark_flag",     sa.Boolean, nullable=False, server_default="FALSE"),
        sa.Column("overruled_by",      sa.Text,   nullable=True),
        sa.Column("citation_key",      sa.Text,   nullable=False, unique=True),
        sa.Column("citation_hash",     sa.Text,   nullable=False, unique=True),
        sa.Column("recency_weight",    sa.Numeric(5, 4), nullable=False, server_default="0.75"),
        sa.Column("amendment_history", JSONB,     nullable=False, server_default="'[]'"),
        sa.Column("created_at",        sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at",        sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("char_length(name) BETWEEN 3 AND 512", name="ck_acts_name_len"),
        sa.CheckConstraint("year_enacted BETWEEN 1860 AND 2026",  name="ck_acts_year"),
        sa.CheckConstraint(
            "status IN ('in_force','repealed','amended','suspended')", name="ck_acts_status"
        ),
        sa.CheckConstraint("branch IN ('A','B','C')", name="ck_acts_branch"),
        sa.CheckConstraint("recency_weight BETWEEN 0.0 AND 1.0",  name="ck_acts_recency"),
    )
    op.create_index("idx_acts_taxonomy",
                    "acts", ["branch", "sub_branch", "status", "year_enacted"])
    op.execute(
        """
        CREATE INDEX idx_acts_landmark_active ON acts (year_enacted DESC, citation_hash)
        WHERE status = 'in_force' AND landmark_flag = TRUE
        """
    )
    op.execute(
        "CREATE OR REPLACE FUNCTION update_updated_at() "
        "RETURNS TRIGGER LANGUAGE plpgsql AS $$ "
        "BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$"
    )
    op.execute(
        "CREATE TRIGGER acts_updated_at BEFORE UPDATE ON acts "
        "FOR EACH ROW EXECUTE FUNCTION update_updated_at()"
    )

    # ── sections ──────────────────────────────────────────────────────────────
    op.create_table(
        "sections",
        sa.Column("section_id",  UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("act_id",      UUID(as_uuid=True), nullable=False),
        sa.Column("section_num", sa.Text, nullable=False),
        sa.Column("title",       sa.Text, nullable=True),
        sa.Column("text",        sa.Text, nullable=False),
        sa.Column("sub_branch",  sa.Text, nullable=False),
        sa.Column("is_repealed", sa.Boolean, nullable=False, server_default="FALSE"),
        sa.Column("created_at",  sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["act_id"], ["acts.act_id"], ondelete="CASCADE"),
    )
    # pgvector column: added via raw SQL (Alembic doesn't natively know VECTOR type)
    op.execute("ALTER TABLE sections ADD COLUMN embedding VECTOR(1536)")
    op.execute(
        "CREATE INDEX idx_sections_embedding_hnsw ON sections "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.create_index("idx_sections_act_id", "sections", ["act_id"])
    op.execute(
        "CREATE INDEX idx_sections_active ON sections (act_id, section_num) "
        "WHERE is_repealed = FALSE"
    )

    # ── citations ─────────────────────────────────────────────────────────────
    op.create_table(
        "citations",
        sa.Column("citation_hash", sa.Text,  primary_key=True),
        sa.Column("citation_key",  sa.Text,  nullable=False),
        sa.Column("score",         sa.Numeric(5, 4), nullable=False),
        sa.Column("status",        sa.Text,  nullable=False),
        sa.Column("last_verified", sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("overruled_by",  sa.Text,  nullable=True),
        sa.Column("ttl_seconds",   sa.Integer, nullable=False, server_default="3600"),
        sa.Column("created_at",    sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("score BETWEEN 0.0 AND 1.0",         name="ck_cit_score"),
        sa.CheckConstraint("ttl_seconds > 0",                   name="ck_cit_ttl"),
        sa.CheckConstraint(
            "status IN ('in_force','repealed','amended','suspended')", name="ck_cit_status"
        ),
    )
    op.create_index("idx_citations_score",    "citations", ["score"])
    op.create_index("idx_citations_verified", "citations", ["last_verified"])
    op.execute(
        "CREATE INDEX idx_citations_passing ON citations (citation_hash, score) "
        "WHERE score >= 0.98 AND status = 'in_force'"
    )

    # ── audit_log (append-only) ───────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id",                UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("citation_hash",     sa.Text,  nullable=False),
        sa.Column("score",             sa.Numeric(5, 4), nullable=False),
        sa.Column("agent_id",          sa.Text,  nullable=False),
        sa.Column("timestamp",         sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("phase_token_spend", JSONB, nullable=False, server_default="'{}'"),
    )
    op.create_index("idx_audit_citation", "audit_log", ["citation_hash", "timestamp"])
    op.create_index("idx_audit_agent",    "audit_log", ["agent_id", "timestamp"])

    # Revoke destructive privileges from app role
    op.execute("REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS acts_updated_at ON acts")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at()")
    op.drop_table("audit_log")
    op.drop_table("citations")
    op.drop_table("sections")
    op.drop_table("acts")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS \"uuid-ossp\"")
