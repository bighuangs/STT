"""TODO-08 block evidence, source locations and citation facts.

Revision ID: 20260723_0008
Revises: 20260722_0007
"""

from alembic import op
import sqlalchemy as sa


revision = "20260723_0008"
down_revision = "20260722_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evidence_records",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "schema_version",
            sa.String(32),
            nullable=False,
            server_default="evidence.v1",
        ),
        sa.Column(
            "source_type", sa.String(16), nullable=False, server_default="local"
        ),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "index_version_id",
            sa.String(64),
            sa.ForeignKey("index_versions.id", ondelete="RESTRICT"),
        ),
        sa.Column("chunk_id", sa.String(128)),
        sa.Column("source_url", sa.Text()),
        sa.Column("source_fetched_at", sa.DateTime(timezone=True)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(71), nullable=False),
        sa.Column("source_block_ids_json", sa.JSON(), nullable=False),
        sa.Column("provenance_spans_json", sa.JSON(), nullable=False),
        sa.Column("heading_path_json", sa.JSON(), nullable=False),
        sa.Column("asset_refs_json", sa.JSON(), nullable=False),
        sa.Column("source_label", sa.String(1024), nullable=False, server_default=""),
        sa.Column("primary_block_id", sa.String(128), nullable=False),
        sa.Column("primary_availability", sa.String(16), nullable=False),
        sa.Column(
            "primary_coordinate_space",
            sa.String(32),
            nullable=False,
            server_default="none",
        ),
        sa.Column("primary_page_no", sa.Integer()),
        sa.Column("primary_bbox_json", sa.JSON()),
        sa.Column("primary_slide_no", sa.Integer()),
        sa.Column("primary_sheet", sa.String(255)),
        sa.Column("primary_cell_range", sa.String(64)),
        sa.Column("primary_line_start", sa.Integer()),
        sa.Column("primary_line_end", sa.Integer()),
        sa.Column("primary_reason_code", sa.String(128)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "document_version_id",
            "index_version_id",
            "chunk_id",
            name="uq_evidence_version_index_chunk",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "source_url",
            "content_hash",
            name="uq_evidence_web_tenant_url_hash",
        ),
        sa.CheckConstraint(
            "content_hash LIKE 'sha256:%' AND length(content_hash) = 71",
            name="ck_evidence_content_hash",
        ),
        sa.CheckConstraint(
            "(source_type = 'local' AND tenant_id IS NOT NULL AND kb_id IS NOT NULL "
            "AND document_id IS NOT NULL AND document_version_id IS NOT NULL "
            "AND index_version_id IS NOT NULL AND chunk_id IS NOT NULL "
            "AND source_url IS NULL AND source_fetched_at IS NULL) OR "
            "(source_type = 'web' AND source_url IS NOT NULL AND source_fetched_at IS NOT NULL "
            "AND tenant_id IS NOT NULL "
            "AND kb_id IS NULL AND document_id IS NULL AND document_version_id IS NULL "
            "AND index_version_id IS NULL AND chunk_id IS NULL)",
            name="ck_evidence_source_shape",
        ),
        sa.CheckConstraint(
            "primary_availability IN ('exact', 'derived', 'unavailable')",
            name="ck_evidence_primary_availability",
        ),
        sa.CheckConstraint(
            "primary_coordinate_space IN "
            "('none', 'pdf_points', 'normalized_page', 'image_pixels', "
            "'slide_points', 'spreadsheet_cells', 'text_lines')",
            name="ck_evidence_coordinate_space",
        ),
        sa.CheckConstraint(
            "primary_availability != 'unavailable' OR "
            "(primary_page_no IS NULL AND primary_bbox_json IS NULL AND "
            "primary_slide_no IS NULL AND primary_sheet IS NULL AND "
            "primary_cell_range IS NULL AND primary_line_start IS NULL AND "
            "primary_line_end IS NULL AND primary_coordinate_space = 'none' AND "
            "primary_reason_code IS NOT NULL)",
            name="ck_evidence_unavailable_empty_location",
        ),
        sa.CheckConstraint(
            "primary_availability = 'unavailable' OR "
            "(primary_page_no IS NOT NULL OR primary_bbox_json IS NOT NULL OR "
            "primary_slide_no IS NOT NULL OR primary_sheet IS NOT NULL OR "
            "primary_cell_range IS NOT NULL OR primary_line_start IS NOT NULL)",
            name="ck_evidence_available_has_location",
        ),
    )
    op.create_index("ix_evidence_records_tenant_id", "evidence_records", ["tenant_id"])
    op.create_index("ix_evidence_records_kb_id", "evidence_records", ["kb_id"])
    op.create_index("ix_evidence_records_document_id", "evidence_records", ["document_id"])
    op.create_index(
        "ix_evidence_records_document_version_id",
        "evidence_records",
        ["document_version_id"],
    )
    op.create_index(
        "ix_evidence_records_index_version_id",
        "evidence_records",
        ["index_version_id"],
    )
    op.create_index(
        "ix_evidence_tenant_kb", "evidence_records", ["tenant_id", "kb_id"]
    )
    op.create_index(
        "ix_evidence_document_version",
        "evidence_records",
        ["document_id", "document_version_id"],
    )
    op.create_index(
        "ix_evidence_index_chunk",
        "evidence_records",
        ["index_version_id", "chunk_id"],
    )
    op.create_index(
        "ix_evidence_primary_page",
        "evidence_records",
        ["document_version_id", "primary_page_no"],
    )

    op.create_table(
        "citation_facts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "schema_version",
            sa.String(32),
            nullable=False,
            server_default="citation.v1",
        ),
        sa.Column("answer_id", sa.String(128), nullable=False),
        sa.Column(
            "evidence_id",
            sa.String(64),
            sa.ForeignKey("evidence_records.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("answer_span_start", sa.Integer(), nullable=False),
        sa.Column("answer_span_end", sa.Integer(), nullable=False),
        sa.Column("source_quote", sa.Text(), nullable=False),
        sa.Column("quote_hash", sa.String(71), nullable=False),
        sa.Column("source_label", sa.String(1024), nullable=False),
        sa.Column("validation_status", sa.String(16), nullable=False),
        sa.Column("reason_codes_json", sa.JSON(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "answer_id",
            "evidence_id",
            "answer_span_start",
            "answer_span_end",
            "quote_hash",
            name="uq_citation_answer_evidence_span_quote",
        ),
        sa.CheckConstraint(
            "answer_span_start >= 0 AND answer_span_end > answer_span_start",
            name="ck_citation_answer_span",
        ),
        sa.CheckConstraint(
            "quote_hash LIKE 'sha256:%' AND length(quote_hash) = 71",
            name="ck_citation_quote_hash",
        ),
        sa.CheckConstraint(
            "validation_status IN ('valid', 'invalid', 'degraded')",
            name="ck_citation_validation_status",
        ),
    )
    op.create_index("ix_citation_facts_answer_id", "citation_facts", ["answer_id"])
    op.create_index(
        "ix_citation_answer_ordinal", "citation_facts", ["answer_id", "ordinal"]
    )
    op.create_index("ix_citation_evidence", "citation_facts", ["evidence_id"])


def downgrade() -> None:
    # Citation FK 为 RESTRICT，必须先删除引用事实，再删除证据。
    op.drop_table("citation_facts")
    op.drop_table("evidence_records")
