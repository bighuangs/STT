"""TODO-06/18 evaluation datasets, runs, results, feedback and annotations.

Revision ID: 20260721_0006
Revises: 20260716_0005
"""

from alembic import op
import sqlalchemy as sa


revision = "20260721_0006"
down_revision = "20260716_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("dataset_key", sa.String(128), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("source_snapshot", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("parent_dataset_id", sa.String(64)),
        sa.Column("created_by", sa.String(64), nullable=False, server_default="system"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "dataset_key", "version", name="uq_eval_dataset_version"),
    )
    op.create_index("ix_evaluation_datasets_tenant_id", "evaluation_datasets", ["tenant_id"])
    op.create_index("ix_eval_dataset_tenant_status", "evaluation_datasets", ["tenant_id", "status"])
    op.create_index("ix_evaluation_datasets_parent_dataset_id", "evaluation_datasets", ["parent_dataset_id"])

    op.create_table(
        "evaluation_cases",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("dataset_id", sa.String(64), sa.ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(128), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("conversation", sa.JSON(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("unanswerable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("relevant_document_ids", sa.JSON(), nullable=False),
        sa.Column("relevant_document_names", sa.JSON(), nullable=False),
        sa.Column("relevant_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("relevant_evidence", sa.JSON(), nullable=False),
        sa.Column("expected_terms", sa.JSON(), nullable=False),
        sa.Column("forbidden_terms", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("sensitivity", sa.String(32), nullable=False, server_default="internal"),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("dataset_id", "case_id", name="uq_eval_case_dataset_case"),
    )
    op.create_index("ix_eval_case_dataset_order", "evaluation_cases", ["dataset_id", "ordinal"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("dataset_id", sa.String(64), sa.ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("baseline_run_id", sa.String(64)),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False, server_default="live"),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("metrics_summary", sa.JSON(), nullable=False),
        sa.Column("comparison_summary", sa.JSON(), nullable=False),
        sa.Column("gate_result", sa.JSON(), nullable=False),
        sa.Column("report_paths", sa.JSON(), nullable=False),
        sa.Column("error_detail", sa.Text()),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_evaluation_runs_tenant_id", "evaluation_runs", ["tenant_id"])
    op.create_index("ix_evaluation_runs_kb_id", "evaluation_runs", ["kb_id"])
    op.create_index("ix_evaluation_runs_baseline_run_id", "evaluation_runs", ["baseline_run_id"])
    op.create_index("ix_eval_run_tenant_created", "evaluation_runs", ["tenant_id", "created_at"])
    op.create_index("ix_eval_run_dataset_status", "evaluation_runs", ["dataset_id", "status"])

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(128), nullable=False),
        sa.Column("observation", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_type", sa.String(128)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "case_id", name="uq_eval_result_run_case"),
    )
    op.create_index("ix_eval_result_run_passed", "evaluation_results", ["run_id", "passed"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("actor_id", sa.String(64), nullable=False),
        sa.Column("answer_id", sa.String(128), nullable=False),
        sa.Column("evaluation_result_id", sa.String(64)),
        sa.Column("session_id", sa.String(128)),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("selected_citation", sa.JSON(), nullable=False),
        sa.Column("corrected_answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", sa.String(64)),
        sa.Column("review_note", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_tenant_id", "feedback", ["tenant_id"])
    op.create_index("ix_feedback_kb_id", "feedback", ["kb_id"])
    op.create_index("ix_feedback_actor_id", "feedback", ["actor_id"])
    op.create_index("ix_feedback_evaluation_result_id", "feedback", ["evaluation_result_id"])
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"])
    op.create_index("ix_feedback_tenant_status", "feedback", ["tenant_id", "status"])
    op.create_index("ix_feedback_answer", "feedback", ["answer_id"])

    op.create_table(
        "annotation_tasks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("feedback_id", sa.String(64), sa.ForeignKey("feedback.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("target_dataset_key", sa.String(128), nullable=False),
        sa.Column("proposed_case", sa.JSON(), nullable=False),
        sa.Column("reviews", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("arbiter_id", sa.String(64)),
        sa.Column("resolution", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_annotation_tasks_tenant_id", "annotation_tasks", ["tenant_id"])
    op.create_index("ix_annotation_tasks_feedback_id", "annotation_tasks", ["feedback_id"])
    op.create_index("ix_annotation_tenant_status", "annotation_tasks", ["tenant_id", "status"])


def downgrade() -> None:
    op.drop_table("annotation_tasks")
    op.drop_table("feedback")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_cases")
    op.drop_table("evaluation_datasets")
