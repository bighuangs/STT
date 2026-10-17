"""P3 query reliability, security cache and vector topology control plane.

Revision ID: 20260725_0011
Revises: 20260724_0010
Create Date: 2026-07-25
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_0011"
down_revision: str | None = "20260724_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
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
    )


def upgrade() -> None:
    op.create_table(
        "query_conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(512), nullable=False, server_default=""),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_turn_seq", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("resolved_turn_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("history_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.CheckConstraint("revision >= 0", name="ck_query_conversation_revision"),
        sa.CheckConstraint("next_turn_seq >= 1", name="ck_query_conversation_next_turn"),
        sa.CheckConstraint(
            "resolved_turn_seq >= 0 AND resolved_turn_seq < next_turn_seq",
            name="ck_query_conversation_frontier",
        ),
        sa.CheckConstraint(
            "history_version >= 0", name="ck_query_conversation_history_version"
        ),
    )
    for column in ("tenant_id", "user_id", "kb_id"):
        op.create_index(
            f"ix_query_conversations_{column}", "query_conversations", [column]
        )
    op.create_index(
        "ix_query_conversation_owner_updated",
        "query_conversations",
        ["tenant_id", "user_id", "updated_at"],
    )

    op.create_table(
        "query_turns",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("query_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("client_turn_id", sa.String(128), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("base_history_seq", sa.Integer(), nullable=False),
        sa.Column("history_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("input_hash", sa.String(71), nullable=False),
        sa.Column("user_content", sa.Text(), nullable=False),
        sa.Column("active_query_run_id", sa.String(64)),
        sa.Column("ready_result_json", sa.JSON()),
        sa.Column("committed_at", sa.DateTime(timezone=True)),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "conversation_id",
            "client_turn_id",
            name="uq_query_turn_client_identity",
        ),
        sa.UniqueConstraint(
            "conversation_id", "seq", name="uq_query_turn_sequence"
        ),
        sa.CheckConstraint("seq >= 1", name="ck_query_turn_seq"),
        sa.CheckConstraint("base_history_seq >= 0", name="ck_query_turn_base_history"),
        sa.CheckConstraint("history_version >= 0", name="ck_query_turn_history_version"),
    )
    op.create_index("ix_query_turns_conversation_id", "query_turns", ["conversation_id"])
    op.create_index("ix_query_turns_active_query_run_id", "query_turns", ["active_query_run_id"])
    op.create_index(
        "ix_query_turn_conversation_status",
        "query_turns",
        ["conversation_id", "status", "seq"],
    )

    op.create_table(
        "query_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("query_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "turn_id",
            sa.String(64),
            sa.ForeignKey("query_turns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("parent_query_run_id", sa.String(64)),
        sa.Column("replay_of_query_run_id", sa.String(64)),
        sa.Column("idempotency_key", sa.String(256)),
        sa.Column("status", sa.String(32), nullable=False, server_default="accepted"),
        sa.Column(
            "current_stage", sa.String(64), nullable=False, server_default="accepted"
        ),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("profile_version", sa.String(128), nullable=False, server_default=""),
        sa.Column("index_snapshot_json", sa.JSON(), nullable=False),
        sa.Column("config_snapshot_json", sa.JSON(), nullable=False),
        sa.Column("history_snapshot_hash", sa.String(71)),
        sa.Column(
            "history_base_turn_seq", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("history_snapshot_json", sa.JSON()),
        sa.Column("deadline_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("safe_error_message", sa.String(1024)),
        sa.Column("error_detail_ref", sa.Text()),
        sa.Column("retryable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("trace_id", sa.String(128), nullable=False),
        sa.Column("result_json", sa.JSON()),
        *_timestamps(),
        sa.UniqueConstraint("turn_id", "attempt", name="uq_query_run_turn_attempt"),
        sa.UniqueConstraint(
            "turn_id", "idempotency_key", name="uq_query_run_retry_idempotency"
        ),
        sa.CheckConstraint("attempt >= 1", name="ck_query_run_attempt"),
        sa.CheckConstraint("revision >= 0", name="ck_query_run_revision"),
        sa.CheckConstraint("event_sequence >= 0", name="ck_query_run_event_sequence"),
        sa.CheckConstraint(
            "history_base_turn_seq >= 0", name="ck_query_run_history_base_turn"
        ),
    )
    for column in (
        "tenant_id",
        "user_id",
        "conversation_id",
        "turn_id",
        "parent_query_run_id",
        "replay_of_query_run_id",
        "deadline_at",
        "trace_id",
    ):
        op.create_index(f"ix_query_runs_{column}", "query_runs", [column])
    op.create_index(
        "ix_query_run_owner_status",
        "query_runs",
        ["tenant_id", "user_id", "status"],
    )
    op.create_index(
        "ix_query_run_conversation_created",
        "query_runs",
        ["conversation_id", "created_at"],
    )

    op.create_table(
        "query_messages",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("query_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "turn_id",
            sa.String(64),
            sa.ForeignKey("query_turns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_run_id", sa.String(64)),
        sa.Column("turn_seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("committed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "turn_id", "ordinal", name="uq_query_message_turn_ordinal"
        ),
        sa.CheckConstraint("turn_seq >= 1", name="ck_query_message_turn_seq"),
        sa.CheckConstraint("ordinal >= 0", name="ck_query_message_ordinal"),
    )
    for column in ("tenant_id", "conversation_id", "turn_id", "query_run_id"):
        op.create_index(f"ix_query_messages_{column}", "query_messages", [column])
    op.create_index(
        "ix_query_message_history",
        "query_messages",
        ["conversation_id", "turn_seq", "ordinal"],
    )

    op.create_table(
        "query_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column(
            "query_run_id",
            sa.String(64),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("stage", sa.String(64)),
        sa.Column("data_json", sa.JSON(), nullable=False),
        sa.Column("trace_id", sa.String(128), nullable=False),
        sa.Column("visibility", sa.String(16), nullable=False, server_default="public"),
        sa.Column(
            "persistence_class",
            sa.String(16),
            nullable=False,
            server_default="durable",
        ),
        sa.Column(
            "schema_version",
            sa.String(32),
            nullable=False,
            server_default="query-event.v1",
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "query_run_id", "seq", name="uq_query_event_run_sequence"
        ),
        sa.CheckConstraint("seq >= 1", name="ck_query_event_seq"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_query_event_size"),
    )
    for column in ("tenant_id", "query_run_id", "trace_id"):
        op.create_index(f"ix_query_events_{column}", "query_events", [column])
    op.create_index(
        "ix_query_event_run_occurred",
        "query_events",
        ["query_run_id", "occurred_at"],
    )
    op.create_index("ix_query_event_expiry", "query_events", ["expires_at"])

    op.create_table(
        "query_idempotency_keys",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("key_hash", sa.String(71), nullable=False),
        sa.Column("payload_hash", sa.String(71), nullable=False),
        sa.Column(
            "query_run_id",
            sa.String(64),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("response_status", sa.Integer()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "key_hash", name="uq_query_idempotency_tenant_key"
        ),
        sa.CheckConstraint(
            "key_hash LIKE 'sha256:%' AND length(key_hash) = 71",
            name="ck_query_idempotency_key_hash",
        ),
        sa.CheckConstraint(
            "payload_hash LIKE 'sha256:%' AND length(payload_hash) = 71",
            name="ck_query_idempotency_payload_hash",
        ),
    )
    for column in ("tenant_id", "query_run_id", "expires_at"):
        op.create_index(
            f"ix_query_idempotency_keys_{column}",
            "query_idempotency_keys",
            [column],
        )
    op.create_index(
        "ix_query_idempotency_expiry", "query_idempotency_keys", ["expires_at"]
    )

    op.create_table(
        "query_cache_entries",
        sa.Column("cache_key", sa.String(71), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("authorized_scope_hash", sa.String(71), nullable=False),
        sa.Column("layer", sa.String(32), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("versions_json", sa.JSON(), nullable=False),
        sa.Column("classification", sa.String(16), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "cache_key LIKE 'sha256:%' AND length(cache_key) = 71",
            name="ck_query_cache_key",
        ),
        sa.CheckConstraint(
            "authorized_scope_hash LIKE 'sha256:%' "
            "AND length(authorized_scope_hash) = 71",
            name="ck_query_cache_scope_hash",
        ),
        sa.CheckConstraint(
            "classification IN ('public', 'internal')",
            name="ck_query_cache_classification",
        ),
    )
    op.create_index(
        "ix_query_cache_tenant_layer_expiry",
        "query_cache_entries",
        ["tenant_id", "layer", "expires_at"],
    )
    op.create_index(
        "ix_query_cache_scope",
        "query_cache_entries",
        ["tenant_id", "authorized_scope_hash"],
    )
    for column in ("tenant_id", "expires_at"):
        op.create_index(
            f"ix_query_cache_entries_{column}", "query_cache_entries", [column]
        )

    op.create_table(
        "query_singleflight_leases",
        sa.Column("fingerprint", sa.String(71), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(128), nullable=False),
        sa.Column("lease_token_hash", sa.String(71), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("result_cache_key", sa.String(71)),
        *_timestamps(),
        sa.CheckConstraint("fencing_token >= 1", name="ck_query_singleflight_fencing"),
    )
    op.create_index(
        "ix_query_singleflight_leases_tenant_id",
        "query_singleflight_leases",
        ["tenant_id"],
    )
    op.create_index(
        "ix_query_singleflight_expiry",
        "query_singleflight_leases",
        ["lease_expires_at"],
    )

    op.create_table(
        "query_replay_manifests",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column(
            "query_run_id",
            sa.String(64),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("fingerprint", sa.String(71), nullable=False),
        sa.Column("request_hash", sa.String(71), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("event_stream_hash", sa.String(71), nullable=False),
        sa.Column("object_refs_json", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint("query_run_id", name="uq_query_replay_run"),
        sa.CheckConstraint(
            "request_hash LIKE 'sha256:%' AND length(request_hash) = 71",
            name="ck_query_replay_request_hash",
        ),
        sa.CheckConstraint(
            "event_stream_hash LIKE 'sha256:%' AND length(event_stream_hash) = 71",
            name="ck_query_replay_event_hash",
        ),
    )
    for column in ("tenant_id", "query_run_id", "fingerprint", "expires_at"):
        op.create_index(
            f"ix_query_replay_manifests_{column}",
            "query_replay_manifests",
            [column],
        )
    op.create_index(
        "ix_query_replay_tenant_created",
        "query_replay_manifests",
        ["tenant_id", "created_at"],
    )

    op.create_table(
        "model_usage_records",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("kb_id", sa.String(64)),
        sa.Column("query_run_id", sa.String(64)),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("scene", sa.String(64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("calls", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="CNY"),
        sa.Column("unit_price_version", sa.String(64), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("estimated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fallback_sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND cache_read_tokens >= 0",
            name="ck_model_usage_tokens",
        ),
        sa.CheckConstraint("calls >= 1", name="ck_model_usage_calls"),
        sa.CheckConstraint("cost >= 0", name="ck_model_usage_cost"),
    )
    for column in ("tenant_id", "kb_id", "query_run_id"):
        op.create_index(
            f"ix_model_usage_records_{column}", "model_usage_records", [column]
        )
    op.create_index(
        "ix_model_usage_tenant_created",
        "model_usage_records",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_model_usage_query_run", "model_usage_records", ["query_run_id"]
    )

    op.create_table(
        "vector_collection_families",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("physical_database", sa.String(255), nullable=False),
        sa.Column("physical_collection", sa.String(255), nullable=False),
        sa.Column("placement_mode", sa.String(32), nullable=False),
        sa.Column("compatibility_fingerprint", sa.String(64), nullable=False),
        sa.Column("schema_fingerprint", sa.String(64), nullable=False),
        sa.Column("index_fingerprint", sa.String(64), nullable=False),
        sa.Column("compatibility_payload", sa.JSON(), nullable=False),
        sa.Column("partition_strategy", sa.String(64), nullable=False),
        sa.Column("num_partitions", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "physical_database",
            "physical_collection",
            name="uq_vector_family_physical_name",
        ),
        sa.CheckConstraint("epoch >= 1", name="ck_vector_family_epoch"),
        sa.CheckConstraint(
            "status IN ('creating', 'validating', 'active', 'draining', "
            "'retired', 'failed')",
            name="ck_vector_family_status",
        ),
    )
    op.create_index(
        "ix_vector_family_compatibility_status",
        "vector_collection_families",
        ["compatibility_fingerprint", "status"],
    )

    op.create_table(
        "vector_placements",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "index_version_id",
            sa.String(64),
            sa.ForeignKey("index_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "family_id",
            sa.String(255),
            sa.ForeignKey("vector_collection_families.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("partition_bucket", sa.String(255)),
        sa.Column("policy_id", sa.String(128), nullable=False),
        sa.Column("policy_version", sa.Integer(), nullable=False),
        sa.Column("placement_reason_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("readable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("writable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("read_slot", sa.String(16)),
        sa.Column("write_slot", sa.String(16)),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "index_version_id", "read_slot", name="uq_vector_active_read"
        ),
        sa.UniqueConstraint(
            "index_version_id", "write_slot", name="uq_vector_active_write"
        ),
        sa.CheckConstraint("epoch >= 1", name="ck_vector_placement_epoch"),
        sa.CheckConstraint(
            "status IN ('planned', 'active', 'source', 'target', 'retired')",
            name="ck_vector_placement_status",
        ),
        sa.CheckConstraint(
            "(readable = true AND read_slot = 'active') OR "
            "(readable = false AND read_slot IS NULL)",
            name="ck_vector_placement_read_slot",
        ),
        sa.CheckConstraint(
            "(writable = true AND write_slot = 'active') OR "
            "(writable = false AND write_slot IS NULL)",
            name="ck_vector_placement_write_slot",
        ),
    )
    for column in ("tenant_id", "kb_id", "index_version_id", "family_id"):
        op.create_index(
            f"ix_vector_placements_{column}", "vector_placements", [column]
        )
    op.create_index(
        "ix_vector_placement_scope",
        "vector_placements",
        ["tenant_id", "kb_id", "status"],
    )
    op.create_index(
        "ix_vector_placement_family",
        "vector_placements",
        ["family_id", "status"],
    )

    op.create_table(
        "vector_gc_runs",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=False),
        sa.Column("document_version_id", sa.String(64), nullable=False),
        sa.Column("index_version_id", sa.String(64), nullable=False),
        sa.Column("family_id", sa.String(255), nullable=False),
        sa.Column("scope_json", sa.JSON(), nullable=False),
        sa.Column("expected_count", sa.BigInteger(), nullable=False),
        sa.Column("matched_before", sa.BigInteger()),
        sa.Column("delete_acknowledged", sa.Boolean()),
        sa.Column("remaining_after", sa.BigInteger()),
        sa.Column("compaction_id", sa.String(255)),
        sa.Column("compaction_status", sa.String(32)),
        sa.Column("status", sa.String(32), nullable=False, server_default="planned"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(128)),
        sa.Column("lease_token_hash", sa.String(71)),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("next_retry_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint("idempotency_key", name="uq_vector_gc_idempotency"),
        sa.CheckConstraint("expected_count >= 0", name="ck_vector_gc_expected"),
        sa.CheckConstraint("attempt >= 0", name="ck_vector_gc_attempt"),
        sa.CheckConstraint("fencing_token >= 0", name="ck_vector_gc_fencing"),
    )
    for column in (
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "family_id",
    ):
        op.create_index(f"ix_vector_gc_runs_{column}", "vector_gc_runs", [column])
    op.create_index(
        "ix_vector_gc_scope",
        "vector_gc_runs",
        [
            "tenant_id",
            "kb_id",
            "document_id",
            "document_version_id",
            "index_version_id",
        ],
    )
    op.create_index(
        "ix_vector_gc_status_retry",
        "vector_gc_runs",
        ["status", "next_retry_at"],
    )

    op.create_table(
        "vector_migration_runs",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("index_version_id", sa.String(64), nullable=False),
        sa.Column("source_family_id", sa.String(255), nullable=False),
        sa.Column("target_family_id", sa.String(255), nullable=False),
        sa.Column("expected_count", sa.BigInteger(), nullable=False),
        sa.Column("chunk_ids_sha256", sa.String(64), nullable=False),
        sa.Column("target_compatibility_fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="planned"),
        sa.Column("checkpoint_json", sa.JSON(), nullable=False),
        sa.Column("validation_json", sa.JSON(), nullable=False),
        sa.Column("rollback_deadline", sa.DateTime(timezone=True)),
        sa.Column("placement_epoch", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(128)),
        sa.Column("lease_token_hash", sa.String(71)),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "source_family_id",
            "target_family_id",
            "index_version_id",
            name="uq_vector_migration_path",
        ),
        sa.CheckConstraint(
            "expected_count >= 0", name="ck_vector_migration_expected"
        ),
        sa.CheckConstraint("attempt >= 0", name="ck_vector_migration_attempt"),
        sa.CheckConstraint(
            "placement_epoch >= 0", name="ck_vector_migration_epoch"
        ),
    )
    for column in (
        "tenant_id",
        "kb_id",
        "index_version_id",
        "source_family_id",
        "target_family_id",
    ):
        op.create_index(
            f"ix_vector_migration_runs_{column}",
            "vector_migration_runs",
            [column],
        )
    op.create_index(
        "ix_vector_migration_scope_status",
        "vector_migration_runs",
        ["tenant_id", "kb_id", "status"],
    )


def downgrade() -> None:
    for table in (
        "vector_migration_runs",
        "vector_gc_runs",
        "vector_placements",
        "vector_collection_families",
        "model_usage_records",
        "query_replay_manifests",
        "query_singleflight_leases",
        "query_cache_entries",
        "query_idempotency_keys",
        "query_events",
        "query_messages",
        "query_runs",
        "query_turns",
        "query_conversations",
    ):
        op.drop_table(table)

