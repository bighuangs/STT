from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_initial_migration_upgrade_and_downgrade(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "migration.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("CONTROL_DATABASE_URL", url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")
    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert {
        "tenants",
        "workspaces",
        "knowledge_bases",
        "documents",
        "document_versions",
        "ingestion_jobs",
        "index_versions",
        "document_index_bindings",
        "index_build_manifests",
        "cleanup_jobs",
        "document_parse_artifacts",
        "document_chunking_runs",
        "query_conversations",
        "query_turns",
        "query_runs",
        "query_messages",
        "query_events",
        "query_idempotency_keys",
        "query_cache_entries",
        "query_singleflight_leases",
        "query_replay_manifests",
        "model_usage_records",
        "vector_collection_families",
        "vector_placements",
        "vector_gc_runs",
        "vector_migration_runs",
        "alembic_version",
    } <= tables
    version_columns = {
        column["name"] for column in inspect(engine).get_columns("document_versions")
    }
    assert "pipeline_fingerprint" in version_columns
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(url)
    tables_after = set(inspect(engine).get_table_names())
    assert "knowledge_bases" not in tables_after
    engine.dispose()


def test_todo02_upgrade_backfills_an_existing_todo01_job(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "existing-job.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("CONTROL_DATABASE_URL", url)
    config = Config("alembic.ini")
    command.upgrade(config, "20260715_0001")

    engine = create_engine(url)
    statements = [
        "INSERT INTO tenants (id,name,status) VALUES ('t','Tenant','active')",
        "INSERT INTO workspaces (id,tenant_id,name,status) VALUES ('w','t','Workspace','active')",
        "INSERT INTO knowledge_bases (id,tenant_id,workspace_id,name,description,status,created_by) VALUES ('kb','t','w','KB','','active','test')",
        "INSERT INTO documents (id,kb_id,logical_name,source_type,status) VALUES ('doc','kb','manual.md','upload','ready')",
        "INSERT INTO document_versions (id,document_id,version_no,content_sha256,size_bytes,mime_type,object_key,parser_config_snapshot,status) VALUES ('dv','doc',1,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',1,'text/markdown','manual.md','{}','published')",
        "INSERT INTO index_versions (id,kb_id,version_no,embedding_model_id,vector_dim,schema_version,status,active_slot,collection_alias) VALUES ('idx','kb',1,'BAAI/bge-m3',1024,'v1','active','active','kb_chunks')",
        "INSERT INTO ingestion_jobs (id,document_version_id,index_version_id,status,stage,progress,attempt) VALUES ('job_existing','dv','idx','completed','published',1.0,1)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(url)
    with engine.connect() as connection:
        event = connection.execute(
            text(
                "SELECT job_id, sequence, event_type FROM ingestion_job_events "
                "WHERE job_id='job_existing'"
            )
        ).one()
        last_sequence = connection.execute(
            text("SELECT last_event_sequence FROM ingestion_jobs WHERE id='job_existing'")
        ).scalar_one()
    assert tuple(event) == ("job_existing", 1, "migrated")
    assert last_sequence == 1
    engine.dispose()
