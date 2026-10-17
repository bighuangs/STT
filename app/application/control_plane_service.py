"""知识库、文档、版本、导入任务和索引版本应用服务。"""

from collections.abc import Iterable

from app.domain.evidence import CitationFact, Evidence
from app.infra.persistence.sql.database import initialize_control_database, session_scope
from app.infra.persistence.sql.repositories import ControlRepository, canonical_json_sha256
from hashlib import sha256
import json
import time

from sqlalchemy.exc import IntegrityError, OperationalError
from app.shared.runtime.logger import logger


def bootstrap_control_plane() -> None:
    logger.info("控制库初始化：开始执行 Alembic 迁移")
    initialize_control_database()
    logger.info("控制库初始化：Alembic 已到达 head，开始校验默认资源")
    with session_scope() as session:
        ControlRepository(session).ensure_defaults()
    logger.info("控制库初始化：默认资源校验完成")


def create_knowledge_base(name: str, description: str = "", **policies) -> dict:
    with session_scope() as session:
        return ControlRepository(session).create_knowledge_base(name, description, **policies)


def list_knowledge_bases() -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session).list_knowledge_bases()


def get_knowledge_base(kb_id: str) -> dict:
    with session_scope() as session:
        repo = ControlRepository(session)
        return repo.knowledge_base_dict(repo.get_knowledge_base_model(kb_id))


def update_knowledge_base(kb_id: str, **changes) -> dict:
    with session_scope() as session:
        return ControlRepository(session).update_knowledge_base(kb_id, **changes)


def delete_knowledge_base(kb_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).delete_knowledge_base(kb_id)


def list_index_versions(kb_id: str) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session).list_index_versions(kb_id)


def register_uploaded_document(
    *,
    kb_id: str,
    logical_name: str,
    content_sha256: str,
    size_bytes: int,
    mime_type: str,
    object_key: str,
    parser_config_snapshot: dict | None = None,
    document_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict:
    fingerprint = sha256(
        json.dumps(
            {
                "kb_id": kb_id,
                "document_id": document_id,
                "logical_name": logical_name,
                "content_sha256": content_sha256,
                "parser_config": parser_config_snapshot or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    # 幂等键并发击穿时，唯一约束是最终裁判；失败事务回滚后重新读取胜出者。
    attempts = 6 if idempotency_key else 1
    for index in range(attempts):
        try:
            with session_scope() as session:
                return ControlRepository(session).register_upload_and_job(
                    kb_id=kb_id,
                    logical_name=logical_name,
                    content_sha256=content_sha256,
                    size_bytes=size_bytes,
                    mime_type=mime_type,
                    object_key=object_key,
                    parser_config_snapshot=parser_config_snapshot,
                    source_uri=object_key,
                    document_id=document_id,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
        except (IntegrityError, OperationalError):
            if index + 1 >= attempts:
                raise
            time.sleep(0.02 * (2**index))
    raise RuntimeError("unreachable idempotency retry state")


def list_documents(kb_id: str) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session).list_documents(kb_id)


def get_document(document_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).get_document_detail(document_id)


def get_document_version(document_version_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).get_document_version_detail(document_version_id)


def record_document_parse_artifact(**values) -> dict:
    """幂等记录解析制品；同 parser cache key 的成功制品不可被不同内容覆盖。"""
    with session_scope() as session:
        return ControlRepository(session).record_parse_artifact(**values)


def get_document_parse_artifact(artifact_id: str) -> dict:
    with session_scope() as session:
        repo = ControlRepository(session)
        return repo.parse_artifact_detail(repo.get_parse_artifact_model(artifact_id))


def find_document_parse_artifact(**cache_key) -> dict | None:
    with session_scope() as session:
        return ControlRepository(session).find_parse_artifact(**cache_key)


def list_document_parse_artifacts(document_version_id: str) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session).list_parse_artifacts(document_version_id)


def record_document_chunking_run(
    *,
    config_hash: str | None = None,
    config_snapshot: dict | None = None,
    **values,
) -> dict:
    """记录切块运行；调用方可传规范哈希，或传配置快照由应用层计算。"""
    resolved_hash = config_hash or canonical_json_sha256(config_snapshot or {})
    with session_scope() as session:
        return ControlRepository(session).record_chunking_run(
            config_hash=resolved_hash,
            **values,
        )


def get_document_chunking_run(run_id: str) -> dict:
    with session_scope() as session:
        repo = ControlRepository(session)
        return repo.chunking_run_dict(repo.get_chunking_run_model(run_id))


def find_document_chunking_run(
    *,
    config_hash: str | None = None,
    config_snapshot: dict | None = None,
    **cache_key,
) -> dict | None:
    resolved_hash = config_hash or canonical_json_sha256(config_snapshot or {})
    with session_scope() as session:
        return ControlRepository(session).find_chunking_run(
            config_hash=resolved_hash,
            **cache_key,
        )


def list_document_chunking_runs(document_version_id: str) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session).list_chunking_runs(document_version_id)


def record_evidence(
    evidence: Evidence | dict | None = None,
    *,
    tenant_id: str | None = None,
    **values,
) -> dict:
    """幂等保存本地或 Web Evidence；查询分数不会进入持久事实。"""

    if evidence is not None and values:
        raise ValueError("pass either evidence or keyword fields, not both")
    payload = evidence if evidence is not None else values
    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).record_evidence(payload)


def record_evidence_batch(
    evidences: Iterable[Evidence | dict],
    *,
    tenant_id: str | None = None,
) -> list[dict]:
    """单事务保存一批 Evidence；任一失败会回滚整批。"""

    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).record_evidence_batch(
            evidences
        )


def record_citation_bundle(
    evidences: Iterable[Evidence | dict],
    citations: Iterable[CitationFact | dict],
    *,
    tenant_id: str | None = None,
) -> dict[str, list[dict]]:
    """单事务保存回答使用的 Evidence/Citation；任一失败时整批回滚。"""

    with session_scope() as session:
        return ControlRepository(
            session, tenant_id=tenant_id
        ).record_citation_bundle(evidences, citations)


def get_evidence(evidence_id: str, *, tenant_id: str | None = None) -> dict:
    with session_scope() as session:
        repo = ControlRepository(session, tenant_id=tenant_id)
        return repo.evidence_dict(repo.get_evidence_model(evidence_id))


def find_evidence(
    *,
    document_version_id: str,
    index_version_id: str,
    chunk_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).find_evidence(
            document_version_id=document_version_id,
            index_version_id=index_version_id,
            chunk_id=chunk_id,
        )


def list_evidence(
    document_version_id: str,
    *,
    index_version_id: str | None = None,
    tenant_id: str | None = None,
) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).list_evidence(
            document_version_id,
            index_version_id=index_version_id,
        )


def record_citation_fact(
    citation: CitationFact | dict | None = None,
    *,
    tenant_id: str | None = None,
    **values,
) -> dict:
    """幂等保存服务端已经验证或降级的 CitationFact。"""

    if citation is not None and values:
        raise ValueError("pass either citation or keyword fields, not both")
    payload = citation if citation is not None else values
    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).record_citation_fact(
            payload
        )


def get_citation_fact(citation_id: str, *, tenant_id: str | None = None) -> dict:
    with session_scope() as session:
        repo = ControlRepository(session, tenant_id=tenant_id)
        return repo.citation_fact_dict(repo.get_citation_fact_model(citation_id))


def list_citation_facts(answer_id: str, *, tenant_id: str | None = None) -> list[dict]:
    with session_scope() as session:
        return ControlRepository(session, tenant_id=tenant_id).list_citation_facts(
            answer_id
        )


# 查询应用层使用更短的命名；保留 *_fact 以强调数据库中的不可变事实。
record_citation = record_citation_fact
get_citation = get_citation_fact
list_citations = list_citation_facts


def delete_document(document_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).delete_document(document_id)


def create_ingestion_job(version_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).create_job_for_version(version_id)


def get_ingestion_job(job_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).get_ingestion_job_detail(job_id)


def start_ingestion_job(job_id: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).start_job(job_id)


def update_ingestion_job_stage(job_id: str, stage: str, progress: float | None = None) -> dict:
    with session_scope() as session:
        return ControlRepository(session).update_job_stage(job_id, stage, progress)


def complete_ingestion_job(job_id: str, chunk_count: int, token_count: int = 0) -> dict:
    with session_scope() as session:
        return ControlRepository(session).complete_job(
            job_id, chunk_count=chunk_count, token_count=token_count
        )


def fail_ingestion_job(job_id: str, error_code: str, error_detail: str) -> dict:
    with session_scope() as session:
        return ControlRepository(session).fail_job(
            job_id, error_code=error_code, error_detail=error_detail
        )


def import_legacy_document(**values) -> dict:
    with session_scope() as session:
        return ControlRepository(session).import_legacy_document(**values)


def rollback_legacy_batch(batch_id: str) -> int:
    with session_scope() as session:
        return ControlRepository(session).rollback_legacy_batch(batch_id)
