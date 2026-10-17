"""可恢复的 LangGraph 导入执行器。

HTTP API 不调用本模块；独立 Worker 领取租约后调用。每个安全节点完成后先落制品，
再以同一租约令牌登记检查点。进程崩溃时，新 Worker 从最新已登记检查点继续。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.application.checkpoint_store import LocalCheckpointStore
from app.application.ingestion_job_service import (
    complete_job,
    get_latest_checkpoint,
    is_cancellation_requested,
    record_job_checkpoint,
    record_index_manifest,
    start_job_stage,
    update_job_stage,
)
from app.application.ingestion_progress_hooks import (
    IngestionProgressHook,
    bind_ingestion_progress_hook,
)
from app.application.job_errors import ValidationJobError
from app.infra.persistence.sql.job_repository import JobCancelledError
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger


NEXT_NODE = {
    "node_pdf_to_md": "node_metadata_extract",
    "node_metadata_extract": "node_md_img",
    "node_md_img": "node_link_and_table_chunk",
    "node_link_and_table_chunk": "node_chunk_merge",
    "node_chunk_merge": "node_image_upload",
    "node_image_upload": "node_llm_fanout_start",
    "llm_enhance_reducer": "node_bge_embedding",
    "node_bge_embedding": "node_import_milvus",
    "node_import_milvus": "__complete__",
}

NODE_PROGRESS = {
    "node_entry": 0.03,
    "node_pdf_to_md": 0.12,
    "node_metadata_extract": 0.22,
    "node_md_img": 0.34,
    "node_link_and_table_chunk": 0.42,
    "node_chunk_merge": 0.50,
    "node_image_upload": 0.58,
    "node_llm_fanout_start": 0.60,
    "node_keyword_extraction": 0.66,
    "node_question_proposal": 0.66,
    "node_metadata_generate": 0.66,
    "node_content_tagging": 0.66,
    "node_toc": 0.66,
    "llm_enhance_reducer": 0.72,
    "node_bge_embedding": 0.86,
    "node_import_milvus": 0.97,
}

# NODE_PROGRESS 表示节点完成后的里程碑；开始阶段使用上一里程碑，既不虚报
# 未完成工作，也能在长耗时节点运行期间持久化准确的 stage。
NODE_START_PROGRESS = {
    "node_entry": 0.02,
    "node_pdf_to_md": NODE_PROGRESS["node_entry"],
    "node_metadata_extract": NODE_PROGRESS["node_pdf_to_md"],
    "node_md_img": NODE_PROGRESS["node_metadata_extract"],
    "node_link_and_table_chunk": NODE_PROGRESS["node_md_img"],
    "node_chunk_merge": NODE_PROGRESS["node_link_and_table_chunk"],
    "node_image_upload": NODE_PROGRESS["node_chunk_merge"],
    "node_llm_fanout_start": NODE_PROGRESS["node_image_upload"],
    "node_keyword_extraction": NODE_PROGRESS["node_llm_fanout_start"],
    "node_question_proposal": NODE_PROGRESS["node_llm_fanout_start"],
    "node_metadata_generate": NODE_PROGRESS["node_llm_fanout_start"],
    "node_content_tagging": NODE_PROGRESS["node_llm_fanout_start"],
    "node_toc": NODE_PROGRESS["node_llm_fanout_start"],
    "llm_enhance_reducer": NODE_PROGRESS["node_toc"],
    "node_bge_embedding": NODE_PROGRESS["llm_enhance_reducer"],
    "node_import_milvus": NODE_PROGRESS["node_bge_embedding"],
}


def _config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _count_chunks(state: dict) -> int:
    chunks = state.get("chunks") or []
    count = sum(1 for chunk in chunks if chunk.get("chunk_id") is not None)
    if (state.get("toc") or {}).get("chunk_id") is not None:
        count += 1
    return count


def _stage_event_detail(node_name: str, state: dict) -> dict:
    """构造阶段事件；node_md_img 降级时附带经过脱敏的可选依赖 warning。"""
    detail: dict = {"node": node_name}
    if node_name != "node_md_img":
        return detail
    warnings = [
        dict(item)
        for item in (state.get("vlm_warnings") or [])
        if isinstance(item, dict)
    ]
    if warnings:
        detail.update(
            {
                "degraded": True,
                "warning_count": len(warnings),
                # 防止异常文档产生超大事件；完整列表仍保存在节点检查点 state 中。
                "warnings": warnings[:100],
            }
        )
    return detail


def _next_running_stage(node_name: str, state: dict) -> str | None:
    """返回节点完成后将要执行的单一节点；并行 fan-out 不作错误猜测。"""
    if node_name == "node_entry":
        readable = (
            state.get("is_document_read_enabled")
            or state.get("is_pdf_read_enabled")
            or state.get("is_md_read_enabled")
        )
        return "node_pdf_to_md" if readable else None
    next_stage = NEXT_NODE.get(node_name)
    return next_stage if next_stage != "__complete__" else None


def _record_stage_started(
    job_id: str,
    lease_token: str,
    stage: str | None,
    *,
    resumed: bool = False,
) -> None:
    if not stage or stage == "__complete__":
        return
    detail = {"node": stage, "phase": "started"}
    if resumed:
        detail["resumed"] = True
    start_job_stage(
        job_id,
        lease_token,
        stage,
        NODE_START_PROGRESS.get(stage, 0.60),
        detail,
    )


def prepare_checkpoint_state(node_name: str, state: dict, resume_node: str) -> dict:
    """生成可安全序列化的 checkpoint state，不修改 LangGraph 运行态。"""
    checkpoint_state = dict(state)
    checkpoint_state["resume_node"] = resume_node
    if node_name in {
        "node_image_upload",
        "llm_enhance_reducer",
        "node_bge_embedding",
        "node_import_milvus",
    }:
        # 图片上传阶段可能关闭底层文件句柄；后续节点只消费 img_id/chunks，
        # 不再需要 PIL.Image。清空可避免 pickle 触发 closed-image.load()。
        checkpoint_state["image_pil_dict"] = {}
    return checkpoint_state


def _stream_with_ingestion_progress(graph, state: dict, hook: IngestionProgressHook):
    """在 Graph generator 的完整迭代周期内绑定当前 Job 进度上下文。"""

    with bind_ingestion_progress_hook(hook):
        yield from graph.stream(state)


def run_graph_task(job: dict, lease_token: str) -> dict:
    """在有效租约内运行任务；异常交给 Worker 分类并安排重试。"""
    from app.runtime.graphs import get_ingestion_graph
    from app.process.import_.agent.state import get_default_state

    job_id = job["id"]
    parser_config = job.get("parser_config") or {}
    config_sha256 = _config_hash(parser_config)
    input_sha256 = job["content_sha256"]
    checkpoint_store = LocalCheckpointStore(settings.ingestion_checkpoint_root)
    latest = get_latest_checkpoint(job_id)

    if latest:
        if latest["input_sha256"] != input_sha256 or latest["config_sha256"] != config_sha256:
            raise ValidationJobError("checkpoint input/config fingerprint mismatch")
        state = checkpoint_store.load(latest)
        state["resume_node"] = latest["resume_node"]
        logger.info(
            f"[{job_id}] 恢复检查点 #{latest['sequence']} {latest['stage']} -> {latest['resume_node']}"
        )
    else:
        state = get_default_state()
        state.update(
            {
                "task_id": job_id,
                "job_id": job_id,
                "document_id": job["document_id"],
                "document_version_id": job["document_version_id"],
                "index_version_id": job["index_version_id"],
                "kb_id": job["kb_id"],
                "tenant_id": job["tenant_id"],
                "local_dir": str(Path(job["object_key"]).parent),
                "local_file_path": job["object_key"],
                "resume_node": "node_entry",
            }
        )
        state["parser_config"].update(parser_config)

    # 目标索引配置来自持久任务，不从检查点或全局默认值推断。这样兼容性重建可
    # 复用 collection，schema/维度变更时则会严格写入新 collection；恢复执行
    # 也会重新覆盖为数据库中的事实，避免旧检查点把任务写回错误索引。
    state.update(
        {
            "collection_name": job["collection_name"],
            "vector_dim": job["vector_dim"],
            "schema_version": job["schema_version"],
            "embedding_model_id": job["embedding_model_id"],
        }
    )

    if state.get("resume_node") == "__complete__":
        return complete_job(job_id, lease_token, _count_chunks(state))

    logger.info(
        f"[{job_id}] Worker 开始/恢复导入：document={job['document_id']}, "
        f"version={job['document_version_id']}, run_attempt={job['run_attempt']}"
    )
    last_state = state
    checkpoint_sequence = int(job.get("checkpoint_sequence") or 0)

    # LangGraph 的默认 stream 只会在节点返回后产出 update。若此处不先写
    # stage，PDF 解析、embedding、Milvus 等长节点运行期间会一直显示上一个
    # 已完成节点。开始事件与完成事件分离，恢复任务也能立即展示 resume_node。
    _record_stage_started(
        job_id,
        lease_token,
        state.get("resume_node") or "node_entry",
        resumed=latest is not None,
    )

    progress_hook = IngestionProgressHook(job_id=job_id, lease_token=lease_token)
    for event in _stream_with_ingestion_progress(get_ingestion_graph(), state, progress_hook):
        for node_name, node_result in event.items():
            if is_cancellation_requested(job_id, lease_token):
                raise JobCancelledError(f"cancel requested after {node_name}")
            if isinstance(node_result, dict):
                last_state = node_result
            if node_name == "node_import_milvus" and isinstance(last_state, dict):
                manifest = dict(last_state.get("index_manifest") or {})
                if not manifest:
                    raise ValidationJobError("node_import_milvus returned no index manifest")
                manifest.update(
                    {
                        "input_sha256": input_sha256,
                        "config_sha256": config_sha256,
                    }
                )
                record_index_manifest(job_id, lease_token, manifest)
            update_job_stage(
                job_id,
                lease_token,
                node_name,
                NODE_PROGRESS.get(node_name, 0.60),
                _stage_event_detail(node_name, last_state),
            )
            logger.info(f"[{job_id}] LangGraph 节点完成：{node_name}")

            resume_node = NEXT_NODE.get(node_name)
            if resume_node and isinstance(last_state, dict):
                checkpoint_sequence += 1
                checkpoint_state = prepare_checkpoint_state(
                    node_name, last_state, resume_node
                )
                artifact = checkpoint_store.save(
                    job_id, checkpoint_sequence, node_name, checkpoint_state
                )
                record_job_checkpoint(
                    job_id,
                    lease_token,
                    stage=node_name,
                    resume_node=resume_node,
                    input_sha256=input_sha256,
                    config_sha256=config_sha256,
                    **artifact,
                )

            _record_stage_started(
                job_id,
                lease_token,
                _next_running_stage(node_name, last_state),
            )

    if is_cancellation_requested(job_id, lease_token):
        raise JobCancelledError("cancel requested before publish")
    result = complete_job(job_id, lease_token, _count_chunks(last_state))
    logger.info(f"[{job_id}] 导入完成，共发布 {_count_chunks(last_state)} 个块")
    return result
