"""
Milvus 向量库入库服务。

职责：把 LangGraph state 中历经各节点处理后的 chunks，构建为 Milvus 混合搜索所需的字段，
      批量写入 Milvus 集合，对齐 RAGFlow rag/svr/task_executor.py:512-579 的 chunk 上传逻辑。

数据流转（从 state 到 Milvus 字段）：
    文档级（来自 state）：
        state["task_id"]                   → doc_id
        state["kb_id"]                     → kb_id
        state["file_title"]                → docnm_kwd（对齐 ragflow 文档标题字段名）
    chunk 通用：
        chunk["content"]                   → content_with_weight（对齐 ragflow 字段名）
        chunk["type"]                      → type（text/table/hyperlinks，STT 扩展）
        chunk["dense_vector"]              → dense_vector（content 视角，带 title 融合）
        chunk["sparse_vector"]             → sparse_vector（content 视角）
        chunk["img_id"]                    → img_id
        chunk["id"]                        → id（xxhash64 确定性唯一 ID）
        chunk["create_time"]               → create_time
        chunk["create_timestamp_flt"]      → create_timestamp_flt
        chunk["available_int"]             → available_int（默认 1，TOC 块 0）
        chunk["position_int"]              → position_int（块在文档中的顺序，默认 0）
    chunk LLM 增强（来自 4 个并行节点）：
        chunk["key_words"]                 → important_kwd（JSON 字符串，对齐 ragflow 命名）
        chunk["key_tokens"]                → important_tks（JSON 字符串）
        chunk["question_kwd"]              → question_kwd（JSON 字符串，LLM 假设问句列表）
        chunk["question_tks"]              → question_tks（JSON 字符串）
        chunk["question_dense_vector"]     → question_dense_vector（LLM 假设问句视角嵌入，独立，不融合 title）
        chunk["question_sparse_vector"]     → question_sparse_vector（LLM 假设问句视角嵌入，独立）
        chunk["metadata"]                  → meta_fields（JSON 字符串，对齐 ragflow 字段名）
        chunk["tags"]                      → tag_kwd（list[str] → JSON 字符串，ragflow 用 dict[str,int]）
    TOC 块（来自 state["toc"]，STT 扩展，与 ragflow task_executor.py:1099-1112 对齐）：
        toc["content_with_weight"]         → content_with_weight（items 序列化的 JSON）
        toc["available_int"]               → 0
        toc["create_time"]                 → create_time（与普通块共用字段）
        toc["type"] = "toc"                → 与 type="toc" 字段对齐（不再用 toc_kwd 区分）
        toc["id"]                          → xxhash64(content_with_weight + doc_id)
        toc["items"]                       → 不入 Milvus（仅 content_with_weight 中保留）
"""

import hashlib
import json
import math
from threading import RLock
from typing import Any

from app.domain.vector_topology import (
    CompatibilityProfile,
    compatibility_fingerprint,
    require_compatible,
)
from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.application.control_plane_service import record_evidence_batch
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository
from app.shared.runtime.logger import logger, step_log
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.vectorstore.write_compatibility import (
    MilvusWriteCompatibilityGate,
    VectorWriteGateError,
    current_chunks_profile,
    resolve_import_vector_target,
)
from app.rag.import_.config import (
    MILVUS_CHUNK_CONTENT_MAX_LENGTH,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)

# 已通过 schema 一致性校验的集合名缓存（进程级，避免每次导入都发起 has_collection + describe_collection 网络调用）
_verified_collections: set[str] = set()
_collection_prepare_lock = RLock()


def _stable_source_chunk_id(
    chunk: dict,
    *,
    document_version_id: str,
    position: int,
    content: str,
) -> str:
    """返回与 Milvus 自增主键无关的稳定来源块 ID。

    TODO-07 模板会同时提供 ``id``/``chunk_id``（``chk_*``）。历史链路曾把
    Milvus INT64 主键回填到 ``chunk_id``，因此这里只信任带稳定命名空间的值；
    老数据则以版本、位置和正文确定性补齐，避免重试产生新的引用身份。
    """

    for field_name in ("id", "chunk_id", "source_chunk_id"):
        candidate = chunk.get(field_name)
        if isinstance(candidate, str) and candidate.strip().startswith("chk_"):
            return candidate.strip()[:128]
    payload = f"{document_version_id}:{position}:{content}"
    return "chk_" + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:40]


def _unavailable_span(block_id: str, reason_code: str) -> EvidenceLocationSpan:
    return EvidenceLocationSpan(
        block_id=block_id,
        availability="unavailable",
        coordinate_space="none",
        reason_code=reason_code,
    )


def _evidence_location_span(
    block_id: str,
    raw_location: dict[str, Any] | None,
) -> EvidenceLocationSpan:
    """把解析器位置转成严格 Evidence span；未知坐标不做猜测。

    页码、幻灯片号、工作表/单元格和文本行都是解析器直接给出的事实。bbox
    只有在解析器同时声明 coordinate_space 时才保留；旧解析器的无单位 bbox
    会退化为页级/幻灯片级定位，绝不把像素猜成 PDF point。
    """

    if not isinstance(raw_location, dict):
        return _unavailable_span(block_id, "parser_location_unavailable")

    explicit_availability = str(raw_location.get("availability") or "exact")
    if explicit_availability == "unavailable":
        return _unavailable_span(
            block_id,
            str(raw_location.get("reason_code") or "parser_location_unavailable"),
        )
    availability = "derived" if explicit_availability == "derived" else "exact"
    derivation = (
        str(raw_location.get("derivation") or "parser_derived_location")
        if availability == "derived"
        else None
    )

    page_no = raw_location.get("page_no")
    slide_no = raw_location.get("slide_no")
    sheet = raw_location.get("sheet")
    cell_range = raw_location.get("cell_range")
    line_start = raw_location.get("line_start")
    line_end = raw_location.get("line_end")
    bbox = raw_location.get("bbox")
    coordinate_space = str(raw_location.get("coordinate_space") or "none")
    supported_spaces = {
        "none",
        "pdf_points",
        "normalized_page",
        "image_pixels",
        "slide_points",
        "spreadsheet_cells",
        "text_lines",
    }
    if coordinate_space not in supported_spaces:
        coordinate_space = "none"

    # 结构化坐标的单位可由字段本身无歧义确定。
    if sheet is not None and cell_range is not None:
        coordinate_space = "spreadsheet_cells"
    elif line_start is not None:
        coordinate_space = "text_lines"

    # 旧契约的 bbox 没有坐标系。保留可靠的页/幻灯片定位，舍弃无单位 bbox。
    spatial_spaces = {
        "pdf_points",
        "normalized_page",
        "image_pixels",
        "slide_points",
    }
    safe_bbox = None
    if (
        bbox is not None
        and coordinate_space in spatial_spaces
        and isinstance(bbox, (list, tuple))
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
    ):
        try:
            values = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            values = ()
        if (
            len(values) == 4
            and all(math.isfinite(value) for value in values)
            and min(values) >= 0
            and values[2] > values[0]
            and values[3] > values[1]
            and (coordinate_space != "normalized_page" or all(value <= 1.0 for value in values))
        ):
            safe_bbox = values
    if safe_bbox is None and coordinate_space in spatial_spaces:
        coordinate_space = "none"

    has_locator = any(
        value is not None
        for value in (
            page_no,
            slide_no,
            sheet,
            cell_range,
            line_start,
            line_end,
            safe_bbox,
        )
    )
    if not has_locator:
        reason = (
            "coordinate_space_unavailable" if bbox is not None else "parser_location_unavailable"
        )
        return _unavailable_span(block_id, reason)

    return EvidenceLocationSpan(
        block_id=block_id,
        availability=availability,
        coordinate_space=coordinate_space,
        page_no=page_no,
        bbox=safe_bbox,
        slide_no=slide_no,
        sheet=sheet,
        cell_range=cell_range,
        line_start=line_start,
        line_end=line_end,
        derivation=derivation,
    )


def _build_local_evidence(
    chunk: dict,
    *,
    doc_id: str,
    kb_id: str,
    tenant_id: str,
    document_version_id: str,
    index_version_id: str,
    file_title: str,
    position: int,
) -> Evidence:
    """从普通可检索 chunk 构建一条不可变、可持久化的本地 Evidence。"""

    content = str(chunk.get("content") or "")
    source_chunk_id = _stable_source_chunk_id(
        chunk,
        document_version_id=document_version_id,
        position=position,
        content=content,
    )
    provenance = chunk.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}

    raw_source_ids = provenance.get("source_block_ids")
    source_block_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in (raw_source_ids if isinstance(raw_source_ids, list) else [])
            if str(value).strip()
        )
    )
    if not source_block_ids:
        # 历史 chunk 没有解析块身份时显式生成一个 legacy block，并把位置标记为
        # unavailable；这比伪造页码更可审计，也让 Evidence 契约保持闭合。
        source_block_ids = ["blk_" + hashlib.sha256(source_chunk_id.encode()).hexdigest()[:32]]

    raw_locations = provenance.get("locations")
    locations_by_block: dict[str, dict[str, Any]] = {}
    if isinstance(raw_locations, list):
        for raw in raw_locations:
            if not isinstance(raw, dict):
                continue
            raw_block_id = str(raw.get("block_id") or "").strip()
            if raw_block_id and raw_block_id not in locations_by_block:
                locations_by_block[raw_block_id] = raw
    spans = [
        _evidence_location_span(block_id, locations_by_block.get(block_id))
        for block_id in source_block_ids
    ]

    raw_heading_path = provenance.get("heading_path")
    heading_path = list(
        dict.fromkeys(
            str(value).strip()
            for value in (raw_heading_path if isinstance(raw_heading_path, list) else [])
            if str(value).strip()
        )
    )
    raw_assets = chunk.get("images")
    asset_refs = [
        str(value).strip()
        for value in (raw_assets if isinstance(raw_assets, list) else [])
        if str(value).strip()
    ]
    raw_img_ids = chunk.get("img_ids")
    asset_refs.extend(
        str(value).strip()
        for value in (raw_img_ids if isinstance(raw_img_ids, list) else [])
        if str(value).strip()
    )
    img_id = str(chunk.get("img_id") or "").strip()
    if img_id:
        asset_refs.append(img_id)
    asset_refs = list(dict.fromkeys(asset_refs))

    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id=tenant_id,
            kb_id=kb_id,
            document_id=doc_id,
            document_version_id=document_version_id,
            index_version_id=index_version_id,
            chunk_id=source_chunk_id,
            source_block_ids=source_block_ids,
        ),
        content=content,
        provenance_spans=spans,
        heading_path=heading_path,
        asset_refs=asset_refs,
        source_label=(file_title or doc_id),
    )


def _jsonify(value, default: str = "", max_length: int | None = None) -> str:
    """
    将 Python 对象（list / dict）安全序列化为 JSON 字符串。
    用于 list[str] / dict 等非标量字段存入 Milvus VARCHAR 列。
    序列化失败或输入为空时返回 default，保证字段类型稳定。

    当 max_length 给定时，超长 JSON 会被安全截断到 max_length 字节（按字符边界切，末尾补 '"}]' 平衡括号），
    触发截断时 logger.warning 提醒调用方——避免 Milvus 写入时因 VARCHAR 超长整批回滚。
    """
    import json

    if value is None:
        return default
    if isinstance(value, str):
        return value[:max_length] if max_length else value
    try:
        s = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return default

    if max_length and len(s) > max_length:
        # 在 UTF-8 安全边界截断，并补上 JSON 闭合括号（list/dict 必有开括号）
        truncated = s.encode("utf-8")[: max_length - 3].decode("utf-8", errors="ignore")
        # 简单处理：末尾补 '"}]' 兼容 list/dict 顶层形态（仅防御性，ragflow 也容忍半截 JSON）
        logger.warning(
            f"[_jsonify] JSON 截断: 原长={len(s)} -> {max_length} 字节（值类型={type(value).__name__}）"
        )
        return truncated + '"}]'
    return s


def _build_chunk_milvus_fields(
    chunk: dict,
    doc_id: str,
    kb_id: str,
    tenant_id: str,
    document_version_id: str,
    index_version_id: str,
    file_title: str,
    position: int,
) -> dict:
    """
    将单个 chunk 字典映射为 Milvus 集合字段格式（不含 TOC 专属字段）。

    字段命名约定：
        - 与 ragflow task_executor.py:539-557 + 1104-1112 1:1 对齐的字段保留 ragflow 命名
          （content_with_weight / docnm_kwd / important_kwd / meta_fields 等）
        - STT 独有字段（type / position_int 等）保留 STT 命名

    :param chunk: 单个 chunk 字典（经各节点处理后）
    :param doc_id: 文档 ID
    :param kb_id: 知识库 ID
    :param file_title: 文档标题
    :param position: chunk 在文档中的顺序（写入 position_int）
    :return: Milvus 字段格式的字典
    """
    milvus_chunk: dict = {}

    # ---- 文档级基础字段 ----
    milvus_chunk["doc_id"] = doc_id
    milvus_chunk["kb_id"] = kb_id
    # TODO-01：动态字段由 enable_dynamic_field 持久化，不触发开发期 drop/recreate。
    milvus_chunk["tenant_id"] = tenant_id
    milvus_chunk["document_version_id"] = document_version_id
    milvus_chunk["index_version_id"] = index_version_id
    milvus_chunk["docnm_kwd"] = file_title  # 对齐 ragflow docnm_kwd

    # ---- chunk 内容字段（content 同时写入 content_with_weight，对齐 ragflow）----
    content = chunk.get("content", "") or ""
    milvus_chunk["content_with_weight"] = content

    # ---- chunk 类型（STT 扩展：text / table / hyperlinks）----
    milvus_chunk["type"] = chunk.get("type", "text")

    # ---- 向量字段（hyperlinks / 失败批 chunk 无向量，由 index_chunks 入库前过滤）----
    milvus_chunk["dense_vector"] = chunk.get("dense_vector", [])
    milvus_chunk["sparse_vector"] = chunk.get("sparse_vector", {})

    # ---- 单图主图 ID（多图场景的辅助列表不持久化，避免 VARCHAR 超长）----
    milvus_chunk["img_id"] = chunk.get("img_id", "")

    # ---- chunk 唯一标识（由 bind_img_id_to_chunks 生成 xxhash64(content + doc_id)）----
    deterministic_source = (
        f"{document_version_id}:{position}:{chunk.get('type', 'text')}:{content.strip()}"
    )
    milvus_chunk["id"] = hashlib.sha256(
        deterministic_source.encode("utf-8", "surrogatepass")
    ).hexdigest()

    # ---- TODO-08 稳定 Evidence 紧凑键 ----
    # 只有携带 sparse_vector、实际会进入可检索 manifest 的普通块才创建 Evidence。
    # 完整位置/标题/资产事实进入 SQL；Milvus 仅保留检索命中后恢复 Evidence 所需的
    # 稳定键和一小段位置摘要，避免把完整 provenance JSON 当作事实源。
    if milvus_chunk["sparse_vector"]:
        evidence = _build_local_evidence(
            chunk,
            doc_id=doc_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            index_version_id=index_version_id,
            file_title=file_title,
            position=position,
        )
        primary_location = evidence.primary_location_span()
        ranked_locations = sorted(
            enumerate(evidence.provenance_spans),
            key=lambda item: (
                {"exact": 0, "derived": 1, "unavailable": 2}[item[1].availability],
                item[0],
            ),
        )
        primary_page_no = next(
            (span.page_no for _, span in ranked_locations if span.page_no is not None),
            0,
        )
        milvus_chunk["evidence_id"] = evidence.evidence_id
        milvus_chunk["content_hash"] = evidence.content_hash
        milvus_chunk["source_chunk_id"] = evidence.source.chunk_id
        milvus_chunk["location_status"] = primary_location.availability
        # 0 是 Milvus 紧凑摘要中的“无可靠页码”哨兵；SQL Evidence 中仍为 null +
        # reason_code，不会把第 0 页暴露为真实来源。
        milvus_chunk["primary_page_no"] = int(primary_page_no or 0)

    # ---- 时间戳字段（由 bind_img_id_to_chunks 填充）----
    milvus_chunk["create_time"] = chunk.get("create_time", "")
    milvus_chunk["create_timestamp_flt"] = chunk.get("create_timestamp_flt", 0.0)

    # ---- 可用性标志（1=可被用户检索；0=TOC 等内部块，对齐 ragflow）----
    # 普通块默认 1；TOC 块在 _build_toc_chunk_milvus_fields 中显式置 0
    milvus_chunk["available_int"] = int(chunk.get("available_int", 1))

    # ---- 块在文档中的顺序（对齐 ragflow position_int；STT 用 enumerate 注入）----
    milvus_chunk["position_int"] = int(chunk.get("position_int", position))

    # ---- 4 个 LLM 增强节点的字段（与 ragflow 字段名 1:1 对齐）----
    # keyword_extraction_service 写入 key_words / key_tokens
    # → ragflow 字段：important_kwd / important_tks（list[str]）
    milvus_chunk["important_kwd"] = _jsonify(
        chunk.get("key_words") or chunk.get("important_kwd") or [],
        default="[]",
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )
    milvus_chunk["important_tks"] = _jsonify(
        chunk.get("key_tokens") or chunk.get("important_tks") or [],
        default="[]",
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )

    # question_proposal_service 写入 question_kwd / question_tks
    # → ragflow 字段：question_kwd / question_tks（同名，list[str]）
    milvus_chunk["question_kwd"] = _jsonify(
        chunk.get("question_kwd") or [],
        default="[]",
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )
    milvus_chunk["question_tks"] = _jsonify(
        chunk.get("question_tks") or [],
        default="[]",
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )

    # metadata_generate_service 写入 metadata（dict）
    # → ragflow 字段：meta_fields（dict，STT 用同名字段）
    # TODO-07：来源证据与 LLM 生成的业务 metadata 是两个独立维度。
    # LLM 节点可以覆盖 chunk["metadata"]，但不得覆盖解析器/模板产生的 provenance。
    source_metadata: dict = {}
    legacy_meta = chunk.get("meta_fields")
    if isinstance(legacy_meta, dict):
        source_metadata.update(legacy_meta)
    generated_meta = chunk.get("metadata")
    if isinstance(generated_meta, dict):
        source_metadata.update(generated_meta)
    provenance = chunk.get("provenance")
    if isinstance(provenance, dict) and provenance:
        source_metadata["_source"] = provenance
    milvus_chunk["meta_fields"] = _jsonify(
        source_metadata,
        default="{}",
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )

    # content_tagging_service 写入 tags（list[str]）
    # → ragflow 字段：TAG_FLD（dict[str, int]，即 {tag: score}）
    # STT 当前不持久化 score，统一以 {"tag": 1} 形式回填，保证与 ragflow 字段语义兼容
    tags = chunk.get("tags") or chunk.get("tag_kwd") or []
    if tags and isinstance(tags[0], str):
        # STT 格式：list[str] → 转为 {"tag": 1} dict 再 JSON 化
        tag_dict = {t: 1 for t in tags}
        milvus_chunk["tag_kwd"] = _jsonify(
            tag_dict, default="{}", max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
        )
    elif tags and isinstance(tags[0], dict):
        # 已经是 dict[str, int] 格式（兼容 ragflow 输入）
        milvus_chunk["tag_kwd"] = _jsonify(
            tags, default="{}", max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
        )
    else:
        milvus_chunk["tag_kwd"] = ""

    return milvus_chunk


def _build_toc_chunk_milvus_fields(
    toc: dict,
    doc_id: str,
    kb_id: str,
    tenant_id: str,
    document_version_id: str,
    index_version_id: str,
    file_title: str,
) -> dict:
    """
    将 state["toc"] 转换为 Milvus 字段格式（与 ragflow task_executor.py:1104-1112 对齐）。

    TOC 块特点：
        - available_int=0：不可被用户检索（仅前端展示导航）
        - type="toc"：与普通块的 "text"/"table"/"hyperlinks" 区分，前端据此识别
        - content_with_weight=items 序列化的 JSON（替代 text 正文）
        - id=xxhash64(content_with_weight + doc_id)

    :param toc: state["toc"] 字典（含 items / content_with_weight / id 等）
    :param doc_id: 文档 ID
    :param kb_id: 知识库 ID
    :param file_title: 文档标题
    :return: Milvus 字段格式的 TOC 块字典
    """
    if not toc:
        return {}

    # content_with_weight：toc_service 已生成 items 序列化的 JSON
    content_with_weight = toc.get("content_with_weight", "")
    if not content_with_weight:
        # 兜底：从 items 自行序列化
        import json

        content_with_weight = json.dumps(toc.get("items", []), ensure_ascii=False)

    # 顺序：TOC 块语义与普通块区分由 available_int=0 + type="toc" 字段承担
    # （原 page_num_int 字段设计为"让 TOC 排到末尾"用极大值，但与 position_int 语义重叠，
    #   且无法要求所有普通 chunk 都写该字段——已从 schema 移除，避免插入失败）
    # page_num = toc.get("page_num_int") or [100000000]  # 已弃用

    # id：toc_service 已生成 xxhash64 hex；缺失时兜底生成
    toc_id = hashlib.sha256(
        f"{document_version_id}:toc:{content_with_weight}".encode("utf-8", "surrogatepass")
    ).hexdigest()

    milvus_toc: dict = {
        # 文档级
        "doc_id": doc_id,
        "kb_id": kb_id,
        "tenant_id": tenant_id,
        "document_version_id": document_version_id,
        "index_version_id": index_version_id,
        "docnm_kwd": file_title,
        # TOC 专属
        "content_with_weight": content_with_weight,
        "available_int": 0,  # 不可被用户检索
        "id": toc_id,
        # 占位字段（保证 schema 完整）
        "type": "toc",
        "dense_vector": [],  # TOC 不参与向量检索
        "sparse_vector": {},
        "img_id": "",
        "create_time": toc.get("create_time", ""),
        "create_timestamp_flt": toc.get("create_timestamp_flt", 0.0),
        "position_int": toc.get("position_int", 0),
        "important_kwd": "",
        "important_tks": "",
        "question_kwd": "",
        "question_tks": "",
        "meta_fields": "{}",
        "tag_kwd": "",
    }
    return milvus_toc


def _build_milvus_chunks(state: dict) -> list[dict]:
    """
    将 state["chunks"]（各节点逐步增强后的切片）+ state["toc"]（TOC 块）转换为 Milvus 集合字段格式。

    处理流程：
        1. 为每个普通 chunk 注入 position_int（enumerate 顺序），映射所有 20+ 字段
        2. 若 state["toc"] 存在，转换为 TOC 块（available_int=0），追加到结果末尾
        3. 返回 Milvus 格式的 chunk 列表

    对齐 RAGFlow task_executor.py:539-557 + 1104-1112 + 1947-2070。

    :param state: LangGraph 流程状态，包含 task_id / kb_id / file_title / chunks / toc 等
    :return: Milvus 格式的 chunk 列表（普通块 + TOC 块）
    """
    chunks = state.get("chunks", [])
    if not chunks and not state.get("toc"):
        return []

    # 新链路使用稳定 document_id；task_id 只作为旧调用的一版兼容兜底。
    doc_id = state.get("document_id") or state.get("task_id", "")
    kb_id = str(state.get("kb_id", ""))
    tenant_id = str(state.get("tenant_id", "tenant_default"))
    document_version_id = str(state.get("document_version_id", ""))
    index_version_id = str(state.get("index_version_id", ""))
    file_title = state.get("file_title", "")

    result: list[dict] = []

    # 1. 普通 chunks（注入 position_int 标识块在文档中的顺序）
    for idx, chunk in enumerate(chunks):
        result.append(
            _build_chunk_milvus_fields(
                chunk=chunk,
                doc_id=doc_id,
                kb_id=kb_id,
                tenant_id=tenant_id,
                document_version_id=document_version_id,
                index_version_id=index_version_id,
                file_title=file_title,
                position=idx,
            )
        )

    # 2. TOC 块（与 ragflow build_TOC 行为对齐，available_int=0）
    toc = state.get("toc")
    if toc:
        toc_chunk = _build_toc_chunk_milvus_fields(
            toc=toc,
            doc_id=doc_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            index_version_id=index_version_id,
            file_title=file_title,
        )
        if toc_chunk:
            result.append(toc_chunk)
            logger.info(
                f"[_build_milvus_chunks] 已构造 TOC 块: id={toc_chunk.get('id', '')[:16]}…, "
                f"items={len(toc.get('items', []))}"
            )

    return result


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """校验 state 中是否存在 chunks，无则抛异常。"""
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空,无法继续业务处理!")
        raise ValueError("chunks为空,无法继续业务处理!")
    return chunks


@step_log("prepare_chunks_collection")
def prepare_chunks_collection(
    collection_name: str | None = None,
    vector_dim: int = MILVUS_VECTOR_DIM,
    *,
    schema_version: str = "chunks_v1",
    embedding_model_id: str = "bge-m3",
    managed_profile: CompatibilityProfile | None = None,
) -> None:
    """
    准备 Milvus chunks 集合。检查集合是否存在，不存在则创建 schema 和索引。

    字段设计（对齐 RAGFlow rag/svr/task_executor.py:539-557 + 1104-1112 + rag/app/naive.py 切块流程）：
        - chunk_id              : 主键，自增 INT64
        - doc_id                : 文档 ID（task_id），VARCHAR
        - kb_id                 : 知识库 ID，VARCHAR
        - docnm_kwd             : 文档标题（对齐 ragflow 字段名），VARCHAR
        - content               : 块文本内容，VARCHAR（截断至 MILVUS_CHUNK_CONTENT_MAX_LENGTH）
        - content_with_weight   : 对齐 ragflow 字段（与 content 双重存储），VARCHAR
        - dense_vector          : BGE-M3 稠密向量，1024 维 FLOAT_VECTOR（title+content 融合，与 ragflow 对齐）
        - sparse_vector         : BGE-M3 稀疏向量（BM25 风格 dict），SPARSE_FLOAT_VECTOR
        - type                  : 块类型（text / table / hyperlinks / toc），VARCHAR(32)
        - img_id                : MinIO 图片主图 ID，VARCHAR
        - id                    : xxhash64(content + doc_id) 确定性唯一 ID，VARCHAR(64)
        - create_time           : 创建时间字符串，VARCHAR(32)
        - create_timestamp_flt  : 创建时间戳，DOUBLE
        - available_int         : 块可用性（1=可检索；0=TOC 内部块），INT64
        - position_int          : 块在文档中的顺序，INT64
        - important_kwd         : 关键词列表（对齐 ragflow 字段名，JSON 字符串），VARCHAR
        - important_tks         : 关键词 jieba 切分（对齐 ragflow 字段名，JSON 字符串），VARCHAR
        - question_kwd          : 配套提问列表（JSON 字符串），VARCHAR
        - question_tks          : 提问 jieba 切分（JSON 字符串），VARCHAR
        - meta_fields           : 结构化元数据（对齐 ragflow 字段名，JSON 字符串），VARCHAR
        - tag_kwd               : 标签（对齐 ragflow TAG_FLD 字段名，JSON 字符串），VARCHAR
    """
    milvus_client = milvus_gateway.client()
    collection_name = collection_name or milvus_gateway.chunks_collection

    if managed_profile is not None:
        # 受管目标只能由当前写链能够完整证明的 schema/index 合同创建。合同升级而
        # 创建器未同步时，会在任何 Milvus mutation 前失败关闭。
        generated_profile = current_chunks_profile(
            vector_dim=vector_dim,
            schema_version=schema_version,
            embedding_model_id=embedding_model_id,
        )
        require_compatible(
            compatibility_fingerprint(managed_profile),
            compatibility_fingerprint(generated_profile),
        )
        if (
            managed_profile.partition.strategy != "none"
            or managed_profile.partition.num_partitions != 1
        ):
            raise VectorWriteGateError("当前导入创建器不支持受管 partition 合同")

    # 进程级缓存：schema 已验证过则跳过网络调用（每次导入节省 ~150ms has_collection + describe_collection）
    if collection_name in _verified_collections:
        logger.debug(
            f"[prepare_chunks_collection] 集合 [{collection_name}] schema 已验证（缓存命中），跳过检查"
        )
        return

    # 预期 schema 字段名集合（schema 构造前先列出，便于与远端比对）
    core_field_names = {
        "chunk_id",
        "doc_id",
        "kb_id",
        "docnm_kwd",
        "content_with_weight",
        "dense_vector",
        "sparse_vector",
        "type",
        "img_id",
        "id",
        "create_time",
        "create_timestamp_flt",
        "available_int",
        "position_int",
        "important_kwd",
        "important_tks",
        "question_kwd",
        "question_tks",
        "meta_fields",
        "tag_kwd",
    }
    identity_field_names = {"tenant_id", "document_version_id", "index_version_id"}
    evidence_field_names = {
        "evidence_id",
        "content_hash",
        "source_chunk_id",
        "location_status",
        "primary_page_no",
    }
    expected_field_names = core_field_names | identity_field_names | evidence_field_names

    if milvus_client.has_collection(collection_name):
        # 远端 schema 与本函数预期不一致 → drop & 重建（开发期便捷路径）
        try:
            remote_fields = {
                f.get("name")
                for f in milvus_client.describe_collection(collection_name).get("fields", [])
            }
        except Exception as e:
            logger.warning(f"读取远端 schema 失败，按一致处理：{e}")
            remote_fields = set()
        # TODO-01/03 的旧集合通过 dynamic field 保存版本身份，TODO-08 同样可以
        # 动态保存 Evidence 紧凑键。只要远端是本项目已知 schema 的子集就兼容使用，
        # 不为“补列”drop 线上集合；新 index version 会显式创建全部字段。
        if core_field_names <= remote_fields <= expected_field_names:
            logger.info(f"Milvus 集合 [{collection_name}] 已存在且 schema 一致，跳过创建")
            _verified_collections.add(collection_name)
            return
        raise RuntimeError(
            f"Milvus 集合 [{collection_name}] 远端 schema 与预期不一致："
            f"差集(远端-预期)={remote_fields - expected_field_names}, "
            f"差集(预期-远端)={expected_field_names - remote_fields}，"
            "拒绝 drop 线上集合；请创建新的 index version/collection"
        )

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=5, is_primary=True, auto_id=True)  # INT64
    schema.add_field(
        field_name="doc_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="kb_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="tenant_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )
    schema.add_field(
        field_name="document_version_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )
    schema.add_field(
        field_name="index_version_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )
    schema.add_field(field_name="evidence_id", datatype=21, max_length=64)
    schema.add_field(field_name="content_hash", datatype=21, max_length=80)
    schema.add_field(field_name="source_chunk_id", datatype=21, max_length=128)
    schema.add_field(field_name="location_status", datatype=21, max_length=16)
    schema.add_field(field_name="primary_page_no", datatype=5)
    schema.add_field(
        field_name="docnm_kwd", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="content_with_weight", datatype=21, max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(field_name="dense_vector", datatype=101, dim=vector_dim)  # FLOAT_VECTOR
    schema.add_field(
        field_name="sparse_vector", datatype=104
    )  # SPARSE_FLOAT_VECTOR（Milvus 2.4 不支持 nullable，空值 chunk 由 index_chunks 过滤）
    schema.add_field(field_name="type", datatype=21, max_length=32)  # VARCHAR
    schema.add_field(
        field_name="img_id", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="id", datatype=21, max_length=64
    )  # VARCHAR（xxhash64 固定 16 字节 hex）
    schema.add_field(field_name="create_time", datatype=21, max_length=32)  # VARCHAR
    schema.add_field(field_name="create_timestamp_flt", datatype=11)  # DOUBLE
    schema.add_field(field_name="available_int", datatype=5)  # INT64（0=不可检索，1=可检索）
    schema.add_field(
        field_name="position_int", datatype=5
    )  # INT64（块在文档中的顺序，支持 >127 个 chunk）
    schema.add_field(
        field_name="important_kwd", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="important_tks", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="question_kwd", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="question_tks", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="meta_fields", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR
    schema.add_field(
        field_name="tag_kwd", datatype=21, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
    )  # VARCHAR

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 进程内串行化同名首次创建；跨进程竞态时 Milvus 的唯一 collection 名决定
    # 胜者，失败者只在确认目标确已存在后继续，随后仍须通过完整物理兼容门禁。
    with _collection_prepare_lock:
        if not milvus_client.has_collection(collection_name):
            try:
                milvus_client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params,
                )
            except Exception:
                if not milvus_client.has_collection(collection_name):
                    raise
    _verified_collections.add(collection_name)
    logger.info(
        f"Milvus 集合 [{collection_name}] 创建完成，字段: "
        f"chunk_id/doc_id/kb_id/tenant_id/document_version_id/index_version_id/"
        f"evidence_id/content_hash/source_chunk_id/location_status/primary_page_no/"
        f"docnm_kwd/content_with_weight/"
        f"dense_vector/sparse_vector/"
        f"type/img_id/id/create_time/"
        f"create_timestamp_flt/available_int/position_int/"
        f"important_kwd/important_tks/question_kwd/question_tks/meta_fields/tag_kwd"
    )


@step_log("delete_existing_doc_chunks")
def delete_existing_doc_chunks(doc_id: str) -> None:
    """拒绝旧式 doc-only 删除，防止绕过 TODO-37 五维作用域门禁。

    正式导入必须由 :func:`index_chunks` 使用 tenant、KB、document、
    document-version、index-version 五层身份以及可选 partition 精确清理
    staging 数据。保留此符号只是为了让旧调用方显式失败，而不是悄悄扩大删除面。
    """
    raise RuntimeError("不安全的 doc-only Milvus 删除已禁用；请通过 index_chunks 的五维作用域清理")


@step_log("insert_chunks")
def insert_chunks(
    milvus_chunks: list[dict],
    collection_name: str | None = None,
    *,
    partition_name: str | None = None,
) -> list[int]:
    """
    将 Milvus 格式的切片列表批量插入，返回主键列表。

    :param milvus_chunks: _build_milvus_chunks 转换后的切片列表
    :return: Milvus 插入返回的主键列表
    """
    if not milvus_chunks:
        return []

    collection_name = collection_name or milvus_gateway.chunks_collection
    client = milvus_gateway.client()
    insert_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "data": milvus_chunks,
    }
    if partition_name is not None:
        insert_kwargs["partition_name"] = partition_name
    result = client.insert(**insert_kwargs)
    insert_count = result.get("insert_count", 0)
    chunk_ids = result.get("ids", [])
    logger.info(f"Milvus 插入完成：{insert_count} 条，主键: {chunk_ids}")

    # TODO-03 publish 前必须证明可检索；因此 flush 是一致性屏障，不能再放后台。
    client.flush(collection_name=collection_name)
    logger.info(f"Milvus 集合 [{collection_name}] flush 完成，进入 manifest 校验")

    return chunk_ids


@step_log("index_chunks")
def index_chunks(state: dict) -> dict:
    """
    入库总入口：对齐 RAGFlow task_executor.py:512-579 + 1104-1112 + 1947-2070。

    处理流程：
        1. require_chunks — 校验 chunks 非空
        2. prepare_chunks_collection — 确保 Milvus 集合已就绪（幂等）
        3. _build_milvus_chunks — 将 state chunks + state["toc"] 映射为 Milvus 字段格式
        4. insert_chunks — 批量写入 Milvus（普通块 + TOC 块）
        5. 回填 chunk_id 到原 state["chunks"] 和 state["toc"]，保持引用一致
    """
    chunks = require_chunks(state)
    collection_name = state.get("collection_name") or milvus_gateway.chunks_collection
    vector_dim = int(state.get("vector_dim") or MILVUS_VECTOR_DIM)
    vector_target = None
    # 真实持久 Worker state 必须通过 SQL placement + Milvus 物理事实双门禁。
    # 不带 job_id 的历史离线 helper 保持兼容，但不能作为生产导入入口。
    if state.get("job_id"):
        vector_target = resolve_import_vector_target(
            state,
            fallback_collection=milvus_gateway.chunks_collection,
        )
        collection_name = vector_target.physical_collection
    if vector_target is not None and vector_target.managed:
        prepare_chunks_collection(
            collection_name,
            vector_dim,
            schema_version=str(state.get("schema_version") or "chunks_v1"),
            embedding_model_id=str(state.get("embedding_model_id") or "bge-m3"),
            managed_profile=vector_target.expected_profile,
        )
    else:
        prepare_chunks_collection(collection_name, vector_dim)
    if vector_target is not None:
        gate_result = MilvusWriteCompatibilityGate(
            milvus_gateway.client(),
            configured_database="default",
        ).verify(
            vector_target,
            vector_dim=vector_dim,
            schema_version=str(state.get("schema_version") or "chunks_v1"),
            embedding_model_id=str(state.get("embedding_model_id") or "bge-m3"),
        )
        state["vector_placement_id"] = vector_target.placement_id
        state["vector_placement_epoch"] = vector_target.placement_epoch
        state["vector_compatibility_fingerprint"] = (
            gate_result.compatibility_fingerprint.compatibility
        )
        state["vector_legacy_policy"] = gate_result.legacy_policy.value
        state["collection_name"] = collection_name

    milvus_chunks = _build_milvus_chunks(state)

    # 过滤无稀疏向量的 chunk：Milvus 2.4 SPARSE_FLOAT_VECTOR 不支持空值/Nullable。
    # 这些 chunk 不参与向量检索，但 content 已被前端/上层使用——只跳过 Milvus 写入，不修改 state。
    original_count = len(milvus_chunks)
    milvus_chunks_filtered = [c for c in milvus_chunks if c.get("sparse_vector")]
    skipped = original_count - len(milvus_chunks_filtered)
    if skipped:
        logger.warning(
            f"[index_chunks] 过滤 {skipped}/{original_count} 个稀疏向量缺失的 chunk "
            f"（hyperlinks / 嵌入失败批，Milvus 2.4 SPARSE_FLOAT_VECTOR 不支持空值）"
        )

    document_id = str(state.get("document_id") or state.get("task_id", ""))
    kb_id = str(state.get("kb_id", ""))
    tenant_id = str(state.get("tenant_id", "tenant_default"))
    document_version_id = str(state.get("document_version_id", ""))
    index_version_id = str(state.get("index_version_id", ""))
    file_title = str(state.get("file_title", ""))
    validation_errors: list[str] = []
    deterministic_ids = [str(row.get("id") or "") for row in milvus_chunks_filtered]
    evidence_ids = [str(row.get("evidence_id") or "") for row in milvus_chunks_filtered]
    evidence_hashes = [str(row.get("content_hash") or "") for row in milvus_chunks_filtered]
    source_chunk_ids = [str(row.get("source_chunk_id") or "") for row in milvus_chunks_filtered]
    if not all((tenant_id, kb_id, document_id, document_version_id, index_version_id)):
        validation_errors.append(
            "missing tenant/kb/document/document_version/index_version identity"
        )
    if len(set(deterministic_ids)) != len(deterministic_ids) or not all(deterministic_ids):
        validation_errors.append("chunk deterministic ids are empty or duplicated")
    if len(set(evidence_ids)) != len(evidence_ids) or not all(evidence_ids):
        validation_errors.append("evidence ids are empty or duplicated")
    if len(set(source_chunk_ids)) != len(source_chunk_ids) or not all(source_chunk_ids):
        validation_errors.append("source chunk ids are empty or duplicated")
    if not all(value.startswith("sha256:") and len(value) == 71 for value in evidence_hashes):
        validation_errors.append("evidence content hashes are missing or malformed")
    wrong_dimensions = [
        idx
        for idx, row in enumerate(milvus_chunks_filtered)
        if len(row.get("dense_vector") or []) != vector_dim
    ]
    if wrong_dimensions:
        validation_errors.append(
            f"dense vector dimension mismatch at positions {wrong_dimensions[:10]}"
        )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    client = milvus_gateway.client()

    def _quoted(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    staging_filter = " and ".join(
        (
            f"tenant_id == {_quoted(tenant_id)}",
            f"kb_id == {_quoted(kb_id)}",
            f"doc_id == {_quoted(document_id)}",
            f"document_version_id == {_quoted(document_version_id)}",
            f"index_version_id == {_quoted(index_version_id)}",
        )
    )
    # 只清理同一个 staging 身份的上次部分写入；绝不删除 active document/version。
    delete_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "filter": staging_filter,
    }
    partition_name = vector_target.partition_bucket if vector_target is not None else None
    if partition_name is not None:
        delete_kwargs["partition_name"] = partition_name
    client.delete(**delete_kwargs)
    chunk_ids = insert_chunks(
        milvus_chunks_filtered,
        collection_name,
        partition_name=partition_name,
    )
    written_count = len(chunk_ids)
    rows = client.query(
        collection_name=collection_name,
        filter=staging_filter,
        output_fields=[
            "chunk_id",
            "id",
            "img_id",
            "document_version_id",
            "index_version_id",
            "evidence_id",
            "content_hash",
            "source_chunk_id",
            "location_status",
            "primary_page_no",
        ],
        limit=max(1, len(milvus_chunks_filtered) + 10),
    )
    searchable_ids = {str(row.get("id") or "") for row in rows}
    searchable_evidence = {
        str(row.get("evidence_id") or ""): str(row.get("content_hash") or "") for row in rows
    }
    expected_evidence = dict(zip(evidence_ids, evidence_hashes, strict=True))
    if written_count != len(milvus_chunks_filtered):
        validation_errors.append(
            f"written count mismatch: expected={len(milvus_chunks_filtered)}, written={written_count}"
        )
    if searchable_ids != set(deterministic_ids):
        validation_errors.append(
            f"searchable id mismatch: expected={len(set(deterministic_ids))}, searchable={len(searchable_ids)}"
        )
    if searchable_evidence != expected_evidence:
        validation_errors.append(
            "searchable evidence/hash manifest mismatch: "
            f"expected={len(expected_evidence)}, searchable={len(searchable_evidence)}"
        )

    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    # 完整 Evidence 在 Milvus 可检索 manifest 通过后才原子写入控制库。任何一个
    # binding/不可变事实冲突都会让整批事务回滚并使当前 ingestion job 失败；此时
    # index version 仍是 staging，已有 active 版本不受影响，重试会先清理上述向量。
    evidences = [
        _build_local_evidence(
            chunk,
            doc_id=document_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            document_version_id=document_version_id,
            index_version_id=index_version_id,
            file_title=file_title,
            position=position,
        )
        for position, chunk in enumerate(chunks)
        if chunk.get("sparse_vector")
    ]
    persisted_evidence = record_evidence_batch(evidences, tenant_id=tenant_id)
    persisted_evidence_ids = {
        str(value.get("evidence_id") or value.get("id") or "") for value in persisted_evidence
    }
    if persisted_evidence_ids != set(evidence_ids):
        raise ValueError(
            "persisted evidence manifest mismatch: "
            f"expected={len(set(evidence_ids))}, persisted={len(persisted_evidence_ids)}"
        )

    # 查询期元数据过滤只读 SQL 控制目录，禁止再扫描最多 16384 个 Milvus chunk。
    # 目录与当前 document/index version 同步替换；写入失败即让 staging job 失败，
    # 不会发布一个“向量存在但元数据过滤不可复现”的半成品版本。
    catalog_metadata: dict[str, list[Any]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for raw_key, raw_value in metadata.items():
            key = str(raw_key)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            bucket = catalog_metadata.setdefault(key, [])
            for value in values:
                if value is not None and value not in bucket:
                    bucket.append(value)
    compact_metadata: dict[str, Any] = {
        key: values[0] if len(values) == 1 else values
        for key, values in catalog_metadata.items()
        if values
    }
    catalog_count = 0
    # 正式 Worker state 始终带 job_id；空元数据时也要清除同一 staging scope
    # 的旧目录。轻量单元测试没有控制库事实且不带 job_id，因此只验证向量/Evidence。
    if compact_metadata or state.get("job_id"):
        with session_scope() as session:
            catalog_count = RetrievalProfileRepository(
                session,
                tenant_id=tenant_id,
                actor_id="ingestion-worker",
            ).replace_metadata_entries(
                kb_id,
                document_id,
                document_version_id,
                index_version_id,
                compact_metadata,
            )

    # Milvus auto PK 只回填到 vector_pk，绝不覆盖 TODO-07 的稳定 chunk_id。
    # 按“是否带 sparse_vector”分别消费 PK/Evidence，避免过滤造成 zip 错位。
    written_ids = iter(chunk_ids)
    written_evidence = iter(evidences)
    for position, chunk in enumerate(chunks):
        if chunk.get("sparse_vector"):
            try:
                vector_pk = next(written_ids)
                evidence = next(written_evidence)
                chunk["chunk_id"] = evidence.source.chunk_id
                chunk["source_chunk_id"] = evidence.source.chunk_id
                chunk["vector_pk"] = vector_pk
                chunk["evidence_id"] = evidence.evidence_id
                chunk["content_hash"] = evidence.content_hash
            except StopIteration:
                logger.warning("[index_chunks] vector_pk/Evidence 不足，可能存在未预期的过滤")
        else:
            chunk["vector_pk"] = None

    # TOC 没有 sparse vector，当前 Milvus schema 不允许空稀疏向量，因此它与
    # hyperlinks/嵌入失败块一样不会进入本次 searchable manifest。不能伪造一个
    # 自增 chunk_id，否则 job complete 会把未写入 TOC 计入并造成 manifest mismatch。
    toc = state.get("toc")
    if toc:
        toc["chunk_id"] = None
        toc["vector_pk"] = None

    state["chunks"] = chunks
    # cleanup 必须记录 MinIO 的真实 object_name，而不是 Milvus 中的 img_id。
    object_keys = sorted(set(state.get("uploaded_object_keys") or []))
    state["index_manifest"] = {
        "document_version_id": document_version_id,
        "index_version_id": index_version_id,
        "expected_chunk_count": len(milvus_chunks_filtered),
        "written_chunk_count": written_count,
        "searchable_chunk_count": len(searchable_ids),
        "chunk_ids": deterministic_ids,
        "chunk_ids_sha256": hashlib.sha256(
            "\n".join(sorted(deterministic_ids)).encode("utf-8")
        ).hexdigest(),
        "evidence_count": len(evidence_ids),
        "evidence_ids": evidence_ids,
        "evidence_ids_sha256": hashlib.sha256(
            "\n".join(sorted(evidence_ids)).encode("utf-8")
        ).hexdigest(),
        "evidence_content_manifest_sha256": hashlib.sha256(
            "\n".join(
                f"{evidence_id}:{expected_evidence[evidence_id]}"
                for evidence_id in sorted(expected_evidence)
            ).encode("utf-8")
        ).hexdigest(),
        "object_keys": object_keys,
        "vector_dim": vector_dim,
        "collection_name": collection_name,
        "partition_name": partition_name,
        "vector_placement_id": state.get("vector_placement_id"),
        "vector_placement_epoch": state.get("vector_placement_epoch"),
        "vector_compatibility_fingerprint": state.get("vector_compatibility_fingerprint"),
        "vector_legacy_policy": state.get("vector_legacy_policy"),
        "metadata_catalog_entry_count": catalog_count,
        "validation_status": "validated" if not validation_errors else "failed",
        "validation_errors": validation_errors,
    }
    logger.info(
        f"Milvus 入库完成：{len(milvus_chunks_filtered)} 个 chunk / "
        f"{len(evidence_ids)} 条 Evidence 入库 + "
        f"{skipped} 个跳过（含无稀疏向量的 TOC/超链接）"
    )
    return state
