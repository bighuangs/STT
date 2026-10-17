"""
Chunk 向量化服务模块。

将文本切片批量编码为稠密向量 + 稀疏向量，对齐 RAGFlow rag/svr/task_executor.py:1135
embedding() 的设计理念（融合向量）。

关键流程：
    1. 文本截断：每条文本按 max_chars 粗截断，防止 BGE-M3 上下文溢出。
    2. 批次处理：EMBEDDING_BATCH_SIZE 控制每批切片数量，平衡显存与吞吐量。
    3. 异常隔离：批次级 try-except，单批失败不影响其他批次。
    4. 标题-内容加权融合：与 ragflow task_executor.py:1235-1250 对齐
       - 标题向量一次编码后 tile 复用
       - 融合：final = title_w * title_vec + (1 - title_w) * content_vec
       - title_w = parser_config["filename_embd_weight"]（默认 0.1）
    5. 文本优先级：与 ragflow task_executor.py:1170-1173 对齐
       - 优先 "\n".join(question_kwd)，空时回退 content（互斥）
    6. HTML 表格标签清洗：与 ragflow task_executor.py:1176 对齐
       - 去除 <table> <tr> <td> <th> <caption> 等标签，避免干扰语义
    7. 空白占位：与 ragflow task_executor.py:1180-1182 对齐
       - 空字符串用 "None" 占位，防止嵌入模型退化为零向量
"""
import re
from collections.abc import Callable

import numpy as np

from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import EMBEDDING_BATCH_SIZE


# BGE-M3 模型单次编码的文本最大长度（字符数，安全裕度）
# STT 与 ragflow 差异：ragflow 按 mdl.max_length（token）截断，STT 按字符截断避免引入额外 tokenizer
# 8192 是 BGE-M3 原生上限，截断阈值留足余量
# 该值为兜底常量，实际由 embed_chunks 从 parser_config["embedding_max_chars"] 读取
_EMBEDDING_MAX_CHARS = 8192

# 标题向量在最终融合向量中的权重（与 ragflow task_executor.py:1240-1243 对齐）
# 兜底常量；实际由 _get_title_weight 从 parser_config["filename_embd_weight"] 读取
_DEFAULT_TITLE_EMBD_WEIGHT = 0.1

# HTML 表格标签正则（与 ragflow task_executor.py:1176 一致）
_HTML_TABLE_TAG_RE = re.compile(
    r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>",
    re.IGNORECASE,
)

# 空白块占位符（与 ragflow task_executor.py:1182 一致）
_EMPTY_PLACEHOLDER = "None"


def _truncate_text(text: str, max_chars: int) -> str:
    """
    按字符数截断文本，超长部分截断。

    STT 简化：ragflow 用 mdl.max_length（token）截断，STT 用 max_chars（字符）粗截断。
    BGE-M3 单条文本上限 8192 token，字符阈值留足余量。

    :param text: 待截断文本
    :param max_chars: 最大字符数
    :return: 截断后的文本
    """
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _build_embedding_text(chunk: dict) -> str:
    """
    构造 content 视角的嵌入输入文本（STT 独有设计，不再与 ragflow task_executor.py:1166-1183 对齐）。

    STT 设计：dense/sparse_vector 用纯 content 嵌入，question_* 字段独占 LLM 假设问句。
    两路输入严格分离，各自独立检索语义空间。

    要点：
        1. 直接取 content，不走 question_kwd 优先（避免与 question_dense_vector 撞输入）
        2. 清洗 HTML 表格标签
        3. 空白占位（防零向量）
        4. 不做 title 融合（融合在 embed_chunks 主流程中统一处理）

    :param chunk: 单个 chunk 字典
    :return: 嵌入输入文本
    """
    # content 是 STT chunk 的核心语义载体（mineru 解析后的 Markdown 正文）
    text = chunk.get("content") or ""

    # 清洗 HTML 表格标签
    text = _HTML_TABLE_TAG_RE.sub(" ", text)

    # 空白占位
    if not text.strip():
        text = _EMPTY_PLACEHOLDER
    return text


def _get_title_weight(parser_config: dict) -> float:
    """
    读取标题向量融合权重（与 ragflow task_executor.py:1240-1243 对齐）。

    字段：parser_config["filename_embd_weight"]，默认 0.1
    """
    raw = parser_config.get("filename_embd_weight", _DEFAULT_TITLE_EMBD_WEIGHT)
    try:
        w = float(raw)
    except (TypeError, ValueError):
        w = _DEFAULT_TITLE_EMBD_WEIGHT
    # 与 ragflow 行为一致：0/缺失/无效均退化为默认值
    if not w:
        w = _DEFAULT_TITLE_EMBD_WEIGHT
    return w


@step_log("embed_chunks")
def embed_chunks(
    chunks: list[dict],
    file_title: str = "",
    parser_config: dict | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    """
    为每个非 hyperlinks chunk 追加稠密向量和稀疏向量（STT 独有设计，不再与 ragflow 对齐）。

    设计要点（STT 独有）：
        - dense_vector/sparse_vector：纯 content 视角嵌入（带 title 融合）
        - question_dense_vector/question_sparse_vector：LLM 假设问句视角嵌入（独立，不融合 title）
        - 两路输入严格分离，各自独立检索语义空间

    流程：
        1. 过滤掉 type=hyperlinks 的 chunk（无需嵌入）
        2. 构造嵌入输入文本：纯 content（与 ragflow 不同，STT 独有设计）
        3. 清洗 HTML 表格标签 + 空白占位
        4. 截断到 max_chars 防止越界
        5. 编码一次标题向量（tile 复用）
        6. 分批编码内容向量
        7. 加权融合稠密向量：final = title_w * title_vec + (1 - title_w) * content_vec
        8. 为每个 chunk 追加 dense_vector / sparse_vector 字段
        9. 批次级异常隔离，单批失败不影响其他批次

    :param chunks: 文档切片列表，每个元素为 dict（至少含 content 字段）
    :param file_title: 文档标题（对齐 ragflow docnm_kwd）
    :param parser_config: 解析器配置（含 filename_embd_weight 标题向量权重等）
    :param batch_size: 每批切片数量，默认由 EMBEDDING_BATCH_SIZE 控制
    :param progress_callback: 每个内容批次结束后的 ``completed/total/failed`` 回调
    :return: 带向量字段的切片列表
    """
    if parser_config is None:
        parser_config = {}

    # 跳过 hyperlinks（其 content 为 URL 列表，无需嵌入）
    regular_chunks = [c for c in chunks if c.get("type") != "hyperlinks"]
    hl_skipped = len(chunks) - len(regular_chunks)
    if not regular_chunks:
        logger.info(
            f"[embed_chunks] 无需嵌入（chunks={len(chunks)}，全部为 hyperlinks）"
        )
        return chunks

    title_weight = _get_title_weight(parser_config)
    safe_title = file_title or "标题"  # 与 ragflow task_executor.py:1168 占位一致
    # 文本截断上限：优先从 parser_config["embedding_max_chars"] 读取，未设置时用兜底常量
    max_chars = int(parser_config.get("embedding_max_chars", _EMBEDDING_MAX_CHARS))

    # -------------------------------------------------------------------------
    # 步骤 1：构造所有 chunk 的嵌入输入文本
    # -------------------------------------------------------------------------
    texts = [
        _truncate_text(_build_embedding_text(c), max_chars)
        for c in regular_chunks
    ]

    # -------------------------------------------------------------------------
    # 步骤 2：编码标题向量（一次，tile 复用）
    # -------------------------------------------------------------------------
    # 与 ragflow task_executor.py:1191-1196 对齐：所有块共享同一标题向量
    title_vec: np.ndarray | None = None
    try:
        title_result = llm_provider.embed_documents([safe_title])
        if title_result.get("dense"):
            title_vec = np.asarray(title_result["dense"][0], dtype=np.float32)
    except Exception as e:
        logger.warning(
            f"[embed_chunks] 标题向量编码失败，将跳过标题加权: {e}",
            exc_info=True,
        )

    # -------------------------------------------------------------------------
    # 步骤 3：分批编码内容向量
    # -------------------------------------------------------------------------
    dense_vecs: list[np.ndarray] = []
    sparse_vecs: list[dict] = []
    failed_batches: list[int] = []

    for batch_idx in range(0, len(texts), batch_size):
        step_texts = texts[batch_idx:batch_idx + batch_size]
        try:
            result = llm_provider.embed_documents(step_texts)
            batch_dense = [np.asarray(v, dtype=np.float32) for v in result["dense"]]
            batch_sparse = result["sparse"]
            if len(batch_dense) != len(step_texts) or len(batch_sparse) != len(step_texts):
                raise ValueError(
                    f"批次 {batch_idx} 向量数量与文本数量不匹配: "
                    f"dense={len(batch_dense)}, sparse={len(batch_sparse)}, texts={len(step_texts)}"
                )
            dense_vecs.extend(batch_dense)
            sparse_vecs.extend(batch_sparse)
        except Exception as e:
            logger.warning(
                f"[embed_chunks] 向量化批次 [{batch_idx}:{batch_idx + batch_size}] "
                f"生成失败，{len(step_texts)} 个 chunk 将无向量: {e}",
                exc_info=True,
            )
            failed_batches.append(batch_idx)
            # 占位：保持索引对齐（失败的 chunk dense/sparse 都置空）
            for _ in step_texts:
                dense_vecs.append(np.zeros(0, dtype=np.float32))
                sparse_vecs.append({})
        if progress_callback is not None:
            progress_callback(
                min(batch_idx + len(step_texts), len(texts)),
                len(texts),
                len(failed_batches),
            )

    if failed_batches:
        logger.warning(
            f"[embed_chunks] 共有 {len(failed_batches)} 个批次失败，受影响 chunk 数 ≈ "
            f"{len(failed_batches) * batch_size}"
        )

    # -------------------------------------------------------------------------
    # 步骤 4：标题-内容加权融合（仅稠密向量）
    # -------------------------------------------------------------------------
    # 与 ragflow task_executor.py:1238-1250 对齐
    # 融合后重新 L2 归一化，以保持 Milvus IP 内积检索的正确性
    if title_vec is not None and title_vec.ndim == 1 and title_vec.size > 0:
        try:
            for i, dv in enumerate(dense_vecs):
                if dv.size == 0:
                    continue
                if dv.shape != title_vec.shape:
                    # 维度不一致时退化为仅使用内容向量（与 ragflow 行为一致）
                    logger.debug(
                        f"[embed_chunks] chunk[{i}] dense 维度 {dv.shape} 与标题维度 {title_vec.shape} 不一致，跳过融合"
                    )
                    continue
                fused = title_weight * title_vec + (1.0 - title_weight) * dv
                # 重新 L2 归一化（保持 Milvus IP 内积检索语义）
                norm = float(np.linalg.norm(fused))
                if norm > 0:
                    fused = fused / norm
                dense_vecs[i] = fused.astype(np.float32)
        except Exception as e:
            logger.warning(
                f"[embed_chunks] 标题-内容向量融合失败，退化为仅内容向量: {e}",
                exc_info=True,
            )

    # -------------------------------------------------------------------------
    # 步骤 5：绑定到每个 chunk
    # -------------------------------------------------------------------------
    success_count = 0
    for chunk, dv, sv in zip(regular_chunks, dense_vecs, sparse_vecs):
        if dv.size == 0 or not sv:
            # 失败的 chunk：跳过字段绑定（保持 chunks 字段缺失，与现有 node_import_milvus 行为一致）
            continue
        chunk["dense_vector"] = dv.tolist()
        chunk["sparse_vector"] = sv
        success_count += 1

    logger.info(
        f"[embed_chunks] 向量化完成：输入 {len(chunks)} 个切片，跳过 hyperlinks={hl_skipped}，"
        f"实际嵌入 {success_count}/{len(regular_chunks)} 个（融合语义），"
        f"title_weight={title_weight}，file_title={safe_title[:30]!r}"
    )
    return chunks
