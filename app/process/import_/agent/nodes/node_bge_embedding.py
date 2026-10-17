"""
节点: 向量化 (node_bge_embedding)

并行分支汇聚点（5 个 LLM 增强节点共同汇入此节点）：

    node_image_upload ─┬─▶ node_keyword_extraction  ─┐
                       ├─▶ node_question_proposal   ─┤
                       ├─▶ node_metadata_generate   ─┤
                       ├─▶ node_content_tagging     ─┼─▶ node_bge_embedding ─▶ node_import_milvus
                       └─▶ node_toc                 ─┘

参考 ragflow rag/svr/task_executor.py:embedding (L1135-1262)

职责：使用 BGE-M3 模型为每个文本块追加稠密向量 + 稀疏向量。
向量化在原 chunks 上原地追加字段，无需新建列表。

与 ragflow 流程的关键差异（与 embedding_service 注释对齐）：
    - LLM 输入字段：ragflow 用 chunk["content_with_weight"]，STT 用 chunk["question_kwd"] 优先 / chunk["content"] 回退
    - 标题字段：ragflow 用 chunk["docnm_kwd"]，STT 用 state["file_title"]（单文档共享）
    - 融合：仅对稠密向量做加权融合，稀疏向量直接使用
    - HTML 表格标签清洗：去除 <table> <tr> <td> 等
    - 空白占位：空文本用 "None" 防零向量
    - 文本截断：ragflow 按 mdl.max_length（token）截断，STT 按 max_chars（字符）截断
"""
import os
from pathlib import Path

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.embedding_service import embed_chunks
from app.application.knowledge_workbench_service import (
    apply_chunk_override_for_worker,
)
from app.application.ingestion_progress_hooks import (
    notify_ingestion_embedding_batch_progress,
)


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)

    职责：使用 BGE-M3 模型为每个文本块追加稠密向量 + 稀疏向量。
    向量化在原 chunks 上原地追加字段，无需新建列表。

    与 ragflow embedding() 的字段映射：
        - ragflow docnm_kwd（每块独立） → STT state["file_title"]（单文档共享）
        - ragflow content_with_weight → STT chunk["question_kwd"] 优先 / chunk["content"] 回退
        - ragflow parser_config["filename_embd_weight"] → STT 同名（默认 0.1）

    行为：
        - chunks 为空：安全跳过，记录警告
        - 单批失败：该批 chunk 跳过向量绑定，其他批次继续（与 ragflow 一致）
        - 标题向量编码失败：退化为仅内容向量（融合步骤跳过）
    """
    add_running_task(state["task_id"], "node_bge_embedding")

    chunks = apply_chunk_override_for_worker(state, state.get("chunks", []))
    state["chunks"] = chunks
    if not chunks:
        logger.warning("[node_bge_embedding] chunks 为空，跳过向量化")
    else:
        # 从 state 上下文提取配置（与 ragflow task_executor.py:1149-1154 对齐）
        file_title = state.get("file_title", "") or ""
        parser_config = state.get("parser_config") or {}
        state["chunks"] = embed_chunks(
            chunks=chunks,
            file_title=file_title,
            parser_config=parser_config,
            progress_callback=notify_ingestion_embedding_batch_progress,
        )

    add_done_task(state["task_id"], "node_bge_embedding")
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_bge_embedding
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → 5 个 LLM 增强（并行） → node_bge_embedding（**本节点**）
    #
    # 关键校验：
    #   - 所有非 hyperlinks chunk 必含 dense_vector / sparse_vector
    #   - 稠密向量维度 == MILVUS_VECTOR_DIM
    #   - 稀疏向量类型为 dict
    #   - 前序 LLM 节点写入的字段（key_words / key_tokens / question_kwd / question_tks）不被破坏
    #   - 验证 file_title + filename_embd_weight 融合（取首条 chunk 打印前 3 维向量）
    # ============================================================
    import os

    from app.shared.runtime.logger import logger, PROJECT_ROOT
    from app.process.import_.agent.state import create_default_state
    from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
    from app.rag.import_.metadata_extract_service import extract_document_metadata
    from app.rag.import_.markdown_image_service import enrich_markdown_images
    from app.process.import_.agent.nodes.node_link_and_table_chunk import node_link_and_table_chunk
    from app.process.import_.agent.nodes.node_chunk_merge import node_chunk_merge
    from app.process.import_.agent.nodes.node_image_upload import node_image_upload
    from app.process.import_.agent.nodes.node_keyword_extraction import node_keyword_extraction
    from app.process.import_.agent.nodes.node_question_proposal import node_question_proposal
    from app.process.import_.agent.nodes.node_content_tagging import node_content_tagging
    from app.process.import_.agent.nodes.node_metadata_generate import node_metadata_generate
    from app.process.import_.agent.nodes.node_toc import node_toc
    from app.rag.import_.config import MILVUS_VECTOR_DIM

    logger.info("===== 开始 node_bge_embedding 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_bge_embedding_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )
    # 显式注入 file_title（与 ragflow docnm_kwd 对齐，对应 PDF 文件 stem）
    # 注：完整 LangGraph 链路下由 node_entry 节点填充，本 __main__ 跳过 entry，故手动注入
    init_state["file_title"] = Path(test_pdf_path).stem
    logger.info(f"[注入] file_title={init_state['file_title']!r}")

    # ============ 阶段 1: PDF → MD（联网 MinerU 解析） ============
    state = parse_pdf_to_markdown(init_state)
    logger.info(f"[阶段1] PDF 转 MD 完成，md_path: {state['md_path']}")

    # ============ 阶段 2: 切分（metadata_extract） ============
    state = extract_document_metadata(state)
    logger.info(f"[阶段2] 切分完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 3: VLM 图片增强（node_md_img） ============
    state = enrich_markdown_images(state)
    logger.info(f"[阶段3] VLM 增强完成，加载图片: {len(state.get('image_pil_dict', {}))} 张")

    # ============ 阶段 4: 链接/表格拆分 ============
    state = node_link_and_table_chunk(state)
    logger.info(f"[阶段4] 链接/表格拆分完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 5: Chunk 合并 ============
    state = node_chunk_merge(state)
    logger.info(f"[阶段5] 合并完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 6: MinIO 上传（image_upload） ============
    state = node_image_upload(state)
    logger.info("[阶段6] MinIO 上传完成")

    # ============ 阶段 7: 5 个 LLM 增强（并行汇聚） ============
    pre_bge_chunks_ref = list(state["chunks"])  # 关键：保留进入 bge 前的原始引用
    state = node_keyword_extraction(state)
    state = node_question_proposal(state)
    state = node_content_tagging(state)
    state = node_metadata_generate(state)
    state = node_toc(state)
    logger.info("[阶段7] 5 个 LLM 增强节点全部完成，准备向量化")

    # ============ 阶段 8: 向量化（**本节点**） ============
    result = node_bge_embedding(state)
    embedded_chunks = result.get("chunks", [])

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(embedded_chunks) == len(pre_bge_chunks_ref), (
        f"向量化后 chunks 数量应保持一致: {len(pre_bge_chunks_ref)} -> {len(embedded_chunks)}"
    )

    # ========== 断言 2：非 hyperlinks chunk 必含 dense_vector / sparse_vector ==========
    non_hl_chunks = [c for c in embedded_chunks if c.get("type") != "hyperlinks"]
    missing_dense = [i for i, c in enumerate(non_hl_chunks) if "dense_vector" not in c]
    missing_sparse = [i for i, c in enumerate(non_hl_chunks) if "sparse_vector" not in c]
    assert not missing_dense, f"非 hyperlinks chunk 缺少 dense_vector 的索引: {missing_dense}"
    assert not missing_sparse, f"非 hyperlinks chunk 缺少 sparse_vector 的索引: {missing_sparse}"
    assert non_hl_chunks, "应有非 hyperlinks 的 chunk 被向量化"

    # ========== 断言 3：hyperlinks chunks 不应携带向量 ==========
    hl_chunks_vec = [c for c in embedded_chunks if c.get("type") == "hyperlinks"]
    hl_with_vec = [
        i for i, c in enumerate(hl_chunks_vec)
        if any(k in c for k in ("dense_vector", "sparse_vector"))
    ]
    assert not hl_with_vec, f"hyperlinks chunk 不应携带向量，异常: {len(hl_with_vec)} 个"

    # ========== 断言 4：向量维度与类型 ==========
    wrong_dim = [i for i, c in enumerate(non_hl_chunks) if len(c["dense_vector"]) != MILVUS_VECTOR_DIM]
    not_dict = [i for i, c in enumerate(non_hl_chunks) if not isinstance(c["sparse_vector"], dict)]
    assert not wrong_dim, f"dense_vector 维度应为 {MILVUS_VECTOR_DIM}，异常 chunk: {wrong_dim[:5]}"
    assert not not_dict, f"sparse_vector 应为 dict，异常 chunk: {not_dict[:5]}"

    # ========== 断言 5：sparse_vector 权重不应全为 0 ==========
    empty_sparse = [i for i, c in enumerate(non_hl_chunks) if not c["sparse_vector"]]
    assert not empty_sparse, f"sparse_vector 不应为空 dict: {empty_sparse[:5]}"

    # ========== 断言 6：前序 LLM 节点字段未被破坏 ==========
    # 验证 5 个 LLM 节点的代表字段在向量化后仍保留
    assert all("key_words" in c for c in embedded_chunks), "bge 后应保留 key_words"
    assert all("key_tokens" in c for c in embedded_chunks), "bge 后应保留 key_tokens"
    assert all("question_kwd" in c for c in embedded_chunks), "bge 后应保留 question_kwd"
    assert all("question_tks" in c for c in embedded_chunks), "bge 后应保留 question_tks"

    # ========== 断言 7：验证 file_title + 融合对稠密向量确有影响 ==========
    # 融合后稠密向量的 L2 范数应 <= 纯内容向量的 L2 范数（凸组合性质）
    import math

    norms = []
    for c in non_hl_chunks[:5]:
        vec = c["dense_vector"]
        norm = math.sqrt(sum(x * x for x in vec))
        norms.append(norm)
    logger.info(f"[诊断] 前 5 条 non-hl chunk dense L2 范数: {[f'{n:.4f}' for n in norms]}")
    assert all(n > 0 for n in norms), "稠密向量 L2 范数应为正"

    # ========== 断言 7：验证 file_title 真的传到了 embed_chunks ==========
    # file_title 非空时，与默认占位符 "标题" 编码出的向量应不同
    if state.get("file_title"):
        from app.infra.llm import llm_provider
        title_emb = llm_provider.embed_documents([state["file_title"]])
        title_vec = title_emb["dense"][0]
        first_chunk_vec = non_hl_chunks[0]["dense_vector"]
        cos = sum(a * b for a, b in zip(first_chunk_vec, title_vec)) / (
            math.sqrt(sum(x * x for x in first_chunk_vec))
            * math.sqrt(sum(x * x for x in title_vec))
        )
        logger.info(
            f"[诊断] chunk[0] 与 file_title 向量余弦相似度 = {cos:.4f} "
            f"（应 > 0.5：title_weight=0.1 融合后 chunk 仍与 title 方向相关）"
        )
        assert cos > 0.5, (
            f"file_title={state['file_title']!r} 应与 chunk[0] 向量有较高相似度，"
            f"实际 cos={cos:.4f}（融合可能未生效）"
        )

    # ========== 诊断输出 ==========
    for i, c in enumerate(embedded_chunks):
        chunk_type = c.get("type", "text")
        dim = len(c.get("dense_vector", []))
        sparse_keys = len(c.get("sparse_vector", {}))
        logger.info(
            f"  chunk[{i}] type={chunk_type}, dense_dim={dim}, sparse_keys={sparse_keys}, "
            f"content_len={len(c.get('content', ''))}"
        )

    # 类型汇总
    hl = [c for c in embedded_chunks if c.get("type") == "hyperlinks"]
    tbl = [c for c in embedded_chunks if c.get("type") == "table"]
    txt = [c for c in embedded_chunks if c.get("type") not in ("hyperlinks", "table")]
    logger.info(
        f"向量化摘要: text={len(txt)}, table={len(tbl)}, hyperlinks={len(hl)}（hyperlinks 无向量）"
    )
    logger.info(f"file_title={state.get('file_title', '')!r}")
    logger.info(
        f"filename_embd_weight="
        f"{result.get('parser_config', {}).get('filename_embd_weight', 0.1)}"
    )
    logger.info("===== node_bge_embedding 节点联调测试完成 =====")
