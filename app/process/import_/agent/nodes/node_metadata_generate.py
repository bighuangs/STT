
"""
节点: 文档元数据生成 (node_metadata_generate)

并行分支之一（详见 node_keyword_extraction.py 头部的拓扑图）。

参考 ragflow rag/svr/task_executor.py:gen_metadata_task (L719-856)

核心功能：遍历 state["chunks"]，根据 parser_config 中的元数据 Schema 配置，
    调用 LLM 从每个 chunk 中抽取结构化元数据，结果写入 chunk["metadata"]: dict。
"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.metadata_generate_service import generate_metadata
@node_log("node_metadata_generate")
def node_metadata_generate(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 元数据生成 (node_metadata_generate)
    调用 metadata_generate_service.generate_metadata，根据 parser_config 配置
    为每个 chunk 抽取结构化元数据，写入 chunk["metadata"]: dict。
    """
    add_running_task(state["task_id"], "node_metadata_generate")
    state = generate_metadata(state)
    add_done_task(state["task_id"], "node_metadata_generate")
    # 上报增量给 llm_enhance_reducer 显式合并（不依赖共享引用）；
    # 仍返回完整 state，保证本文件 __main__ 独立联调可直接读 state["chunks"]
    state["llm_enhance_results"] = [{"chunks": state.get("chunks", []), "toc": state.get("toc")}]
    return state

if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_metadata_generate
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → node_metadata_generate（5 个 LLM 增强之一）
    #
    # 与 node_question_proposal 的关键差异：
    #   - 需在 parser_config 中注入 enable_metadata=True + metadata 字段配置
    #   - 支持 list[dict]（字段列表）和 dict（JSON Schema）两种配置格式
    #   - 成功时每个 chunk["metadata"] 为 dict（字段由 Schema 决定）
    # ============================================================
    import os
    import sys
    from pathlib import Path

    from app.shared.runtime.logger import logger, PROJECT_ROOT
    from app.process.import_.agent.state import create_default_state
    from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
    from app.rag.import_.metadata_extract_service import extract_document_metadata
    from app.rag.import_.markdown_image_service import enrich_markdown_images
    from app.process.import_.agent.nodes.node_link_and_table_chunk import node_link_and_table_chunk
    from app.process.import_.agent.nodes.node_chunk_merge import node_chunk_merge
    from app.process.import_.agent.nodes.node_image_upload import node_image_upload

    logger.info("===== 开始 node_metadata_generate 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_metadata_generate_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    # ============ 阶段 1: PDF → MD（联网 MinerU 解析） ============
    state = parse_pdf_to_markdown(init_state)
    logger.info(f"[阶段1] PDF 转 MD 完成，md_path: {state['md_path']}")

    # ============ 阶段 2: 切分（metadata_extract） ============
    state = extract_document_metadata(state)
    logger.info(f"[阶段2] 切分完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 3: VLM 图片增强（node_md_img） ============
    state = enrich_markdown_images(state)
    logger.info(
        f"[阶段3] VLM 增强完成，加载图片: {len(state.get('image_pil_dict', {}))} 张"
    )

    # ============ 阶段 4: 链接/表格拆分（node_link_and_table_chunk） ============
    state = node_link_and_table_chunk(state)
    logger.info(f"[阶段4] 链接/表格拆分完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 5: Chunk 合并（node_chunk_merge） ============
    state = node_chunk_merge(state)
    logger.info(f"[阶段5] 合并完成，chunks 总数: {len(state.get('chunks', []))}")

    # ============ 阶段 6: MinIO 上传（node_image_upload） ============
    state = node_image_upload(state)
    logger.info("[阶段6] MinIO 上传完成")

    # ============ 注入元数据配置 ============
    # 对齐 ragflow parser_config.enable_metadata + parser_config.metadata
    # 测试字段列表格式（list[dict]），支持 key / description / enum
    state["parser_config"] = {
        **state.get("parser_config", {}),
        "enable_metadata": True,
        # list[dict] 格式：字段列表（list 会被 _turn2jsonschema 转为 JSON Schema）
        "metadata": [
            {
                "key": "product_name",
                "description": "产品名称，如华为 MateBook X Pro",
            },
            {
                "key": "language",
                "description": "文档语言",
                "enum": ["中文", "英文", "中英双语"],
            },
            {
                "key": "document_type",
                "description": "文档类型，如用户指南、快速入门、技术规格",
            },
        ],
    }
    logger.info(
        f"[注入] enable_metadata=True，metadata 字段: {[f['key'] for f in state['parser_config']['metadata']]}"
    )

    # ============ 阶段 7: 元数据生成（node_metadata_generate，**本节点**） ============
    pre_chunks = list(state["chunks"])  # 关键：保留原始引用用于断言比对
    result = node_metadata_generate(state)
    chunks = result.get("chunks", [])

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(chunks) == len(pre_chunks), (
        f"chunks 数量应保持不变: {len(pre_chunks)} -> {len(chunks)}"
    )

    # ========== 断言 2：所有 chunk 必含 metadata 字段 ==========
    missing_meta = [i for i, c in enumerate(chunks) if "metadata" not in c]
    assert not missing_meta, f"chunks 缺少 metadata: {missing_meta}"

    # ========== 断言 3：metadata 必为 dict 类型 ==========
    bad_meta_types = [
        (i, type(c.get("metadata")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("metadata"), dict)
    ]
    assert not bad_meta_types, f"metadata 类型错误: {bad_meta_types}"

    # ========== 断言 4：每个 chunk 的 metadata 字段键一致（来自同一 Schema） ==========
    meta_keys = set().union(*[c.get("metadata", {}).keys() for c in chunks]) if chunks else set()
    expected_keys = {"product_name", "language", "document_type"}
    logger.info(f"[诊断] 实际 metadata 字段键集合: {meta_keys}")
    # 只要有 chunk 成功提取了 Schema 字段即可（非空 metadata）
    non_empty = [c for c in chunks if c.get("metadata")]
    logger.info(
        f"元数据摘要: {len(non_empty)}/{len(chunks)} 个 chunk 成功生成元数据"
    )

    # ========== 诊断输出 ==========
    for i, c in enumerate(chunks):
        meta = c.get("metadata") or {}
        type_ = c.get("type", "text")
        meta_str = ", ".join(f"{k}={v!r}" for k, v in meta.items()) if meta else "(空)"
        logger.info(
            f"  chunk[{i}] type={type_} | metadata={meta_str}"
        )

    logger.info("===== node_metadata_generate 节点联调测试完成 =====")
