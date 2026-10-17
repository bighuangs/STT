"""
节点: 文档目录构建 (node_toc)

并行分支之一（与 4 个 LLM 增强节点并列）：

    node_image_upload ─┬─▶ node_keyword_extraction  ─┐
                       ├─▶ node_question_proposal   ─┤
                       ├─▶ node_metadata_generate   ─┤
                       ├─▶ node_content_tagging     ─┼─▶ node_bge_embedding ─▶ node_import_milvus
                       └─▶ node_toc                 ─┘

参考 ragflow rag/svr/task_executor.py:build_TOC (L1009-1113)
       ragflow rag/svr/task_executor.py:2333（异步触发入口）

与 4 个 LLM 增强节点的关键区别：
    - 输出粒度：整篇文档 1 份 TOC（写在 state["toc"]），不是逐 chunk 写字段
    - 不会修改 state["chunks"]，TOC 块与普通块解耦
    - 下游 node_bge_embedding → node_import_milvus 需自行读取 state["toc"] 决定是否入库
    - 触发开关：parser_config["toc_extraction"]，未启用时跳过整个流程

实现说明：
    - 调用 toc_service.build_toc(state)
    - 写入 state["toc"]: dict（与 ragflow build_TOC 块元数据对齐）
    - toc_extraction=False 时安全跳过，state["toc"] 保持空 dict
"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.toc_service import build_toc


@node_log("node_toc")
def node_toc(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 目录构建 (node_toc)

    遍历 state["chunks"]，根据 parser_config["toc_extraction"] 配置，
    调用 LLM 提取整篇文档的目录结构，写入 state["toc"]: dict。

    parser_config 配置字段：
        - toc_extraction: bool — 是否启用目录提取（与 ragflow parser_config.toc_extraction 对齐）

    行为：
        - toc_extraction=False：安全跳过，state["toc"] 保持空 dict
        - LLM 调用失败：安全跳过，state["toc"] 保持空 dict
        - 无 chunks：安全跳过，state["toc"] 保持空 dict
    """
    add_running_task(state["task_id"], "node_toc")
    state = build_toc(state)
    add_done_task(state["task_id"], "node_toc")
    # 上报增量给 llm_enhance_reducer 显式合并（不依赖共享引用）；
    # node_toc 同时会给 chunks 写入 id，因此 chunks 也需上报
    state["llm_enhance_results"] = [{"chunks": state.get("chunks", []), "toc": state.get("toc")}]
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_toc
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → node_toc（5 个 LLM 增强之一）
    #
    # 与 node_content_tagging 的关键差异：
    #   - 需在 parser_config 中注入 toc_extraction=True 启用 TOC 生成
    #   - 成功时 state["toc"] 为 dict（含 items / content_with_weight / toc_kwd 等）
    #   - chunks 数量与字段保持不变，TOC 与 chunks 解耦
    # ============================================================
    import os
    import json

    from app.shared.runtime.logger import logger, PROJECT_ROOT
    from app.process.import_.agent.state import create_default_state
    from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
    from app.rag.import_.metadata_extract_service import extract_document_metadata
    from app.rag.import_.markdown_image_service import enrich_markdown_images
    from app.process.import_.agent.nodes.node_link_and_table_chunk import node_link_and_table_chunk
    from app.process.import_.agent.nodes.node_chunk_merge import node_chunk_merge
    from app.process.import_.agent.nodes.node_image_upload import node_image_upload

    logger.info("===== 开始 node_toc 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_toc_task_001",
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

    # ============ 注入 TOC 配置 ============
    # 对齐 ragflow parser_config.toc_extraction
    state["parser_config"] = {
        **state.get("parser_config", {}),
        "toc_extraction": True,
    }
    logger.info("[注入] toc_extraction=True")

    # ============ 阶段 7: 目录构建（node_toc，**本节点**） ============
    pre_chunks = list(state["chunks"])  # 关键：保留原始引用用于断言比对
    pre_toc = state.get("toc", {})  # 节点前 toc 应为空 dict
    result = node_toc(state)
    chunks = result.get("chunks", [])
    toc = result.get("toc", {})

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(chunks) == len(pre_chunks), (
        f"chunks 数量应保持不变: {len(pre_chunks)} -> {len(chunks)}"
    )

    # ========== 断言 2：chunks 必含 id 字段（toc 内部会兜底生成） ==========
    missing_id = [i for i, c in enumerate(chunks) if not c.get("id")]
    assert not missing_id, f"chunks 缺少 id 字段: {missing_id[:5]}"

    # ========== 断言 3：state["toc"] 已被填充 ==========
    assert isinstance(toc, dict), f"toc 应为 dict 类型，实际: {type(toc).__name__}"
    assert toc.get("items"), f"toc.items 不应为空，实际: {toc}"
    assert toc.get("content_with_weight"), "toc.content_with_weight 必填"
    assert toc.get("toc_kwd") == "toc", f"toc_kwd 应为 'toc'，实际: {toc.get('toc_kwd')}"

    # ========== 断言 4：toc 块元数据与 ragflow 对齐 ==========
    assert toc.get("available_int") == 0, f"available_int 应为 0，实际: {toc.get('available_int')}"
    assert toc.get("page_num_int") == [100000000], (
        f"page_num_int 应为 [100000000]，实际: {toc.get('page_num_int')}"
    )
    assert toc.get("id"), "toc 块应含 id 字段（xxhash64 hex）"
    assert len(toc["id"]) == 16, f"toc.id 应为 16 字符 hex，实际长度: {len(toc['id'])}"

    # ========== 断言 5：items 字段结构 ==========
    items = toc["items"]
    assert isinstance(items, list), f"items 应为 list，实际: {type(items).__name__}"
    assert all(isinstance(it, dict) for it in items), "items 元素必须为 dict"
    for i, it in enumerate(items):
        assert "level" in it and "title" in it and "ids" in it, (
            f"items[{i}] 缺少必填字段: {it}"
        )
        assert isinstance(it["ids"], list) and it["ids"], (
            f"items[{i}].ids 应为非空 list，实际: {it['ids']}"
        )
        # ids 里的所有 chunk_id 都必须能在 state["chunks"] 找到
        for cid in it["ids"]:
            assert any(c.get("id") == cid for c in chunks), (
                f"items[{i}].ids 含未在 chunks 中出现的 id: {cid}"
            )

    # ========== 断言 6：content_with_weight 与 items 一致 ==========
    cw_items = json.loads(toc["content_with_weight"])
    assert len(cw_items) == len(items), (
        f"content_with_weight 项数与 items 不一致: {len(cw_items)} vs {len(items)}"
    )

    # ========== 诊断输出 ==========
    for i, it in enumerate(items):
        logger.info(
            f"  toc[{i}] level={it['level']} | title={it['title']!r} | "
            f"ids={len(it['ids'])}"
        )

    logger.info(
        f"目录摘要: {len(items)} 个条目，"
        f"levels={sorted({it['level'] for it in items})}，"
        f"toc_id={toc['id']}"
    )
    logger.info("===== node_toc 节点联调测试完成 =====")
