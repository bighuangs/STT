"""
节点: 文档内容打标 (node_content_tagging)

并行分支之一：
    node_image_upload ─┬─▶ node_keyword_extraction  ─┐
                       ├─▶ node_question_proposal   ─┤
                       ├─▶ node_metadata_generate   ─┤
                       ├─▶ node_content_tagging     ─┼─▶ node_bge_embedding ─▶ node_import_milvus
                       └─▶ node_toc                 ─┘

参考 ragflow rag/svr/task_executor.py:doc_content_tagging (L862-992)
       ragflow rag/prompts/generator.py:content_tagging (L433-480)
       ragflow rag/prompts/content_tagging_prompt.md

实现说明：
    - 调用 content_tagging_service.tag_content(state)
    - 写入每个 chunk["tags"]: list[str]（LLM 为 chunk 匹配的最相关标签列表）
    - 失败/空 chunks/tags 配置为空时安全跳过，字段仍初始化为空列表
"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.content_tagging_service import tag_content


@node_log("node_content_tagging")
def node_content_tagging(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 内容打标 (node_content_tagging)

    遍历 state["chunks"]，根据 parser_config["tags"] 配置，
    调用 LLM 为每个 chunk 打上最相关的标签，写入 chunk["tags"]: list[str]。

    parser_config 配置字段：
        - tags: list[str] — 标签列表，LLM 从中选取最相关的标签
        - topn_tags: int — 每个 chunk 最多打几个标签，默认 3
        - tag_kb_ids: list[str] — 参考知识库 ID 列表（STT 暂不支持，留空即可）

    行为：
        - tags 配置为空：安全跳过，所有 chunk["tags"] = []
        - LLM 调用失败：安全跳过，chunk["tags"] = []
        - 无 chunks：直接返回
    """
    add_running_task(state["task_id"], "node_content_tagging")
    state = tag_content(state)
    add_done_task(state["task_id"], "node_content_tagging")
    # 上报增量给 llm_enhance_reducer 显式合并（不依赖共享引用）；
    # 仍返回完整 state，保证本文件 __main__ 独立联调可直接读 state["chunks"]
    state["llm_enhance_results"] = [{"chunks": state.get("chunks", []), "toc": state.get("toc")}]
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_content_tagging
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → node_content_tagging（5 个 LLM 增强之一）
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

    logger.info("===== 开始 node_content_tagging 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_content_tagging_task_001",
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
    # 新拓扑：chunk_merge → image_upload（bge_embedding 已下沉至 5 个 LLM 之后）
    state = node_image_upload(state)
    logger.info("[阶段6] MinIO 上传完成")

    # ============ 阶段 7: 配置标签参数 + 内容打标（node_content_tagging，**本节点**） ============
    # 配置参考标签列表和 topn（模拟用户通过前端传入 parser_config）
    state["parser_config"]["tags"] = [
        "产品介绍",
        "技术规格",
        "使用方法",
        "安全警告",
        "故障排除",
        "维护保养",
        "快速入门",
        "设置配置",
        "网络连接",
        "驱动程序",
    ]
    state["parser_config"]["topn_tags"] = 3

    pre_chunks = list(state["chunks"])  # 关键：保留原始引用用于断言比对
    result = node_content_tagging(state)
    chunks = result.get("chunks", [])

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(chunks) == len(pre_chunks), (
        f"chunks 数量应保持不变: {len(pre_chunks)} -> {len(chunks)}"
    )

    # ========== 断言 2：所有 chunk 必含 tags 字段 ==========
    missing_tags = [i for i, c in enumerate(chunks) if "tags" not in c]
    assert not missing_tags, f"chunks 缺少 tags 字段: {missing_tags}"

    # ========== 断言 3：tags 必为 list 类型 ==========
    bad_types = [
        (i, type(c.get("tags")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("tags"), list)
    ]
    assert not bad_types, f"tags 类型错误: {bad_types}"

    # ========== 断言 4：tags 中的标签必为字符串类型 ==========
    bad_tag_types = []
    for i, c in enumerate(chunks):
        tags = c.get("tags") or []
        for t in tags:
            if not isinstance(t, str):
                bad_tag_types.append((i, type(t).__name__))
                break
    assert not bad_tag_types, f"tags 元素类型错误（应为 str）: {bad_tag_types}"

    # ========== 断言 5：成功的 tags 必来自参考标签集 ==========
    parser_config = result.get("parser_config", {}) or {}
    tag_set = set(parser_config.get("tags", []) or [])
    topn = int(parser_config.get("topn_tags", 3) or 3)
    invalid_tags = []
    for i, c in enumerate(chunks):
        tags = c.get("tags") or []
        if tags and tag_set:
            for t in tags:
                if t not in tag_set:
                    invalid_tags.append((i, t))
    assert not invalid_tags, f"tags 包含参考标签集之外的标签: {invalid_tags}"

    # ========== 断言 6：成功的 chunk 满足 topn 上限 ==========
    over_topn = [i for i, c in enumerate(chunks) if len(c.get("tags", [])) > topn]
    assert not over_topn, f"tags 数量超过 topn={topn}: {over_topn}"

    # ========== 诊断输出 ==========
    succ = 0
    for i, c in enumerate(chunks):
        tags = c.get("tags") or []
        tags_str = " | ".join(tags) if tags else "(无)"
        if tags:
            succ += 1
        logger.info(
            f"  chunk[{i}] type={c.get('type', 'text')} | "
            f"tags={len(tags)}={tags_str}"
        )

    logger.info(
        f"内容打标摘要: {succ}/{len(chunks)} 个 chunk 成功生成标签，"
        f"topn={topn}, 参考标签集={len(tag_set)} 个"
    )
    logger.info("===== node_content_tagging 节点联调测试完成 =====")
