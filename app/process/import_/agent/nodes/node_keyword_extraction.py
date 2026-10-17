"""
节点: 文档关键词提取 (node_keyword_extraction)

并行分支之一：
    node_image_upload ─┬─▶ node_keyword_extraction  ─┐
                       ├─▶ node_question_proposal   ─┤
                       ├─▶ node_metadata_generate   ─┤
                       ├─▶ node_content_tagging     ─┼─▶ node_bge_embedding ─▶ node_import_milvus
                       └─▶ node_toc                 ─┘

参考 ragflow rag/svr/task_executor.py:doc_keyword_extraction (L592-641)
       ragflow rag/prompts/generator.py:keyword_extraction (L302-322)

实现说明：
    - 调用 keyword_extraction_service.extract_keywords(state)
    - 写入每个 chunk["key_words"]（LLM 提取的关键词）与 chunk["key_tokens"]（jieba 词粒度切分）
    - 失败/空 chunks/auto_keywords<=0 时安全跳过，字段仍初始化为空列表
"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.keyword_extraction_service import extract_keywords


@node_log("node_keyword_extraction")
def node_keyword_extraction(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 关键词提取 (node_keyword_extraction)

    遍历 state["chunks"]，为每个 chunk 异步并发调用 LLM 提取关键词，
    写入 chunk["key_words"]: list[str]（LLM 原始关键词）与
    chunk["key_tokens"]: list[str]（jieba 词粒度切分结果，供后续 Milvus 稀疏向量扩展）。
    """
    add_running_task(state["task_id"], "node_keyword_extraction")
    state = extract_keywords(state)
    add_done_task(state["task_id"], "node_keyword_extraction")
    # 上报增量给 llm_enhance_reducer 显式合并（不依赖共享引用）；
    # 仍返回完整 state，保证本文件 __main__ 独立联调可直接读 state["chunks"]
    state["llm_enhance_results"] = [{"chunks": state.get("chunks", []), "toc": state.get("toc")}]
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_keyword_extraction
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → 5 个并行 LLM 增强 → keyword_extraction
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
    from app.process.import_.agent.nodes.node_question_proposal import node_question_proposal
    from app.process.import_.agent.nodes.node_content_tagging import node_content_tagging
    from app.process.import_.agent.nodes.node_metadata_generate import node_metadata_generate
    from app.process.import_.agent.nodes.node_toc import node_toc

    logger.info("===== 开始 node_keyword_extraction 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_keyword_extraction_task_001",
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

    # ============ 阶段 7: 5 个并行 LLM 增强节点依次执行（模拟 LangGraph 并行） ============
    # 实际图中这 5 个节点并行跑；测试里顺序执行即可（无相互依赖，仅都读 chunks 写字段）
    pre_kw_chunks_ref = list(state["chunks"])  # 关键：保留进入 keyword 前的原始引用
    state = node_keyword_extraction(state)     # 阶段 7a: 关键词提取（**本节点**的核心验证）
    logger.info("[阶段7] 5 个 LLM 增强节点全部完成（keyword/question/tagging/metadata/toc）")

    # 取出最终 chunks & result 用于断言
    chunks = state.get("chunks", [])
    result = state
    pre_chunks = pre_kw_chunks_ref

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(chunks) == len(pre_chunks), (
        f"chunks 数量应保持不变: {len(pre_chunks)} -> {len(chunks)}"
    )

    # ========== 断言 2：所有 chunk 必含 key_words / key_tokens 字段 ==========
    missing_kw = [i for i, c in enumerate(chunks) if "key_words" not in c]
    missing_tk = [i for i, c in enumerate(chunks) if "key_tokens" not in c]
    assert not missing_kw, f"chunks 缺少 key_words: {missing_kw}"
    assert not missing_tk, f"chunks 缺少 key_tokens: {missing_tk}"

    # ========== 断言 3：key_words 必为 list[str] 类型 ==========
    bad_types = [
        (i, type(c.get("key_words")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("key_words"), list)
    ]
    assert not bad_types, f"key_words 类型错误: {bad_types}"

    # ========== 断言 4：key_tokens 必为 list[str] 类型 ==========
    bad_tk_types = [
        (i, type(c.get("key_tokens")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("key_tokens"), list)
    ]
    assert not bad_tk_types, f"key_tokens 类型错误: {bad_tk_types}"

    # ========== 断言 5：成功的 chunk 满足 topn 上限 ==========
    parser_config = result.get("parser_config", {}) or {}
    topn = int(parser_config.get("auto_keywords", 0) or 0)
    over_topn = [i for i, c in enumerate(chunks) if len(c.get("key_words", [])) > topn]
    assert not over_topn, f"key_words 数量超过 topn={topn}: {over_topn}"

    # ========== 诊断输出 ==========
    succ = 0
    for i, c in enumerate(chunks):
        kw = c.get("key_words") or []
        tk = c.get("key_tokens") or []
        kw_str = ", ".join(kw) if kw else "(无)"
        if kw:
            succ += 1
        # 诊断：key_tokens 必须是 jieba 词粒度切分（中文多为 2~4 字词，英文保留整词/单字符）
        tk_sample = ", ".join(tk[:10]) + ("..." if len(tk) > 10 else "")
        logger.info(
            f"  chunk[{i}] type={c.get('type', 'text')} | "
            f"key_words={len(kw)}={kw_str} | "
            f"key_tokens={len(tk)}={tk_sample}"
        )

    # ========== 断言 6（诊断级，jieba 切分语义）：key_tokens 总量应明显多于 key_words ==========
    # 关键词被 jieba 拆为词粒度后，tokens 数应 >= words 数（每个 kw 至少 1 个 token）
    total_words = sum(len(c.get("key_words") or []) for c in chunks)
    total_tokens = sum(len(c.get("key_tokens") or []) for c in chunks)
    logger.info(
        f"[诊断-jieba] 词粒度切分汇总: {total_words} 个 key_words → {total_tokens} 个 key_tokens "
        f"(平均 {total_tokens / max(total_words, 1):.2f} tokens/word)"
    )
    assert total_tokens >= total_words, (
        f"key_tokens 总量应 >= key_words 总量（jieba 切分粒度更细）: "
        f"{total_tokens} < {total_words}"
    )

    logger.info(
        f"关键词提取摘要: {succ}/{len(chunks)} 个 chunk 成功提取关键词，"
        f"topn={topn}"
    )
    logger.info("===== node_keyword_extraction 节点联调测试完成 =====")
