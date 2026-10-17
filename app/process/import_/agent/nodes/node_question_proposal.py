"""
节点: 文档配套提问 / 设问生成 (node_question_proposal)

并行分支之一：
    node_image_upload ─┬─▶ node_keyword_extraction  ─┐
                       ├─▶ node_question_proposal   ─┤
                       ├─▶ node_metadata_generate   ─┤
                       ├─▶ node_content_tagging     ─┼─▶ node_bge_embedding ─▶ node_import_milvus
                       └─▶ node_toc                 ─┘

参考 ragflow rag/svr/task_executor.py:doc_question_proposal (L650-695)
       ragflow rag/prompts/generator.py:question_proposal (L327-347)

实现说明：
    - 调用 question_proposal_service.propose_questions(state)
    - 写入每个 chunk["question_kwd"]（LLM 生成的问题列表）与
      chunk["question_tks"]（jieba 词粒度切分结果）
    - 字段命名与 ragflow 1:1 对齐（question_kwd / question_tks），便于后续 Milvus 字段映射
    - 失败/空 chunks/auto_questions<=0 时安全跳过，字段仍初始化为空列表
"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.question_proposal_service import propose_questions


@node_log("node_question_proposal")
def node_question_proposal(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 配套提问 / 设问生成 (node_question_proposal)

    遍历 state["chunks"]，为每个 chunk 异步并发调用 LLM 生成 topn 个候选问题，
    写入 chunk["question_kwd"]: list[str]（LLM 原始问题列表）与
    chunk["question_tks"]: list[str]（jieba 词粒度切分结果，供后续 Milvus 稀疏向量扩展）。
    """
    add_running_task(state["task_id"], "node_question_proposal")
    state = propose_questions(state)
    add_done_task(state["task_id"], "node_question_proposal")
    # 上报增量给 llm_enhance_reducer 显式合并（不依赖共享引用）；
    # 仍返回完整 state，保证本文件 __main__ 独立联调可直接读 state["chunks"]
    state["llm_enhance_results"] = [{"chunks": state.get("chunks", []), "toc": state.get("toc")}]
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_question_proposal
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → node_question_proposal（5 个 LLM 增强之一）
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

    logger.info("===== 开始 node_question_proposal 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf",
    )
    init_state = create_default_state(
        task_id="test_question_proposal_task_001",
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

    # ============ 阶段 7: 配套提问生成（node_question_proposal，**本节点**） ============
    pre_chunks = list(state["chunks"])  # 关键：保留原始引用用于断言比对
    result = node_question_proposal(state)
    chunks = result.get("chunks", [])

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(chunks) == len(pre_chunks), (
        f"chunks 数量应保持不变: {len(pre_chunks)} -> {len(chunks)}"
    )

    # ========== 断言 2：所有 chunk 必含 question_kwd / question_tks 字段 ==========
    missing_qk = [i for i, c in enumerate(chunks) if "question_kwd" not in c]
    missing_qt = [i for i, c in enumerate(chunks) if "question_tks" not in c]
    assert not missing_qk, f"chunks 缺少 question_kwd: {missing_qk}"
    assert not missing_qt, f"chunks 缺少 question_tks: {missing_qt}"

    # ========== 断言 3：question_kwd 必为 list[str] 类型 ==========
    bad_qk_types = [
        (i, type(c.get("question_kwd")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("question_kwd"), list)
    ]
    assert not bad_qk_types, f"question_kwd 类型错误: {bad_qk_types}"

    # ========== 断言 4：question_tks 必为 list[str] 类型 ==========
    bad_qt_types = [
        (i, type(c.get("question_tks")).__name__)
        for i, c in enumerate(chunks)
        if not isinstance(c.get("question_tks"), list)
    ]
    assert not bad_qt_types, f"question_tks 类型错误: {bad_qt_types}"

    # ========== 断言 5：成功的 chunk 满足 topn 上限 ==========
    parser_config = result.get("parser_config", {}) or {}
    topn = int(parser_config.get("auto_questions", 0) or 0)
    over_topn = [i for i, c in enumerate(chunks) if len(c.get("question_kwd", [])) > topn]
    assert not over_topn, f"question_kwd 数量超过 topn={topn}: {over_topn}"

    # ========== 诊断输出 ==========
    succ = 0
    for i, c in enumerate(chunks):
        qk = c.get("question_kwd") or []
        qt = c.get("question_tks") or []
        qk_str = " | ".join(qk) if qk else "(无)"
        if qk:
            succ += 1
        # 诊断：question_tks 必须是 jieba 词粒度切分（中文多为 2~4 字词，英文保留整词/单字符）
        qt_sample = ", ".join(qt[:10]) + ("..." if len(qt) > 10 else "")
        logger.info(
            f"  chunk[{i}] type={c.get('type', 'text')} | "
            f"question_kwd={len(qk)}={qk_str} | "
            f"question_tks={len(qt)}={qt_sample}"
        )

    # ========== 断言 6（诊断级，jieba 切分语义）：question_tks 总量应明显多于 question_kwd ==========
    # 问题被 jieba 拆为词粒度后，tokens 数应 >= questions 数（每个 question 至少 1 个 token）
    total_questions = sum(len(c.get("question_kwd") or []) for c in chunks)
    total_tokens = sum(len(c.get("question_tks") or []) for c in chunks)
    logger.info(
        f"[诊断-jieba] 词粒度切分汇总: {total_questions} 个 question_kwd → {total_tokens} 个 question_tks "
        f"(平均 {total_tokens / max(total_questions, 1):.2f} tokens/question)"
    )
    assert total_tokens >= total_questions, (
        f"question_tks 总量应 >= question_kwd 总量（jieba 切分粒度更细）: "
        f"{total_tokens} < {total_questions}"
    )

    logger.info(
        f"配套提问摘要: {succ}/{len(chunks)} 个 chunk 成功生成问题，"
        f"topn={topn}"
    )
    logger.info("===== node_question_proposal 节点联调测试完成 =====")
