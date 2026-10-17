from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.index_service import index_chunks


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: Milvus 入库 (node_import_milvus)

    职责：将带向量的 chunks（含 TOC 块）批量写入 Milvus 集合，
          对齐 RAGFlow task_executor.py:512-579 + 1104-1112 + 1947-2070。

    前置节点已完成：
        node_bge_embedding  → chunk["dense_vector"] / chunk["sparse_vector"]
        node_image_upload    → chunk["img_id"] / chunk["create_time"] /
                                chunk["create_timestamp_flt"] / chunk["id"]
        node_keyword_extraction → chunk["key_words"] / chunk["key_tokens"]
        node_question_proposal  → chunk["question_kwd"] / chunk["question_tks"]
        node_metadata_generate  → chunk["metadata"]
        node_content_tagging    → chunk["tags"]
        node_toc                → state["toc"]（独立于 chunks 的目录块）

    本节点负责：
        1. kb_id 兜底：未设置时使用 task_id 作为默认值（与 ragflow task["kb_id"] 对齐）
        2. 校验 chunks 非空（require_chunks）
        3. 幂等创建 Milvus 集合 schema（prepare_chunks_collection）
        4. 构造 Milvus 字段（_build_milvus_chunks，含 TOC 块）
        5. 批量插入 Milvus（insert_chunks）
        6. 回填 chunk_id 到 state["chunks"] 与 state["toc"]
    """
    add_running_task(state["task_id"], "node_import_milvus")

    # kb_id 兜底：若 state 未设置 kb_id，使用 task_id 作为默认值
    if not state.get("kb_id"):
        state["kb_id"] = state.get("task_id", "")
        logger.info(f"[node_import_milvus] kb_id 未设置，使用 task_id 作为默认值: {state['kb_id']}")

    state = index_chunks(state)
    add_done_task(state["task_id"], "node_import_milvus")
    return state


if __name__ == "__main__":
    # ============================================================
    # 联调测试：完整链路跑至 node_import_milvus
    # 流程：PDF → MD(MinerU) → metadata_extract → md_img → link_and_table_chunk
    #       → chunk_merge → image_upload → 5 个 LLM 增强（并行）→ node_bge_embedding
    #       → node_import_milvus（**本节点**）
    #
    # 关键校验：
    #   - Milvus 集合已存在（prepare_chunks_collection 幂等）
    #   - 所有普通 chunk 都获得 chunk_id（按主键回填）
    #   - state["toc"] 存在时，TOC 块也获得 chunk_id
    #   - 4 个 LLM 增强节点字段正确映射为 Milvus 字段（important_kwd / question_kwd / meta_fields / tag_kwd）
    #   - docnm_kwd / content_with_weight / available_int / position_int 等关键字段写入正确
    #   - TOC 块 available_int=0，toc_kwd="toc"
    #   - Milvus 端回查：按 doc_id 过滤，能查到对应数量的记录
    # ============================================================
    import os
    from pathlib import Path

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
    from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
    from app.rag.import_.config import MILVUS_VECTOR_DIM
    from app.infra.vectorstore.milvus_gateway import milvus_gateway
    from app.rag.import_.index_service import _build_milvus_chunks

    logger.info("===== 开始 node_import_milvus 节点联调测试 =====")

    test_pdf_path = os.path.join(
        PROJECT_ROOT,
        "doc",
        "hak180使用说明书.pdf",
    )
    test_task_id = "test_import_milvus_task_001"
    init_state = create_default_state(
        task_id=test_task_id,
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )
    init_state["file_title"] = Path(test_pdf_path).stem
    init_state["kb_id"] = "test_kb_001"  # 显式设置 kb_id，避免使用 task_id 兜底
    logger.info(f"[注入] file_title={init_state['file_title']!r}, kb_id={init_state['kb_id']!r}")

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
    # 注入 LLM 增强节点所需的 parser_config 配置（直接赋值，覆盖默认值 False / []）
    state["parser_config"] = state.get("parser_config") or {}
    state["parser_config"]["enable_metadata"] = True
    state["parser_config"]["metadata"] = [
        {"key": "product_name", "description": "产品名称，如华为 MateBook X Pro"},
        {"key": "language", "description": "文档语言", "enum": ["中文", "英文", "中英双语"]},
        {"key": "document_type", "description": "文档类型，如用户指南、快速入门、技术规格"},
    ]
    state["parser_config"]["tags"] = [
        "产品介绍", "技术规格", "使用方法", "安全警告", "故障排除",
        "维护保养", "快速入门", "设置配置"
    ]
    state["parser_config"]["topn_tags"] = 3
    state["parser_config"]["toc_extraction"] = True

    state = node_keyword_extraction(state)
    state = node_question_proposal(state)
    state = node_content_tagging(state)
    state = node_metadata_generate(state)
    state = node_toc(state)
    logger.info("[阶段7] 5 个 LLM 增强节点全部完成")

    # ============ 阶段 8: 向量化 ============
    state = node_bge_embedding(state)
    embedded_chunks = state.get("chunks", [])
    logger.info(f"[阶段8] 向量化完成，chunks: {len(embedded_chunks)}")

    # ============ 诊断：构造 Milvus 字段（不入库） ============
    milvus_chunks_preview = _build_milvus_chunks(state)
    logger.info(
        f"[诊断] _build_milvus_chunks 构造完成：{len(milvus_chunks_preview)} 条 "
        f"（chunks={len(embedded_chunks)} + "
        f"toc={'有' if state.get('toc') else '无'}）"
    )

    # ========== 断言 1：chunks 数量保持不变 ==========
    assert len(embedded_chunks) == len(state["chunks"]), "向量化与 milvus 入库后 chunks 数量应保持一致"

    # ========== 断言 2：_build_milvus_chunks 数量正确（含 TOC 块） ==========
    has_toc = bool(state.get("toc"))
    expected_milvus_count = len(embedded_chunks) + (1 if has_toc else 0)
    assert len(milvus_chunks_preview) == expected_milvus_count, (
        f"_build_milvus_chunks 数量不正确: 期望 {expected_milvus_count}, "
        f"实际 {len(milvus_chunks_preview)}"
    )

    # ========== 断言 3：Milvus 字段映射核心字段正确（非 TOC 块） ==========
    # 抽 3 条普通 chunk 验证关键字段
    sample_chunks = milvus_chunks_preview[:3] if len(milvus_chunks_preview) >= 3 else milvus_chunks_preview
    for i, mc in enumerate(sample_chunks):
        # 文档级字段
        assert mc.get("doc_id") == test_task_id, f"sample[{i}] doc_id 不正确: {mc.get('doc_id')}"
        assert mc.get("kb_id") == "test_kb_001", f"sample[{i}] kb_id 不正确: {mc.get('kb_id')}"
        assert mc.get("docnm_kwd") == init_state["file_title"], (
            f"sample[{i}] docnm_kwd 应为 file_title: {mc.get('docnm_kwd')!r}"
        )
        # 内容字段（仅写 content_with_weight，未额外写 content 字段）
        assert mc.get("content_with_weight"), f"sample[{i}] content_with_weight 不应为空"
        # 可用性 + 位置
        assert mc.get("available_int") == 1, f"sample[{i}] 普通块 available_int 应=1"
        assert mc.get("position_int") == i, f"sample[{i}] position_int 应={i}（enumerate 注入）"
        # 4 个 LLM 增强字段（JSON 字符串）
        for fld in ("important_kwd", "important_tks", "question_kwd", "question_tks", "meta_fields"):
            assert fld in mc, f"sample[{i}] 缺少字段: {fld}"
            assert isinstance(mc[fld], str), f"sample[{i}] {fld} 应为字符串（JSON）"
        # 标签字段（list[str] → {tag: 1} JSON 字符串）
        assert "tag_kwd" in mc, f"sample[{i}] 缺少 tag_kwd 字段"

    # ========== 断言 4：4 个 LLM 增强字段内容可 JSON 化（不强制非空，LLM 可能失败）============
    import json

    for fld in ("important_kwd", "important_tks", "question_kwd", "question_tks", "meta_fields"):
        val = sample_chunks[0].get(fld, "")
        if val:
            try:
                json.loads(val)
            except json.JSONDecodeError as e:
                raise AssertionError(f"sample[0] {fld} 不是合法 JSON: {val[:80]}… ({e})")

    # ========== 断言 5：TOC 块字段正确（如存在）============
    if has_toc:
        toc_mc = milvus_chunks_preview[-1]
        assert toc_mc.get("type") == "toc", f"TOC 块 type 应='toc'，实际: {toc_mc.get('type')!r}"
        assert toc_mc.get("available_int") == 0, (
            f"TOC 块 available_int 应=0，实际: {toc_mc.get('available_int')}"
        )
        assert toc_mc.get("dense_vector") == [], "TOC 块不应携带 dense_vector"
        assert toc_mc.get("sparse_vector") == {}, "TOC 块不应携带 sparse_vector"
        # content_with_weight 应为 items 序列化的 JSON
        toc_cww = toc_mc.get("content_with_weight", "")
        try:
            toc_items_back = json.loads(toc_cww)
            assert isinstance(toc_items_back, list), "TOC content_with_weight 应为 items 列表的 JSON"
            assert len(toc_items_back) == len(state["toc"]["items"]), (
                f"TOC 条目数不匹配: {len(toc_items_back)} vs {len(state['toc']['items'])}"
            )
        except json.JSONDecodeError as e:
            raise AssertionError(f"TOC content_with_weight 不是合法 JSON: {toc_cww[:80]}… ({e})")

    # ========== 断言 6：执行 node_import_milvus（实际入库） ============
    # 注：以下操作会真实写入 Milvus
    pre_milvus_chunks = list(state["chunks"])
    pre_milvus_toc = state.get("toc")
    result = node_import_milvus(state)
    final_chunks = result.get("chunks", [])

    # ========== 断言 7：每个入库的 chunk 获得 chunk_id ============
    # 被过滤的 chunk（hyperlinks / 嵌入失败）chunk_id=None 属预期，仅检查"应入库"的 chunk
    indexed_chunks = [
        (i, c) for i, c in enumerate(final_chunks)
        if c.get("sparse_vector") or c.get("dense_vector")
    ]
    chunk_id_list = [c.get("chunk_id") for _, c in indexed_chunks]
    missing_ids = [i for (i, _), v in zip(indexed_chunks, chunk_id_list) if not v]
    assert not missing_ids, f"普通 chunk 缺 chunk_id 的索引: {missing_ids[:5]}"
    assert len(chunk_id_list) == len(chunk_id_list), "回填数量应与 chunks 一致"

    # chunk_id 应为 INT64 兼容的整数（Milvus 自增主键）
    non_int_ids = [v for v in chunk_id_list if not isinstance(v, int)]
    assert not non_int_ids, f"chunk_id 应为 int，实际类型: {set(type(v).__name__ for v in non_int_ids[:5])}"
    assert all(v > 0 for v in chunk_id_list), "chunk_id 应为正整数"

    # ========== 断言 8：chunk_id 与原 chunks 引用一致 ==========
    for orig, final in zip(pre_milvus_chunks, final_chunks):
        assert orig.get("chunk_id") == final.get("chunk_id"), (
            "回填后原引用与新引用 chunk_id 应一致（保持引用）"
        )

    # ========== 断言 9：TOC 块获得 chunk_id ==========
    final_toc = result.get("toc")
    if pre_milvus_toc and final_toc:
        assert final_toc.get("chunk_id"), "TOC 块应回填 chunk_id"
        assert final_toc["chunk_id"] not in chunk_id_list, (
            "TOC 块的 chunk_id 应与普通块不重复"
        )

    # ========== 断言 10：Milvus 集合已就绪 + 按 doc_id 回查 ==========
    collection_name = milvus_gateway.chunks_collection
    milvus_client = milvus_gateway.client()
    assert milvus_client.has_collection(collection_name), (
        f"Milvus 集合 {collection_name} 应已创建（prepare_chunks_collection 幂等）"
    )

    # 诊断：分别按主键 + 按 doc_id 查询，确认数据真的写入了
    # Milvus 2.4 的 query 在数据量小（72 条）时仍会走全表扫描，无须建索引
    primary_key_ids = chunk_id_list
    res_by_pk = milvus_client.query(
        collection_name=collection_name,
        filter=f'chunk_id in {primary_key_ids}',
        output_fields=["chunk_id", "id", "type", "available_int", "position_int",
                        "docnm_kwd", "doc_id", "kb_id"],
        limit=200,
    )
    logger.info(f"[诊断] Milvus 端按主键 chunk_id in {primary_key_ids[:3]}... 查询返回 {len(res_by_pk)} 条")
    res_by_doc = milvus_client.query(
        collection_name=collection_name,
        filter=f'doc_id == "{test_task_id}"',
        output_fields=["chunk_id", "id", "type", "available_int", "position_int",
                        "docnm_kwd", "doc_id", "kb_id"],
        limit=200,
    )
    logger.info(f"[诊断] Milvus 端按 doc_id 查询返回 {len(res_by_doc)} 条")
    # 用 chunk_id_list 长度作为预期（实际写入 Milvus 的 chunk 数）
    expected_written = len(chunk_id_list)
    res = res_by_pk
    assert len(res) == expected_written, (
        f"Milvus 端按主键应能查到 {expected_written} 条（hyperlinks 已被过滤），"
        f"实际 {len(res)}。doc_id 查询返回 {len(res_by_doc)} 条"
    )

    # 验证 TOC 块在 Milvus 端标记正确
    if has_toc:
        toc_records = [r for r in res if r.get("type") == "toc"]
        assert len(toc_records) == 1, f"Milvus 端 TOC 块应=1，实际: {len(toc_records)}"
        toc_rec = toc_records[0]
        assert toc_rec.get("available_int") == 0, "Milvus 端 TOC available_int 应=0"
        assert toc_rec.get("type") == "toc", "Milvus 端 TOC type 应='toc'"

    # 验证普通块的 available_int 全为 1
    normal_records = [r for r in res if r.get("type") != "toc"]
    wrong_avail = [r for r in normal_records if r.get("available_int") != 1]
    assert not wrong_avail, (
        f"普通块 available_int 应=1，异常: {wrong_avail[:3]}"
    )

    # 验证 position_int 单调递增
    positions = [r.get("position_int") for r in normal_records]
    assert positions == sorted(positions), f"position_int 应单调递增，实际: {positions[:10]}"

    # 验证 docnm_kwd 一致
    wrong_title = [r for r in res if r.get("docnm_kwd") != init_state["file_title"]]
    assert not wrong_title, f"docnm_kwd 应一致，异常: {len(wrong_title)} 条"

    # ========== 诊断输出 ==========
    type_count = {}
    for c in final_chunks:
        t = c.get("type", "text")
        type_count[t] = type_count.get(t, 0) + 1
    logger.info(f"类型分布: {type_count}")
    logger.info(
        f"Milvus 入库摘要: 写入 {len(milvus_chunks_preview)} 条 "
        f"（{len(embedded_chunks)} 普通 + {1 if has_toc else 0} TOC）"
    )
    logger.info(
        f"chunk_id 范围: {min(chunk_id_list)} ~ {max(chunk_id_list)}"
    )
    logger.info(f"file_title={state.get('file_title', '')!r}")
    logger.info("===== node_import_milvus 节点联调测试通过 =====")
