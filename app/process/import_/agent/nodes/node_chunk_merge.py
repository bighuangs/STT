"""
节点: 文档块合并 (node_chunk_merge)

职责：对 metadata_extract_service.extract_document_metadata 生成的切分结果做精细化合并。
      切分逻辑由 extract_document_metadata 统一实现，此节点只负责长度标准化：
      - 过长的块 → 拆分（不超过 max_len）
      - 过短的块 → 合并（不低于 min_len）
      - special chunk（type=hyperlinks/table）原样保留不参与合并
      - 备份结果到 chunks.json

前置依赖：node_metadata_extract 已生成 state["chunks"]（每项含 images）
后续节点：node_bge_embedding
"""
from app.shared.runtime.logger import node_log, logger, PROJECT_ROOT
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.chunk_merge_service import merge_and_backup_chunks
from app.rag.import_.link_table_chunk_service import process_hyperlinks_and_tables


@node_log("node_chunk_merge")
def node_chunk_merge(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档块合并 (node_chunk_merge)
    职责：
        1. 从 state["chunks"] 读取原始切块（由 extract_document_metadata 生成）
        2. refine_chunks 做长度标准化（过长拆分、过短合并）
        3. 备份切块结果到 chunks.json
        4. 结果写回 state["chunks"]
    """
    add_running_task(state["task_id"], "node_chunk_merge")
    # TODO-07 模板已处理长度/重叠；旧合并器会重建 dict 并丢 provenance。
    if not state.get("structured_chunks_ready"):
        state = merge_and_backup_chunks(state)
    add_done_task(state["task_id"], "node_chunk_merge")
    return state


if __name__ == "__main__":
    import os
    from app.shared.runtime.logger import PROJECT_ROOT
    from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
    from app.rag.import_.metadata_extract_service import extract_document_metadata
    from app.process.import_.agent.state import create_default_state

    # -------------------------------------------------------------------------
    # 完整流程集成测试（从 PDF 到 chunk merge）
    # -------------------------------------------------------------------------
    logger.info("===== 开始 node_chunk_merge 节点联调测试 =====")

    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", "hak180使用说明书.pdf")
    init_state = create_default_state(
        task_id="test_chunk_merge_integration_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    # 1. PDF -> MD
    md_state = parse_pdf_to_markdown(init_state)
    logger.info(f"PDF 转 MD 完成，md_path: {md_state['md_path']}")

    # 2. 文档元数据提取与粗切分
    md_state = extract_document_metadata(md_state)
    text_chunks_before = [c for c in md_state.get("chunks", [])
                          if c.get("type") not in ("hyperlinks", "table")]
    logger.info(f"粗切分完成，text chunks: {len(text_chunks_before)}")

    # 3. 超链接与表格提取（生成 hyperlinks/table chunk）
    md_state = process_hyperlinks_and_tables(md_state)
    all_chunks_before_merge = md_state.get("chunks", [])
    hl_before = [c for c in all_chunks_before_merge if c.get("type") == "hyperlinks"]
    tbl_before = [c for c in all_chunks_before_merge if c.get("type") == "table"]
    txt_before = [c for c in all_chunks_before_merge
                  if c.get("type") not in ("hyperlinks", "table")]
    logger.info(f"link_table 处理后: text={len(txt_before)}, "
                f"hyperlinks={len(hl_before)}, table={len(tbl_before)}")

    # 4. 执行 node_chunk_merge（token 级合并）
    result = node_chunk_merge(md_state)
    all_chunks = result.get("chunks", [])

    hl_chunks = [c for c in all_chunks if c.get("type") == "hyperlinks"]
    tbl_chunks = [c for c in all_chunks if c.get("type") == "table"]
    txt_chunks = [c for c in all_chunks if c.get("type") not in ("hyperlinks", "table")]

    logger.info(f"合并后 chunks 总量: {len(all_chunks)}，"
                f"text={len(txt_chunks)}, hyperlinks={len(hl_chunks)}, table={len(tbl_chunks)}")

    # 5. 验证 hyperlinks 内容原样保留
    if hl_chunks:
        logger.info(f"hyperlinks chunk: {hl_chunks[0].get('content', '')[:200]}")
        assert len(hl_chunks) == 1, "hyperlinks 应只有 1 个"
        if hl_before:
            assert hl_chunks[0].get("content") == hl_before[0].get("content"), \
                "hyperlinks 内容应原样保留"

    # 6. 验证 table 内容原样保留
    for i, c in enumerate(tbl_chunks):
        logger.info(f"  table chunk[{i}]: {c.get('content', '')[:80]}...")
        assert c.get("type") == "table", "type 应保持为 table"

    # 7. 打印 text chunks 摘要
    for i, c in enumerate(txt_chunks):
        content = c.get("content", "")
        imgs = c.get("images")
        logger.info(f"  text chunk[{i}] ({len(content)} 字符, images={imgs}): {content[:60]}...")

    logger.info("===== 集成测试结束 =====")
