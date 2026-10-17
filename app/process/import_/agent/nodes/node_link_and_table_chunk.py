"""
节点: 超链接提取与表格结构化 (node_link_and_table_chunk)

职责：
    1. 从 Markdown 内容中提取所有超链接，生成 "hyperlinks" 类型 chunk 追加到 state["chunks"]
    2. 将 state["tables"] 中的原始表格结构化并按 batch_size=10 分批生成 "table" 类型 chunk
    3. 操作主体为 state["chunks"]，md_content 不再被修改

前置依赖：node_md_img（已生成 md_content / tables）
后续节点：node_chunk_merge（基于 chunks 进行分块）
"""
from app.shared.runtime.logger import node_log, logger, PROJECT_ROOT
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.link_table_chunk_service import process_hyperlinks_and_tables
from app.rag.import_.metadata_extract_service import extract_document_metadata
from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
import os


@node_log("node_link_and_table_chunk")
def node_link_and_table_chunk(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 超链接提取与表格结构化 (node_link_and_table_chunk)
    """
    add_running_task(state["task_id"], "node_link_and_table_chunk")
    # 结构化 table blocks 已由模板切块，禁止再次从 Markdown 提取造成重复与证据丢失。
    if not state.get("structured_chunks_ready"):
        state = process_hyperlinks_and_tables(state)
    add_done_task(state["task_id"], "node_link_and_table_chunk")
    return state


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    logger.info("===== 开始 node_link_and_table_chunk 节点联调测试 =====")

    # 1. PDF -> MD
    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf")
    init_state = create_default_state(
        task_id="test_link_table_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )
    md_state = parse_pdf_to_markdown(init_state)
    logger.info(f"PDF 转 MD 完成，md_path: {md_state['md_path']}")

    # 2. 提取文档元数据（生成正文 chunks，每项含 images 字段；表格被抽离到 state["tables"]）
    md_state = extract_document_metadata(md_state)
    text_chunks_before = md_state.get("chunks", [])
    tables_before = md_state.get("tables", [])
    logger.info(
        f"文档切分完成，正文 chunks: {len(text_chunks_before)}，"
        f"抽离表格: {len(tables_before)}"
    )

    # 3. 统计正文 chunks 与原始表格的图片分布
    text_with_images = sum(1 for c in text_chunks_before if c.get("images"))
    table_image_count_dedup = 0
    table_with_img = 0
    seen_table_urls: set = set()
    from app.rag.import_.metadata_extract_service import extract_table_image_urls
    for t in tables_before:
        urls = extract_table_image_urls(t)
        if urls:
            table_with_img += 1
        seen_table_urls.update(urls)
    table_image_count_dedup = len(seen_table_urls)
    logger.info(
        f"  正文含图片 chunk: {text_with_images}/{len(text_chunks_before)}"
    )
    logger.info(
        f"  原始表格含图片: {table_with_img}/{len(tables_before)} 张，"
        f"去重后关联图片: {table_image_count_dedup}"
    )

    # 4. 执行 node_link_and_table_chunk
    result = node_link_and_table_chunk(md_state)
    all_chunks = result.get("chunks", [])

    hl_chunks = [c for c in all_chunks if c.get("type") == "hyperlinks"]
    tbl_chunks = [c for c in all_chunks if c.get("type") == "table"]
    txt_chunks = [c for c in all_chunks if c.get("type") not in ("hyperlinks", "table")]

    logger.info(
        f"chunks 总量: {len(all_chunks)}，"
        f"其中 hyperlinks={len(hl_chunks)}，table={len(tbl_chunks)}，text={len(txt_chunks)}"
    )

    # ========== 断言 1：chunks 数量结构正确 ==========
    assert len(hl_chunks) == 1, f"hyperlinks chunk 应只有 1 个，实际: {len(hl_chunks)}"
    assert len(txt_chunks) == len(text_chunks_before), \
        f"正文 chunks 应保持不变: {len(text_chunks_before)} -> {len(txt_chunks)}"
    assert tbl_chunks, "应有至少 1 个 table chunk"

    # ========== 断言 2：hyperlinks chunk 完整性 ==========
    hl_content = hl_chunks[0].get("content", "")
    logger.info(f"hyperlinks chunk 内容 ({len(hl_content)} 字符):\n{hl_content[:500]}")
    # 检查关键字段存在性（file_title 在 file_title 为空字符串时是 falsy，改用 in 判断）
    assert "file_title" in hl_chunks[0], "hyperlinks chunk 缺少 file_title 字段"
    assert "content" in hl_chunks[0], "hyperlinks chunk 缺少 content 字段"
    assert "type" in hl_chunks[0], "hyperlinks chunk 缺少 type 字段"

    # ========== 断言 3：table chunks 存储结构与正文同构 ==========
    for i, c in enumerate(tbl_chunks):
        assert c.get("type") == "table", f"table chunk[{i}] type 错误: {c.get('type')}"
        assert c.get("content"), f"table chunk[{i}] content 为空"
        # file_title 可能为空字符串（测试中未显式设置），只检查 key 存在
        assert "file_title" in c, f"table chunk[{i}] 缺少 file_title 字段"
        # images 字段必须存在（None 或 list），与正文 chunks 保持完全相同的存储结构
        assert "images" in c, f"table chunk[{i}] 缺少 images 字段（应与正文 chunk 同构）"
        images_val = c.get("images")
        assert images_val is None or isinstance(images_val, list), \
            f"table chunk[{i}] images 字段类型错误: {type(images_val).__name__}"
        logger.info(
            f"  table chunk[{i}]: images={images_val}, "
            f"content[:80]={c.get('content', '')[:80]!r}"
        )

    # ========== 断言 4 + 5：表格内嵌图片的完整性（合并：因 batch 切分会重复关联）
    # 设计语义：单张表的 image_urls 会被复制到该表所有 batch chunks（图片归属表格而非行），
    # 因此不能用"总出现次数"比较，而要用"去重 URL 集合"做双向完整性校验。
    original_table_urls: set = set()
    for t in tables_before:
        original_table_urls.update(extract_table_image_urls(t))
    chunk_table_urls: set = set()
    for c in tbl_chunks:
        chunk_table_urls.update(c.get("images") or [])

    tbl_chunks_with_images = [c for c in tbl_chunks if c.get("images")]
    logger.info(
        f"  table chunks 含图片: {len(tbl_chunks_with_images)}/{len(tbl_chunks)}，"
        f"原始表格去重 URL: {len(original_table_urls)} 个，"
        f"chunk 去重 URL: {len(chunk_table_urls)} 个"
    )
    missing_urls = original_table_urls - chunk_table_urls
    extra_urls = chunk_table_urls - original_table_urls
    assert not missing_urls, f"表格 chunk 缺失原始表格中的图片 URL: {missing_urls}"
    assert not extra_urls, f"表格 chunk 出现原始表格中不存在的图片 URL: {extra_urls}"
    logger.info(
        f"  表格内嵌图片 URL 双向完整性校验通过: {len(chunk_table_urls)} 个 URL 全部对齐"
    )

    # ========== 断言 6：文本 chunks 内容未被破坏（图片标记应已被 _strip_image_refs 清除） ==========
    for i, c in enumerate(txt_chunks[:5]):
        content = c.get("content", "")
        # 正文 chunks 在 metadata_extract 时已通过 _strip_image_refs 清除图片标记
        assert "![](images/" not in content, (
            f"text chunk[{i}] 残留 Markdown 图片标记: {content[:200]}"
        )
        logger.info(f"  text chunk[{i}]: {content[:80]!r}...")

    # ========== 诊断：表格 chunks 跨 batch 时的图片关联一致性 ==========
    # 设计语义：单张表的所有 batch chunks 应共享相同的 image_urls 集合
    # 用 (images 元组) 作为分组 key，统计每张原始表生成几个 batch chunk
    from collections import defaultdict
    batch_groups: dict[tuple, int] = defaultdict(int)
    for c in tbl_chunks:
        if c.get("images"):
            batch_groups[tuple(c["images"])] += 1
    multi_batch = [imgs for imgs, cnt in batch_groups.items() if cnt > 1]
    if multi_batch:
        logger.info(
            f"  跨 batch 一致性: {len(multi_batch)} 张原始表被切为多 batch，"
            f"其中 {sum(batch_groups.values())} 个 batch chunks 共享图片集"
        )
    else:
        logger.info("  跨 batch 一致性: 所有表格 chunk 均为单 batch，无需跨 batch 校验")

    logger.info("===== node_link_and_table_chunk 节点联调测试结束 =====")
