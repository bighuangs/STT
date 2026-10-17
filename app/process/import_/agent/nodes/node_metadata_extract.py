from app.shared.runtime.logger import node_log, logger, PROJECT_ROOT
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.metadata_extract_service import extract_document_metadata
import os

@node_log("node_metadata_extract")
def node_metadata_extract(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 全局元数据提取 (node_metadata_extract)
    为什么叫这个名字: 从 PDF/MD 文件中提取全局元数据（如标题、作者、关键词、摘要、页数、创建时间等），
    汇聚 pdf 转 md 与 entry service 之后的结果，供下游节点使用。
    """
    add_running_task(state["task_id"], "node_metadata_extract")
    # 通用 ParsedDocument 兼容壳已经按模板产出 chunks；旧 checkpoint 没有标记时
    # 仍执行原 Markdown 粗切，保证升级前任务可恢复。
    if not state.get("structured_chunks_ready"):
        state = extract_document_metadata(state)
    add_done_task(state["task_id"], "node_metadata_extract")
    return state


if __name__ == "__main__":
    logger.info("===== 开始 node_metadata_extract 节点联调测试 =====")

    test_md_path = os.path.join(
        PROJECT_ROOT, "output", "hak180使用说明书", "hak180使用说明书.md"
    )
    init_state = create_default_state(
        task_id="test_metadata_extract_task_001",
        md_path=test_md_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    # ========== 完整流程：node_metadata_extract 节点 ==========
    result = node_metadata_extract(init_state)

    # ========== 断言验证 ==========
    assert "chunks" in result, "返回 state 应包含 chunks"
    assert "tables" in result, "返回 state 应包含 tables"
    assert "image_refs" in result, "返回 state 应包含 image_refs"

    chunks = result["chunks"]
    image_refs = result["image_refs"]
    assert chunks, "chunks 不应为空"

    # 空 chunk 定义：content 为空且 images 为空（纯图片行 content 为空但有 images 是正常的）
    empty_chunks = [
        i for i, c in enumerate(chunks)
        if not c.get("content") and not c.get("images")
    ]
    assert not empty_chunks, f"存在空 content 且无图片的 chunk: {empty_chunks}"

    # 图片关联：所有 image_refs 都应该出现在某个 chunk 的 images 中
    chunks_with_images = [i for i, c in enumerate(chunks) if c.get("images")]
    all_chunk_img_urls = {
        url
        for c in chunks
        for url in (c.get("images") or [])
    }
    ref_urls = {ref["url"] for ref in image_refs}
    unmatched = ref_urls - all_chunk_img_urls
    assert not unmatched, \
        f"有 {len(unmatched)} 张图未被任何 chunk 关联: {unmatched}"
    assert len(chunks_with_images) == len(ref_urls), \
        f"有图片的 chunk 数应等于 image_refs 数: {len(chunks_with_images)} vs {len(ref_urls)}"

    # ========== 诊断打印 ==========
    logger.info(f"文档加载完成，文本长度={len(result.get('md_content', ''))} 字符")
    logger.info(f"表格数量={len(result.get('tables', []))}")
    logger.info(f"image_refs 数量={len(image_refs)}")
    logger.info(f"chunks 数量={len(chunks)}")
    logger.info(f"含图片的 chunk 数量={len(chunks_with_images)}")
    for idx in chunks_with_images:
        logger.info(f"  chunk[{idx}] 图片: {chunks[idx]['images']}")

    logger.info(f"绑定摘要: {len(chunks)} 个 chunk，{len(chunks_with_images)} 个含图片，"
                f"全部 {len(image_refs)} 张图均已关联")
    logger.info("===== 集成测试结束 =====")
