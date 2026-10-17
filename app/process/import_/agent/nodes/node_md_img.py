from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.markdown_image_service import enrich_markdown_images
from app.rag.import_.metadata_extract_service import extract_document_metadata
from app.shared.runtime.logger import logger, PROJECT_ROOT
import os

from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片 VLM 增强 (node_md_img)
    职责: 调用 enrich_markdown_images，扫描 Markdown 中的图片并生成描述字典，
          将 {图片名: 描述} 和图片列表存入 state。
          若 state["chunks"] 已有（由 extract_document_metadata 预先切分好，每项含 images），
          则对有图的 chunk 执行二次 VLM 增强，将描述追加到 chunk 文本。
    MinIO 上传由 node_image_upload 在 main_graph 最后统一执行。
    """
    add_running_task(state["task_id"], "node_md_img")
    state = enrich_markdown_images(state)
    add_done_task(state["task_id"], "node_md_img")
    return state


if __name__ == "__main__":
    logger.info("===== 开始 node_md_img 节点联调测试 =====")

    # 1. PDF -> MD
    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", "HUAWEI MateBook B5-440 用户指南-(KLVFZ,Windows11_02,zh-cn).pdf")
    init_state = create_default_state(
        task_id="test_md_img_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )
    md_state = parse_pdf_to_markdown(init_state)
    logger.info(f"PDF 转 MD 完成，md_path: {md_state['md_path']}")

    # 2. 提取文档元数据（生成 chunks，每项含 images）
    md_state = extract_document_metadata(md_state)
    logger.info(f"文档切分完成，chunks 数量: {len(md_state.get('chunks', []))}")

    # 打印切分后有图片的 chunk
    for i, c in enumerate(md_state.get("chunks", [])):
        imgs = c.get("images") if isinstance(c, dict) else None
        if imgs:
            logger.info(f"  chunk[{i}] 关联图片: {imgs}")

    # 3. 验证 chunk 增强
    original_chunks_ref = md_state["chunks"]
    result = node_md_img(md_state)
    enriched_chunks = result["chunks"]
    changed = 0
    for i, (orig, enriched) in enumerate(zip(original_chunks_ref, enriched_chunks)):
        orig_content = orig.get("content", "") if isinstance(orig, dict) else str(orig)
        enriched_content = enriched.get("content", "") if isinstance(enriched, dict) else str(enriched)
        delta = enriched_content[len(orig_content):] if enriched_content.startswith(orig_content) else "[内容不包含原始前缀]"
        logger.info(f"  chunk[{i}] changed={orig_content != enriched_content} | 原始len={len(orig_content)} | 增强后len={len(enriched_content)} | delta(repr)={repr(delta)}")
        changed += int(orig_content != enriched_content)
    logger.info(f"  共 {changed} / {len(original_chunks_ref)} 个 chunk 发生变化")

    logger.info("===== 结束 node_md_img 节点联调测试 =====")
