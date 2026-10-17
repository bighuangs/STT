import os
import re
from pathlib import Path

from app.shared.runtime.logger import node_log, logger, PROJECT_ROOT
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.image_minio_service import upload_images_to_minio, bind_img_id_to_chunks, backup_markdown
from app.infra.object_storage import minio_gateway


@node_log("node_image_upload")
def node_image_upload(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片 MinIO 上传 (node_image_upload)

    对齐 ragflow/rag/svr/task_executor.py:512-575 image2id 流程：
        PIL Image → JPEG 编码 → MinIO 上传 → 替换 md_content 中的本地引用为在线 URL
        → 为 chunks 绑定 img_id（主图）+ img_ids（全部图）+ 时间戳字段

    依赖:
        1. node_md_img 已通过 enrich_markdown_images 生成:
           - image_pil_dict: {图片名: PIL.Image}
           - image_summaries: {图片名: 描述}（可选）
        2. node_link_and_table_chunk 已把表格内嵌图片挂到 chunk.images（图片归属表格而非行）
    """
    add_running_task(state["task_id"], "node_image_upload")

    md_path = state.get("md_path", "")
    md_content = state.get("md_content", "")
    # TODO-03：图片派生物按 document_version + index_version 隔离。全量重建会
    # 复用同一 document_version，因此只用文档版本仍可能清空 active 索引图片。
    document_version_id = state.get("document_version_id")
    index_version_id = state.get("index_version_id")
    stem = (
        f"{document_version_id}/{index_version_id}"
        if document_version_id and index_version_id
        else document_version_id or Path(md_path).stem
    )
    doc_id = state.get("document_id") or state.get("task_id", stem)

    image_pil_dict = state.get("image_pil_dict", {})
    image_summaries = state.get("image_summaries", {})
    chunks = state.get("chunks", [])

    if not image_pil_dict:
        logger.info(f"[{state['task_id']}] 无图片，跳过 MinIO 上传")
        add_done_task(state["task_id"], "node_image_upload")
        return state

    # 1. MinIO 上传 + md_content 中的图片引用替换
    new_md_content, img_id_dict = upload_images_to_minio(
        image_pil_dict=image_pil_dict,
        image_summaries_dict=image_summaries,
        md_content=md_content,
        stem=stem,
    )

    # 2. 为 chunks 绑定 img_id（主图）+ img_ids（全部图）+ 时间戳字段
    if chunks and img_id_dict:
        state["chunks"] = bind_img_id_to_chunks(
            chunks=chunks,
            img_id_dict=img_id_dict,
            doc_id=doc_id,
        )

    state["md_content"] = new_md_content
    state["img_id_dict"] = img_id_dict
    image_prefix = minio_gateway.image_dir.lstrip("/")
    state["uploaded_object_keys"] = sorted(
        f"{image_prefix}/{stem}/{image_name}" for image_name in img_id_dict
    )
    state["md_path"] = backup_markdown(new_md_content, Path(md_path))

    add_done_task(state["task_id"], "node_image_upload")
    return state


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from app.shared.runtime.logger import PROJECT_ROOT
    from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
    from app.rag.import_.metadata_extract_service import extract_document_metadata
    from app.rag.import_.markdown_image_service import enrich_markdown_images
    from app.process.import_.agent.nodes.node_link_and_table_chunk import node_link_and_table_chunk
    from app.process.import_.agent.nodes.node_chunk_merge import node_chunk_merge

    logger.info("===== 开始全流程测试（PDF 导入 → MinIO 上传完成） =====")

    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", "hak180使用说明书.pdf")
    init_state = create_default_state(
        task_id="test_image_upload_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    # ============ 阶段 1: PDF → MD（node_pdf_to_md） ============
    md_state = parse_pdf_to_markdown(init_state)
    logger.info(f"[阶段1] PDF 转 MD 完成，md_path: {md_state['md_path']}")

    # ============ 阶段 2: 文档切分（node_metadata_extract） ============
    md_state = extract_document_metadata(md_state)
    chunks_after_split = md_state.get("chunks", [])
    chunks_with_images_split = [i for i, c in enumerate(chunks_after_split) if c.get("images")]
    logger.info(
        f"[阶段2] 切分完成，chunks 总数: {len(chunks_after_split)}，"
        f"含图片的 chunk: {chunks_with_images_split}"
    )

    # ============ 阶段 3: VLM 图片增强（node_md_img） ============
    md_state = enrich_markdown_images(md_state)
    image_pil = md_state.get("image_pil_dict", {})
    image_sums = md_state.get("image_summaries", {})
    logger.info(f"[阶段3] VLM 增强完成，加载图片: {len(image_pil)} 张，摘要数量: {len(image_sums)}")
    for name in image_pil:
        logger.info(f"  图片: {name} | 摘要: {image_sums.get(name, '(无)')[:80]}...")

    # ============ 阶段 4: 超链接 / 表格结构化（node_link_and_table_chunk） ============
    # 关键：把 state["tables"] 拆为 8 个 table chunk（含 28 张表格图），
    # 追加 1 个 hyperlinks chunk。后续 node_chunk_merge 只做长度标准化，不会自己拆表格。
    md_state = node_link_and_table_chunk(md_state)
    chunks_after_link_table = md_state.get("chunks", [])
    tbl_chunks = [c for c in chunks_after_link_table if c.get("type") == "table"]
    hl_chunks = [c for c in chunks_after_link_table if c.get("type") == "hyperlinks"]
    tbl_with_img = [c for c in tbl_chunks if c.get("images")]
    logger.info(
        f"[阶段4] 链接/表格拆分完成，chunks 总数: {len(chunks_after_link_table)}，"
        f"table chunks: {len(tbl_chunks)}（含图 {len(tbl_with_img)}），"
        f"hyperlinks chunks: {len(hl_chunks)}"
    )

    # ============ 阶段 5: Chunk 合并（node_chunk_merge，table/hyperlinks 原样保留） ============
    md_state = node_chunk_merge(md_state)
    chunks_after_merge = md_state.get("chunks", [])
    tbl_chunks_after_merge = [c for c in chunks_after_merge if c.get("type") == "table"]
    logger.info(
        f"[阶段5] 合并完成，chunks 总数: {len(chunks_after_merge)}，"
        f"table chunks（合并后保留）: {len(tbl_chunks_after_merge)}"
    )

    # 关键快照：记录 node_image_upload 入参的原始 chunks（用于断言 1/3/7/9 比对）
    original_chunks_ref = list(md_state["chunks"])

    # ============ 阶段 6: MinIO 上传（node_image_upload，**本节点**） ============
    # 新拓扑：chunk_merge → image_upload → 5 个 LLM 并行 → bge_embedding → import_milvus
    result = node_image_upload(md_state)

    img_id_dict = result.get("img_id_dict", {})
    enriched_chunks = result.get("chunks", [])
    backup_md_path = result.get("md_path", "")
    logger.info(f"[阶段6] MinIO 上传完成，img_id_dict 数量: {len(img_id_dict)}")
    logger.info(f"  备份 Markdown: {backup_md_path}")

    # ========== 断言 1：chunks 数量与原始引用一致（merge 后） ==========
    assert len(enriched_chunks) == len(original_chunks_ref), \
        f"chunks 数量应保持不变: {len(original_chunks_ref)} -> {len(enriched_chunks)}"

    # ========== 断言 2：含图片的 chunk（合并后位置索引）必绑 img_id ==========
    chunks_with_img_id = [i for i, c in enumerate(enriched_chunks) if c.get("img_id")]
    assert chunks_with_img_id, "应有 chunk 绑定了 img_id"

    all_chunks_with_images_idx = [
        i for i, c in enumerate(original_chunks_ref) if c.get("images")
    ]
    assert set(chunks_with_img_id) >= set(all_chunks_with_images_idx), \
        f"含图片 chunk 必绑 img_id，未绑定的: {set(all_chunks_with_images_idx) - set(chunks_with_img_id)}"

    # ========== 断言 3：所有 chunk 必含时间戳字段 ==========
    missing_create_time = [i for i, c in enumerate(enriched_chunks) if not c.get("create_time")]
    missing_create_ts = [i for i, c in enumerate(enriched_chunks) if "create_timestamp_flt" not in c]
    assert not missing_create_time, f"chunks 缺少 create_time: {missing_create_time}"
    assert not missing_create_ts, f"chunks 缺少 create_timestamp_flt: {missing_create_ts}"

    # ========== 断言 4：旧 image 字段必清空 ==========
    has_image_field = [i for i, c in enumerate(enriched_chunks) if "image" in c]
    assert not has_image_field, f"chunks 不应包含 image 字段: {has_image_field}"

    # ========== 断言 5：img_id 格式必须符合 "imagetemps-{stem}/{name}" ==========
    for img_name, img_id in img_id_dict.items():
        assert img_id.startswith("imagetemps-"), \
            f"img_id 格式错误，期望 'imagetemps-*'，实际: {img_id}"

    # ========== 断言 6：img_ids 字段必须覆盖 images 全部图片（多图场景） ==========
    bad_img_ids: list[tuple[int, int, list, list]] = []
    for i, (orig, enriched) in enumerate(zip(original_chunks_ref, enriched_chunks)):
        image_urls = orig.get("images") or []
        enriched_img_ids = enriched.get("img_ids", [])
        if image_urls and len(enriched_img_ids) != len(image_urls):
            bad_img_ids.append((i, len(image_urls), enriched_img_ids, image_urls))
    assert not bad_img_ids, (
        f"img_ids 数量应等于 images 数量（多图场景必须全覆盖）: {bad_img_ids[:3]}"
    )

    # ========== 断言 7：img_id（主图）必须与 img_ids[0] 一致（语义对齐 RAGFlow） ==========
    mismatched_primary = [
        (i, c.get("img_id"), c.get("img_ids"))
        for i, c in enumerate(enriched_chunks)
        if c.get("img_ids") and c.get("img_id") != c["img_ids"][0]
    ]
    assert not mismatched_primary, (
        f"img_id 应等于 img_ids[0]（主图绑定语义）: {mismatched_primary[:3]}"
    )

    # ========== 断言 8：无图片的 chunk img_ids 必为空列表 ==========
    for i, (orig, enriched) in enumerate(zip(original_chunks_ref, enriched_chunks)):
        if not (orig.get("images") or []):
            assert enriched.get("img_ids") == [], (
                f"chunk[{i}] 无图片但 img_ids 不为空: {enriched.get('img_ids')}"
            )

    # ========== 断言 9：含图片 chunk 的 img_ids 内每个 id 都必须在 img_id_dict 里 ==========
    #         （防止出现"伪造的" img_id 进入 chunks）
    valid_ids = set(img_id_dict.values())
    for i, c in enumerate(enriched_chunks):
        for img_id in c.get("img_ids", []):
            assert img_id in valid_ids, \
                f"chunk[{i}] 出现未在 img_id_dict 注册的 img_id: {img_id}"

    # ========== 诊断 1：img_id_dict 中未绑定到任何 chunk 的图片 ==========
    bound_img_names = set()
    for c in enriched_chunks:
        for img_id in c.get("img_ids", []):
            if img_id:
                bound_img_names.add(Path(img_id).name)
    unbound = [n for n in img_id_dict if n not in bound_img_names]
    logger.info(
        f"[诊断1] img_id_dict 未绑定图片: {len(unbound)}/{len(img_id_dict)} 个，"
        f"示例: {unbound[:5]}{'...' if len(unbound) > 5 else ''}"
    )

    # ========== 诊断 2：表格 chunk 的 img_ids 多图覆盖 ==========
    table_with_img_chunks = [
        (i, c) for i, c in enumerate(enriched_chunks)
        if c.get("type") == "table" and c.get("img_ids")
    ]
    if table_with_img_chunks:
        max_table_imgs = max(len(c["img_ids"]) for _, c in table_with_img_chunks)
        avg_table_imgs = sum(len(c["img_ids"]) for _, c in table_with_img_chunks) / len(table_with_img_chunks)
        logger.info(
            f"[诊断2] 表格 chunk 图片覆盖: {len(table_with_img_chunks)} 个含图 table，"
            f"最多 {max_table_imgs} 图/表，平均 {avg_table_imgs:.1f} 图/表"
        )

    # ========== 诊断 3：备份 Markdown 中应不残留本地 images/ 路径 ==========
    with open(backup_md_path, "r", encoding="utf-8") as f:
        backup_text = f.read()
    remaining_local = re.findall(r'src="images/[^"]+\.jpg"', backup_text)
    logger.info(
        f"[诊断3] 备份 Markdown 中残留本地 images/ 引用: {len(remaining_local)} 处，"
        f"示例: {remaining_local[:3]}"
    )

    # ========== 打印摘要 ==========
    img_bound = 0
    for i, (orig, enriched) in enumerate(zip(original_chunks_ref, enriched_chunks)):
        img_id = enriched.get("img_id", "")
        img_ids = enriched.get("img_ids", [])
        create_time = enriched.get("create_time", "")
        images_field = orig.get("images") if isinstance(orig, dict) else None
        logger.info(
            f"  chunk[{i}] type={orig.get('type', 'text')} | images={len(images_field or [])} | "
            f"img_id={bool(img_id)} | img_ids={len(img_ids)} | "
            f"create_time={bool(create_time)}"
        )
        if img_id:
            img_bound += 1

    logger.info(
        f"绑定摘要: {img_bound}/{len(enriched_chunks)} 个 chunk 绑定了 img_id，"
        f"img_id_dict: {len(img_id_dict)} 张图片，备份文件: {backup_md_path}"
    )

    logger.info("===== 全流程测试结束（PDF → MD → 切分 → 增强 → 合并 → MinIO 上传） =====")
