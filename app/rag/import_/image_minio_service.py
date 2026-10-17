"""
图片 MinIO 上传服务。

将本地图片上传至 MinIO（对象存储），用 `![摘要](在线URL)` 替换 Markdown 中的本地引用，
并为 chunks 绑定 img_id，对齐 ragflow/rag/svr/task_executor.py:512-575 image2id 流程。

PIL Image → JPEG 编码 → MinIO 上传 → 生成确定性 img_id（xxhash64）→ 绑定到 chunk。
"""

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import xxhash
from PIL import Image

from app.shared.runtime.logger import logger, step_log
from app.infra.object_storage import minio_gateway


def _encode_image_to_jpeg(img: Image.Image) -> Optional[bytes]:
    """
    将 PIL Image 编码为 JPEG 字节流（参考 ragflow image2id:encode_image）。
    - RGBA/P 模式先转 RGB
    - 编码失败返回 None
    """
    try:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.getvalue()
    except OSError as e:
        logger.warning(f"JPEG 编码失败: {e}")
        return None


def _generate_chunk_img_id(content: str, doc_id: str) -> str:
    """
    对齐 RAGFlow rag/svr/task_executor.py:542-543 的 img_id 生成逻辑。

    用 xxhash64 对 (content + doc_id) 生成确定性唯一 ID，
    保证同一内容+文档的切片始终映射到同一 img_id，实现幂等性。

    :param content: chunk 文本内容
    :param doc_id: 文档 ID（task_id 或 md_path 的 hash）
    :return: img_id 字符串
    """
    return xxhash.xxh64((content + doc_id).encode("utf-8", "surrogatepass")).hexdigest()


@step_log("bind_img_id_to_chunks")
def bind_img_id_to_chunks(
    chunks: list[dict],
    img_id_dict: Dict[str, str],
    doc_id: str,
) -> list[dict]:
    """
    对齐 RAGFlow task_executor.py:539-561 逻辑：将图片绑定到 chunk。

    chunks 现在已是 list[dict]，每项为 {"content": ..., "images": list[str]|None}，
    不再需要从 md_content 正则匹配图片名，直接用 images 字段分配 img_id。

    处理逻辑：
        1. 遍历 chunks，取 chunk["images"] 列表
        2. img_id 字段：取 images[0] 在 img_id_dict 中的 img_id（与 RAGFlow 行为一致，保证主图参与向量检索）
        3. img_ids 字段：images 全部图片的 img_id 列表，覆盖表格等"图片归属表格而非行"的多图场景
        4. 用 xxhash64 对 (content + doc_id) 生成 chunk["id"]（对齐 RAGFlow task_executor.py:542-543）
        5. 写入 create_time / create_timestamp_flt 时间戳字段
        6. 若无图片则 img_id 置空字符串、img_ids 置空列表
        7. 清除 image 字段（对齐 RAGFlow task_executor.py:552）

    :param chunks: chunk 列表，每项为 {"content": str, "images": list[str]|None}
    :param img_id_dict: {图片名: img_id} 映射，由 upload_images_to_minio 返回
    :param doc_id: 文档唯一标识（用于日志）
    :return: 带 img_id / img_ids / create_time 的 chunk 列表（新副本，不修改原列表）
    """
    result = []
    now_str = str(datetime.now()).replace("T", " ")[:19]
    now_ts = datetime.now().timestamp()

    # 收集所有 chunk.images 中出现过的图片名（用于诊断 img_id_dict 的命中率）
    all_chunk_img_names = set()
    for chunk in chunks:
        for raw in (chunk.get("images") or []):
            # 兼容 url 为 list 的情况（如被 flatten 后残留）
            for url in (raw if isinstance(raw, list) else [raw]):
                if isinstance(url, str):
                    all_chunk_img_names.add(Path(url).name)

    unbound_in_dict = {k: v for k, v in img_id_dict.items() if k not in all_chunk_img_names}
    if unbound_in_dict:
        logger.warning(
            f"[bind_img_id_to_chunks] img_id_dict 中有 {len(unbound_in_dict)}/{len(img_id_dict)} "
            f"个图片未在任何 chunk.images 中出现（可能 md_content 未引用该图）: "
            f"{list(unbound_in_dict.keys())[:5]}{'...' if len(unbound_in_dict) > 5 else ''}"
        )

    for chunk in chunks:
        chunk_new = chunk.copy()

        # 用 chunk["images"] 字段直接分配 img_id（对齐 RAGFlow 直接取 image 字段）
        image_urls: list | None = chunk.get("images")
        if image_urls:
            # 统一将 url（可能是嵌套 list）展开为纯字符串路径，再取文件名
            def _resolve_img_name(url) -> str:
                if isinstance(url, list):
                    return _resolve_img_name(url[0]) if url else ""
                return Path(url).name
            first_name = _resolve_img_name(image_urls[0])
            chunk_new["img_id"] = img_id_dict.get(first_name, f"imagetemps-{doc_id}/{first_name}")
            # img_ids：覆盖 images 全部图片的 img_id（处理表格等多图场景，避免其余图片被丢弃）
            chunk_new["img_ids"] = [
                img_id_dict.get(_resolve_img_name(url), f"imagetemps-{doc_id}/{_resolve_img_name(url)}")
                for url in image_urls
            ]
        else:
            chunk_new["img_id"] = ""
            chunk_new["img_ids"] = []

        # 清除 image 字段（对齐 RAGFlow task_executor.py:552）
        chunk_new.pop("image", None)

        # 时间戳字段（对齐 RAGFlow task_executor.py:544-545）
        chunk_new["create_time"] = now_str
        chunk_new["create_timestamp_flt"] = now_ts

        # chunk 唯一 ID：对齐 RAGFlow task_executor.py:542-543
        # 用 xxhash64 对 (content + doc_id) 生成确定性唯一 ID
        if "id" not in chunk_new:
            chunk_new["id"] = xxhash.xxh64(
                (chunk.get("content", "") + doc_id).encode("utf-8", "surrogatepass")
            ).hexdigest()

        result.append(chunk_new)

    logger.info(f"为 {len(result)} 个 chunk 绑定 img_id，时间戳={now_str}")
    return result


@step_log("upload_images_to_minio")
def upload_images_to_minio(
    image_pil_dict: Dict[str, Image.Image],
    image_summaries_dict: Dict[str, str],
    md_content: str,
    stem: str,
) -> tuple[str, Dict[str, str]]:
    """
    将图片上传到 MinIO，用 `![摘要](在线URL)` 替换原有本地图片引用。

    处理流程（对齐 ragflow image2id）：
        1. 清空 MinIO 中该文档的旧图片（幂等，避免脏数据）
        2. 对 image_pil_dict 中每张图执行 JPEG 编码 → MinIO 上传
        3. 建立图片名 → 在线 URL / img_id 的映射
        4. 用 img_id 替换 MD 中本地引用（支持 image_summaries 做 alt text）

    参数:
        image_pil_dict: {图片名: PIL.Image}，由 enrich_markdown_images 扫描目录后加载
        image_summaries_dict: {图片名: 描述}，由 batch_describe_images 生成（可选）
        md_content: 当前 Markdown 文本
        stem: 文档名（用于 MinIO 路径）

    返回:
        (替换后的 md_content, img_id_dict) — img_id_dict 格式 {图片名: "imagetemps-{stem}/{name}"}
    """
    minio_client = minio_gateway.client()
    image_dir = minio_gateway.image_dir[1:]
    bucket = minio_gateway.bucket_name

    # 1. 清空 MinIO 中该文档的旧图片（幂等性）
    try:
        from minio.deleteobjects import DeleteObject
        object_list = minio_client.list_objects(
            bucket_name=bucket,
            prefix=f"{image_dir}/{stem}/",
            recursive=True,
        )
        delete_list = [DeleteObject(name=obj.object_name) for obj in object_list]
        errors = minio_client.remove_objects(bucket_name=bucket, delete_object_list=delete_list)
        for error in errors:
            logger.warning(f"删除旧图片失败: {error}")
    except Exception as e:
        logger.warning(f"清空 MinIO 旧图片失败: {e}")

    if not image_pil_dict:
        logger.info("没有图片需要上传")
        return md_content, {}

    # 2. JPEG 编码 → MinIO 上传，建立图片名 → 在线 URL / img_id 的映射
    img_url_dict: Dict[str, str] = {}
    img_id_dict: Dict[str, str] = {}

    for image_name, img in image_pil_dict.items():
        try:
            jpeg_binary = _encode_image_to_jpeg(img)
            if jpeg_binary is None:
                continue

            object_name = f"{image_dir}/{stem}/{image_name}"

            minio_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(jpeg_binary),
                length=len(jpeg_binary),
                content_type="image/jpeg",
            )

            img_url = minio_gateway.build_image_url(stem=stem, image_name=image_name)
            img_url_dict[image_name] = img_url
            # img_id 格式对齐 RAGFlow：imagetemps-{stem}/{name}
            img_id_dict[image_name] = f"imagetemps-{stem}/{image_name}"

            try:
                img.close()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"上传图片 {image_name} 失败: {e}")
            continue

    if not img_url_dict:
        logger.warning("没有上传成功任何图片")
        return md_content, {}

    # 3. 将 MD 中本地图片引用替换为【图片摘要 / 图片名】+【在线 URL】
    #    支持两种引用语法：
    #    - Markdown 图片: ![alt](images/xxx.jpg)  （正文典型用法）
    #    - HTML img 标签: <img src="images/xxx.jpg" />  （表格单元格内嵌图片，Markdown 标准要求用 HTML）
    for image_name, image_url in img_url_dict.items():
        image_summary = image_summaries_dict.get(image_name, image_name)

        # 3.1 替换 Markdown 语法：![任意 alt](任意 URL 含 image_name)
        md_rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = md_rep.sub(
            lambda x: f"![{image_summary}]({image_url})", md_content
        )

        # 3.2 替换 HTML img 标签：<img ... src="images/xxx" ... />（或 '...'/无引号）
        #     用字符类 ['\"] 避免被前面 (.*?) 吞掉 src 后的引号
        #     group 1 = src 之前的属性；group 2 = image_name 之后的属性
        full_src = "images/" + image_name
        html_rep = re.compile(
            r'<img\b(.*?)src=["' + "'" + r'"]\s*' + re.escape(full_src) + r'["' + "'" + r'"](.*?)/?>'
        )
        md_content = html_rep.sub(
            lambda x: f'<img{x.group(1)}src="{image_url}"{x.group(2)}/>',
            md_content,
        )

    # 4. 诊断：上传成功但 md_content 中未找到引用的图片
    #    通常意味着图片存在于 images/ 目录，但 md 中是其他语法（背景图、HTML attr 等）未被覆盖
    not_referenced = [
        name for name in img_url_dict
        if f"images/{name}" in md_content
    ]
    if not_referenced:
        logger.warning(
            f"[upload_images_to_minio] {len(not_referenced)}/{len(img_url_dict)} 张上传成功但 md_content 中仍存在 'images/xxx' 本地引用: "
            f"{not_referenced[:5]}{'...' if len(not_referenced) > 5 else ''}"
        )

    return md_content, img_id_dict


@step_log("backup_markdown")
def backup_markdown(new_md_content: str, md_path_obj: Path) -> str:
    """备份并保存增强后的新 Markdown 文件（写入 *_new.md）"""
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    return str(new_md_path_obj)
