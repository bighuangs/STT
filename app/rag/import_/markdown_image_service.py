"""
Markdown 图片 VLM 增强服务
负责：扫描图片 → 逐图并行调用 VLM 生成描述 → 返回 {图片名: 描述} 字典。
MinIO 上传和 Markdown 替换由 node_image_upload 节点统一在 main_graph 最后执行。

chunk 级 VLM 增强（集成在 enrich_markdown_images 末尾）：
- 若 state["chunks"] 已有（由 node_metadata_extract 预先切分好，每项为 {"content": ..., "images": ...}），
  则对有对应图片的 chunk 执行二次 VLM 增强，将描述追加到 chunk["content"]
- 所有逻辑均在 STT 内实现，不依赖 ragflow

MinIO 上传流程（参考 ragflow image2id）：
- scan_image_refs 扫描目录，返回 {图片名: PIL.Image}，由 node_image_upload 统一上传
"""
import base64
import io
import re
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS


VLM_DEGRADED_WARNING_CODE = "VLM_IMAGE_DESCRIPTION_DEGRADED"


def _iter_exception_chain(exc: BaseException):
    """遍历显式异常链，避免 LangChain 包装后丢失底层网络异常类型。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_transient_vlm_error(exc: BaseException) -> bool:
    """仅把连接、超时、限流和服务端 5xx 视为可安全降级的 VLM 故障。"""
    for current in _iter_exception_chain(exc):
        if isinstance(
            current,
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                httpx.TransportError,
                ssl.SSLError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return True
        if isinstance(current, APIStatusError):
            status_code = int(getattr(current, "status_code", 0) or 0)
            if status_code == 429 or status_code >= 500:
                return True
    return False


def _is_optional_vlm_not_configured(exc: BaseException) -> bool:
    """识别 get_llm_client 的明确缺配异常；其他 ValueError 仍按编程错误抛出。"""
    return any(
        isinstance(current, ValueError) and "[LLM客户端] 配置缺失" in str(current)
        for current in _iter_exception_chain(exc)
    )


def _record_vlm_warning(
    warnings: list[dict] | None,
    *,
    scope: str,
    subject: str,
    exc: BaseException | None = None,
) -> None:
    """写入可持久化、无密钥/原始异常文本的降级事实，并进行稳定去重。"""
    if warnings is None:
        return
    warning = {
        "code": VLM_DEGRADED_WARNING_CODE,
        "severity": "warning",
        "message": "视觉描述服务暂时不可用，已保留原始图片和正文并继续导入。",
        "stage": scope,
        "subject": subject,
        "error_type": type(exc).__name__ if exc is not None else "EMPTY_RESPONSE",
    }
    identity = (warning["code"], warning["stage"], warning["subject"], warning["error_type"])
    if any(
        (
            item.get("code"),
            item.get("stage"),
            item.get("subject"),
            item.get("error_type"),
        )
        == identity
        for item in warnings
        if isinstance(item, dict)
    ):
        return
    warnings.append(warning)


def _clean_markdown_block(text: str) -> str:
    """
    去除 Markdown 代码块语法（从 ragflow/common/string_utils.py 移植）。
    去掉首尾的 ```markdown 标记和 ``` 标记，返回纯净文本。
    """
    text = re.sub(r'^\s*```markdown\s*\n?', '', text)
    text = re.sub(r'\n?\s*```\s*$', '', text)
    return text.strip()


def _concat_images(img1: Image.Image, img2: Image.Image) -> Optional[Image.Image]:
    """
    纵向拼接两张图片（从 ragflow/rag/nlp/__init__.py:concat_img 移植）。
    - img1 在上，img2 在下
    - 宽度取两者最大值
    - 去除重复图片检测（pixel_data 完全相同则直接返回 img1）
    """
    if img1 is None and img2 is None:
        return None
    if img1 and img2 is None:
        return img1
    if img2 and img1 is None:
        return img2
    if img1 is img2:
        return img1

    pixel_data1 = img1.tobytes()
    pixel_data2 = img2.tobytes()
    if pixel_data1 == pixel_data2:
        return img1

    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))
    return new_image


def _call_vision_model(
    img: Image.Image,
    vision_model,
    prompt: str,
    min_image_side: int = 11,
    callback=None,
) -> str:
    """
    调用视觉模型描述图片。与 batch_describe_images 保持一致的调用方式：
    - 将 PIL Image 转为 base64
    - 通过 HumanMessage (image_url + text) + StrOutputParser 调用
    - _clean_markdown_block 清洗输出

    min_image_side: 跳过过小图片的最小边长（像素），默认 11（与 parser_config.vlm_min_image_side 对齐）
    """
    callback = callback or (lambda prog, msg: None)

    logger.info(f"  [_call_vision_model] 调用 _call_vision_model，img size={img.size if hasattr(img, 'size') else 'unknown'}, prompt len={len(prompt)}")

    if hasattr(img, "size") and (img.size[0] < min_image_side or img.size[1] < min_image_side):
        callback(0.0, f"Skip tiny image for VLM: {img.size[0]}x{img.size[1]}")
        return ""

    try:
        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG")
        except OSError:
            buf.seek(0)
            buf.truncate()
            img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]
        )
        logger.info(f"  [_call_vision_model] 调用 vision_model，prompt 长度={len(prompt)}, image size={img.size}")
        raw_output = (vision_model | StrOutputParser()).invoke([message])
        description = raw_output.strip() if raw_output else ""
        logger.info(f"  [_call_vision_model] 清洗后描述: {description[:100] if description else '(空)'}")
        return _clean_markdown_block(description)
    except Exception as e:
        callback(-1, str(e))
        # 降级决策由携带图片/chunk 标识的上层负责。这里不能把权限、取消或
        # 编程错误统一伪装成“空响应”，否则导入会在数据不完整时静默成功。
        raise


# =============================================================================
# 原有函数（仅移除 ragflow 导入，逻辑不变）
# =============================================================================


def is_supported_image(image_name):
    """
    判断文件是否为支持的图片格式。
    返回 True 表示是支持的图片格式，调用方应跳过（因为只处理 MD 引用的图片，无需再判断格式）。
    """
    ext = "." + image_name.lower().rsplit(".", 1)[-1]
    return ext in SUPPORTED_IMAGE_EXTENSIONS


@step_log("scan_image_refs")
def scan_image_refs(md_content: str, images_path_obj: Path) -> Tuple[List, Dict[str, Image.Image]]:
    """
    扫描图片目录 + Markdown 内容，找出【真正被 MD 引用的图片】，加载为 PIL.Image 并返回。

    返回:
        - image_context_list: [(图片名, 图片路径, (上文, 下文)), ...]（供 batch_describe_images 调用）
        - image_pil_dict: {图片名: PIL.Image}（供 MinIO 上传使用）
    """
    image_context_list = []
    image_pil_dict: Dict[str, Image.Image] = {}

    logger.info(f"[scan_image_refs] 开始扫描: md_content长度={len(md_content)}, images_path={images_path_obj}, exists={images_path_obj.exists()}")
    if not images_path_obj.exists():
        return image_context_list, image_pil_dict

    image_files = list(images_path_obj.iterdir())
    logger.info(f"[scan_image_refs] images目录文件数: {len(image_files)}")
    for image_file in image_files:
        image_name = image_file.name
        if not is_supported_image(image_name):
            continue

        # 兼容两种引用语法：
        #   - Markdown: ![alt](images/xxx.jpg)             （正文典型用法）
        #   - HTML:     <img src="images/xxx.jpg" />      （表格单元格内嵌图片）
        escaped = re.escape(image_name)
        md_rep = re.compile(r"\!\[.*?\]\(.*?" + escaped + r".*?\)")
        html_rep = re.compile(
            r'<img\b[^>]*src=["' + "'" + r'"]\s*' + re.escape("images/" + image_name) + r'["' + "'" + r'"]'
        )
        match_obj = md_rep.search(md_content) or html_rep.search(md_content)
        logger.info(f"[scan_image_refs] 图片={image_name}, 正则匹配={bool(match_obj)}")
        if not match_obj:
            continue

        start, end = match_obj.span()
        pre_context = md_content[max(0, start - 100) : start]
        pos_context = md_content[end : min(len(md_content), end + 100)]

        image_context_list.append((image_name, image_file, (pre_context, pos_context)))

        # 加载为 PIL.Image（RGBA/P 转 RGB，供 MinIO JPEG 编码使用）
        try:
            img = Image.open(image_file).convert("RGB")
            image_pil_dict[image_name] = img
        except PermissionError:
            raise
        except OSError as e:
            logger.warning(f"[scan_image_refs] 加载图片 {image_name} 失败: {e}")

    return image_context_list, image_pil_dict


def batch_describe_images(
    image_context_list: list,
    stem: str,
    vlm_max_workers: int = 10,
    *,
    warnings: list[dict] | None = None,
) -> Dict[str, str]:
    """
    逐图并行调用多模态模型生成描述，返回 {图片名: 描述} 字典。
    与 _call_vision_model 链路保持一致：每张图独立调用，vlm_max_workers 线程并行。

    vlm_max_workers: VLM 并发线程数，从 parser_config["vlm_max_workers"] 读取，默认 10
    """
    if not image_context_list:
        return {}

    summaries: Dict[str, str] = {}
    try:
        vm_model = llm_provider.vision_chat()
    except Exception as exc:
        if not (_is_transient_vlm_error(exc) or _is_optional_vlm_not_configured(exc)):
            raise
        logger.warning(
            "[batch_describe_images] VLM 初始化暂不可用，"
            f"{len(image_context_list)} 张图片使用占位描述继续导入: {type(exc).__name__}: {exc}"
        )
        _record_vlm_warning(
            warnings,
            scope="image_summary_batch",
            subject=f"{stem}:{len(image_context_list)} images",
            exc=exc,
        )
        return {
            image_name: f"图片 {idx + 1}"
            for idx, (image_name, _image_path, _ctx) in enumerate(image_context_list)
        }

    def process_one(idx: int, image_name: str, image_path: Path, ctx_above: str, ctx_below: str) -> tuple[int, str, str]:
        """线程 worker：加载图片，调用 VLM，返回 (idx, image_name, description)。"""
        try:
            img = Image.open(image_path).convert("RGB")
        except PermissionError:
            raise
        except OSError as e:
            logger.warning(f"[batch_describe_images] 加载图片 {image_name} 失败: {e}")
            return idx, image_name, ""

        try:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()

            if ctx_above or ctx_below:
                prompt_text = load_prompt(
                    "vision_llm_figure_describe_prompt_with_context",
                    context_above=ctx_above,
                    context_below=ctx_below,
                )
            else:
                prompt_text = load_prompt("vision_llm_figure_describe_prompt")

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ]
            )
            description = (vm_model | StrOutputParser()).invoke([message])
            description = description.strip() if description else ""
        finally:
            img.close()

        return idx, image_name, description

    futures = {}
    with ThreadPoolExecutor(max_workers=vlm_max_workers) as executor:
        for idx, (image_name, image_path, ctx) in enumerate(image_context_list):
            ctx_above, ctx_below = ctx if ctx else ("", "")
            future = executor.submit(
                process_one, idx, image_name, image_path, ctx_above, ctx_below
            )
            futures[future] = (idx, image_name)

        for future in as_completed(futures):
            expected_idx, expected_name = futures[future]
            degraded_exc: BaseException | None = None
            try:
                idx, image_name, description = future.result()
            except Exception as exc:
                if not _is_transient_vlm_error(exc):
                    raise
                idx, image_name, description = expected_idx, expected_name, ""
                degraded_exc = exc
                logger.warning(
                    f"[batch_describe_images] 图片 {image_name} 的 VLM 描述失败，"
                    f"保留原图并使用占位描述: {type(exc).__name__}: {exc}"
                )
                _record_vlm_warning(
                    warnings,
                    scope="image_summary",
                    subject=image_name,
                    exc=exc,
                )
            if description:
                summaries[image_name] = description
            else:
                summaries[image_name] = f"图片 {idx + 1}"
                if degraded_exc is None:
                    _record_vlm_warning(
                        warnings,
                        scope="image_summary",
                        subject=image_name,
                    )

    logger.info(f"批量 VLM 调用完成，生成 {len(summaries)} 个图片描述")
    return summaries


def _enrich_chunks_with_vision(
    state: dict,
    images_path_obj: Path,
    vlm_max_workers: int = 10,
    vlm_min_image_side: int = 11,
) -> dict:
    """
    对 state 中的每个 chunk 执行 VLM 增强（对齐 naive.py:1505-1528）。
    遍历每个 section，将其中图片合并后调用 VLM，将描述追加到 section 文本。

    vlm_max_workers: VLM 并发线程数，从 parser_config["vlm_max_workers"] 读取，默认 10
    vlm_min_image_side: 跳过过小图片的最小边长（像素），从 parser_config["vlm_min_image_side"] 读取，默认 11

    naive.py 流程：
        for idx, (section_text, _) in enumerate(sections):
            images = [section_images[idx]]            # 取出该 section 的图片列表
            combined_image = reduce(concat_img, images)  # 多图纵向合并
            boosted_figures = VisionFigureParser(...)
            sections[idx] = (section_text + "\n\n" + "\n\n".join([fig[0][1] for fig in boosted_figures]), ...)

    STT 实现（不使用 VisionFigureParser，直接调 VLM）：
        1. 从 chunk["images"] 取图片 URL 列表
        2. _resolve_image_urls 加载为 PIL Image 列表
        3. 多图纵向合并（reduce _concat_images）
        4. _call_vision_model 调用 VLM
        5. 描述追加到 chunk["content"]，结果写回 state["chunks"]
    """
    chunks: List = state.get("chunks", [])
    warnings: list[dict] = state.setdefault("vlm_warnings", [])

    try:
        vision_model = llm_provider.vision_chat()
        logger.info("[_enrich_chunks_with_vision] 检测到视觉模型，开始增强图形提取...")
    except Exception as exc:
        if not (_is_transient_vlm_error(exc) or _is_optional_vlm_not_configured(exc)):
            raise
        logger.warning(
            "[_enrich_chunks_with_vision] VLM 初始化暂不可用，保留 chunk 原文继续导入: "
            f"{type(exc).__name__}: {exc}"
        )
        _record_vlm_warning(
            warnings,
            scope="chunk_vision_batch",
            subject=f"{len(chunks)} chunks",
            exc=exc,
        )
        return state

    # chunks 现在是 list[dict]，每项为 {"content": str, "images": list[str]|None}
    # chunk_images 作为独立字段已被废弃，不再依赖它获取图片 URL
    enriched_chunks: List = [None] * len(chunks)

    def _resolve_image_urls(urls, images_path_obj):
        """将相对 URL 列表转换为 PIL Image 列表（跳过不存在的文件）。"""
        resolved = []
        for url in urls:
            clean_url = url.removeprefix("images/").removeprefix("images\\").lstrip("/\\")
            img_path = images_path_obj / clean_url if not Path(url).is_absolute() else Path(url)
            if img_path.exists():
                try:
                    resolved.append(Image.open(img_path).convert("RGB"))
                except PermissionError:
                    raise
                except OSError as exc:
                    logger.warning(f"[_resolve_image_urls] 加载图片 {img_path.name} 失败: {exc}")
        return resolved

    def _enrich_single(idx, chunk: dict, vision_model):
        chunk_text = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        image_urls = chunk.get("images") if isinstance(chunk, dict) else None
        if not image_urls:
            logger.info(f"  [_enrich_single] chunk[{idx}] image_urls 为空，跳过")
            return idx, chunk_text, None, False

        images = _resolve_image_urls(image_urls, images_path_obj)
        if not images:
            logger.info(f"  [_enrich_single] chunk[{idx}] 未能加载任何图片，跳过")
            return idx, chunk_text, None, False

        combined_image = None
        try:
            combined_image = images[0]
            for img in images[1:]:
                combined_image = _concat_images(combined_image, img)

            prompt_text = load_prompt("vision_llm_figure_describe_prompt")
            description = _call_vision_model(
                combined_image, vision_model, prompt_text, min_image_side=vlm_min_image_side
            )

            if description and description.strip():
                enriched_text = chunk_text + "\n\n" + description
                logger.info(f"  [_enrich_single] chunk[{idx}] 增强成功，描述长度={len(description)}")
                return idx, enriched_text, None, False
            else:
                logger.info(f"  [_enrich_single] chunk[{idx}] VLM 返回空，保留原文本")
                is_tiny = (
                    hasattr(combined_image, "size")
                    and (
                        combined_image.size[0] < vlm_min_image_side
                        or combined_image.size[1] < vlm_min_image_side
                    )
                )
                return idx, chunk_text, None, not is_tiny
        except Exception as exc:
            if not _is_transient_vlm_error(exc):
                raise
            logger.warning(
                f"[_enrich_single] chunk[{idx}] VLM 描述失败，保留原文本继续导入: "
                f"{type(exc).__name__}: {exc}"
            )
            return idx, chunk_text, exc, False
        finally:
            for img in images:
                try:
                    img.close()
                except OSError:
                    pass
            if combined_image is not None and all(combined_image is not img for img in images):
                try:
                    combined_image.close()
                except OSError:
                    pass

    futures = {}
    with ThreadPoolExecutor(max_workers=vlm_max_workers) as executor:
        for idx, chunk in enumerate(chunks):
            future = executor.submit(_enrich_single, idx, chunk, vision_model)
            futures[future] = idx

        for future in as_completed(futures):
            idx, enriched_text, degraded_exc, empty_response = future.result()
            chunk = chunks[idx] if isinstance(chunks[idx], dict) else {}
            enriched_chunks[idx] = {**chunk, "content": enriched_text}
            if degraded_exc is not None:
                _record_vlm_warning(
                    warnings,
                    scope="chunk_vision",
                    subject=f"chunk[{idx}]",
                    exc=degraded_exc,
                )
            elif empty_response:
                _record_vlm_warning(
                    warnings,
                    scope="chunk_vision",
                    subject=f"chunk[{idx}]",
                )

    state["chunks"] = enriched_chunks
    logger.info(f"[_enrich_chunks_with_vision] 完成，{len(enriched_chunks)} 个 chunk 已增强")
    return state


@step_log("enrich_markdown_images")
def enrich_markdown_images(state: dict) -> dict:
    """
    对 Markdown 中的图片做 VLM 增强（仅生成描述，不执行 MinIO 上传）。
    MinIO 上传和 Markdown 替换由 node_image_upload 在 main_graph 最后统一执行。

    流程：
        1. 扫描图片目录，返回 image_context_list（供 batch_describe_images）和 image_pil_dict（供 MinIO 上传）
        2. 调用 VLM 生成 image_summaries
        3. 若 state["chunks"] 已有（由 node_metadata_extract 预先切分好，每项为 {"content": ..., "images": ...}），
           则对有图的 chunk 执行二次 VLM 增强，将描述追加到 chunk["content"]，
           并将增强后的 chunks 更新回 state
    """
    md_path = state.get("md_path", "")
    md_content = state.get("md_content", "")

    if not md_path:
        logger.warning("[enrich_markdown_images] md_path 不存在，跳过")
        return state

    md_path_obj = Path(md_path)
    if not md_content:
        md_content = md_path_obj.read_text(encoding="utf-8")
        state["md_content"] = md_content

    images_path_obj = md_path_obj.parent / "images"

    # 回退搜索：当 md_path 重命名到带空格路径时（download_and_extract_markdown 的 rename 行为），
    # images/ 目录可能还在原始解压路径（无空格）下，此时向上找同名父目录
    if not images_path_obj.exists():
        parent_parent = md_path_obj.parent.parent
        parent_stem = md_path_obj.parent.stem
        logger.info(f"[enrich_markdown_images] 回退搜索: parent_stem='{parent_stem}', 遍历 {parent_parent}:")
        for sibling in parent_parent.iterdir():
            logger.info(f"  sibling: {sibling.name} (stem={sibling.stem}, is_dir={sibling.is_dir()})")
            if sibling.is_dir() and sibling.stem == parent_stem:
                candidate = sibling / "images"
                if candidate.exists():
                    images_path_obj = candidate
                    logger.info(f"[enrich_markdown_images] images 目录回退到: {images_path_obj}")
                    break

    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.info(f"[enrich_markdown_images] 无图片目录，跳过: {md_path_obj}")
        return state

    # VLM 超参数：从 parser_config 读取，统一使用默认值兜底
    parser_config = state.get("parser_config", {}) or {}
    vlm_max_workers = int(parser_config.get("vlm_max_workers", 10))
    vlm_min_image_side = int(parser_config.get("vlm_min_image_side", 11))
    logger.debug(
        f"[enrich_markdown_images] VLM 超参数: "
        f"vlm_max_workers={vlm_max_workers}, vlm_min_image_side={vlm_min_image_side}"
    )

    image_context_list, image_pil_dict = scan_image_refs(md_content, images_path_obj)
    logger.info(f"[enrich_markdown_images] 扫描到 {len(image_context_list)} 张图片")
    state["image_pil_dict"] = image_pil_dict
    vlm_warnings: list[dict] = state.setdefault("vlm_warnings", [])

    if image_context_list:
        image_summaries = batch_describe_images(
            image_context_list,
            md_path_obj.stem,
            vlm_max_workers=vlm_max_workers,
            warnings=vlm_warnings,
        )
        state["image_summaries"] = image_summaries

    # chunk 级 VLM 增强（chunks 由 node_metadata_extract 预先切分好，每项为 {"content": ..., "images": ...}）
    chunks = state.get("chunks", [])
    if chunks:
        state = _enrich_chunks_with_vision(
            state, images_path_obj, vlm_max_workers=vlm_max_workers,
            vlm_min_image_side=vlm_min_image_side,
        )

    return state
