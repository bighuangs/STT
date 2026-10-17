"""
Service: 文档元数据生成 (metadata_generate_service)

参考 ragflow rag/svr/task_executor.py:gen_metadata_task (L719-856)
       ragflow rag/prompts/generator.py:gen_metadata (L1321-1340)
       ragflow rag/prompts/meta_data.md（已迁移至 app/resources/prompts/metadata_generate.prompt）

核心功能：遍历 state["chunks"]，根据 parser_config 中的元数据 Schema 配置，
    调用 LLM 从每个 chunk 中抽取结构化元数据，结果写入 chunk["metadata"]: dict。

与 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
    - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
    - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
    - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
"""
import asyncio
import json
import re
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.llm_limiter import ainvoke_with_retry


# 元数据生成提示词文件名（位于 app/resources/prompts/）
# 模板原为 ragflow rag/prompts/meta_data.md
_METADATA_PROMPT_NAME = "metadata_generate"

# 文档级（scope=document）非枚举字段聚合的最小出现频次：
# 逐 chunk 抽取会产生噪声取值（只在个别 chunk 出现），低于该频次的取值视为噪声丢弃。
_DOC_SCOPE_MIN_FREQ = 2


# ---------------------------------------------------------------------------
# Schema 转换（对齐 ragflow common/metadata_utils.py）
# ---------------------------------------------------------------------------

def _is_json_schema(obj: Any) -> bool:
    """
    判断 obj 是否已是 JSON Schema 格式。

    对齐 ragflow metadata_utils.py:_is_json_schema：
        - 含 $schema 键，或
        - type=object 且 properties 为 dict
    """
    if not isinstance(obj, dict):
        return False
    if "$schema" in obj:
        return True
    return obj.get("type") == "object" and isinstance(obj.get("properties"), dict)


def _is_metadata_list(obj: Any) -> bool:
    """
    判断 obj 是否为元数据字段列表格式。

    对齐 ragflow metadata_utils.py:_is_metadata_list：
        - list
        - 每个元素是 dict
        - 每个元素含非空字符串 key
        - 可选含 enum（list）或 description（str）
    """
    if not isinstance(obj, list) or not obj:
        return False
    for item in obj:
        if not isinstance(item, dict):
            return False
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return False
        if "enum" in item and not isinstance(item["enum"], list):
            return False
        if "description" in item and not isinstance(item["description"], str):
            return False
        if "descriptions" in item and not isinstance(item["descriptions"], str):
            return False
    return True


def _metadata_schema(metadata: list) -> dict:
    """
    将元数据字段列表转换为 JSON Schema（对齐 ragflow metadata_utils.py:metadata_schema）。

    Args:
        metadata: 字段列表，每项含 key / description / enum（可选）

    Returns:
        {"type": "object", "properties": {...}, "additionalProperties": False}
    """
    if not metadata:
        return {}
    properties = {}
    for item in metadata:
        key = item.get("key")
        if not key:
            continue
        prop_schema: dict[str, Any] = {
            "description": item.get("description", "")
        }
        if "enum" in item and item["enum"]:
            prop_schema["enum"] = item["enum"]
            prop_schema["type"] = "string"
        properties[key] = prop_schema

    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def _turn2jsonschema(obj: Any) -> dict:
    """
    将元数据配置转换为 JSON Schema（对齐 ragflow metadata_utils.py:turn2jsonschema）。

    支持两种输入格式：
        - dict：直接是 JSON Schema（type=object 且 properties=dict），直接返回
        - list：字段列表，调用 _metadata_schema 转换为 JSON Schema

    Args:
        obj: 元数据配置（dict 或 list）

    Returns:
        JSON Schema dict；若无法转换返回空 dict
    """
    if isinstance(obj, dict) and _is_json_schema(obj):
        return obj
    if isinstance(obj, list) and _is_metadata_list(obj):
        normalized = []
        for item in obj:
            description = item.get("description", item.get("descriptions", ""))
            normalized_item: dict[str, Any] = {
                "key": item.get("key"),
                "description": description,
            }
            if "enum" in item:
                normalized_item["enum"] = item["enum"]
            normalized.append(normalized_item)
        return _metadata_schema(normalized)
    return {}


def _build_metadata_prompt(content: str, schema: dict) -> str:
    """
    加载并渲染元数据提取 prompt。

    Args:
        content: 待分析的 chunk 原文
        schema: JSON Schema dict

    Returns:
        完整 prompt 字符串
    """
    from app.shared.runtime.logger import PROJECT_ROOT

    # 直接读写文件做占位符替换，避免 str.format 的 {{}} 解析与 schema JSON 冲突
    prompt_path = PROJECT_ROOT / "app" / "resources" / "prompts" / f"{_METADATA_PROMPT_NAME}.prompt"
    raw = prompt_path.read_text(encoding="utf-8")

    schema_text = json.dumps(schema, ensure_ascii=False)
    # {{ schema }} 在 .prompt 文件中作为占位符，避免被 str.format 先解析
    rendered = raw.replace("{{ schema }}", schema_text)
    # {content} 也用纯字符串替换，避免 schema JSON 中的 {}/{{}} 干扰 str.format
    rendered = rendered.replace("{content}", content)
    return rendered


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

async def _call_llm_for_metadata(content: str, schema: dict) -> str:
    """
    调用 LLM 提取元数据（JSON 输出模式）。

    与 ragflow generator.py:gen_metadata (L1321-1340) 对齐：
        1. 渲染 prompt（schema + content）
        2. 使用 json_mode=True 强制返回合法 JSON
        3. 去掉 RAGFlow 思考链残留标签
        4. 出错时返回 "{}" 兜底

    Args:
        content: chunk 原文
        schema: JSON Schema dict

    Returns:
        LLM 输出的 JSON 字符串；出错时返回 "{}"
    """
    try:
        llm = llm_provider.chat(json_mode=True)
        prompt_text = _build_metadata_prompt(content, schema)
        msg = [
            SystemMessage(content=prompt_text),
            HumanMessage(content="输出："),
        ]
        # 并发限流 + 失败指数退避重试（吸收代理/网络抖动导致的连接错误）
        resp = await ainvoke_with_retry(llm, msg)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 与 ragflow generator.py:1340 对齐：去掉思考链残留
        text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        logger.warning(f"[_call_llm_for_metadata] 调用失败: {e}", exc_info=True)
        return "{}"


async def _extract_for_chunk(chunk: dict, schema: dict, content_max_chars: int = 4000) -> None:
    """
    异步处理单个 chunk，原地写入 metadata。

    不抛异常：失败时写入空 dict，保证主流程不被单个 chunk 拖垮。

    Args:
        chunk: 单个 chunk 字典（必须有 "content" 字段）
        schema: JSON Schema dict
        content_max_chars: LLM 输入内容截断上限（从 parser_config.llm_content_max_chars 读取）
    """
    content = (chunk.get("content") or "").strip()
    if not content:
        chunk["metadata"] = {}
        return

    # 内容过长时截断（与 keyword/question service 策略一致）
    if len(content) > content_max_chars:
        content = content[:content_max_chars]

    raw = await _call_llm_for_metadata(content, schema)

    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"[_extract_for_chunk] JSON 解析失败，跳过: {raw[:100]}")
        chunk["metadata"] = {}
        return

    # 安全类型兜底：确保返回值为 dict
    if not isinstance(result, dict):
        chunk["metadata"] = {}
        return

    chunk["metadata"] = result


# ---------------------------------------------------------------------------
# 文档级字段聚合（scope=document）
# ---------------------------------------------------------------------------

def _collect_scope_meta(raw_metadata: Any) -> tuple[set[str], set[str]]:
    """
    从原始 metadata 配置中解析出「文档级字段」与「枚举字段」集合。

    - scope 取 "document"/"doc" 的字段视为文档级（整篇应一致）。
    - 含非空 enum 的字段视为枚举字段（单选）。

    支持 list（字段列表）与 dict（JSON Schema，scope 写在 properties[key] 上）两种配置。

    Returns:
        (文档级字段名集合, 枚举字段名集合)
    """
    doc_keys: set[str] = set()
    enum_keys: set[str] = set()

    def _scope_is_doc(spec: dict) -> bool:
        return str(spec.get("scope", spec.get("x-scope", ""))).lower() in ("document", "doc")

    if isinstance(raw_metadata, list):
        for item in raw_metadata:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not key:
                continue
            if _scope_is_doc(item):
                doc_keys.add(key)
            if item.get("enum"):
                enum_keys.add(key)
    elif isinstance(raw_metadata, dict):
        props = raw_metadata.get("properties")
        if isinstance(props, dict):
            for key, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                if _scope_is_doc(spec):
                    doc_keys.add(key)
                if spec.get("enum"):
                    enum_keys.add(key)
    return doc_keys, enum_keys


def _aggregate_document_scope(chunks: list, doc_keys: set[str], enum_keys: set[str]) -> None:
    """
    把「文档级」字段在整篇 doc 维度聚合，并统一回写到该 doc 的所有 chunk（原地修改）。

    动机：文档类型/设备型号等属性本应整篇一致，但逐 chunk 抽取会让少数 chunk 误标
    （例如一本操作手册里的“注意事项”段被标成 安全手册），这些少数派会污染 doc 级目录、
    让元数据过滤错误命中无关文档。

    聚合规则（按字段类型）：
        - 枚举字段：取出现次数最多的单值（多数表决），整篇统一为该值。
        - 非枚举字段：保留出现次数 >= _DOC_SCOPE_MIN_FREQ 的去重取值（列表，频次降序），
          丢弃只出现一次的噪声；若没有任何取值达到阈值，则退化为最高频单值。

    注意：本函数假设传入的 chunks 属于同一篇文档（导入流程按单文档执行）。
    """
    for key in doc_keys:
        counter: Counter = Counter()
        for c in chunks:
            md = c.get("metadata") or {}
            val = md.get(key)
            if val in (None, ""):
                continue
            for item in (val if isinstance(val, list) else [val]):
                item_str = str(item).strip()
                if item_str:
                    counter[item_str] += 1
        if not counter:
            continue

        if key in enum_keys:
            agg_value: Any = counter.most_common(1)[0][0]
        else:
            kept = [v for v, n in counter.most_common() if n >= _DOC_SCOPE_MIN_FREQ]
            agg_value = kept or counter.most_common(1)[0][0]

        # 回写到所有 chunk，使整篇文档对该字段标注一致（含原本为空的 chunk）
        for c in chunks:
            c.setdefault("metadata", {})[key] = agg_value

    logger.info(f"[generate_metadata] 文档级字段聚合完成：{sorted(doc_keys)}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@step_log("generate_metadata")
def generate_metadata(state: ImportGraphState) -> ImportGraphState:
    """
    遍历 state["chunks"]，根据 parser_config 中的元数据 Schema 逐 chunk 抽取结构化元数据，
    结果写入每个 chunk["metadata"]: dict。

    parser_config 元数据配置字段（与 ragflow parser_config 对齐）：

        parser_config["enable_metadata"]: bool
            - True / >0：启用元数据生成
            - False / 0 / 缺失：不生成，跳过

        parser_config["metadata"]: list[dict] | dict
            - list：自定义字段列表，每项含 key / description / enum（可选）
            - dict：直接是 JSON Schema（type=object，properties=dict）
            - 缺失或空：不生成

        parser_config["built_in_metadata"]: list[dict]
            - 内置字段列表，key 与 ragflow 内置字段名对齐（如 "author", "language"）
            - 与 metadata 合并后统一传给 LLM

    字段映射（与 ragflow task_executor.py:726-806 对齐）：
        - chunk["metadata"]: {"field_key": "field_value", ...}

    业务逻辑：
        1. 读取 parser_config 元数据配置；enable_metadata 为 False 时跳过
        2. 合并 metadata + built_in_metadata 为统一 Schema
        3. 对每个 chunk 并发调用 LLM（json_mode=True），提取结构化 JSON
        4. 错误兜底：JSON 解析失败或 LLM 调用失败时，写入空 dict {}
        5. 统计完成数量

    对齐 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
        - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
        - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
        - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现

    Args:
        state: LangGraph 状态字典

    Returns:
        state（原地修改 chunks）
    """
    parser_config = state.get("parser_config") or {}

    # -------------------------------------------------------------------------
    # 步骤 1：读取配置，检查是否启用
    # -------------------------------------------------------------------------
    enable_metadata = parser_config.get("enable_metadata", False)
    if not enable_metadata:
        logger.info("[generate_metadata] enable_metadata=False，跳过元数据生成")
        return state

    # LLM 输入内容截断上限（与 keyword/question service 策略一致）
    content_max_chars = int(parser_config.get("llm_content_max_chars", 4000))

    # -------------------------------------------------------------------------
    # 步骤 2：读取并合并 Schema（metadata + built_in_metadata）
    # -------------------------------------------------------------------------
    raw_metadata = parser_config.get("metadata", [])
    built_in = list(parser_config.get("built_in_metadata") or [])

    if isinstance(raw_metadata, dict):
        # dict 格式：直接是 JSON Schema
        metadata_conf = raw_metadata
        if not isinstance(metadata_conf.get("properties"), dict):
            metadata_conf = {"type": "object", "properties": {}}
        # built_in_metadata 合并到 properties（用户自定义优先）
        if built_in:
            schema_builtin = _turn2jsonschema(built_in)
            merged_properties = {**metadata_conf.get("properties", {}), **schema_builtin.get("properties", {})}
            metadata_conf = {
                **metadata_conf,
                "properties": merged_properties,
            }
    elif isinstance(raw_metadata, list):
        # list 格式：字段列表，拼接 built_in_metadata
        metadata_conf = raw_metadata + built_in
    else:
        # 其他格式：退化为仅 built_in_metadata
        metadata_conf = built_in

    schema = _turn2jsonschema(metadata_conf)
    if not schema or not schema.get("properties"):
        logger.info("[generate_metadata] Schema 为空，跳过元数据生成")
        return state

    logger.info(f"[generate_metadata] 元数据 Schema 字段: {list(schema['properties'].keys())}")

    # -------------------------------------------------------------------------
    # 步骤 3：为每个 chunk 并发调用 LLM
    # -------------------------------------------------------------------------
    chunks = state.get("chunks", [])
    if not chunks:
        logger.info("[generate_metadata] chunks 为空，跳过")
        return state

    async def _run_all() -> None:
        tasks = [_extract_for_chunk(c, schema, content_max_chars) for c in chunks]
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run_all())

    # -------------------------------------------------------------------------
    # 步骤 3.5：文档级字段聚合（scope=document）
    # 把整篇应一致的属性（如 文档类型/设备型号）按全文聚合并统一回写所有 chunk，
    # 消除逐 chunk 抽取产生的少数派噪声，避免污染 doc 级元数据目录。
    # -------------------------------------------------------------------------
    doc_keys, enum_keys = _collect_scope_meta(raw_metadata)
    if doc_keys:
        _aggregate_document_scope(chunks, doc_keys, enum_keys)

    # -------------------------------------------------------------------------
    # 步骤 4：统计
    # -------------------------------------------------------------------------
    succ = sum(1 for c in chunks if c.get("metadata"))
    logger.info(
        f"[generate_metadata] 完成：{succ}/{len(chunks)} 个 chunk 成功生成元数据"
    )
    return state
