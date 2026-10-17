"""
Service: 查询改写 (query_rewrite_service)

本服务对齐 RAGFlow chat 流程中「检索前的问题精炼」环节：
    ragflow api/db/services/dialog_service.py:643-663
    ├─ full_question(...)          多轮对话 + refine_multiturn：把带上下文的追问改写成独立问题
    ├─ cross_languages(...)        跨语言扩展：把问题翻译成多种语言，提升跨语种召回
    ├─ apply_meta_data_filter(...) 元数据过滤：依据用户问题命中文档元数据，生成候选 doc_ids
    └─ keyword_extraction(...)     LLM 抽取关键词，追加到检索问题尾部

对应 RAGFlow 源函数：
    rag/prompts/generator.py:full_question / cross_languages / keyword_extraction / gen_meta_filter
    common/metadata_utils.py:apply_meta_data_filter / meta_filter

与 RAGFlow 的关键差异（适配 STT 现状）：
    - LLM 栈：STT 用 LangChain（llm_provider），同步 invoke；RAGFlow 用自研 LLMBundle 异步 async_chat
    - 多租户/模型路由：租户和 KB 范围由认证上下文与控制库强制解析
    - 元数据来源：RAGFlow 走独立的文档元数据索引（ES/Infinity DocMetadataService）；
      STT 在导入阶段把 chunk["metadata"] 同步成 SQL MetadataCatalogEntry，
      在线查询从活动索引对应的控制目录读取，不扫描 Milvus chunk
    - 主体确认（item_names）：本服务的前身 item_name_confirm_service 已废弃，检索过滤改由 doc_ids 承担

模块级开关（默认值；后续可改为按请求配置）：
    ENABLE_MULTITURN_REFINE   多轮独立问题改写（有历史时才触发）
    CROSS_LANGUAGE_TARGETS    跨语言目标列表（空 = 关闭）
    ENABLE_KEYWORD_EXTRACTION 关键词抽取追加
    KEYWORD_TOPN              关键词数量
    元数据过滤由 state["meta_data_filter"] 驱动（缺省 None = 关闭）
"""

import datetime
from hashlib import sha256
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.application.model_usage_service import invoke_model_call
from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository
from app.security.principal import current_principal
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

# ====================== 全局配置 ======================
# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 是否启用多轮独立问题改写（full_question）；仅在存在历史时实际触发
ENABLE_MULTITURN_REFINE = True
# 跨语言扩展目标语言列表；空列表表示关闭
CROSS_LANGUAGE_TARGETS: list[str] = []
# 是否启用关键词抽取并追加到检索问题尾部
ENABLE_KEYWORD_EXTRACTION = True
# 关键词抽取数量
KEYWORD_TOPN = 3

# 关键词分隔符：与 ragflow task_executor.py:623 保持一致（中英文逗号/分号/顿号/换行）
_KEYWORD_DELIMITER = re.compile(r"[,，;；、\r\n]+")


# ---------------------------------------------------------------------------
# 基础工具：参数校验 / 历史加载 / 上下文拼接 / 历史写回
# ---------------------------------------------------------------------------
@step_log("validate_query_identity")
def validate_query_identity(state: QueryGraphState) -> tuple[str, str]:
    """
    校验查询状态中是否包含改写所需的核心字段。

    Args:
        state: 查询图当前状态，必须包含 original_query（用户问题）和 session_id（会话ID）

    Returns:
        tuple[str, str]: (原始问题, 会话ID)
    """
    original_query = state.get("original_query")
    session_id = state.get("session_id")

    # 两个字段都需要校验，任一缺失均需抛出 error
    if not original_query or not session_id:
        logger.error(
            f"查询状态缺失关键字段: original_query={original_query}, session_id={session_id}"
        )
        raise ValueError("查询状态缺失关键字段: original_query 或 session_id")

    return original_query, session_id


@step_log("load_history")
def load_history(session_id: str) -> list[dict]:
    """
    从 MongoDB 加载当前会话的最近聊天记录。
    用于多轮指代消解（代词、简称、上下文依赖）。
    """
    return history_repository.list_recent(session_id, limit=QUERY_HISTORY_LIMIT)


@step_log("build_history_context")
def build_history_context(history_messages: list[dict]) -> str:
    """
    把历史消息拼接成大模型能看懂的上下文格式。

    该函数是 Query rewrite 的历史上下文装配边界。
    """
    context: list[str] = []
    for message in history_messages:
        # 用户消息使用改写后的 query，助手消息使用原始 text
        content = (
            message.get("rewritten_query") if message.get("role") == "user" else message.get("text")
        )
        context.append(f"角色：{message.get('role')}\n内容：{content}")
    return "\n".join(context)


@step_log("save_user_message")
def save_user_message(state: dict) -> None:
    """
    把当前用户提问 + 改写结果写入 MongoDB 历史记录，供下一轮上下文使用。

    注意：保存的是「干净的独立问题」rewritten_query（不含追加的关键词），
    关键词只用于检索（写在 questions[-1] 里），不污染历史展示。
    """
    if state.get("runtime_managed_history", False):
        return
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
    )


# ---------------------------------------------------------------------------
# 1) full_question：多轮 → 独立完整问题（对齐 ragflow generator.py:full_question）
# ---------------------------------------------------------------------------
@step_log("full_question")
def full_question(
    history_messages: list[dict],
    original_query: str,
    *,
    state: dict | None = None,
) -> str:
    """
    结合历史对话，把带上下文的追问改写为「完整、独立、可单独检索」的问题。

    对齐 ragflow generator.py:full_question：
        1. 把 user/assistant 历史 + 当前问题拼成 conversation
        2. 注入今天/昨天/明天，便于相对日期归一化
        3. LLM 输出独立问题；去 <think> 残留；命中 **ERROR** 时回退为原问题

    Args:
        history_messages: 最近历史消息（不含当前问题）
        original_query: 当前用户原始问题

    Returns:
        改写后的独立问题；失败时回退为 original_query
    """
    # 拼接对话：历史 user 用改写后的 query，assistant 用原文，最后追加当前问题
    conv: list[str] = []
    for m in history_messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("rewritten_query") if role == "user" else m.get("text")
        conv.append(f"{role.upper()}: {content}")
    conv.append(f"USER: {original_query}")
    conversation = "\n".join(conv)

    today = datetime.date.today()
    prompt = load_prompt(
        "full_question",
        today=today.isoformat(),
        yesterday=(today - datetime.timedelta(days=1)).isoformat(),
        tomorrow=(today + datetime.timedelta(days=1)).isoformat(),
        conversation=conversation,
    )

    try:
        messages = [SystemMessage(content=prompt), HumanMessage(content="输出：")]
        response = invoke_model_call(
            state,
            scene="query_rewrite",
            invoke=lambda: llm_provider.chat().invoke(messages),
        )
        ans = StrOutputParser().invoke(response)
        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL).strip()
        if not ans or "**ERROR**" in ans:
            return original_query
        return ans
    except Exception as e:
        logger.warning(f"[full_question] 改写失败，回退原问题: {e}")
        return original_query


# ---------------------------------------------------------------------------
# 2) cross_languages：跨语言扩展（对齐 ragflow generator.py:cross_languages）
# ---------------------------------------------------------------------------
@step_log("cross_languages")
def cross_languages(
    query: str,
    languages: list[str],
    *,
    state: dict | None = None,
) -> list[str]:
    """
    把问题翻译成多个目标语言，得到一组检索变体，提升跨语种召回。

    对齐 ragflow generator.py:cross_languages：sys/user 双 prompt，译文以 `###` 分隔。
    STT 额外把「原问题」也保留在变体首位（原文 + 各译文一起参与检索）。

    Args:
        query: 待翻译问题
        languages: 目标语言列表（如 ["英语", "日语"]）

    Returns:
        去重后的检索变体列表；失败或未配置时返回 [query]
    """
    if not languages:
        return [query]

    try:
        sys_prompt = load_prompt("cross_languages_sys")  # 无占位符，原样返回
        user_prompt = load_prompt(
            "cross_languages_user", query=query, languages="、".join(languages)
        )
        response = invoke_model_call(
            state,
            scene="cross_language",
            invoke=lambda: llm_provider.chat().invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
            ),
        )
        ans = StrOutputParser().invoke(response)
        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL).strip()
        if "**ERROR**" in ans:
            return [query]

        # 译文以 `###` 分隔；兼容模型偶尔用空行分隔
        ans = re.sub(r"^输出：?\s*|^Output:\s*", "", ans)
        parts = [p.strip() for p in re.split(r"###|\n{2,}", ans) if p.strip()]

        # 原问题置于首位，连同译文一起去重保序
        variants: list[str] = []
        for q in [query, *parts]:
            if q and q not in variants:
                variants.append(q)
        return variants or [query]
    except Exception as e:
        logger.warning(f"[cross_languages] 跨语言扩展失败，回退原问题: {e}")
        return [query]


# ---------------------------------------------------------------------------
# 3) keyword_extraction：抽取关键词（对齐 ragflow generator.py:keyword_extraction）
# ---------------------------------------------------------------------------
@step_log("keyword_extraction")
def keyword_extraction(
    content: str,
    topn: int = 3,
    *,
    state: dict | None = None,
) -> str:
    """
    调用 LLM 从问题中抽取 topn 个关键词，返回以逗号拼接的关键词字符串。

    复用导入阶段同款 prompt（app/resources/prompts/keyword_extraction.prompt）。
    去 <think> 残留；命中 **ERROR** 返回空串。

    Args:
        content: 待抽取文本（通常是改写后的问题）
        topn: 关键词数量

    Returns:
        关键词字符串（如“使用说明，故障排除”）；失败返回 ""
    """
    if not content:
        return ""
    try:
        prompt = load_prompt("keyword_extraction", topn=topn, content=content)
        response = invoke_model_call(
            state,
            scene="keyword_extraction",
            invoke=lambda: llm_provider.chat().invoke(
                [SystemMessage(content=prompt), HumanMessage(content="输出：")]
            ),
        )
        ans = StrOutputParser().invoke(response)
        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL).strip()
        if "**ERROR**" in ans:
            return ""
        # 规范化分隔符并截断到 topn，避免模型多输出
        parts = [p.strip() for p in _KEYWORD_DELIMITER.split(ans) if p and p.strip()]
        return "，".join(parts[:topn])
    except Exception as e:
        logger.warning(f"[keyword_extraction] 关键词抽取失败: {e}")
        return ""


# ---------------------------------------------------------------------------
# 4) 元数据过滤：apply_meta_data_filter → doc_ids
#    （对齐 ragflow common/metadata_utils.py + generator.py:gen_meta_filter）
# ---------------------------------------------------------------------------
@step_log("load_meta_catalog")
def load_meta_catalog(kb_id: str | None = None) -> dict:
    """
    从 SQL 控制目录读取「元数据目录」：{key: {value(str): [doc_id, ...]}}。

    TODO-09 起，目录在索引 staging 阶段随 document/index version 幂等写入，
    在线查询不再扫描向量库 chunk。

    Args:
        kb_id: 可选知识库过滤；为空则聚合全量

    Returns:
        {属性名: {属性值: [doc_id, ...]}}；无活动索引或无元数据时返回 {}
    """
    principal = current_principal()
    if principal is None or not kb_id:
        logger.warning("[load_meta_catalog] 缺少认证主体或 KB，拒绝读取目录")
        return {}
    metas: dict[str, dict[str, list[str]]] = {}
    with session_scope() as session:
        repository = RetrievalProfileRepository(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
        )
        rows = repository.load_metadata_entries(kb_id)
        for row in rows:
            value = str(row.value_text).strip()
            if not value:
                continue
            bucket = metas.setdefault(row.metadata_key, {}).setdefault(value, [])
            if row.document_id not in bucket:
                bucket.append(row.document_id)

    logger.info(f"[load_meta_catalog] 从控制目录读取元数据属性 {list(metas.keys())}")
    return metas


@step_log("gen_meta_filter")
def gen_meta_filter(
    meta_data: dict,
    query: str,
    constraints: dict | None = None,
    *,
    state: dict | None = None,
) -> dict:
    """
    用 LLM 依据用户问题 + 可用元数据，生成过滤条件 {"logic", "conditions"}。

    对齐 ragflow generator.py:gen_meta_filter。STT 用 json_mode + JsonOutputParser，
    解析失败时回退 {"conditions": []}。

    Args:
        meta_data: load_meta_catalog 的结果 {key: {value: [doc_ids]}}
        query: 用户问题
        constraints: 可选 {key: op} 运算符限制

    Returns:
        {"logic": "and"/"or", "conditions": [{"key","value","op"}, ...]}
    """
    # 仅把「属性名 → 可选值列表」喂给 LLM（隐藏 doc_id 明细，降 token）
    structure = {key: list(values.keys()) for key, values in meta_data.items()}

    # meta_filter.prompt 含大量 JSON 花括号，不能走 str.format，改用占位符替换
    from app.shared.runtime.logger import PROJECT_ROOT

    prompt_path = PROJECT_ROOT / "app" / "resources" / "prompts" / "meta_filter.prompt"
    raw = prompt_path.read_text(encoding="utf-8")
    constraints_block = (
        f"- 运算符限制：{json.dumps(constraints, ensure_ascii=False)}" if constraints else ""
    )
    prompt = (
        raw.replace("__CURRENT_DATE__", datetime.date.today().isoformat())
        .replace("__METADATA_KEYS__", json.dumps(structure, ensure_ascii=False))
        .replace("__USER_QUESTION__", query)
        .replace("__CONSTRAINTS_BLOCK__", constraints_block)
    )

    try:
        response = invoke_model_call(
            state,
            scene="metadata_filter",
            invoke=lambda: llm_provider.chat(json_mode=True).invoke(
                [SystemMessage(content=prompt), HumanMessage(content="生成过滤器：")]
            ),
        )
        result = JsonOutputParser().invoke(response)
        if isinstance(result, dict) and isinstance(result.get("conditions"), list):
            return result
    except Exception as e:
        logger.warning(f"[gen_meta_filter] 生成过滤条件失败: {e}")
    return {"conditions": []}


def _meta_filter(metas: dict, filters: list[dict], logic: str = "and") -> list[str]:
    """
    依据过滤条件在元数据目录上做匹配，返回命中的 doc_ids（对齐 ragflow metadata_utils.meta_filter）。

    支持运算符：contains/not contains/in/not in/start with/end with/empty/not empty/=/≠/>/</≥/≤。
    多条件按 logic 取交集（and）或并集（or）。

    Args:
        metas: {key: {value(str): [doc_id]}}
        filters: [{"key","value","op"}, ...]
        logic: "and" / "or"

    Returns:
        命中的 doc_id 列表
    """
    if logic not in {"and", "or"}:
        raise ValueError("metadata filter logic must be 'and' or 'or'")

    def _match(input_val: str, operator: str, value: str) -> bool:
        a, b = str(input_val), str(value)
        # 数值/日期比较类运算符尝试转数值，失败则按字符串小写比较
        if operator in ("=", "≠", ">", "<", "≥", "≤"):
            la, lb = a.lower(), b.lower()
            try:
                na, nb = float(a), float(b)
                la, lb = na, nb  # 数值可比
            except (ValueError, TypeError):
                # 大小关系不能退回字典序，否则 "not-a-number" > "10"
                # 会把非法元数据误纳入范围。相等/不等仍允许普通字符串语义。
                if operator in (">", "<", "≥", "≤"):
                    return False
            try:
                if operator == "=":
                    return la == lb
                if operator == "≠":
                    return la != lb
                if operator == ">":
                    return la > lb
                if operator == "<":
                    return la < lb
                if operator == "≥":
                    return la >= lb
                if operator == "≤":
                    return la <= lb
            except TypeError:
                return False
        a_low, b_low = a.lower(), b.lower()
        if operator in ("contains", "not contains") and not b_low.strip():
            return False
        if operator == "contains":
            return b_low in a_low
        if operator == "not contains":
            return b_low not in a_low
        if operator in ("in", "not in"):
            # value 是「、/,/，」分隔的候选值列表；in = 元数据值精确等于其中之一。
            # 注意：不能用子串兜底（a_low in b_low），否则空串/任意子串会误命中
            # （Python 中 "" in "任意" 恒为 True），把无关文档全部拉进候选。
            tokens = [t.strip() for t in re.split(r"[、,，]", b_low) if t.strip()]
            matched = a_low in tokens
            return matched if operator == "in" else not matched
        if operator == "start with":
            return a_low.startswith(b_low)
        if operator == "end with":
            return a_low.endswith(b_low)
        if operator == "empty":
            return not a
        if operator == "not empty":
            return bool(a)
        return False

    doc_ids: set[str] | None = None
    for flt in filters:
        key = flt.get("key")
        op = flt.get("op")
        value = flt.get("value")
        ids: list[str] = []
        if key in metas:
            for val_str, docids in metas[key].items():
                if _match(val_str, op, value):
                    ids.extend(docids)

        if doc_ids is None:
            doc_ids = set(ids)
        elif logic == "and":
            doc_ids &= set(ids)
            if not doc_ids:
                return []
        else:
            doc_ids |= set(ids)

    return sorted(doc_ids or [])


@step_log("apply_meta_data_filter")
def apply_meta_data_filter(
    meta_data_filter: dict | None,
    question: str,
    base_doc_ids: list[str] | None = None,
    kb_id: str | None = None,
    *,
    state: dict | None = None,
) -> list[str] | None:
    """
    依据元数据过滤配置，命中文档元数据，返回候选 doc_ids。

    对齐 ragflow common/metadata_utils.py:apply_meta_data_filter 的三种模式：
        - auto       LLM 依据问题 + 全量元数据目录生成过滤条件
        - semi_auto  仅在选定的属性键上让 LLM 生成过滤条件
        - manual     直接使用配置里给定的条件

    meta_data_filter 结构示例：
        {"method": "auto"}
        {"method": "semi_auto", "semi_auto": ["color", {"key": "price", "op": ">"}]}
        {"method": "manual", "logic": "and",
         "manual": [{"key": "color", "value": "red", "op": "="}]}

    Args:
        meta_data_filter: 过滤配置；None/空表示不过滤
        question: 用户问题（auto/semi_auto 模式需要）
        base_doc_ids: 已有的基础 doc_ids（会被并入结果）
        kb_id: 可选知识库范围

    Returns:
        doc_id 列表；manual 命中为空时返回 ["-999"]（语义=没有任何文档命中）；
        auto/semi_auto 命中为空时返回 None
    """
    doc_ids: list[str] = list(base_doc_ids) if base_doc_ids else []
    if not meta_data_filter:
        return doc_ids

    method = meta_data_filter.get("method")

    # 惰性加载 SQL 元数据目录：只有真正需要时才访问控制库
    _cache: dict = {}

    def _metas() -> dict:
        if "v" not in _cache:
            _cache["v"] = load_meta_catalog(kb_id)
        return _cache["v"]

    if method == "auto":
        filters = gen_meta_filter(_metas(), question, state=state)
        doc_ids.extend(
            _meta_filter(_metas(), filters.get("conditions", []), filters.get("logic", "and"))
        )
        if not doc_ids:
            return None
    elif method == "semi_auto":
        selected_keys: list[str] = []
        constraints: dict = {}
        for item in meta_data_filter.get("semi_auto", []):
            if isinstance(item, str):
                selected_keys.append(item)
            elif isinstance(item, dict) and item.get("key"):
                selected_keys.append(item["key"])
                if item.get("op"):
                    constraints[item["key"]] = item["op"]
        if selected_keys:
            current = _metas()
            filtered_metas = {k: current[k] for k in selected_keys if k in current}
            if filtered_metas:
                filters = gen_meta_filter(
                    filtered_metas,
                    question,
                    constraints=constraints or None,
                    state=state,
                )
                doc_ids.extend(
                    _meta_filter(
                        current, filters.get("conditions", []), filters.get("logic", "and")
                    )
                )
                if not doc_ids:
                    return None
    elif method == "manual":
        filters = meta_data_filter.get("manual", [])
        doc_ids.extend(_meta_filter(_metas(), filters, meta_data_filter.get("logic", "and")))
        if filters and not doc_ids:
            doc_ids = ["-999"]

    logger.info(f"[apply_meta_data_filter] method={method}, doc_ids={doc_ids}")
    return doc_ids


# ---------------------------------------------------------------------------
# 主入口（节点调用）：编排 full_question / cross_languages / 元数据过滤 / keyword
# ---------------------------------------------------------------------------
@step_log("rewrite_query")
def rewrite_query(state: QueryGraphState) -> QueryGraphState:
    """
    查询改写服务（对齐 ragflow dialog_service.py:643-663 的检索前问题精炼）：

        1. 校验 original_query / session_id
        2. full_question：有历史且开启时，把追问改写成独立问题，否则用原问题
        3. cross_languages：可选跨语言扩展，得到检索变体 questions
        4. apply_meta_data_filter：可选元数据过滤，得到候选 doc_ids
        5. keyword_extraction：可选关键词抽取，追加到 questions[-1]
        6. 回写 state：rewritten_query / questions / keywords / doc_ids
        7. 写回 MongoDB 历史（保存干净的 rewritten_query）

    state 输出字段：
        rewritten_query: 干净的独立问题（供历史展示 / web 检索）
        questions:       检索变体列表（末位含追加关键词，供向量/HyDE 检索 join 使用）
        keywords:        抽取出的关键词字符串
        doc_ids:         元数据过滤命中的候选文档ID（空 = 不按文档过滤）
    """
    # 1. 校验必填参数
    original_query, session_id = validate_query_identity(state)

    # 2. full_question 多轮独立问题改写（仅在有历史且开启时触发）
    # 评测 case 显式注入 conversation；禁止读取线上 Mongo，保证结果与顺序无关。
    history_messages = (
        list(state.get("history") or [])
        if state.get("evaluation_mode", False) or state.get("history_snapshot_loaded", False)
        else load_history(session_id)
    )
    # 后续 QueryPlanner、ContextPack 固定预算和最终 Prompt 必须消费同一份历史
    # 快照；禁止各节点再次读取 Mongo 造成一次 query run 内视图漂移。
    state["history"] = history_messages
    if ENABLE_MULTITURN_REFINE and history_messages:
        rewritten_query = full_question(
            history_messages,
            original_query,
            state=state,
        )
    else:
        rewritten_query = original_query

    # 3. cross_languages 跨语言扩展（未配置目标语言时 = [rewritten_query]）
    questions = cross_languages(
        rewritten_query,
        CROSS_LANGUAGE_TARGETS,
        state=state,
    )

    # 4. 元数据过滤 → doc_ids（由 state["meta_data_filter"] 驱动，缺省关闭）
    doc_ids: list[str] = []
    meta_cfg = state.get("meta_data_filter")
    if meta_cfg:
        result = apply_meta_data_filter(
            meta_cfg,
            questions[-1],
            kb_id=state.get("kb_id"),
            state=state,
        )
        # 启用过滤后，目录缺失、过滤生成失败或零命中都必须 fail closed，
        # 不能把空结果解释成“全知识库检索”而扩大数据范围。
        doc_ids = result if isinstance(result, list) else ["__stt_no_matching_document__"]

    # 5. keyword_extraction 关键词抽取并追加到最后一个检索变体尾部（对齐 ragflow）
    keywords = ""
    if ENABLE_KEYWORD_EXTRACTION:
        keywords = keyword_extraction(
            questions[-1],
            topn=KEYWORD_TOPN,
            state=state,
        )
        if keywords:
            questions[-1] = f"{questions[-1]},{keywords}"

    # 6. 回写 state
    state["rewritten_query"] = rewritten_query
    state["questions"] = questions
    state["keywords"] = keywords
    state["doc_ids"] = doc_ids

    # 7. 同步历史记录到 MongoDB（保存干净问题，不含追加关键词）
    if not state.get("evaluation_mode", False) and not state.get("runtime_managed_history", False):
        save_user_message(state)

    # 正文、改写文本和关键词默认不进入日志；仅记录可定位但低敏的数量。
    logger.info(
        "[rewrite_query] query_hash={}, variant_count={}, keyword_present={}, "
        "document_filter_count={}",
        sha256(rewritten_query.encode("utf-8")).hexdigest()[:16],
        len(questions),
        bool(keywords),
        len(doc_ids),
    )
    return state
