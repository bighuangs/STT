import re
from pathlib import Path

import jieba

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

jieba.setLogLevel("INFO")


@step_log("resolve_input_file")
def resolve_input_file(state: dict) -> ImportGraphState:
    """
    文件类型识别与状态初始化节点（导入流程入口）
    核心功能：根据本地文件路径识别文件类型，自动装配对应状态字段，为后续流程路由提供依据

    业务逻辑：
        1. 校验 local_file_path 是否为空
        2. 通用策略预处理（与文件类型无关，提前执行）：
           - parser_config：合并用户传入的配置与默认配置（默认值唯一来源为 state.py，
             详见 graph_default_state["parser_config"]，涵盖 27 字段/7 大类）
           - child_delimiter：从 parser_config["children_delimiter"] 解析出自定义子分隔符
             （结果写入 state["child_delimiter"]，供下游分块/切分节点使用）
           - title_tks / title_sm_tks：文件名 jieba 粗/细粒度分词结果
             （细粒度阈值由 parser_config["min_token_len_for_subtokenize"] 控制）
           - file_title：文档标题（不含扩展名）
        3. TODO-07 使用 magic/MIME/OOXML signature 联合识别已注册格式
        4. 所有格式统一进入兼容节点 node_pdf_to_md；扩展格式不修改图拓扑

    Args:
        state: 导入流程全局状态，必须包含 local_file_path 字段

    Returns:
        ImportGraphState: 补全文件信息后的完整状态对象
    """
    # 1.获取本地文件路径
    local_file_path = state.get("local_file_path")

    # 2.校验文件路径是否为空
    if not local_file_path:
        logger.error("local_file_path 为空，无法进行文件类型识别")
        raise ValueError("local_file_path 为空，无法进行文件类型识别")

    # 3.通用策略预处理（与文件类型无关，提前执行）
    #    - parser_config：用户可覆盖任意 27 字段之一（详见 state.py 顶部 7 大类分组）
    #    - child_delimiter：从 parser_config["children_delimiter"] 解析出自定义子分隔符
    #      （结果写入 state["child_delimiter"]，供分块/切分节点使用）
    #    - title_tks / title_sm_tks：文件名 jieba 粗/细粒度分词结果
    #      （细粒度阈值由 parser_config["min_token_len_for_subtokenize"] 控制，默认 2）
    #    - file_title：文档标题（不含扩展名）
    #    - 这四项在任何文件类型下都可能用到，因此放在识别之前处理
    _parse_parser_config(state)
    _parse_child_delimiter(state)
    _tokenize_filename(state)

    # 4.设置文件标题（所有文件类型均需设置，供下游节点兜底语义使用）
    state["file_title"] = Path(local_file_path).stem

    # 5. 用服务端内容探测代替单一后缀判断。上传阶段已经把 resolved profile 写入
    #    DocumentVersion；Worker 仍会复核一次，防止文件在排队期间被替换。
    from app.application.document_parsing_service import resolve_parser_profile

    declared_mime = state["parser_config"].get("detected_mime_type")
    profile = resolve_parser_profile(
        local_file_path,
        declared_mime=declared_mime,
        parser_config=state["parser_config"],
    )
    state["is_document_read_enabled"] = True
    state["input_format"] = profile.detected_format
    state["input_mime_type"] = profile.detected_mime_type
    state["parser_id"] = profile.parser_id
    state["parser_version"] = profile.parser_version
    state["chunking_template_id"] = profile.template_id
    state["chunking_template_version"] = profile.template_version
    state["suggested_chunking_template_id"] = profile.suggested_template_id
    state["is_md_read_enabled"] = profile.detected_format == "markdown"
    state["is_pdf_read_enabled"] = profile.detected_format == "pdf"
    if state["is_md_read_enabled"]:
        state["md_path"] = local_file_path
    if state["is_pdf_read_enabled"]:
        state["pdf_path"] = local_file_path

    # 实际选择成为 config fingerprint 的一部分，后续 checkpoint 不会跨 parser/template
    # 版本混跑；这些值在正常上传路径中与持久快照完全一致。
    state["parser_config"].update(
        {
            "parser_id": profile.parser_id,
            "parser_version": profile.parser_version,
            "detected_format": profile.detected_format,
            "detected_mime_type": profile.detected_mime_type,
            "chunking_template": profile.template_id,
            "chunking_template_version": profile.template_version,
        }
    )

    return state


def _tokenize_text(text: str) -> str:
    """
    粗粒度分词：对文本进行语义分词。
    使用 jieba 精确模式进行中文分词，英文单词保持原样。
    """
    if not text:
        return ""
    tokens = jieba.cut(text, cut_all=False, HMM=False)
    return " ".join(tokens)


def _fine_grained_tokenize(state: ImportGraphState, text: str) -> str:
    """
    细粒度分词：在粗粒度分词结果基础上进一步拆分。
    对粗粒度切出的 2+ 字中文词再用 jieba 切分一次，最终保留 jieba 切分出的最小词。
    英文/数字/单字词保持原词不变。

    长度阈值从 state["parser_config"]["min_token_len_for_subtokenize"] 读取，
    默认 2（与原硬编码一致），可通过 parser_config 调整。
    阈值含义：中文 token 长度 >= min_len 时会再被 jieba 切分（拆得更细）；
    阈值越大，被二次切分的中文词越少，title_sm_tks 越接近 title_tks。
    """
    if not text:
        return ""
    parser_config = state.get("parser_config", {}) or {}
    min_len = int(parser_config.get("min_token_len_for_subtokenize", 2))
    tokens = text.split()
    result = []
    for token in tokens:
        if re.search(r'[\u4e00-\u9fff]', token) and len(token) >= min_len:
            sub_tokens = jieba.cut(token, cut_all=False)
            result.extend(t for t in sub_tokens if t)
        else:
            result.append(token)
    return " ".join(result)


def _tokenize_filename(state: ImportGraphState) -> None:
    """
    对文件名（去除扩展名）进行 jieba 粗/细粒度分词：
    - title_tks：jieba 精确模式粗粒度分词结果
    - title_sm_tks：在粗粒度结果上进一步逐字拆分（细粒度）
    注意：直接从 local_file_path 提取文件名，不依赖 file_title，
    因为本函数在通用策略预处理阶段执行，file_title 尚未设置。
    """
    local_file_path = state.get("local_file_path", "")
    filename = Path(local_file_path).stem
    filename = re.sub(r"\.[a-zA-Z]+$", "", filename)

    state["title_tks"] = _tokenize_text(filename)
    state["title_sm_tks"] = _fine_grained_tokenize(state, state["title_tks"])
    logger.debug(f"文件名分词结果：原始={filename}, 粗粒度={state['title_tks']}, 细粒度={state['title_sm_tks']}")


def _parse_parser_config(state: ImportGraphState) -> None:
    """
    解析 parser_config：从 state 中提取用户传入的 parser_config，与默认配置合并后写回 state。

    默认值唯一来源：app.process.import_.agent.state.graph_default_state["parser_config"]
    本函数不重复定义默认值（避免与 state.py 同步漂移）。

    用户可通过 parser_config 覆盖任意默认参数；未提供的键保持 state.py 默认。
    """
    user_config = state.get("parser_config", {}) or {}
    # 从 state.py 引入默认配置，作为兜底
    # 延迟导入避免循环依赖（state.py 不依赖 entry_service）
    from app.process.import_.agent.state import graph_default_state

    default_config = graph_default_state["parser_config"]
    # 用户配置优先，默认配置兜底
    merged = {**default_config, **user_config}
    state["parser_config"] = merged
    logger.debug(f"parser_config 合并结果：{merged}")


def _parse_child_delimiter(state: ImportGraphState) -> None:
    """
    解析自定义子分隔符（children_delimiter）：
    - 输入：state["parser_config"]["children_delimiter"]（原始字符串，**不在 parser_config 默认值中**，
            由用户在创建 state 时显式传入，例如 'r"\\n||`---`"'）
    - 输出：state["child_delimiter"]（已转义反引号、按长度降序拼接、可直接用于 re.split 的正则字符串）
    - 处理流程：
        1. 将 \\n 等转义字符还原为真正的换行符等
        2. 提取反引号包裹的自定义分隔符（反引号语法优先级最高）
        3. 去掉 child_deli 中的反引号部分，保留原始分隔符（如 \n）
        4. 将多个自定义分隔符按长度降序排列后用 | 连接，防止短分隔符干扰
        5. 最终拼接系统默认分隔符，一并写入 state["child_delimiter"]
    参考 ragflow naive.py:1247-1259 的实现。
    """
    parser_config = state.get("parser_config", {})
    raw_deli = (parser_config.get("children_delimiter") or "")

    # 1.将 \\n 等转义字符还原为真正的换行符
    child_deli = raw_deli.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")

    # 2.提取反引号包裹的自定义分隔符
    cust_child_deli = re.findall(r"`([^`]+)`", child_deli)

    # 3.去掉 child_deli 中的反引号部分，保留原始分隔符（如 \n）
    child_deli = "|".join(re.sub(r"`([^`]+)`", "", child_deli))

    # 4.处理自定义分隔符
    if cust_child_deli:
        # 长度降序排序，防止短分隔符优先匹配导致长分隔符被切碎
        # 例如 || 和 | 同时存在时，先匹配 ||。
        cust_child_deli = sorted(set(cust_child_deli), key=lambda x: -len(x))
        # 用 | 作为多个自定义分隔符之间的连接符（作为正则中的"或"操作符）
        cust_child_deli = "|".join(re.escape(t) for t in cust_child_deli if t)
        child_deli += cust_child_deli

    # 5.写入 state["child_delimiter"]，供后续分块节点使用
    state["child_delimiter"] = child_deli
    logger.debug(f"解析后的子分隔符（child_delimiter）：{repr(child_deli)}")

