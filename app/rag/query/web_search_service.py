from agents.mcp import MCPServerStreamableHttp

from app.shared.config import mcp_config
from app.shared.runtime.logger import logger, step_log
import asyncio
import json

# 从全局配置读取MCP联网搜索服务地址、密钥
DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key


@step_log("validate_web_search_inputs")
def validate_web_search_inputs(state: dict) -> str:
    """
    校验联网检索前置参数
    联网搜索依赖改写后的标准问句，无有效问句则无法执行检索
    Args:
        state: 查询流程图全局状态
    Returns:
        str: 校验通过的标准化检索问句
    """
    rewritten_query = state.get("rewritten_query")
    # 联网搜索必须依赖语义完整的改写问句，原始口语化问句检索效果差
    if not rewritten_query:
        logger.error("rewritten_query不能为空!")
        raise ValueError("rewritten_query不能为空!")
    return rewritten_query


@step_log("search_web_documents_async")
async def search_web_documents_async(rewritten_query: str, count: int = 5):
    """
    异步调用MCP联网搜索工具，获取互联网检索结果
    Args:
        rewritten_query: 标准化检索问句
        count: 返回网页检索结果条数
    Returns:
        联网搜索工具原始返回数据对象
    """

    # 初始化MCP流式HTTP服务端
    mcp_server = MCPServerStreamableHttp(
        name="search_mcp",
        client_session_timeout_seconds=300,
        params={
            "url": DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300,
        },
    )

    try:
        await mcp_server.connect()
        # 打印当前可用工具列表
        tool_list = await mcp_server.list_tools()
        logger.info(f"当前可用工具列表: {tool_list}")
        return await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": count,
            },
        )
    finally:
        # 无论成功失败，最终释放连接资源，避免连接堆积
        await mcp_server.cleanup()


@step_log("search_by_web")
def search_by_web(state: dict, count: int = 10) -> list[dict]:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    适配流程图同步调用规范，封装异步联网搜索能力
    Args:
        state: 查询流程图全局状态
        count: 最大返回网页结果条数，默认10条
    Returns:
        list[dict]: 结构化网页检索结果列表
    """
    # 校验检索前置参数完整性
    rewritten_query = validate_web_search_inputs(state)

    # 执行异步联网搜索
    mcp_result = asyncio.run(search_web_documents_async(rewritten_query, count=count))

    # 解析返回的json
    text_dict = json.loads(mcp_result.content[0].text)

    # 提取网页详情列表返回，无结果则返回空列表
    return text_dict.get("pages", [])
