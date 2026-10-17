"""
工具模块，负责提供 mongo history 相关的辅助能力。
"""
from typing import Any
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

from app.shared.runtime.logger import logger
from app.shared.config.settings_config import get_settings


class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类 (基于原生 PyMongo 实现)
    核心功能：封装MongoDB的连接、集合初始化、索引创建，为上层提供统一的数据库操作入口
    扩展功能：支持与LangChain消息对象的格式转换（原代码预留能力）
    """
    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库/集合获取、索引创建
        初始化失败会抛出异常并记录错误日志，确保程序感知连接问题
        """
        try:
            config = get_settings().mongo
            self.mongo_url = config.url
            self.db_name = config.database

            # 创建MongoDB客户端实例，建立与数据库的连接
            timeout_ms = int(get_settings().health.timeout_seconds * 1000)
            self.client = MongoClient(
                self.mongo_url,
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
            )
            # 获取指定名称的数据库对象 user 库
            self.db = self.client[self.db_name]
            # 获取对话记录的集合（相当于关系型数据库的表），集合名：chat_message  db.chat_message
            self.chat_message = self.db["chat_message"]

            # 为chat_message集合创建复合索引，提升查询性能
            # 索引规则：session_id升序 + ts降序，适配"按会话查最新记录"的核心查询场景
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])
            self.chat_message.create_index([("answer_id", 1)], sparse=True, unique=True)

            # 记录成功日志，确认数据库连接和初始化完成
            logger.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            # 捕获所有初始化异常，记录详细错误日志
            logger.error(f"MongoDB 初始化失败：{type(e).__name__}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise


# 定义全局变量：存储HistoryMongoTool的单例实例
# 作用：避免多次创建HistoryMongoTool实例，从而避免重复建立MongoDB连接
_history_mongo_tool: HistoryMongoTool | None = None
def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例（懒加载模式）
    核心逻辑：全局实例为空时创建，不为空时直接返回，保证整个程序只有一个数据库连接实例
    :return: HistoryMongoTool的单例实例
    """
    # 声明使用全局变量，避免函数内视为局部变量
    global _history_mongo_tool
    # 懒加载：仅当全局实例为空时，才创建新的实例
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # 返回单例实例
    return _history_mongo_tool


def close_history_mongo_tool() -> None:
    """由 FastAPI/Worker 生命周期调用，释放 Mongo 连接池。"""
    global _history_mongo_tool
    if _history_mongo_tool is not None:
        _history_mongo_tool.client.close()
        _history_mongo_tool = None



def clear_history(
    session_id: str, tenant_id: str | None = None, user_id: str | None = None
) -> int:
    """
    清空指定会话的所有历史对话记录
    :param session_id: 会话唯一标识，用于筛选要删除的记录
    :return: 实际删除的文档数量，删除失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式避免重复创建数据库连接
    mongo_tool = get_history_mongo_tool()
    try:
        # 执行批量删除操作：删除所有session_id匹配的文档
        query = {"session_id": session_id}
        if tenant_id is not None:
            query["tenant_id"] = tenant_id
        if user_id is not None:
            query["user_id"] = user_id
        result = mongo_tool.chat_message.delete_many(query)
        # 记录删除成功日志，包含删除数量和会话ID，便于问题排查
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        # 返回实际删除的数量（delete_many的返回对象包含deleted_count属性）
        return result.deleted_count
    except Exception as e:
        # 捕获删除异常，记录错误日志，包含会话ID
        logger.error(f"Error clearing history for session {session_id}: {e}")
        # 异常时返回0，标识删除失败
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        image_urls: list[str] | None = None,
        message_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        answer_id: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        sources: list[dict[str, Any]] | None = None,
        citation_validation: dict[str, Any] | None = None,
        retrieval_profile: dict[str, Any] | None = None,
        retrieval_diagnostics: dict[str, Any] | None = None,
        answer_envelope: dict[str, Any] | None = None,
) -> str:
    """
    写入/更新单条会话记录到MongoDB
    支持两种模式：无message_id时新增记录，有message_id时更新已有记录
    :param session_id: 会话唯一标识，关联对话所属的会话
    :param role: 消息角色，固定值：user（用户）/assistant（助手）
    :param text: 对话核心内容，用户的提问或助手的回答
    :param rewritten_query: 重写后的查询语句（可选，用于检索增强等场景，默认空字符串）
    :param image_urls: 关联的图片URL列表（可选，默认None）
    :param message_id: 记录主键ID（可选，有值则更新，无值则新增）
    :return: 插入/更新的记录唯一标识（新增返回ObjectId字符串，更新返回传入的message_id）
    """
    # 生成当前时间的时间戳（秒级），用于记录消息的创建时间，后续用于排序和查询
    ts = datetime.now().timestamp()

    # 构造要插入/更新的文档数据（MongoDB的基本数据单元是文档，类似Python字典）
    document = {
        "session_id": session_id,  # 会话ID，关联维度
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 重写查询，空值处理为空字符串
        "image_urls": image_urls,  # 关联图片URL列表
        "citations": citations or [],
        "sources": sources or [],
        "citation_validation": citation_validation or {},
        "retrieval_profile": retrieval_profile or {},
        "retrieval_diagnostics": retrieval_diagnostics or {},
        "answer_envelope": answer_envelope or {},
        "ts": ts  # 时间戳，排序和时间筛选维度
    }
    if answer_id:
        document["answer_id"] = answer_id
    if tenant_id is not None:
        document["tenant_id"] = tenant_id
    if user_id is not None:
        document["user_id"] = user_id

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    # 判断是否传入主键ID，区分更新/新增逻辑
    if message_id:
        # 有message_id：执行更新操作（根据主键更新）
        update_filter = {"_id": ObjectId(message_id)}
        if tenant_id is not None:
            update_filter["tenant_id"] = tenant_id
        if user_id is not None:
            update_filter["user_id"] = user_id
        result = mongo_tool.chat_message.update_one(
            update_filter,
            {"$set": document}  # 更新操作：$set表示只更新指定字段，保留其他字段
        )
        # 更新操作返回传入的message_id作为标识
        return answer_id or message_id
    else:
        # 无message_id：执行新增操作
        result = mongo_tool.chat_message.insert_one(document)
        # 新增操作返回插入的ObjectId并转为字符串，便于上层使用（避免直接返回ObjectId对象）
        return answer_id or str(result.inserted_id)


def get_recent_messages(
    session_id: str, limit: int = 10,
    tenant_id: str | None = None, user_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录，返回原始字典格式
    结果按时间正序排列，可直接喂给LLM作为上下文
    :param session_id: 会话唯一标识，用于筛选指定会话的记录
    :param limit: 条数限制，默认返回最近10条
    :return: 对话记录列表（字典格式），查询失败返回空列表
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 构造查询条件：仅查询指定session_id的记录
        query = {"session_id": session_id}
        if tenant_id is not None:
            query["tenant_id"] = tenant_id
        if user_id is not None:
            query["user_id"] = user_id

        # 执行查询：按时间戳倒序取最近记录，限制返回条数
        # find(query)：获取符合条件的游标（惰性加载，不立即查询）
        # sort("ts", -1)：按ts字段倒序（从新到旧），用于快速获取最近消息
        # limit(limit)：限制返回的最大条数
        cursor = mongo_tool.chat_message.find(query).sort("ts", -1).limit(limit)
        # 将游标转为列表，触发实际数据库查询，获取所有符合条件的文档
        messages = list(cursor)
        # 返回查询结果列表
        return messages
    except Exception as e:
        # 捕获查询异常，记录错误日志
        logger.error(f"Error getting recent messages: {e}")
        # 异常时返回空列表，避免上层处理None报错
        return []


# 主程序入口：仅当直接运行该脚本时执行，用于简单的功能测试
if __name__ == "__main__":
    # 简单测试代码：验证数据库的写入和查询功能是否正常
    # 测试会话ID，用于标识测试的对话记录
    sid = "000015_hybrid"
    # 1. 写入用户消息（手动指定ts=1000，便于测试排序）
    save_chat_message(sid, "user", "你好 (Hybrid)")
    # 2. 写入助手回复（手动指定ts=1001，按时间顺序紧跟用户消息）
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo + LangChain 对象的助手。")
    # 3. 写入带改写问题的用户消息（手动指定ts=1002，测试 rewritten_query 字段）
    save_chat_message(sid, "user", "这个万用表怎么换电池？", rewritten_query="混合万用表怎么换电池？")

    # 4. 查询指定会话的最近5条记录，验证查询功能
    print("--- 查询 LangChain 对象记录 ---")
    messages = get_recent_messages(sid, limit=5)
    # 打印查询到的记录数量
    print(f"查询到的记录数: {len(messages)}")
    # 遍历打印每条记录的详细内容
    for m in messages:
        print(f" {m}  ")
