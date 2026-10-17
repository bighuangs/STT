"""
应用主包 / 接口层 / 数据模型层中的 query 模块，负责承载对应场景的具体实现逻辑。
"""
from typing import Any

from pydantic import BaseModel, Field

from app.domain.answer import AnswerEnvelope


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询内容")
    session_id: str | None = Field(default=None, description="会话ID，为空时自动生成")
    is_stream: bool = Field(default=False, description="是否使用SSE流式输出")
    kb_id: str | None = Field(default=None, description="可选知识库ID；服务端仍强制解析 active index")
    retrieval_filter: dict[str, Any] | None = Field(
        default=None,
        description="可选结构化检索过滤 AST；只接受业务元数据字段，系统隔离条件由服务端注入",
    )


class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer: str = ""
    done_list: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    answer_id: str = Field(default="", description="助手答案的稳定身份，可用于提交反馈")
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    retrieval_profile: dict[str, Any] = Field(
        default_factory=dict, description="本会话固定使用的已发布检索配置身份"
    )
    retrieval_diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="逐路召回、融合及故障诊断，不包含向量和密钥"
    )
    query_plan: dict[str, Any] = Field(
        default_factory=dict, description="服务端校验后的查询计划与实际路由预算"
    )
    fusion_diagnostics: dict[str, Any] = Field(
        default_factory=dict, description="校准、融合、去重及多样性选择诊断"
    )
    context_pack_summary: dict[str, Any] = Field(
        default_factory=dict, description="上下文 Token 预算和装包摘要，不返回未引用正文"
    )
    evidence_assessment: dict[str, Any] = Field(
        default_factory=dict, description="生成前证据充分性评估"
    )
    answer_decision: dict[str, Any] = Field(
        default_factory=dict, description="回答、部分回答、澄清、拒答或直接回答决策"
    )
    answer_envelope: AnswerEnvelope | None = Field(
        default=None, description="answer.v1 权威查询终态"
    )


class HistoryItem(BaseModel):
    id: str = Field(default="", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    image_urls: list[str] = Field(default_factory=list)
    answer_id: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    retrieval_profile: dict[str, Any] = Field(default_factory=dict)
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    answer_envelope: AnswerEnvelope | None = None
    ts: Any = None


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem] = Field(default_factory=list)
