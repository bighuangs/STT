"""TODO-27 确定性查询理解、改写校验、分解和计划服务。

第一版采用“规则优先 + 结构化 proposal 可选”的方式：硬规则负责直接回答、关键
歧义、时效和数据外发边界；未来 LLM 只能提交 ``QueryPlan`` proposal，并且必须
经过本服务的 schema、作用域、策略和硬规则复核。
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import re
from typing import Any, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.query_planning import (
    AuthorizedScopeRef,
    ConversationTurn,
    FACET_LEDGER_SCHEMA_VERSION,
    FacetClosure,
    FacetLedger,
    FacetNecessity,
    FacetPolarity,
    MAX_QUERY_LENGTH,
    MAX_FACET_EVIDENCE_IDS,
    MAX_QUERY_FACETS,
    PlanDisposition,
    PlanningPolicy,
    PLANNER_VERSION,
    QueryAmbiguity,
    QueryConstraint,
    QueryFacet,
    QueryIntent,
    QueryLanguage,
    QueryPlan,
    QueryRoute,
    QueryRouteType,
    QuerySubQuery,
    QueryUnderstanding,
    RewriteDecision,
    RewriteStatus,
    RouteFallback,
    RouteFilter,
    SubQueryPurpose,
    TemporalKind,
    TemporalRequirement,
    make_plan_id,
    validate_plan_for_execution,
)


_GREETING_ONLY = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好|谢谢|感谢|再见)[!！。,.，\s]*$",
    re.IGNORECASE,
)
_HELP_WORDS = ("怎么", "如何", "在哪", "哪里", "使用", "操作", "帮助")
_GROUNDED_REFERENCE = re.compile(
    r"(?:手册|说明书|操作指南|用户指南|"
    r"(?:文档|资料)(?:中|内|里|所述)|"
    r"根据(?:该|这份|当前)?(?:文档|资料|手册|说明书)|"
    r"原文|引用|证据|页码|第\s*\d+\s*页|图示|插图|表格|"
    r"(?:展示|提供|给出|召回|加载).{0,12}(?:图片|截图)|"
    r"(?:图片|截图).{0,12}(?:页码|手册|文档|证据))"
)
_EXPLICIT_STT_SURFACE = re.compile(
    r"(?:STT|掌柜智库|知识库工作台|"
    r"(?:当前|本|这个|该)(?:页面|网页|工作台|系统)|"
    r"顶部|侧边栏|导航栏|输入框)",
    re.IGNORECASE,
)
_STT_APP_ACTION = re.compile(
    r"(?:(?:切换|选择|创建|删除|编辑|发布|重建)知识库|"
    r"(?:上传|导入|删除|预览)(?:文档|文件)|"
    r"(?:清空|新建|归档|恢复|删除)(?:对话|会话)|"
    r"(?:登录|退出|登出)(?:知识库|工作台|STT|掌柜智库|系统)|"
    r"(?:查看|刷新)(?:导入任务|查询运行|任务进度))",
    re.IGNORECASE,
)
_UI_CONTROL_ACTION = re.compile(
    r"(?:(?:登录|退出|登出|上传|导入)|"
    r"(?:点击|选择|查看|找到|定位).{0,12}(?:按钮|选项|菜单|进度条))"
)
_DOMAIN_CONTENT = re.compile(
    r"(?:设备|打印机|扫描仪|驱动|固件|产品|型号|机器|硬件|耗材|"
    r"纸盒|卡纸|墨盒|硒鼓|安装|维护|维修|故障)"
)
_CURRENT_TIME = re.compile(r"(最新|目前|现在|当前|今天|今日|实时|刚刚|近期)")
_AS_OF = re.compile(
    r"(截至\s*(?:\d{4}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2}日?)?|"
    r"昨天|今日|今天|目前|现在|当前))"
)
_RANGE = re.compile(
    r"((?:\d{4}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2}日?)?)\s*"
    r"(?:至|到|—|-)\s*(?:\d{4}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2}日?)?))"
)
_WEAK_CURRENT_TIME = re.compile(r"(?:目前|现在|当前)")
_EXTERNAL_RECENCY = re.compile(r"(?:最新|今天|今日|实时|刚刚|近期|截至)")
_EXTERNALLY_VOLATILE_FACT = re.compile(
    r"(?:官网|新闻|公告|通知|召回|漏洞|CVE|补丁|价格|售价|报价|库存|"
    r"供货|促销|折扣|销量|市场行情|服务状态|在线状态|停机|可用性|"
    r"(?:固件|驱动|软件|系统|应用).{0,8}(?:版本|发布|更新)|"
    r"版本号|发布日期|更新时间)",
    re.IGNORECASE,
)
_MANUAL_STRUCTURE = re.compile(
    r"(?:菜单(?:结构|项)?|选项|设置(?:项|菜单)?|参数(?:项)?|状态码|指示灯)"
)
_STATIC_CAPABILITY = re.compile(r"(?:支持|兼容|适用|适合|包含|提供|具备|配备|允许|可用|可选)")
_STATIC_OPERATION_SEMANTICS = re.compile(
    r"(?:菜单(?:结构|项)?|选项|设置(?:项|菜单)?|参数(?:项)?|功能|作用|"
    r"用于|表示|代表|含义|是什么意思|如何查看|怎么查看|在哪查看|哪里查看)"
)
_CURRENT_CLAUSE_SPLIT = re.compile(r"[,，。.!！?？;；]+")

_FreshnessScope = Literal["external", "authorized_kb_current"]
_EXTERNAL_FRESHNESS: _FreshnessScope = "external"
_AUTHORIZED_KB_CURRENT: _FreshnessScope = "authorized_kb_current"
_CHAPTER_TITLE_INTENT = re.compile(
    r"(?:标题|题名|章名|节名|heading|title)",
    re.IGNORECASE,
)
_DOCUMENT_TITLE_INTENT = re.compile(
    r"(?:(?:文档|文件|手册|资料)(?:的)?(?:标题|题名)|"
    r"(?:document|file|manual)\s+(?:heading|title))",
    re.IGNORECASE,
)
_CHAPTER_TITLE_CLAUSE_BOUNDARY = re.compile(r"[。！？!?；;\r\n]+|(?<!\d)\.|\.(?!\d)")
_DOTTED_CHAPTER_REFERENCE = re.compile(
    r"(?:"
    r"(?:第\s*)?(?P<suffix_path>\d+(?:\s*\.\s*\d+)+)\s*(?:章|节|章节)|"
    r"(?:章节|章|节)(?:编号|号)?\s*[:：]?\s*(?P<prefix_path>\d+(?:\s*\.\s*\d+)+)|"
    r"(?:chapter|section)\s*(?P<english_path>\d+(?:\s*\.\s*\d+)+)"
    r")",
    re.IGNORECASE,
)
_MAX_CHAPTER_TITLE_REFERENCES = 4
_MODEL_ID = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{2,}[-_ ]?\d[A-Za-z0-9._-]*|"
    r"\d+[A-Za-z][A-Za-z0-9._-]*)(?![A-Za-z0-9])"
)
_QUOTED_ENTITY = re.compile(r"[“\"《](.{1,80}?)[”\"》]")
_QUESTION_AXIS: dict[str, tuple[str, ...]] = {
    "procedure": ("如何", "怎么", "步骤", "怎样"),
    "reason": ("为什么", "为何", "原因"),
    "definition": ("是什么", "什么是"),
    "location": ("哪里", "在哪"),
    "time": ("何时", "什么时候", "多久"),
    "quantity": ("多少", "几次", "几个"),
}
_ENGLISH_NEGATION = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|won't|mustn't|shouldn't)\b",
    re.IGNORECASE,
)
_CHINESE_NEGATION = re.compile(
    r"(?:禁止|不能|不得|不可|不应|不要|无需|无须|没有|无法|未能|勿|避免|从未|尚未|"
    r"不(?:支持|允许|需要|建议|执行|使用|安装|连接|打印|启动|工作|响应|显示|识别|"
    r"包括|包含|属于|接地|满足)|"
    r"没(?:有|能|法|检测|找到|发现)|"
    r"未(?:完成|连接|安装|启动|检测|识别|响应|找到|发现|配置|设置|校准|更新|满足))"
)
_VAGUE_REFERENTS = ("它", "这个", "那个", "该设备", "该产品", "上述", "前者", "后者")
_SCOPED_DOMAIN_REFERENT = re.compile(
    r"(?:该|本|当前|这个|这台|这款)(?:产品|设备|打印机|扫描仪|机器|型号|文档|手册|说明书|知识库)"
)
_PLURAL_REFERENT = re.compile(r"(?:它们|这些|上述|前述|前者|后者|两者)")
_COLLECTION_ANCHOR = re.compile(
    r"(?:哪些|哪几|多少|几种|列出|包括|包含|类型|种类|方式|步骤|条件|项目|"
    r"选项|部件|功能|热点|节点|对象|文件|文档|型号|菜单|参数|要求|情况)"
)
_TROUBLESHOOTING = ("故障", "报错", "错误", "失败", "无法", "异常", "不能", "怎么办")
_COMPARISON = (
    "对比",
    "比较",
    "区别",
    "差异",
    "相比",
    "哪个好",
    "哪个更好",
    "哪一个好",
)
_EXPLICIT_COMPARISON_PAIR = re.compile(
    r"(?:比较|对比)\s*(?P<left>.{2,80}?)\s*(?:与|和|及)\s*"
    r"(?P<right>.{2,80}?)(?:[:：,，;；?？]|$)"
)
_PROCEDURE = ("如何", "怎么", "步骤", "操作", "使用", "安装", "维护", "校准", "启动", "关机")
_HARD_CLAUSE_SPLIT = re.compile(r"[?？;；]+")
_SOFT_CLAUSE_CONNECTOR = re.compile(r"(?:并且|同时|另外|以及)")
_INDEPENDENT_INFORMATION_CLAUSE = re.compile(
    r"(?:如何|怎么|怎样|为什么|为何|是什么|什么是|哪里|在哪|何时|什么时候|多久|"
    r"多少|几次|几个|哪些|哪几|是否.+|能否.+|可否.+|有没有|有何|列出|说明|"
    r"比较|对比|介绍|给出|什么(?:前提条件|条件|要求|信息|内容|操作))"
)
_COORDINATED_EACH = re.compile(
    r"^(?P<left>.{2,}?)\s*(?:和|与|以及)\s*(?P<right>.{2,}?)"
    r"(?:[,，]\s*)?(?:分别|各)(?P<suffix>.{2,})$"
)
_SHARED_ATTRIBUTE_CONTEXT = re.compile(
    r"(?:^(?:当|如果|若|在).{2,48}?(?:时|下|中|内)[,，]|"
    r"(?:其|(?:该|本|这个|这台|这款)(?:产品|设备|打印机|扫描仪|机器|型号)?的)|"
    r"(?:产品|设备|打印机|扫描仪|机器|型号)的)"
)
_SHARED_ATTRIBUTE_PREDICATE = re.compile(
    r"^(?:是什么|是多少|有多少|有哪些|包括哪些|包含哪些)(?:[?？]|$)"
)
_EXPLICIT_ATTRIBUTE_OWNER = re.compile(
    r"(?:其|(?:该|本|这个|这台|这款)?(?:产品|设备|打印机|扫描仪|机器|型号)的|"
    r"^(?:当|如果|若|在))"
)
_TRIM_PUNCTUATION = " \t\r\n,，。.!！?？:：;；、"
_STOP_CHARS = set("的了呢吗啊呀请问一下如何怎么什么是否能否可以")
_CONTEXT_DEPENDENT_CLAUSE = re.compile(
    r"^(?:其|该|此|这|这些|上述|前者|后者|两者|它|否则|与|和|相比|相较)"
)
_RELATIONAL_RESULT_DEPENDENCY = re.compile(r"^(?:对应(?:的)?|解决办法中)")
_COLLECTION_RESULT_DEPENDENCY = re.compile(
    r"^(?:(?:它们|这些(?:项目|对象|记录|条目|纸张|热点|原因|故障)?)(?:的|应)|"
    r"其中(?:哪个|哪些|哪项|哪几项|的)|(?:请)?列出两者都)"
)
_RESULT_SET_ANTECEDENT = re.compile(
    r"(?:哪些|哪几|所有|全部|分别|两者|类型|种类|项目|条目|记录|原因|故障|列表|表中)"
)
_RESULT_SET_DEPENDENCY_BOUNDARY = re.compile(
    r"[?？;；]\s*(?:(?:对应(?:的)?|解决办法中)|"
    r"(?:(?:它们|这些(?:项目|对象|记录|条目|纸张|热点|原因|故障)?)(?:的|应))|"
    r"其中(?:哪个|哪些|哪项|哪几项|的)|(?:请)?列出两者都)"
)
_CONDITIONAL_CLAUSE = re.compile(r"^(?:如果|若|如未|当)")
_CONDITIONAL_ANAPHORIC_REFERENCE = re.compile(
    r"(?:这些|上述|前述|前者|后者|两者|它们|其(?:中|他)?|"
    r"(?:该|此)(?:产品|设备|打印机|扫描仪|机器|型号|文档|手册|条件|步骤|"
    r"情况|操作|项目|项|值|结果|内容|问题|方式|功能))"
)
_CONDITIONAL_DEPENDENCY_REFERENCE = re.compile(
    r"(?:这些|上述|前述|前者|后者|两者|它们|"
    r"(?:该|此)(?:条件|步骤|情况|操作|项目|项|值|结果|内容|问题|方式|功能))"
)
_ELLIPTICAL_ANSWER_CONDITION = re.compile(
    r"^(?:如果|若)\s*(?:(?:该|此)?(?:产品|设备|打印机|扫描仪|机器|型号))?\s*"
    r"(?:支持|不支持|具备|不具备|允许|不允许|可以|不可以|能够|不能|存在|不存在|"
    r"满足|不满足|是|不是)\s*(?:时)?\s*$"
)
_FALLBACK_RESULT_CONDITION = re.compile(
    r"(?:自动恢复|重试|再次尝试|重新尝试|"
    r"(?:上述|前述|该|此)(?:操作|处理|步骤|尝试))"
    r".{0,8}?(?:失败|无效|未成功|仍未|无法|不能)"
)
_ENGLISH_DEPENDENT_CONDITION = re.compile(
    r"^(?:if|when)\s+(?:it|they|this|that|these|those|the\s+(?:result|step|condition))\b|"
    r"^(?:if|when)\s+(?:supported|unsupported|successful|unsuccessful)\b|"
    r"^(?:if|when).{0,40}\b(?:retry|recovery)\s+(?:fails?|failed)\b",
    re.IGNORECASE,
)
_SOURCE_RESPONSE_INSTRUCTION = re.compile(
    r"^(?:请)?(?:分别|逐项|依次)?(?:依据|根据|结合|参考)"
    r"(?P<source>.{1,96}?)(?:分别)?(?:作答|回答|说明|描述|给出答案)$"
)
_SOURCE_ONLY_RESPONSE_TAIL = re.compile(
    r"(?:证据|资料|文档|手册|内容|(?:安装)?流程|"
    r"(?:安装向导|设置|操作)?界面(?:信息)?)"
    r"(?:中(?:的)?(?:具体)?(?:操作说明|事实|信息|内容|说明|操作)?)?$"
)
_PURE_OUTPUT_INSTRUCTION = re.compile(
    r"^(?:请)?(?:分别|逐项)?(?:"
    r"(?:(?:按(?:照)?顺序|依次|逐步)(?:说明|列出|描述)"
    r"(?:具体|完整)?(?:操作)?(?:步骤|流程|内容)?|"
    r"(?:说明|描述)(?:具体|完整)(?:操作|下载)?(?:步骤|流程|路径)|"
    r"列出(?:所有(?:相关)?|全部)(?:内容|检查点|事项|步骤)"
    r"))$"
)
_OPEN_ENUMERATION = re.compile(
    r"(?:哪些|哪几|所有|全部|分别|各自|逐项|逐一|"
    r"(?:需要|应当|应该|必须|要)(?:先)?"
    r"(?:知道|了解|掌握|提供|填写|输入|确认|检查|准备|获取|满足)"
    r"什么(?:信息|内容|事项|条件|前提条件|要求)|"
    r"列出.{0,20}(?:尺寸|项目|内容|菜单项|检查点|操作)|"
    r"(?:列出|说明|给出).{0,12}(?:步骤|事项|要求|条件)|"
    r"(?:步骤|事项|要求|条件).{0,8}(?:有哪些|是什么|包括|包含)|"
    r"(?:适用|使用|运行|工作|部署|安装|安全)(?:的)?"
    r"(?:地理)?(?:环境)?限制.{0,4}(?:有哪些|是什么|包括|包含))"
)
_SCOPED_MULTI_STEP_PROCEDURE = re.compile(
    r"(?:如何|怎么|怎样)\s*(?:通过|利用|借助|在|从|用)"
    r".{1,80}?(?:打印|安装|配置|设置|添加|连接|进入|查看|查询|获取|启动|"
    r"打开|启用|禁用|关闭|取消|恢复|更换|清洁|校准|操作|执行|完成|导出|"
    r"导入|发送|扫描|复印)"
)
_EXPLICIT_MULTI_STEP_STRUCTURE = re.compile(
    r"(?:"
    r"(?:列出|给出|写出|枚举|说明|描述).{0,80}?(?:具体|完整)?(?:操作)?步骤|"
    r"(?:哪些|哪几|几个|所有|全部|\d{1,3}|[一二两三四五六七八九十])"
    r"\s*(?:个|项|条|步)?\s*(?:具体|完整)?(?:操作)?步骤|"
    r"(?:具体|完整)?(?:操作)?步骤(?:是什么|有哪些|包括|包含)|"
    r"的(?:具体|完整)?操作步骤$|"
    r"(?:操作|处理|配置)流程(?:是什么|有哪些|包括|包含|如何|怎么|怎样|$)"
    r")"
)
_ORDERED_RESPONSE_STRUCTURE = re.compile(
    r"(?:分别|逐项)?(?:按(?:照)?顺序|依次|逐步)(?:说明|列出|描述)"
)
_COMPLETE_PROCEDURE_RESPONSE_STRUCTURE = re.compile(
    r"(?:说明|描述)(?:具体|完整)(?:操作)?(?:步骤|流程)"
)
_COMPLETE_RESPONSE_STRUCTURE = re.compile(r"列出(?:所有(?:相关)?|全部)(?:内容|检查点|事项|步骤)")
_ORDERED_ACTION_COLLECTION = re.compile(
    r"(?:需要|应当|应该|应|需|必须|要).{0,20}?"
    r"(?:完成|执行|遵循|进行)(?:哪些|哪几)"
    r"[^。！？?；;]{0,24}?(?:操作|步骤)"
)
_CONTEXTUAL_MULTI_STEP_PROCEDURE = re.compile(
    r"(?:在|通过|使用|利用|借助|从).{1,80}?(?:中|上|内|后)?[,，]?"
    r"(?:应)?(?:如何|怎么|怎样)(?:操作|开启|关闭|启用|禁用|设置|配置|更改|调整|恢复)"
)
_ACTION_INFORMATION_NEED = re.compile(
    r"(?:"
    r"(?:如何|怎么|怎样).{0,40}?"
    r"(?:打印|安装|配置|设置|添加|连接|进入|查看|查询|获取|启动|打开|启用|禁用|"
    r"关闭|取消|恢复|更换|清洁|校准|装纸|取纸|排纸|处理|操作|执行|完成|导出|"
    r"导入|发送|扫描|复印|检查|排除)|"
    r"(?:需要|应当|应该|应|需|必须|要).{0,32}?"
    r"(?:完成|执行|遵循|进行|处理|安装|配置|设置|连接|恢复|更换|清洁|校准|装纸|"
    r"取纸|排纸|检查|排除)"
    r")"
)
_COUNT_VALUE = r"(?P<count>\d{1,3}|[一二两三四五六七八九十])"
_COUNT_CLASSIFIER = r"(?:个|项|条|种|步|点|类|处|份|组|次)"
_COUNTED_NOUN = re.compile(rf"(?<!第){_COUNT_VALUE}\s*{_COUNT_CLASSIFIER}")
_ANSWER_BOUND_WH_COUNT = re.compile(rf"哪(?!些|几)\s*{_COUNT_VALUE}\s*{_COUNT_CLASSIFIER}")
_PLURAL_WH_ANSWER = re.compile(r"(?:哪些|哪几)")
_ANSWER_BOUND_POSTPOSED_COUNT = re.compile(
    rf"(?:有|共有|共计|共|包括|包含)\s*{_COUNT_VALUE}\s*{_COUNT_CLASSIFIER}"
    r"[^。！？?]{0,16}?[,，]?\s*(?:分别|各自)\s*"
    r"(?:是|为|包括|包含)?(?:什么|哪些|哪几)"
)
_ANSWER_REQUEST_VERB = re.compile(r"(?:列出|给出|写出|枚举|说明)")
_ANSWER_COUNT_SCOPE_PREFIX = re.compile(r"(?:在|当|按|根据|每|支持|满足|包含|采用)\s*$")
_ANSWER_COUNT_SCOPED_SUFFIX = re.compile(r"(?:下|时|中|内|期间|情况下|条件下)(?:的|所需)?")
_ANSWER_COUNT_NOUN_SUFFIX = re.compile(
    r"(?:步骤|事项|要求|条件|方式|方法|原因|部件|组件|项目|答案|内容|模式|"
    r"功能|类型|类别|方案|要点|位置|场景|协议)$"
)


class QueryPlanningInput(BaseModel):
    """planner 的完整输入快照；所有线上副作用均由调用方在进入服务前完成。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str = Field(min_length=1, max_length=4096)
    history: tuple[ConversationTurn, ...] = Field(default=(), max_length=20)
    rewrite_candidate: str | None = Field(default=None, max_length=4096)
    authorized_scope_ref: AuthorizedScopeRef
    profile_version: int = Field(default=0, ge=0)
    policy: PlanningPolicy = Field(default_factory=PlanningPolicy)
    deadline_at: datetime | None = None
    route_filters: tuple[RouteFilter, ...] = Field(default=(), max_length=32)

    @field_validator("original_query")
    @classmethod
    def normalize_original_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("original_query cannot be blank")
        return normalized

    @field_validator("rewrite_candidate")
    @classmethod
    def normalize_rewrite_candidate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_deadline(self) -> "QueryPlanningInput":
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")
        return self


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


def _question_axes(value: str) -> set[str]:
    lowered = value.lower()
    return {
        axis
        for axis, markers in _QUESTION_AXIS.items()
        if any(marker in lowered for marker in markers)
    }


def _contains_negation(value: str) -> bool:
    return bool(_ENGLISH_NEGATION.search(value) or _CHINESE_NEGATION.search(value))


def _content_bigrams(value: str) -> set[str]:
    compact = "".join(char for char in _normalize_for_match(value) if char not in _STOP_CHARS)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def extract_entities(query: str) -> tuple[str, ...]:
    """抽取规划所需的保守实体锚点，不尝试替代完整 NER。"""

    entities: list[str] = []
    for match in _MODEL_ID.finditer(query):
        entity = re.sub(r"[\s_-]+", "", match.group(0)).upper()
        if entity not in entities:
            entities.append(entity)
    for match in _QUOTED_ENTITY.finditer(query):
        entity = match.group(1).strip()
        if entity and entity not in entities:
            entities.append(entity)

    if any(marker in query for marker in _COMPARISON):
        before_marker = query
        for marker in _COMPARISON:
            if marker in before_marker:
                before_marker = before_marker.split(marker, 1)[0]
                break
        parts = re.split(r"\s*(?:和|与|跟|vs\.?|VS\.?)\s*", before_marker)
        if len(parts) >= 2:
            for part in parts[-2:]:
                candidate = part.strip(_TRIM_PUNCTUATION)
                candidate = re.sub(r"^(?:请|帮我|比较|对比)\s*", "", candidate)
                candidate = re.sub(r"(?:之间|两者|的)$", "", candidate).strip()
                if 1 < len(candidate) <= 80 and candidate not in entities:
                    entities.append(candidate)
    return tuple(entities[:32])


def validate_intent_preserving_rewrite(
    original_query: str,
    candidate: str,
) -> tuple[bool, float, tuple[str, ...]]:
    """用可解释硬约束拦截改变用户意图的 standalone rewrite。

    该检查不是通用语义相似模型，而是高精度安全闸：数字/型号、否定语义和疑问
    维度必须保持；内容锚点需有最低重合。低分时始终退回原问题。
    """

    original = " ".join(original_query.split())
    rewritten = " ".join(candidate.split())
    if not rewritten:
        return False, 0.0, ("rewrite_blank",)
    if _normalize_for_match(original) == _normalize_for_match(rewritten):
        return True, 1.0, ("rewrite_equivalent",)
    if len(rewritten) > len(original) * 4 + 80:
        return False, 0.1, ("rewrite_expansion_too_large",)

    reasons: list[str] = []
    original_anchors = {
        anchor.upper()
        for anchor in re.findall(r"[A-Za-z]+\d[A-Za-z0-9._-]*|\d+(?:\.\d+)?", original)
    }
    rewritten_anchors = {
        anchor.upper()
        for anchor in re.findall(r"[A-Za-z]+\d[A-Za-z0-9._-]*|\d+(?:\.\d+)?", rewritten)
    }
    if not original_anchors.issubset(rewritten_anchors):
        return False, 0.15, ("rewrite_lost_identifier_or_number",)

    if _contains_negation(original) != _contains_negation(rewritten):
        return False, 0.2, ("rewrite_changed_negation",)

    original_axes = _question_axes(original)
    rewritten_axes = _question_axes(rewritten)
    if original_axes and not original_axes.issubset(rewritten_axes):
        return False, 0.25, ("rewrite_changed_question_axis",)

    original_bigrams = _content_bigrams(original)
    rewritten_bigrams = _content_bigrams(rewritten)
    lexical_overlap = (
        len(original_bigrams & rewritten_bigrams) / len(original_bigrams)
        if original_bigrams
        else 1.0
    )
    anchor_score = (
        1.0
        if not original_anchors
        else (len(original_anchors & rewritten_anchors) / len(original_anchors))
    )
    axis_score = (
        1.0 if not original_axes else (len(original_axes & rewritten_axes) / len(original_axes))
    )
    score = round(0.5 * lexical_overlap + 0.3 * anchor_score + 0.2 * axis_score, 4)
    if lexical_overlap < 0.25 or score < 0.55:
        return False, score, ("rewrite_low_semantic_overlap",)
    reasons.append("rewrite_preserved_identifiers")
    reasons.append("rewrite_preserved_intent")
    return True, score, tuple(reasons)


def _history_rewrite(request: QueryPlanningInput) -> str | None:
    if not any(token in request.original_query for token in _VAGUE_REFERENTS):
        return None
    for turn in reversed(request.history):
        entities = extract_entities(turn.text)
        if not entities:
            continue
        candidate = request.original_query
        for referent in _VAGUE_REFERENTS:
            if referent in candidate:
                candidate = candidate.replace(referent, entities[0], 1)
                return candidate
    return None


def _select_standalone_query(
    request: QueryPlanningInput,
) -> tuple[str, RewriteDecision]:
    candidate = request.rewrite_candidate or _history_rewrite(request)
    if not candidate:
        return request.original_query, RewriteDecision(
            status=RewriteStatus.ORIGINAL,
            semantic_preservation_score=1.0,
            reasons=("no_rewrite_candidate",),
        )
    accepted, score, reasons = validate_intent_preserving_rewrite(
        request.original_query,
        candidate,
    )
    if accepted:
        return candidate, RewriteDecision(
            status=RewriteStatus.ACCEPTED,
            semantic_preservation_score=score,
            reasons=reasons,
        )
    return request.original_query, RewriteDecision(
        status=RewriteStatus.REJECTED,
        semantic_preservation_score=score,
        reasons=reasons,
    )


def _detect_language(query: str) -> QueryLanguage:
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", query))
    latin_count = len(re.findall(r"[A-Za-z]", query))
    if cjk_count and latin_count:
        return QueryLanguage.ZH if latin_count <= max(3, cjk_count // 2) else QueryLanguage.MIXED
    if cjk_count:
        return QueryLanguage.ZH
    if latin_count:
        return QueryLanguage.EN
    return QueryLanguage.UNKNOWN


def _detect_temporal(query: str) -> TemporalRequirement:
    range_match = _RANGE.search(query)
    if range_match:
        return TemporalRequirement(kind=TemporalKind.RANGE, expression=range_match.group(1))
    as_of_match = _AS_OF.search(query)
    if as_of_match:
        return TemporalRequirement(kind=TemporalKind.AS_OF, expression=as_of_match.group(1))
    current_match = _CURRENT_TIME.search(query)
    if current_match:
        return TemporalRequirement(
            kind=TemporalKind.CURRENT,
            expression=current_match.group(1),
        )
    return TemporalRequirement()


def _has_static_authorized_kb_current_semantics(query: str) -> bool:
    """Return whether weak current wording describes stable manual semantics.

    The authorized single-KB read path already binds local evidence to the active
    document version.  In that scope, wording such as ``当前支持`` or a menu item
    ``用于查看当前连接状态`` asks about the published manual, not a live external
    fact.  Keep this deliberately conservative: a domain/manual anchor and a
    static capability or operation marker must occur in the same clause as the
    weak current expression.
    """

    has_manual_anchor = bool(
        _GROUNDED_REFERENCE.search(query)
        or _DOMAIN_CONTENT.search(query)
        or _MANUAL_STRUCTURE.search(query)
    )
    if not has_manual_anchor:
        return False

    for clause in _CURRENT_CLAUSE_SPLIT.split(query):
        if not _WEAK_CURRENT_TIME.search(clause):
            continue
        if _STATIC_CAPABILITY.search(clause) or _STATIC_OPERATION_SEMANTICS.search(clause):
            return True
    return False


def _classify_freshness_scope(
    query: str,
    temporal: TemporalRequirement,
    *,
    single_kb_scope: bool,
) -> _FreshnessScope | None:
    """Separate external recency from the active authorized-KB snapshot.

    ``external`` retains the existing fresh-Web requirement.  The
    ``authorized_kb_current`` scope is only available for weak-current wording in
    a single authorized KB when the clause asks stable product/manual semantics.
    Historical ranges remain temporal constraints without implying live Web.
    """

    if temporal.kind == TemporalKind.NONE:
        return None
    if temporal.kind == TemporalKind.AS_OF or _EXTERNAL_RECENCY.search(query):
        return _EXTERNAL_FRESHNESS
    if temporal.kind == TemporalKind.RANGE:
        return None
    if _EXTERNALLY_VOLATILE_FACT.search(query):
        return _EXTERNAL_FRESHNESS
    if (
        single_kb_scope
        and _WEAK_CURRENT_TIME.search(query)
        and _has_static_authorized_kb_current_semantics(query)
    ):
        return _AUTHORIZED_KB_CURRENT
    return _EXTERNAL_FRESHNESS


def _is_ui_help(query: str) -> bool:
    """只把明确的 STT 工作台操作问题归为 UI 帮助。

    ``页面``、``界面`` 和 ``如何`` 也经常出现在产品手册问题中（例如要求展示
    驱动安装界面）。因此证据请求必须优先走知识库检索；产品/设备领域问题也不
    能仅因出现通用 UI 名词而绕过检索。没有证据或领域信号时，知识库、文档、
    会话等 STT 专属操作，或带明确工作台位置的操作问题，才使用内置页面引导。
    """

    if _GROUNDED_REFERENCE.search(query):
        return False

    has_help_word = any(word in query for word in _HELP_WORDS)
    if not has_help_word:
        return False

    has_explicit_surface = _EXPLICIT_STT_SURFACE.search(query) is not None
    has_stt_action = _STT_APP_ACTION.search(query) is not None
    has_ui_control_action = _UI_CONTROL_ACTION.search(query) is not None
    if has_explicit_surface and (has_stt_action or has_ui_control_action):
        return True

    if _MODEL_ID.search(query) or _DOMAIN_CONTENT.search(query):
        return False

    return has_stt_action


def _classify_intent(query: str) -> QueryIntent:
    normalized = query.strip()
    if _GREETING_ONLY.fullmatch(normalized):
        return QueryIntent.GREETING
    if _is_ui_help(query):
        return QueryIntent.UI_HELP
    if any(marker in query for marker in _COMPARISON):
        return QueryIntent.COMPARISON
    if any(marker in query for marker in _TROUBLESHOOTING):
        return QueryIntent.TROUBLESHOOTING
    if any(marker in query for marker in _PROCEDURE):
        return QueryIntent.PROCEDURE
    if _CURRENT_TIME.search(query) or _AS_OF.search(query) or _RANGE.search(query):
        return QueryIntent.TEMPORAL_FACT
    return QueryIntent.KNOWLEDGE_FACT


def _detect_ambiguity(
    query: str,
    *,
    intent: QueryIntent,
    entities: tuple[str, ...],
    single_kb_scope: bool,
) -> QueryAmbiguity:
    compact = query.strip(_TRIM_PUNCTUATION)
    if intent in {QueryIntent.GREETING, QueryIntent.UI_HELP}:
        return QueryAmbiguity()

    vague_referent = any(token in compact for token in _VAGUE_REFERENTS)
    scoped_domain_referent = bool(single_kb_scope and _SCOPED_DOMAIN_REFERENT.search(compact))
    same_query_antecedent = _has_same_query_collection_antecedent(compact)
    if vague_referent and not entities and not scoped_domain_referent and not same_query_antecedent:
        return QueryAmbiguity(
            is_ambiguous=True,
            missing_slots=("target",),
            clarification_question="请说明“它/这个”具体指哪台设备、哪份文档或哪个对象？",
        )
    comparison_pair = _EXPLICIT_COMPARISON_PAIR.search(compact)
    explicit_comparison_targets = bool(
        comparison_pair
        and len(_normalize_for_match(comparison_pair.group("left"))) >= 2
        and len(_normalize_for_match(comparison_pair.group("right"))) >= 2
    )
    if intent == QueryIntent.COMPARISON and len(entities) < 2 and not explicit_comparison_targets:
        return QueryAmbiguity(
            is_ambiguous=True,
            missing_slots=("comparison_targets",),
            clarification_question="请说明需要比较的两个或多个具体对象。",
        )
    generic_procedure = re.fullmatch(
        r"(?:请问)?(?:这个|它)?(?:要)?(?:怎么|如何)(?:使用|操作|处理|解决|办)",
        compact,
    )
    if generic_procedure:
        return QueryAmbiguity(
            is_ambiguous=True,
            missing_slots=("target",),
            clarification_question="请说明要使用或处理的具体设备、文档或问题。",
        )
    if entities and _normalize_for_match(compact) in {
        _normalize_for_match(entity) for entity in entities
    }:
        return QueryAmbiguity(
            is_ambiguous=True,
            missing_slots=("information_need",),
            clarification_question="请补充你希望了解该对象的哪一方面，例如安装、使用或安全要求。",
        )
    if len(_normalize_for_match(compact)) <= 2:
        return QueryAmbiguity(
            is_ambiguous=True,
            missing_slots=("information_need",),
            clarification_question="请补充你想了解的具体问题或范围。",
        )
    return QueryAmbiguity()


def _has_same_query_collection_antecedent(query: str) -> bool:
    """识别同一问题中复数代词之前已经出现的集合信息需求。"""

    for matched in _PLURAL_REFERENT.finditer(query):
        prefix = query[: matched.start()].rstrip(_TRIM_PUNCTUATION)
        if not prefix:
            continue
        clauses = [
            item.strip(_TRIM_PUNCTUATION)
            for item in re.split(r"[?？;；。!！]+", prefix)
            if item.strip(_TRIM_PUNCTUATION)
        ]
        if any(
            len(_normalize_for_match(clause)) >= 4 and _COLLECTION_ANCHOR.search(clause)
            for clause in clauses
        ):
            return True
    return False


def _coordinated_each_parts(query: str) -> tuple[str, ...]:
    """拆分 ``X 和 Y（时），各/分别……`` 形式的并列信息需求。

    普通的 ``A 和 B`` 可能只是同一事实中的联合宾语，不能一律拆分；``各`` 或
    ``分别`` 则明确要求两个方面分别作答。公共问题后缀会复制到两个检索问题，
    从而避免完整长问题的向量表示被其中一个方面（例如“校准”）主导。
    """

    compact = query.strip(_TRIM_PUNCTUATION)
    # 句末另有独立信息问题时，``分别`` 只修饰当前分句。若跨越问号匹配，会把
    # 后续交集问题错误复制到两个并列对象上；“请标明页码”一类输出要求仍共享。
    hard_parts = tuple(
        part.strip(_TRIM_PUNCTUATION)
        for part in _HARD_CLAUSE_SPLIT.split(compact)
        if part.strip(_TRIM_PUNCTUATION)
    )
    if len(hard_parts) > 1 and any(
        _looks_like_complete_information_clause(part) for part in hard_parts[1:]
    ):
        return ()
    matched = _COORDINATED_EACH.fullmatch(compact)
    if matched is None:
        return ()
    left = matched.group("left").strip(_TRIM_PUNCTUATION)
    right = matched.group("right").strip(_TRIM_PUNCTUATION)
    suffix = matched.group("suffix").strip(_TRIM_PUNCTUATION)
    # “X 和 Y 时，各……”中的“时/中/方面”是两个方面共享的语法连接词，
    # 不是第二个方面的专属检索词。
    right = re.sub(r"(?:时|中|方面)$", "", right).strip(_TRIM_PUNCTUATION)
    if min(len(left), len(right), len(suffix)) < 2:
        return ()
    # “设备的 A 和 B 分别是什么”中的 B 共享首项的实体/场景前提。把它拆成
    # 裸 ``B 是什么`` 会丢掉检索作用域；此类共享谓词属性组保留为完整查询。
    # 右项若自带 owner，则仍是两个可独立检索的方面。
    if (
        _SHARED_ATTRIBUTE_CONTEXT.search(left)
        and _SHARED_ATTRIBUTE_PREDICATE.match(suffix)
        and _EXPLICIT_ATTRIBUTE_OWNER.search(right) is None
    ):
        return (compact,)
    return (f"{left}{suffix}", f"{right}{suffix}")


def _looks_like_complete_information_clause(value: str) -> bool:
    """保守识别可独立检索的完整问句，拒绝 ``设备是否`` 一类残句。"""

    clause = value.strip(_TRIM_PUNCTUATION)
    if len(clause) < 3:
        return False
    if re.search(r"(?:是否|能否|可否|如何|怎么|怎样|哪些|哪几|为何|为什么)$", clause):
        return False
    return bool(_question_axes(clause) or _INDEPENDENT_INFORMATION_CLAUSE.search(clause))


def _split_soft_information_clauses(value: str) -> tuple[str, ...]:
    """只在连接词或逗号两侧都是完整信息需求时递归拆分。"""

    segment = value.strip(_TRIM_PUNCTUATION)
    if not segment:
        return ()
    for matched in _SOFT_CLAUSE_CONNECTOR.finditer(segment):
        left = segment[: matched.start()].strip(_TRIM_PUNCTUATION)
        right = segment[matched.end() :].strip(_TRIM_PUNCTUATION)
        if _looks_like_complete_information_clause(
            left
        ) and _looks_like_complete_information_clause(right):
            return (
                *_split_soft_information_clauses(left),
                *_split_soft_information_clauses(right),
            )
    for matched in re.finditer(r"[,，]", segment):
        left = segment[: matched.start()].strip(_TRIM_PUNCTUATION)
        right = segment[matched.end() :].strip(_TRIM_PUNCTUATION)
        if _looks_like_complete_information_clause(
            left
        ) and _looks_like_complete_information_clause(right):
            return (
                *_split_soft_information_clauses(left),
                *_split_soft_information_clauses(right),
            )
    return (segment,)


def _is_response_instruction_clause(value: str) -> bool:
    """识别不新增事实槽位的信源或输出组织指令。"""

    normalized = value.strip(_TRIM_PUNCTUATION)
    if not normalized:
        return False
    if _PURE_OUTPUT_INSTRUCTION.fullmatch(normalized):
        return True
    matched = _SOURCE_RESPONSE_INSTRUCTION.fullmatch(normalized)
    return bool(
        matched is not None
        and _SOURCE_ONLY_RESPONSE_TAIL.search(matched.group("source")) is not None
    )


def _merge_result_set_dependent_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    """合并必须复用前序结果集/表行的关系投影，避免重复证明同一记录组。"""

    merged: list[str] = []
    for part in parts:
        # 显式基数只约束当前分句的答案集合。若把 ``原因是什么？解决办法中哪两项``
        # 合成一个 facet，后句的数量会错误覆盖前句事实义务，且无法证明目标表列。
        # 检索表达式仍会在下游携带首句上下文，因此这里保留两个独立义务。
        has_local_closed_count = _facet_closure(part)[0] == FacetClosure.CLOSED_COUNT
        depends_on_previous = bool(
            merged
            and not has_local_closed_count
            and (
                _RELATIONAL_RESULT_DEPENDENCY.match(part)
                or (
                    _COLLECTION_RESULT_DEPENDENCY.match(part)
                    and _RESULT_SET_ANTECEDENT.search(merged[-1])
                )
            )
        )
        candidate = f"{merged[-1]}；{part}" if depends_on_previous else part
        if depends_on_previous and len(candidate) <= MAX_QUERY_LENGTH:
            merged[-1] = candidate
        else:
            merged.append(part)
    return tuple(merged)


def _decomposition_parts(query: str) -> tuple[str, ...]:
    response_instructions = _response_instruction_clauses(query)
    coordinated_query = query
    if response_instructions:
        semantic_hard_parts = tuple(
            normalized
            for raw in _HARD_CLAUSE_SPLIT.split(query)
            if (normalized := raw.strip(_TRIM_PUNCTUATION))
            and normalized not in response_instructions
        )
        coordinated_query = "？".join(semantic_hard_parts)
    coordinated = _coordinated_each_parts(coordinated_query)
    if coordinated:
        return coordinated
    parts: list[str] = []
    for hard_part in _HARD_CLAUSE_SPLIT.split(query):
        for item in _split_soft_information_clauses(hard_part):
            normalized = re.sub(
                r"^(?:并(?=在|通过|从|用|如何|怎么|怎样)|并且|同时|另外|以及)[,，:：\s]*",
                "",
                item.strip(_TRIM_PUNCTUATION),
            )
            if len(normalized) >= 3:
                parts.append(normalized)
    # 信源引用和“按顺序说明”只约束答案组织方式，不表达新的信息需求。把它们
    # 作为检索分支会稀释真正的多跳查询，还会制造无法从 Evidence 证明的伪义务。
    information_parts = tuple(item for item in parts if not _is_response_instruction_clause(item))
    return _merge_result_set_dependent_parts(information_parts)


def _requires_first_part_context(part: str) -> bool:
    """仅为真正承接前文的子句补首句上下文。"""

    if _CONTEXT_DEPENDENT_CLAUSE.match(part) or _RELATIONAL_RESULT_DEPENDENCY.match(part):
        return True
    return bool(_CONDITIONAL_CLAUSE.match(part) and _CONDITIONAL_ANAPHORIC_REFERENCE.search(part))


def _contextualize_decomposition_part(
    part: str,
    *,
    first_part: str,
    index: int,
) -> str:
    """让依赖前文的检索子句保留其主题，不尝试猜测具体指代对象。

    确定性拆句会把 ``其/这些/如果未核对这些条件……`` 一类承接子句从首句
    主题中剥离；直接检索该子句会造成中间流程语义丢失。条件句若已经明确给出
    自己的主语，则保持为独立检索分支，避免首分支反过来污染它。
    """

    if index <= 1 or not _requires_first_part_context(part):
        return part
    contextualized = f"{first_part}；{part}"
    # QuerySubQuery 的硬上限与原始问题一致；异常超长输入保持原子句，避免规划失败。
    return contextualized if len(contextualized) <= 4096 else part


def _safe_shared_entity(entities: tuple[str, ...]) -> str | None:
    """只允许明确的短实体补全独立子句，拒绝把整段疑问误当前缀。"""

    for entity in entities:
        compact = entity.strip(_TRIM_PUNCTUATION)
        if not 2 <= len(compact) <= 64:
            continue
        if _question_axes(compact):
            continue
        if compact.endswith(("的", "主要", "相关", "方面", "中的", "上的")):
            continue
        if any(
            marker in compact
            for marker in (
                "？",
                "?",
                "相比",
                "区别",
                "差异",
                "两者",
                "前者",
                "后者",
                "哪些",
                "什么",
                "为何",
                "为什么",
            )
        ):
            continue
        return compact
    return None


def _needs_multihop(query: str, intent: QueryIntent) -> bool:
    if intent == QueryIntent.COMPARISON:
        return True
    if len(_coordinated_each_parts(query)) >= 2:
        return True
    return len(_decomposition_parts(query)) >= 2


def _constraints(
    temporal: TemporalRequirement,
    *,
    freshness_scope: _FreshnessScope | None,
) -> tuple[QueryConstraint, ...]:
    constraints: list[QueryConstraint] = []
    if temporal.kind != TemporalKind.NONE:
        operator = (
            "between"
            if temporal.kind == TemporalKind.RANGE
            else ("as_of" if temporal.kind == TemporalKind.AS_OF else "eq")
        )
        constraints.append(
            QueryConstraint(
                name="temporal_requirement",
                operator=operator,
                value=temporal.expression or temporal.kind.value,
            )
        )
    if freshness_scope is not None:
        constraints.append(
            QueryConstraint(
                name="freshness_scope",
                operator="eq",
                value=freshness_scope,
            )
        )
    return tuple(constraints)


def extract_chapter_title_references(query: str) -> tuple[str, ...]:
    """提取高置信章节标题问题中的规范化章节路径。"""

    shadow = unicodedata.normalize("NFKC", str(query))
    if not shadow.strip():
        return ()

    paths: list[str] = []
    seen_paths: set[str] = set()
    for clause in _CHAPTER_TITLE_CLAUSE_BOUNDARY.split(shadow):
        # “文档标题 / document title”不是章节标题意图；删除这一局部短语后，
        # 同一分句仍须存在独立 title intent 与 dotted 章节引用。
        intent_shadow = _DOCUMENT_TITLE_INTENT.sub("", clause)
        if _CHAPTER_TITLE_INTENT.search(intent_shadow) is None:
            continue
        for matched in _DOTTED_CHAPTER_REFERENCE.finditer(clause):
            raw_path = next(value for value in matched.groupdict().values() if value)
            normalized_path = ".".join(part.strip() for part in raw_path.split("."))
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            paths.append(normalized_path)
            if len(seen_paths) >= _MAX_CHAPTER_TITLE_REFERENCES:
                return tuple(paths)
    return tuple(paths)


def project_chapter_title_retrieval_expression(query: str) -> str:
    """为明确的章节标题查询追加与索引一致的结构别名。"""

    original = str(query)
    shadow = unicodedata.normalize("NFKC", original)
    aliases: list[str] = []
    for path in extract_chapter_title_references(original):
        alias = f"标题路径：{path}"
        if unicodedata.normalize("NFKC", alias) not in shadow:
            aliases.append(alias)

    if not aliases:
        return original
    projected = f"{original} {' '.join(aliases)}"
    return projected if len(projected) <= MAX_QUERY_LENGTH else original


def _single_facet_retrieval_query(query: str) -> str:
    """单 facet 检索只保留语义问句，回答组织/信源尾句留给契约层。"""

    has_response_instruction = bool(_response_instruction_clauses(query))
    information_parts = _decomposition_parts(query)
    if has_response_instruction and len(information_parts) == 1:
        return information_parts[0]
    return query


def _decompose(
    understanding: QueryUnderstanding,
    *,
    max_subqueries: int,
) -> tuple[tuple[QuerySubQuery, ...], tuple[str, ...]]:
    query = understanding.standalone_query
    if not understanding.needs_multihop or max_subqueries < 2:
        primary_query = _single_facet_retrieval_query(query)
        limitations = (
            ("multihop_decomposition_limited_by_budget",)
            if understanding.needs_multihop and max_subqueries < 2
            else ()
        )
        return (
            (
                QuerySubQuery(
                    sub_query_id="sq_primary",
                    query=project_chapter_title_retrieval_expression(primary_query),
                    purpose=SubQueryPurpose.RETRIEVAL,
                ),
            ),
            limitations,
        )

    parts = list(_decomposition_parts(query))
    # 比较问题通常无法仅靠连接词安全拆解；保留完整问题比臆造两个目标子问题更安全。
    if len(parts) < 2:
        primary_query = _single_facet_retrieval_query(query)
        return (
            (
                QuerySubQuery(
                    sub_query_id="sq_primary",
                    query=project_chapter_title_retrieval_expression(primary_query),
                    purpose=SubQueryPurpose.RETRIEVAL,
                ),
            ),
            ("multihop_detected_but_not_safely_decomposable",),
        )

    reserve_synthesis = max_subqueries >= 3
    leaf_budget = max_subqueries - 1 if reserve_synthesis else max_subqueries
    shared_entity = _safe_shared_entity(understanding.entities)
    first_part = parts[0]
    leaves: list[QuerySubQuery] = []
    for index, part in enumerate(parts[:leaf_budget], start=1):
        part = _contextualize_decomposition_part(
            part,
            first_part=first_part,
            index=index,
        )
        if shared_entity and shared_entity.lower() not in part.lower():
            part = f"{shared_entity}：{part}"
        leaves.append(
            QuerySubQuery(
                sub_query_id=f"sq_part_{index}",
                query=project_chapter_title_retrieval_expression(part),
                purpose=SubQueryPurpose.RETRIEVAL,
            )
        )
    limitations: list[str] = []
    if len(parts) > leaf_budget:
        limitations.append("subqueries_truncated_by_budget")
    if reserve_synthesis and len(leaves) >= 2:
        leaves.append(
            QuerySubQuery(
                sub_query_id="sq_synthesis",
                query=query,
                purpose=SubQueryPurpose.SYNTHESIS,
                depends_on=tuple(item.sub_query_id for item in leaves),
            )
        )
    return tuple(leaves), tuple(limitations)


def _canonical_facet_text(value: str) -> str:
    """生成稳定键，同时保留技术标识符中具有语义的符号。"""

    normalized = unicodedata.normalize("NFKC", " ".join(value.split())).casefold()
    normalized = re.sub(
        r"^(?:并(?=在|通过|从|用|如何|怎么|怎样)|并且|同时|另外|以及)[,，:：\s]*",
        "",
        normalized,
    )
    result: list[str] = []
    for index, character in enumerate(normalized):
        if character.isalnum() or character in "+#_":
            result.append(character)
            continue
        if (
            character == "."
            and index > 0
            and index + 1 < len(normalized)
            and normalized[index - 1].isdigit()
            and normalized[index + 1].isdigit()
        ):
            result.append(character)
    return "".join(result)


def _facet_semantic_signature(value: str) -> str:
    """保留有意义符号的无损签名，用于阻止 lossy canonical key 碰撞。"""

    normalized = unicodedata.normalize("NFKC", " ".join(value.split())).casefold()
    normalized = re.sub(
        r"^(?:并(?=在|通过|从|用|如何|怎么|怎样)|并且|同时|另外|以及)[,，:：\s]*",
        "",
        normalized,
    )
    return normalized.strip(_TRIM_PUNCTUATION)


def _facet_information_parts(understanding: QueryUnderstanding) -> tuple[str, ...]:
    """抽取独立信息需求，同时保留普通联合宾语为单一 facet。"""

    if not understanding.need_retrieval:
        return ()
    query = understanding.standalone_query.strip(_TRIM_PUNCTUATION)
    decomposed = _decomposition_parts(query)
    has_projected_instruction = any(
        _is_response_instruction_clause(part)
        for part in _HARD_CLAUSE_SPLIT.split(query)
        if part.strip(_TRIM_PUNCTUATION)
    )
    if understanding.needs_multihop or has_projected_instruction:
        raw_parts = decomposed
    else:
        raw_parts = (query,)
    if not raw_parts:
        # 纯指令或异常空投影不应构造空检索契约；保留原问句让既有歧义与拒答
        # 逻辑继续 fail-closed。
        raw_parts = (query,)

    parts: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        part = re.sub(
            r"^(?:并(?=在|通过|从|用|如何|怎么|怎样)|并且|同时|另外|以及)[,，:：\s]*",
            "",
            raw.strip(_TRIM_PUNCTUATION),
        )
        signature = _facet_semantic_signature(part)
        canonical = _canonical_facet_text(part)
        if not canonical or signature in seen:
            continue
        parts.append(part)
        seen.add(signature)
    if len(parts) > MAX_QUERY_FACETS:
        overflow = "；".join(parts[MAX_QUERY_FACETS - 1 :])
        parts = [*parts[: MAX_QUERY_FACETS - 1], overflow]
    return tuple(parts) or (query,)


def _conditional_premise(value: str) -> str | None:
    """提取句首条件前提；没有显式条件标记时返回 ``None``。"""

    normalized = value.strip(_TRIM_PUNCTUATION)
    if re.match(r"^(?:如果|若|如未|当|if\b|when\b)", normalized, flags=re.IGNORECASE) is None:
        return None
    return re.split(r"[,，]", normalized, maxsplit=1)[0].strip()


def _normalized_answer_state_claim(premise: str) -> str | None:
    """规范化可能复述前序二元答案的状态谓词。"""

    claim = re.sub(r"^(?:如果|若)\s*", "", premise)
    claim = re.sub(
        r"^(?:(?:该|此)?(?:产品|设备|打印机|扫描仪|机器|型号))\s*",
        "",
        claim,
    )
    claim = re.sub(r"时$", "", claim).strip()
    if (
        re.match(
            r"^(?:支持|不支持|具备|不具备|允许|不允许|可以|不可以|能够|不能|"
            r"存在|不存在|满足|不满足|是|不是)",
            claim,
        )
        is None
    ):
        return None
    normalized = _normalize_for_match(claim)
    return normalized if len(normalized) >= 4 else None


def _is_answer_dependent_condition(
    value: str,
    *,
    preceding_parts: tuple[str, ...],
) -> bool:
    """只把依赖前序答案或执行结果的条件标为 conditional。"""

    premise = _conditional_premise(value)
    if premise is None:
        return False
    if _CONDITIONAL_DEPENDENCY_REFERENCE.search(premise):
        return True
    if _ELLIPTICAL_ANSWER_CONDITION.fullmatch(premise):
        return True
    if _FALLBACK_RESULT_CONDITION.search(premise):
        return True
    if _ENGLISH_DEPENDENT_CONDITION.search(premise):
        return True

    # 显式复述前序二元谓词（如“是否支持扩展纸盒”）也属于答案依赖；只比较
    # 状态谓词，避免把“当设备发生卡纸”一类独立场景误判为条件义务。
    answer_state_claim = _normalized_answer_state_claim(premise)
    return bool(
        answer_state_claim
        and any(
            answer_state_claim in _normalize_for_match(previous) for previous in preceding_parts
        )
    )


def _facet_necessity(
    value: str,
    *,
    preceding_parts: tuple[str, ...] = (),
) -> FacetNecessity:
    if preceding_parts and _is_answer_dependent_condition(
        value,
        preceding_parts=preceding_parts,
    ):
        return FacetNecessity.CONDITIONAL
    return FacetNecessity.REQUIRED


def _facet_necessities(parts: tuple[str, ...]) -> tuple[FacetNecessity, ...]:
    """按前提依赖关系判定 necessity，而不是按句首条件词或位置判定。"""

    return tuple(
        _facet_necessity(part, preceding_parts=parts[:index]) for index, part in enumerate(parts)
    )


_CHINESE_COUNT_VALUES = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _count_from_match(matched: re.Match[str]) -> int | None:
    raw_count = matched.group("count")
    count = int(raw_count) if raw_count.isascii() and raw_count.isdigit() else None
    if count is None:
        count = _CHINESE_COUNT_VALUES.get(raw_count)
    if count is None or not 1 <= count <= MAX_FACET_EVIDENCE_IDS:
        return None
    return count


def _imperative_answer_count(value: str) -> int | None:
    """只接受直接受答案请求动词支配的数量短语。"""

    for request in _ANSWER_REQUEST_VERB.finditer(value):
        remainder = value[request.end() :].strip(_TRIM_PUNCTUATION)
        for matched in _COUNTED_NOUN.finditer(remainder):
            prefix = remainder[: matched.start()].strip()
            suffix = remainder[matched.end() :].strip(_TRIM_PUNCTUATION)
            # ``在 N 种模式下``、``支持 N 种协议的设备`` 中的数字约束谓词，
            # 并不声明答案应包含 N 项。闭合契约宁可保守降级，也不能错误加码。
            if _ANSWER_COUNT_SCOPE_PREFIX.search(prefix):
                continue
            if _ANSWER_COUNT_SCOPED_SUFFIX.search(suffix):
                continue
            if suffix and not _ANSWER_COUNT_NOUN_SUFFIX.search(suffix):
                continue
            if count := _count_from_match(matched):
                return count
    return None


def _facet_closure(value: str) -> tuple[FacetClosure, int | None]:
    """仅从明确绑定答案槽的数量表达构造闭合约束。"""

    matched = _ANSWER_BOUND_WH_COUNT.search(value)
    if matched is not None and (count := _count_from_match(matched)) is not None:
        return FacetClosure.CLOSED_COUNT, count
    # 复数 wh-head 已经把答案定义为未知大小的集合；其后的数量通常修饰筛选
    # 谓词或属性维度，不能反向附着成答案 cardinality。
    if _PLURAL_WH_ANSWER.search(value):
        return FacetClosure.OPEN_ENUMERATION, None
    matched = _ANSWER_BOUND_POSTPOSED_COUNT.search(value)
    if matched is not None and (count := _count_from_match(matched)) is not None:
        return FacetClosure.CLOSED_COUNT, count
    if count := _imperative_answer_count(value):
        return FacetClosure.CLOSED_COUNT, count
    if _OPEN_ENUMERATION.search(value):
        return FacetClosure.OPEN_ENUMERATION, None
    # 带明确操作载体与目标动作的“如何”问题需要保留多个有序步骤。普通的
    # “设备如何启动”仍维持 SINGLE，避免把所有过程意图都无依据地扩成集合。
    if is_multi_step_procedure_request(value):
        return FacetClosure.OPEN_ENUMERATION, None
    return FacetClosure.SINGLE, None


def is_multi_step_procedure_request(value: str) -> bool:
    """仅识别由问题文本明确要求的多步骤输出，不从 Evidence 反推基数。"""

    response_requests_procedure = bool(
        _ORDERED_RESPONSE_STRUCTURE.search(value)
        or _COMPLETE_PROCEDURE_RESPONSE_STRUCTURE.search(value)
    )
    return bool(
        _SCOPED_MULTI_STEP_PROCEDURE.search(value)
        or _EXPLICIT_MULTI_STEP_STRUCTURE.search(value)
        or _ORDERED_ACTION_COLLECTION.search(value)
        or _CONTEXTUAL_MULTI_STEP_PROCEDURE.search(value)
        or (response_requests_procedure and _ACTION_INFORMATION_NEED.search(value))
    )


def _response_instruction_clauses(query: str) -> tuple[str, ...]:
    """返回已确认只约束回答组织方式的尾部子句。"""

    return tuple(
        normalized
        for raw in _HARD_CLAUSE_SPLIT.split(query)
        if (normalized := raw.strip(_TRIM_PUNCTUATION))
        and _is_response_instruction_clause(normalized)
    )


def _is_action_information_need(value: str) -> bool:
    """保守识别可承接步骤型输出约束的动作信息需求。"""

    return bool(
        _ACTION_INFORMATION_NEED.search(value)
        or _SCOPED_MULTI_STEP_PROCEDURE.search(value)
        or _CONTEXTUAL_MULTI_STEP_PROCEDURE.search(value)
        or _EXPLICIT_MULTI_STEP_STRUCTURE.search(value)
        or _ORDERED_ACTION_COLLECTION.search(value)
    )


def _project_response_instructions(
    query: str,
    parts: tuple[str, ...],
) -> tuple[tuple[str, ...], frozenset[int]]:
    """把输出约束投影到既有 facet，不把指令本身制造成新事实槽位。"""

    instructions = _response_instruction_clauses(query)
    if not instructions:
        return parts, frozenset()

    procedure_instruction = any(
        _ORDERED_RESPONSE_STRUCTURE.search(item)
        or _COMPLETE_PROCEDURE_RESPONSE_STRUCTURE.search(item)
        or (_COMPLETE_RESPONSE_STRUCTURE.search(item) and "步骤" in item)
        for item in instructions
    )
    completeness_instruction = any(
        _COMPLETE_RESPONSE_STRUCTURE.search(item) for item in instructions
    )
    procedure_targets = {
        index
        for index, part in enumerate(parts)
        if procedure_instruction and _is_action_information_need(part)
    }

    completeness_targets: set[int] = set()
    if completeness_instruction and parts:
        distributed = any(
            re.match(r"^(?:请)?(?:分别|逐项)", item) is not None for item in instructions
        )
        if len(parts) == 1:
            completeness_targets.add(0)
        elif distributed:
            completeness_targets.update(range(len(parts)))
        else:
            # 未声明“分别/逐项”时，尾部完整性指令只约束最近的信息需求。
            completeness_targets.add(len(parts) - 1)

    projected = list(parts)
    if len(parts) > 1:
        for index in procedure_targets:
            candidate = f"{parts[index]}；请按顺序说明操作步骤"
            if len(candidate) <= MAX_QUERY_LENGTH:
                projected[index] = candidate
    return tuple(projected), frozenset(procedure_targets | completeness_targets)


def _facet_identity_keys(parts: tuple[str, ...]) -> tuple[str, ...]:
    """在保持扰动稳定性的同时，为同一问题内的 canonical 碰撞域分。"""

    canonical = [_canonical_facet_text(item) for item in parts]
    counts = {value: canonical.count(value) for value in set(canonical)}
    return tuple(
        value
        if counts[value] == 1
        else (
            value
            + "\x1e"
            + hashlib.sha256(_facet_semantic_signature(part).encode("utf-8")).hexdigest()[:16]
        )
        for part, value in zip(parts, canonical, strict=True)
    )


def _facet_retrieval_expression(
    part: str,
    *,
    understanding: QueryUnderstanding,
    first_part: str,
    index: int,
) -> str:
    expression = _contextualize_decomposition_part(
        part,
        first_part=first_part,
        index=index,
    )
    shared_entity = _safe_shared_entity(understanding.entities)
    if shared_entity and shared_entity.lower() not in expression.lower():
        expression = f"{shared_entity}：{expression}"
    return expression


def build_query_facet_ledger(
    understanding: QueryUnderstanding,
    sub_queries: tuple[QuerySubQuery, ...],
) -> FacetLedger:
    """从用户独立问题和已批准 sub-query 构造确定性 Facet Ledger。

    该函数只读取 QueryUnderstanding，不接收评测 case、参考答案或 concept groups。
    Facet ID 由无序信息需求集合与单个需求的规范文本生成，因此分句换序、标点和
    ``同时/另外/以及/并且`` 扰动不会改变同一信息需求的 ID。
    """

    query_hash = (
        "sha256:" + hashlib.sha256(understanding.standalone_query.encode("utf-8")).hexdigest()
    )
    base_parts = _facet_information_parts(understanding)
    if not base_parts:
        return FacetLedger(source_query_hash=query_hash)

    information_parts, output_open_targets = _project_response_instructions(
        understanding.standalone_query,
        base_parts,
    )
    facet_identity_keys = _facet_identity_keys(information_parts)
    facet_necessities = _facet_necessities(base_parts)
    context_signature = "|".join(sorted(facet_identity_keys))
    retrieval_by_id = {
        item.sub_query_id: item for item in sub_queries if item.purpose == SubQueryPurpose.RETRIEVAL
    }
    facets: list[QueryFacet] = []
    first_part = base_parts[0]
    relational_projection = bool(
        len(base_parts) == 1
        and _RESULT_SET_DEPENDENCY_BOUNDARY.search(understanding.standalone_query)
    )
    single_facet_procedure_requested = len(base_parts) == 1 and is_multi_step_procedure_request(
        understanding.standalone_query
    )
    for index, (part, information_need, identity_key, necessity) in enumerate(
        zip(
            base_parts,
            information_parts,
            facet_identity_keys,
            facet_necessities,
            strict=True,
        ),
        start=1,
    ):
        polarity = FacetPolarity.NEGATIVE if _contains_negation(part) else FacetPolarity.AFFIRMATIVE
        closure, expected_count = _facet_closure(part)
        if relational_projection and closure != FacetClosure.CLOSED_COUNT:
            closure = FacetClosure.OPEN_ENUMERATION
            expected_count = None
        elif (
            single_facet_procedure_requested or index - 1 in output_open_targets
        ) and closure == FacetClosure.SINGLE:
            closure = FacetClosure.OPEN_ENUMERATION
        dedicated = retrieval_by_id.get(f"sq_part_{index}")
        if dedicated is None:
            dedicated = retrieval_by_id.get("sq_primary")
        retrieval_expression = (
            dedicated.query
            if dedicated is not None
            else _facet_retrieval_expression(
                part,
                understanding=understanding,
                first_part=first_part,
                index=index,
            )
        )
        facet_seed = "\x1f".join(
            (
                FACET_LEDGER_SCHEMA_VERSION,
                context_signature,
                identity_key,
                necessity.value,
                polarity.value,
                closure.value,
                str(expected_count or ""),
            )
        )
        facet_id = f"qf_{hashlib.sha256(facet_seed.encode('utf-8')).hexdigest()[:32]}"
        facets.append(
            QueryFacet(
                facet_id=facet_id,
                information_need=information_need,
                retrieval_expression=retrieval_expression,
                necessity=necessity,
                polarity=polarity,
                closure=closure,
                expected_count=expected_count,
                sub_query_ids=((dedicated.sub_query_id,) if dedicated is not None else ()),
            )
        )
    return FacetLedger(source_query_hash=query_hash, facets=tuple(facets))


class DeterministicQueryPlanner:
    """无外部副作用的规则规划器，可直接用于离线评测和请求重放。"""

    def understand(self, request: QueryPlanningInput) -> QueryUnderstanding:
        standalone, rewrite = _select_standalone_query(request)
        entities = extract_entities(standalone)
        temporal = _detect_temporal(standalone)
        single_kb_scope = len(request.authorized_scope_ref.kb_ids) == 1
        freshness_scope = _classify_freshness_scope(
            standalone,
            temporal,
            single_kb_scope=single_kb_scope,
        )
        intent = _classify_intent(standalone)
        ambiguity = _detect_ambiguity(
            standalone,
            intent=intent,
            entities=entities,
            single_kb_scope=single_kb_scope,
        )
        need_retrieval = not ambiguity.is_ambiguous and intent not in {
            QueryIntent.GREETING,
            QueryIntent.UI_HELP,
        }
        return QueryUnderstanding(
            original_query=request.original_query,
            display_query=request.original_query,
            standalone_query=standalone,
            language=_detect_language(standalone),
            intent=intent,
            temporal_requirement=temporal,
            entities=entities,
            constraints=_constraints(temporal, freshness_scope=freshness_scope),
            ambiguity=ambiguity,
            need_retrieval=need_retrieval,
            need_freshness=freshness_scope == _EXTERNAL_FRESHNESS,
            needs_multihop=_needs_multihop(standalone, intent),
            rewrite=rewrite,
        )

    def plan(
        self,
        request: QueryPlanningInput,
        *,
        additional_reasons: tuple[str, ...] = (),
    ) -> QueryPlan:
        understanding = self.understand(request)
        reasons: list[str] = list(additional_reasons)
        limitations: list[str] = []

        if understanding.ambiguity.is_ambiguous:
            disposition = PlanDisposition.NEED_CLARIFICATION
            reasons.extend(("critical_slot_missing", "retrieval_blocked_until_clarified"))
            return self._assemble(
                request=request,
                understanding=understanding,
                disposition=disposition,
                sub_queries=(),
                routes=(),
                reasons=tuple(reasons),
                confidence=0.96,
                response_key=None,
                limitations=(),
            )

        if not understanding.need_retrieval:
            disposition = PlanDisposition.DIRECT_ANSWER
            response_key = (
                "query.greeting"
                if understanding.intent == QueryIntent.GREETING
                else "query.ui_help"
            )
            reasons.extend(("deterministic_direct_intent", "retrieval_not_required"))
            return self._assemble(
                request=request,
                understanding=understanding,
                disposition=disposition,
                sub_queries=(),
                routes=(),
                reasons=tuple(reasons),
                confidence=0.99,
                response_key=response_key,
                limitations=(),
            )

        sub_queries, decomposition_limitations = _decompose(
            understanding,
            max_subqueries=request.policy.max_subqueries,
        )
        limitations.extend(decomposition_limitations)
        retrieval_ids = tuple(
            item.sub_query_id for item in sub_queries if item.purpose == SubQueryPurpose.RETRIEVAL
        )
        routes: list[QueryRoute] = [
            QueryRoute(
                route_id="route_local_hybrid",
                route_type=QueryRouteType.LOCAL_HYBRID,
                required=True,
                sub_query_ids=retrieval_ids,
                # 本地混合检索是计划成功所必需的路线，必须获得发布检索策略
                # 明确授予的完整超时预算。PlanningPolicy / QueryPlan 已将该值
                # 限制在可信上界内，执行器还会与 QueryRun 剩余 deadline 取交集。
                timeout_ms=request.policy.total_timeout_ms,
                top_k=request.policy.default_top_k,
                filters=request.route_filters,
                fallback=RouteFallback.FAIL_PLAN,
                max_cost=0.05,
                evidence_eligible=True,
            )
        ]
        spent = 0.05
        reasons.append("local_hybrid_required")

        hyde_warranted = understanding.needs_multihop or len(understanding.standalone_query) >= 100
        if hyde_warranted and request.policy.allow_hyde:
            if (
                len(routes) < request.policy.max_routes
                and spent + 0.20 <= request.policy.cost_budget + 1e-9
            ):
                routes.append(
                    QueryRoute(
                        route_id="route_hyde",
                        route_type=QueryRouteType.HYDE,
                        required=False,
                        sub_query_ids=retrieval_ids,
                        timeout_ms=min(2_500, request.policy.total_timeout_ms),
                        top_k=request.policy.default_top_k,
                        filters=request.route_filters,
                        fallback=RouteFallback.SKIP_ROUTE,
                        max_cost=0.20,
                        evidence_eligible=False,
                    )
                )
                spent += 0.20
                reasons.append("hyde_enabled_for_complex_query")
            else:
                limitations.append("hyde_omitted_by_budget")
        elif hyde_warranted:
            limitations.append("hyde_forbidden_by_policy")
        else:
            reasons.append("simple_query_avoids_hyde")

        if understanding.need_freshness:
            if request.policy.allow_web:
                if (
                    len(routes) < request.policy.max_routes
                    and spent + 0.40 <= request.policy.cost_budget + 1e-9
                ):
                    routes.append(
                        QueryRoute(
                            route_id="route_web",
                            route_type=QueryRouteType.WEB,
                            required=False,
                            sub_query_ids=retrieval_ids,
                            timeout_ms=min(4_000, request.policy.total_timeout_ms),
                            top_k=request.policy.default_top_k,
                            fallback=RouteFallback.LOCAL_ONLY,
                            max_cost=0.40,
                            evidence_eligible=True,
                        )
                    )
                    reasons.append("freshness_requires_policy_approved_web")
                else:
                    limitations.append("web_omitted_by_budget")
            else:
                limitations.append("freshness_requested_but_web_forbidden")
                reasons.append("web_not_enabled_without_policy")
        else:
            reasons.append("web_not_needed_without_freshness")

        if understanding.needs_multihop:
            reasons.append("controlled_multihop_plan")
        else:
            reasons.append("single_question_low_cost_plan")
        return self._assemble(
            request=request,
            understanding=understanding,
            disposition=PlanDisposition.RETRIEVE,
            sub_queries=sub_queries,
            routes=tuple(routes),
            reasons=tuple(reasons),
            confidence=0.82 if understanding.needs_multihop else 0.94,
            response_key=None,
            limitations=tuple(limitations),
        )

    def approve_or_fallback(
        self,
        request: QueryPlanningInput,
        raw_proposal: QueryPlan | dict[str, Any] | None,
        *,
        proposal_error: str | None = None,
    ) -> QueryPlan:
        """批准结构化 proposal；超时、非法或越权时回退确定性安全计划。"""

        if raw_proposal is None:
            reason = "planner_proposal_unavailable"
            if proposal_error:
                reason = f"{reason}:{proposal_error[:64]}"
            return self.plan(request, additional_reasons=(reason, "safe_rule_fallback"))
        try:
            proposal = validate_plan_for_execution(
                raw_proposal,
                trusted_scope=request.authorized_scope_ref,
                trusted_policy=request.policy,
            )
            baseline = self.plan(request)
            if proposal.profile_version != request.profile_version:
                raise ValueError("proposal changed the retrieval profile version")
            if proposal.understanding.original_query != request.original_query:
                raise ValueError("proposal changed the original query")
            if proposal.understanding != baseline.understanding:
                raise ValueError("proposal bypassed deterministic query understanding")
            if proposal.disposition != baseline.disposition:
                raise ValueError("proposal bypassed a deterministic hard decision")
            if proposal.facet_ledger != baseline.facet_ledger:
                raise ValueError("proposal changed the deterministic facet ledger")
            if proposal.sub_queries != baseline.sub_queries:
                raise ValueError("proposal changed the approved intent-preserving sub-queries")
            allowed_routes = {route.route_type for route in baseline.routes}
            if any(route.route_type not in allowed_routes for route in proposal.routes):
                raise ValueError("proposal enabled a route not allowed by hard rules")
            return proposal
        except (TypeError, ValueError) as exc:
            return self.plan(
                request,
                additional_reasons=(
                    "planner_proposal_rejected",
                    f"safe_rule_fallback:{type(exc).__name__}",
                ),
            )

    @staticmethod
    def _assemble(
        *,
        request: QueryPlanningInput,
        understanding: QueryUnderstanding,
        disposition: PlanDisposition,
        sub_queries: tuple[QuerySubQuery, ...],
        routes: tuple[QueryRoute, ...],
        reasons: tuple[str, ...],
        confidence: float,
        response_key: str | None,
        limitations: tuple[str, ...],
    ) -> QueryPlan:
        facet_ledger = build_query_facet_ledger(understanding, sub_queries)
        seed = {
            "planner_version": PLANNER_VERSION,
            "profile_version": request.profile_version,
            "disposition": disposition.value,
            "understanding": understanding,
            "facet_ledger": facet_ledger,
            "sub_queries": sub_queries,
            "routes": routes,
            "scope": request.authorized_scope_ref,
            "policy": request.policy,
            "deadline_at": request.deadline_at,
            "reasons": reasons,
            "limitations": limitations,
        }
        return QueryPlan(
            plan_id=make_plan_id(seed),
            planner_version=PLANNER_VERSION,
            profile_version=request.profile_version,
            disposition=disposition,
            understanding=understanding,
            facet_ledger=facet_ledger,
            sub_queries=sub_queries,
            routes=routes,
            authorized_scope_ref=request.authorized_scope_ref,
            policy=request.policy,
            deadline_at=request.deadline_at,
            token_budget=request.policy.token_budget,
            cost_budget=request.policy.cost_budget,
            decision_reasons=reasons,
            confidence=confidence,
            response_key=response_key,
            limitations=limitations,
        )


def build_query_plan(request: QueryPlanningInput) -> QueryPlan:
    return DeterministicQueryPlanner().plan(request)


__all__ = [
    "DeterministicQueryPlanner",
    "QueryPlanningInput",
    "build_query_facet_ledger",
    "build_query_plan",
    "extract_chapter_title_references",
    "extract_entities",
    "project_chapter_title_retrieval_expression",
    "validate_intent_preserving_rewrite",
]
