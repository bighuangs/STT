"""纯确定性的局部角色—值与约束极性绑定校验。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_CLAUSE_RE = re.compile(r"[^。！？!?；;，,\n]+")
_BINARY_SENTENCE_RE = re.compile(
    r"(?P<sentence>[^。！？!?；;\n]+?)(?:[。！？!?；;]|\.(?=\s|$)|\n|$)"
)
_ASCII_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*(?![A-Za-z0-9])")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_NEGATIVE_RE = re.compile(
    r"(?:请勿|不得|禁止|严禁|切勿|不要|不能|不可以|恕不|无需|不需要|不会|不再|不另行|"
    r"未(?:经|使用|打印|完成|去除)?|没有|无法)"
    r"|\b(?:not|never|without|no)\b",
    re.IGNORECASE,
)
_REQUIREMENT_RE = re.compile(
    r"(?:必须|务必|应当|应先|需要|需先|确保|确认|"
    r"请(?:将|先|使用|确保|按|选择|打开|关闭|插|拔|连接|放|装|取|检查)|"
    r"require(?:d)?|must|shall)",
    re.IGNORECASE,
)
_ANCHOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("side:left", re.compile(r"(?:左侧|左边|左方|left)", re.IGNORECASE)),
    ("side:right", re.compile(r"(?:右侧|右边|右方|right)", re.IGNORECASE)),
    ("vertical:top", re.compile(r"(?:上侧|上方|上面|顶部|top)", re.IGNORECASE)),
    (
        "vertical:bottom",
        re.compile(r"(?:下侧|下方|下面|底部|bottom)", re.IGNORECASE),
    ),
    ("color:black", re.compile(r"(?:黑色|黑墨|black)", re.IGNORECASE)),
    ("color:color", re.compile(r"(?:彩色|彩墨|colour|color)", re.IGNORECASE)),
)
_RELATION_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "﹣": "-",
        "≦": "≤",
        "≧": "≥",
        "﹤": "<",
        "﹥": ">",
    }
)
_CJK_UNIT_PATTERN = "|".join(
    sorted(
        (
            "摄氏度",
            "华氏度",
            "百分比",
            "千瓦时",
            "千帕",
            "兆帕",
            "毫巴",
            "毫米",
            "厘米",
            "千米",
            "毫克",
            "千克",
            "毫升",
            "毫秒",
            "分钟",
            "小时",
            "赫兹",
            "字节",
            "转每分",
            "伏",
            "安",
            "瓦",
            "帕",
            "巴",
            "米",
            "克",
            "升",
            "秒",
            "分",
            "时",
            "天",
            "周",
            "月",
            "年",
            "度",
            "转",
            "次",
            "个",
            "页",
            "张",
        ),
        key=len,
        reverse=True,
    )
)
_RELATION_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"(?P<number>[+-]?(?:\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|"
    rf"\d+(?:\.\d+)?|\.\d+))"
    rf"\s*(?P<unit>%|‰|°\s*[cf]|"
    rf"[a-zµμω]+(?:[0-9²³]*)(?:[/·][a-zµμω]+(?:[0-9²³]*))*|"
    rf"{_CJK_UNIT_PATTERN})?"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_RANGE_CONNECTOR_RE = re.compile(r"\s*(?:至|到|~|～|to)\s*", re.IGNORECASE)
_RELATION_OPERATOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "le",
        re.compile(
            r"(?:≤|<=|小于(?:或)?等于|不(?:得|能|可|应)?(?:超过|高于|大于)|"
            r"至多|最多|less\s+than\s+or\s+equal\s+to|"
            r"no\s+more\s+than|at\s+most)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ge",
        re.compile(
            r"(?:≥|>=|大于(?:或)?等于|不(?:得|能|可|应)?(?:低于|小于|少于)|"
            r"至少|最少|greater\s+than\s+or\s+equal\s+to|"
            r"no\s+less\s+than|at\s+least)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "lt",
        re.compile(r"(?:<|小于|低于|less\s+than)\s*$", re.IGNORECASE),
    ),
    (
        "gt",
        re.compile(r"(?:>|大于|高于|greater\s+than)\s*$", re.IGNORECASE),
    ),
    (
        "eq",
        re.compile(r"(?:=|等于|为|是|\bis|equal\s+to)\s*$", re.IGNORECASE),
    ),
)
_UNIT_EQUIVALENTS = {
    "摄氏度": "°c",
    "华氏度": "°f",
    "百分比": "%",
}
_FIELD_LEADING_CONNECTOR_RE = re.compile(
    r"^\s*(?:(?:且|和|及|以及|并|并且|同时|而|但)|\b(?:and|or|while)\b)\s*",
    re.IGNORECASE,
)
_FIELD_TRAILING_MODAL_RE = re.compile(
    r"(?:必须|务必|应当|应该|应|需要|需|须|"
    r"\bmust(?:\s+be)?|\bshall(?:\s+be)?|\bshould(?:\s+be)?|"
    r"\brequired(?:\s+to|\s+be)?|\bto\s+be)\s*$",
    re.IGNORECASE,
)
_FIELD_ENGLISH_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)
_TEMPORAL_NEGATIVE_VERDICT_RE = re.compile(
    r"(?:不能|不可以|不可|不得|请勿|切勿|禁止|严禁|不应|"
    r"\b(?:cannot|can't|must\s+not|may\s+not|not\s+allowed)\b)",
    re.IGNORECASE,
)
_TEMPORAL_IMMEDIATE_RE = re.compile(
    r"(?:马上|立即|立刻|现在|当前|此时|当下|"
    r"\b(?:immediately|right\s+now|currently|at\s+once|now)\b)",
    re.IGNORECASE,
)
_TEMPORAL_WEAK_ADVICE_RE = re.compile(
    r"(?:建议|最好|可以考虑|可考虑|推荐|尽量|酌情|"
    r"\b(?:recommend(?:ed)?|suggest(?:ed)?|may\s+want|might\s+want|consider)\b)",
    re.IGNORECASE,
)
_TEMPORAL_POSITIVE_PERMISSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:可以|允许|获准|准许|能够|可)\s*(?:马上|立即|立刻|现在|当前|此时|当下)"
        r"\s*(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:马上|立即|立刻|现在|当前|此时|当下|现已)"
        r"[^。！？!?；;\n]{0,32}?(?:可以|允许|获准|准许|能够|可)\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"无需[^。！？!?；;\n]{0,30}(?:即可|便可)\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:may|can|(?:is\s+)?allowed\s+to|(?:is\s+)?permitted\s+to)\s+"
        r"(?:immediately|right\s+now|currently|at\s+once|now)\s+"
        r"(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:may|can|(?:is\s+)?allowed\s+to|(?:is\s+)?permitted\s+to)\s+"
        r"(?P<action>[^.!?;\n]{2,80}?)\s+"
        r"(?:immediately|right\s+now|currently|at\s+once|now)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:may|can)\s+(?:right\s+now|currently|now)\s+"
        r"(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are)\s+(?:right\s+now|currently|now)\s+"
        r"(?:allowed|permitted)\s+to\s+(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.!?;\n])\s*(?:right\s+now|currently|now)\s+"
        r"[^.!?;\n]{0,48}?\b(?:may|can|(?:is\s+)?allowed\s+to|"
        r"(?:is\s+)?permitted\s+to)\s+(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
)
_TEMPORAL_GATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:(?:必须|务必|应当|应该|需要|需|请)\s*先|"
        r"(?:^|[。！？!?；;\n：:])\s*先)\s*"
        r"(?P<gate>[^。！？!?；;\n]{2,48}?)\s*[,，]?\s*再\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:必须|务必|应当|应该|需要|需)\s*"
        r"(?P<gate>[^。！？!?；;\n]{2,48}?)\s*(?:后|之后|以后|完成后)\s*"
        r"(?:再|才(?:能|可以|可)?|方可)\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmust\s+(?P<gate>[^.!?;\n]{2,80}?)\s+before\s+"
        r"(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bonly\s+after\s+(?P<gate>[^.!?;\n]{2,80}?)\s+"
        r"(?:may|can|should)\s+(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
)
_BINARY_QUERY_MARKER_RE = re.compile(
    r"(?:是否|能否|可否|应否|有没有|是不是|能不能|可不可以|该不该)\s*"
    r"(?P<predicate>[^。！？!?；;\n]{1,96})",
    re.IGNORECASE,
)
_BINARY_QUERY_MODAL_RE = re.compile(
    r"(?:不可以|不能|不可|可以|能够|(?<!功)(?<!性)能|允许|应该|应当|需要|必须)"
    r"\s*(?P<predicate>[^。！？!?；;\n]{1,96}?)\s*吗\s*[？?]?\s*$",
    re.IGNORECASE,
)
_BINARY_QUERY_ENGLISH_RE = re.compile(
    r"^\s*(?:is|are|am|was|were|can|could|may|might|should|must|do|does|did|"
    r"has|have|had|will|would)\s+(?P<predicate>[^.!?;\n]{1,120})",
    re.IGNORECASE,
)
_BINARY_LEADING_ENGLISH_CONDITION_RE = re.compile(
    r"^\s*(?P<condition>(?:if|when|whenever|unless|once|until|before|after|"
    r"only\s+after|under|during|in\s+case\s+of)\s+[^.!?;,\n]{1,80})"
    r"\s*[,;]\s*(?P<core>[^.!?;\n]{1,160})$",
    re.IGNORECASE,
)
_BINARY_TRAILING_ENGLISH_CONDITION_RE = re.compile(
    r"^\s*(?P<core>[^.!?;,\n]{1,160}?)\s+"
    r"(?P<condition>(?:if|when|whenever|unless|once|until|before|after|"
    r"only\s+after|under|during|in\s+case\s+of)\s+[^.!?;,\n]{1,80})\s*$",
    re.IGNORECASE,
)
_BINARY_NESTED_MODAL_RE = re.compile(
    r"^\s*(?:不可以|不能|不可|可以|能够|(?<!功)(?<!性)能|允许|应该|应当|需要|必须|"
    r"(?:is|are)\s+(?:not\s+)?(?:allowed|permitted|required)\s+to|"
    r"cannot|can't|can|may|must|should)\s+",
    re.IGNORECASE,
)
_BINARY_VERDICT_MODAL_RE = re.compile(
    r"(?:不可以|不能|不得|请勿|切勿|禁止|严禁|不可|不应|不需要|无需|无须|"
    r"可以|允许|应该|应当|需要|必须|能够|(?<!功)(?<!性)能|"
    r"不(?=(?:支持|具备|存在|适用|符合|满足|兼容|提供|配备))|"
    r"\b(?:is|are)\s+(?:not\s+)?(?:allowed|permitted|required)\s+to\b|"
    r"\b(?:do|does|did)\s+not(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\b(?:doesn't|don't|didn't)(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\bnever(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\b(?:cannot|can't|must\s+not|should\s+not|may\s+not|"
    r"can|may|must|should|yes|no)\b)",
    re.IGNORECASE,
)
_BINARY_NEGATIVE_MODAL_RE = re.compile(
    r"(?:不可以|不能|不得|请勿|切勿|禁止|严禁|不可|不应|不需要|无需|无须|"
    r"^不(?=(?:支持|具备|存在|适用|符合|满足|兼容|提供|配备))|"
    r"\b(?:cannot|can't|must\s+not|should\s+not|may\s+not|no)\b|"
    r"\b(?:do|does|did)\s+not(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\b(?:doesn't|don't|didn't)(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\bnever(?:\s+(?:currently|presently|now|yet|still))*\s+"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)|"
    r"\b(?:is|are)\s+not\s+(?:allowed|permitted|required)\s+to\b)",
    re.IGNORECASE,
)
_BINARY_EPISTEMIC_UNCERTAINTY_RE = re.compile(
    r"(?:可能|也许|或许|不一定|未必|尚不确定|取决于|视[^。！？!?；;\n]{0,32}而定)|"
    r"\b(?:might|could|possibly|perhaps|maybe|uncertain|unclear|"
    r"depends?\s+on|depending\s+on)\b|"
    r"\bnot\s+necessarily\b|"
    r"\b(?:do|does|did)\s+not\s+(?:appear|seem)\s+to\b|"
    r"\b(?:is|are|was|were)\s+not\s+"
    r"(?:known|expected|guaranteed|certain|proven)\s+to\b|"
    r"\bmay\s+or\s+may\s+not\b|\beither\b[^.!?;\n]{0,80}\bor\b",
    re.IGNORECASE,
)
_BINARY_POLARITY_ALTERNATIVE_RE = re.compile(
    r"(?:可以|能够|能|允许|支持|具备|存在|适用|符合|满足|兼容|提供|配备)"
    r"\s*(?:或|或者|也|和|与)\s*"
    r"(?:不可以|不能|不可|不允许|不支持|不具备|不存在|不适用|不符合|"
    r"不满足|不兼容|不提供|不配备)|"
    r"\b(?:can|may|must|should)\s+(?:and|or)\s+"
    r"(?:cannot|can't|may\s+not|must\s+not|should\s+not)\b|"
    r"\b(?:supports?|includes?|provides?|complies?|meets?|has|have)\b"
    r"[^.!?;\n]{0,24}\b(?:and|or)\s+(?:does?|do)\s+not\s+"
    r"(?:support|include|provide|comply|meet|have)\b",
    re.IGNORECASE,
)
_BINARY_PREREQUISITE_MODAL_RE = re.compile(
    r"^(?:应该|应当|需要|必须|"
    r"(?:is|are)\s+(?:required)\s+to|\b(?:must|should)\b)$",
    re.IGNORECASE,
)
_BINARY_PREREQUISITE_CONNECTOR_RE = re.compile(
    r"(?:然后|并且|且|再|\b(?:and|then)\b)\s*$",
    re.IGNORECASE,
)
_BINARY_INTRINSIC_ADVERSATIVE_RE = re.compile(
    r"(?:但是|然而|不过|但|却|而|\b(?:but|yet|although|however)\b)",
    re.IGNORECASE,
)
_BINARY_INTRINSIC_COORDINATOR_RE = re.compile(
    r"\b(?:as\s+well\s+as|along\s+with|together\s+with|and|or|plus|as\s+well|"
    r"while|whereas)\b|"
    r"(?:并且|同时|以及|还有|而且|并|及|且|又|和|与|或)",
    re.IGNORECASE,
)
_BINARY_INTRINSIC_NEGATION_PARENTHETICAL_RE = re.compile(
    r"\b(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never)\s*"
    r"(?P<left>,|—|–)\s*"
    r"(?:in\s+fact|actually|really|currently|presently|now|at\s+all|in\s+any\s+way)"
    r"\s*(?P<right>,|—|–)\s*"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)",
    re.IGNORECASE,
)
_BINARY_INTRINSIC_UNPARSED_NEGATION_PARENTHETICAL_RE = re.compile(
    r"\b(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never)\s*"
    r"(?:,|—|–)[^.!?;\n]{1,40}(?:,|—|–)\s*"
    r"(?=(?:support|include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)",
    re.IGNORECASE,
)
_BINARY_ACTION_OR_PREDICATE_RE = re.compile(
    r"(?:停止|启动|关闭|打开|断开|连接|检查|确认|按下|按住|点击|选择|通知|"
    r"呼叫|咨询|隔离|疏散|通风|重启|复位|处理|处置|应对|调整|设置|执行|"
    r"维护|维修|采取|联系|移除|清除|保持|放置|避免|使用|操作|运行|安装|"
    r"更换|清洁|等待|记录|报告|阅读|降低|提高|移动|触摸|接触|碰触|"
    r"拆解|拆卸|拆开|开启|关机|存储|储存|存放|校准|校正|支持|具备|"
    r"存在|适用|符合|满足|兼容|提供|配备|"
    r"\b(?:stop|start|restart|reset|shut\s+down|turn\s+(?:on|off)|"
    r"disconnect|connect|check|inspect|verify|confirm|press|click|select|"
    r"handle|remediate|adjust|notify|call|consult|isolate|evacuate|ventilate|"
    r"set|perform|maintain|repair|contact|remove|clear|keep|place|avoid|use|"
    r"operate|install|replace|clean|wait|record|report|read|reduce|increase|"
    r"move|touch|disassemble|dismantle|open|close|store|calibrate|support|"
    r"include|provide|comply|meet|have)(?:s|es|ed|ing)?\b)",
    re.IGNORECASE,
)
_BINARY_CONDITION_RE = re.compile(
    r"(?P<condition>[^。！？!?；;，,\n]{1,40}?(?:之后|以后|完成后|后|之前|以前|"
    r"前|时|情况下|环境下|状态下|条件下|期间|之中|中))\s*$|"
    r"\b(?P<english_condition>(?:if|when|whenever|unless|once|until|before|after|"
    r"only\s+after|under|during|in\s+case\s+of)\s+[^.!?;,\n]{1,80})\s*$",
    re.IGNORECASE,
)
_BINARY_GOVERNED_CONDITION_RE = re.compile(
    r"^\s*(?:在)?(?P<condition>[^。！？!?；;，,\n]{1,48}?)"
    r"(?:之后|以后|完成后|后|之前|以前|前|时|情况下|环境下|状态下|条件下|"
    r"期间|之中|中)\s*[,，]?\s*"
    r"(?P<governed>[^。！？!?；;\n]{1,96})$|"
    r"^\s*(?P<english_condition>(?:if|when|whenever|unless|once|until|before|after|"
    r"only\s+after|under|during|in\s+case\s+of)\s+[^.!?;,\n]{1,80})\s*[,;]\s*"
    r"(?P<english_governed>[^.!?;\n]{1,120})$",
    re.IGNORECASE,
)
_BINARY_SENTENCE_CONDITION_RE = re.compile(
    r"^\s*(?:(?:如果|若|倘若|仅当)\s*(?P<chinese_if>[^。！？!?；;，,\n]{1,48})\s*[,，]|"
    r"(?:当|在)?\s*(?P<chinese_temporal>[^。！？!?；;，,\n]{1,48}?)"
    r"(?:之后|以后|完成后|后|之前|以前|前|时|情况下|环境下|状态下|条件下|"
    r"期间|之中|中)\s*[,，]?|"
    r"(?P<english>(?:if|when|whenever|unless|once|until|before|after|only\s+after|"
    r"under|during|in\s+case\s+of)"
    r"\s+[^.!?;,\n]{1,80})\s*[,;])",
    re.IGNORECASE,
)
_BINARY_PREDICATE_TAIL_RE = re.compile(
    r"(?:因为|由于|以免|否则|原因是|\b(?:because|since|so\s+that|in\s+order\s+to)\b).*$",
    re.IGNORECASE,
)
_BINARY_TRAILING_CONNECTOR_RE = re.compile(
    r"(?:但|但是|而|且|并且|同时|\b(?:but|while|and|or)\b)\s*$",
    re.IGNORECASE,
)
_BINARY_OBJECT_PRESENTATION_RE = re.compile(
    r"^(?:(?:本|该|此|这个)?(?:设备|机器|装置|系统|产品)(?:的)?|"
    r"\b(?:the|a|an|device|machine|system|unit)(?:'s)?\b\s*)+",
    re.IGNORECASE,
)
_BINARY_GOVERNED_PRESENTATION_RE = re.compile(
    r"^\s*(?:(?:马上|立即|立刻|现在|当前|此时|当下|直接|先|再|同时|依次|务必)\s*|"
    r"\b(?:to|be|immediately|currently|now|directly|first|then)\b\s*)*",
    re.IGNORECASE,
)
_BINARY_INTRINSIC_PREDICATES = frozenset(
    {
        "支持",
        "具备",
        "存在",
        "适用",
        "符合",
        "满足",
        "兼容",
        "提供",
        "配备",
        "support",
        "include",
        "provide",
        "comply",
        "meet",
        "have",
    }
)
_TEMPORAL_ACTION_NOISE_RE = re.compile(
    r"(?:不能|不可以|不可|不得|请勿|切勿|禁止|严禁|不应|"
    r"马上|立即|立刻|现在|当前|此时|当下|"
    r"必须|务必|应当|应该|需要|需|请|再|才|方可|"
    r"本设备|该设备|此设备|设备的|\b(?:the|a|an|device|machine|"
    r"cannot|can't|must|not|may|can|should|immediately|right|now|at|once|"
    r"before|after|then|only|to)\b)",
    re.IGNORECASE,
)
_TEMPORAL_ASCII_ACTION_NOISE = frozenset(
    {
        "a",
        "an",
        "at",
        "before",
        "device",
        "immediately",
        "machine",
        "now",
        "once",
        "right",
        "the",
        "then",
        "to",
    }
)


@dataclass
class _NumericRelation:
    """局部数值关系原子；span 只用于识别同一范围的相邻端点。"""

    operator: str
    value: str
    unit: str
    field_key: str
    start: int
    end: int


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def _relation_text(value: str) -> str:
    """为关系提取统一兼容字符，但不删除承载事实的符号。"""

    return unicodedata.normalize("NFKC", value).casefold().translate(_RELATION_TRANSLATION)


def _canonical_number(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _canonical_unit(value: str | None) -> str:
    unit = "".join(_relation_text(value or "").split())
    unit = unit.replace("µ", "μ")
    return _UNIT_EQUIVALENTS.get(unit, unit)


def _operator_before(value: str, offset: int) -> str:
    # 最长英文表达也远小于 64 字符；限制局部窗口，避免前一字段的比较符串线。
    prefix = value[max(0, offset - 64) : offset]
    for operator, pattern in _RELATION_OPERATOR_PATTERNS:
        if pattern.search(prefix):
            return operator
    return "eq"


def _without_trailing_operator(value: str) -> str:
    for _, pattern in _RELATION_OPERATOR_PATTERNS:
        match = pattern.search(value)
        if match is not None:
            return value[: match.start()]
    return value


def _field_key_before(
    value: str,
    *,
    segment_start: int,
    offset: int,
    inherited_key: str,
) -> str:
    """提取当前数值左侧最近的 subject/predicate，禁止跨字段借用数值。"""

    segment = value[segment_start:offset]
    segment = _without_trailing_operator(segment)
    while True:
        trimmed = _FIELD_TRAILING_MODAL_RE.sub("", segment).rstrip()
        if trimmed == segment:
            break
        segment = trimmed
    segment = _FIELD_LEADING_CONNECTOR_RE.sub("", segment).strip()
    if not segment:
        return inherited_key
    segment = _FIELD_ENGLISH_ARTICLE_RE.sub("", segment)
    # “设备的运行温度”与“设备运行温度”是同一字段；其余字母数字均保留，
    # 尤其不能丢掉 A/B、左/右、运行/存储等判别成分。
    return "".join(character for character in segment if character.isalnum() and character != "的")


def _field_keys_compatible(
    claim_key: str,
    evidence_key: str,
    *,
    literal_clause_match: bool,
) -> bool:
    if not claim_key or not evidence_key:
        return not claim_key and not evidence_key and literal_clause_match
    if claim_key == evidence_key:
        return True
    # claim“运行温度”可由 Evidence“设备运行温度”支持；反向不成立，因为
    # claim 新增的对象限定不能由泛化 Evidence 补出。不同字段的判别词同样
    # 不能由相同数字或单位弥补。
    return len(claim_key) >= 2 and len(evidence_key) >= 2 and claim_key in evidence_key


def _numeric_relations(value: str) -> list[_NumericRelation]:
    """提取 NFKC 数值、比较符、正负号和单位，并规范闭区间端点。"""

    normalized = _relation_text(value)
    relations: list[_NumericRelation] = []
    for match in _RELATION_NUMBER_RE.finditer(normalized):
        number = _canonical_number(match.group("number"))
        if number is None:
            continue
        previous = relations[-1] if relations else None
        segment_start = previous.end if previous is not None else 0
        relations.append(
            _NumericRelation(
                operator=_operator_before(normalized, match.start()),
                value=number,
                unit=_canonical_unit(match.group("unit")),
                field_key=_field_key_before(
                    normalized,
                    segment_start=segment_start,
                    offset=match.start(),
                    inherited_key=previous.field_key if previous is not None else "",
                ),
                start=match.start(),
                end=match.end(),
            )
        )

    # “-20 至 50℃”与“≥-20℃ 且 ≤50℃”具有同一闭区间语义。范围只在
    # 相邻数值之间存在纯连接符时成立；普通句子里的额外数值不会被合并。
    for left, right in zip(relations, relations[1:]):
        connector = normalized[left.end : right.start]
        if _RANGE_CONNECTOR_RE.fullmatch(connector) is None:
            continue
        if not left.unit and right.unit:
            left.unit = right.unit
        elif left.unit and not right.unit:
            right.unit = left.unit
        left.operator = "ge"
        right.operator = "le"
        right.field_key = left.field_key
    return relations


def _numeric_clause_relations_preserved(claim: str, evidence: str) -> bool:
    """验证一个 claim 子句的数值关系、字段、比较符、符号和值。"""

    claim_relations = _numeric_relations(claim)
    if not claim_relations:
        return True
    evidence_relations = _numeric_relations(evidence)
    remaining = list(evidence_relations)
    claim_literal = "".join(_relation_text(claim).split()).strip("|`*_")
    evidence_literal = "".join(_relation_text(evidence).split()).strip("|`*_")
    literal_clause_match = claim_literal == evidence_literal
    for claim_relation in claim_relations:
        same_value = [
            relation
            for relation in evidence_relations
            if relation.value == claim_relation.value
            and relation.unit == claim_relation.unit
            and _field_keys_compatible(
                claim_relation.field_key,
                relation.field_key,
                literal_clause_match=literal_clause_match,
            )
        ]
        evidence_bounds = {
            relation.operator
            for relation in same_value
            if relation.operator in {"le", "ge", "lt", "gt"}
        }
        # Evidence 对同一 signed value/unit 明示了界限时，claim 不得省略成
        # 等值，也不得换成相反或宽松的比较符。
        if evidence_bounds and claim_relation.operator not in evidence_bounds:
            return False
        for index, relation in enumerate(remaining):
            if (
                relation.operator == claim_relation.operator
                and relation.value == claim_relation.value
                and relation.unit == claim_relation.unit
                and _field_keys_compatible(
                    claim_relation.field_key,
                    relation.field_key,
                    literal_clause_match=literal_clause_match,
                )
            ):
                remaining.pop(index)
                break
        else:
            return False
    return True


def _numeric_relation_clauses(value: str) -> list[str]:
    """切分关系子句，但不把合法千分位逗号误当作 claim 边界。"""

    normalized = _relation_text(value)
    clauses: list[str] = []
    start = 0
    for index, character in enumerate(normalized):
        if character not in "。！？!?；;,\n":
            continue
        grouping_comma = bool(
            character == ","
            and re.search(r"(?<!\d)\d{1,3}$", normalized[start:index])
            and re.match(r"\d{3}(?!\d)", normalized[index + 1 :])
        )
        if grouping_comma:
            continue
        clause = normalized[start:index].strip()
        if clause:
            clauses.append(clause)
        start = index + 1
    tail = normalized[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def local_numeric_relations_preserved(claim: str, evidence: str) -> bool:
    """逐局部子句绑定数值关系与字段；仅供严格 citation/grounding 门调用。"""

    claim_clauses = _numeric_relation_clauses(claim)
    evidence_clauses = _numeric_relation_clauses(evidence)
    if not claim_clauses:
        return True
    if not evidence_clauses:
        return not any(_numeric_relations(clause) for clause in claim_clauses)
    for claim_clause in claim_clauses:
        if not _numeric_relations(claim_clause):
            continue
        if not any(
            _numeric_clause_relations_preserved(claim_clause, evidence_clause)
            for evidence_clause in evidence_clauses
        ):
            return False
    return True


def ascii_token_keys(value: str) -> set[str]:
    """返回保留内部标点边界的 NFKC/casefold ASCII token 集。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {match.group(0) for match in _ASCII_TOKEN_RE.finditer(normalized)}


def _semantic_units(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    units = {match.group(0) for match in _ASCII_TOKEN_RE.finditer(normalized)}
    for match in _CJK_RUN_RE.finditer(normalized):
        run = match.group(0)
        if len(run) == 1:
            units.add(run)
        else:
            units.update(run[index : index + 2] for index in range(len(run) - 1))
    return units


def _polarity(value: str) -> str | None:
    if _NEGATIVE_RE.search(value):
        return "negative"
    if _REQUIREMENT_RE.search(value):
        return "required"
    return None


def _anchors(value: str) -> set[str]:
    result = {name for name, pattern in _ANCHOR_PATTERNS if pattern.search(value)}
    result.update(
        f"literal:{_normalized(token)}"
        for token in _ASCII_TOKEN_RE.findall(unicodedata.normalize("NFKC", value))
        if any(character.isdigit() for character in token)
    )
    return result


def restore_same_evidence_placeholders(
    claim: str,
    local_evidence: str,
    full_evidence: str,
) -> str:
    """用同 Evidence 中存在的完整字母数字 token 恢复局部单占位 token。"""

    restored = local_evidence
    full_tokens = ascii_token_keys(full_evidence)
    claim_literals = {
        unicodedata.normalize("NFKC", token).casefold(): token
        for token in _ASCII_TOKEN_RE.findall(unicodedata.normalize("NFKC", claim))
        if any(character.isdigit() for character in token)
    }
    for literal, display in claim_literals.items():
        if literal not in full_tokens or literal in ascii_token_keys(restored):
            continue
        matches = list(re.finditer(r"[A-Za-z0-9\ufffd]+", restored))
        for match in reversed(matches):
            token = match.group(0)
            normalized_token = "".join(
                character.casefold()
                for character in unicodedata.normalize("NFKC", token)
                if character.isalnum() or character == "\ufffd"
            )
            if (
                normalized_token.count("\ufffd") == 1
                and len(normalized_token) == len(literal)
                and all(
                    left == "\ufffd" or left == right
                    for left, right in zip(normalized_token, literal, strict=True)
                )
            ):
                restored = restored[: match.start()] + display + restored[match.end() :]
    return restored


def _semantic_overlap(claim: str, evidence: str) -> float:
    claim_units = _semantic_units(claim)
    if not claim_units:
        return 0.0
    return len(claim_units & _semantic_units(evidence)) / len(claim_units)


def _temporal_action_key(value: str) -> str:
    """规范 temporal gate 的动作/对象；不删除内外、左右等判别成分。"""

    value = unicodedata.normalize("NFKC", value).casefold()
    value = _TEMPORAL_ACTION_NOISE_RE.sub("", value)
    return "".join(character for character in value if character.isalnum())


def _temporal_ascii_token_candidates(token: str) -> frozenset[str]:
    """生成有限规则词形候选，不用词典猜测动作语义。"""

    candidates = {token}
    if len(token) >= 5 and token.endswith("ing"):
        stem = token[:-3]
        candidates.update({stem, f"{stem}e"})
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            candidates.add(stem[:-1])
    elif len(token) >= 4 and token.endswith("ed"):
        stem = token[:-2]
        candidates.update({stem, f"{stem}e"})
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            candidates.add(stem[:-1])
    elif len(token) >= 4 and token.endswith("es"):
        candidates.update({token[:-2], token[:-1]})
    elif len(token) >= 3 and token.endswith("s"):
        candidates.add(token[:-1])
    return frozenset(candidate for candidate in candidates if candidate)


def _temporal_ascii_action_tokens(value: str) -> tuple[frozenset[str], ...]:
    """按原顺序保留动作 token，并为每一位生成有限词形候选。"""

    return tuple(
        _temporal_ascii_token_candidates(raw)
        for raw in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold())
        if raw not in _TEMPORAL_ASCII_ACTION_NOISE
    )


def temporal_actions_compatible(claim_action: str, evidence_action: str) -> bool:
    """保守验证 temporal claim 与 Evidence 绑定同一动作和对象。"""

    claim_ascii = _temporal_ascii_action_tokens(claim_action)
    evidence_ascii = _temporal_ascii_action_tokens(evidence_action)
    if claim_ascii or evidence_ascii:
        return bool(
            claim_ascii
            and len(claim_ascii) == len(evidence_ascii)
            and all(
                claim_candidates & evidence_candidates
                for claim_candidates, evidence_candidates in zip(
                    claim_ascii,
                    evidence_ascii,
                    strict=True,
                )
            )
        )
    claim_key = _temporal_action_key(claim_action)
    evidence_key = _temporal_action_key(evidence_action)
    if min(len(claim_key), len(evidence_key)) < 4:
        return False
    if claim_key in evidence_key or evidence_key in claim_key:
        return True
    claim_pairs = {claim_key[index : index + 2] for index in range(len(claim_key) - 1)}
    evidence_pairs = {evidence_key[index : index + 2] for index in range(len(evidence_key) - 1)}
    if not claim_pairs or not evidence_pairs:
        return False
    return len(claim_pairs & evidence_pairs) / len(claim_pairs) >= 0.75


def _binary_english_condition_parts(value: str) -> tuple[str, str]:
    """对称提取英文句首或句尾的白名单条件。"""

    normalized = unicodedata.normalize("NFKC", value).strip().rstrip("。.!?！？；; ")
    for pattern in (
        _BINARY_LEADING_ENGLISH_CONDITION_RE,
        _BINARY_TRAILING_ENGLISH_CONDITION_RE,
    ):
        match = pattern.match(normalized)
        if match is not None:
            return str(match.group("core") or "").strip(), str(
                match.group("condition") or ""
            ).strip()
    return normalized, ""


def _direct_binary_query_predicate(value: str) -> str:
    """只提取二元问句中 modal 实际支配的 predicate。"""

    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized, _ = _binary_english_condition_parts(normalized)
    for pattern in (
        _BINARY_QUERY_MARKER_RE,
        _BINARY_QUERY_MODAL_RE,
        _BINARY_QUERY_ENGLISH_RE,
    ):
        match = pattern.search(normalized)
        if match is None:
            continue
        predicate = str(match.group("predicate") or "").strip().rstrip("吗呢 ?？")
        predicate = _BINARY_NESTED_MODAL_RE.sub("", predicate, count=1).strip()
        predicate, _ = _binary_governed_predicate(predicate)
        if predicate:
            return predicate
    return ""


def _binary_action_parts(value: str, *, anchored: bool = False) -> tuple[str, str] | None:
    """返回 predicate 的首个动作头及其动作—对象短语。"""

    candidate = _BINARY_PREDICATE_TAIL_RE.sub("", value).strip().rstrip("。.!?！？；; ")
    if anchored:
        candidate = _BINARY_GOVERNED_PRESENTATION_RE.sub("", candidate, count=1)
        match = _BINARY_ACTION_OR_PREDICATE_RE.match(candidate)
    else:
        match = _BINARY_ACTION_OR_PREDICATE_RE.search(candidate)
    if match is None:
        return None
    head = unicodedata.normalize("NFKC", match.group(0)).casefold().strip()
    phrase = candidate[match.start() :].strip()
    return head, phrase


def _binary_governed_predicate(value: str) -> tuple[str, str]:
    """剥离 modal 后的显式前置状语，返回真正被支配的 predicate。"""

    candidate = value.strip()
    match = _BINARY_GOVERNED_CONDITION_RE.match(candidate)
    if match is None:
        return candidate, ""
    condition = str(match.group("condition") or match.group("english_condition") or "").strip()
    governed = str(match.group("governed") or match.group("english_governed") or "").strip()
    return governed, condition


def _binary_action_heads_compatible(left: str, right: str) -> bool:
    """有限词形与显式同义组内比较，禁止用主题词猜动作。"""

    left_key = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", left.casefold())
    right_key = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", right.casefold())
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_ascii = _temporal_ascii_token_candidates(left_key)
    right_ascii = _temporal_ascii_token_candidates(right_key)
    if left_key.isascii() or right_key.isascii():
        return bool(left_key.isascii() and right_key.isascii() and left_ascii & right_ascii)
    return False


def _binary_predicates_compatible(
    query_predicate: str,
    claim_predicate: str,
    *,
    claim_anchored: bool = False,
) -> bool:
    """要求 predicate 的动作头与对象同时兼容。"""

    query_parts = _binary_action_parts(query_predicate)
    claim_parts = _binary_action_parts(claim_predicate, anchored=claim_anchored)
    if query_parts is None or claim_parts is None:
        return False
    query_head, query_action = query_parts
    claim_head, claim_action = claim_parts
    if not _binary_action_heads_compatible(query_head, claim_head):
        return False
    if query_head.isascii() or claim_head.isascii():
        query_object = query_action[len(query_head) :].strip()
        claim_object = claim_action[len(claim_head) :].strip()
        return bool(
            query_head.isascii()
            and claim_head.isascii()
            and query_object
            and claim_object
            and temporal_actions_compatible(claim_action, query_action)
        )
    query_object = query_action[len(query_head) :].strip()
    claim_object = claim_action[len(claim_head) :].strip()
    normalized_query_object = _BINARY_OBJECT_PRESENTATION_RE.sub("", query_object).strip()
    normalized_claim_object = _BINARY_OBJECT_PRESENTATION_RE.sub("", claim_object).strip()
    # “启动设备”中的“设备”是完整对象，只有后面仍有实质对象时才把设备词当展示前缀。
    if normalized_query_object:
        query_object = normalized_query_object
    if normalized_claim_object:
        claim_object = normalized_claim_object
    query_object_key = re.sub(
        r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", query_object).casefold(),
    )
    claim_object_key = re.sub(
        r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", claim_object).casefold(),
    )
    return bool(query_object_key and claim_object_key and query_object_key == claim_object_key)


def _binary_condition_is_in_query(prefix: str, query_clause: str) -> bool:
    """quote 中额外出现的前置条件不能被当成无条件 verdict。"""

    match = _BINARY_CONDITION_RE.search(prefix)
    if match is None:
        return True
    condition = str(match.group("condition") or match.group("english_condition") or "")
    condition_key = _temporal_action_key(condition)
    query_key = _temporal_action_key(query_clause)
    return bool(len(condition_key) >= 2 and condition_key in query_key)


def _binary_condition_text_is_in_query(condition: str, query_clause: str) -> bool:
    if not condition:
        return True
    condition_key = _temporal_action_key(condition)
    query_key = _temporal_action_key(query_clause)
    return bool(len(condition_key) >= 2 and condition_key in query_key)


def _binary_sentence_condition(value: str) -> str:
    match = _BINARY_SENTENCE_CONDITION_RE.match(value)
    if match is None:
        return ""
    return str(
        match.group("chinese_if") or match.group("chinese_temporal") or match.group("english") or ""
    ).strip()


def _binary_modal_polarity(value: str) -> int:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if normalized == "不" or re.match(
        r"^(?:(?:do|does|did)\s+not\b|don't\b|doesn't\b|didn't\b|never\b)",
        normalized,
    ):
        return -1
    return -1 if _BINARY_NEGATIVE_MODAL_RE.search(value) else 1


def _binary_intrinsic_segments(value: str) -> list[tuple[int, str]]:
    """按转折连接词拆分 intrinsic 断言，并保留相对偏移。"""

    segments: list[tuple[int, str]] = []
    start = 0
    for match in _BINARY_INTRINSIC_ADVERSATIVE_RE.finditer(value):
        segment = value[start : match.start()].strip()
        if segment:
            segment_start = (
                start
                + len(value[start : match.start()])
                - len(value[start : match.start()].lstrip())
            )
            segments.append((segment_start, segment))
        start = match.end()
    segment = value[start:].strip()
    if segment:
        segment_start = start + len(value[start:]) - len(value[start:].lstrip())
        segments.append((segment_start, segment))
    return segments or [(0, value)]


def _binary_intrinsic_residual_polarity(prefix: str) -> int | None:
    """解析未被 modal regex 消费、但紧邻 intrinsic head 的显式否定。"""

    normalized = unicodedata.normalize("NFKC", prefix).casefold().strip()
    if re.search(
        r"\b(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never)\s+"
        r"(?:only|merely|just)\s*$",
        normalized,
    ):
        return 1
    english = re.search(
        r"\b(?P<negator>(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never))"
        r"(?P<bridge>(?:\s+(?!(?:and|or|but|yet|then|although)\b)"
        r"[a-z][a-z-]*){0,6})\s*$",
        normalized,
    )
    if english is not None:
        bridge = " ".join(str(english.group("bridge") or "").split())
        if re.search(r"\b(?:not|never|no)\b", bridge):
            return 0
        if not bridge or re.fullmatch(
            r"(?:(?:actually|really|truly|currently|presently|now|yet|still)|"
            r"(?:in\s+fact|in\s+any\s+way|at\s+all))"
            r"(?:\s+(?:(?:actually|really|truly|currently|presently|now|yet|still)|"
            r"(?:in\s+fact|in\s+any\s+way|at\s+all)))*",
            bridge,
        ):
            return -1
        return 0
    chinese = re.search(
        r"(?P<negator>不再|从未|尚未|并未|未曾|从不|并不|没有)"
        r"(?P<bridge>[㐀-䶿一-鿿]{0,6})\s*$",
        normalized,
    )
    if chinese is not None:
        bridge = str(chinese.group("bridge") or "")
        if re.search(r"[不未没无]|[并且但却而或]", bridge):
            return 0
        if not bridge or re.fullmatch(
            r"(?:(?:真正|实际|确实|目前|当前|现在|继续|仍然|依然|再继续)){1,2}",
            bridge,
        ):
            return -1
        return 0
    if re.search(
        r"\b(?:(?:do|does|did)\s+not|don't|doesn't|didn't|never)\b|"
        r"(?:不再|从未|尚未|并未|未曾|从不|并不|没有)",
        normalized,
    ):
        return 0
    return None


def _binary_preserve_intrinsic_parenthetical_negation(
    value: str,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """去掉已知否定插入语的分隔符，并返回剩余未知结构的坐标。"""

    def replace(match: re.Match[str]) -> str:
        return "".join(
            " " if character in {",", "—", "–"} else character for character in match.group(0)
        )

    normalized = _BINARY_INTRINSIC_NEGATION_PARENTHETICAL_RE.sub(replace, value)
    unparsed_spans = tuple(
        match.span()
        for match in _BINARY_INTRINSIC_UNPARSED_NEGATION_PARENTHETICAL_RE.finditer(normalized)
    )
    return normalized, unparsed_spans


def _binary_unparsed_parenthetical_targets_query(
    value: str,
    spans: tuple[tuple[int, int], ...],
    *,
    query_predicate: str,
) -> bool:
    """未知否定插入语只在支配查询动作与对象时触发 fail-closed。"""

    for _, end in spans:
        tail = value[end:]
        immediate_clause = _CLAUSE_RE.match(tail)
        if immediate_clause is None:
            continue
        segments = _binary_intrinsic_segments(immediate_clause.group(0))
        if not segments:
            continue
        governed = segments[0][1]
        candidates = [governed]
        action_matches = list(_BINARY_ACTION_OR_PREDICATE_RE.finditer(governed))
        candidates.extend(
            governed[match.start() : action_matches[index + 1].start()].strip()
            for index, match in enumerate(action_matches)
            if index + 1 < len(action_matches)
        )
        if action_matches:
            candidates.append(governed[action_matches[-1].start() :].strip())
        coordinated = [
            item.strip()
            for item in _BINARY_INTRINSIC_COORDINATOR_RE.split(governed)
            if item.strip()
        ]
        candidates.extend(coordinated)
        first_parts = _binary_action_parts(coordinated[0], anchored=True) if coordinated else None
        inherited_head = first_parts[0] if first_parts is not None else ""
        for item in coordinated[1:]:
            if inherited_head and _binary_action_parts(item, anchored=True) is None:
                separator = " " if inherited_head.isascii() else ""
                candidates.append(f"{inherited_head}{separator}{item}")
        for candidate in candidates:
            candidate = _BINARY_TRAILING_CONNECTOR_RE.sub("", candidate).strip()
            candidate, _ = _binary_governed_predicate(candidate)
            candidate, _ = _binary_english_condition_parts(candidate)
            if _binary_predicates_compatible(
                query_predicate,
                candidate,
                claim_anchored=True,
            ):
                return True
    return False


def _binary_statement_is_uncertain(value: str, *, query_head: str) -> bool:
    """拒绝只表达可能性、依赖关系或互斥肯否的非确定结论。"""

    if _BINARY_EPISTEMIC_UNCERTAINTY_RE.search(value):
        return True
    if _BINARY_POLARITY_ALTERNATIVE_RE.search(value):
        return True
    return bool(
        query_head in _BINARY_INTRINSIC_PREDICATES and re.search(r"\bmay\b", value, re.IGNORECASE)
    )


def _binary_has_preceding_prerequisite(
    clause: str,
    modal_matches: list[re.Match[str]],
    *,
    modal_index: int,
) -> bool:
    """识别同一原子句中先于 verdict 的显式义务，不把有条件许可当无条件结论。"""

    for earlier_index in range(modal_index):
        earlier = modal_matches[earlier_index]
        if not _BINARY_PREREQUISITE_MODAL_RE.match(earlier.group(0).strip()):
            continue
        between = clause[earlier.end() : modal_matches[earlier_index + 1].start()]
        if not _BINARY_PREREQUISITE_CONNECTOR_RE.search(between):
            continue
        prerequisite = _BINARY_PREREQUISITE_CONNECTOR_RE.sub("", between).strip()
        if prerequisite:
            return True
    return False


def direct_binary_verdict_binds_query(quote: str, query_clause: str) -> bool:
    """验证同一原子句中的肯否 modal 确实支配 query 的动作与对象。"""

    normalized_quote = unicodedata.normalize("NFKC", quote).strip()
    normalized_query = unicodedata.normalize("NFKC", query_clause).strip()
    query_predicate = _direct_binary_query_predicate(normalized_query)
    query_parts = _binary_action_parts(query_predicate)
    if not normalized_quote or query_parts is None:
        return False

    relevant_gate_spans: list[tuple[int, int]] = []
    for pattern in _TEMPORAL_GATE_PATTERNS:
        for match in pattern.finditer(normalized_quote):
            gate_action = str(match.group("action") or "").strip()
            if gate_action and _binary_predicates_compatible(
                query_predicate,
                gate_action,
                claim_anchored=True,
            ):
                relevant_gate_spans.append(match.span())

    bound_polarities: set[int] = set()
    query_head = query_parts[0]
    for sentence_match in _BINARY_SENTENCE_RE.finditer(normalized_quote):
        sentence = sentence_match.group("sentence")
        sentence_condition = _binary_sentence_condition(sentence)
        if not sentence.strip() or not _binary_condition_text_is_in_query(
            sentence_condition,
            normalized_query,
        ):
            continue
        clause_source = sentence
        if query_head in _BINARY_INTRINSIC_PREDICATES:
            clause_source, unparsed_parenthetical_spans = (
                _binary_preserve_intrinsic_parenthetical_negation(sentence)
            )
            if _binary_unparsed_parenthetical_targets_query(
                clause_source,
                unparsed_parenthetical_spans,
                query_predicate=query_predicate,
            ):
                return False
        for clause_match in _CLAUSE_RE.finditer(clause_source):
            clause = clause_match.group(0)
            if not clause.strip():
                continue
            atomic_clauses = (
                _binary_intrinsic_segments(clause)
                if query_head in _BINARY_INTRINSIC_PREDICATES
                else [(0, clause)]
            )
            for relative_start, atomic_clause in atomic_clauses:
                clause_start = sentence_match.start() + clause_match.start() + relative_start
                modal_matches = list(_BINARY_VERDICT_MODAL_RE.finditer(atomic_clause))
                for modal_index, modal_match in enumerate(modal_matches):
                    absolute_modal = clause_start + modal_match.start()
                    if any(start <= absolute_modal < end for start, end in relevant_gate_spans):
                        continue
                    prefix = atomic_clause[: modal_match.start()]
                    if not _binary_condition_is_in_query(prefix, normalized_query):
                        continue
                    governed_end = (
                        modal_matches[modal_index + 1].start()
                        if modal_index + 1 < len(modal_matches)
                        else len(atomic_clause)
                    )
                    governed = atomic_clause[modal_match.end() : governed_end].strip()
                    governed = _BINARY_TRAILING_CONNECTOR_RE.sub("", governed).strip()
                    governed, governed_condition = _binary_governed_predicate(governed)
                    governed, trailing_condition = _binary_english_condition_parts(governed)
                    if any(
                        not _binary_condition_text_is_in_query(condition, normalized_query)
                        for condition in (governed_condition, trailing_condition)
                        if condition
                    ):
                        continue
                    if _binary_predicates_compatible(
                        query_predicate,
                        governed,
                        claim_anchored=True,
                    ):
                        if (
                            query_head in _BINARY_INTRINSIC_PREDICATES
                            and modal_match.group(0).strip() == "不"
                            and _binary_intrinsic_residual_polarity(prefix) in {-1, 0}
                        ):
                            return False
                        if _binary_statement_is_uncertain(
                            sentence,
                            query_head=query_head,
                        ) or _binary_has_preceding_prerequisite(
                            atomic_clause,
                            modal_matches,
                            modal_index=modal_index,
                        ):
                            return False
                        bound_polarities.add(_binary_modal_polarity(modal_match.group(0)))

                # 支持/具备等 predicate 的陈述本身就是二元事实；仍要求同一原子句
                # 重述完整动作与对象，且不能位于额外前置条件中。
                if query_head not in _BINARY_INTRINSIC_PREDICATES or modal_matches:
                    continue
                claim_core, claim_condition = _binary_english_condition_parts(atomic_clause)
                if not _binary_condition_text_is_in_query(
                    claim_condition,
                    normalized_query,
                ):
                    continue
                claim_parts = _binary_action_parts(claim_core)
                if (
                    claim_parts is not None
                    and _binary_action_heads_compatible(query_head, claim_parts[0])
                    and _binary_condition_is_in_query(
                        claim_core[: claim_core.lower().find(claim_parts[0].lower())],
                        normalized_query,
                    )
                    and _binary_predicates_compatible(query_predicate, claim_core)
                ):
                    if _binary_statement_is_uncertain(
                        sentence,
                        query_head=query_head,
                    ):
                        return False
                    claim_prefix = claim_core[: claim_core.lower().find(claim_parts[0].lower())]
                    residual_polarity = _binary_intrinsic_residual_polarity(claim_prefix)
                    if residual_polarity == 0:
                        return False
                    bound_polarities.add(residual_polarity or 1)
    return len(bound_polarities) == 1


def temporal_positive_immediate_permission_actions(value: str) -> list[str]:
    """提取当前/立即许可的动作，用于与 temporal gate 做同对象冲突检查。"""

    return [
        action
        for pattern in _TEMPORAL_POSITIVE_PERMISSION_PATTERNS
        for match in pattern.finditer(value)
        if (action := str(match.group("action") or "").strip())
    ]


def _temporal_negative_immediate_action(value: str) -> str | None:
    """提取单一“当前不能”命题的动作/对象，其余句型返回 ``None``。"""

    clauses = [match.group(0).strip() for match in _CLAUSE_RE.finditer(value)]
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    if (
        _TEMPORAL_NEGATIVE_VERDICT_RE.search(clause) is None
        or _TEMPORAL_IMMEDIATE_RE.search(clause) is None
    ):
        return None
    action = _TEMPORAL_NEGATIVE_VERDICT_RE.sub("", clause, count=1)
    action = _TEMPORAL_IMMEDIATE_RE.sub("", action, count=1).strip()
    return action if len(_temporal_action_key(action)) >= 4 else None


def local_temporal_gating_verdict_preserved(claim: str, evidence: str) -> bool:
    """验证“当前不能行动”是否仅由同动作/对象的强制前置 gate 单向推出。

    这是 citation/grounding 专用的窄义 alternative，不改变通用局部绑定语义。
    只允许 negative+immediate claim 由 ``先 gate 再 action`` 等强制顺序支持；
    正向许可、弱建议、相邻但无顺序绑定的子句和不同动作/对象均返回 ``False``。
    """

    claim_action = _temporal_negative_immediate_action(claim)
    if claim_action is None:
        return False
    if _TEMPORAL_WEAK_ADVICE_RE.search(evidence) or temporal_positive_immediate_permission_actions(
        evidence
    ):
        return False

    matches = [
        match
        for pattern in _TEMPORAL_GATE_PATTERNS
        for match in pattern.finditer(evidence)
        if temporal_actions_compatible(claim_action, str(match.group("action") or ""))
    ]
    return len(matches) == 1


def local_temporal_claim_support_preserved(claim: str, evidence: str) -> bool:
    """为 temporal claim 增加不可被通用局部绑定绕过的硬门。"""

    claim_action = _temporal_negative_immediate_action(claim)
    if claim_action is None:
        return True
    if _TEMPORAL_WEAK_ADVICE_RE.search(evidence) or temporal_positive_immediate_permission_actions(
        evidence
    ):
        return False
    for clause_match in _CLAUSE_RE.finditer(evidence):
        evidence_action = _temporal_negative_immediate_action(clause_match.group(0).strip())
        if evidence_action is not None and temporal_actions_compatible(
            claim_action,
            evidence_action,
        ):
            return True
    return local_temporal_gating_verdict_preserved(claim, evidence)


def local_claim_bindings_preserved(claim: str, evidence: str) -> bool:
    """确保每个局部子句的角色、值/数字和否定/必须极性保持绑定。

    该函数只做硬边界检查，不单独声明事实已获支持。普通无关系、无约束的
    说明句返回 ``True``，继续由调用方的词面/顺序算法判断。
    """

    claim_clauses = [match.group(0).strip() for match in _CLAUSE_RE.finditer(claim)]
    evidence_clauses = [match.group(0).strip() for match in _CLAUSE_RE.finditer(evidence)]
    if not claim_clauses or not evidence_clauses:
        return True

    for claim_clause in claim_clauses:
        anchors = _anchors(claim_clause)
        polarity = _polarity(claim_clause)
        # 单个角色词没有形成角色—值关系；单个数字仍由调用方的关键字面量
        # 门禁处理。两个以上锚点，或任何明确极性，才启用局部绑定硬门禁。
        if len(anchors) < 2 and polarity is None:
            continue
        matched = False
        for evidence_clause in evidence_clauses:
            if not anchors <= _anchors(evidence_clause):
                continue
            evidence_polarity = _polarity(evidence_clause)
            if polarity is not None and evidence_polarity != polarity:
                # “应确认设备已去除包装”与“请勿在未去除包装时使用”是同一
                # 安全状态的肯定/禁止双重表达。只允许带“已/确认”的完成态
                # 对同动作“未”状态做这一窄等价；“必须打开”绝不能因此被
                # “不得打开”支持。
                completed_state_equivalent = (
                    polarity == "required"
                    and evidence_polarity == "negative"
                    and ("已" in claim_clause or "确认" in claim_clause)
                    and "未" in evidence_clause
                    and _semantic_overlap(claim_clause, evidence_clause) >= 0.55
                )
                if not completed_state_equivalent:
                    continue
            # 极性必须和同一动作/对象绑定，不能只在相邻子句中出现同一个
            # “不得/必须”。角色—值对本身已足够区分时只需较低词面门槛。
            minimum_overlap = 0.35 if polarity is not None else 0.20
            if _semantic_overlap(claim_clause, evidence_clause) < minimum_overlap:
                continue
            matched = True
            break
        if not matched:
            return False
    return True
