"""
诊断脚本：观察 Milvus 中"实际写入的" question_kwd 字段与 content 字段的关系。

目的：
    不修改任何现有代码，纯观察，用于判断：
    1. question_kwd 字段是否真的有 LLM 生成的问句（包含"什么/怎么/哪里"等疑问词）
    2. question_kwd 与 content 实际差异有多大
    3. 验证当前 dense_vector 输入 = question_kwd（而非 content） 的假设

用法：
    python -m scripts.diagnose_question_kwd
    或
    cd STT && python scripts/diagnose_question_kwd.py
"""
import os
import sys
from pathlib import Path

# ---- 让脚本能 import STT 包 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- 加载 .env ----
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # dotenv 未安装时回退到手动解析
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from app.shared.clients.milvus_utils import get_milvus_client
from app.shared.config.milvus_config import milvus_config
from app.shared.runtime.logger import logger


def truncate(text: str, n: int = 200) -> str:
    if not text:
        return "<空>"
    if len(text) <= n:
        return text
    return text[:n] + f"... <截断，总长={len(text)}>"


def main():
    collection_name = milvus_config.chunks_collection  # "kb_chunks"
    print(f"\n{'='*80}")
    print(f"Milvus 诊断：观察 {collection_name} 中的 question_kwd 实际值")
    print(f"{'='*80}\n")

    client = get_milvus_client()
    if client is None:
        print("ERROR: Milvus 客户端连接失败，请检查 MILVUS_URL")
        return

    # 1. 先按 doc_id 查 1 条（用之前测试的 task_id）
    test_task_id = os.getenv("TEST_TASK_ID", "test_import_milvus_task_001")
    print(f"[1] 尝试按 doc_id == '{test_task_id}' 拉取 1 条样本 ...")

    res = client.query(
        collection_name=collection_name,
        filter=f'doc_id == "{test_task_id}"',
        output_fields=[
            "chunk_id", "content", "question_kwd",
            "important_kwd", "available_int", "position_int", "type",
        ],
        limit=3,
    )

    if not res:
        print(f"   ⚠️  doc_id='{test_task_id}' 没有记录，尝试拉取任意 3 条 ...")
        res = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=[
                "chunk_id", "content", "question_kwd",
                "important_kwd", "available_int", "position_int", "type",
            ],
            limit=3,
        )

    if not res:
        print("ERROR: Milvus 中没有可观察的数据（请先跑一次 node_import_milvus 联调测试）")
        return

    print(f"   ✅ 拉到 {len(res)} 条样本\n")

    # 2. 逐条打印
    for i, row in enumerate(res, 1):
        print(f"{'─'*80}")
        print(f"样本 {i}: chunk_id={row.get('chunk_id')}, type={row.get('type')}, "
              f"position={row.get('position_int')}, available={row.get('available_int')}")
        print(f"{'─'*80}")
        print(f"[content]      ({len(row.get('content') or '')} 字):")
        print(f"  {truncate(row.get('content'))}")
        print()
        print(f"[question_kwd] ({type(row.get('question_kwd')).__name__}):")
        qk = row.get("question_kwd")
        if isinstance(qk, list):
            for j, q in enumerate(qk, 1):
                print(f"  [{j}] {q}")
        else:
            print(f"  {qk!r}")
        print()
        print(f"[important_kwd] ({type(row.get('important_kwd')).__name__}):")
        ik = row.get("important_kwd")
        if isinstance(ik, list):
            for j, k in enumerate(ik[:10], 1):
                print(f"  [{j}] {k}")
            if len(ik) > 10:
                print(f"  ... <共 {len(ik)} 个>")
        else:
            print(f"  {ik!r}")
        print()

    # 3. 关键判定
    print(f"\n{'='*80}")
    print("关键判定：")
    print(f"{'='*80}")
    sample = res[0]
    qk = sample.get("question_kwd")
    if isinstance(qk, list) and qk:
        # 简单判定：是否有"什么/怎么/哪里/如何/吗/？/?" 这类疑问信号
        question_markers = ["什么", "怎么", "哪里", "如何", "为何", "为什么", "吗", "？", "?", "哪", "谁", "几个", "是否"]
        has_marker = any(
            any(marker in str(q) for marker in question_markers)
            for q in qk
        )
        if has_marker:
            print("✅ question_kwd 包含疑问信号（什么/怎么/？ 等）")
            print("   → 说明 LLM 节点确实生成了假设问句")
        else:
            print("⚠️  question_kwd 没有疑问信号，看起来像从原句抽词")
            print("   → 可能 LLM 节点没真正生成问句，需要回查 node_bge_embedding 上游")

        # 文本重合度估算
        content = sample.get("content") or ""
        qk_joined = "\n".join(str(q) for q in qk)
        overlap_chars = sum(1 for c in qk_joined if c in content)
        print(f"\n   content 长度={len(content)}, question_kwd 拼接长度={len(qk_joined)}")
        print(f"   字符重叠数={overlap_chars}（粗略估计，不计 token 切分）")
    else:
        print(f"⚠️  question_kwd 为空或非列表: {qk!r}")
        print("   → 说明该 chunk 的 LLM 增强阶段未生成问句（可能上游没跑 / 失败）")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
