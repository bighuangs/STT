import copy
from langgraph.graph import StateGraph, END
from langgraph.types import Send  # v1.0+ 使用此路径
from app.foundation import LazyObject
from app.shared.runtime.logger import logger

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_metadata_extract import node_metadata_extract
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_link_and_table_chunk import node_link_and_table_chunk
from app.process.import_.agent.nodes.node_chunk_merge import node_chunk_merge
from app.process.import_.agent.nodes.node_image_upload import node_image_upload
from app.process.import_.agent.nodes.node_keyword_extraction import node_keyword_extraction
from app.process.import_.agent.nodes.node_question_proposal import node_question_proposal
from app.process.import_.agent.nodes.node_metadata_generate import node_metadata_generate
from app.process.import_.agent.nodes.node_content_tagging import node_content_tagging
from app.process.import_.agent.nodes.node_toc import node_toc
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus

workflow = StateGraph(ImportGraphState)

# ─────────────────────────────────────────────────────────────────────────────
# 基础节点（非并行）
# ─────────────────────────────────────────────────────────────────────────────
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_metadata_extract", node_metadata_extract)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_link_and_table_chunk", node_link_and_table_chunk)
workflow.add_node("node_chunk_merge", node_chunk_merge)
workflow.add_node("node_image_upload", node_image_upload)


def node_llm_fanout_start(state: ImportGraphState) -> ImportGraphState:
    """TODO-02 可恢复屏障：图片已上传后，从这里安全地重新触发并行增强。"""
    return state


workflow.add_node("node_llm_fanout_start", node_llm_fanout_start)

# ─────────────────────────────────────────────────────────────────────────────
# 并行 LLM 增强节点的结果合并（Reducer 节点）
# ─────────────────────────────────────────────────────────────────────────────
def llm_enhanceReducer(state: ImportGraphState) -> ImportGraphState:
    """
    Reducer 节点：合并 5 个并行 LLM 增强节点的增量结果到主 state。

    并行节点各自在 fanout 阶段拿到一份 chunks 的**深拷贝**（彼此隔离，避免跨线程
    并发原地修改同一批 dict），各自增强后通过 state["llm_enhance_results"] 上报
    {"chunks": <该节点增强后的 chunks 快照>, "toc": <可选>}。
    本函数把这些增量按索引 dict.update() 合并回基准 chunks，确保即使后续启用
    checkpointer / 序列化导致共享引用失效，5 个维度的增强字段也不会相互覆盖丢失。

    合并策略：
    - chunks: 各节点的增量字段（key_words / question_kwd / metadata / tags / id 等）
              按索引 dict.update() 合并（只新增/覆盖各节点写入的键，互不影响）
    - toc: 使用最后一个非空结果（仅 node_toc 会上报 toc）
    """
    merged = state.copy()
    results = state.get("llm_enhance_results", [])

    if not results:
        return merged

    # 获取原始 chunks（image_upload 阶段的结果）
    original_chunks = merged.get("chunks", [])
    if not original_chunks:
        logger.warning("[llm_enhance_reducer] 原始 chunks 为空，跳过合并")
        return merged

    # 复制一份用于合并
    merged_chunks = [chunk.copy() for chunk in original_chunks]

    # 遍历所有并行节点的结果，合并增量字段
    for result in results:
        if not result:
            continue

        # 合并 chunks 增量
        result_chunks = result.get("chunks", [])
        if result_chunks and len(result_chunks) == len(merged_chunks):
            for i, chunk_update in enumerate(result_chunks):
                if isinstance(chunk_update, dict):
                    merged_chunks[i].update(chunk_update)

        # 合并 toc（使用最后一个有效结果）
        if result.get("toc"):
            merged["toc"] = result["toc"]

    merged["chunks"] = merged_chunks
    merged["llm_enhance_results"] = []  # 清空收集结果，准备下一轮

    logger.info(f"[llm_enhance_reducer] 合并完成，共 {len(results)} 个并行节点结果")
    return merged


workflow.add_node("llm_enhance_reducer", llm_enhanceReducer)

# ─────────────────────────────────────────────────────────────────────────────
# 单独的并行节点定义（不直接连接到主图，而是通过 Send API 调用）
# ─────────────────────────────────────────────────────────────────────────────
workflow.add_node("node_keyword_extraction", node_keyword_extraction)
workflow.add_node("node_question_proposal", node_question_proposal)
workflow.add_node("node_metadata_generate", node_metadata_generate)
workflow.add_node("node_content_tagging", node_content_tagging)
workflow.add_node("node_toc", node_toc)

workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ─────────────────────────────────────────────────────────────────────────────
# 边定义
# ─────────────────────────────────────────────────────────────────────────────
# 入口路由
def after_entry_node(state: ImportGraphState) -> str:
    if state.get("is_document_read_enabled"):
        logger.info(
            f"{state['local_file_path']} 检测为 {state.get('input_format')}，"
            "路由到兼容节点 node_pdf_to_md（内部为通用 ParserRegistry）"
        )
        return "node_pdf_to_md"
    # 兼容升级前持久 checkpoint 中只有两个旧布尔字段的状态。
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "node_pdf_to_md"
    else:
        logger.warning(f"{state['local_file_path']} 无法识别文件类型，返回END")
        return END

workflow.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END,
    },
)

# 顺序执行阶段
workflow.add_edge("node_pdf_to_md", "node_metadata_extract")
workflow.add_edge("node_metadata_extract", "node_md_img")
workflow.add_edge("node_md_img", "node_link_and_table_chunk")
workflow.add_edge("node_link_and_table_chunk", "node_chunk_merge")
workflow.add_edge("node_chunk_merge", "node_image_upload")

# ─────────────────────────────────────────────────────────────────────────────
# 并行阶段：使用 conditional_edges + Send API 实现真正的并行
# ─────────────────────────────────────────────────────────────────────────────
LLM_ENHANCE_NODES = [
    "node_keyword_extraction",
    "node_question_proposal",
    "node_metadata_generate",
    "node_content_tagging",
    "node_toc",
]


def fanout_to_llm_enhance(state: ImportGraphState) -> list[dict]:
    """
    Fanout 函数：返回 Send 对象列表，实现真正的并行执行。

    使用 Send API：
    - 每个 Send 对象指定目标节点和要传递的 state 快照
    - LangGraph 会并行执行所有 Send 目标节点
    - 结果由 llm_enhance_reducer 收集合并

    关键：为每个并行节点单独**深拷贝** chunks（其余大对象如 image_pil_dict 这些
    并行节点只读、按引用共享以省内存），使各节点在独立的 chunk 副本上原地修改，
    避免多线程并发写同一批 dict 的竞态；增量再由 llm_enhance_reducer 显式合并，
    不依赖"共享引用碰巧生效"这一隐式契约。
    """
    logger.info(f"[fanout] 启动 {len(LLM_ENHANCE_NODES)} 个并行 LLM 增强节点")
    base_chunks = state.get("chunks", []) or []
    sends: list[Send] = []
    for node_name in LLM_ENHANCE_NODES:
        snapshot = dict(state)  # 浅拷贝外层；下方仅隔离 chunks
        snapshot["chunks"] = copy.deepcopy(base_chunks)
        snapshot["llm_enhance_results"] = []  # 避免把上游累积值带入并行分支
        sends.append(Send(node_name, snapshot))
    return sends


# image_upload 之后先经过可恢复屏障，再使用 conditional_edges 触发并行执行。
# 这样 checkpoint 不必重放 image_upload，也不需要序列化已关闭的 PIL.Image。
workflow.add_edge("node_image_upload", "node_llm_fanout_start")
workflow.add_conditional_edges(
    "node_llm_fanout_start",
    fanout_to_llm_enhance,
    # 由于使用了 Send API，targets 参数在这里是占位
    {node: node for node in LLM_ENHANCE_NODES}
)

# reducer 节点从所有并行节点收集结果
for node_name in LLM_ENHANCE_NODES:
    workflow.add_edge(node_name, "llm_enhance_reducer")

# reducer 之后继续向量化
workflow.add_edge("llm_enhance_reducer", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

RESUMABLE_ENTRY_NODES = [
    "node_entry",
    "node_pdf_to_md",
    "node_metadata_extract",
    "node_md_img",
    "node_link_and_table_chunk",
    "node_chunk_merge",
    "node_image_upload",
    "node_llm_fanout_start",
    "llm_enhance_reducer",
    "node_bge_embedding",
    "node_import_milvus",
]


def route_resume_entry(state: ImportGraphState) -> str:
    requested = state.get("resume_node") or "node_entry"
    if requested not in RESUMABLE_ENTRY_NODES:
        logger.warning(f"[resume] 非法恢复节点 {requested}，退回 node_entry")
        return "node_entry"
    if requested != "node_entry":
        logger.info(f"[resume] 从持久检查点继续：{requested}")
    return requested


workflow.set_conditional_entry_point(
    route_resume_entry,
    {node: node for node in RESUMABLE_ENTRY_NODES},
)

def compile_ingestion_graph():
    """编译导入图；生产 Worker 应通过 ``app.runtime.graphs`` 调用。"""

    return workflow.compile()


# 兼容旧脚本的导入符号；图只在首次使用时编译。
kb_import_app = LazyObject(compile_ingestion_graph)


# ─────────────────────────────────────────────────────────────────────────────
# Mermaid 流程图生成
# 运行：python -m app.process.import_.agent.main_graph mermaid
# ─────────────────────────────────────────────────────────────────────────────
def generate_mermaid_flowchart() -> str:
    """
    生成当前 LangGraph 导入工作流的 Mermaid 流程图字符串。

    忠实反映图的实际拓扑：
      - 顺序段：entry → 文件类型条件路由 → 分块 → 图片上传
      - 条件路由：PDF / Markdown / 未知格式
      - 并行扇出（Send API）：image_upload → 5 个 LLM 增强节点同时执行
      - 汇聚：5 个节点 → llm_enhance_reducer
      - 收尾顺序：bge_embedding → import_milvus → END

    Returns:
        Mermaid flowchart TD 格式字符串，可直接粘贴到 https://mermaid.live 预览，
        或写入 .md 文件的代码块中。
    """
    parallel_color = "fill:#d4edda,stroke:#28a745,color:#155724"
    lines: list[str] = [
        "flowchart TD",
        "",
        "    %% ── 起止节点 ──",
        "    Start([开始])",
        "    End([结束])",
        "",
        "    %% ── 入口与文件类型路由 ──",
        '    Start --> entry["node_entry\n入口 / 文件类型解析"]',
        '    entry -->|PDF/MD/TXT/HTML/CSV/DOCX/PPTX/XLSX| pdf_to_md["node_pdf_to_md\nParserRegistry + 版本化模板"]',
        "    entry -->|未知格式| End",
        "",
        "    %% ── 统一结构解析阶段；旧节点名用于 checkpoint 兼容 ──",
        '    pdf_to_md --> meta_ext["node_metadata_extract\n结构化路径兼容检查"]',
        "    meta_ext --> md_img",
        "",
        "    %% ── 顺序分块阶段 ──",
        '    md_img --> link_table["node_link_and_table_chunk\n结构化路径保留 provenance"]',
        '    link_table --> chunk_merge["node_chunk_merge\n模板 chunk 兼容检查"]',
        '    chunk_merge --> img_upload["node_image_upload\n图片上传 MinIO"]',
        "",
        "    %% ── 并行 LLM 增强（Send API，5 节点同时执行）──",
        '    img_upload --> keyword["node_keyword_extraction\nLLM 关键词提取"]',
        '    img_upload --> question["node_question_proposal\nLLM 问题生成"]',
        '    img_upload --> meta_gen["node_metadata_generate\nLLM 元数据生成"]',
        '    img_upload --> tagging["node_content_tagging\nLLM 内容打标"]',
        '    img_upload --> toc["node_toc\nLLM 目录构建"]',
        "",
        "    %% ── 并行结果汇聚（Reducer）──",
        '    keyword --> reducer["llm_enhance_reducer\n增量结果合并"]',
        "    question --> reducer",
        "    meta_gen --> reducer",
        "    tagging --> reducer",
        "    toc --> reducer",
        "",
        "    %% ── 向量化与入库 ──",
        '    reducer --> bge["node_bge_embedding\nBGE-M3 混合向量化"]',
        '    bge --> milvus["node_import_milvus\nMilvus 向量入库"]',
        "    milvus --> End",
        "",
        "    %% ── 样式：颜色分区标注节点角色 ──",
        f"    style keyword {parallel_color}",
        f"    style question {parallel_color}",
        f"    style meta_gen {parallel_color}",
        f"    style tagging {parallel_color}",
        f"    style toc     {parallel_color}",
        "    style reducer  fill:#fff3cd,stroke:#ffc107,color:#856404",
        "    style img_upload fill:#cce5ff,stroke:#0062cc,color:#004085",
        "    style bge      fill:#f8d7da,stroke:#c82333,color:#721c24",
        "    style milvus   fill:#f8d7da,stroke:#c82333,color:#721c24",
        "    style Start    fill:#e2e3e5,stroke:#6c757d",
        "    style End      fill:#e2e3e5,stroke:#6c757d",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 并发限流测试（自包含，无需 LLM / PDF / Milvus）
# 运行：python -m app.process.import_.agent.main_graph limiter
# ─────────────────────────────────────────────────────────────────────────────
def _test_chat_limiter_concurrency() -> None:
    """
    验证 llm_limiter.chat_limiter / LoopLocalSemaphore 的并发限流行为。

    纯逻辑测试：用可控的 async 探针任务模拟"扇出并发 LLM 调用"，统计实际并发峰值，
    断言其不超过信号量上限。不触发任何真实 LLM/网络调用，可秒级运行。

    覆盖用例：
        1. 全新 LoopLocalSemaphore 在【单个事件循环】内严格把并发限制到上限
        2. 共享实例 chat_limiter 同样生效（上限 = MAX_CONCURRENT_CHATS）
        3. 「每事件循环独立」语义：多线程各自 asyncio.run 时，limiter 在每个循环内
           分别限流（这正是 STT 5 个并行 LLM 节点的真实运行模型）——每循环峰值 <= 上限，
           但全局峰值可达 ~线程数×上限（已知语义边界，见 llm_limiter 模块说明）
    """
    import asyncio
    import sys
    import threading

    from app.shared.runtime.logger import logger
    from app.rag.import_.llm_limiter import (
        LoopLocalSemaphore,
        chat_limiter,
        MAX_CONCURRENT_CHATS,
    )

    logger.info("=" * 70)
    logger.info(" 并发限流测试 (chat_limiter / LoopLocalSemaphore)")
    logger.info("=" * 70)

    errors: list[str] = []

    def _probe_single_loop(limiter, total_tasks: int, work_ms: int = 30,
                           global_tracker: dict | None = None,
                           global_lock=None) -> int:
        """
        在一个独立事件循环内并发跑 total_tasks 个探针任务，返回本循环观测到的并发峰值。

        每个任务先经 `async with limiter` 获取许可，再更新并发计数（单线程 asyncio 中，
        计数更新两行之间无 await，是原子的），sleep 模拟 LLM 耗时后释放。
        可选传入跨线程的 global_tracker + global_lock 以统计全局峰值。
        """
        local = {"current": 0, "peak": 0}

        async def _worker():
            async with limiter:
                local["current"] += 1
                local["peak"] = max(local["peak"], local["current"])
                if global_tracker is not None and global_lock is not None:
                    with global_lock:
                        global_tracker["current"] += 1
                        global_tracker["peak"] = max(
                            global_tracker["peak"], global_tracker["current"]
                        )
                await asyncio.sleep(work_ms / 1000)
                if global_tracker is not None and global_lock is not None:
                    with global_lock:
                        global_tracker["current"] -= 1
                local["current"] -= 1

        async def _runner():
            await asyncio.gather(*[_worker() for _ in range(total_tasks)])

        asyncio.run(_runner())
        return local["peak"]

    # ── 用例 1：全新 limiter 在单循环内严格不超过上限，且能打满到上限 ──
    limit1 = 3
    peak1 = _probe_single_loop(LoopLocalSemaphore(limit1), total_tasks=12)
    logger.info(f"[用例1] 上限={limit1}, 任务=12 → 观测并发峰值={peak1}")
    if peak1 > limit1:
        errors.append(f"用例1：并发峰值 {peak1} 超过上限 {limit1}（限流失效）")
    if peak1 != limit1:
        errors.append(f"用例1：并发峰值 {peak1} 未打满到上限 {limit1}（未充分并发，测试无意义）")

    # ── 用例 2：共享 chat_limiter 在单循环内不超过 MAX_CONCURRENT_CHATS ──
    if MAX_CONCURRENT_CHATS >= 1:
        peak2 = _probe_single_loop(chat_limiter, total_tasks=MAX_CONCURRENT_CHATS * 3)
        logger.info(
            f"[用例2] chat_limiter 上限={MAX_CONCURRENT_CHATS}, "
            f"任务={MAX_CONCURRENT_CHATS * 3} → 观测并发峰值={peak2}"
        )
        if peak2 > MAX_CONCURRENT_CHATS:
            errors.append(f"用例2：并发峰值 {peak2} 超过上限 {MAX_CONCURRENT_CHATS}")
        if peak2 != MAX_CONCURRENT_CHATS:
            errors.append(
                f"用例2：并发峰值 {peak2} 未打满到上限 {MAX_CONCURRENT_CHATS}"
            )
    else:
        logger.warning(f"[用例2] MAX_CONCURRENT_CHATS={MAX_CONCURRENT_CHATS} < 1，跳过")

    # ── 用例 3：每事件循环独立（STT 并行节点真实模型）──
    # 两个线程各自 asyncio.run，共享同一个 limiter 实例。
    # 断言：每个循环内峰值 <= 上限；并记录全局峰值以印证"每循环独立"的语义边界。
    limit3 = 3
    shared = LoopLocalSemaphore(limit3)
    g_tracker = {"current": 0, "peak": 0}
    g_lock = threading.Lock()
    per_loop_peaks: list[int] = []
    peaks_lock = threading.Lock()

    def _thread_body():
        p = _probe_single_loop(
            shared, total_tasks=12, global_tracker=g_tracker, global_lock=g_lock
        )
        with peaks_lock:
            per_loop_peaks.append(p)

    threads = [threading.Thread(target=_thread_body) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info(
        f"[用例3] 2 线程各 12 任务, 每循环上限={limit3} → "
        f"各循环峰值={per_loop_peaks}, 全局峰值={g_tracker['peak']}"
    )
    for i, p in enumerate(per_loop_peaks):
        if p > limit3:
            errors.append(f"用例3：线程{i} 循环内峰值 {p} 超过上限 {limit3}")
    if g_tracker["peak"] > limit3:
        logger.info(
            f"[用例3] 注：全局峰值 {g_tracker['peak']} > 单循环上限 {limit3}，"
            f"印证 LoopLocalSemaphore 为『每事件循环独立』语义"
            f"（跨并行节点非全局限流，详见 llm_limiter 模块说明）"
        )

    # ── 汇总 ──
    if errors:
        logger.error("并发限流测试未通过：")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    logger.info("=" * 70)
    logger.info(" 并发限流测试全部通过 ✅")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys

    # 子命令分发
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        # `limiter`：仅跑并发限流测试（无需 LLM/PDF/Milvus，秒级完成）
        if cmd == "limiter":
            _test_chat_limiter_concurrency()
            sys.exit(0)

        # `mermaid`：打印 Mermaid 流程图；可追加文件路径写入 .md
        if cmd == "mermaid":
            diagram = generate_mermaid_flowchart()
            if len(sys.argv) > 2:
                out_path = sys.argv[2]
                with open(out_path, "w", encoding="utf-8") as _f:
                    _f.write("```mermaid\n")
                    _f.write(diagram)
                    _f.write("\n```\n")
                print(f"Mermaid 流程图已写入: {out_path}")
            else:
                print(diagram)
            sys.exit(0)

    # =========================================================================
    # 全流程测试：验证 PDF 导入 → Milvus 入库完整链路
    # =========================================================================
    import os
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.shared.runtime.logger import logger

    logger.info("=" * 70)
    logger.info(" 知识图谱导入全流程测试")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. 配置测试参数
    # ─────────────────────────────────────────────────────────────────────────
    test_pdf_name = "联想海豚用户手册.pdf"
    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", test_pdf_name)
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    test_kb_id = "test_kb_001"

    os.makedirs(test_output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 校验测试文件
    # ─────────────────────────────────────────────────────────────────────────
    if not os.path.exists(test_pdf_path):
        logger.error(f"测试 PDF 文件不存在: {test_pdf_path}")
        logger.info("请将测试 PDF 文件放入项目根目录的 doc 文件夹中")
        sys.exit(1)

    logger.info(f"  PDF 路径: {test_pdf_path}")
    logger.info(f"  输出目录: {test_output_dir}")
    logger.info(f"  知识库 ID: {test_kb_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. 构造初始状态
    # ─────────────────────────────────────────────────────────────────────────
    from app.process.import_.agent.state import create_default_state

    test_state = create_default_state(
        task_id="test_kg_import_workflow_003",
        local_file_path=test_pdf_path,
        local_dir=test_output_dir,
        kb_id=test_kb_id,
        is_pdf_read_enabled=True,
        is_md_read_enabled=False,
        parser_config={
            "auto_keywords": 5,
            "auto_questions": 3,
            "toc_extraction": False,
            "chunk_char_num": 512,
            "overlapped_percent": 0,  # 25% 重叠
            # 元数据生成（node_metadata_generate → Milvus meta_fields）
            # 不配 enable_metadata / metadata 时，meta_fields 会落默认 "{}"
            "enable_metadata": True,
            "metadata": [
                # 注意：自由文本字段的 description 不要写具体示例值（如某个真实型号），
                # 否则 LLM 会把示例当默认答案抄到无关文档上，造成跨文档元数据串味。
                # scope=document：文档级属性，逐 chunk 抽取后按全文聚合并统一回写，
                #   避免少数 chunk 误标污染 doc 级目录（如操作手册里的注意事项段被标成安全手册）。
                {"key": "设备型号", "scope": "document",
                 "description": "本段内容中明确出现的设备/产品型号名称；若未提及具体型号则不要输出该字段"},
                {
                    "key": "文档类型",
                    "scope": "document",
                    "description": "文档类别",
                    "enum": ["操作手册", "规格书", "安全手册", "维护手册"],
                },
                {"key": "适用场景", "description": "本段内容所描述的操作或主题（动作/环节名词），仅在内容确有相关描述时输出"},
            ],
        }
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. 执行全流程
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("-" * 70)
    logger.info(" 开始执行 LangGraph 全流程...")
    logger.info(" 节点: entry → pdf_to_md → metadata_extract → md_img")
    logger.info("      → link_table → chunk_merge → image_upload")
    logger.info("      → [5 并行 LLM 增强] → bge_embedding → import_milvus")
    logger.info("-" * 70)

    final_state = None
    node_execution_order = []

    try:
        # 同时订阅两种流模式：
        #   - "updates": 产出 {节点名: 增量}，用于正确统计节点执行顺序
        #   - "values" : 产出完整 state，用于拿到最终状态
        for mode, payload in kb_import_app.stream(
            test_state, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                for current_node in payload:
                    node_execution_order.append(current_node)
                    logger.info(f"  [OK] {current_node}")
            elif mode == "values":
                final_state = payload

    except Exception as e:
        logger.exception(f"全流程执行失败: {e}")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. 结果验证与统计
    # ─────────────────────────────────────────────────────────────────────────
    if not final_state:
        logger.error("全流程执行结果为空")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info(" 全流程测试结果汇总")
    logger.info("=" * 70)

    # 5.1 执行统计
    logger.info("")
    logger.info("[执行统计]")
    logger.info(f"  节点总数: {len(node_execution_order)}")
    logger.info(f"  执行顺序: {' → '.join(node_execution_order)}")

    # 5.2 内容预览
    logger.info("")
    logger.info("[内容预览]")
    md_content = final_state.get("md_content", "")
    logger.info(f"  MD 内容长度: {len(md_content)} 字符")
    logger.info(f"  MD 预览: {md_content[:200]}...")

    # 5.3 分块统计
    chunks = final_state.get("chunks", [])
    logger.info("")
    logger.info("[分块统计]")
    logger.info(f"  切片总数: {len(chunks)}")

    if chunks:
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get("type", "text")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        logger.info(f"  类型分布: {type_counts}")

        logger.info("  内容预览（前3个）:")
        for i, chunk in enumerate(chunks[:3], 1):
            content = chunk.get("content", "")[:100]
            chunk_type = chunk.get("type", "text")
            logger.info(f"    [{i}] type={chunk_type}, len={len(chunk.get('content', ''))}, content={content}...")

    # 5.4 向量化验证
    logger.info("")
    logger.info("[向量化验证]")
    # hyperlinks 类型的 chunk 按设计不嵌入向量，只检查非 hyperlinks 的 chunk
    non_hl_chunks = [c for c in chunks if c.get("type") != "hyperlinks"] if chunks else []
    has_dense = all("dense_vector" in c for c in non_hl_chunks) if non_hl_chunks else True
    has_sparse = all("sparse_vector" in c for c in non_hl_chunks) if non_hl_chunks else True
    hl_count = len(chunks) - len(non_hl_chunks) if chunks else 0
    logger.info(f"  dense_vector: {'OK' if has_dense else 'MISS'} ({len(non_hl_chunks)} 个非 hyperlinks chunk)")
    logger.info(f"  sparse_vector: {'OK' if has_sparse else 'MISS'} ({len(non_hl_chunks)} 个非 hyperlinks chunk)")
    if hl_count > 0:
        logger.info(f"  注: {hl_count} 个 hyperlinks chunk 按设计跳过向量化")

    # 5.5 Milvus 入库验证
    logger.info("")
    logger.info("[Milvus 入库验证]")
    has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
    logger.info(f"  chunk_id: {'OK' if has_chunk_id else 'MISS'}")

    # 5.6 TOC 验证
    toc = final_state.get("toc", {})
    logger.info("")
    logger.info("[目录（TOC）验证]")
    if toc:
        toc_items = toc.get("items", [])
        logger.info(f"  TOC 条目数: {len(toc_items)}")
        for item in toc_items[:5]:
            logger.info(f"    level={item.get('level')}, title={item.get('title')}")
    else:
        logger.info("  TOC 未生成（或未启用 toc_extraction）")

    # 5.7 图片处理验证
    img_id_dict = final_state.get("img_id_dict", {})
    logger.info("")
    logger.info("[图片处理验证]")
    logger.info(f"  上传图片数量: {len(img_id_dict)}")
    for img_name, img_id in list(img_id_dict.items())[:3]:
        logger.info(f"    {img_name}: {img_id}")

    # 5.8 LLM 增强验证
    logger.info("")
    logger.info("[LLM 增强验证]")
    has_keywords = all("key_words" in c for c in chunks) if chunks else False
    has_questions = all("question_kwd" in c for c in chunks) if chunks else False
    logger.info(f"  关键词提取: {'OK' if has_keywords else 'SKIP'}")
    logger.info(f"  问题生成: {'OK' if has_questions else 'SKIP'}")

    # 5.9 最终状态键
    logger.info("")
    logger.info("[最终状态键]")
    logger.info(f"  {list(final_state.keys())}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. 断言验证
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info(" 断言验证")
    logger.info("=" * 70)

    errors = []
    if not chunks:
        errors.append("切片列表为空")
    else:
        # hyperlinks 类型的 chunk 按设计不需要向量，只检查非 hyperlinks 的 chunk
        if non_hl_chunks:
            if not has_dense:
                errors.append("部分非 hyperlinks 切片缺少 dense_vector")
            if not has_sparse:
                errors.append("部分非 hyperlinks 切片缺少 sparse_vector")
        if not has_chunk_id:
            errors.append("部分切片缺少 chunk_id")

    if not md_content:
        errors.append("MD 内容为空")

    if errors:
        logger.error("断言失败:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info(" 全流程测试全部通过!")
    logger.info("=" * 70)
    logger.info(f"  PDF: {test_pdf_name}")
    logger.info(f"  切片数: {len(chunks)}")
    logger.info(f"  MD 长度: {len(md_content)} 字符")
    logger.info(f"  节点数: {len(node_execution_order)}")
    logger.info("=" * 70)
