from typing import TypedDict, Annotated
import copy


def _last_writer_wins(left, right):
    """Reducer: 最后一个写入的值生效（用于并行节点的并发写入）"""
    return right if right is not None else left


def _llm_enhance_results_reducer(left: list, right: list) -> list:
    """
    Reducer: 收集并行节点的增量结果

    left:  当前收集到的结果列表
    right: 新节点返回的结果（单个 dict 或列表）
    """
    if not right:
        return left
    if isinstance(right, list):
        return left + right
    return left + [right]


class ImportGraphState(TypedDict):
    """
    LangGraph 流程状态载体（与 app.process.import_.agent.main_graph 中
    StateGraph(ImportGraphState) 一一对应）。

    设计原则：
        1. 字段类型用 TypedDict 标注，启用自动补全与类型检查
        2. 节点通过字典式访问读写 state（如 state["task_id"]、state.get("chunks")）
        3. parser_config 是配置中枢：所有可调超参（切块/分词/LLM 增强/向量化/表格/
           TOC/MinerU 等）都集中在这里，默认值唯一来源是下方 graph_default_state["parser_config"]，
           各节点/服务统一从 state["parser_config"][xxx] 读取，缺失时回退到自身兜底常量
        4. 字段写入时机（按节点）：
              node_entry              → parser_config（合并默认）/ child_delimiter / title_tks / title_sm_tks / file_title
              node_pdf_to_md          → md_path / md_content
              node_metadata_extract   → tables / image_refs / chunks（初版）/ doc_metadata
              node_md_img             → image_pil_dict / image_summaries / chunks（VLM 增强追加到 content）
              node_link_and_table_chunk → chunks（追加 hyperlinks/table 类型）
              node_chunk_merge        → chunks（长度标准化合并）
              node_image_upload       → chunks（追加 img_id/img_ids/create_time/create_timestamp_flt）/ img_id_dict / md_content
              node_keyword_extraction → chunks（追加 key_words / key_tokens）
              node_question_proposal  → chunks（追加 question_kwd / question_tks）
              node_metadata_generate  → chunks（追加 metadata）
              node_content_tagging    → chunks（追加 tags）
              node_toc                → toc（独立字段，不修改 chunks）
              node_bge_embedding      → chunks（追加 dense_vector / sparse_vector，hyperlinks 跳过）
              node_import_milvus      → 消费 chunks + toc 写入 Milvus
    """

    # 任务标志id（使用 _last_writer_wins reducer，支持并行节点并发写入）
    task_id: Annotated[str, _last_writer_wins]
    # TODO-01 稳定领域标识：task_id 仅为旧进度投影，不能再充当 doc_id。
    job_id: Annotated[str, _last_writer_wins]
    document_id: Annotated[str, _last_writer_wins]
    document_version_id: Annotated[str, _last_writer_wins]
    index_version_id: Annotated[str, _last_writer_wins]
    collection_name: Annotated[str, _last_writer_wins]
    vector_dim: Annotated[int, _last_writer_wins]
    schema_version: Annotated[str, _last_writer_wins]
    embedding_model_id: Annotated[str, _last_writer_wins]
    tenant_id: Annotated[str, _last_writer_wins]
    # TODO-37：由 SQL placement + Milvus 物理门禁解析，不能由上传请求指定。
    vector_placement_id: Annotated[str | None, _last_writer_wins]
    vector_placement_epoch: Annotated[int | None, _last_writer_wins]
    vector_compatibility_fingerprint: Annotated[str, _last_writer_wins]
    vector_legacy_policy: Annotated[str, _last_writer_wins]
    # TODO-02：Worker 从最新安全检查点恢复时，图从该节点继续。
    resume_node: Annotated[str, _last_writer_wins]

    # 路由分发判断标记
    is_md_read_enabled: Annotated[bool, _last_writer_wins]
    is_pdf_read_enabled: Annotated[bool, _last_writer_wins]
    # TODO-07：所有已注册格式统一走历史节点名 node_pdf_to_md（兼容旧检查点）。
    is_document_read_enabled: Annotated[bool, _last_writer_wins]
    input_format: Annotated[str, _last_writer_wins]
    input_mime_type: Annotated[str, _last_writer_wins]
    parser_id: Annotated[str, _last_writer_wins]
    parser_version: Annotated[str, _last_writer_wins]
    chunking_template_id: Annotated[str, _last_writer_wins]
    chunking_template_version: Annotated[str, _last_writer_wins]
    suggested_chunking_template_id: Annotated[str, _last_writer_wins]

    # 路径相关
    local_file_path: Annotated[str, _last_writer_wins]  # 原始pdf/md文件存储位置
    pdf_path: Annotated[str, _last_writer_wins]  # 存储解析为pdf的文件
    md_path: Annotated[str, _last_writer_wins]  # 存储解析为md的文件
    local_dir: Annotated[str, _last_writer_wins]  # pdf转md的目标路径
    file_title: Annotated[str, _last_writer_wins]  # chunk所属的文档标题
    kb_id: Annotated[str, _last_writer_wins]  # 知识库 ID，Milvus 入库时映射到 kb_id 字段

    # 内容数据
    md_content: Annotated[str, _last_writer_wins]
    # ParsedDocument 本体持久化为独立制品，checkpoint 只携带以下引用与摘要。
    parsed_artifact_uri: Annotated[str, _last_writer_wins]
    parsed_artifact_object_key: Annotated[str, _last_writer_wins]
    parsed_artifact_sha256: Annotated[str, _last_writer_wins]
    parsed_artifact_cache_key: Annotated[str, _last_writer_wins]
    parsed_artifact_cache_hit: Annotated[bool, _last_writer_wins]
    parsed_document_summary: Annotated[dict, _last_writer_wins]
    parse_warnings: Annotated[list, _last_writer_wins]
    structured_chunks_ready: Annotated[bool, _last_writer_wins]
    tables: Annotated[list, _last_writer_wins]
    hyperlinks: Annotated[list, _last_writer_wins]
    tbls: Annotated[list, _last_writer_wins]
    image_refs: Annotated[list, _last_writer_wins]
    chunks: Annotated[list, _last_writer_wins]  # 文档切片列表，每项 dict 字段动态扩展：
                   # 通用：content, images, file_title, type
                   # node_metadata_extract 阶段（粗切分）：
                   #   - content, images, file_title
                   # node_md_img 阶段（VLM 增强）：
                   #   - content 末尾追加 VLM 生成的图片描述
                   # node_link_and_table_chunk 阶段：
                   #   - 追加 hyperlinks/table 类型 chunk（含内嵌图片 URL 列表）
                   # node_chunk_merge 阶段（长度标准化合并）：
                   #   - 合并后 special chunk（type=hyperlinks/table）原样保留
                   # node_image_upload 阶段（MinIO 绑定）：
                   #   - 追加 img_id（主图）/ img_ids（全部）/ create_time / create_timestamp_flt
                   #   - 删除原始 images 字段
                   # 4 个并行 LLM 增强节点（共享同一 chunks 引用）：
                   #   - node_keyword_extraction  → key_words: list[str]（关键词）+ key_tokens: list[str]（jieba 切分）
                   #   - node_question_proposal   → question_kwd: list[str]（LLM 提问）+ question_tks: list[str]（jieba 切分）
                   #   - node_metadata_generate   → metadata: dict（字段由 parser_config.metadata 配置）
                   #   - node_content_tagging     → tags: list[str]（来自 parser_config["tags"]，topn 由 topn_tags 控制）
                   # node_bge_embedding 阶段（向量化）：
                   #   - text/table 类型追加 2 字段：
                   #     * dense_vector（BGE 稠密，title+content 融合，1024 维）
                   #     * sparse_vector（BGE 稀疏，融合语义的稀疏部分）
                   #   - hyperlinks 类型无向量（content 是 URL 列表）

    # 图片处理
    image_pil_dict: Annotated[dict, _last_writer_wins]  # {图片名: PIL.Image}，由 enrich_markdown_images 扫描加载，供 MinIO 上传
    image_summaries: Annotated[dict, _last_writer_wins]  # VLM 生成的图片描述，{图片名: 描述}
    # VLM 是导入角色的可选依赖；连接/超时等可恢复故障会保留原图与正文，
    # 并在此记录可持久化降级事实，供检查点和任务事件观测。
    vlm_warnings: Annotated[list, _last_writer_wins]
    image_context_list: Annotated[list, _last_writer_wins]  # 图片扫描结果 [(图片名, 图片路径), ...]，不含上下文（与 naive.py 一致）
    img_id_dict: Annotated[dict, _last_writer_wins]  # MinIO 上传后生成的 img_id 映射，{图片名: img_id}，对齐 ragflow chunk["img_id"]
    uploaded_object_keys: Annotated[list, _last_writer_wins]  # cleanup 使用的真实 MinIO object_name

    # 全局元数据（从 PDF/MD 文件提取）
    # 当前由 node_metadata_extract 兜底写入空 dict {}，完整字段（标题/作者/关键词/
    # 摘要/页数/创建时间等）的提取逻辑后续在 metadata_extract_service 中补充
    doc_metadata: Annotated[dict, _last_writer_wins]  # 全局元数据字典

    # 文档目录（TOC，由 node_toc 生成）
    # 与 chunks 解耦：仅记录整篇文档的目录结构与最终块元数据，node_import_milvus 据此入库
    # 字段约定（参考 ragflow task_executor.py:build_TOC L1009-1113）：
    #   - doc_id: 文档 ID（用于 hash 生成 TOC 块 ID）
    #   - items: toc 条目列表，每项含 level/title/ids(关联块 ID 列表)
    #   - content_with_weight: items 序列化的 JSON 字符串（与 ragflow 一致）
    #   - available_int: 0=不可被用户检索，仅内部使用（前端展示导航）
    #   - page_num_int: 极大值（默认 1e8，可由 parser_config["toc_page_num_int"] 覆盖），
    #                   使 TOC 排在所有内容块之后
    #   - toc_kwd: 固定为 "toc"，用于前端/下游识别
    #   - id: xxhash64(content_with_weight + doc_id) 生成的 16 字符 hex
    toc: Annotated[dict, _last_writer_wins]
    # TODO-03：Milvus flush 后生成的完整性证明，由 Worker 持久化到控制库。
    index_manifest: Annotated[dict, _last_writer_wins]

    # 文件名分词结果
    title_tks: Annotated[str, _last_writer_wins]  # 文件名粗粒度分词结果
    title_sm_tks: Annotated[str, _last_writer_wins]  # 文件名细粒度分词结果

    # 分块解析器配置（27 字段，详细见 graph_default_state["parser_config"]）
    # 涵盖：分块/分隔符/超链接分析/4 个 LLM 增强开关与阈值/向量化融合权重/分批参数/
    #      VLM 超参/TOC 超参/MinerU 超时 等；与 ragflow parser_config 字段名 1:1 对齐，
    #      缺失/无效时各 service 退化为自身兜底常量（行为向后兼容）
    parser_config: Annotated[dict, _last_writer_wins]
    # 合并后的子分隔符（系统默认 + parser_config["children_delimiter"] 解析出的自定义分隔符），
    # 由 node_entry 在 _parse_child_delimiter 阶段写入，供后续分块/切分节点使用
    child_delimiter: Annotated[str, _last_writer_wins]

    # 并行 LLM 增强节点的增量结果收集（用于合并 chunks 增量字段）
    # 每个并行节点返回 {"chunks": [...], "toc": {...}} 格式的增量更新
    # 使用 list reducer 收集所有节点结果，再由 reducer 节点合并
    llm_enhance_results: Annotated[list, _llm_enhance_results_reducer]


# parser_config 默认值（STT 全流程所有可调超参的唯一来源）
# 设计原则：
#   1. 各节点/service 一律从 state["parser_config"][xxx] 读取，未在节点内部硬编码
#   2. service 内部仍保留"兜底常量"作为缺失/无效值的回退（向后兼容）
#   3. 与 ragflow parser_config 字段名 1:1 对齐，便于后续 ragflow 字段扩展时直接迁移
# 字段按 7 大类分组：
#   A. 分块/分隔符   ：chunk_char_num, delimiter, analyze_hyperlink, overlapped_percent,
#                       short_header_max_tokens, table_batch_size, table_row_delimiter,
#                       separate_tables, min_token_len_for_subtokenize
#   B. LLM 增强开关  ：auto_keywords, auto_questions, enable_metadata, metadata,
#                       built_in_metadata, tag_kb_ids, tags, topn_tags, toc_extraction
#   C. LLM 公共      ：llm_content_max_chars, tag_min_score
#   D. 向量化        ：filename_embd_weight, embedding_max_chars
#   E. VLM 图像增强  ：vlm_max_workers, vlm_min_image_side
#   F. TOC           ：toc_title_max_len_cn, toc_title_max_len_en, toc_batch_max_chars,
#                       toc_page_num_int
#   G. PDF 解析（MinerU）：mineru_poll_timeout, mineru_poll_interval, mineru_download_timeout
# 注：parser_config["children_delimiter"]（自定义子分隔符原始字符串）由 entry_service._parse_child_delimiter
# 单独读取，不在本默认值 dict 中——其解析结果写入 state["child_delimiter"]。
graph_default_state: ImportGraphState = {
    "task_id": "",
    "job_id": "",
    "document_id": "",
    "document_version_id": "",
    "index_version_id": "",
    "collection_name": "",
    "vector_dim": 1024,
    "schema_version": "chunks_v1",
    "embedding_model_id": "bge-m3",
    "tenant_id": "tenant_default",
    "vector_placement_id": None,
    "vector_placement_epoch": None,
    "vector_compatibility_fingerprint": "",
    "vector_legacy_policy": "",
    "resume_node": "node_entry",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_document_read_enabled": False,
    "input_format": "",
    "input_mime_type": "application/octet-stream",
    "parser_id": "",
    "parser_version": "",
    "chunking_template_id": "general",
    "chunking_template_version": "",
    "suggested_chunking_template_id": "general",
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "kb_id": "",
    "md_content": "",
    "parsed_artifact_uri": "",
    "parsed_artifact_object_key": "",
    "parsed_artifact_sha256": "",
    "parsed_artifact_cache_key": "",
    "parsed_artifact_cache_hit": False,
    "parsed_document_summary": {},
    "parse_warnings": [],
    "structured_chunks_ready": False,
    "tables": [],
    "hyperlinks": [],
    "tbls": [],
    "image_refs": [],
    "chunks": [],
    "image_pil_dict": {},
    "image_summaries": {},
    "vlm_warnings": [],
    "image_context_list": [],
    "img_id_dict": {},
    "uploaded_object_keys": [],
    "doc_metadata": {},
    "toc": {},
    "index_manifest": {},
    "title_tks": "",
    "title_sm_tks": "",
    "parser_config": {
        # ============ A. 分块/分隔符 ============
        # metadata_extract_service + chunk_merge_service 共享
        "chunk_char_num": 512,         # 单块目标大小；chunk_merge_service 用 tiktoken 累计 token 数
                                       # （字符名仅为历史延续，下游仍按 1 字符≈1 token 估算）
        "delimiter": "\n!?。；！？",     # 细切分子分隔符（metadata_extract 阶段用）
        "analyze_hyperlink": True,     # 是否启用超链接提取（当前实现总是启用，预留开关）
        "overlapped_percent": 0,       # chunk_merge_service 块间重叠百分比（0~100）
        "short_header_max_tokens": 100,  # chunk_merge_service 短标题判定阈值（触发"立即 flush + 锁下一段"）
        "table_batch_size": 10,        # link_table_chunk_service 表格分批行数
        "table_row_delimiter": "； ",  # link_table_chunk_service 表格行间分隔符
        "separate_tables": True,       # metadata_extract_service 表格分离开关
        "min_token_len_for_subtokenize": 2,  # entry_service 细粒度分词触发阈值

        # ============ B. LLM 增强开关（4 个并行 LLM 节点）============
        # node_keyword_extraction 启用开关：
        #   - 0 / False / 缺失：跳过关键词提取
        #   - >0：每个 chunk 提取的关键词数量（与 ragflow parser_config.auto_keywords 对齐）
        "auto_keywords": 5,
        # node_question_proposal 启用开关：
        #   - 0 / False / 缺失：跳过问题生成
        #   - >0：每个 chunk 生成的问题数量（与 ragflow parser_config.auto_questions 对齐）
        "auto_questions": 3,
        # node_metadata_generate 启用开关（与 ragflow parser_config.enable_metadata 对齐）：
        #   - True / >0：启用元数据生成
        #   - False / 0 / 缺失：不生成
        "enable_metadata": False,
        # node_metadata_generate 自定义元数据字段列表（与 ragflow parser_config.metadata 对齐）：
        #   - list[dict]：每项含 key（字段名）、description（描述）、enum（可选枚举列表）
        #   - dict：直接是 JSON Schema（type=object，properties=dict）
        #   - 空 / 缺失：不生成
        "metadata": [],
        # node_metadata_generate 内置元数据字段列表（与 ragflow parser_config.built_in_metadata 对齐）：
        #   - list[dict]：每项含 key（字段名）、description（描述）
        #   - 与 metadata 合并后统一传给 LLM（metadata 优先级更高）
        "built_in_metadata": [],
        # node_content_tagging 参考知识库 ID 列表（与 ragflow parser_config.tag_kb_ids 对齐）：
        #   - list[str]：指定用于打标的参考知识库，从已有标签集匹配
        #   - STT 暂不支持从此拉取标签，仅作兼容保留
        "tag_kb_ids": [],
        # node_content_tagging 标签列表（与 ragflow parser_config.tags 对齐）：
        #   - list[str]：LLM 为每个 chunk 选取最相关的标签
        #   - 为空时跳过标签生成
        "tags": [],
        # node_content_tagging 每 chunk 最多打几个标签（与 ragflow parser_config.topn_tags 对齐）：
        #   - int：默认 3
        "topn_tags": 3,
        # node_toc 启用开关（与 ragflow parser_config.toc_extraction 对齐）：
        #   - True / >0：启用目录提取
        #   - False / 0 / 缺失：不生成
        "toc_extraction": False,

        # ============ C. LLM 公共超参（4 个 LLM service 共享）============
        # 内容截断上限（避免 LLM 上下文溢出），适用于：
        #   keyword_extraction_service / question_proposal_service /
        #   metadata_generate_service / content_tagging_service
        # 默认 4000（与改造前 4 处硬编码一致）
        "llm_content_max_chars": 4000,
        # content_tagging_service 标签评分下限（与 ragflow 1-10 分制对齐）：
        #   - int：LLM 打的标签分数 < 该值被过滤
        "tag_min_score": 1,

        # ============ D. 向量化（node_bge_embedding）============
        # 标题向量融合权重（与 ragflow parser_config.filename_embd_weight 对齐）：
        #   - float：取值 0~1
        #     - 0：标题向量不参与融合（退化为仅内容向量）
        #     - 1：仅使用标题向量（忽略内容）
        #     - 默认 0.1：标题方向占 10%，与 ragflow 默认一致
        #   - 字段缺失/无效/<=0 时，embedding_service 退化为 0.1
        "filename_embd_weight": 0.1,
        # 单条文本最大字符数（BGE-M3 8192 token 安全裕度）：
        #   - int：默认 8192
        #   - 超过则按字符截断（STT 用字符而非 token，与 embedding_service 现状一致）
        "embedding_max_chars": 8192,

        # ============ E. VLM 图像增强（node_md_img）============
        # VLM 并发线程数
        #   - int：图片 VLM 描述的最大并发数
        "vlm_max_workers": 10,
        # 跳过过小图片的最小边长（像素）
        #   - int：图片宽或高 < 该值时跳过 VLM 调用
        "vlm_min_image_side": 11,

        # ============ F. TOC（node_toc）============
        # 中文标题最大字符数（超过则视为无效 TOC 标题）
        "toc_title_max_len_cn": 25,
        # 英文标题最大字符数
        "toc_title_max_len_en": 80,
        # TOC 阶段 1（标题抽取）每批字符预算。
        # 这是决定 TOC 速度的关键：阶段 1 会把全文按此预算切成多批并【并发】调用 LLM
        # （asyncio.gather）。预算越小→批次越多→并发度越高→墙钟越短（但 API 调用次数/成本上升）。
        # 对齐 ragflow run_toc_from_text 的设计（其把每批预算上限压到 1024 token 以换取高并发，
        # 见 generator.py:1198）；STT 用字符预算近似，默认 3000（约等价于 1024 token 的中英文混合量级），
        # 使中等文档也能切出多批并发，避免退化成一次串行大调用。
        "toc_batch_max_chars": 3000,
        # TOC 块 page_num 极大值（确保 TOC 排在所有内容块之后）
        "toc_page_num_int": 100000000,

        # ============ G. PDF 解析（MinerU，node_pdf_to_md）============
        # 轮询超时（秒）
        "mineru_poll_timeout": 600,
        # 轮询间隔（秒）
        "mineru_poll_interval": 3,
        # 下载超时（秒，PDF 地址获取与 ZIP 下载两处共用）
        "mineru_download_timeout": 30,

        # ============ H. TODO-07 统一解析与版本化切块模板 ============
        # parser_id=auto 由 magic/MIME/OOXML signature 决定；扩展名只作辅助证据。
        "parser_id": "auto",
        # 未显式选择时稳定使用 general；suggestion 只展示，不隐式改变版本事实。
        "chunking_template": "general",
        "chunking_template_version": "",
        "chunk_token_num": 512,
        "chunk_overlap_percent": 0,
        "inherit_heading": True,
        "table_rows_per_chunk": 20,
        "max_table_rows_per_block": 200,
        "parser_timeout_seconds": 600,
        "zip_max_entries": 10000,
        "zip_max_uncompressed_bytes": 512 * 1024 * 1024,
        "zip_max_compression_ratio": 200,
        # 正常 Worker 会镜像到 MinIO；离线测试/预览可显式关闭。
        "persist_parsed_artifact_to_minio": True,
        "require_parsed_artifact_minio": False,
    },
    "child_delimiter": "",
    "llm_enhance_results": [],  # 收集并行节点的增量结果
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖。

    overrides 传入任何字段均可（包括 parser_config 子键），最终与 graph_default_state 做 update 合并。
    parser_config 的合并策略详见 _parse_parser_config（entry_service 中）：
    用户传入的 parser_config 子键优先，默认值兜底。

    例子：
        state = create_default_state(
            task_id="abc123",
            local_file_path="/data/doc.pdf",
            is_pdf_read_enabled=True,
            parser_config={"chunk_char_num": 1024, "auto_keywords": 10},
        )
        等价于 overrides = {"task_id": "abc123", "local_file_path": "...", ...}
        → 最终 state["parser_config"]["chunk_char_num"] = 1024（用户覆盖）
        → 其他键（如 filename_embd_weight=0.1）保持 graph_default_state 默认
    """
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(graph_default_state)
