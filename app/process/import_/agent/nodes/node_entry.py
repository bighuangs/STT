from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state


if __name__ == '__main__':
    import sys
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state

    logger.info("===== 开始 node_entry 节点单元测试 =====")
    logger.info("")

    # 测试用例表：local_file_path → (期望类型, 期望路由)
    test_cases = [
        ("华为平板 C5 平板用户指南-(BZT3-W59&W69,EMUI10.1_01,ZH-CN)",    "UNSUPPORTED"),
        ("华为擎云 P5 激光多功能一体机 CV81Z-LDM 用户指南-(CV81Z-LDM,06,zh-cn).md",      "MD"),
        ("华为显示器 B3-242H 用户指南-(SSN-24BZ,VGA,04,zh-cn).pdf",     "PDF"),
    ]

    for i, (file_path, expected_type) in enumerate(test_cases, 1):
        from pathlib import Path  # 内部导入，避免干扰模块级 linter
        state = create_default_state(
            task_id=f"test_task_{i:03d}",
            local_file_path=file_path,
        )
        result = node_entry(state)

        # 断言：通用策略（所有类型均应填充）
        assert "parser_config" in result and result["parser_config"], \
            f"[{file_path}] parser_config 未填充"
        assert "child_delimiter" in result, \
            f"[{file_path}] child_delimiter 未填充"
        assert result.get("title_tks") and result.get("title_sm_tks"), \
            f"[{file_path}] 文件名分词结果为空"
        assert result.get("file_title") == Path(file_path).stem, \
            f"[{file_path}] file_title 不正确，期望 {Path(file_path).stem}，实际 {result.get('file_title')}"

        # 断言：按类型验证路由开关
        if expected_type == "MD":
            assert result["is_md_read_enabled"],      f"[{file_path}] MD 未开启"
            assert not result["is_pdf_read_enabled"], f"[{file_path}] PDF 不应开启"
            assert result["md_path"] == file_path,     f"[{file_path}] md_path 不正确"
        elif expected_type == "PDF":
            assert not result["is_md_read_enabled"],  f"[{file_path}] MD 不应开启"
            assert result["is_pdf_read_enabled"],      f"[{file_path}] PDF 未开启"
            assert result["pdf_path"] == file_path,    f"[{file_path}] pdf_path 不正确"
        elif expected_type == "UNSUPPORTED":
            assert not result["is_md_read_enabled"],  f"[{file_path}] MD 不应开启"
            assert not result["is_pdf_read_enabled"], f"[{file_path}] PDF 不应开启"

        logger.info(f"  [{i}/{len(test_cases)}] {file_path} → {expected_type} ✓")
        logger.debug(f"  详细结果：file_title={result['file_title']}, "
                     f"title_tks={result['title_tks'][:30]}..., "
                     f"child_delimiter={repr(result['child_delimiter'])}")

    # 测试用例外：空路径应抛出 ValueError
    logger.info("")
    try:
        bad_state = create_default_state(task_id="test_bad", local_file_path="")
        node_entry(bad_state)
        logger.error("  空路径测试：未抛出异常 ✗")
        sys.exit(1)
    except ValueError as e:
        logger.info(f"  [额外] 空路径测试 → ValueError ✓")

    # 测试用例外：自定义分隔符解析
    logger.info("")
    state_custom = create_default_state(
        task_id="test_custom_delimiter",
        local_file_path="测试.pdf",
        parser_config={"children_delimiter": r"\n||`---`"},
    )
    result_custom = node_entry(state_custom)
    assert result_custom["child_delimiter"], "自定义分隔符解析结果为空"
    logger.info(f"  [额外] 自定义分隔符解析 → {repr(result_custom['child_delimiter'])} ✓")

    logger.info("")
    logger.info("===== node_entry 节点单元测试全部通过 =====")
    sys.exit(0)