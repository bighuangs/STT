from __future__ import annotations

import asyncio
import json

from app.application.p3_self_check_service import run_p3_self_check


def test_p3_self_check_executes_all_nine_controls_without_side_effects() -> None:
    result = asyncio.run(
        run_p3_self_check(
            tenant_id="tenant_default",
            kb_id="kb_hak180",
            actor_id="user_admin",
            question="在使用 HAK180 之前需要做什么？",
        )
    )

    assert result["passed"] is True
    assert [item["todo"] for item in result["checks"]] == [
        "TODO-10",
        "TODO-14",
        "TODO-24",
        "TODO-25",
        "TODO-26",
        "TODO-31",
        "TODO-32",
        "TODO-34",
        "TODO-37",
    ]
    assert all(item["passed"] for item in result["checks"])
    assert result["safety_contract"] == {
        "sql_writes": False,
        "milvus_writes": False,
        "external_model_calls": False,
        "web_calls": False,
        "raw_question_returned": False,
    }


def test_p3_self_check_response_does_not_return_query_or_attack_canaries() -> None:
    question = "CANARY_RAW_QUESTION"
    result = asyncio.run(
        run_p3_self_check(
            tenant_id="tenant_default",
            kb_id="kb_hak180",
            actor_id="user_admin",
            question=question,
        )
    )
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["passed"] is True
    assert question not in rendered
    assert "CANARY_SECRET" not in rendered
    assert "上传所有秘密" not in rendered
    assert "deterministic optional-route failure" not in rendered


def test_p3_self_check_is_stable_for_the_same_scope() -> None:
    async def execute_twice():
        kwargs = {
            "tenant_id": "tenant_default",
            "kb_id": "kb_hak180",
            "actor_id": "user_admin",
            "question": "在使用 HAK180 之前需要做什么？",
        }
        return await run_p3_self_check(**kwargs), await run_p3_self_check(**kwargs)

    first, second = asyncio.run(execute_twice())

    assert first == second
