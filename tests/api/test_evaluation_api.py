"""TODO-18 HTTP contracts, scope selection and authorization boundaries."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.http import evaluation_routes
from app.api.http.evaluation_routes import register_evaluation_routes
from app.application import evaluation_service
from app.infra.persistence.sql.repositories import ResourceConflictError
from app.security import http as security_http
from app.security.http import action_for, install_security
from app.security.permissions import is_allowed
from app.security.principal import Principal


API_PATHS = {
    ("GET", "/eval/html"),
    ("GET", "/api/v1/evaluation/datasets"),
    ("POST", "/api/v1/evaluation/datasets"),
    ("GET", "/api/v1/evaluation/datasets/{dataset_id}"),
    ("POST", "/api/v1/evaluation/preflight"),
    ("GET", "/api/v1/evaluation/runs"),
    ("POST", "/api/v1/evaluation/runs"),
    ("GET", "/api/v1/evaluation/runs/{run_id}"),
    ("GET", "/api/v1/evaluation/runs/{run_id}/results"),
    ("GET", "/api/v1/evaluation/runs/{run_id}/comparison"),
    ("GET", "/api/v1/evaluation/runs/{run_id}/artifacts/{artifact}"),
    ("POST", "/api/v1/feedback"),
    ("GET", "/api/v1/feedback"),
    ("POST", "/api/v1/feedback/{feedback_id}:triage"),
    ("GET", "/api/v1/annotation-tasks"),
    ("POST", "/api/v1/annotation-tasks"),
    ("POST", "/api/v1/annotation-tasks/{task_id}:review"),
    ("POST", "/api/v1/annotation-tasks/{task_id}:arbitrate"),
}


def _principal(role: str = "KBAdmin") -> Principal:
    return Principal(
        user_id="user-api",
        tenant_id="tenant-api",
        roles=(role,),
        scopes=(),
        auth_type="test",
        credential_id="credential-api",
        request_id="",
    )


def _app(monkeypatch: pytest.MonkeyPatch, principal: Principal | None = None) -> FastAPI:
    app = FastAPI()
    register_evaluation_routes(app)
    if principal is not None:
        monkeypatch.setattr(
            security_http,
            "authenticate_access",
            lambda _token, request_id="": Principal(
                **{**principal.__dict__, "request_id": request_id}
            ),
        )
        monkeypatch.setattr(security_http, "audit", lambda *args, **kwargs: None)
        install_security(app)
    return app


def test_all_18_evaluation_feedback_routes_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch)
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path == "/eval/html"
        or route.path.startswith("/api/v1/evaluation")
        or route.path.startswith("/api/v1/feedback")
        or route.path.startswith("/api/v1/annotation-tasks")
    }
    assert actual == API_PATHS
    assert len(actual) == 18


def test_eval_api_requires_authentication_and_uses_rbac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation_routes, "list_datasets", lambda: [])
    app = _app(monkeypatch, _principal("Viewer"))

    with TestClient(app) as client:
        assert client.get("/api/v1/evaluation/datasets").status_code == 401
        response = client.get(
            "/api/v1/evaluation/datasets",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert response.json()["data"] == []
    assert action_for("GET", "/api/v1/evaluation/runs/r/artifacts/json") == "eval:read"
    assert action_for("POST", "/api/v1/evaluation/runs") == "eval:run"
    assert is_allowed(("Viewer",), (), "eval:read") is True
    assert is_allowed(("Viewer",), (), "eval:run") is False


def test_create_run_copies_principal_and_forwards_retrieval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _principal("KBAdmin")
    captured: dict = {}

    def fake_create_run(**kwargs):
        captured["create"] = kwargs
        return {"id": "erun-api", "status": "queued", **kwargs}

    def fake_execute(run_id: str, principal: Principal):
        captured["execute"] = (run_id, principal)

    monkeypatch.setattr(evaluation_routes, "create_run", fake_create_run)
    monkeypatch.setattr(evaluation_routes, "execute_run", fake_execute)
    app = _app(monkeypatch, source)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/runs",
            headers={"Authorization": "Bearer test-token"},
            json={
                "dataset_id": "eds-api",
                "kb_id": "kb-api",
                "mode": "retrieval_only",
                "name": "retrieval smoke",
            },
        )

    assert response.status_code == 202
    assert captured["create"]["mode"] == "retrieval_only"
    run_id, copied = captured["execute"]
    assert run_id == "erun-api"
    assert copied is not source
    assert copied.user_id == source.user_id
    assert copied.tenant_id == source.tenant_id
    assert copied.roles == source.roles
    assert copied.request_id


def test_create_run_forwards_the_requested_paired_case_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_create_run(**kwargs):
        captured.update(kwargs)
        return {"id": "erun-subset", "status": "queued"}

    monkeypatch.setattr(evaluation_routes, "create_run", fake_create_run)
    monkeypatch.setattr(evaluation_routes, "execute_run", lambda *_args: None)
    app = _app(monkeypatch, _principal("KBAdmin"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/runs",
            headers={"Authorization": "Bearer test-token"},
            json={
                "dataset_id": "eds-api",
                "kb_id": "kb-api",
                "mode": "retrieval_only",
                "baseline_run_id": "erun-baseline",
                "case_ids": ["case-2", "case-1"],
            },
        )

    assert response.status_code == 202
    assert captured["baseline_run_id"] == "erun-baseline"
    assert captured["case_ids"] == ["case-2", "case-1"]


def test_preflight_forwards_full_mode_to_server_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_preflight(dataset_id: str, kb_id: str, *, mode: str) -> dict:
        captured.update({"dataset_id": dataset_id, "kb_id": kb_id, "mode": mode})
        return {"ready": True, "evaluation_mode": mode, "checks": {}}

    monkeypatch.setattr(evaluation_routes, "evaluation_preflight", fake_preflight)
    app = _app(monkeypatch, _principal("KBAdmin"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/preflight",
            headers={"Authorization": "Bearer test-token"},
            json={"dataset_id": "eds-api", "kb_id": "kb-api", "mode": "live"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["evaluation_mode"] == "live"
    assert captured == {
        "dataset_id": "eds-api",
        "kb_id": "kb-api",
        "mode": "live",
    }


def test_preflight_and_run_require_complete_profile_pin_and_forward_exact_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_preflight(dataset_id: str, kb_id: str, **kwargs) -> dict:
        captured["preflight"] = (dataset_id, kb_id, kwargs)
        return {"ready": True, "checks": {}}

    def fake_create_run(**kwargs) -> dict:
        captured["run"] = kwargs
        return {"id": "erun-profile-pin", "status": "queued"}

    monkeypatch.setattr(evaluation_routes, "evaluation_preflight", fake_preflight)
    monkeypatch.setattr(evaluation_routes, "create_run", fake_create_run)
    monkeypatch.setattr(evaluation_routes, "execute_run", lambda *_args: None)
    app = _app(monkeypatch, _principal("KBAdmin"))
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        incomplete = client.post(
            "/api/v1/evaluation/preflight",
            headers=headers,
            json={
                "dataset_id": "eds-api",
                "kb_id": "kb-api",
                "retrieval_profile_id": "rpf-candidate",
            },
        )
        preflight = client.post(
            "/api/v1/evaluation/preflight",
            headers=headers,
            json={
                "dataset_id": "eds-api",
                "kb_id": "kb-api",
                "retrieval_profile_id": "rpf-candidate",
                "retrieval_profile_version": 7,
            },
        )
        run = client.post(
            "/api/v1/evaluation/runs",
            headers=headers,
            json={
                "dataset_id": "eds-api",
                "kb_id": "kb-api",
                "mode": "retrieval_only",
                "retrieval_profile_id": "rpf-candidate",
                "retrieval_profile_version": 7,
            },
        )

    assert incomplete.status_code == 422
    assert preflight.status_code == 200
    assert run.status_code == 202
    assert captured["preflight"] == (
        "eds-api",
        "kb-api",
        {
            "mode": "retrieval_only",
            "retrieval_profile_id": "rpf-candidate",
            "retrieval_profile_version": 7,
        },
    )
    assert captured["run"]["retrieval_profile_id"] == "rpf-candidate"
    assert captured["run"]["retrieval_profile_version"] == 7


def test_dataset_import_rejects_invalid_gate_policy_at_api_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_routes,
        "import_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid gate policy must be rejected before application service"
        ),
    )
    app = _app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/datasets",
            json={
                "manifest": {
                    "gate_policy": {"mrr_max_drop": -0.01},
                }
            },
        )

    assert response.status_code == 422
    assert "mrr_max_drop" in response.text


def test_dataset_import_rejects_case_policy_that_can_pass_unsupported_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_routes,
        "import_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "unsafe case pass policy must be rejected before application service"
        ),
    )
    app = _app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/datasets",
            json={
                "manifest": {
                    "gate_policy": {
                        "case_pass_policy": {
                            "citation_coverage_minimum": 0.0,
                        }
                    }
                }
            },
        )

    assert response.status_code == 422


def test_results_api_preserves_explainable_case_outcome_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible_answer_policy = {
        "schema_version": "visible-answer-policy.v1",
        "mode": "safe_pre_degraded",
        "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        "strict_grounding_eligible": False,
    }
    completeness_self_review = {
        "schema_version": "completeness-self-review-audit.v1",
        "status": "committed",
        "calls_attempted": 1,
        "model_id": "model-a",
        "usage": {"status": "available", "total_tokens": 120},
        "trigger_reason_codes": ["UNCOVERED_QUESTION_CLAUSES"],
        "prompt_sha256": "archive-only",
    }
    outcome = {
        "policy_version": "stt.eval.case.v2",
        "passed": False,
        "pass_rate": 0.0,
        "strict_supported_case_pass": False,
        "strict_supported_case_pass_rate": 0.0,
        "quality_score": 0.7,
        "answer_correctness": 1.0,
        "final_answer_correctness": 1.0,
        "final_answer_correct": True,
        "final_answer_accuracy": 1.0,
        "faithfulness": 0.0,
        "citation_coverage": 0.0,
        "retrieval_support": 0.0,
        "reason_codes": [
            "faithfulness_below_minimum",
            "citation_coverage_below_minimum",
        ],
        "diagnostic_codes": ["retrieval_ground_truth_miss"],
        "visible_answer_policy": visible_answer_policy,
        "answer_display_audit": {
            "status": "degraded",
            "badge_code": "ANSWER_COMPLETENESS_PRESERVED_CITATION_DEGRADED",
            "label": "答案完整性优先，部分内容未通过引用校验",
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
            "strict_grounding_eligible": False,
        },
    }
    monkeypatch.setattr(
        evaluation_routes,
        "list_results",
        lambda run_id, only_failed=False: [
            {
                "id": "eres-contract",
                "run_id": run_id,
                "case_id": "generic-case",
                "passed": False,
                "observation": {
                    "visible_answer_policy": visible_answer_policy,
                    "completeness_self_review": completeness_self_review,
                },
                "metrics": {"metrics": {"case_outcome": outcome}},
            }
        ],
    )
    app = _app(monkeypatch, _principal("Viewer"))

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/evaluation/runs/erun-contract/results?only_failed=true",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    persisted = response.json()["data"][0]["metrics"]["metrics"]["case_outcome"]
    assert persisted == outcome
    assert (
        response.json()["data"][0]["observation"]["visible_answer_policy"]
        == visible_answer_policy
    )
    assert (
        response.json()["data"][0]["observation"]["completeness_self_review"]
        == completeness_self_review
    )


def test_live_run_creation_fails_before_persistence_when_egress_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "evaluation_preflight",
        lambda _dataset_id, _kb_id, *, mode: {
            "ready": False,
            "checks": {
                "query_scope_authorized": True,
                "external_llm_authorized": False,
            },
            "evaluation_mode": mode,
        },
    )

    with pytest.raises(ResourceConflictError, match="external_llm_authorized"):
        evaluation_service.create_run(
            dataset_id="eds-api",
            kb_id="kb-api",
            mode="live",
            name="must not persist",
            baseline_run_id=None,
            case_ids=None,
        )


def test_feedback_accepts_owner_scoped_reliable_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    principal = _principal("Viewer")
    captured: dict = {}

    @contextmanager
    def fake_session_scope():
        yield object()

    class FakeRuntimeRepository:
        def __init__(self, _session, *, tenant_id: str, user_id: str) -> None:
            assert tenant_id == principal.tenant_id
            assert user_id == principal.user_id

        def get_conversation(self, conversation_id: str):
            captured["conversation_id"] = conversation_id
            return SimpleNamespace(kb_id="kb-api")

    class FakeControlRepository:
        def __init__(self, _session) -> None:
            pass

        def get_knowledge_base_model(self, kb_id: str) -> object:
            assert kb_id == "kb-api"
            return object()

    class FakeEvaluationRepository:
        def __init__(self, _session) -> None:
            pass

        def create_feedback(self, payload: dict) -> dict:
            captured["payload"] = payload
            return {"id": "fb-api", **payload}

    monkeypatch.setattr(evaluation_service, "current_principal", lambda: principal)
    monkeypatch.setattr(evaluation_service, "session_scope", fake_session_scope)
    monkeypatch.setattr(
        evaluation_service, "QueryRuntimeRepository", FakeRuntimeRepository
    )
    monkeypatch.setattr(evaluation_service, "ControlRepository", FakeControlRepository)
    monkeypatch.setattr(
        evaluation_service, "EvaluationRepository", FakeEvaluationRepository
    )
    monkeypatch.setattr(
        evaluation_service,
        "assert_query_session_owner",
        lambda *_args, **_kwargs: pytest.fail(
            "reliable Conversation must not be resolved as a legacy QuerySession"
        ),
    )

    payload = {
        "kb_id": "kb-api",
        "answer_id": "ans-api",
        "session_id": "conv_0123456789abcdef0123456789abcdef",
        "rating": 1,
    }
    result = evaluation_service.create_feedback(payload)

    assert result["id"] == "fb-api"
    assert captured == {
        "conversation_id": payload["session_id"],
        "payload": payload,
    }


def test_artifact_name_is_an_allowlist_before_report_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        evaluation_routes,
        "report_path",
        lambda run_id, artifact: calls.append((run_id, artifact)),
    )
    app = _app(monkeypatch, _principal("Viewer"))

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/evaluation/runs/erun-api/artifacts/secrets",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 422
    assert calls == []


def test_mode_adapter_selection_never_uses_query_graph_for_retrieval_only() -> None:
    retrieval = evaluation_service._adapter_for_mode("retrieval_only")
    live = evaluation_service._adapter_for_mode("live")

    assert retrieval.__class__.__name__ == "RetrievalOnlyEvalAdapter"
    assert live.__class__.__name__ == "QueryGraphEvalAdapter"
    with pytest.raises(ResourceConflictError, match="unsupported"):
        evaluation_service._adapter_for_mode("anything_else")


def test_report_path_is_bound_to_run_directory_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_root = tmp_path / "output" / "evaluation"
    own = report_root / "run-a" / "report.json"
    other = report_root / "run-b" / "report.json"
    own.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    own.write_text('{"run":"a"}', encoding="utf-8")
    other.write_text('{"run":"b"}', encoding="utf-8")
    import hashlib

    monkeypatch.setattr(evaluation_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_service, "REPORT_ROOT", report_root)

    def stored(path: Path) -> dict:
        content = path.read_bytes()
        return {
            "report_paths": {
                "json": {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            }
        }

    own_record = stored(own)
    monkeypatch.setattr(evaluation_service, "get_run", lambda _run_id: own_record)
    assert evaluation_service.report_path("run-a", "json") == own

    monkeypatch.setattr(evaluation_service, "get_run", lambda _run_id: stored(other))
    with pytest.raises(ResourceConflictError, match="invalid evaluation artifact path"):
        evaluation_service.report_path("run-a", "json")

    monkeypatch.setattr(evaluation_service, "get_run", lambda _run_id: own_record)
    own.write_text("tampered", encoding="utf-8")
    with pytest.raises(ResourceConflictError, match="integrity"):
        evaluation_service.report_path("run-a", "json")


def test_artifact_hash_is_computed_from_the_bytes_stored_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 换行转换后，产物仍应通过自身的完整性校验。"""

    import hashlib

    report_root = tmp_path / "output" / "evaluation"
    monkeypatch.setattr(evaluation_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_service, "REPORT_ROOT", report_root)
    payload = {
        "run_id": "run-newline",
        "run": {"id": "run-newline"},
        "summary": {"lines": ["a", "b"]},
        "results": [],
        "comparison": {},
    }

    paths = evaluation_service._write_artifacts(payload)

    for record in paths.values():
        stored = tmp_path / record["path"]
        stored_bytes = stored.read_bytes()
        assert record["sha256"] == hashlib.sha256(stored_bytes).hexdigest()
        assert record["size_bytes"] == len(stored_bytes)


def test_evaluation_page_exposes_both_modes_and_posts_mode() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")

    assert 'id="runModeSelect"' in page
    assert 'value="live"' in page
    assert 'value="retrieval_only"' in page
    assert "mode: byId('runModeSelect').value" in page
    assert "生成、引用和拒答指标将显示“未采集”" in page
    assert "window.confirm" in page
    assert "liveRunConfirmationMessage(selectedDatasetForConfirmation())" in page
    assert "发送 HAK180 问题/上下文" not in page
    assert "已取消完整链路运行，未向服务端创建任务" in page


def test_evaluation_page_orders_difficulties_applies_manifest_defaults_and_limits_baselines() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"    var DIFFICULTY_META = .*?(?=\n    function byId)",
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    assert "retrieval_profile_resolved" in page
    assert "paired_case_set: '基线与候选配对题集'" in page
    assert "if (isRootBaselineRun(run))" in page
    assert "return isRootBaselineRun(run)" in page
    assert "applyDatasetDefaultMode(dataset)" in page
    assert "select.value = preferredDatasetId(orderedDatasets, previous);" in page
    assert "sessionStorage.getItem(STORAGE.dataset)" in page
    assert "localStorage.getItem(STORAGE.dataset)" not in page
    assert "[' + datasetDescriptor(dataset) + '] ' + dataset.name" in page
    assert "'suite=' + (datasetSuiteKey(dataset) || '—')" in page
    assert "'case_policy=' + casePolicy.label" in page

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript function contract")
    script = (
        match.group(0)
        + "\nconst finalSuite=QUALITY_TARGETS.suite_key;"
        + "\nconst easy={id:'easy',name:'简单集',case_count:51,"
        + "difficulty:'easy',suite_key:'suite-a',minimum_case_count:50,"
        + "evaluation_mode_default:'retrieval_only'};"
        + "\nconst medium={id:'medium',name:'中等集',case_count:49,manifest_json:{"
        + "difficulty:'medium',suite_key:'suite-a',minimum_case_count:50,"
        + "evaluation_mode_default:'live'}};"
        + "\nconst hard={id:'hard',name:'困难集',case_count:50,manifest_json:{"
        + "difficulty:'hard',suite_key:'suite-a',minimum_case_count:50,"
        + "evaluation_mode_default:'live'}};"
        + "\nconst finalEasy={id:'final-easy',name:'最终简单集',case_count:50,"
        + "difficulty:'easy',suite_key:finalSuite,minimum_case_count:50,"
        + "evaluation_mode_default:'live'};"
        + "\nconst finalMedium={id:'final-medium',name:'最终中等集',case_count:50,"
        + "difficulty:'medium',suite_key:finalSuite,minimum_case_count:50,"
        + "evaluation_mode_default:'live'};"
        + "\nconst finalHard={id:'final-hard',name:'最终困难集',case_count:50,"
        + "difficulty:'hard',suite_key:finalSuite,minimum_case_count:50,"
        + "evaluation_mode_default:'live'};"
        + "\nconst explicitV3={manifest_json:{gate_policy:{case_pass_policy:{"
        + "policy_version:'stt.eval.case.v3'}}}};"
        + "\nconsole.log(JSON.stringify({"
        + "suite:datasetField(hard,'suite_key',''),"
        + "difficulty:difficultyMeta(hard),"
        + "hardStatus:datasetCaseStatus(hard),"
        + "mediumStatus:datasetCaseStatus(medium),"
        + "defaultMode:datasetDefaultMode(hard),"
        + "ordered:[hard,finalMedium,easy,finalHard,medium,finalEasy]"
        + ".sort(compareDatasets).map(item=>item.id),"
        + "fresh:preferredDatasetId([hard,finalMedium,easy,finalHard,medium,finalEasy],''),"
        + "explicit:preferredDatasetId([hard,finalMedium,easy,finalHard,medium,finalEasy],'hard'),"
        + "missing:preferredDatasetId([hard,finalMedium,easy,finalHard,medium,finalEasy],'gone'),"
        + "descriptor:datasetDescriptor(finalEasy),"
        + "defaultPolicy:datasetCasePolicy(finalEasy),"
        + "v3Policy:datasetCasePolicy(explicitV3),"
        + "rootEligible:isRootBaselineRun({status:'succeeded',baseline_run_id:null}),"
        + "candidateEligible:isRootBaselineRun({status:'succeeded',baseline_run_id:'root'}),"
        + "failedEligible:isRootBaselineRun({status:'failed',baseline_run_id:null})"
        + "}));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert result["suite"] == "suite-a"
    assert result["difficulty"] == {
        "key": "hard",
        "label": "困难",
        "order": 2,
        "tone": "danger",
    }
    assert result["hardStatus"] == {
        "count": 50,
        "minimum": 50,
        "meets": True,
        "label": "题数达标 50 / 50",
        "tone": "success",
    }
    assert result["mediumStatus"]["meets"] is False
    assert result["mediumStatus"]["label"] == "题数不足 49 / 50"
    assert result["defaultMode"] == "live"
    assert result["ordered"] == [
        "final-easy",
        "final-medium",
        "final-hard",
        "easy",
        "medium",
        "hard",
    ]
    assert result["fresh"] == "final-easy"
    assert result["explicit"] == "hard"
    assert result["missing"] == "final-easy"
    assert result["descriptor"] == (
        "suite=three-difficulty-live-2026-08-11-reviewed-v2"
        " · 难度=简单 · 范围=完整链路（live）"
    )
    assert result["defaultPolicy"] == {
        "version": "stt.eval.case.v2",
        "explicit": False,
        "label": "默认 stt.eval.case.v2",
    }
    assert result["v3Policy"] == {
        "version": "stt.eval.case.v3",
        "explicit": True,
        "label": "显式 stt.eval.case.v3（候选概念合同）",
    }
    assert result["rootEligible"] is True
    assert result["candidateEligible"] is False
    assert result["failedEligible"] is False


def test_live_run_confirmation_uses_selected_dataset_name_and_safe_fallback() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"function liveRunConfirmationMessage\(dataset\) \{.*?\n    \}",
        page,
        flags=re.DOTALL,
    )
    assert match is not None

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript function contract")
    script = (
        match.group(0)
        + "\nconsole.log(JSON.stringify(["
        + "liveRunConfirmationMessage({name:'联想海豚用户手册黄金集'}),"
        + "liveRunConfirmationMessage({name:'   '}),"
        + "liveRunConfirmationMessage(null)"
        + "]));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    selected, blank, missing = json.loads(completed.stdout)
    assert "“联想海豚用户手册黄金集”的问题/上下文" in selected
    assert "“当前选中的评测数据集”的问题/上下文" in blank
    assert missing == blank


def test_gate_presentation_distinguishes_baseline_failures_from_candidate_failures() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"function gatePresentation\(run\) \{.*?\n    \}",
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    assert page.index("baseline_recorded_with_failures") < page.index(
        "候选运行未通过质量门禁"
    )

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript function contract")
    script = (
        match.group(0)
        + "\nconsole.log(JSON.stringify(["
        + "gatePresentation({gate_result:{status:'baseline_recorded',passed:true}}),"
        + "gatePresentation({gate_result:{status:'baseline_recorded_with_failures',passed:false}}),"
        + "gatePresentation({gate_result:{status:'failed',passed:false}})"
        + "]));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    baseline, baseline_failed, candidate_failed = json.loads(completed.stdout)
    assert baseline == {
        "tone": "warning",
        "message": "首次运行已记录为事实基线。",
    }
    assert baseline_failed == {
        "tone": "danger",
        "message": "首次运行已建立事实基线，但关键 case 失败，不能视为门禁通过。",
    }
    assert candidate_failed == {
        "tone": "danger",
        "message": "候选运行未通过质量门禁。",
    }


def test_gate_failure_rules_are_rendered_as_human_readable_rank_regressions() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"function gateFailureText\(failure\) \{.*?\n    \}",
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    assert "gateFailureText(failure)" in page
    assert "门禁策略：" in page
    assert "metrics.retrieval.ndcg_at_k.10" in page

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript function contract")
    script = (
        match.group(0)
        + "\nconsole.log(gateFailureText({"
        + "rule:'mrr',delta:-0.107142857,max_drop:0.02,"
        + "paired_count:7,unknown_pair_count:1}));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.stdout.strip() == (
        "MRR 绝对下降 10.71 个百分点，允许最多下降 2.00 个百分点"
        "（有效配对 7，未知配对 1）"
    )


def test_evaluation_page_explains_answer_first_case_outcomes() -> None:
    page = (
        Path(__file__).resolve().parents[2]
        / "app/process/query/page/evaluation.html"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"function caseOutcomeReasonText\(outcome\) \{.*?\n    \}",
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    assert (
        "['严格有据通过率', ['metrics', 'case_outcome', "
        "'strict_supported_case_pass_rate']" in page
    )
    assert "['综合用例质量', ['metrics', 'case_outcome', 'quality_score']" in page
    assert (
        "['最终答案正确率', ['metrics', 'case_outcome', "
        "'final_answer_accuracy']" in page
    )
    assert "最终答案是否正确（目标）：" in page
    assert "最终答案正确性得分：" in page
    assert "严格有据判定：" in page
    assert "用例判定策略：" in page
    assert "manifest 省略用例策略时使用 case.v2" in page
    assert "case.v3 候选概念合同" in page
    assert "answer_correctness: '最终答案正确性'" in page
    assert "case_quality: '综合用例质量'" in page
    assert "function visibleAnswerAudit(result)" in page
    assert "答案完整性优先，部分内容未通过引用校验" in page
    assert "strict_grounding_eligible=false" in page
    assert "function completenessSelfReviewAudit(result)" in page
    assert "function completenessSelfReviewText(audit)" in page
    assert "完整性自审审计" in page
    assert "prompt_sha256" not in page
    assert "provider_response_sha256" not in page

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript function contract")
    script = (
        match.group(0)
        + "\nconsole.log(JSON.stringify(["
        + "caseOutcomeReasonText({reason_codes:['faithfulness_below_minimum',"
        + "'citation_coverage_below_minimum'],diagnostic_codes:[]}),"
        + "caseOutcomeReasonText({reason_codes:[],"
        + "diagnostic_codes:['retrieval_ground_truth_miss']}),"
        + "caseOutcomeReasonText({reason_codes:[],diagnostic_codes:[]})"
        + "]));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    unsupported, retrieval_diagnostic, passed = json.loads(completed.stdout)
    assert unsupported == "答案缺少独立证据支持；引用覆盖未达门槛"
    assert retrieval_diagnostic == "召回标注未命中（仅诊断）"
    assert passed == "最终答案与证据守门均通过"
