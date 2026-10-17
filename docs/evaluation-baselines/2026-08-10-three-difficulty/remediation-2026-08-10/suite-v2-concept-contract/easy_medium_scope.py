"""定义只评估 easy 与 medium 的不可晋升执行范围。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


FULL_SCOPE_NAME = "full"
SCOPE_NAME = "easy-medium-evaluation"
SCOPE_ID = "easy-medium-evaluation-2026-08-13-v1"
SCOPE_SCHEMA_VERSION = "stt.eval.easy-medium-evaluation-scope.v1"
SCOPE_DIFFICULTIES = ("easy", "medium")
SCOPE_QUALITY_TARGETS = MappingProxyType({"easy": 48, "medium": 43})
SCOPE_HARD_REFERENCE_ONLY = True
SCOPE_CANDIDATE_RUN_ALLOWED = True
SCOPE_CANONICAL_PROMOTION_ALLOWED = False
SCOPE_PROMOTABLE = False
SCOPE_BASELINE_REGISTRATION_ALLOWED = False
SCOPE_FROZEN_CANDIDATE_MANIFEST_FILE_SHA256 = MappingProxyType(
    {
        "easy": "e370d19b750f1b0c861f06e6cd2494817f98c7940b91e352e171ed98f85fd6b2",
        "medium": "1acec643cc67dcfc1e843920d1db4fffa92d3c3ac6c9072a450389277ec53a09",
        "hard": "e688a826a3e54687111f52c40a50ee12d5c69171827aa998425358e7474e139b",
    }
)
SCOPE_FROZEN_CANDIDATE_MANIFEST_SHA256 = MappingProxyType(
    {
        "easy": "a4eb57b2588ad85797c4e5176645755d622e28e8dd1d9a04a67d0103abcc7c20",
        "medium": "84f43d42eccdf273da0fd67f24810fd947785748e113198eb757e698c2b3f18b",
        "hard": "e3debda5901e4f548e7b3823ccb5cb22661b5e0c71074f3ca87d20b8b2628f63",
    }
)
SCOPE_FROZEN_SUITE_FILE_SHA256 = (
    "4518b653b8782c88b358266d7f6fd7076810945ee2c72d06da03be0f3926ed17"
)
SCOPE_FROZEN_QA_SUMMARY_FILE_SHA256 = (
    "47181d1ada7bb6982c3327f61f8adea1cf21c1c7cfa968e736f580691f552cbc"
)
SCOPE_FROZEN_BLOCKER_CANDIDATES_SHA256 = (
    "e1fee7d040fd9d232c38d39b42eaa9e1cf157b24924dc7b75d4edc15c7d06dae"
)
SCOPE_FROZEN_BLOCKER_REQUEST_MAP_SHA256 = (
    "bbf9ce4872b197d98c97e3bb0c4b7f7be8035f47cb67eb0447851ae259a00961"
)
_AUTHORITY_FIELDS = (
    "custom_id",
    "case_id",
    "parent_case_id",
    "difficulty",
    "payload_sha256",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """流式计算scope冻结文件的物理SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_lineage(
    candidate: Mapping[str, Any],
    *,
    expected_difficulty: str | None = None,
) -> tuple[str, str, str, int]:
    """校验case/parent的难度命名空间与三位序号完全一致。"""

    metadata = candidate.get("metadata") or {}
    difficulty = str(metadata.get("difficulty") or "")
    case_id = str(candidate.get("case_id") or "")
    parent_case_id = str(metadata.get("parent_case_id") or "")
    namespace = {"easy": "EASY", "medium": "MEDIUM", "hard": "HARD"}.get(
        difficulty
    )
    match = re.fullmatch(r"P3500-(EASY|MEDIUM|HARD)-V2-(\d{3})", case_id)
    if (
        namespace is None
        or (expected_difficulty is not None and difficulty != expected_difficulty)
        or match is None
        or match.group(1) != namespace
        or parent_case_id != f"P3500-{namespace}-{match.group(2)}"
    ):
        raise RuntimeError(f"candidate manifest lineage namespace drift: {case_id}")
    ordinal = int(match.group(2))
    if not 1 <= ordinal <= 50:
        raise RuntimeError(f"candidate manifest lineage ordinal drift: {case_id}")
    return case_id, parent_case_id, difficulty, ordinal


def scope_profile() -> dict[str, Any]:
    """返回可序列化且字段固定的 easy/medium 范围契约。"""

    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "scope_name": SCOPE_NAME,
        "scope_id": SCOPE_ID,
        "difficulties": list(SCOPE_DIFFICULTIES),
        "quality_targets": dict(SCOPE_QUALITY_TARGETS),
        "hard_reference_only": SCOPE_HARD_REFERENCE_ONLY,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
    }


def _request_payload_identity(request: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """解析盲审payload，并验证物理内容与请求中的payload哈希一致。"""

    messages = (request.get("request") or {}).get("messages") or []
    if len(messages) != 2 or not isinstance(messages[1], Mapping):
        raise RuntimeError("scope request message structure is invalid")
    try:
        payload = json.loads(str(messages[1].get("content") or ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("scope request payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("scope request payload must be an object")
    payload_sha256 = _sha256_json(payload)
    if request.get("payload_sha256") != payload_sha256:
        raise RuntimeError("scope request content/payload_sha256 binding drifted")
    return payload, payload_sha256


def _candidate_review_payload(candidate_case: Mapping[str, Any]) -> dict[str, Any]:
    """按冻结盲审投影从候选题重建语义payload。"""

    contract = candidate_case.get("answer_contract") or {}
    branches = contract.get("required_branches") or []
    evidence = candidate_case.get("relevant_evidence") or []
    review_flags = [
        str(item.get("code") or "")
        for item in (contract.get("review") or {}).get("flags") or []
    ]
    return {
        "query": str(candidate_case.get("query") or ""),
        "reference_answer": str(candidate_case.get("reference_answer") or ""),
        "unanswerable": bool(candidate_case.get("unanswerable")),
        "legacy_expected_terms": [
            str(value) for value in candidate_case.get("expected_terms") or []
        ],
        "forbidden_terms": [
            str(value) for value in candidate_case.get("forbidden_terms") or []
        ],
        "branches": [
            {"branch_label": f"B{index + 1}", "query_span": branch.get("query_span")}
            for index, branch in enumerate(branches)
        ],
        "evidence": [
            {
                "evidence_label": f"E{index + 1}",
                "page": item.get("page"),
                "text": str(item.get("text") or ""),
            }
            for index, item in enumerate(evidence)
        ],
        "deterministic_candidate": {
            "answerability": contract.get("answerability"),
            "required_concepts": contract.get("required_concepts"),
            "critical_constraints": contract.get("critical_constraints"),
            "required_branches": branches,
            "evidence_concept_coverage": contract.get("evidence_concept_coverage"),
            "review_flags": review_flags,
        },
    }


def build_candidate_scope_authority(
    candidate_cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从已经过冻结manifest校验的候选题构造scope权威行。"""

    authority: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, candidate in enumerate(candidate_cases, start=1):
        case_id, parent_case_id, difficulty, _ = _case_lineage(candidate)
        payload_sha256 = _sha256_json(_candidate_review_payload(candidate))
        custom_id = f"concept-review-{payload_sha256[:20]}"
        row = {
            "custom_id": custom_id,
            "case_id": case_id,
            "parent_case_id": parent_case_id,
            "difficulty": difficulty,
            "payload_sha256": payload_sha256,
        }
        if (
            custom_id in seen_ids
            or any(not row[field] for field in _AUTHORITY_FIELDS)
            or row["difficulty"] not in {"easy", "medium", "hard"}
        ):
            raise RuntimeError(f"candidate authority 第 {position} 行身份非法或重复")
        seen_ids.add(custom_id)
        authority.append(row)
    if not authority:
        raise RuntimeError("candidate authority 不能为空")
    return authority


def build_candidate_manifest_scope_authority(
    candidate_manifests: Mapping[str, dict[str, Any]],
    *,
    suite_receipt: Mapping[str, Any],
    qa_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """以scope代码内冻结哈希和双receipt验证三份候选manifest。"""

    difficulties = ("easy", "medium", "hard")
    if set(candidate_manifests) != set(difficulties):
        raise RuntimeError("candidate manifest set does not exactly cover three difficulties")
    suite_datasets = suite_receipt.get("datasets") or {}
    qa_hashes = qa_summary.get("candidate_manifest_sha256") or {}
    authority: list[dict[str, Any]] = []
    for difficulty in difficulties:
        manifest = candidate_manifests[difficulty]
        expected_hash = SCOPE_FROZEN_CANDIDATE_MANIFEST_SHA256[difficulty]
        suite_row = suite_datasets.get(difficulty) or {}
        cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
        if (
            _sha256_json(manifest) != expected_hash
            or suite_row.get("checksum") != expected_hash
            or qa_hashes.get(difficulty) != expected_hash
            or manifest.get("difficulty") != difficulty
            or not isinstance(cases, list)
            or len(cases) != 50
            or suite_row.get("case_count") != 50
            or suite_row.get("path") != f"datasets/{difficulty}.v2.json"
        ):
            raise RuntimeError(f"candidate manifest frozen identity drift: {difficulty}")
        ordinals: set[int] = set()
        for candidate in cases:
            _, _, _, ordinal = _case_lineage(
                candidate,
                expected_difficulty=difficulty,
            )
            if ordinal in ordinals:
                raise RuntimeError(
                    f"candidate manifest duplicate lineage ordinal: {difficulty}/{ordinal:03d}"
                )
            ordinals.add(ordinal)
        if ordinals != set(range(1, 51)):
            raise RuntimeError(f"candidate manifest ordinal coverage drift: {difficulty}")
        authority.extend(build_candidate_scope_authority(cases))
    if len(authority) != 150 or len({row["custom_id"] for row in authority}) != 150:
        raise RuntimeError("candidate manifest authority must contain 150 unique requests")
    return authority


def load_frozen_candidate_manifest_scope_authority(root: Path) -> list[dict[str, Any]]:
    """从磁盘加载并用代码内物理SHA锚验证冻结候选manifest。"""

    manifest_paths = {
        difficulty: root / "datasets" / f"{difficulty}.v2.json"
        for difficulty in ("easy", "medium", "hard")
    }
    suite_path = root / "suite-v2.json"
    qa_summary_path = root / "qa" / "summary.json"
    for difficulty, path in manifest_paths.items():
        if _sha256_file(path) != SCOPE_FROZEN_CANDIDATE_MANIFEST_FILE_SHA256[difficulty]:
            raise RuntimeError(f"candidate manifest frozen file SHA drift: {difficulty}")
    if _sha256_file(suite_path) != SCOPE_FROZEN_SUITE_FILE_SHA256:
        raise RuntimeError("candidate suite frozen file SHA drift")
    if _sha256_file(qa_summary_path) != SCOPE_FROZEN_QA_SUMMARY_FILE_SHA256:
        raise RuntimeError("candidate QA summary frozen file SHA drift")
    manifests = {
        difficulty: json.loads(path.read_text(encoding="utf-8"))
        for difficulty, path in manifest_paths.items()
    }
    return build_candidate_manifest_scope_authority(
        manifests,
        suite_receipt=json.loads(suite_path.read_text(encoding="utf-8")),
        qa_summary=json.loads(qa_summary_path.read_text(encoding="utf-8")),
    )


def build_blocker_scope_authority(
    candidate_cases: Sequence[dict[str, Any]],
    *,
    request_map: Sequence[dict[str, Any]],
    proposals: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """以冻结pack哈希及proposal谱系构造blocker权威行。"""

    if _sha256_json(list(candidate_cases)) != SCOPE_FROZEN_BLOCKER_CANDIDATES_SHA256:
        raise RuntimeError("blocker candidates frozen identity drift")
    if _sha256_json(list(request_map)) != SCOPE_FROZEN_BLOCKER_REQUEST_MAP_SHA256:
        raise RuntimeError("blocker request-map authority frozen identity drift")
    proposal_by_parent = {
        str(row.get("case_id") or ""): row for row in proposals
    }
    if len(proposal_by_parent) != len(proposals) or len(proposal_by_parent) != 15:
        raise RuntimeError("blocker proposal authority must contain 15 unique rows")
    candidate_by_case_id: dict[str, dict[str, Any]] = {}
    for candidate in candidate_cases:
        case_id, parent_case_id, difficulty, _ = _case_lineage(candidate)
        metadata = candidate.get("metadata") or {}
        proposal = proposal_by_parent.get(parent_case_id)
        proposal_sha256 = str((proposal or {}).get("audit", {}).get("proposed_case_sha256") or "")
        if (
            proposal is None
            or proposal.get("difficulty") != difficulty
            or metadata.get("blocker_remediation_proposal_sha256") != proposal_sha256
            or case_id in candidate_by_case_id
        ):
            raise RuntimeError(f"blocker candidate/proposal namespace drift: {parent_case_id}")
        candidate_by_case_id[case_id] = dict(candidate)
    if len(candidate_by_case_id) != 15:
        raise RuntimeError("blocker candidate authority must contain 15 unique rows")
    seen_custom_ids: set[str] = set()
    for mapping in request_map:
        custom_id = str(mapping.get("custom_id") or "")
        case_id = str(mapping.get("case_id") or "")
        candidate = candidate_by_case_id.get(case_id)
        metadata = (candidate or {}).get("metadata") or {}
        parent_case_id = str(metadata.get("parent_case_id") or "")
        proposal = proposal_by_parent.get(parent_case_id) or {}
        proposal_sha256 = str((proposal.get("audit") or {}).get("proposed_case_sha256") or "")
        if (
            not custom_id
            or custom_id in seen_custom_ids
            or candidate is None
            or mapping.get("candidate_case_sha256") != _sha256_json(candidate)
            or mapping.get("proposed_case_sha256") != proposal_sha256
            or mapping.get("parent_case_id") != parent_case_id
            or mapping.get("difficulty") != metadata.get("difficulty")
        ):
            raise RuntimeError(f"blocker request-map/candidate authority drift: {custom_id}")
        seen_custom_ids.add(custom_id)
    if len(seen_custom_ids) != 15:
        raise RuntimeError("blocker request-map authority must contain 15 unique rows")
    return build_candidate_scope_authority(candidate_cases)


def build_contract_scope_authority(
    rereview_requests: Sequence[dict[str, Any]],
    original_requests: Sequence[dict[str, Any]],
    original_request_map: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从已验证的原始150题请求和映射构造contract rereview权威行。"""

    original_by_id: dict[str, dict[str, Any]] = {}
    derived_by_rereview_id: dict[str, tuple[dict[str, Any], str]] = {}
    for request in original_requests:
        original_id = str(request.get("custom_id") or "")
        _, payload_sha256 = _request_payload_identity(request)
        expected_original_id = f"concept-review-{payload_sha256[:20]}"
        if not original_id or original_id != expected_original_id or original_id in original_by_id:
            raise RuntimeError("original authority request identity drifted or duplicated")
        original_by_id[original_id] = dict(request)
        identity = f"contract-rereview-v1:{original_id}:{payload_sha256}"
        rereview_id = f"concept-contract-rereview-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        if rereview_id in derived_by_rereview_id:
            raise RuntimeError("contract authority rereview id duplicated")
        derived_by_rereview_id[rereview_id] = (dict(request), payload_sha256)

    lineage_by_original_id: dict[str, dict[str, Any]] = {}
    for position, mapping in enumerate(original_request_map, start=1):
        original_id = str(mapping.get("custom_id") or "")
        if not original_id or original_id in lineage_by_original_id:
            raise RuntimeError(f"original authority map 第 {position} 行缺失或重复")
        lineage_by_original_id[original_id] = dict(mapping)
    if set(lineage_by_original_id) != set(original_by_id):
        raise RuntimeError("original authority map must exactly cover original requests")

    authority: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for request in rereview_requests:
        rereview_id = str(request.get("custom_id") or "")
        derived = derived_by_rereview_id.get(rereview_id)
        if derived is None or rereview_id in seen_ids:
            raise RuntimeError("contract rereview request is outside original authority")
        original_request, payload_sha256 = derived
        original_id = str(original_request["custom_id"])
        lineage = lineage_by_original_id[original_id]
        payload, actual_payload_sha256 = _request_payload_identity(request)
        original_payload, _ = _request_payload_identity(original_request)
        if actual_payload_sha256 != payload_sha256 or payload != original_payload:
            raise RuntimeError("contract rereview semantic payload differs from original authority")
        row = {
            "custom_id": rereview_id,
            "case_id": str(lineage.get("case_id") or ""),
            "parent_case_id": str(lineage.get("parent_case_id") or ""),
            "difficulty": str(lineage.get("difficulty") or ""),
            "payload_sha256": payload_sha256,
        }
        if (
            any(not row[field] for field in _AUTHORITY_FIELDS)
            or row["difficulty"] not in {"easy", "medium", "hard"}
        ):
            raise RuntimeError("contract authority lineage is incomplete")
        seen_ids.add(rereview_id)
        authority.append(row)
    if not authority:
        raise RuntimeError("contract authority 不能为空")
    return authority


@dataclass(frozen=True)
class ScopeSelection:
    """保存完整请求包与目标子集之间的可审计映射。"""

    target_requests: tuple[dict[str, Any], ...]
    target_custom_ids: tuple[str, ...]
    target_parent_case_ids: tuple[str, ...]
    target_case_ids: tuple[str, ...]
    target_difficulty_by_custom_id: Mapping[str, str]
    target_difficulty_counts: dict[str, int]
    full_case_count: int
    full_request_pack_sha256: str
    target_request_pack_sha256: str
    request_map_sha256: str

    @property
    def target_case_count(self) -> int:
        return len(self.target_custom_ids)

    def receipt(self) -> dict[str, Any]:
        """返回绑定完整请求包、request-map 与目标ID的范围收据。"""

        profile = scope_profile()
        return {
            **profile,
            "scope_profile_sha256": _sha256_json(profile),
            "scope_target_custom_ids": list(self.target_custom_ids),
            "scope_target_custom_ids_sha256": _sha256_json(
                list(self.target_custom_ids)
            ),
            "scope_target_parent_case_ids": list(self.target_parent_case_ids),
            "scope_target_parent_case_ids_sha256": _sha256_json(
                list(self.target_parent_case_ids)
            ),
            "scope_target_case_ids": list(self.target_case_ids),
            "scope_target_case_ids_sha256": _sha256_json(list(self.target_case_ids)),
            "scope_target_difficulty_counts": dict(self.target_difficulty_counts),
            "scope_case_expected_count": self.target_case_count,
            "full_case_expected_count": self.full_case_count,
            "full_request_pack_sha256": self.full_request_pack_sha256,
            "scope_target_request_pack_sha256": self.target_request_pack_sha256,
            "request_map_sha256": self.request_map_sha256,
        }


def select_scope_requests(
    requests: Sequence[dict[str, Any]],
    request_map: Sequence[dict[str, Any]],
    scope_name: str,
    *,
    authority_rows: Sequence[dict[str, Any]],
) -> ScopeSelection:
    """用独立权威行绑定完整请求包后选择easy/medium目标。"""

    if scope_name != SCOPE_NAME:
        raise RuntimeError(f"未知评估范围: {scope_name}")
    request_ids = [str(row.get("custom_id") or "") for row in requests]
    if not request_ids or "" in request_ids or len(set(request_ids)) != len(request_ids):
        raise RuntimeError("完整请求包的 custom_id 缺失或重复")
    authority_by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(authority_rows, start=1):
        custom_id = str(row.get("custom_id") or "")
        normalized = {field: str(row.get(field) or "") for field in _AUTHORITY_FIELDS}
        if (
            not custom_id
            or custom_id in authority_by_id
            or any(not normalized[field] for field in _AUTHORITY_FIELDS)
            or normalized["difficulty"] not in {"easy", "medium", "hard"}
        ):
            raise RuntimeError(f"scope authority 第 {position} 行身份非法或重复")
        authority_by_id[custom_id] = normalized
    if set(authority_by_id) != set(request_ids):
        raise RuntimeError("scope authority 必须完整覆盖完整请求包且不得含外来ID")

    for position, request in enumerate(requests, start=1):
        custom_id = str(request["custom_id"])
        _, payload_sha256 = _request_payload_identity(request)
        authority = authority_by_id[custom_id]
        if payload_sha256 != authority["payload_sha256"]:
            raise RuntimeError(f"scope request 第 {position} 行与authority payload不一致")
    mapping_by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(request_map, start=1):
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in mapping_by_id:
            raise RuntimeError(f"request-map 第 {position} 行 custom_id 缺失或重复")
        difficulty = str(row.get("difficulty") or "")
        parent_case_id = str(row.get("parent_case_id") or "")
        case_id = str(row.get("case_id") or "")
        if difficulty not in {"easy", "medium", "hard"}:
            raise RuntimeError(f"request-map 第 {position} 行 difficulty 非法")
        if not parent_case_id or not case_id:
            raise RuntimeError(f"request-map 第 {position} 行 lineage 不完整")
        authority = authority_by_id.get(custom_id)
        exact_values = {
            "custom_id": custom_id,
            "case_id": case_id,
            "parent_case_id": parent_case_id,
            "difficulty": difficulty,
        }
        if authority is None or any(
            exact_values[field] != authority[field] for field in exact_values
        ):
            raise RuntimeError(f"request-map 第 {position} 行与scope authority不一致")
        if "payload_sha256" in row and str(row.get("payload_sha256") or "") != authority[
            "payload_sha256"
        ]:
            raise RuntimeError(f"request-map 第 {position} 行payload与scope authority不一致")
        mapping_by_id[custom_id] = dict(row)
    if set(mapping_by_id) != set(request_ids):
        raise RuntimeError("request-map 必须完整覆盖完整请求包且不得含外来ID")

    target_requests: list[dict[str, Any]] = []
    target_custom_ids: list[str] = []
    target_parent_case_ids: list[str] = []
    target_case_ids: list[str] = []
    difficulty_by_id: dict[str, str] = {}
    difficulty_counts = {difficulty: 0 for difficulty in SCOPE_DIFFICULTIES}
    for request in requests:
        custom_id = str(request["custom_id"])
        mapping = mapping_by_id[custom_id]
        difficulty = str(mapping["difficulty"])
        if difficulty not in SCOPE_DIFFICULTIES:
            continue
        target_requests.append(dict(request))
        target_custom_ids.append(custom_id)
        target_parent_case_ids.append(str(mapping["parent_case_id"]))
        target_case_ids.append(str(mapping["case_id"]))
        difficulty_by_id[custom_id] = difficulty
        difficulty_counts[difficulty] += 1
    if not target_requests:
        raise RuntimeError("easy/medium 评估范围没有目标请求")

    return ScopeSelection(
        target_requests=tuple(target_requests),
        target_custom_ids=tuple(target_custom_ids),
        target_parent_case_ids=tuple(target_parent_case_ids),
        target_case_ids=tuple(target_case_ids),
        target_difficulty_by_custom_id=MappingProxyType(difficulty_by_id),
        target_difficulty_counts=difficulty_counts,
        full_case_count=len(requests),
        full_request_pack_sha256=_sha256_json(list(requests)),
        target_request_pack_sha256=_sha256_json(target_requests),
        request_map_sha256=_sha256_json(list(request_map)),
    )


def scope_progress(
    state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selection: ScopeSelection,
) -> dict[str, Any]:
    """同时报告目标范围进度与完整账本进度。"""

    full_ids = {
        str(custom_id) for custom_id in (state.get("successful_by_id") or {})
    }
    completed = [
        custom_id for custom_id in selection.target_custom_ids if custom_id in full_ids
    ]
    remaining = [
        custom_id for custom_id in selection.target_custom_ids if custom_id not in full_ids
    ]
    return {
        "scope_case_expected_count": selection.target_case_count,
        "scope_case_success_count": len(completed),
        "scope_remaining_case_calls": len(remaining),
        "scope_completed_custom_ids": completed,
        "scope_remaining_custom_ids": remaining,
        "full_case_expected_count": selection.full_case_count,
        "full_case_success_count": len(full_ids),
        "full_remaining_case_calls": selection.full_case_count - len(full_ids),
    }


def scope_completion_errors(
    state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selection: ScopeSelection,
) -> list[str]:
    """要求目标ID全部成功，且目标ID没有未决materializer状态。"""

    target_ids = set(selection.target_custom_ids)
    successful = set(state.get("successful_by_id") or {}) & target_ids
    errors: list[str] = []
    if successful != target_ids:
        errors.append("scope successful ids do not exactly cover scope target ids")
    for key in (
        "retry_eligible_by_id",
        "invalidated_by_id",
        "retry_started_by_id",
        "terminal_by_id",
    ):
        if set(state.get(key) or {}) & target_ids:
            errors.append(f"scope completion retains nonterminal state: {key}")
    return errors


def scope_summary_path(rereview_root: Path, scope_name: str) -> Path:
    """把范围摘要与完整执行摘要物理隔离。"""

    if scope_name == FULL_SCOPE_NAME:
        return rereview_root / "execution-summary.json"
    if scope_name != SCOPE_NAME:
        raise RuntimeError(f"未知评估范围: {scope_name}")
    return rereview_root / f"execution-summary.{scope_name}.json"
