from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.application.retrieval_fusion import (
    CalibrationConfig,
    FusionConfig,
    RetrievalCandidate,
    RouteObservation,
    ScoreSemantics,
    canonicalize_web_uri,
    fuse_route_results,
    fusion_result_to_mappings,
    normalized_query_fingerprint,
)
from app.rag.query.rrf_service import fuse_by_rrf


def _local(
    *,
    marker: str,
    document_id: str,
    content: str,
    score: float | None,
) -> dict:
    result = {
        "evidence_id": f"ev_{marker * 40}",
        "source_type": "local",
        "tenant_id": "tenant-a",
        "kb_id": "kb-a",
        "document_id": document_id,
        "document_version_id": f"dver-{document_id}",
        "index_version_id": "index-a",
        "chunk_id": f"chunk-{marker}",
        "content": content,
        "title": f"title-{marker}",
    }
    if score is not None:
        result["raw_score"] = score
    return result


def _ranked_ids(result) -> list[str | None]:
    return [item.candidate.evidence_id for item in result.ranked]


def test_rrf_merges_same_evidence_across_routes_without_overwriting_raw_scores():
    content = "在维护 HAK180 前必须断开电源，并等待设备完全停止。"
    dense = _local(marker="a", document_id="doc-a", content=content, score=0.91)
    hyde = dict(dense)
    hyde["raw_score"] = 17.5

    result = fuse_route_results(
        {"hyde": [hyde], "dense": [dense]},
        FusionConfig(output_k=3, near_duplicate_threshold=None),
    )

    assert len(result.ranked) == 1
    ranked = result.ranked[0]
    assert ranked.candidate.evidence_id == dense["evidence_id"]
    assert {
        observation.route_id: observation.raw_scores["raw_score"]
        for observation in ranked.candidate.observations
    } == {"dense": 0.91, "hyde": 17.5}
    assert ranked.fusion_contributions == pytest.approx({"dense": 1 / 61, "hyde": 1 / 61})
    assert ranked.fusion_score == pytest.approx(2 / 61)
    assert result.diagnostics.exact_duplicate_count == 1
    assert result.diagnostics.valid_count == 2
    assert any(decision.reason == "exact_duplicate" for decision in result.diagnostics.decisions)


def test_weighted_calibration_respects_lower_is_better_and_keeps_raw_values():
    first = _local(
        marker="a",
        document_id="doc-a",
        content="距离更近的正确证据",
        score=0.1,
    )
    second = _local(
        marker="b",
        document_id="doc-b",
        content="距离更远的次要证据",
        score=0.9,
    )
    semantics = {
        "metric": "L2",
        "direction": "lower_is_better",
        "model_version": "embedding-v3",
    }
    first["score_semantics"] = semantics
    second["score_semantics"] = semantics

    result = fuse_route_results(
        {"distance": [first, second]},
        FusionConfig(
            method="weighted",
            output_k=2,
            near_duplicate_threshold=None,
            mmr_lambda=1.0,
            route_calibrations={
                "distance": CalibrationConfig(
                    method="min_max",
                    version="l2-minmax.v1",
                )
            },
        ),
    )

    assert _ranked_ids(result) == [first["evidence_id"], second["evidence_id"]]
    assert result.ranked[0].fusion_score == pytest.approx(1.0)
    assert result.ranked[1].fusion_score == pytest.approx(0.0)
    assert result.ranked[0].candidate.observations[0].raw_scores["raw_score"] == 0.1
    assert result.ranked[1].candidate.observations[0].raw_scores["raw_score"] == 0.9
    route = result.diagnostics.routes[0]
    assert route.calibration_method == "min_max"
    assert route.calibration_version == "l2-minmax.v1"
    assert route.raw_score_min == pytest.approx(0.1)
    assert route.raw_score_max == pytest.approx(0.9)


def test_constant_raw_scores_have_a_deterministic_neutral_calibration():
    first = _local(marker="a", document_id="doc-a", content="候选甲", score=5.0)
    second = _local(marker="b", document_id="doc-b", content="候选乙", score=5.0)

    result = fuse_route_results(
        {"dense": [first, second]},
        FusionConfig(
            method="weighted",
            output_k=2,
            near_duplicate_threshold=None,
            route_calibrations={
                "dense": CalibrationConfig(method="min_max", version="constant.v1")
            },
        ),
    )

    assert [item.fusion_score for item in result.ranked] == pytest.approx([0.5, 0.5])
    # 同分时以稳定 Evidence 身份裁决，不依赖 mapping 插入顺序或对象地址。
    assert _ranked_ids(result) == [first["evidence_id"], second["evidence_id"]]


def test_web_canonical_url_deduplicates_tracking_variants_but_never_local_evidence():
    content = "同一份安全公告正文，内容快照必须参与 Web 身份。"
    web_one = {
        "type": "web",
        "url": "HTTPS://Example.COM:443/safety/?utm_source=test&b=2&a=1#part",
        "snippet": content,
        "provider_score": 0.8,
    }
    web_two = {
        "type": "web",
        "url": "https://example.com/safety?a=1&b=2",
        "snippet": content,
        "provider_score": 0.7,
    }
    local = _local(
        marker="c",
        document_id="doc-local",
        content=content,
        score=0.6,
    )

    result = fuse_route_results(
        {"web": [web_one, web_two], "dense": [local]},
        FusionConfig(output_k=5, near_duplicate_threshold=0.8),
    )

    assert canonicalize_web_uri(web_one["url"]) == "https://example.com/safety?a=1&b=2"
    assert result.diagnostics.exact_duplicate_count == 1
    assert result.diagnostics.near_duplicate_count == 0
    assert {item.candidate.source_type for item in result.ranked} == {"local", "web"}


def test_near_duplicate_collapses_local_content_and_retains_member_mapping():
    base = (
        "在安装、使用、维护 HAK180 设备之前，应完整阅读产品安全手册，"
        "理解所有警告并确认电源已经可靠断开。"
    )
    variant = base.replace("，", " ").replace("。", "")
    first = _local(marker="a", document_id="doc-a", content=base, score=0.9)
    second = _local(marker="b", document_id="doc-b", content=variant, score=0.8)

    result = fuse_route_results(
        {"dense": [first, second]},
        FusionConfig(
            output_k=2,
            near_duplicate_threshold=0.9,
            near_duplicate_min_chars=20,
        ),
    )

    assert len(result.ranked) == 1
    assert result.diagnostics.near_duplicate_count == 1
    assert len(result.ranked[0].duplicate_member_refs) == 2
    assert any(decision.reason == "near_duplicate" for decision in result.diagnostics.decisions)


def test_document_and_source_quotas_prevent_one_document_from_filling_the_window():
    candidates = [
        _local(marker="a", document_id="doc-a", content="A 文档第一块", score=0.99),
        {
            **_local(
                marker="b",
                document_id="doc-a",
                content="A 文档第二块",
                score=0.98,
            ),
            "chunk_id": "chunk-b",
        },
        _local(marker="c", document_id="doc-b", content="B 文档证据", score=0.8),
    ]
    web = {
        "type": "web",
        "url": "https://example.com/manual",
        "snippet": "供应商的补充公告",
        "provider_score": 0.7,
    }

    result = fuse_route_results(
        {"dense": candidates, "web": [web]},
        FusionConfig(
            output_k=3,
            max_per_document=1,
            source_quotas={"local": 2, "web": 1},
            near_duplicate_threshold=None,
            mmr_lambda=1.0,
        ),
    )

    assert len(result.ranked) == 3
    assert [item.candidate.source_type for item in result.ranked].count("local") == 2
    assert [item.candidate.source_type for item in result.ranked].count("web") == 1
    assert {
        item.candidate.document_id
        for item in result.ranked
        if item.candidate.source_type == "local"
    } == {"doc-a", "doc-b"}
    assert any(
        decision.reason == "document_quota_exceeded" for decision in result.diagnostics.decisions
    )


def test_empty_invalid_surrogate_and_overflow_inputs_are_bounded_and_explicit():
    valid_without_ids = {
        "content": "缺少 chunk_id 但仍可用内容哈希形成显式 surrogate 身份。",
        "raw_score": 0.5,
    }
    invalid = {"content": "非法非有限分数", "raw_score": math.nan}
    overflow = _local(
        marker="c",
        document_id="doc-c",
        content="不会被消费的第三条候选",
        score=0.4,
    )

    result = fuse_route_results(
        {"empty": [], "dense": [valid_without_ids, invalid, overflow]},
        FusionConfig(
            max_input_candidates=2,
            output_k=1,
            near_duplicate_threshold=None,
        ),
    )

    assert len(result.ranked) == 1
    assert result.ranked[0].candidate.identity_quality() == "surrogate"
    assert result.diagnostics.input_count == 2
    assert result.diagnostics.valid_count == 1
    assert result.diagnostics.invalid_count == 1
    assert result.diagnostics.invalid_reasons == {"non_finite_score": 1}
    assert result.diagnostics.overflow_count == 1
    assert result.diagnostics.input_truncated is True
    assert result.diagnostics.identity_derived_count == 1
    assert {decision.status for decision in result.diagnostics.decisions} == {"invalid", "selected"}


def test_empty_routes_return_a_valid_empty_result():
    result = fuse_route_results({"dense": [], "web": []})

    assert result.ranked == ()
    assert result.diagnostics.input_count == 0
    assert result.diagnostics.selected_count == 0
    assert result.diagnostics.routes == ()


def test_route_mapping_order_does_not_change_rank_or_score():
    dense = [_local(marker="a", document_id="doc-a", content="内部证据", score=0.9)]
    web = [
        {
            "type": "web",
            "url": "https://example.com/notice",
            "snippet": "外部公告",
            "provider_score": 0.8,
        }
    ]
    config = FusionConfig(output_k=2, near_duplicate_threshold=None)

    first = fuse_route_results({"dense": dense, "web": web}, config)
    second = fuse_route_results({"web": web, "dense": dense}, config)

    assert [
        (item.candidate.identity_key(), item.fusion_score, item.final_rank) for item in first.ranked
    ] == [
        (item.candidate.identity_key(), item.fusion_score, item.final_rank)
        for item in second.ranked
    ]


def test_rrf_tie_break_is_stable_across_rebuilt_evidence_ids():
    first_content = "打印机网络配置完成后保存设置。"
    second_content = "打印机驱动搜索完成后选择设备。"

    def rebuilt(marker_a: str, marker_b: str):
        first = _local(
            marker=marker_a,
            document_id="manual",
            content=first_content,
            score=0.9,
        )
        second = _local(
            marker=marker_b,
            document_id="manual",
            content=second_content,
            score=0.9,
        )
        return fuse_route_results(
            {"route_a": [first], "route_b": [second]},
            FusionConfig(output_k=2, near_duplicate_threshold=None, mmr_lambda=1.0),
        )

    old_index = rebuilt("0", "f")
    new_index = rebuilt("f", "0")

    assert [item.candidate.content for item in old_index.ranked] == [
        item.candidate.content for item in new_index.ranked
    ]
    assert {
        item.candidate.content: item.candidate.evidence_id for item in old_index.ranked
    } != {
        item.candidate.content: item.candidate.evidence_id for item in new_index.ranked
    }


def test_rrf_tie_break_uses_logical_block_occurrence_when_content_and_page_collide():
    content = "同一页重复出现的完全相同注意事项。"

    def candidate(marker: str, block_number: int, route: str) -> RetrievalCandidate:
        block_id = f"block-{block_number:08d}"
        return RetrievalCandidate(
            evidence_id=f"ev_{marker * 40}",
            source_type="local",
            tenant_id="tenant-a",
            kb_id="kb-a",
            document_id="manual",
            document_version_id=f"version-{marker}",
            index_version_id=f"index-{marker}",
            chunk_id=f"chunk-{marker}",
            content=content,
            page_no=10,
            source_block_ids=(block_id,),
            provenance_spans=(
                {
                    "block_id": block_id,
                    "availability": "exact",
                    "coordinate_space": "none",
                    "page_no": 10,
                },
            ),
            observations=(RouteObservation(route_id=route, route_rank=1),),
        )

    def rebuilt(first_marker: str, second_marker: str):
        return fuse_route_results(
            {
                "route_a": [candidate(first_marker, 12, "route_a")],
                "route_b": [candidate(second_marker, 48, "route_b")],
            },
            FusionConfig(output_k=2, near_duplicate_threshold=None, mmr_lambda=1.0),
        )

    old_index = rebuilt("0", "f")
    new_index = rebuilt("f", "0")

    assert [item.candidate.source_block_ids for item in old_index.ranked] == [
        item.candidate.source_block_ids for item in new_index.ranked
    ]
    assert [item.candidate.evidence_id for item in old_index.ranked] != [
        item.candidate.evidence_id for item in new_index.ranked
    ]


def test_subquery_coverage_bonus_is_bounded_and_duplicate_query_cannot_inflate_score():
    duplicate_query = {
        **_local(
            marker="a",
            document_id="manual",
            content="同一个子查询被重复返回。",
            score=0.9,
        ),
        "retrieval_observations": [
            {"route": "base.hybrid", "sub_query_id": "sq_one", "rank": rank}
            for rank in (1, 2, 3)
        ],
    }
    multi_query = {
        **_local(
            marker="b",
            document_id="manual",
            content="一条证据同时覆盖两个独立子查询。",
            score=0.9,
        ),
        "retrieval_observations": [
            {"route": "base.hybrid", "sub_query_id": "sq_one", "rank": 1},
            {"route": "base.hybrid", "sub_query_id": "sq_two", "rank": 2},
        ],
    }
    many_queries = {
        **_local(
            marker="c",
            document_id="manual",
            content="很多独立子查询也只能得到有界奖励。",
            score=0.9,
        ),
        "retrieval_observations": [
            {
                "route": "base.hybrid",
                "sub_query_id": f"sq_{index}",
                "rank": index,
            }
            for index in range(1, 21)
        ],
    }
    config = FusionConfig(
        output_k=3,
        near_duplicate_threshold=None,
        mmr_lambda=1.0,
        subquery_coverage_bonus_weight=0.1,
        subquery_coverage_bonus_cap=0.25,
    )

    result = fuse_route_results(
        {"base": [duplicate_query, multi_query, many_queries]},
        config,
        approved_subquery_fingerprints={
            "sq_one": normalized_query_fingerprint("第一个检索问题"),
            "sq_two": normalized_query_fingerprint("第二个检索问题"),
            **{
                f"sq_{index}": normalized_query_fingerprint(f"独立问题 {index}")
                for index in range(1, 21)
            },
        },
    )
    scores = {
        item.candidate.evidence_id: item.fusion_score for item in result.ranked
    }
    base_score = 1 / 61

    assert scores[duplicate_query["evidence_id"]] == pytest.approx(base_score)
    assert scores[multi_query["evidence_id"]] > base_score
    assert scores[many_queries["evidence_id"]] == pytest.approx(base_score * 1.25)


def test_subquery_bonus_rejects_unapproved_ids_and_deduplicates_normalized_queries():
    candidate = {
        **_local(
            marker="d",
            document_id="manual",
            content="别名子查询不能重复领取覆盖奖励。",
            score=0.9,
        ),
        "retrieval_observations": [
            {"route": "base.hybrid", "sub_query_id": "sq_a", "rank": 1},
            {"route": "base.hybrid", "sub_query_id": "sq_alias", "rank": 1},
            {"route": "base.hybrid", "sub_query_id": "sq_unapproved", "rank": 1},
        ],
    }
    result = fuse_route_results(
        {"base": [candidate]},
        FusionConfig(
            output_k=1,
            near_duplicate_threshold=None,
            subquery_coverage_bonus_weight=0.5,
            subquery_coverage_bonus_cap=0.5,
        ),
        approved_subquery_fingerprints={
            "sq_a": normalized_query_fingerprint("A4 打印速度？"),
            "sq_alias": normalized_query_fingerprint("a4打印速度"),
        },
    )

    assert result.ranked[0].fusion_score == pytest.approx(1 / 61)


def test_query_graph_legacy_profile_has_no_bonus_and_explicit_profile_enables_it():
    content = "该证据覆盖打印速度与首页输出时间两个字段。"

    def state(weight: float | None) -> dict:
        candidate = _local(
            marker="e",
            document_id="manual",
            content=content,
            score=0.9,
        )
        first = {
            **candidate,
            "retrieval_observations": [
                {"route": "base.hybrid", "sub_query_id": "sq_speed", "rank": 1}
            ],
        }
        second = {
            **candidate,
            "retrieval_observations": [
                {"route": "base.hybrid", "sub_query_id": "sq_first_page", "rank": 2}
            ],
        }
        fusion = {"near_duplicate_threshold": None}
        if weight is not None:
            fusion.update(
                {
                    "subquery_coverage_bonus_weight": weight,
                    "subquery_coverage_bonus_cap": 0.25,
                }
            )
        return {
            "embedding_chunks": [first, second],
            "hyde_embedding_chunks": [],
            "retrieval_profile": {"candidate_k": 2, "fusion": fusion},
            "query_plan": {
                "sub_queries": [
                    {
                        "sub_query_id": "sq_speed",
                        "query": "A4 打印速度是多少？",
                        "purpose": "retrieval",
                    },
                    {
                        "sub_query_id": "sq_first_page",
                        "query": "首页输出时间是多少？",
                        "purpose": "retrieval",
                    },
                ],
                "routes": [
                    {
                        "route_type": "local_hybrid",
                        "sub_query_ids": ["sq_speed", "sq_first_page"],
                    }
                ],
            },
        }

    legacy = fuse_by_rrf(state(None))
    explicit = fuse_by_rrf(state(0.1))

    assert legacy[0]["fusion_score"] == pytest.approx(1 / 61)
    assert explicit[0]["fusion_score"] > legacy[0]["fusion_score"]


def test_typed_candidate_accepts_rank_only_observation_and_preserves_evidence_identity():
    candidate = RetrievalCandidate(
        evidence_id="ev_" + "d" * 40,
        source_type="local",
        tenant_id="tenant-a",
        kb_id="kb-a",
        document_id="doc-d",
        document_version_id="dver-d",
        index_version_id="idx-d",
        chunk_id="chunk-d",
        content="只有路线排名、没有 raw score 的候选仍可进入 RRF。",
        observations=(
            RouteObservation(
                route_id="toc",
                sub_query_id="section-safety",
                route_rank=2,
            ),
        ),
    )

    result = fuse_route_results({"toc": [candidate]})

    assert result.ranked[0].candidate.evidence_id == candidate.evidence_id
    assert result.ranked[0].fusion_score == pytest.approx(1 / 62)
    assert result.diagnostics.routes[0].raw_score_count == 0
    assert result.diagnostics.routes[0].calibration_fallback_count == 0


def test_query_graph_fusion_preserves_exact_subquery_coverage_per_candidate():
    install = {
        **_local(
            marker="a",
            document_id="doc-a",
            content="首次安装打印头时，不要触摸芯片和喷孔。",
            score=0.91,
        ),
        "retrieval_observations": [
            {
                "route": "base.hybrid",
                "sub_query_id": "sq_install",
                "rank": 1,
                "raw_score": 0.91,
            }
        ],
    }
    calibration = {
        **_local(
            marker="b",
            document_id="doc-b",
            content="首次校准必须使用未打印过的普通 A4 纸。",
            score=0.89,
        ),
        "retrieval_observations": [
            {
                "route": "base.hybrid",
                "sub_query_id": "sq_calibration",
                "rank": 1,
                "raw_score": 0.89,
            }
        ],
    }
    state = {
        "embedding_chunks": [install, calibration],
        "hyde_embedding_chunks": [],
        "retrieval_profile": {
            "candidate_k": 2,
            "fusion": {"near_duplicate_threshold": None},
        },
        "query_plan": {
            "routes": [
                {
                    "route_type": "local_hybrid",
                    "sub_query_ids": ["sq_install", "sq_calibration"],
                }
            ]
        },
    }

    fused = fuse_by_rrf(state)

    coverage = {item["chunk_id"]: item["sub_query_ids"] for item in fused}
    assert coverage == {
        "chunk-a": ["sq_install"],
        "chunk-b": ["sq_calibration"],
    }


def test_legacy_hydrated_evidence_keeps_page_bbox_and_source_identity():
    content = "带页码坐标的 HAK180 块级证据。"
    raw = {
        "type": "milvus",
        "score": 0.88,
        "evidence": {
            "evidence_id": "ev_" + "e" * 40,
            "content": content,
            "source_label": "hak180产品安全手册.pdf",
            "source": {
                "source_type": "local",
                "tenant_id": "tenant-a",
                "kb_id": "kb-a",
                "document_id": "doc-e",
                "document_version_id": "dver-e",
                "index_version_id": "idx-e",
                "chunk_id": "chunk-e",
                "source_block_ids": ["block-e"],
            },
            "provenance_spans": [
                {
                    "block_id": "block-e",
                    "availability": "exact",
                    "coordinate_space": "normalized_page",
                    "page_no": 3,
                    "bbox": [0.1, 0.2, 0.8, 0.9],
                }
            ],
            "heading_path": ["安全事项", "使用前"],
            "asset_refs": ["asset-e"],
        },
    }

    result = fuse_route_results({"base.hybrid": [raw]})
    candidate = result.ranked[0].candidate

    assert candidate.source_type == "local"
    assert candidate.evidence_id == raw["evidence"]["evidence_id"]
    assert candidate.document_version_id == "dver-e"
    assert candidate.index_version_id == "idx-e"
    assert candidate.page_no == 3
    assert candidate.bbox == pytest.approx((0.1, 0.2, 0.8, 0.9))
    assert candidate.source_block_ids == ("block-e",)
    assert candidate.provenance_spans[0]["block_id"] == "block-e"
    assert candidate.heading_path == ("安全事项", "使用前")
    assert candidate.asset_refs == ("asset-e",)
    legacy = fusion_result_to_mappings(result)[0]
    assert legacy["evidence"] == raw["evidence"]
    assert legacy["type"] == "milvus"
    assert legacy["score"] == result.ranked[0].fusion_score
    assert legacy["retrieval_observations"][0]["raw_scores"]["score"] == 0.88
    assert legacy["retrieval_observations"][0]["raw_score"] == 0.88


def test_invalid_fusion_configuration_fails_before_candidate_processing():
    with pytest.raises(ValidationError, match="positive route weight"):
        FusionConfig(default_route_weight=0)

    with pytest.raises(ValidationError, match="output_k"):
        FusionConfig(output_k=3, max_input_candidates=2)

    with pytest.raises(ValidationError, match="range_max"):
        ScoreSemantics(range_min=1.0, range_max=1.0)
