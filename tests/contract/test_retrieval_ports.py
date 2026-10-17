from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from app.application.retrieval_ports import RetrievalPort, RerankerPort
from app.domain.retrieval import (
    DenseSearchConfig,
    FilterPredicate,
    RerankRequest,
    RetrievalEngineError,
    RetrievalHit,
    RetrievalProfile,
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RouteOutcome,
    RouteScore,
)
from app.infra.retrieval.milvus_adapter import (
    ActiveIndexScope,
    MilvusRetrievalAdapter,
)
from app.infra.retrieval.reranker_adapter import BGERerankerAdapter


def _profile(
    *,
    routes: tuple[str, ...] = ("hybrid",),
    candidate_k: int = 2,
    top_k: int = 2,
) -> RetrievalProfile:
    return RetrievalProfile(
        tenant_id="tenant-a",
        routes=routes,
        candidate_k=candidate_k,
        top_k=top_k,
        dense=DenseSearchConfig(dimension=4),
    )


def _request(profile: RetrievalProfile | None = None, **updates: Any) -> RetrievalRequest:
    values: dict[str, Any] = {
        "query": "压力如何测量",
        "variants": ("压力测量方法",),
        "scope": RetrievalScope(
            tenant_id="tenant-a",
            user_id="user-a",
            kb_ids=("kb-a",),
        ),
        "profile": profile or _profile(),
        "trace_id": "trace-contract",
    }
    values.update(updates)
    return RetrievalRequest(**values)


def _hit(chunk_id: str, *, score: float = 0.9, route: str = "hybrid") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        evidence_id=f"ev_{'1' * 40}",
        tenant_id="tenant-a",
        kb_id="kb-a",
        document_id=f"doc-{chunk_id[-1]}",
        document_version_id=f"dv-{chunk_id[-1]}",
        index_version_id=f"iv-{chunk_id[-1]}",
        content=f"{chunk_id} 的正文",
        scores=(RouteScore(route=route, raw_score=score, rank=1),),
        final_score=score,
    )


class FakeRetrievalAdapter:
    """纯内存 Fake 与生产端口返回同一个稳定契约。"""

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        hit = _hit("chunk-1")
        return RetrievalResult(
            trace_id=request.trace_id,
            profile_id=request.profile.profile_id,
            profile_version=request.profile.version,
            engine="fake",
            hits=(hit,),
            route_outcomes=(
                RouteOutcome(
                    route="hybrid",
                    status="succeeded",
                    candidate_count=1,
                ),
            ),
        )


class FakeEmbedding:
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> dict[str, list[Any]]:
        self.calls.append(texts)
        return {
            "dense": [[0.5] * self.dimension for _ in texts],
            "sparse": [{1: 0.8} for _ in texts],
        }


class AdvancingEmbedding(FakeEmbedding):
    def __init__(self, advance, dimension: int = 4):
        super().__init__(dimension=dimension)
        self._advance = advance

    def embed_documents(self, texts: list[str]) -> dict[str, list[Any]]:
        result = super().embed_documents(texts)
        self._advance()
        return result


class FakeScopeResolver:
    def resolve(self, scope: RetrievalScope) -> list[ActiveIndexScope]:
        return [
            ActiveIndexScope(
                tenant_id=scope.tenant_id,
                kb_id="kb-a",
                document_id="doc-1",
                document_version_id="dv-1",
                index_version_id="iv-1",
                collection_name="chunks-a",
                partition_name="tenant-a-bucket",
            ),
            ActiveIndexScope(
                tenant_id=scope.tenant_id,
                kb_id="kb-a",
                document_id="doc-2",
                document_version_id="dv-2",
                index_version_id="iv-2",
                collection_name="chunks-a",
                partition_name="tenant-a-bucket",
            ),
        ]


class FakeFilterResolver:
    def __init__(self, document_ids: tuple[str, ...]):
        self.document_ids = document_ids
        self.calls: list[Any] = []

    def resolve_document_ids(self, scope, filters):
        self.calls.append((scope, filters))
        return self.document_ids


class FakeMilvusGateway:
    def __init__(self):
        self.request_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    def create_requests(
        self,
        dense_vector,
        sparse_vector,
        *,
        expr=None,
        limit=5,
    ):
        self.request_calls.append(
            {
                "dense": dense_vector,
                "sparse": sparse_vector,
                "expr": expr,
                "limit": limit,
            }
        )
        return [{"kind": "dense", "expr": expr}, {"kind": "sparse", "expr": expr}]

    def hybrid_search(self, **kwargs):
        self.search_calls.append(kwargs)
        expr = kwargs["reqs"][0]["expr"]
        entities = []
        if '"doc-1"' in expr:
            entities.append(
                {
                    "id": 101,
                    "distance": 0.91,
                    "entity": {
                        "source_chunk_id": "chunk-1",
                        "evidence_id": f"ev_{'1' * 40}",
                        "content_with_weight": "压力测量步骤",
                        "tenant_id": "tenant-a",
                        "kb_id": "kb-a",
                        "doc_id": "doc-1",
                        "document_version_id": "dv-1",
                        "index_version_id": "iv-1",
                        "meta_fields": '{"department":"研发"}',
                    },
                }
            )
        if '"doc-2"' in expr:
            entities.append(
                {
                    "id": 102,
                    "distance": 0.82,
                    "entity": {
                        "source_chunk_id": "chunk-2",
                        "evidence_id": f"ev_{'2' * 40}",
                        "content_with_weight": "温度校准步骤",
                        "tenant_id": "tenant-a",
                        "kb_id": "kb-a",
                        "doc_id": "doc-2",
                        "document_version_id": "dv-2",
                        "index_version_id": "iv-2",
                    },
                }
            )
        return [entities[: kwargs["limit"]]]


class TimeoutMilvusGateway(FakeMilvusGateway):
    def hybrid_search(self, **kwargs):
        self.search_calls.append(kwargs)
        raise TimeoutError("deadline exceeded")


def _milvus_adapter(**updates: Any) -> MilvusRetrievalAdapter:
    dependencies: dict[str, Any] = {
        "embedding": FakeEmbedding(),
        "gateway": FakeMilvusGateway(),
        "scope_resolver": FakeScopeResolver(),
    }
    dependencies.update(updates)
    return MilvusRetrievalAdapter(**dependencies)


@pytest.mark.parametrize("adapter", [FakeRetrievalAdapter(), _milvus_adapter()])
def test_fake_and_milvus_adapters_share_the_retrieval_port_contract(adapter):
    assert isinstance(adapter, RetrievalPort)

    result = adapter.search(_request())

    assert result.trace_id == "trace-contract"
    assert result.profile_id == "retrieval-default"
    assert result.profile_version == 1
    assert 0 < len(result.hits) <= 2
    assert result.route_outcomes
    assert all(hit.scores for hit in result.hits)
    assert all(hit.final_score is not None for hit in result.hits)


def test_milvus_adapter_executes_configured_routes_and_preserves_route_scores():
    gateway = FakeMilvusGateway()
    adapter = _milvus_adapter(gateway=gateway)
    profile = _profile(
        routes=("dense", "sparse", "hybrid"),
        candidate_k=2,
        top_k=2,
    )

    result = adapter.search(_request(profile))

    assert [call["ranker_weights"] for call in gateway.search_calls] == [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.9, 0.1),
    ]
    assert all(
        call["partition_names"] == ["tenant-a-bucket"]
        for call in gateway.search_calls
    )
    assert {outcome.route for outcome in result.route_outcomes} == {
        "dense",
        "sparse",
        "hybrid",
    }
    first = result.hits[0]
    assert first.evidence_id == f"ev_{'1' * 40}"
    assert {score.route for score in first.scores} == {
        "dense",
        "sparse",
        "hybrid",
    }
    assert all(score.raw_score is not None for score in first.scores)
    assert all(score.rrf_score is not None for score in first.scores)


def test_milvus_adapter_keeps_candidate_window_for_outer_rrf_and_reranker():
    gateway = FakeMilvusGateway()
    adapter = _milvus_adapter(gateway=gateway)
    profile = _profile(candidate_k=2, top_k=1)

    result = adapter.search(_request(profile))

    assert gateway.request_calls[0]["limit"] == 2
    assert gateway.search_calls[0]["limit"] == 2
    assert len(result.hits) == 2
    assert profile.top_k == 1


def test_mandatory_scope_filter_is_server_built_and_metadata_filter_only_selects_docs():
    gateway = FakeMilvusGateway()
    filter_resolver = FakeFilterResolver(("doc-1",))
    adapter = _milvus_adapter(
        gateway=gateway,
        filter_resolver=filter_resolver,
    )
    filters = FilterPredicate(
        field="metadata.department",
        operator="eq",
        value="研发",
    )

    result = adapter.search(_request(filters=filters))

    assert [hit.document_id for hit in result.hits] == ["doc-1"]
    assert len(filter_resolver.calls) == 1
    expr = gateway.request_calls[0]["expr"]
    assert 'tenant_id == "tenant-a"' in expr
    assert 'kb_id == "kb-a"' in expr
    assert 'doc_id == "doc-1"' in expr
    assert 'document_version_id == "dv-1"' in expr
    assert 'index_version_id == "iv-1"' in expr
    assert "available_int == 1" in expr
    assert "department" not in expr
    assert "doc-2" not in expr


def test_milvus_adapter_rejects_filter_without_control_catalog_resolver():
    adapter = _milvus_adapter()
    with pytest.raises(RetrievalEngineError) as caught:
        adapter.search(
            _request(
                filters=FilterPredicate(
                    field="metadata.department",
                    operator="eq",
                    value="研发",
                )
            )
        )
    assert caught.value.code == "RETRIEVAL_FILTER_UNSUPPORTED"
    assert caught.value.retryable is False


def test_milvus_adapter_exposes_dimension_mismatch_as_typed_failure():
    adapter = _milvus_adapter(embedding=FakeEmbedding(dimension=3))
    with pytest.raises(RetrievalEngineError) as caught:
        adapter.search(_request())
    assert caught.value.code == "RETRIEVAL_DIMENSION_MISMATCH"
    assert caught.value.retryable is False
    assert caught.value.details == {"expected": 4, "actual": 3}


def test_milvus_adapter_rechecks_deadline_after_embedding_before_milvus() -> None:
    now = [0.0]
    gateway = FakeMilvusGateway()
    adapter = _milvus_adapter(
        embedding=AdvancingEmbedding(lambda: now.__setitem__(0, 31.0)),
        gateway=gateway,
        clock=lambda: now[0],
    )
    profile = _profile().model_copy(update={"timeout_ms": 30_000})

    with pytest.raises(RetrievalEngineError) as caught:
        adapter.search(_request(profile))

    assert caught.value.code == "RETRIEVAL_TIMEOUT"
    assert gateway.search_calls == []


def test_milvus_adapter_stops_after_embedding_when_route_was_cancelled() -> None:
    cancelled = [False]
    gateway = FakeMilvusGateway()
    adapter = _milvus_adapter(
        embedding=AdvancingEmbedding(lambda: cancelled.__setitem__(0, True)),
        gateway=gateway,
        cancellation=lambda: cancelled[0],
    )

    with pytest.raises(RetrievalEngineError) as caught:
        adapter.search(_request())

    assert caught.value.code == "RETRIEVAL_CANCELLED"
    assert gateway.search_calls == []


def test_milvus_adapter_passes_remaining_profile_budget_to_sdk() -> None:
    gateway = FakeMilvusGateway()
    adapter = _milvus_adapter(gateway=gateway, clock=lambda: 10.0)
    profile = _profile().model_copy(update={"timeout_ms": 90_000})

    adapter.search(_request(profile))

    assert gateway.search_calls[0]["timeout_seconds"] == pytest.approx(90.0)


def test_milvus_adapter_maps_sdk_deadline_to_typed_timeout() -> None:
    gateway = TimeoutMilvusGateway()
    adapter = _milvus_adapter(gateway=gateway)

    with pytest.raises(RetrievalEngineError) as caught:
        adapter.search(_request())

    assert caught.value.code == "RETRIEVAL_TIMEOUT"
    assert caught.value.retryable is True
    assert caught.value.route == "hybrid"


def test_milvus_adapter_does_not_import_pymilvus_in_the_domain_or_ports():
    repository = Path(__file__).resolve().parents[2]
    files = [
        repository / "app/domain/retrieval.py",
        repository / "app/application/retrieval_ports.py",
        repository / "app/infra/retrieval/milvus_adapter.py",
    ]
    imported_modules: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
    assert not any(name == "pymilvus" or name.startswith("pymilvus.") for name in imported_modules)


class FakeScoreModel:
    def compute_score(self, pairs, normalize=True):
        assert normalize is True
        return [0.2, 0.95]


def test_bge_adapter_satisfies_reranker_port_and_keeps_retrieval_observations():
    adapter = BGERerankerAdapter(model=FakeScoreModel())
    assert isinstance(adapter, RerankerPort)
    candidates = (_hit("chunk-1", score=0.8), _hit("chunk-2", score=0.7))

    result = adapter.rerank(
        RerankRequest(
            query="压力如何测量",
            candidates=candidates,
            config=_profile().reranker,
            trace_id="trace-rerank",
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["chunk-2"]
    assert {score.route for score in result.hits[0].scores} == {
        "hybrid",
        "reranker",
    }
    assert result.hits[0].final_score == pytest.approx(0.95)


class InvalidScoreModel:
    def compute_score(self, pairs, normalize=True):
        return [float("nan")]


def test_bge_adapter_turns_invalid_model_output_into_typed_failure():
    adapter = BGERerankerAdapter(model=InvalidScoreModel())
    with pytest.raises(RetrievalEngineError) as caught:
        adapter.rerank(
            RerankRequest(
                query="q",
                candidates=(_hit("chunk-1"), _hit("chunk-2")),
                config=_profile().reranker,
                trace_id="trace-invalid-reranker",
            )
        )
    assert caught.value.code == "RERANKER_INVALID_RESPONSE"
    assert caught.value.retryable is False
