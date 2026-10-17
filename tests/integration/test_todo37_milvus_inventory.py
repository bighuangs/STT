"""TODO-37 opt-in real Milvus read-only inventory contract.

Set STT_RUN_EXTERNAL_MILVUS_TESTS=1 when the local Docker dependencies are
available. The adapter surface intentionally contains no mutating method.
"""

from __future__ import annotations

import os

import pytest

from app.application.vector_topology_service import VectorTopologyService
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.vectorstore.milvus_topology_inspector import (
    MilvusReadOnlyTopologyInspector,
)


pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        os.getenv("STT_RUN_EXTERNAL_MILVUS_TESTS") != "1",
        reason="set STT_RUN_EXTERNAL_MILVUS_TESTS=1 for read-only live Milvus audit",
    ),
]


def test_live_chunks_collection_inventory_is_identity_only_and_non_mutating():
    client = milvus_gateway.client()
    if client is None:
        pytest.skip("configured Milvus dependency is unavailable")
    inspector = MilvusReadOnlyTopologyInspector(client)
    report = VectorTopologyService.audit_collection(
        inspector,
        physical_database="default",
        physical_collection=milvus_gateway.chunks_collection,
        sample_limit=200,
    )

    assert report.dry_run is True
    assert report.destructive_actions == ()
    assert report.entity_count >= report.sampled_count >= 0
    assert report.partition_count >= 1
    assert report.index_count >= 1
