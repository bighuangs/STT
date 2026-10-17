"""导入向量批次进度回调的纯单元回归。"""

from __future__ import annotations

import numpy as np

from app.rag.import_ import embedding_service


def _embedding_result(texts: list[str]) -> dict:
    return {
        "dense": [np.asarray([1.0, 0.0, 0.0], dtype=np.float32) for _ in texts],
        "sparse": [{0: 1.0} for _ in texts],
    }


def test_embed_chunks_reports_each_completed_batch(monkeypatch):
    monkeypatch.setattr(
        embedding_service.llm_provider,
        "embed_documents",
        _embedding_result,
    )
    observed: list[tuple[int, int, int]] = []
    chunks = [{"content": f"chunk-{index}"} for index in range(12)]

    result = embedding_service.embed_chunks(
        chunks,
        file_title="manual",
        batch_size=5,
        progress_callback=lambda completed, total, failed: observed.append(
            (completed, total, failed)
        ),
    )

    assert observed == [(5, 12, 0), (10, 12, 0), (12, 12, 0)]
    assert all(len(chunk["dense_vector"]) == 3 for chunk in result)
    assert all(chunk["sparse_vector"] for chunk in result)


def test_embed_chunks_failed_batch_still_advances_progress(monkeypatch):
    calls = 0

    def fake_embed(texts: list[str]) -> dict:
        nonlocal calls
        calls += 1
        # 第一次调用编码标题，第三次调用是第二个内容批次。
        if calls == 3:
            raise RuntimeError("synthetic batch failure")
        return _embedding_result(texts)

    monkeypatch.setattr(
        embedding_service.llm_provider,
        "embed_documents",
        fake_embed,
    )
    observed: list[tuple[int, int, int]] = []
    chunks = [{"content": f"chunk-{index}"} for index in range(12)]

    result = embedding_service.embed_chunks(
        chunks,
        file_title="manual",
        batch_size=5,
        progress_callback=lambda completed, total, failed: observed.append(
            (completed, total, failed)
        ),
    )

    assert observed == [(5, 12, 0), (10, 12, 1), (12, 12, 1)]
    assert all("dense_vector" in chunk for chunk in result[:5])
    assert all("dense_vector" not in chunk for chunk in result[5:10])
    assert all("dense_vector" in chunk for chunk in result[10:])
