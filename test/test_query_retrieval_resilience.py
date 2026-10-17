import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from app.process.query.agent.nodes import node_search_embedding_hyde as hyde_node_module
from app.rag.query.hyde_search_service import build_hyde_query
from app.shared.model import embedding_utils
from app.shared.runtime.load_prompt import load_prompt


class EmbeddingConcurrencyTests(unittest.TestCase):
    def test_bge_m3_lazy_singleton_is_initialized_once(self):
        created = []

        class FakeEmbeddingFunction:
            def __init__(self, **kwargs):
                time.sleep(0.02)
                created.append(kwargs)

        original_model = embedding_utils._bge_m3_ef
        try:
            embedding_utils._bge_m3_ef = None
            with patch.object(
                embedding_utils,
                "BGEM3EmbeddingFunction",
                FakeEmbeddingFunction,
            ):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    models = list(
                        executor.map(
                            lambda _: embedding_utils.get_bge_m3_ef(),
                            range(8),
                        )
                    )
        finally:
            embedding_utils._bge_m3_ef = original_model

        self.assertEqual(len(created), 1)
        self.assertEqual(len({id(model) for model in models}), 1)

    def test_embedding_encode_is_serialized(self):
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        class FakeSparse:
            indices = np.array([], dtype=np.int64)
            data = np.array([], dtype=np.float32)
            indptr = np.array([0, 0], dtype=np.int64)

        class FakeDense:
            def tolist(self):
                return [1.0, 0.0]

        class FakeModel:
            def encode_documents(self, texts):
                nonlocal active, max_active
                with counter_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with counter_lock:
                    active -= 1
                return {"dense": [FakeDense()], "sparse": FakeSparse()}

        original_model = embedding_utils._bge_m3_ef
        try:
            embedding_utils._bge_m3_ef = FakeModel()
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(
                    executor.map(
                        lambda i: embedding_utils.generate_embeddings([f"q-{i}"]),
                        range(6),
                    )
                )
        finally:
            embedding_utils._bge_m3_ef = original_model

        self.assertEqual(max_active, 1)
        self.assertTrue(
            all(result["dense"] == [[1.0, 0.0]] for result in results)
        )


class HyDEResilienceTests(unittest.TestCase):
    def test_hyde_prompt_renders_real_query(self):
        query = "HAK 180 烫金机使用什么电源规格？"

        prompt = load_prompt("hyde_prompt", rewritten_query=query)

        self.assertIn(query, prompt)
        self.assertNotIn("{{", prompt)
        self.assertNotIn("rewritten_query", prompt)

    def test_invalid_hyde_answer_falls_back_to_original_query(self):
        query = "HAK 180 烫金机使用什么电源规格？"

        actual = build_hyde_query(
            query,
            "请提供您想要查询的内容，我将为您重新整理并优化查询语句。",
        )

        self.assertEqual(actual, query)

    def test_hyde_node_degrades_to_empty_results(self):
        with (
            patch.object(
                hyde_node_module,
                "search_embedding_hyde",
                side_effect=NotImplementedError("meta tensor"),
            ),
            patch.object(hyde_node_module, "add_running_task"),
            patch.object(hyde_node_module, "add_done_task"),
        ):
            result = hyde_node_module.node_search_embedding_hyde(
                {"session_id": "test-hyde-degrade", "is_stream": False}
            )

        self.assertEqual(result, {"hyde_embedding_chunks": []})


class AnswerPromptTests(unittest.TestCase):
    def test_answer_prompt_renders_context_history_and_question(self):
        prompt = load_prompt(
            "answer_out",
            context="HAK 180 额定电源规格参考内容",
            history="用户之前询问了 HAK 180",
            question="HAK 180 烫金机使用什么电源规格？",
        )

        self.assertIn("HAK 180 额定电源规格参考内容", prompt)
        self.assertIn("用户之前询问了 HAK 180", prompt)
        self.assertIn("HAK 180 烫金机使用什么电源规格？", prompt)
        self.assertNotIn("{{", prompt)


if __name__ == "__main__":
    unittest.main()
