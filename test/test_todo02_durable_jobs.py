"""TODO-02 持久队列、幂等、租约与检查点回归测试（仅依赖 unittest）。"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.application import control_plane_service as control
from app.application import ingestion_job_service as jobs
from app.application.checkpoint_store import CheckpointIntegrityError, LocalCheckpointStore
from app.application.import_runner import (
    _next_running_stage,
    _stream_with_ingestion_progress,
    prepare_checkpoint_state,
)
from app.application.ingestion_progress_hooks import (
    IngestionProgressHook,
    bind_ingestion_progress_hook,
    notify_ingestion_embedding_batch_progress,
)
from app.application.job_errors import (
    DependencyTransientError,
    RateLimitJobError,
    ValidationJobError,
    classify_job_error,
)
from app.infra.persistence.sql.database import configure_database, get_engine, session_scope
from app.infra.persistence.sql.job_repository import JobCancelledError, LeaseLostError
from app.infra.persistence.sql.models import IngestionJobModel
from app.infra.persistence.sql.repositories import ResourceConflictError


class ClosedImageSentinel:
    """模拟 PIL.Image.close() 后 pickle 调用 __getstate__ 失败。"""

    def __getstate__(self):
        raise ValueError("Operation on closed image")


class DurableJobTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        configure_database(f"sqlite:///{(self.root / 'control.db').as_posix()}")
        control.bootstrap_control_plane()

    def tearDown(self):
        get_engine().dispose()
        self.temp.cleanup()

    def register(self, name: str = "manual.md", key: str | None = None) -> dict:
        source = self.root / name
        source.write_text("# Durable queue\ncheckpoint and retry", encoding="utf-8")
        return control.register_uploaded_document(
            kb_id="kb_default",
            logical_name=name,
            content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            size_bytes=source.stat().st_size,
            mime_type="text/markdown",
            object_key=str(source),
            idempotency_key=key,
        )

    def test_only_one_worker_can_claim_and_stale_lease_is_fenced(self):
        job_id = self.register()["ingestion_job"]["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda worker: jobs.claim_next_job(worker, 30), ["w1", "w2"]))
        winners = [claim for claim in claims if claim]
        self.assertEqual(1, len(winners))
        first = winners[0]

        with session_scope() as session:
            row = session.get(IngestionJobModel, job_id)
            row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        recovered = jobs.claim_next_job("w3", 30)
        self.assertEqual(job_id, recovered["id"])
        self.assertEqual(1, recovered["recovered_count"])
        with self.assertRaises(LeaseLostError):
            jobs.mark_job_running(job_id, first["lease_token"])

    def test_retry_then_reclaim_increments_run_attempt(self):
        job_id = self.register()["ingestion_job"]["id"]
        first = jobs.claim_next_job("retry-worker", 30)
        jobs.mark_job_running(job_id, first["lease_token"])
        waiting = jobs.schedule_job_retry(
            job_id,
            first["lease_token"],
            error_category="dependency_transient",
            error_code="DEPENDENCY_TRANSIENT",
            safe_message="temporary dependency outage",
            delay_seconds=0,
        )
        self.assertEqual("retry_wait", waiting["status"])
        second = jobs.claim_next_job("retry-worker-2", 30)
        self.assertEqual(2, second["run_attempt"])
        self.assertNotEqual(first["lease_token"], second["lease_token"])
        event_types = [event["event_type"] for event in jobs.list_job_events(job_id)]
        self.assertIn("retry_scheduled", event_types)

    def test_cancel_queued_and_running_are_persistent(self):
        queued_id = self.register("queued.md")["ingestion_job"]["id"]
        self.assertEqual("cancelled", jobs.cancel_job(queued_id)["status"])

        running_id = self.register("running.md")["ingestion_job"]["id"]
        claimed = jobs.claim_next_job("cancel-worker", 30)
        self.assertEqual(running_id, claimed["id"])
        jobs.mark_job_running(running_id, claimed["lease_token"])
        requested = jobs.cancel_job(running_id)
        self.assertEqual("running", requested["status"])
        self.assertIsNotNone(requested["cancel_requested_at"])
        cancelled = jobs.cancel_claimed_job(running_id, claimed["lease_token"], "safe point")
        self.assertEqual("cancelled", cancelled["status"])

    def test_checkpoint_artifact_hash_and_event_reconnect(self):
        registered = self.register()
        job_id = registered["ingestion_job"]["id"]
        claimed = jobs.claim_next_job("checkpoint-worker", 30)
        jobs.mark_job_running(job_id, claimed["lease_token"])
        store = LocalCheckpointStore(self.root / "checkpoints")
        state = {"chunks": [{"content": "hello"}], "resume_node": "node_bge_embedding"}
        artifact = store.save(job_id, 1, "llm_enhance_reducer", state)
        metadata = jobs.record_job_checkpoint(
            job_id,
            claimed["lease_token"],
            stage="llm_enhance_reducer",
            resume_node="node_bge_embedding",
            input_sha256=registered["document_version"]["content_sha256"],
            config_sha256=hashlib.sha256(b"{}").hexdigest(),
            **artifact,
        )
        self.assertEqual(state, store.load(metadata))
        first_page = jobs.list_job_events(job_id, 0, 2)
        remainder = jobs.list_job_events(job_id, first_page[-1]["sequence"])
        self.assertGreaterEqual(len(remainder), 2)
        self.assertEqual(
            list(range(1, len(first_page + remainder) + 1)),
            [event["sequence"] for event in first_page + remainder],
        )
        target = Path(metadata["artifact_uri"])
        target.write_bytes(target.read_bytes() + b"tampered")
        with self.assertRaises(CheckpointIntegrityError):
            store.load(metadata)

    def test_stage_started_projects_running_node_without_claiming_completion(self):
        job_id = self.register("stage-started.md")["ingestion_job"]["id"]
        claimed = jobs.claim_next_job("stage-worker", 30)
        jobs.mark_job_running(job_id, claimed["lease_token"])

        jobs.update_job_stage(
            job_id,
            claimed["lease_token"],
            "llm_enhance_reducer",
            0.72,
        )
        running = jobs.start_job_stage(
            job_id,
            claimed["lease_token"],
            "node_bge_embedding",
            0.72,
        )

        self.assertEqual("running", running["status"])
        self.assertEqual("node_bge_embedding", running["stage"])
        self.assertEqual(0.72, running["progress"])
        events = jobs.list_job_events(job_id)
        self.assertEqual("stage_started", events[-1]["event_type"])
        self.assertEqual("node_bge_embedding", events[-1]["stage"])
        self.assertEqual("started", events[-1]["payload"]["phase"])
        completed = [
            event["stage"]
            for event in events
            if event["event_type"] == "stage_completed"
        ]
        self.assertEqual(["llm_enhance_reducer"], completed)

    def test_embedding_batch_progress_is_monotonic_throttled_and_final(self):
        job_id = self.register("embedding-progress.md")["ingestion_job"]["id"]
        claimed = jobs.claim_next_job("embedding-worker", 30)
        jobs.mark_job_running(job_id, claimed["lease_token"])
        jobs.update_job_stage(
            job_id,
            claimed["lease_token"],
            "llm_enhance_reducer",
            0.72,
        )
        hook = IngestionProgressHook(job_id, claimed["lease_token"])

        with bind_ingestion_progress_hook(hook):
            notify_ingestion_embedding_batch_progress(5, 218, 0)
            notify_ingestion_embedding_batch_progress(20, 218, 0)
            notify_ingestion_embedding_batch_progress(21, 218, 0)
            notify_ingestion_embedding_batch_progress(218, 218, 1)

        batch_events = [
            event
            for event in jobs.list_job_events(job_id)
            if event["event_type"] == "stage_started"
            and event["payload"].get("phase") == "batch"
        ]
        self.assertEqual(2, len(batch_events))
        self.assertLess(batch_events[-1]["progress"], 0.86)
        self.assertGreater(batch_events[-1]["progress"], batch_events[0]["progress"])
        self.assertEqual(218, batch_events[-1]["payload"]["completed"])
        self.assertEqual(218, batch_events[-1]["payload"]["total"])
        self.assertEqual(1, batch_events[-1]["payload"]["failed_batches"])

        # ContextVar 在离开 bind 后必须复位，不能把下一任务的回调写入旧 Job。
        count_after_bind = len(jobs.list_job_events(job_id))
        notify_ingestion_embedding_batch_progress(218, 218, 0)
        self.assertEqual(count_after_bind, len(jobs.list_job_events(job_id)))

    def test_embedding_batch_progress_respects_lease_and_cancel_fences(self):
        stale_id = self.register("embedding-stale-lease.md")["ingestion_job"]["id"]
        stale_claim = jobs.claim_next_job("embedding-stale-worker", 30)
        self.assertEqual(stale_id, stale_claim["id"])
        jobs.mark_job_running(stale_id, stale_claim["lease_token"])
        stale_hook = IngestionProgressHook(stale_id, "not-the-current-lease")
        with self.assertRaises(LeaseLostError):
            stale_hook.embedding_batch_progress(20, 218)

        cancelled_id = self.register("embedding-cancelled.md")["ingestion_job"]["id"]
        jobs.cancel_claimed_job(stale_id, stale_claim["lease_token"], "next fixture")
        cancelled_claim = jobs.claim_next_job("embedding-cancel-worker", 30)
        self.assertEqual(cancelled_id, cancelled_claim["id"])
        jobs.mark_job_running(cancelled_id, cancelled_claim["lease_token"])
        jobs.cancel_job(cancelled_id)
        cancelled_hook = IngestionProgressHook(
            cancelled_id,
            cancelled_claim["lease_token"],
        )
        with self.assertRaises(JobCancelledError):
            cancelled_hook.embedding_batch_progress(20, 218)

    def test_graph_progress_binding_resets_after_stream_failure(self):
        job_id = self.register("embedding-stream-reset.md")["ingestion_job"]["id"]
        claimed = jobs.claim_next_job("embedding-stream-worker", 30)
        jobs.mark_job_running(job_id, claimed["lease_token"])
        hook = IngestionProgressHook(job_id, claimed["lease_token"])

        class ExplodingGraph:
            @staticmethod
            def stream(_state):
                notify_ingestion_embedding_batch_progress(20, 100)
                yield {"node_bge_embedding": {}}
                raise RuntimeError("synthetic graph failure")

        stream = _stream_with_ingestion_progress(ExplodingGraph(), {}, hook)
        self.assertEqual({"node_bge_embedding": {}}, next(stream))
        with self.assertRaises(RuntimeError):
            next(stream)
        count_after_failure = len(jobs.list_job_events(job_id))
        notify_ingestion_embedding_batch_progress(100, 100)
        self.assertEqual(count_after_failure, len(jobs.list_job_events(job_id)))

    def test_next_running_stage_only_predicts_deterministic_graph_edges(self):
        self.assertEqual(
            "node_pdf_to_md",
            _next_running_stage("node_entry", {"is_document_read_enabled": True}),
        )
        self.assertEqual(
            "node_bge_embedding",
            _next_running_stage("llm_enhance_reducer", {}),
        )
        self.assertEqual(
            "node_import_milvus",
            _next_running_stage("node_bge_embedding", {}),
        )
        self.assertIsNone(_next_running_stage("node_llm_fanout_start", {}))
        self.assertIsNone(_next_running_stage("node_import_milvus", {}))

    def test_checkpoint_after_image_upload_drops_closed_pil_objects(self):
        state = {
            "chunks": [{"content": "image text", "img_id": "img-1"}],
            "image_pil_dict": {"image.png": ClosedImageSentinel()},
        }
        prepared = prepare_checkpoint_state(
            "node_image_upload", state, "node_llm_fanout_start"
        )
        self.assertEqual({}, prepared["image_pil_dict"])
        self.assertEqual("node_llm_fanout_start", prepared["resume_node"])
        self.assertIn("image.png", state["image_pil_dict"])

        store = LocalCheckpointStore(self.root / "checkpoints")
        artifact = store.save("job_closed_image", 1, "node_image_upload", prepared)
        loaded = store.load({**artifact, "job_id": "job_closed_image", "sequence": 1})
        self.assertEqual({}, loaded["image_pil_dict"])

    def test_one_hundred_same_idempotency_requests_reuse_one_job(self):
        source = self.root / "idempotent.md"
        source.write_text("same request", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()

        def submit(_index: int) -> str:
            result = control.register_uploaded_document(
                kb_id="kb_default",
                logical_name=source.name,
                content_sha256=digest,
                size_bytes=source.stat().st_size,
                mime_type="text/markdown",
                object_key=str(source),
                idempotency_key="same-request-100",
            )
            return result["ingestion_job"]["id"]

        with ThreadPoolExecutor(max_workers=16) as pool:
            job_ids = list(pool.map(submit, range(100)))
        self.assertEqual(1, len(set(job_ids)))
        self.assertEqual(1, len(jobs.list_jobs(kb_id="kb_default")))

    def test_five_failure_injections_have_stable_categories(self):
        cases = [
            (ValidationJobError("bad file"), "validation", False),
            (DependencyTransientError("milvus temporarily unavailable"), "dependency_transient", True),
            (RateLimitJobError("429"), "rate_limit", True),
            (MemoryError("out of memory"), "resource_exhausted", True),
            (RuntimeError("unexpected parser crash"), "unknown", True),
        ]
        for error, category, retryable in cases:
            with self.subTest(category=category):
                classified = classify_job_error(error)
                self.assertEqual(category, classified.category)
                self.assertEqual(retryable, classified.retryable)

    def test_idempotency_key_is_bound_even_when_content_was_already_deduplicated(self):
        original = self.register("already-there.md")
        replay = self.register("already-there.md", key="dedup-bound-key")
        self.assertEqual(
            original["ingestion_job"]["id"], replay["ingestion_job"]["id"]
        )
        other = self.root / "different.md"
        other.write_text("different content", encoding="utf-8")
        with self.assertRaises(ResourceConflictError):
            control.register_uploaded_document(
                kb_id="kb_default",
                logical_name=other.name,
                content_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
                size_bytes=other.stat().st_size,
                mime_type="text/markdown",
                object_key=str(other),
                idempotency_key="dedup-bound-key",
            )


if __name__ == "__main__":
    unittest.main()
