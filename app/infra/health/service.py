"""TODO-05 依赖健康：并发、短超时、缓存、单飞及安全错误。"""
from __future__ import annotations

import copy
import re
import ssl
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import requests
import urllib3
from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from sqlalchemy import text

from app.infra.persistence.sql.database import dispose_database, get_engine
from app.shared.config.settings_config import AppSettings, get_settings


Probe = Callable[[], dict]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_error(exc: BaseException) -> tuple[str, str]:
    message = str(exc).replace("\r", " ").replace("\n", " ")[:240]
    message = re.sub(r"(?i)(mongodb(?:\+srv)?://)[^@/\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
    message = re.sub(
        r"(?i)((?:authorization|api[_-]?key|secret|password|token)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return type(exc).__name__.upper(), message or type(exc).__name__


def _timed_probe(probe: Probe) -> tuple[float, dict | None, BaseException | None]:
    started = time.perf_counter()
    try:
        result = probe()
        return round((time.perf_counter() - started) * 1000, 2), result, None
    except BaseException as exc:
        return round((time.perf_counter() - started) * 1000, 2), None, exc


class HealthManager:
    def __init__(self, settings: AppSettings | None = None, probes: dict[str, Probe] | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, dict]] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._last_success: dict[str, str] = {}
        self._failures: dict[str, int] = {}
        self._circuit_until: dict[str, float] = {}
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="health-probe")
        self._probes = probes or {
            "control_database": self._probe_control_database,
            "queue": self._probe_queue,
            "milvus": self._probe_milvus,
            "minio": self._probe_minio,
            "mongo": self._probe_mongo,
            "llm": self._probe_llm,
            "embedding": self._probe_embedding,
            "mineru": self._probe_mineru,
        }

    def _required(self, role: str) -> set[str]:
        health = self.settings.health
        return set({"import": health.import_required, "query": health.query_required, "worker": health.worker_required}.get(role, ()))

    def liveness(self, role: str) -> dict:
        return {
            "status": "alive",
            "role": role,
            "timestamp": _now(),
            "config_version": self.settings.config_version,
        }

    def check(self, role: str, *, force: bool = False, include_config: bool = False) -> dict:
        ttl = self.settings.health.cache_ttl_seconds
        cache_key = role
        now_mono = time.monotonic()
        owner = False
        with self._lock:
            cached = self._cache.get(cache_key)
            if not force and cached and now_mono - cached[0] < ttl:
                result = copy.deepcopy(cached[1])
                result["cached"] = True
                if include_config:
                    result["config"] = self.settings.redacted_snapshot()
                return result
            event = self._inflight.get(cache_key)
            if event is None:
                event = threading.Event()
                self._inflight[cache_key] = event
                owner = True

        if not owner:
            event.wait(self.settings.health.timeout_seconds * 2 + 1)
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached:
                result = copy.deepcopy(cached[1])
                result["cached"] = True
                if include_config:
                    result["config"] = self.settings.redacted_snapshot()
                return result

        try:
            result = self._run_checks(role)
            with self._lock:
                self._cache[cache_key] = (time.monotonic(), copy.deepcopy(result))
            if include_config:
                result["config"] = self.settings.redacted_snapshot()
            return result
        finally:
            if owner:
                with self._lock:
                    finished = self._inflight.pop(cache_key, None)
                    if finished:
                        finished.set()

    def _run_checks(self, role: str) -> dict:
        required = self._required(role)
        started = time.perf_counter()
        futures: dict[Future, str] = {}
        dependencies: dict[str, dict] = {}
        now_mono = time.monotonic()
        for name, probe in self._probes.items():
            if self._circuit_until.get(name, 0) > now_mono:
                dependencies[name] = {
                    "status": "circuit_open", "required": name in required, "latency_ms": 0,
                    "last_success_at": self._last_success.get(name), "error_code": "CIRCUIT_OPEN",
                    "message": "连续失败后暂时停止主动探测",
                }
                continue
            futures[self._executor.submit(_timed_probe, probe)] = name

        timeout = self.settings.health.timeout_seconds
        done, unfinished = wait(futures, timeout=timeout + 0.25)
        for future in done:
            name = futures[future]
            latency, detail, error = future.result()
            try:
                if error is not None:
                    raise error
                assert detail is not None
                status = detail.pop("status", "healthy")
                if status == "healthy":
                    self._last_success[name] = _now()
                    self._failures[name] = 0
                else:
                    self._record_failure(name)
                dependencies[name] = {
                    "status": status, "required": name in required, "latency_ms": latency,
                    "last_success_at": self._last_success.get(name), **detail,
                }
            except BaseException as exc:
                code, message = _safe_error(exc)
                self._record_failure(name)
                dependencies[name] = {
                    "status": "unhealthy", "required": name in required, "latency_ms": latency,
                    "last_success_at": self._last_success.get(name), "error_code": code, "message": message,
                }
        for future in unfinished:
            name = futures[future]
            future.cancel()
            self._record_failure(name)
            dependencies[name] = {
                "status": "unhealthy", "required": name in required,
                "latency_ms": round(timeout * 1000, 2), "last_success_at": self._last_success.get(name),
                "error_code": "PROBE_TIMEOUT", "message": f"依赖探测超过 {timeout:g}s",
            }

        ordered = {name: dependencies[name] for name in sorted(dependencies)}
        required_failed = [name for name, item in ordered.items() if item["required"] and item["status"] != "healthy"]
        optional_failed = [name for name, item in ordered.items() if not item["required"] and item["status"] != "healthy"]
        ready = not required_failed
        status = "not_ready" if not ready else ("degraded" if optional_failed else "ready")
        return {
            "status": status, "ready": ready, "role": role, "cached": False,
            "config_version": self.settings.config_version, "checked_at": _now(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "required_failed": required_failed, "optional_failed": optional_failed,
            "dependencies": ordered,
        }

    def _record_failure(self, name: str) -> None:
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if failures >= 3:
            self._circuit_until[name] = time.monotonic() + self.settings.health.cache_ttl_seconds * 6

    def _probe_control_database(self) -> dict:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "healthy", "mode": "query"}

    def _probe_queue(self) -> dict:
        with get_engine().begin() as connection:
            connection.execute(text("SELECT COUNT(*) FROM ingestion_jobs"))
            connection.execute(text("UPDATE ingestion_workers SET heartbeat_at=heartbeat_at WHERE 1=0"))
        return {"status": "healthy", "mode": "read-write-path"}

    def _probe_milvus(self) -> dict:
        client = MilvusClient(uri=self.settings.milvus_url, timeout=self.settings.health.timeout_seconds)
        try:
            collections = client.list_collections(timeout=self.settings.health.timeout_seconds)
            return {"status": "healthy", "collection_count": len(collections)}
        finally:
            client.close()

    def _probe_minio(self) -> dict:
        timeout = urllib3.Timeout(connect=self.settings.health.timeout_seconds, read=self.settings.health.timeout_seconds)
        http = urllib3.PoolManager(
            timeout=timeout,
            retries=False,
            cert_reqs=ssl.CERT_REQUIRED if self.settings.minio_secure else ssl.CERT_NONE,
        )
        client = Minio(
            self.settings.minio_endpoint, access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key, secure=self.settings.minio_secure,
            http_client=http,
        )
        try:
            exists = client.bucket_exists(self.settings.minio_bucket_name)
            return {"status": "healthy" if exists else "unhealthy", "bucket_exists": exists, "error_code": None if exists else "BUCKET_MISSING"}
        finally:
            http.clear()

    def _probe_mongo(self) -> dict:
        timeout_ms = int(self.settings.health.timeout_seconds * 1000)
        client = MongoClient(
            self.settings.mongo_url, serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms, socketTimeoutMS=timeout_ms,
        )
        try:
            client.admin.command("ping")
            return {"status": "healthy", "database": self.settings.mongo_db_name}
        finally:
            client.close()

    def _probe_llm(self) -> dict:
        if not self.settings.openai_base_url or not self.settings.llm_default_model or not self.settings.llm.api_key:
            return {"status": "not_configured", "error_code": "LLM_NOT_CONFIGURED", "message": "LLM endpoint/model/credential 未完整配置"}
        response = requests.get(
            f"{self.settings.openai_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
            timeout=self.settings.health.timeout_seconds,
        )
        if response.status_code >= 400:
            return {"status": "unhealthy", "error_code": f"HTTP_{response.status_code}", "message": "LLM /models 探测失败"}
        return {"status": "healthy", "model": self.settings.llm_default_model}

    def _probe_embedding(self) -> dict:
        local = Path(self.settings.bge_m3_path).expanduser()
        if local.exists():
            return {"status": "healthy", "mode": "local", "model": local.name}
        if self.settings.bge_m3:
            return {"status": "healthy", "mode": "model-id", "model": self.settings.bge_m3}
        return {"status": "not_configured", "error_code": "EMBEDDING_NOT_CONFIGURED", "message": "Embedding 模型未配置"}

    def _probe_mineru(self) -> dict:
        if not self.settings.mineru_base_url or not self.settings.mineru.api_key:
            return {"status": "not_configured", "error_code": "MINERU_NOT_CONFIGURED", "message": "MinerU endpoint/token 未配置"}
        response = requests.head(self.settings.mineru_base_url, timeout=self.settings.health.timeout_seconds, allow_redirects=True)
        if response.status_code >= 500:
            return {"status": "unhealthy", "error_code": f"HTTP_{response.status_code}", "message": "MinerU 连通性探测失败"}
        return {"status": "healthy", "mode": "connectivity"}

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


health_manager = HealthManager()


def validate_startup(role: str) -> None:
    """配置对象构造已完成字段校验；生产/显式门禁再要求依赖 ready。"""
    settings = get_settings()
    must_be_ready = settings.health.startup_require_ready or settings.app_env.lower() in {"prod", "production"}
    if not must_be_ready:
        return
    result = health_manager.check(role, force=True)
    if not result["ready"]:
        failed = ",".join(result["required_failed"])
        raise RuntimeError(f"启动就绪校验失败：role={role}, required_failed={failed}, config_version={settings.config_version}")


def close_runtime_resources() -> None:
    """关闭由运行时创建的连接池；任何关闭失败均不阻断其他资源释放。"""
    closers = []
    try:
        from app.shared.clients.mongo_history_utils import close_history_mongo_tool
        closers.append(close_history_mongo_tool)
    except ImportError:
        pass
    try:
        from app.shared.clients.milvus_utils import close_milvus_client
        closers.append(close_milvus_client)
    except ImportError:
        pass
    try:
        from app.shared.clients.minio_utils import close_minio_client
        closers.append(close_minio_client)
    except ImportError:
        pass
    for close in closers:
        try:
            close()
        except Exception:
            pass
    dispose_database()
