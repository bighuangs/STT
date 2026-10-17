"""导入异常分类与指数退避策略。"""

from dataclasses import dataclass

from app.domain.parsing import DocumentParseError


class ValidationJobError(ValueError):
    pass


class DependencyTransientError(RuntimeError):
    pass


class RateLimitJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassifiedJobError:
    category: str
    code: str
    safe_message: str
    retryable: bool


def classify_job_error(exc: BaseException) -> ClassifiedJobError:
    name = type(exc).__name__
    raw = str(exc).strip() or name
    safe = raw[:1000]
    lowered = raw.lower()
    if isinstance(exc, DocumentParseError):
        return ClassifiedJobError(
            "document_parse",
            exc.code,
            exc.safe_message[:1000],
            exc.retryable,
        )
    if isinstance(exc, ValidationJobError):
        return ClassifiedJobError("validation", "VALIDATION_ERROR", safe, False)
    if isinstance(exc, RateLimitJobError) or "429" in lowered or "rate limit" in lowered:
        return ClassifiedJobError("rate_limit", "RATE_LIMIT", safe, True)
    if isinstance(exc, (MemoryError, TimeoutError)) or "out of memory" in lowered:
        return ClassifiedJobError("resource_exhausted", "RESOURCE_EXHAUSTED", safe, True)
    if isinstance(exc, (DependencyTransientError, ConnectionError)) or any(
        token in lowered for token in ("connection reset", "temporarily unavailable", "timeout")
    ):
        return ClassifiedJobError("dependency_transient", "DEPENDENCY_TRANSIENT", safe, True)
    return ClassifiedJobError("unknown", f"UNEXPECTED_{name.upper()[:80]}", safe, True)


def retry_delay_seconds(run_attempt: int, base_seconds: float, max_seconds: float) -> float:
    return min(max_seconds, base_seconds * (2 ** max(0, run_attempt - 1)))
