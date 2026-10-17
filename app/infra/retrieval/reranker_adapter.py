"""BGE 重排器的端口适配器；模型实例由组合根注入。"""

from __future__ import annotations

import gc
import inspect
import math
import time
from typing import Any, Callable, Protocol

from app.application.retrieval_ports import RerankerPort
from app.domain.retrieval import (
    RerankRequest,
    RerankResult,
    RetrievalEngineError,
    RetrievalHit,
    RouteScore,
)


class ScoreModel(Protocol):
    def compute_score(
        self,
        pairs: list[list[str]],
        normalize: bool = True,
        **kwargs: Any,
    ) -> Any: ...


TextPreparer = Callable[[str, str, int, str], str]

RERANK_SPECIAL_TOKEN_RESERVE = 8
RERANK_PRETOKEN_CHARS_PER_TOKEN = 8
DEFAULT_RERANK_BATCH_SIZE = 2
DEFAULT_RERANK_MAX_INPUT_TOKENS = 512
DEFAULT_RERANK_QUERY_MAX_TOKENS = 128
DEFAULT_RERANK_DOCUMENT_MAX_TOKENS = 376
DEFAULT_RERANK_FAILURE_MODE = "preserve_order"


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_token_budget(tokenizer, text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    # Tokenizer 本身也会为超长字符串分配中间缓冲。先做与 token 预算成比例的
    # 硬字符帽，保证即使 Evidence 字段异常膨胀也不会把整段交给 Rust tokenizer。
    max_chars = max(token_budget, token_budget * RERANK_PRETOKEN_CHARS_PER_TOKEN)
    candidate = text[:max_chars]
    token_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if len(token_ids) <= token_budget:
        return candidate
    decode = getattr(tokenizer, "decode", None)
    truncated = ""
    if callable(decode):
        try:
            truncated = str(
                decode(
                    token_ids[:token_budget],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            truncated = str(decode(token_ids[:token_budget]))
    if not truncated or _token_count(tokenizer, truncated) > token_budget:
        low, high, best = 0, len(candidate), 0
        while low <= high:
            middle = (low + high) // 2
            if _token_count(tokenizer, candidate[:middle]) <= token_budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        truncated = candidate[:best]
    while truncated and _token_count(tokenizer, truncated) > token_budget:
        truncated = truncated[:-1]
    return truncated


def _bounded_pair(
    model: ScoreModel,
    question: str,
    content: str,
    *,
    max_input_tokens: int,
    query_max_tokens: int = DEFAULT_RERANK_QUERY_MAX_TOKENS,
    document_max_tokens: int = DEFAULT_RERANK_DOCUMENT_MAX_TOKENS,
) -> tuple[list[str], bool]:
    """用本地 reranker tokenizer 构造硬预算输入；无 tokenizer 时保守按字符截断。"""

    available_tokens = max(2, max_input_tokens - RERANK_SPECIAL_TOKEN_RESERVE)
    query_budget = min(query_max_tokens, max(1, available_tokens - 1))
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        bounded_question = question[:query_budget]
        content_budget = min(
            document_max_tokens,
            max(1, available_tokens - len(bounded_question)),
        )
        bounded_content = content[:content_budget]
        return [bounded_question, bounded_content], (
            bounded_question != question or bounded_content != content
        )

    bounded_question = _truncate_to_token_budget(tokenizer, question, query_budget)
    content_budget = max(
        1,
        available_tokens - _token_count(tokenizer, bounded_question),
    )
    content_budget = min(document_max_tokens, content_budget)
    bounded_content = _truncate_to_token_budget(tokenizer, content, content_budget)
    return [bounded_question, bounded_content], (
        bounded_question != question or bounded_content != content
    )


def _score_kwargs(model: ScoreModel, values: dict[str, Any]) -> dict[str, Any]:
    """只向 fake/第三方模型传递其声明支持的参数。"""

    try:
        parameters = inspect.signature(model.compute_score).parameters.values()
    except (TypeError, ValueError):
        return {"normalize": values["normalize"]}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return values
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in values.items() if name in names}


def _normalized_scores(raw_scores: Any) -> list[float]:
    if isinstance(raw_scores, (float, int)):
        raw_scores = [raw_scores]
    try:
        return [float(value) for value in raw_scores]
    except TypeError:
        return [float(raw_scores)]


class BGERerankerAdapter(RerankerPort):
    """将 ``compute_score`` 模型输出转换为稳定命中与路线观察。"""

    def __init__(
        self,
        *,
        model: ScoreModel,
        text_preparer: TextPreparer | None = None,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_input_tokens: int = DEFAULT_RERANK_MAX_INPUT_TOKENS,
        query_max_tokens: int = DEFAULT_RERANK_QUERY_MAX_TOKENS,
        document_max_tokens: int = DEFAULT_RERANK_DOCUMENT_MAX_TOKENS,
        failure_mode: str = DEFAULT_RERANK_FAILURE_MODE,
        clock=time.monotonic,
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch_size must be positive")
        if max_input_tokens < 32:
            raise ValueError("reranker max_input_tokens must be at least 32")
        if query_max_tokens < 1 or document_max_tokens < 1:
            raise ValueError("reranker query/document token limits must be positive")
        if failure_mode not in {"error", "preserve_order"}:
            raise ValueError("reranker failure_mode must be error or preserve_order")
        self._model = model
        self._text_preparer = text_preparer
        self._batch_size = batch_size
        self._max_input_tokens = max_input_tokens
        self._query_max_tokens = query_max_tokens
        self._document_max_tokens = document_max_tokens
        self._failure_mode = failure_mode
        self._clock = clock

    def rerank(self, request: RerankRequest) -> RerankResult:
        started = self._clock()
        if not request.candidates or not request.config.enabled:
            return RerankResult(
                trace_id=request.trace_id,
                model_id=request.config.model_id,
                hits=request.candidates,
                elapsed_ms=self._elapsed_ms(started),
            )

        effective_max_tokens = min(
            request.config.max_input_tokens,
            self._max_input_tokens,
        )
        scores: list[float] = []
        truncated_count = 0
        batch_count = 0
        for start in range(0, len(request.candidates), self._batch_size):
            candidate_batch = request.candidates[start : start + self._batch_size]
            pairs: list[list[str]] = []
            for candidate in candidate_batch:
                content = candidate.content
                if self._text_preparer is not None:
                    content = self._text_preparer(
                        request.query,
                        content,
                        effective_max_tokens,
                        request.config.long_text_strategy,
                    )
                pair, truncated = _bounded_pair(
                    self._model,
                    request.query,
                    content,
                    max_input_tokens=effective_max_tokens,
                    query_max_tokens=self._query_max_tokens,
                    document_max_tokens=self._document_max_tokens,
                )
                pairs.append(pair)
                truncated_count += int(truncated)
            batch_count += 1
            try:
                raw_scores = self._model.compute_score(
                    pairs,
                    **_score_kwargs(
                        self._model,
                        {
                            "normalize": request.config.normalize_scores,
                            "batch_size": min(self._batch_size, len(pairs)),
                            "query_max_length": min(
                                self._query_max_tokens,
                                effective_max_tokens,
                            ),
                            "max_length": effective_max_tokens,
                        },
                    ),
                )
            except (MemoryError, RuntimeError) as exc:
                return self._model_failure(
                    request,
                    started=started,
                    batch_count=batch_count,
                    truncated_count=truncated_count,
                    max_input_tokens=effective_max_tokens,
                    exc=exc,
                )
            except Exception as exc:
                return self._model_failure(
                    request,
                    started=started,
                    batch_count=batch_count,
                    truncated_count=truncated_count,
                    max_input_tokens=effective_max_tokens,
                    exc=exc,
                )
            try:
                batch_scores = _normalized_scores(raw_scores)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RetrievalEngineError(
                    "RERANKER_INVALID_RESPONSE",
                    "重排模型返回的分数无法解析",
                    retryable=False,
                    route="reranker",
                    details={"batch": batch_count},
                ) from exc
            if len(batch_scores) != len(candidate_batch) or not all(
                math.isfinite(value) for value in batch_scores
            ):
                raise RetrievalEngineError(
                    "RERANKER_INVALID_RESPONSE",
                    "重排模型返回的分数数量或数值无效",
                    retryable=False,
                    route="reranker",
                    details={
                        "expected": len(candidate_batch),
                        "actual": len(batch_scores),
                        "batch": batch_count,
                    },
                )
            scores.extend(batch_scores)
            del pairs, raw_scores, batch_scores
        if len(scores) != len(request.candidates) or not all(
            math.isfinite(value) for value in scores
        ):
            raise RetrievalEngineError(
                "RERANKER_INVALID_RESPONSE",
                "重排模型返回的分数数量或数值无效",
                retryable=False,
                route="reranker",
                details={
                    "expected": len(request.candidates),
                    "actual": len(scores),
                },
            )

        ranked: list[tuple[RetrievalHit, float]] = sorted(
            zip(request.candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        with_scores: list[RetrievalHit] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            observations = tuple(item for item in hit.scores if item.route != "reranker") + (
                RouteScore(
                    route="reranker",
                    raw_score=score,
                    rank=rank,
                    explanation=f"model={request.config.model_id}",
                ),
            )
            with_scores.append(
                hit.model_copy(
                    update={"scores": observations, "final_score": score},
                    deep=True,
                )
            )

        max_top_k = min(request.config.max_top_k, len(with_scores))
        top_k = max_top_k
        for index in range(request.config.min_top_k - 1, max_top_k - 1):
            current = with_scores[index].final_score or 0.0
            following = with_scores[index + 1].final_score or 0.0
            absolute_gap = abs(current - following)
            ratio_gap = max(0.0, current - following) / max(abs(current), 1e-7)
            if ratio_gap >= request.config.gap_ratio or absolute_gap >= request.config.gap_absolute:
                top_k = index + 1
                break

        return RerankResult(
            trace_id=request.trace_id,
            model_id=request.config.model_id,
            hits=tuple(with_scores[:top_k]),
            elapsed_ms=self._elapsed_ms(started),
            status="succeeded",
            batch_size=self._batch_size,
            batch_count=batch_count,
            truncated_count=truncated_count,
            max_input_tokens=effective_max_tokens,
        )

    def _model_failure(
        self,
        request: RerankRequest,
        *,
        started: float,
        batch_count: int,
        truncated_count: int,
        max_input_tokens: int,
        exc: Exception,
    ) -> RerankResult:
        gc.collect()
        if self._failure_mode == "error":
            raise RetrievalEngineError(
                "RERANKER_UNAVAILABLE",
                "重排模型调用失败",
                retryable=True,
                route="reranker",
                details={"batch": batch_count, "error_type": type(exc).__name__},
            ) from exc
        return RerankResult(
            trace_id=request.trace_id,
            model_id=request.config.model_id,
            hits=request.candidates,
            elapsed_ms=self._elapsed_ms(started),
            status="degraded",
            failure_code="RERANKER_UNAVAILABLE",
            message="重排模型不可用，已保留融合候选顺序",
            batch_size=self._batch_size,
            batch_count=batch_count,
            truncated_count=truncated_count,
            max_input_tokens=max_input_tokens,
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))
