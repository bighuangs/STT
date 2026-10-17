from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError
from PIL import Image

from app.application.import_runner import _stage_event_detail
from app.rag.import_ import markdown_image_service


class _SequenceVisionModel:
    def __init__(self, outcomes: list[object]):
        self._outcomes = list(outcomes)
        self._lock = threading.Lock()

    def __or__(self, _parser):
        return self

    def invoke(self, _messages):
        with self._lock:
            outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        message="SSL EOF occurred in violation of protocol",
        request=httpx.Request("POST", "https://vlm.invalid/v1/chat/completions"),
    )


def _write_image(path: Path, color: str = "red") -> None:
    Image.new("RGB", (32, 32), color=color).save(path, format="PNG")


def test_batch_vlm_connection_failure_degrades_per_image_and_keeps_other_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first)
    _write_image(second, color="blue")
    model = _SequenceVisionModel([_connection_error(), "第二张图片的有效描述"])
    monkeypatch.setattr(markdown_image_service.llm_provider, "vision_chat", lambda: model)
    warnings: list[dict] = []

    result = markdown_image_service.batch_describe_images(
        [
            (first.name, first, ("", "")),
            (second.name, second, ("", "")),
        ],
        "manual",
        vlm_max_workers=1,
        warnings=warnings,
    )

    assert result == {
        first.name: "图片 1",
        second.name: "第二张图片的有效描述",
    }
    assert len(warnings) == 1
    assert warnings[0] == {
        "code": "VLM_IMAGE_DESCRIPTION_DEGRADED",
        "severity": "warning",
        "message": "视觉描述服务暂时不可用，已保留原始图片和正文并继续导入。",
        "stage": "image_summary",
        "subject": first.name,
        "error_type": "APIConnectionError",
    }


@pytest.mark.parametrize("fatal_error", [PermissionError("forbidden"), TypeError("bug")])
def test_batch_vlm_does_not_swallow_permission_or_programming_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fatal_error: Exception,
) -> None:
    image_path = tmp_path / "warning.png"
    _write_image(image_path)
    model = _SequenceVisionModel([fatal_error])
    monkeypatch.setattr(markdown_image_service.llm_provider, "vision_chat", lambda: model)

    with pytest.raises(type(fatal_error), match=str(fatal_error)):
        markdown_image_service.batch_describe_images(
            [(image_path.name, image_path, ("", ""))],
            "manual",
            vlm_max_workers=1,
            warnings=[],
        )


def test_unconfigured_optional_vlm_degrades_the_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "warning.png"
    _write_image(image_path)

    def _not_configured():
        raise ValueError("[LLM客户端] 配置缺失：请配置可选 VLM")

    monkeypatch.setattr(markdown_image_service.llm_provider, "vision_chat", _not_configured)
    warnings: list[dict] = []

    result = markdown_image_service.batch_describe_images(
        [(image_path.name, image_path, ("", ""))],
        "manual",
        vlm_max_workers=1,
        warnings=warnings,
    )

    assert result == {image_path.name: "图片 1"}
    assert warnings[0]["stage"] == "image_summary_batch"
    assert warnings[0]["error_type"] == "ValueError"


def test_enrichment_keeps_original_image_and_chunk_text_and_exposes_job_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "warning.png"
    _write_image(image_path)
    markdown_path = tmp_path / "manual.md"
    markdown_text = "使用前请阅读安全须知。\n\n![警告图](images/warning.png)"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    model = _SequenceVisionModel([_connection_error(), _connection_error()])
    monkeypatch.setattr(markdown_image_service.llm_provider, "vision_chat", lambda: model)
    state = {
        "md_path": str(markdown_path),
        "md_content": markdown_text,
        "chunks": [
            {
                "content": "使用前请阅读安全须知。",
                "images": ["images/warning.png"],
            }
        ],
        "parser_config": {"vlm_max_workers": 1, "vlm_min_image_side": 11},
    }

    result = markdown_image_service.enrich_markdown_images(state)

    assert result["md_content"] == markdown_text
    assert result["chunks"][0]["content"] == "使用前请阅读安全须知。"
    assert result["image_summaries"] == {"warning.png": "图片 1"}
    assert "warning.png" in result["image_pil_dict"]
    assert {warning["stage"] for warning in result["vlm_warnings"]} == {
        "image_summary",
        "chunk_vision",
    }

    detail = _stage_event_detail("node_md_img", result)
    assert detail["degraded"] is True
    assert detail["warning_count"] == 2
    assert detail["warnings"] == result["vlm_warnings"]

    result["image_pil_dict"]["warning.png"].close()


def test_non_vlm_stage_event_does_not_leak_unrelated_warning_state() -> None:
    detail = _stage_event_detail(
        "node_link_and_table_chunk",
        {"vlm_warnings": [{"code": "VLM_IMAGE_DESCRIPTION_DEGRADED"}]},
    )

    assert detail == {"node": "node_link_and_table_chunk"}
