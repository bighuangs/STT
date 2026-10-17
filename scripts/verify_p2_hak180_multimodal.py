r"""用真实 HAK180 PDF 验证 TODO-23 的图片资产、派生链与可定位 Evidence。

示例：
    .\.venv\Scripts\python.exe scripts\verify_p2_hak180_multimodal.py
    .\.venv\Scripts\python.exe scripts\verify_p2_hak180_multimodal.py --pdftoppm C:\path\pdftoppm.exe

脚本不会调用外部模型或写控制库。渲染和裁剪位于临时目录，退出后删除；派生文本
来自版本化人工黄金集，并明确标记为 ``human-gold/manual-review-v1``。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.multimodal_service import (  # noqa: E402
    MediaValidationError,
    MultimodalService,
    rank_multimodal_candidates,
)
from app.domain.multimodal import (  # noqa: E402
    MultimodalCandidate,
    MultimodalRetrievalProfile,
)


DATASET_PATH = ROOT / "app" / "evaluation" / "datasets" / "hak180_multimodal_v1.json"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdftoppm(explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path(shutil.which("pdftoppm") or ""),
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "native"
        / "poppler"
        / "Library"
        / "bin"
        / "pdftoppm.exe",
    ]
    for candidate in candidates:
        if (
            candidate
            and candidate.is_file()
            and (os.name != "nt" or candidate.suffix.lower() == ".exe")
        ):
            return candidate.resolve()
    raise RuntimeError("找不到 pdftoppm。请安装 Poppler，或用 --pdftoppm 指定 pdftoppm.exe。")


def _render_page(pdf: Path, output_prefix: Path, executable: Path) -> Path:
    command = [
        str(executable),
        "-f",
        "1",
        "-l",
        "1",
        "-singlefile",
        "-r",
        "120",
        "-png",
        str(pdf),
        str(output_prefix),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"HAK180 第 1 页渲染失败：{message[:1000]}")
    rendered = output_prefix.with_suffix(".png")
    if not rendered.is_file():
        raise RuntimeError("pdftoppm 成功退出但没有生成 PNG")
    return rendered


def _crop_bytes(
    image: Image.Image,
    bbox: list[float],
) -> tuple[bytes, tuple[int, int, int, int]]:
    width, height = image.size
    x0, y0, x1, y1 = bbox
    pixels = (
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    )
    crop = image.crop(pixels)
    buffer = BytesIO()
    crop.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), pixels


def verify(*, pdftoppm: str | None = None) -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    pdf = ROOT / dataset["source"]["relative_path"]
    expected_pdf_hash = dataset["source"]["content_sha256"]
    current_pdf_hash = _sha256(pdf)
    if current_pdf_hash != expected_pdf_hash:
        raise RuntimeError("HAK180 PDF hash 与黄金集不一致；禁止用未复核文件继续验收")

    executable = _pdftoppm(pdftoppm)
    service = MultimodalService()
    model_snapshot_hash = (
        "sha256:"
        + sha256(
            json.dumps(
                {
                    "dataset": dataset["dataset_key"],
                    "version": dataset["version"],
                    "review": "human-gold",
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    evidence_rows: list[dict] = []
    candidates: list[MultimodalCandidate] = []

    with tempfile.TemporaryDirectory(prefix="stt-hak180-mm-") as directory:
        rendered = _render_page(
            pdf,
            Path(directory) / "hak180-page1",
            executable,
        )
        render_hash = _sha256(rendered)
        expected_render_hash = dataset["source"]["render"]["content_sha256"]
        with Image.open(rendered) as page:
            page.load()
            actual_size = page.size
            expected_size = (
                dataset["source"]["render"]["width_pixels"],
                dataset["source"]["render"]["height_pixels"],
            )
            if actual_size != expected_size:
                raise RuntimeError(
                    f"渲染尺寸不一致：expected={expected_size}, actual={actual_size}"
                )

            first_crop: bytes | None = None
            for case in dataset["cases"]:
                for region in case["relevant_regions"]:
                    crop, pixel_bbox = _crop_bytes(page, region["bbox"])
                    first_crop = first_crop or crop
                    asset = service.ingest_image(
                        crop,
                        claimed_media_type="image/png",
                        tenant_id="tenant_default",
                        source_kind="document_asset",
                        classification="internal",
                        kb_id="kb_hak180_eval",
                        document_id="doc_hak180_eval",
                        document_version_id="dv_hak180_eval_v1",
                        page_no=region["page"],
                        bbox=tuple(region["bbox"]),
                    )
                    derivation_kind = (
                        case["expected_derivations"][0]
                        if case["expected_derivations"]
                        else "vision_description"
                    )
                    derivation = service.record_derivation(
                        asset,
                        kind=derivation_kind,
                        text=(f"{region['description']}。人工核验答案：{case['reference_answer']}"),
                        provider_id="human-gold",
                        model_id="manual-review-v1",
                        prompt_version="hak180-multimodal-v1",
                        model_snapshot_hash=model_snapshot_hash,
                        confidence=1.0,
                        external_processing=False,
                    )
                    evidence = service.to_evidence(
                        asset,
                        derivation,
                        index_version_id="idx_hak180_eval_v1",
                        chunk_id=f"chunk_{region['region_id'].replace('-', '_')}",
                        source_block_id=region["region_id"],
                        source_label=dataset["source"]["logical_name"],
                    )
                    assert evidence.evidence_id is not None
                    candidates.append(
                        MultimodalCandidate(
                            evidence_id=evidence.evidence_id,
                            asset_id=str(asset.asset_id),
                            modality="image",
                            text_score=0.5,
                            modality_score=1.0,
                        )
                    )
                    evidence_rows.append(
                        {
                            "case_id": case["case_id"],
                            "region_id": region["region_id"],
                            "normalized_bbox": region["bbox"],
                            "pixel_bbox": pixel_bbox,
                            "asset_id": asset.asset_id,
                            "asset_hash": asset.content_hash,
                            "derivation_id": derivation.derivation_id,
                            "evidence_id": evidence.evidence_id,
                            "evidence_modality": evidence.modality,
                            "external_processing": (
                                evidence.derivation.external_processing
                                if evidence.derivation
                                else None
                            ),
                        }
                    )

            if first_crop is None:
                raise RuntimeError("黄金集没有可验证的图片区域")
            attachment = service.ingest_image(
                first_crop,
                claimed_media_type="image/png",
                tenant_id="tenant_default",
                source_kind="query_attachment",
                owner_user_id="user_hak180_eval",
            )
            attachment_derivation = service.record_derivation(
                attachment,
                kind="vision_description",
                text="查询附件隔离负例",
                provider_id="human-gold",
                model_id="manual-review-v1",
                prompt_version="hak180-multimodal-v1",
                model_snapshot_hash=model_snapshot_hash,
            )
            attachment_context = service.query_attachment_context(
                attachment,
                attachment_derivation,
                tenant_id="tenant_default",
                user_id="user_hak180_eval",
            )
            try:
                service.to_evidence(
                    attachment,
                    attachment_derivation,
                    index_version_id="idx_hak180_eval_v1",
                    chunk_id="forbidden_attachment_chunk",
                    source_block_id="forbidden_attachment",
                    source_label="query attachment",
                )
            except MediaValidationError:
                attachment_became_evidence = False
            else:  # pragma: no cover - fail-closed assertion
                attachment_became_evidence = True

    profile = MultimodalRetrievalProfile(
        profile_id="hak180_multimodal_eval",
        version=1,
        enabled=True,
        status="published",
        modalities=("image",),
        route_weights={"text": 1.0, "image": 1.0},
        max_candidates=20,
        max_per_asset=1,
    )
    ranked, diagnostics = rank_multimodal_candidates(candidates, profile)
    passed = (
        len(evidence_rows) == 3
        and len(ranked) == 3
        and not attachment_became_evidence
        and all(not item["external_processing"] for item in evidence_rows)
    )
    return {
        "schema_version": "hak180-multimodal-verification.v1",
        "passed": passed,
        "pdf": {
            "path": str(pdf),
            "sha256": current_pdf_hash,
            "hash_matches_dataset": True,
        },
        "render": {
            "pdftoppm": str(executable),
            "sha256": render_hash,
            "hash_matches_review_render": render_hash == expected_render_hash,
            "note": (
                "不同 Poppler 版本可能产生不同 PNG 编码；PDF hash、尺寸和区域 "
                "Evidence 才是强制门禁。"
            ),
        },
        "document_evidence": evidence_rows,
        "ranking": {
            "evidence_ids": [item.evidence_id for item in ranked],
            "diagnostics": diagnostics,
        },
        "query_attachment_isolation": {
            "asset_id": attachment.asset_id,
            "context_schema": attachment_context["schema_version"],
            "became_kb_evidence": attachment_became_evidence,
        },
        "external_calls": 0,
        "persistent_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdftoppm",
        help="pdftoppm 可执行文件；未传时检查 PATH 和 Codex bundled Poppler",
    )
    arguments = parser.parse_args()
    result = verify(pdftoppm=arguments.pdftoppm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
