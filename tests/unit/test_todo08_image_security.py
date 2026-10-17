from __future__ import annotations

from app.infra.object_storage.minio_gateway import minio_gateway
from app.infrastructure.answering import delivery as answer_delivery
from app.infrastructure.answering.delivery import cited_image_urls
from app.rag.query import evidence_asset_service
from app.rag.query.evidence_asset_service import (
    evidence_image_assets,
    evidence_image_projection,
    manifested_evidence_image_assets,
    trusted_image_object_key,
)
from app.shared.clients import minio_utils


EVIDENCE_ID = "ev_" + "1" * 40


def test_answer_images_only_come_from_authorized_minio_asset_refs(monkeypatch) -> None:
    trusted = minio_gateway.build_image_url("manual", "safe.png")
    documents = [
        {
            "url": "https://tracker.example/ignored.jpg",
            "text": "![ignored](javascript:alert(1)//fake.png)",
            "evidence": {
                "evidence_id": EVIDENCE_ID,
                "source": {
                    "source_type": "local",
                    "document_version_id": "dver_manual",
                    "index_version_id": "idx_manual",
                },
                "asset_refs": [
                    "images/from-parser.jpg",
                    "imagetemps-manual/safe.png",
                    "imagetemps-dver_manual/idx_manual/pages/page-7.jpg",
                    trusted,
                    "https://tracker.example/pixel.png",
                    "javascript:alert(1)//fake.png",
                    "data:image/png;base64,AAAA",
                    "imagetemps-manual/active.svg",
                    "imagetemps-../escape.jpg",
                ]
            },
        }
    ]

    assets = evidence_image_assets(documents[0]["evidence"])
    assert [asset.object_key for asset in assets] == [
        f"{minio_gateway.image_dir.strip('/')}/dver_manual/idx_manual/from-parser.jpg",
        f"{minio_gateway.image_dir.strip('/')}/manual/safe.png",
        f"{minio_gateway.image_dir.strip('/')}/dver_manual/idx_manual/pages/page-7.jpg",
    ]
    monkeypatch.setattr(answer_delivery, "manifested_evidence_image_assets", lambda _: assets)
    state = {
        "citations": [{"evidence_id": EVIDENCE_ID, "validation_status": "valid"}],
        "context_aliases": [{"evidence": documents[0]["evidence"]}],
    }
    assert cited_image_urls(state) == [
        asset.preview_url(EVIDENCE_ID) for asset in assets
    ]


def test_answer_images_reject_unbound_markdown_and_web_urls() -> None:
    documents = [
        {
            "url": "https://example.test/result.png",
            "text": "![untrusted](https://example.test/content.jpg)",
            "evidence": {"evidence_id": EVIDENCE_ID, "asset_refs": []},
        }
    ]

    assert cited_image_urls(
        {"citations": [], "context_aliases": [{"evidence": documents[0]["evidence"]}]}
    ) == []


def test_image_asset_parser_rejects_traversal_foreign_hosts_and_active_content() -> None:
    assert trusted_image_object_key("imagetemps-a/b/../../../secret.png") is None
    assert trusted_image_object_key("imagetemps-a/b/payload.svg") is None
    assert trusted_image_object_key("https://tracker.example/pixel.png") is None
    assert trusted_image_object_key("data:image/png;base64,AAAA") is None


def test_versioned_image_identity_is_stable_and_does_not_expose_object_key() -> None:
    evidence = {
        "evidence_id": EVIDENCE_ID,
        "asset_refs": [
            "imagetemps-dver_1/idx_2/page-8.jpeg",
            "imagetemps-dver_1/idx_2/page-8.jpeg",
        ],
    }
    first = evidence_image_assets(evidence)
    second = evidence_image_assets(evidence)

    assert first == second
    assert len(first) == 1
    assert first[0].image_id.startswith("img_")
    assert first[0].object_key not in first[0].preview_url(EVIDENCE_ID)


def test_real_evidence_shape_only_projects_validated_uploaded_assets(monkeypatch) -> None:
    """回归联想海豚 Evidence 中解析引用与上传清单并存的真实形态。"""

    document_version_id = "dv_8d4db33a9cb0463bb430463b724f8ba7"
    index_version_id = "idx_05cc71f5bb344ef8b11f5bc87de2155a"
    missing_name = "20c0c836e80cfe98f3730708d395a0cf30b425eb97e35cbe860fe5e68eb8def6.jpg"
    uploaded_names = [
        "fc65ea5cdd38eea71c8fc6ea843653a954ea1bf5711991c740249a371325005f.jpg",
        "e532e04b51542ef589bfc8c5cf651347fddcec26bcb6df189c8164ae1ddb1795.jpg",
        "9a5c6946bbfe4d3e9661b232a2be633371c7b77ff239ea605a8512e37c6ede94.jpg",
    ]
    evidence = {
        "evidence_id": "ev_268297906f695b0eba55113086a3b1f2bc875e9d",
        "source": {
            "source_type": "local",
            "document_version_id": document_version_id,
            "index_version_id": index_version_id,
        },
        "asset_refs": [
            f"images/{missing_name}",
            *(f"images/{name}" for name in uploaded_names),
            f"imagetemps-doc_0e9272569afa456b9e8d19afd288f5c5/{missing_name}",
            *(
                f"imagetemps-{document_version_id}/{index_version_id}/{name}"
                for name in uploaded_names
            ),
        ],
    }
    image_prefix = minio_gateway.image_dir.strip("/")
    manifested_keys = frozenset(
        f"{image_prefix}/{document_version_id}/{index_version_id}/{name}"
        for name in uploaded_names
    )
    monkeypatch.setattr(
        evidence_asset_service,
        "_validated_manifest_object_keys",
        lambda _: manifested_keys,
    )

    parsed_assets = evidence_image_assets(evidence)
    published_assets = manifested_evidence_image_assets(evidence)
    projections = evidence_image_projection(evidence)

    assert len(parsed_assets) == 5
    assert [asset.object_key for asset in published_assets] == [
        f"{image_prefix}/{document_version_id}/{index_version_id}/{name}"
        for name in uploaded_names
    ]
    assert len(projections) == 3
    assert all(item["url"].startswith("/api/v1/evidence/") for item in projections)
    assert not any(asset.image_id == "img_9ba56c08a4a5d71e914cb1090bda7949" for asset in published_assets)


def test_minio_bucket_ready_removes_legacy_anonymous_read_policy() -> None:
    calls: list[tuple[str, str]] = []

    class Client:
        def bucket_exists(self, bucket_name: str) -> bool:
            calls.append(("exists", bucket_name))
            return True

        def delete_bucket_policy(self, bucket_name: str) -> None:
            calls.append(("private", bucket_name))

        def set_bucket_policy(self, *_):  # pragma: no cover - security tripwire
            raise AssertionError("Evidence 图片启用后不得重新开放匿名桶读取")

    bucket_name = minio_utils.minio_config.bucket_name
    minio_utils._create_bucket_ready(Client())

    assert calls == [("exists", bucket_name), ("private", bucket_name)]
