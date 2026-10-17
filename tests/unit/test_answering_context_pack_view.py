from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.application.query_answering.builders import _ContextPackV2View


def _valid_context_pack() -> dict:
    evidence_id = "ev_" + "a" * 40
    representation_id = "rep_" + "b" * 40
    return {
        "schema_version": "context-pack.v2",
        "content_hash": "sha256:" + "c" * 64,
        "packed_blocks": [
            {
                "label": "E1",
                "evidence_id": evidence_id,
                "representation_id": representation_id,
                "original_content_hash": "sha256:" + "d" * 64,
                "text": "可验证的证据正文",
                "sub_query_ids": ["sq_primary"],
                "truncation": None,
            }
        ],
        "omitted": [],
        "label_mapping": [
            {
                "label": "E1",
                "evidence_id": evidence_id,
                "representation_id": representation_id,
            }
        ],
        "upstream_only_field": {"ignored": True},
    }


def test_context_pack_v2_view_accepts_only_required_identity_projection() -> None:
    view = _ContextPackV2View.model_validate(_valid_context_pack())

    assert view.schema_version == "context-pack.v2"
    assert view.packed_blocks[0].label == "E1"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda pack: pack.update(schema_version="context-pack.v1"),
        lambda pack: pack.update(content_hash="not-a-hash"),
        lambda pack: pack["packed_blocks"][0].update(label="E2"),
        lambda pack: pack["label_mapping"][0].update(representation_id="rep_" + "e" * 40),
    ),
)
def test_context_pack_v2_view_rejects_version_hash_or_mapping_drift(mutation) -> None:
    pack = deepcopy(_valid_context_pack())
    mutation(pack)

    with pytest.raises(ValidationError):
        _ContextPackV2View.model_validate(pack)
