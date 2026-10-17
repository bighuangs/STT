"""Portable JSON checkpoint 的纯 AES-256-GCM codec。

该 foundation 模块不读取环境变量，也不依赖任何业务包。调用方必须显式注入独立
key 和完整版本/身份上下文；用途复用与兼容策略由外层组合根决定。
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import re
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


QUERY_CHECKPOINT_KEY_PURPOSE = "query-checkpoint-encryption.v1"
QUERY_CHECKPOINT_MAX_PLAINTEXT_BYTES = 16 * 1024 * 1024
_ENCODED_KEY = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


class QueryCheckpointCryptoError(ValueError):
    """checkpoint 无法安全加解密；消息不得包含正文。"""


class QueryCheckpointKeyInvalid(QueryCheckpointCryptoError):
    """独立 checkpoint key 缺失或格式无效。"""


class QueryCheckpointCorrupt(QueryCheckpointCryptoError):
    """ciphertext、AAD 或 JSON envelope 已损坏。"""


class QueryCheckpointVersionUnsupported(QueryCheckpointCryptoError):
    """持久化版本不在当前执行器的显式兼容集合中。"""


@dataclass(frozen=True, slots=True)
class QueryCheckpointEncryptionContext:
    tenant_id: str
    user_id: str
    query_run_id: str
    sequence: int
    completed_node: str | None
    next_node: str | None
    checkpoint_namespace: str
    workflow_version: str
    state_schema_version: str
    checkpoint_schema_version: str


@dataclass(frozen=True, slots=True)
class EncryptedQueryCheckpoint:
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)
    ciphertext_sha256: str
    size_bytes: int


def decode_query_checkpoint_key(value: str | bytes) -> bytes:
    """解析独立 key；字符串必须是 URL-safe base64 编码的 32 bytes。"""

    if isinstance(value, bytes):
        if len(value) != 32:
            raise QueryCheckpointKeyInvalid("query checkpoint key must contain 32 bytes")
        return bytes(value)
    encoded = value.strip()
    if not encoded or _ENCODED_KEY.fullmatch(encoded) is None:
        raise QueryCheckpointKeyInvalid("query checkpoint key must be URL-safe base64")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise QueryCheckpointKeyInvalid("query checkpoint key must be URL-safe base64") from exc
    if len(decoded) != 32:
        raise QueryCheckpointKeyInvalid("query checkpoint key must decode to 32 bytes")
    return decoded


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QueryCheckpointCryptoError("query checkpoint state must be portable JSON") from exc
    return serialized.encode("utf-8")


def _aad(context: QueryCheckpointEncryptionContext) -> bytes:
    if not context.tenant_id or not context.user_id or not context.query_run_id:
        raise QueryCheckpointCryptoError("query checkpoint identity is incomplete")
    if context.sequence < 0:
        raise QueryCheckpointCryptoError("query checkpoint sequence cannot be negative")
    version_fields = (
        context.checkpoint_namespace,
        context.workflow_version,
        context.state_schema_version,
        context.checkpoint_schema_version,
    )
    if any(not value for value in version_fields):
        raise QueryCheckpointVersionUnsupported("query checkpoint metadata version is incomplete")
    return _canonical_json(
        {
            "purpose": QUERY_CHECKPOINT_KEY_PURPOSE,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "query_run_id": context.query_run_id,
            "sequence": context.sequence,
            "completed_node": context.completed_node,
            "next_node": context.next_node,
            "checkpoint_namespace": context.checkpoint_namespace,
            "workflow_version": context.workflow_version,
            "state_schema_version": context.state_schema_version,
            "checkpoint_schema_version": context.checkpoint_schema_version,
        }
    )


class QueryCheckpointCipher:
    """显式 key 注入的 AES-GCM codec；对象不读取全局 settings。"""

    def __init__(self, key: str | bytes) -> None:
        self._cipher = AESGCM(decode_query_checkpoint_key(key))

    def encrypt_state(
        self,
        state: Mapping[str, Any],
        *,
        context: QueryCheckpointEncryptionContext,
    ) -> EncryptedQueryCheckpoint:
        aad = _aad(context)
        if state.get("tenant_id") != context.tenant_id:
            raise QueryCheckpointCryptoError("query checkpoint tenant binding mismatch")
        if state.get("user_id") != context.user_id:
            raise QueryCheckpointCryptoError("query checkpoint user binding mismatch")
        if state.get("query_run_id") != context.query_run_id:
            raise QueryCheckpointCryptoError("query checkpoint run binding mismatch")
        plaintext = _canonical_json(
            {
                "checkpoint_schema_version": context.checkpoint_schema_version,
                "workflow_version": context.workflow_version,
                "state_schema_version": context.state_schema_version,
                "checkpoint_namespace": context.checkpoint_namespace,
                "sequence": context.sequence,
                "completed_node": context.completed_node,
                "next_node": context.next_node,
                "state": dict(state),
            }
        )
        if len(plaintext) > QUERY_CHECKPOINT_MAX_PLAINTEXT_BYTES:
            raise QueryCheckpointCryptoError("query checkpoint exceeds the maximum size")
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(nonce, plaintext, aad)
        return EncryptedQueryCheckpoint(
            nonce=nonce,
            ciphertext=ciphertext,
            ciphertext_sha256=f"sha256:{sha256(ciphertext).hexdigest()}",
            size_bytes=len(ciphertext),
        )

    def decrypt_state(
        self,
        encrypted: EncryptedQueryCheckpoint,
        *,
        context: QueryCheckpointEncryptionContext,
    ) -> dict[str, Any]:
        aad = _aad(context)
        if len(encrypted.nonce) != 12:
            raise QueryCheckpointCorrupt("query checkpoint nonce is invalid")
        expected_digest = f"sha256:{sha256(encrypted.ciphertext).hexdigest()}"
        if encrypted.ciphertext_sha256 != expected_digest:
            raise QueryCheckpointCorrupt("query checkpoint ciphertext digest mismatch")
        if encrypted.size_bytes != len(encrypted.ciphertext):
            raise QueryCheckpointCorrupt("query checkpoint ciphertext size mismatch")
        try:
            plaintext = self._cipher.decrypt(encrypted.nonce, encrypted.ciphertext, aad)
        except (InvalidTag, ValueError) as exc:
            raise QueryCheckpointCorrupt("query checkpoint authentication failed") from exc
        if len(plaintext) > QUERY_CHECKPOINT_MAX_PLAINTEXT_BYTES:
            raise QueryCheckpointCorrupt("query checkpoint plaintext is oversized")
        try:
            envelope = json.loads(plaintext)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryCheckpointCorrupt("query checkpoint JSON is invalid") from exc
        if not isinstance(envelope, dict) or not isinstance(envelope.get("state"), dict):
            raise QueryCheckpointCorrupt("query checkpoint envelope is invalid")
        expected_metadata = {
            "checkpoint_schema_version": context.checkpoint_schema_version,
            "workflow_version": context.workflow_version,
            "state_schema_version": context.state_schema_version,
            "checkpoint_namespace": context.checkpoint_namespace,
            "sequence": context.sequence,
            "completed_node": context.completed_node,
            "next_node": context.next_node,
        }
        if any(envelope.get(key) != value for key, value in expected_metadata.items()):
            raise QueryCheckpointCorrupt("query checkpoint envelope binding mismatch")
        state = dict(envelope["state"])
        if (
            state.get("tenant_id") != context.tenant_id
            or state.get("user_id") != context.user_id
            or state.get("query_run_id") != context.query_run_id
        ):
            raise QueryCheckpointCorrupt("query checkpoint state identity mismatch")
        return state


__all__ = [
    "EncryptedQueryCheckpoint",
    "QUERY_CHECKPOINT_KEY_PURPOSE",
    "QUERY_CHECKPOINT_MAX_PLAINTEXT_BYTES",
    "QueryCheckpointCipher",
    "QueryCheckpointCorrupt",
    "QueryCheckpointCryptoError",
    "QueryCheckpointEncryptionContext",
    "QueryCheckpointKeyInvalid",
    "QueryCheckpointVersionUnsupported",
    "decode_query_checkpoint_key",
]
