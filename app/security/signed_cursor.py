"""Authenticated opaque cursors shared by the knowledge workbench APIs.

Cursor payloads are not secrets, but they are control-plane input.  Signing them
prevents a caller from forging keyset positions or artifact offsets.  The signing
key is domain-separated from the application's token secret and therefore needs
no additional plaintext configuration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from typing import Any

from app.shared.config.settings_config import settings


_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_KEY_CONTEXT = b"stt/workbench/cursor-signing/v1/"


class SignedCursorError(ValueError):
    """Raised when an opaque cursor is malformed or fails authentication."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not value or _B64URL.fullmatch(value) is None:
        raise SignedCursorError("invalid cursor encoding")
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, TypeError) as exc:
        raise SignedCursorError("invalid cursor encoding") from exc
    if not hmac.compare_digest(_b64encode(decoded), value):
        raise SignedCursorError("non-canonical cursor encoding")
    return decoded


def _key(secret: str, purpose: str) -> bytes:
    return hmac.new(
        secret.encode("utf-8"), _KEY_CONTEXT + purpose.encode("utf-8"), hashlib.sha256
    ).digest()


def encode_signed_cursor(payload: dict[str, Any], *, purpose: str) -> str:
    """Return canonical JSON plus an HMAC-SHA256 signature."""

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    encoded = _b64encode(raw)
    signature = hmac.new(
        _key(settings.auth_token_secret, purpose),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded}.{_b64encode(signature)}"


def decode_signed_cursor(cursor: str, *, purpose: str) -> dict[str, Any]:
    """Authenticate and decode a cursor, accepting configured rotation secrets."""

    if not isinstance(cursor, str) or not cursor or len(cursor) > 8192:
        raise SignedCursorError("invalid cursor")
    try:
        encoded, signature_text = cursor.split(".")
    except ValueError as exc:
        raise SignedCursorError("invalid cursor") from exc
    raw = _b64decode(encoded)
    signature = _b64decode(signature_text)
    if len(signature) != hashlib.sha256().digest_size:
        raise SignedCursorError("invalid cursor signature")

    valid = False
    for secret in settings.security.token_secrets:
        expected = hmac.new(
            _key(secret, purpose), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        valid = hmac.compare_digest(signature, expected) or valid
    if not valid:
        raise SignedCursorError("invalid cursor signature")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignedCursorError("invalid cursor payload") from exc
    if not isinstance(payload, dict):
        raise SignedCursorError("invalid cursor payload")
    return payload


__all__ = ["SignedCursorError", "decode_signed_cursor", "encode_signed_cursor"]
