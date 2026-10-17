import base64
import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe

from app.shared.config.settings_config import settings


class InvalidTokenError(ValueError):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(*, user_id: str, tenant_id: str, token_version: int, token_type: str) -> tuple[str, str, datetime]:
    now = datetime.now(UTC)
    expires = now + (
        timedelta(minutes=settings.auth_access_token_minutes)
        if token_type == "access"
        else timedelta(days=settings.auth_refresh_token_days)
    )
    jti = token_urlsafe(24)
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({
        "iss": settings.auth_token_issuer, "sub": user_id, "tid": tenant_id,
        "typ": token_type, "ver": token_version, "jti": jti,
        "iat": int(now.timestamp()), "exp": int(expires.timestamp()),
    }, separators=(",", ":")).encode())
    signature = _b64(hmac.new(settings.auth_token_secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}", jti, expires


def decode_token(token: str, expected_type: str) -> dict:
    try:
        header, payload, signature = token.split(".")
        signed = f"{header}.{payload}".encode()
        valid = any(
            hmac.compare_digest(
                signature,
                _b64(hmac.new(secret.encode(), signed, hashlib.sha256).digest()),
            )
            for secret in settings.security.token_secrets
        )
        if not valid:
            raise InvalidTokenError("invalid token signature")
        claims = json.loads(_unb64(payload))
    except InvalidTokenError:
        raise
    except Exception as exc:
        raise InvalidTokenError("malformed token") from exc
    if claims.get("iss") != settings.auth_token_issuer or claims.get("typ") != expected_type:
        raise InvalidTokenError("invalid token claims")
    if int(claims.get("exp", 0)) <= int(time.time()):
        raise InvalidTokenError("token expired")
    return claims


def token_digest(token: str, secret: str | None = None) -> str:
    return hmac.new((secret or settings.auth_token_secret).encode(), token.encode(), hashlib.sha256).hexdigest()


def token_digests(token: str) -> tuple[str, ...]:
    """当前与上一密钥的摘要候选，用于无中断轮换刷新令牌/API Key。"""
    return tuple(token_digest(token, secret) for secret in settings.security.token_secrets)
