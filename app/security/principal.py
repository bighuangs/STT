from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    auth_type: str
    credential_id: str
    request_id: str = ""


_principal: ContextVar[Principal | None] = ContextVar("stt_principal", default=None)


def current_principal() -> Principal | None:
    return _principal.get()


def set_current_principal(principal: Principal) -> Token:
    return _principal.set(principal)


def reset_current_principal(token: Token) -> None:
    _principal.reset(token)

