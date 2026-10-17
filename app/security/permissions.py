"""Stable action registry. HTTP code authorizes actions, never hard-coded role branches."""

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "TenantAdmin": frozenset({"*"}),
    "KBAdmin": frozenset({
        "kb:read", "kb:update", "document:read", "document:upload", "document:delete",
        "job:read", "job:manage", "index:read", "index:rebuild", "index:publish",
        "index:rollback", "cleanup:read", "cleanup:run", "query:execute",
        "history:read", "history:delete", "audit:read", "eval:read", "eval:run",
        "eval:manage", "feedback:create", "feedback:review",
        "p2:read", "p2:run", "p2:manage", "p3:read", "p3:run",
    }),
    "Editor": frozenset({
        "kb:read", "document:read", "document:upload", "job:read", "query:execute",
        "history:read", "history:delete", "index:read", "eval:read", "eval:run",
        "feedback:create", "p2:read", "p2:run", "p3:read", "p3:run",
    }),
    "Viewer": frozenset({
        "kb:read", "document:read", "job:read", "query:execute", "history:read", "index:read",
        "eval:read", "feedback:create",
        "p2:read", "p3:read",
    }),
    "ServiceAccount": frozenset(),
}


def effective_permissions(roles: tuple[str, ...], scopes: tuple[str, ...]) -> frozenset[str]:
    permissions = set(scopes)
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, ()))
    return frozenset(permissions)


def is_allowed(roles: tuple[str, ...], scopes: tuple[str, ...], action: str) -> bool:
    permissions = effective_permissions(roles, scopes)
    return "*" in permissions or action in permissions or action == "authenticated"
