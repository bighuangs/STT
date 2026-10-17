"""Version the built-in retrieval timeout for contended local BGE queries.

Revision ID: 20260807_0013
Revises: 20260725_0012
Create Date: 2026-08-07

Only the untouched, system-created ``rpf_default`` profile is upgraded.  Custom
profiles keep their explicit timeout unchanged.  The old row is retired rather
than mutated so frozen QueryRuns remain reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any

from alembic import op
import sqlalchemy as sa


revision = "20260807_0013"
down_revision = "20260725_0012"
branch_labels = None
depends_on = None


_OLD_TIMEOUT_MS = 30_000
_NEW_TIMEOUT_MS = 90_000
_CREATED_BY = "system-timeout-v13"
_DESCRIPTION_PREFIX = "[builtin-timeout-v2:"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _config_hash(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _profile_id(source_row_id: str) -> str:
    digest = sha256(f"{source_row_id}|builtin-timeout-v2".encode()).hexdigest()[:32]
    return f"rpv_{digest}"


def _tables() -> tuple[sa.TableClause, sa.TableClause]:
    profiles = sa.table(
        "retrieval_profiles",
        sa.column("id", sa.String),
        sa.column("tenant_id", sa.String),
        sa.column("kb_id", sa.String),
        sa.column("profile_id", sa.String),
        sa.column("version", sa.Integer),
        sa.column("name", sa.String),
        sa.column("description", sa.Text),
        sa.column("status", sa.String),
        sa.column("config_json", sa.JSON),
        sa.column("config_hash", sa.String),
        sa.column("created_by", sa.String),
        sa.column("published_at", sa.DateTime(timezone=True)),
        sa.column("retired_at", sa.DateTime(timezone=True)),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    knowledge_bases = sa.table(
        "knowledge_bases",
        sa.column("id", sa.String),
        sa.column("retrieval_config_id", sa.String),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    return profiles, knowledge_bases


def _decoded_config(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    return dict(raw) if isinstance(raw, dict) else None


def upgrade() -> None:
    bind = op.get_bind()
    profiles, knowledge_bases = _tables()
    active_defaults = bind.execute(
        sa.select(profiles).join(
            knowledge_bases,
            sa.and_(
                knowledge_bases.c.id == profiles.c.kb_id,
                knowledge_bases.c.retrieval_config_id == profiles.c.id,
            ),
        ).where(
            profiles.c.profile_id == "rpf_default",
            profiles.c.status == "published",
            profiles.c.created_by == "system",
        )
    ).mappings().all()

    now = datetime.now(UTC)
    for source in active_defaults:
        config = _decoded_config(source["config_json"])
        if config is None or int(config.get("timeout_ms") or 0) != _OLD_TIMEOUT_MS:
            continue

        # Do not overwrite a user-created successor of the same logical profile.
        latest_version = bind.scalar(
            sa.select(sa.func.max(profiles.c.version)).where(
                profiles.c.tenant_id == source["tenant_id"],
                profiles.c.kb_id == source["kb_id"],
                profiles.c.profile_id == source["profile_id"],
            )
        )
        if int(latest_version or 0) != int(source["version"]):
            continue

        config["timeout_ms"] = _NEW_TIMEOUT_MS
        target_id = _profile_id(str(source["id"]))
        existing = bind.scalar(
            sa.select(profiles.c.id).where(profiles.c.id == target_id)
        )
        if existing is None:
            bind.execute(
                profiles.insert().values(
                    id=target_id,
                    tenant_id=source["tenant_id"],
                    kb_id=source["kb_id"],
                    profile_id=source["profile_id"],
                    version=int(source["version"]) + 1,
                    name=source["name"],
                    description=(
                        f"{_DESCRIPTION_PREFIX}{source['id']}] "
                        "内建本地检索预算升级：支持 BGE 冷启动及导入并发资源竞争。"
                    ),
                    status="published",
                    config_json=config,
                    config_hash=_config_hash(config),
                    created_by=_CREATED_BY,
                    published_at=now,
                    retired_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        bind.execute(
            profiles.update()
            .where(profiles.c.id == source["id"])
            .values(status="retired", retired_at=now, updated_at=now)
        )
        bind.execute(
            knowledge_bases.update()
            .where(
                knowledge_bases.c.id == source["kb_id"],
                knowledge_bases.c.retrieval_config_id == source["id"],
            )
            .values(retrieval_config_id=target_id, updated_at=now)
        )


def downgrade() -> None:
    bind = op.get_bind()
    profiles, knowledge_bases = _tables()
    migrated = bind.execute(
        sa.select(profiles).where(profiles.c.created_by == _CREATED_BY)
    ).mappings().all()
    now = datetime.now(UTC)
    for target in migrated:
        description = str(target["description"] or "")
        if not description.startswith(_DESCRIPTION_PREFIX) or "]" not in description:
            continue
        source_id = description[len(_DESCRIPTION_PREFIX) : description.index("]")]
        pointer = bind.scalar(
            sa.select(knowledge_bases.c.retrieval_config_id).where(
                knowledge_bases.c.id == target["kb_id"]
            )
        )
        # Preserve a profile that was superseded after this migration.
        if pointer != target["id"]:
            continue
        bind.execute(
            profiles.update()
            .where(profiles.c.id == source_id)
            .values(status="published", retired_at=None, updated_at=now)
        )
        bind.execute(
            knowledge_bases.update()
            .where(knowledge_bases.c.id == target["kb_id"])
            .values(retrieval_config_id=source_id, updated_at=now)
        )
        bind.execute(profiles.delete().where(profiles.c.id == target["id"]))
