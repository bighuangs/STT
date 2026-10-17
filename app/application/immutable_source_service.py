"""Resolve and verify immutable local document-version source objects."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class ImmutableSourceUnavailableError(FileNotFoundError):
    """The version does not point at an allowed local original."""


class ImmutableSourceIntegrityError(RuntimeError):
    """The local bytes no longer match the immutable version fact."""


@dataclass(frozen=True, slots=True)
class ImmutableLocalSource:
    path: Path
    size: int
    sha256: str


def resolve_immutable_local_source(
    *,
    object_key: object,
    expected_size: object,
    expected_sha256: object,
    originals_root: Path,
) -> ImmutableLocalSource:
    """Return a verified original, failing closed outside ``originals_root``."""

    if not isinstance(object_key, str) or not object_key.strip():
        raise ImmutableSourceUnavailableError("source object is unavailable")
    try:
        root = originals_root.expanduser().resolve()
        candidate = Path(object_key).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ImmutableSourceUnavailableError("source object is unavailable") from exc
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise ImmutableSourceUnavailableError("source object is unavailable")

    if type(expected_size) is not int or expected_size < 0:
        raise ImmutableSourceIntegrityError("source version size is invalid")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_HEX.fullmatch(expected_sha256.lower()) is None
    ):
        raise ImmutableSourceIntegrityError("source version digest is invalid")

    digest = sha256()
    actual_size = 0
    try:
        with candidate.open("rb") as stream:
            while payload := stream.read(1024 * 1024):
                actual_size += len(payload)
                digest.update(payload)
    except OSError as exc:
        raise ImmutableSourceUnavailableError("source object is unavailable") from exc
    actual_sha256 = digest.hexdigest()
    if actual_size != expected_size or actual_sha256 != expected_sha256.lower():
        raise ImmutableSourceIntegrityError("source object integrity mismatch")
    return ImmutableLocalSource(
        path=candidate,
        size=actual_size,
        sha256=actual_sha256,
    )


__all__ = [
    "ImmutableLocalSource",
    "ImmutableSourceIntegrityError",
    "ImmutableSourceUnavailableError",
    "resolve_immutable_local_source",
]
