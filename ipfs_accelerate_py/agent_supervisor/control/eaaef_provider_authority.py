"""Host-local storage identities for EAAEF provider-review authority.

Provider authorizations are bound to both a repository CID and a baseline
commit.  A profile directory therefore cannot be reused after either binding
changes: ``initialize_local_profile(force=True)`` correctly loads and verifies
the old repository binding before rotating, so it is not a repository-migration
mechanism.

This module contains only deterministic path projection.  It never creates a
profile, reads a key, signs authority, or mutates lifecycle state.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

EAAEF_PROVIDER_LEGACY_PROFILE_DIR = (
    Path.home()
    / ".ipfs_accelerate"
    / "agent_supervisor"
    / "eaaef-route-profile"
)
EAAEF_PROVIDER_PROFILE_ROOT = (
    Path.home()
    / ".ipfs_accelerate"
    / "agent_supervisor"
    / "eaaef-route-profiles"
)
EAAEF_PROVIDER_LIFECYCLE_DIR = (
    Path.home()
    / ".ipfs_accelerate"
    / "agent_supervisor"
    / "eaaef-route-lifecycle"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")


def eaaef_provider_profile_storage_id(
    *,
    repository_cid: str,
    baseline_commit: str,
) -> str:
    """Return the bounded directory identity for one exact profile binding."""

    if _SHA256_RE.fullmatch(str(repository_cid or "")) is None:
        raise ValueError("EAAEF provider profile repository CID is invalid")
    if _GIT_OID_RE.fullmatch(str(baseline_commit or "")) is None:
        raise ValueError("EAAEF provider profile baseline commit is invalid")
    return hashlib.sha256(
        (
            "eaaef-provider-profile-storage-v1\0"
            + repository_cid
            + "\0"
            + baseline_commit
        ).encode("ascii")
    ).hexdigest()


def eaaef_provider_profile_directory(
    *,
    repository_cid: str,
    baseline_commit: str,
    profile_root: Path | None = None,
) -> Path:
    """Project one source/baseline binding below the private profile root."""

    root = Path(profile_root or EAAEF_PROVIDER_PROFILE_ROOT)
    return root / eaaef_provider_profile_storage_id(
        repository_cid=repository_cid,
        baseline_commit=baseline_commit,
    )


def eaaef_provider_profile_candidates(
    *,
    repository_cid: str,
    baseline_commit: str,
    profile_root: Path | None = None,
    legacy_profile_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Return the source-specific location followed by the legacy location."""

    projected = eaaef_provider_profile_directory(
        repository_cid=repository_cid,
        baseline_commit=baseline_commit,
        profile_root=profile_root,
    )
    legacy = Path(legacy_profile_dir or EAAEF_PROVIDER_LEGACY_PROFILE_DIR)
    return (projected,) if projected == legacy else (projected, legacy)


__all__ = [
    "EAAEF_PROVIDER_LEGACY_PROFILE_DIR",
    "EAAEF_PROVIDER_LIFECYCLE_DIR",
    "EAAEF_PROVIDER_PROFILE_ROOT",
    "eaaef_provider_profile_candidates",
    "eaaef_provider_profile_directory",
    "eaaef_provider_profile_storage_id",
]
