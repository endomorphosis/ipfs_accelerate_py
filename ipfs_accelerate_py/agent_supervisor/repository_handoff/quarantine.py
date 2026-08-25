"""Quarantine reconstructed repositories before onboarding (EAAEF-022)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import urlparse


QUARANTINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-handoff-quarantine@1"
)
MAX_OBJECTS: Final[int] = 1_000_000
MAX_BYTES: Final[int] = 8 * 1024 * 1024 * 1024


class QuarantineError(ValueError):
    """Reconstructed repository failed quarantine."""


@dataclass(frozen=True)
class QuarantineVerdict:
    admitted: bool
    reason_code: str
    tree_id: str
    object_count: int
    object_bytes: int

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": QUARANTINE_SCHEMA,
                "admitted": self.admitted,
                "reason_code": self.reason_code,
                "tree_id": self.tree_id,
                "object_count": int(self.object_count),
                "object_bytes": int(self.object_bytes),
            }
        )


def quarantine_repository(
    *,
    tree_id: str,
    object_count: int,
    object_bytes: int,
    origin_url: str = "",
    hooks_enabled: bool = False,
    symlink_escape: bool = False,
    claimed_tree_id: str = "",
) -> QuarantineVerdict:
    if not str(tree_id).strip():
        raise QuarantineError("tree_id is required")
    if hooks_enabled:
        return QuarantineVerdict(False, "enabled_hooks", tree_id, object_count, object_bytes)
    if symlink_escape:
        return QuarantineVerdict(False, "symlink_escape", tree_id, object_count, object_bytes)
    if int(object_count) > MAX_OBJECTS or int(object_bytes) > MAX_BYTES:
        return QuarantineVerdict(False, "unbounded_objects", tree_id, object_count, object_bytes)
    if origin_url:
        parsed = urlparse(origin_url)
        if parsed.scheme in {"file", "unix"} or origin_url.startswith("/"):
            return QuarantineVerdict(False, "host_path_origin", tree_id, object_count, object_bytes)
    if claimed_tree_id and claimed_tree_id != tree_id:
        return QuarantineVerdict(False, "tree_identity_mismatch", tree_id, object_count, object_bytes)
    return QuarantineVerdict(True, "admitted", tree_id, object_count, object_bytes)
