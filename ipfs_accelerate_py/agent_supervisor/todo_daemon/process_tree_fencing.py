"""Process-tree fencing for integrated security recovery.

Cleanup may terminate an owned descendant tree, including children that
opened a new session.  It must not delete immutable evidence, published
artifacts, hidden tests, or another task's worktree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .core import pid_alive, terminate_pid_tree


PROCESS_TREE_FENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/process-tree-fence@1"
)
PROCESS_TREE_FENCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/process-tree-fence-receipt@1"
)

IMMUTABLE_EVIDENCE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "accepted-work.json",
        "events.jsonl",
        "hidden_labels",
        "hidden-tests",
        "published",
        "source_release",
    }
)
FORBIDDEN_CLEANUP_PREFIXES: Final[tuple[str, ...]] = (
    "delete_evidence",
    "delete_hidden",
    "delete_published",
    "delete_source",
)


class ProcessTreeFenceError(ValueError):
    """Unsafe process-tree cleanup was refused."""


class ProcessTreeFenceDecision(str, Enum):
    FENCED = "fenced"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ProcessTreeFenceReceipt:
    """Outcome of one owned-tree fence.  Not a cleanup permit for evidence."""

    decision: ProcessTreeFenceDecision
    root_pid: int
    terminated: bool
    reasons: tuple[str, ...]
    preserved_paths: tuple[str, ...]
    schema: str = PROCESS_TREE_FENCE_RECEIPT_SCHEMA

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "preserved_paths": list(self.preserved_paths),
            "reasons": list(self.reasons),
            "root_pid": self.root_pid,
            "schema": self.schema,
            "terminated": self.terminated,
        }


def _is_immutable_evidence(path: Path | str) -> bool:
    selected = Path(path)
    parts = {part.casefold().replace(" ", "-") for part in selected.parts}
    if parts & IMMUTABLE_EVIDENCE_NAMES:
        return True
    name = selected.name.casefold()
    return name in IMMUTABLE_EVIDENCE_NAMES or name.endswith(".evidence.json")


def reject_unsafe_cleanup(paths: Sequence[Path | str], *, flags: Sequence[str] = ()) -> None:
    """Fail closed when cleanup would destroy immutable evidence."""

    reasons: list[str] = []
    for flag in flags:
        text = str(flag or "").strip().casefold()
        if any(text.startswith(prefix) for prefix in FORBIDDEN_CLEANUP_PREFIXES):
            reasons.append("unsafe_cleanup")
    for path in paths:
        if _is_immutable_evidence(path):
            reasons.append("unsafe_cleanup")
    if reasons:
        raise ProcessTreeFenceError("unsafe cleanup refused: " + ",".join(dict.fromkeys(reasons)))


def fence_process_tree(
    pid: int,
    *,
    grace_seconds: float = 0.2,
    freeze_first: bool = True,
    require_gone: bool = False,
    cleanup_paths: Sequence[Path | str] = (),
    cleanup_flags: Sequence[str] = (),
) -> ProcessTreeFenceReceipt:
    """Fence an owned tree without deleting immutable evidence."""

    reject_unsafe_cleanup(cleanup_paths, flags=cleanup_flags)
    if pid <= 1:
        raise ProcessTreeFenceError("process-tree fence requires a positive non-init pid")
    terminated = terminate_pid_tree(
        pid,
        grace_seconds=grace_seconds,
        freeze_first=freeze_first,
        require_gone=require_gone,
    )
    preserved = tuple(str(Path(path)) for path in cleanup_paths if Path(path).exists())
    return ProcessTreeFenceReceipt(
        decision=ProcessTreeFenceDecision.FENCED,
        root_pid=pid,
        terminated=bool(terminated) or not pid_alive(pid),
        reasons=(),
        preserved_paths=preserved,
    )


__all__ = (
    "IMMUTABLE_EVIDENCE_NAMES",
    "PROCESS_TREE_FENCE_RECEIPT_SCHEMA",
    "PROCESS_TREE_FENCE_SCHEMA",
    "ProcessTreeFenceDecision",
    "ProcessTreeFenceError",
    "ProcessTreeFenceReceipt",
    "fence_process_tree",
    "reject_unsafe_cleanup",
)
