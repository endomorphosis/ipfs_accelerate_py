"""ProvisionalSemanticState@1 — attempt-local roots cannot become canonical."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

SCHEMA = "lgswf/provisional-semantic-state@1"
CANONICAL_PUBLISHER = "post-merge-supervisor-refresh"


class ProvisionalStateError(ValueError):
    """A prohibited provisional-to-canonical transition was attempted."""


def bind_provisional_root(record: Mapping[str, Any]) -> Mapping[str, Any]:
    required = ("root_cid", "task_id", "attempt_id", "worktree_tree")
    missing = [name for name in required if not record.get(name)]
    if missing:
        raise ProvisionalStateError(f"provisional root missing {missing}")
    return MappingProxyType(
        {
            "schema": SCHEMA,
            "root_cid": record["root_cid"],
            "task_id": record["task_id"],
            "attempt_id": record["attempt_id"],
            "worktree_tree": record["worktree_tree"],
            "usable_for": ("impact", "context", "verification"),
            "canonical": False,
        }
    )


def publish_canonical(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("source") in {"provisional", "worker"}:
        raise ProvisionalStateError("provisional/worker root cannot become canonical")
    if record.get("stale_attempt"):
        raise ProvisionalStateError("stale attempt cannot publish canonical root")
    if record.get("worktree_tree") != record.get("accepted_merge_tree"):
        raise ProvisionalStateError("wrong worktree/tree for canonical publish")
    if not record.get("fresh_datasets_rescan") or not record.get("delta_receipt"):
        raise ProvisionalStateError("canonical publish requires rescan and delta receipt")
    if record.get("publisher") != CANONICAL_PUBLISHER:
        raise ProvisionalStateError("canonical publisher role is prohibited here")
    return MappingProxyType(
        {
            "schema": SCHEMA,
            "canonical": True,
            "root_cid": record["root_cid"],
            "publisher": CANONICAL_PUBLISHER,
        }
    )


def authority_transitions() -> tuple[dict[str, str], ...]:
    return (
        {"from": "worktree-change", "to": "provisional", "allowed": "yes"},
        {"from": "provisional", "to": "canonical", "allowed": "no"},
        {"from": "worker-result", "to": "canonical", "allowed": "no"},
        {"from": "accepted-merge+rescan", "to": "canonical", "allowed": "yes"},
    )
