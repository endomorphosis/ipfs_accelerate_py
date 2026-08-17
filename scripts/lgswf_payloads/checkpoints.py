"""DaemonCheckpoint@1 and typed stale-stop lifecycle."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

TRANSITIONS = {
    "ready": ("running", "cancelled"),
    "running": ("checkpointed", "stale-stop", "completed", "cancelled"),
    "checkpointed": ("running", "stale-stop"),
    "stale-stop": (),
    "completed": (),
    "cancelled": (),
}
STALE_REASONS = (
    "stale-plan",
    "stale-root",
    "stale-lease",
    "stale-fence",
    "stale-state-owner-epoch",
    "stale-scope",
    "cancel",
    "already-accepted",
)


class CheckpointError(ValueError):
    """Lifecycle or checkpoint rejected."""


def transition(state: str, action: str) -> str:
    nxt = {
        "start": "running",
        "checkpoint": "checkpointed",
        "resume": "running",
        "stale-stop": "stale-stop",
        "complete": "completed",
        "cancel": "cancelled",
    }[action]
    if nxt not in TRANSITIONS[state]:
        raise CheckpointError(f"illegal {state}->{nxt}")
    return nxt


def write_checkpoint(record: Mapping[str, Any], path) -> Mapping[str, Any]:
    required = ("attempt_id", "packet_cid", "tree_cid", "fence_epoch", "effects", "obligations")
    missing = [name for name in required if name not in record]
    if missing:
        raise CheckpointError(f"checkpoint missing {missing}")
    if record.get("as_completion"):
        raise CheckpointError("checkpoint cannot be completion")
    payload = dict(record)
    path.write_text(repr(sorted(payload.items())), encoding="utf-8")
    return MappingProxyType({"ok": True, "path": str(path)})


def resume_checkpoint(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("corrupt"):
        raise CheckpointError("corrupt checkpoint")
    if record.get("stale_reason") in STALE_REASONS:
        return MappingProxyType({"resumed": False, "stopped": True, "reason": record["stale_reason"]})
    return MappingProxyType({"resumed": True, "stopped": False, "reason": ""})


def stale_stop(reason: str) -> Mapping[str, Any]:
    if reason not in STALE_REASONS:
        raise CheckpointError(f"unknown stale reason {reason}")
    return MappingProxyType({"stopped": True, "effect_after": False, "reason": reason})
