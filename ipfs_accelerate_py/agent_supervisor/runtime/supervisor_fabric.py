"""Supervisor capability and fenced coordination contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class SupervisorFabricError(ValueError):
    """A fenced coordination contract was violated."""


def issue_fence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if not record.get("supervisor_id") or not record.get("capability"):
        raise SupervisorFabricError("supervisor capability is required")
    if record.get("stale_epoch"):
        raise SupervisorFabricError("stale fence epoch")
    return MappingProxyType(
        {
            "supervisor_id": record["supervisor_id"],
            "capability": record["capability"],
            "epoch": int(record.get("epoch") or 1),
            "fenced": True,
        }
    )
