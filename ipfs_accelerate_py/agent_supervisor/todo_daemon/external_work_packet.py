"""Bounded subagent work packets (EAAEF-084)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


PACKET_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-work-packet@1"


class WorkPacketError(ValueError):
    """Work packet is incomplete or self-approving."""


@dataclass(frozen=True)
class ExternalWorkPacket:
    goal_id: str
    task_id: str
    repository_id: str
    semantic_root: str
    write_scope: tuple[str, ...]
    effect_scope: tuple[str, ...]
    container_id: str
    lease_id: str
    fence_token: int
    worker_principal: str
    reviewer_principal: str
    self_approve: bool = False

    def __post_init__(self) -> None:
        required = (
            "goal_id",
            "task_id",
            "repository_id",
            "semantic_root",
            "container_id",
            "lease_id",
            "worker_principal",
            "reviewer_principal",
        )
        for name in required:
            if not str(getattr(self, name) or "").strip():
                raise WorkPacketError(f"{name} is required")
        if self.self_approve or self.worker_principal == self.reviewer_principal:
            raise WorkPacketError("workers cannot self-approve")
        if int(self.fence_token) < 0:
            raise WorkPacketError("fence_token must be nonnegative")
        object.__setattr__(self, "write_scope", tuple(self.write_scope))
        object.__setattr__(self, "effect_scope", tuple(self.effect_scope))

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": PACKET_SCHEMA,
                "goal_id": self.goal_id,
                "task_id": self.task_id,
                "repository_id": self.repository_id,
                "semantic_root": self.semantic_root,
                "write_scope": list(self.write_scope),
                "effect_scope": list(self.effect_scope),
                "container_id": self.container_id,
                "lease_id": self.lease_id,
                "fence_token": int(self.fence_token),
                "worker_principal": self.worker_principal,
                "reviewer_principal": self.reviewer_principal,
                "self_approve": False,
            }
        )
