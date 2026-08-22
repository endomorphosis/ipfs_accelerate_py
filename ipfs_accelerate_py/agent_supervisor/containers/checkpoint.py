"""Fenced container checkpoints and restart (EAAEF-054)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-checkpoint@1"
)


class CheckpointError(ValueError):
    """Checkpoint is not restart-safe."""


@dataclass(frozen=True)
class ContainerCheckpoint:
    attempt_id: str
    worktree_id: str
    fence_token: int
    lane_id: str
    owner_alive: bool
    semantic_delta_id: str = ""

    def __post_init__(self) -> None:
        if int(self.fence_token) < 0:
            raise CheckpointError("fence_token must be nonnegative")
        if not str(self.attempt_id).strip() or not str(self.lane_id).strip():
            raise CheckpointError("attempt and lane are required")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": CHECKPOINT_SCHEMA,
                "attempt_id": self.attempt_id,
                "worktree_id": self.worktree_id,
                "fence_token": int(self.fence_token),
                "lane_id": self.lane_id,
                "owner_alive": bool(self.owner_alive),
                "semantic_delta_id": self.semantic_delta_id,
            }
        )


def recover(checkpoint: ContainerCheckpoint, *, next_fence: int) -> ContainerCheckpoint:
    """Recover only a provably dead same-lane owner; require a later fence."""

    if checkpoint.owner_alive:
        raise CheckpointError("live owner cannot be recovered")
    if int(next_fence) <= int(checkpoint.fence_token):
        raise CheckpointError("restart requires a later fence")
    return ContainerCheckpoint(
        attempt_id=checkpoint.attempt_id,
        worktree_id=checkpoint.worktree_id,
        fence_token=int(next_fence),
        lane_id=checkpoint.lane_id,
        owner_alive=True,
        semantic_delta_id=checkpoint.semantic_delta_id,
    )
