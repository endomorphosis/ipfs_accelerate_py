"""EAAEF-054: fenced checkpoint restart."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.containers.checkpoint import (
    CheckpointError,
    ContainerCheckpoint,
    recover,
)


def test_dead_owner_recovers_with_later_fence() -> None:
    ckpt = ContainerCheckpoint(
        attempt_id="att-1",
        worktree_id="wt-1",
        fence_token=3,
        lane_id="lane-0",
        owner_alive=False,
    )
    nxt = recover(ckpt, next_fence=4)
    assert nxt.fence_token == 4
    assert nxt.owner_alive is True
    assert nxt.lane_id == "lane-0"


def test_live_owner_and_stale_fence_fail() -> None:
    live = ContainerCheckpoint(
        attempt_id="att-1",
        worktree_id="wt-1",
        fence_token=3,
        lane_id="lane-0",
        owner_alive=True,
    )
    with pytest.raises(CheckpointError, match="live owner"):
        recover(live, next_fence=4)
    dead = ContainerCheckpoint(
        attempt_id="att-1",
        worktree_id="wt-1",
        fence_token=3,
        lane_id="lane-0",
        owner_alive=False,
    )
    with pytest.raises(CheckpointError, match="later fence"):
        recover(dead, next_fence=3)
