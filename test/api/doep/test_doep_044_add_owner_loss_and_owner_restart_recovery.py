"""DOEP-044 owner-loss and owner-restart recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    DuplicateOwnerError,
    StaleOwnerError,
    _bind_external_quack_owner,
    recover_after_owner_loss,
    recover_after_owner_restart,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    ServerLifecycle,
)


BOARD = "agent-supervisor-direct-objective-and-event-driven-planning-v1"
SHARD = "doep-044-recovery"


def _server(
    *,
    generation: int = 7,
    server_id: str = "server:doep-044",
    fence_epoch: int = 7,
):
    identity = SimpleNamespace(
        server_id=server_id,
        store_id="doep-044-control",
        database_uuid="uuid-doep-044",
        generation=generation,
        fence_epoch=fence_epoch,
        secret_handle="handle:doep-044",
        listen_uri="quack:127.0.0.1:19495",
    )
    return SimpleNamespace(
        lifecycle=ServerLifecycle.READY,
        identity=identity,
        _identity=identity,
        _connection=object(),
        _owner=SimpleNamespace(held=True, fence_token="fence-doep-044"),
    )


def test_live_owner_is_not_stolen() -> None:
    server = _server()
    owner = _bind_external_quack_owner(
        owner_server=server, board_namespace=BOARD, shard_id=SHARD
    )
    lease = owner.lease()
    with pytest.raises(DuplicateOwnerError, match="not lost"):
        recover_after_owner_loss(
            previous_lease=lease,
            owner_server=server,
            board_namespace=BOARD,
            shard_id=SHARD,
        )


def test_owner_loss_without_replacement_does_not_mint_or_complete() -> None:
    server = _server()
    owner = _bind_external_quack_owner(
        owner_server=server, board_namespace=BOARD, shard_id=SHARD
    )
    lease = owner.lease()
    receipt = recover_after_owner_loss(
        previous_lease=lease,
        owner_server=None,
        board_namespace=BOARD,
        shard_id=SHARD,
    )
    assert receipt["recovered"] is False
    assert receipt["minted_generation"] is False
    assert receipt["completion_authority"] is False
    assert receipt["previous_generation"] == lease.generation


def test_same_generation_restart_rebinds_without_minting() -> None:
    server = _server()
    owner = _bind_external_quack_owner(
        owner_server=server, board_namespace=BOARD, shard_id=SHARD
    )
    lease = owner.lease()
    receipt = recover_after_owner_restart(
        previous_lease=lease,
        owner_server=server,
        board_namespace=BOARD,
        shard_id=SHARD,
    )
    assert receipt["recovered"] is True
    assert receipt["minted_generation"] is False
    assert receipt["completion_authority"] is False
    assert receipt["generation"] == lease.generation


def test_successor_generation_is_not_owner_loss_recovery() -> None:
    first = _server(generation=7, server_id="server:old")
    owner = _bind_external_quack_owner(
        owner_server=first, board_namespace=BOARD, shard_id=SHARD
    )
    lease = owner.lease()
    second = _server(generation=8, server_id="server:new", fence_epoch=8)
    with pytest.raises(StaleOwnerError, match="must not mint"):
        recover_after_owner_loss(
            previous_lease=lease,
            owner_server=second,
            board_namespace=BOARD,
            shard_id=SHARD,
        )
    with pytest.raises(StaleOwnerError):
        recover_after_owner_restart(
            previous_lease=lease,
            owner_server=second,
            board_namespace=BOARD,
            shard_id=SHARD,
        )
