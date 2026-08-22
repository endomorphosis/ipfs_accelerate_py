"""EAAEF-084: work packets bind identities and forbid self-approval."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.external_work_packet import (
    ExternalWorkPacket,
    WorkPacketError,
)


def _packet(**overrides):
    payload = dict(
        goal_id="EAAEF-G090",
        task_id="EAAEF-084",
        repository_id="ipfs_accelerate_py",
        semantic_root="sha256:" + "a" * 64,
        write_scope=("ipfs_accelerate_py/agent_supervisor/todo_daemon/external_work_packet.py",),
        effect_scope=("isolated_write",),
        container_id="sha256:" + "b" * 64,
        lease_id="lease-1",
        fence_token=1,
        worker_principal="did:key:worker",
        reviewer_principal="did:key:reviewer",
    )
    payload.update(overrides)
    return ExternalWorkPacket(**payload)


def test_packet_binds_required_identities() -> None:
    packet = _packet()
    assert packet.self_approve is False
    assert packet.worker_principal != packet.reviewer_principal


def test_worker_cannot_self_approve() -> None:
    with pytest.raises(WorkPacketError, match="self-approve"):
        _packet(reviewer_principal="did:key:worker")
    with pytest.raises(WorkPacketError, match="self-approve"):
        _packet(self_approve=True)
