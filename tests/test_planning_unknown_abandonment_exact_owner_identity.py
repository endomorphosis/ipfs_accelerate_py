"""Real gateway refuses bool/int owner identity before context admission.

The test supplies deliberately unadmitted context metadata and traps the policy
boundary. It qualifies exact typed owner binding, not launch custody or audit.
"""
from __future__ import annotations

import os

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy


def test_registered_generation_boolean_cannot_impersonate_actual_integer(tmp_path, monkeypatch):
    path = tmp_path / "exact-identity.duckdb"
    repository = IntentRepository(path)
    repository.close()
    connection = duckdb.connect(str(path))
    identity = {"server_id": "TEST-ONLY-exact-owner", "generation": 1, "store_id": "TEST-ONLY-exact-store"}
    gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=tmp_path / "exact-owner.sock",
                                           store_id=identity["store_id"], identity=identity)
    gateway.start()

    def must_not_reach_policy(*_args, **_kwargs):
        raise AssertionError("bool/int owner identity reached context policy admission")

    monkeypatch.setattr(policy, "scope", must_not_reach_policy)
    context = {"context_cid": "TEST-ONLY-unadmitted-context",
        "registered_owner_identity": {**identity, "generation": True},
        "registered_creator": {"pid": os.getpid(), "process_birth_id": typed.kernel_process_birth_id(os.getpid())}}
    # Demonstrate the exact Python coercion that the service must refuse.
    assert context["registered_owner_identity"] == identity
    try:
        with pytest.raises(typed.TypedStateOwnerAuthorizationError, match="registered owner/creator differs"):
            gateway.bind_planning_unknown_abandonment_service(expected_identity=identity, authority_context=context)
        assert gateway._planning_unknown_abandonment_service is None
        assert dict(gateway.identity) == identity
    finally:
        gateway.stop()
        connection.close()
