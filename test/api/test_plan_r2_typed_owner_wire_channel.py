from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
)
from ipfs_accelerate_py.agent_supervisor.planning import (
    external_agent_plan_r2 as r2,
)
from ipfs_accelerate_py.agent_supervisor.runtime.plan_r2_remote_owner import (
    PLAN_R2_TYPED_OWNER_CHANNEL_QUALIFICATION_STATUS,
    PlanR2CanonicalWireChannel,
    PlanR2RemoteOwnerError,
    TypedStateOwnerPlanR2CanonicalWireChannel,
    bind_plan_r2_process_remote_owner_gateway,
    bind_plan_r2_remote_exact_envelope_journal,
    bind_typed_state_owner_plan_r2_canonical_wire_channel,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    eaaef_plan_r2_owner_service as owner_service,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    eaaef_typed_owner_service as bootstrap_owner_service,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionStore,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerRemoteError,
)
from ipfs_accelerate_py.agent_supervisor.validation.plan_r2_remote_owner_admission import (
    PLAN_R2_REMOTE_OPERATIONS,
    PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability as _quack_capability,
)
from test.api.causal_federation.test_bootstrap_runtime import _migrate
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _signed_capability,
)
from test.api.test_eaaef_plan_r2_owner_service import (
    _joined_authorities,
    _seed_bootstrap_board_lease,
)
from test.api.test_external_agent_state_repository import (
    _provision_canonical_plan_r2_owner,
)
from test.api.test_plan_r2_remote_owner import _adapter


def test_typed_owner_canonical_channel_uses_real_authenticated_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_bundle = _signed_capability(
        owner_bindings={
            "store_id": "eaaef-plan-r2-canonical-channel-v1",
            "owner_generation": 1,
            "fence_epoch": 1,
        }
    )
    bootstrap_admission, capability, _context, authority = (
        _joined_authorities(
            tmp_path / "authority",
            capability_bundle=capability_bundle,
            plan_overrides={"expected_version": 0},
        )
    )
    authorization = authority["authorization"]
    assert isinstance(authorization, dict)
    admission = authority["admission"]
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )

    operational = tmp_path / "owner.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did=str(authority["principal_did"]),
        owner_status="ready",
    )
    seed = duckdb.connect(str(operational))
    try:
        _seed_bootstrap_board_lease(seed, capability)
        seed.execute("DELETE FROM state_servers")
        seed.execute("DELETE FROM server_epochs")
        seed.execute("DELETE FROM store_generations")
    finally:
        seed.close()

    server = build_server(
        database_path=operational,
        state_dir=tmp_path / "owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id=str(capability["store_id"]),
        transport=FakeQuackTransport(),
        capability_probe=_quack_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    server.clock = lambda: 4.0
    server.start()
    bootstrap_service = None
    plan_service = None
    owner_connection = None
    adapter = None
    try:
        bootstrap_service = (
            bootstrap_owner_service.bind_eaaef_typed_owner_command_service(
                owner_server=server,
                admission=bootstrap_admission,
            )
        )
        plan_service = (
            owner_service.bind_eaaef_plan_r2_borrowed_owner_service(
                owner_server=server,
                admission=admission,
                plan_r2_operational_capability=authority["plan_capability"],
                authorization=authorization,
                trusted_capability_reviewer_dids=[
                    authority["plan_reviewer_did"]
                ],
                trusted_operator_dids=[authority["operator_did"]],
                trusted_security_reviewer_dids=[authority["security_did"]],
            )
        )
        client_id = "eaaef-plan-r2-canonical-channel-client"
        process_birth_id = "birth:eaaef-plan-r2-canonical-channel-client"
        grant = server.issue_typed_client_grant(
            client_id=client_id,
            process_birth_id=process_birth_id,
            allowed_operations=tuple(sorted(PLAN_R2_REMOTE_OPERATIONS)),
            peer_pid=os.getpid(),
        )
        owner_connection = TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=grant,
            client_id=client_id,
            process_birth_id=process_birth_id,
            store_id=str(capability["store_id"]),
        )
        channel = bind_typed_state_owner_plan_r2_canonical_wire_channel(
            owner_connection=owner_connection,
            admission=admission,
        )
        assert isinstance(channel, PlanR2CanonicalWireChannel)
        assert channel.INTERFACE == PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE
        assert channel.request_channel_id == admission["request_channel_id"]
        assert channel.response_channel_id == admission["response_channel_id"]
        with pytest.raises(AttributeError):
            channel.request_channel_id = "substituted-channel"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            channel.response_channel_id = "substituted-channel"  # type: ignore[misc]
        assert channel.evidence() == {
            "interface": PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
            "qualification_status": (
                PLAN_R2_TYPED_OWNER_CHANNEL_QUALIFICATION_STATUS
            ),
            "production_admitted": False,
            "production_blocker": (
                owner_service.EAAEF_PLAN_R2_SINGLE_OWNER_PRODUCTION_BLOCKER
            ),
            "request_channel_id": admission["request_channel_id"],
            "response_channel_id": admission["response_channel_id"],
            "transport": "authenticated_typed_state_owner_connection",
            "canonical_bytes_only": True,
            "r1_operations_allowed": False,
            "generic_state_command_allowed": False,
            "database_authority_exposed": False,
            "filesystem_path_authority_exposed": False,
            "transport_token_exposed": False,
            "sql_exposed": False,
            "closes_shared_owner_connection": False,
            "attached": False,
        }
        for escaped_name in (
            "bootstrap_socket_path",
            "connection",
            "database_path",
            "execute",
            "execute_sql",
            "owner_connection",
            "socket_path",
            "token",
            "transport_token",
        ):
            assert not hasattr(channel, escaped_name)
        with pytest.raises(
            owner_service.EAAEFPlanR2OwnerServiceError,
            match="production no-go",
        ):
            channel.require_production_admission()

        gateway = bind_plan_r2_process_remote_owner_gateway(
            admission=admission,
            channel=channel,
            journal=bind_plan_r2_remote_exact_envelope_journal(
                store=PlanRevisionStore(tmp_path / "journal"),
                admission=admission,
            ),
        )
        adapter = _adapter(
            authority=authority,
            gateway=gateway,
            slot_allocator=iter(range(1, 10)).__next__,
            now_ms=NOW_MS,
        )
        adapter.attach()
        with pytest.raises(PlanR2RemoteOwnerError, match="signed wait"):
            channel.exchange(
                b"{}",
                request_cid="sha256:" + "0" * 64,
                maximum_wait_ms=int(admission["maximum_wait_ms"]) + 1,
            )
        prepared = adapter.prepare_authorized_plan_r2_transition(
            authorization
        )
        receipt = adapter.apply_authorized_plan_r2_transition(
            authorization,
            prepared,
        )
        observation = adapter.observe_authorized_plan_r2_transition(
            authorization,
            receipt,
        )
        assert prepared["schema"] == r2.PLAN_R2_PREPARED_PROJECTION_SCHEMA
        assert receipt["schema"] == r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA
        assert observation["schema"] == r2.PLAN_R2_STATE_OBSERVATION_SCHEMA
        assert plan_service.evidence()["operations"] == list(
            PLAN_R2_REMOTE_OPERATIONS
        )
        assert channel.evidence()["attached"] is True

        adapter.close()
        adapter = None
        assert owner_connection._closed is False  # noqa: SLF001
        with pytest.raises(TypedStateOwnerRemoteError) as still_open:
            owner_connection._request(  # noqa: SLF001 - liveness-only probe
                "outside-closed-protocol"
            )
        assert still_open.value.error_code == "protocol_denied"
    finally:
        if adapter is not None:
            adapter.close()
        if owner_connection is not None:
            owner_connection.close()
        if plan_service is not None:
            plan_service.close()
        if bootstrap_service is not None:
            bootstrap_service.close()
        server.stop()


def test_typed_owner_channel_rejects_unverified_admission_and_direct_build(
) -> None:
    with pytest.raises(PlanR2RemoteOwnerError, match="verified admission"):
        bind_typed_state_owner_plan_r2_canonical_wire_channel(
            owner_connection=object(),
            admission={},  # type: ignore[arg-type]
        )
    with pytest.raises(
        PlanR2RemoteOwnerError,
        match="exact authenticated client and verified admission",
    ):
        TypedStateOwnerPlanR2CanonicalWireChannel(  # type: ignore[arg-type]
            object(),
            client=object(),
            admission=object(),
        )
