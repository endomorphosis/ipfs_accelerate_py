from __future__ import annotations

import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping

import duckdb
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
)
from ipfs_accelerate_py.agent_supervisor.planning import (
    external_agent_plan_r2 as r2,
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
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    eaaef_board_scheduler_lease_seed,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.external_agent_state_repository import (
    PLAN_R2_OWNER_GATEWAY_INTERFACE,
    PlanR2OwnerAdapterUnavailable,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandFabric,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    quack_daemon_operation_intent,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS,
    TypedStateOwnerConnection,
    TypedStateOwnerGateway,
    TypedStateOwnerRemoteError,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_single_owner_review_request as single_owner_review,
)
from ipfs_accelerate_py.agent_supervisor.validation.plan_r2_remote_owner_admission import (
    PLAN_R2_REMOTE_OPERATIONS,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability as _quack_capability,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _migrate,
)
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _signed_capability,
)
from test.api.test_eaaef_lane_gateway_runtime import (
    _admission_bundle,
    _envelope,
)
from test.api.test_external_agent_state_repository import (
    _provision_canonical_plan_r2_owner,
)
from test.api.test_plan_r2_remote_owner import (
    _adapter,
    _admitted_authority,
)


class _GuardedOwnerConnection:
    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._guard = threading.Lock()
        self._active = 0
        self.maximum_active = 0
        self.statements: list[str] = []

    def execute(self, statement: str, *args: object) -> object:
        normalized = " ".join(statement.strip().upper().split())
        self.statements.append(normalized)
        forbidden = ("ATTACH", "DETACH", "INSTALL", "LOAD")
        if normalized.startswith(forbidden):
            raise AssertionError(
                "authoritative Plan-R2 connection attempted catalog mutation"
            )
        begins = normalized == "BEGIN TRANSACTION"
        if begins:
            with self._guard:
                self._active += 1
                self.maximum_active = max(self.maximum_active, self._active)
            time.sleep(0.005)
        try:
            return self._connection.execute(statement, *args)
        except BaseException:
            if begins:
                with self._guard:
                    self._active -= 1
            raise

    def commit(self) -> None:
        try:
            self._connection.commit()
        finally:
            with self._guard:
                self._active -= 1

    def rollback(self) -> None:
        try:
            self._connection.rollback()
        finally:
            with self._guard:
                if self._active:
                    self._active -= 1


class _TestOnlyPlanR2Gateway:
    """Test harness for a cutover interface production has not signed yet."""

    INTERFACE = PLAN_R2_OWNER_GATEWAY_INTERFACE

    def __init__(
        self,
        service: owner_service.EAAEFPlanR2BorrowedOwnerService,
    ) -> None:
        self._service = service

    def submit_authorized_plan_r2_operation(
        self,
        envelope: object,
        operation_payload: object,
    ) -> object:
        return self._service.submit_authorized_plan_r2_operation(
            envelope,  # type: ignore[arg-type]
            operation_payload,  # type: ignore[arg-type]
        )


def _seed_bootstrap_board_lease(
    connection: object,
    capability: dict[str, object],
) -> None:
    seed = eaaef_board_scheduler_lease_seed(
        board_namespace=str(capability["board_namespace"]),
        shard_id=str(capability["shard_id"]),
        lease_id=str(capability["lease_id"]),
        principal_did=str(capability["command_principal_did"]),
        owner_session_id=str(capability["owner_session_id"]),
        owner_generation=int(capability["owner_generation"]),
        fencing_token=int(capability["fencing_token"]),
        fence_epoch=int(capability["fence_epoch"]),
        issued_at_ms=NOW_MS - 1_000,
        expires_at_ms=NOW_MS + 40_000,
    )
    row = seed["row"]
    columns = (
        "task_cid",
        "claim_cid",
        "resolution_cid",
        "claimant_did",
        "logical_epoch",
        "fencing_token",
        "expires_at_ms",
        "attempt",
        "state",
        "started_at_ms",
        "release_reason",
        "retry_not_before_ms",
        "owner_session_id",
        "fence_epoch",
        "revision",
        "extension_schema",
        "extension_json",
    )
    connection.execute(
        "INSERT INTO leases("
        + ",".join(columns)
        + ") VALUES ("
        + ",".join("?" for _ in columns)
        + ")",
        [
            (
                json.dumps(
                    row[name],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if name == "extension_json"
                else row[name]
            )
            for name in columns
        ],
    )


def _joined_authorities(
    root: Path,
    *,
    capability_bundle: tuple[
        dict[str, object], dict[str, object]
    ]
    | None = None,
    plan_overrides: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], dict[str, object], dict[str, object]]:
    bootstrap_admission, capability, context = _admission_bundle(
        root,
        capability_bundle=capability_bundle,
    )
    policy = context["policy"]
    owner_bindings = {
        "board_namespace": capability["board_namespace"],
        "source_head": capability["source_head"],
        "source_tree": capability["source_tree"],
        "source_generation_cid": capability["configuration_root"],
        "bootstrap_admission_cid": capability[
            "bootstrap_admission_receipt_cid"
        ],
        "r1_launch_capsule_cid": capability[
            "configured_board_capsule_cid"
        ],
        "quack_command_fabric_qualification_cid": capability[
            "command_fabric_qualification_cid"
        ],
        "owner_principal_did": capability["owner_principal_did"],
        "shard_id": capability["shard_id"],
        "store_id": capability["store_id"],
        "owner_generation": capability["owner_generation"],
        "expected_epoch": 4,
        "fencing_token": policy.fence_epoch,
        "expected_active_plan_root_cid": capability[
            "active_plan_root_cid"
        ],
        "expected_active_plan_cid": bootstrap_admission[
            "active_plan_revision_cid"
        ],
        "expected_active_plan_revision": capability[
            "active_plan_revision"
        ],
        "successor_plan_cid": "sha256:" + "a" * 64,
    }
    if plan_overrides is not None:
        owner_bindings.update(plan_overrides)
    plan_authority = _admitted_authority(
        NOW_MS,
        owner_bindings=owner_bindings,
    )
    return bootstrap_admission, capability, context, plan_authority


def _bind_joined_services(
    gateway: TypedStateOwnerGateway,
    *,
    bootstrap_admission: object,
    plan_authority: dict[str, object],
) -> tuple[
    bootstrap_owner_service.EAAEFTypedOwnerCommandService,
    owner_service.EAAEFPlanR2BorrowedOwnerService,
]:
    gateway.start()
    bootstrap_service = (
        gateway._bind_eaaef_typed_owner_command_service_from_server(  # noqa: SLF001
            admission=bootstrap_admission,
        )
    )
    plan_service = gateway._bind_eaaef_plan_r2_owner_service_from_server(  # noqa: SLF001
        admission=plan_authority["admission"],
        plan_r2_operational_capability=plan_authority[
            "plan_capability"
        ],
        authorization=plan_authority["authorization"],
        trusted_capability_reviewer_dids=[
            plan_authority["plan_reviewer_did"]
        ],
        trusted_operator_dids=[plan_authority["operator_did"]],
        trusted_security_reviewer_dids=[
            plan_authority["security_did"]
        ],
    )
    return bootstrap_service, plan_service


def test_plan_r2_borrows_one_owner_without_sidecars_or_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ms = NOW_MS
    bootstrap_admission, capability, context, authority = (
        _joined_authorities(tmp_path / "authority")
    )
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: now_ms * 1_000_000,
    )
    authorization = authority["authorization"]
    assert isinstance(authorization, dict)
    operational = tmp_path / "owner-private.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did=str(authority["principal_did"]),
        owner_status="ready",
    )
    seed = duckdb.connect(str(operational))
    try:
        _seed_bootstrap_board_lease(seed, capability)
    finally:
        seed.close()

    open_count = 0
    native_connect = duckdb.connect

    def counted_connect(*args: object, **kwargs: object) -> object:
        nonlocal open_count
        open_count += 1
        return native_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", counted_connect)
    raw = duckdb.connect(str(operational))
    assert open_count == 1
    guarded = _GuardedOwnerConnection(raw)
    gateway = TypedStateOwnerGateway(
        connection=guarded,
        socket_path=tmp_path / "typed-owner.sock",
        store_id=str(authorization["store_id"]),
        identity={
            "server_id": "server:plan-r2",
            "store_id": authorization["store_id"],
            "database_uuid": "123e4567-e89b-12d3-a456-426614174000",
            "generation": authorization["owner_generation"],
            "fence_epoch": authorization["fencing_token"],
        },
    )
    bootstrap_service, service = _bind_joined_services(
        gateway,
        bootstrap_admission=bootstrap_admission,
        plan_authority=authority,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("single-owner Plan-R2 attempted a database or fabric open")

    monkeypatch.setattr(duckdb, "connect", forbidden)
    monkeypatch.setattr(QuackCommandFabric, "__init__", forbidden)
    monkeypatch.setattr(QuackCommandFabric, "start", forbidden)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_duckdb_connection",
        forbidden,
    )
    with pytest.raises(
        PlanR2OwnerAdapterUnavailable,
        match="atomic_owner_operation_unavailable",
    ):
        _adapter(
            authority=authority,
            gateway=service,  # type: ignore[arg-type]
            slot_allocator=iter([99]).__next__,
            now_ms=now_ms,
        )
    adapter = _adapter(
        authority=authority,
        gateway=_TestOnlyPlanR2Gateway(service),  # type: ignore[arg-type]
        slot_allocator=iter(range(1, 20)).__next__,
        now_ms=now_ms,
    )
    adapter.attach()
    try:
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
        launch = r2.validate_plan_r2_launch_transition(
            repository=adapter,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=observation,
            trusted_operator_dids=[authority["operator_did"]],
            trusted_security_reviewer_dids=[authority["security_did"]],
            now_ms=time.time_ns() // 1_000_000,
        )
        assert launch["valid"] is True
        assert guarded.maximum_active == 1
        assert open_count == 1
        assert not any(
            statement.startswith(("ATTACH", "DETACH", "INSTALL", "LOAD"))
            for statement in guarded.statements
        )
        assert sorted(path.name for path in tmp_path.glob("*.duckdb")) == [
            "owner-private.duckdb"
        ]
        evidence = service.evidence()
        assert evidence["operations"] == [
            "plan_r2.prepare",
            "plan_r2.apply",
            "plan_r2.observe",
        ]
        assert evidence["production_admitted"] is False
        assert evidence["bound_by_typed_state_owner_gateway"] is True
        assert evidence["opens_database"] is False
        assert evidence["local_sidecar_enabled"] is False
    finally:
        adapter.close()
        gateway.stop()
        raw.close()


def test_plan_r2_real_quack_owner_socket_executes_exact_three_grants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_bundle = _signed_capability(
        owner_bindings={
            "store_id": "eaaef-plan-r2-real-wire-v1",
            "owner_generation": 1,
            "fence_epoch": 1,
        }
    )
    bootstrap_admission, capability, context, authority = (
        _joined_authorities(
            tmp_path / "authority",
            capability_bundle=capability_bundle,
            plan_overrides={"expected_version": 0},
        )
    )
    authorization = authority["authorization"]
    assert isinstance(authorization, dict)
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    operational = tmp_path / "real-owner.duckdb"
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
    identity = server.start()
    assert identity.generation == identity.fence_epoch == 1
    bootstrap_service = (
        bootstrap_owner_service.bind_eaaef_typed_owner_command_service(
            owner_server=server,
            admission=bootstrap_admission,
        )
    )
    plan_service = owner_service.bind_eaaef_plan_r2_borrowed_owner_service(
        owner_server=server,
        admission=authority["admission"],
        plan_r2_operational_capability=authority["plan_capability"],
        authorization=authorization,
        trusted_capability_reviewer_dids=[
            authority["plan_reviewer_did"]
        ],
        trusted_operator_dids=[authority["operator_did"]],
        trusted_security_reviewer_dids=[authority["security_did"]],
    )
    assert server._command_gateway is not None  # noqa: SLF001
    assert (  # noqa: SLF001 - one explicit owner-wide boundary
        server._command_gateway._transaction_lock is server._lock
    )
    assert (  # noqa: SLF001
        bootstrap_service._transaction_lock
        is plan_service._transaction_lock
        is server._lock
    )
    client_id = "eaaef-plan-r2-real-wire-client"
    process_birth_id = "birth:eaaef-plan-r2-real-wire-client"
    token = server.issue_typed_client_grant(
        client_id=client_id,
        process_birth_id=process_birth_id,
        allowed_operations=tuple(sorted(PLAN_R2_REMOTE_OPERATIONS)),
        peer_pid=os.getpid(),
    )
    owner_connection = TypedStateOwnerConnection(
        socket_path=server.typed_command_socket_path(),
        token=token,
        client_id=client_id,
        process_birth_id=process_birth_id,
        store_id=str(capability["store_id"]),
    )
    r1_lookup_id = "eaaef-r1-contention-wire-client"
    r1_lookup_birth = "birth:eaaef-r1-contention-wire-client"
    r1_lookup_token = server.issue_typed_client_grant(
        client_id=r1_lookup_id,
        process_birth_id=r1_lookup_birth,
        allowed_operations=("eaaef.command.lookup",),
        peer_pid=os.getpid(),
    )
    r1_connections = [
        TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=r1_lookup_token,
            client_id=r1_lookup_id,
            process_birth_id=r1_lookup_birth,
            store_id=str(capability["store_id"]),
        )
        for _index in range(31)
    ]
    r1_envelopes = []
    for serial, operation in enumerate(
        sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS),
        start=100,
    ):
        intent = dict(
            quack_daemon_operation_intent(
                gateway_binding_cid=str(capability["gateway_binding_cid"]),
                operational_capability_cid=str(capability["capability_cid"]),
                operation=operation,
                arguments={},
            )
        )
        r1_envelopes.append(
            _envelope(intent, capability, context, serial=serial)
        )
    plan_client = owner_service.bind_eaaef_plan_r2_typed_owner_command_client(
        owner_connection=owner_connection,
        admission=authority["admission"],
    )
    adapter = _adapter(
        authority=authority,
        gateway=plan_client,  # type: ignore[arg-type]
        slot_allocator=iter(range(1, 20)).__next__,
        now_ms=NOW_MS,
    )
    malformed = {
        "remote_capability_cid": plan_service.remote_capability_cid,
        "plan_r2_operational_capability_cid": (
            plan_service.operational_capability_cid
        ),
        "plan_r2_authorization_cid": plan_service.authorization_cid,
        "envelope": {},
        "operation_payload": {"operation": "plan_r2.prepare"},
    }
    try:
        with pytest.raises(TypedStateOwnerRemoteError) as extra:
            owner_connection._request(  # noqa: SLF001 - malformed-wire probe
                "plan_r2.prepare",
                **malformed,
                unexpected="rejected",
            )
        assert extra.value.error_code == "protocol_denied"
        with pytest.raises(TypedStateOwnerRemoteError) as wrong_authority:
            owner_connection._request(  # noqa: SLF001 - authority probe
                "plan_r2.prepare",
                **{
                    **malformed,
                    "remote_capability_cid": "sha256:" + "0" * 64,
                },
            )
        assert wrong_authority.value.error_code == "authorization_denied"

        prepare_only_id = "eaaef-plan-r2-prepare-only"
        prepare_only_birth = "birth:eaaef-plan-r2-prepare-only"
        prepare_only_token = server.issue_typed_client_grant(
            client_id=prepare_only_id,
            process_birth_id=prepare_only_birth,
            allowed_operations=("plan_r2.prepare",),
            peer_pid=os.getpid(),
        )
        prepare_only = TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=prepare_only_token,
            client_id=prepare_only_id,
            process_birth_id=prepare_only_birth,
            store_id=str(capability["store_id"]),
        )
        try:
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                prepare_only._request(  # noqa: SLF001 - grant probe
                    "plan_r2.apply",
                    **{
                        **malformed,
                        "operation_payload": {
                            "operation": "plan_r2.apply"
                        },
                    },
                )
            assert denied.value.error_code == "authorization_denied"
        finally:
            prepare_only.close()

        class CapturedPlanSubmission(RuntimeError):
            pass

        class CapturingPlanGateway:
            INTERFACE = PLAN_R2_OWNER_GATEWAY_INTERFACE

            def submit_authorized_plan_r2_operation(
                self,
                envelope: object,
                operation_payload: object,
            ) -> object:
                self.envelope = envelope
                self.operation_payload = operation_payload
                raise CapturedPlanSubmission

        capture = CapturingPlanGateway()
        capture_adapter = _adapter(
            authority=authority,
            gateway=capture,  # type: ignore[arg-type]
            slot_allocator=iter(range(1, 20)).__next__,
            now_ms=NOW_MS,
        )
        with pytest.raises(CapturedPlanSubmission):
            capture_adapter.prepare_authorized_plan_r2_transition(
                authorization
            )
        race_id = "eaaef-plan-r2-revoked-waiter"
        race_birth = "birth:eaaef-plan-r2-revoked-waiter"
        race_token, race_grant = server.issue_typed_client_grant_record(
            client_id=race_id,
            process_birth_id=race_birth,
            allowed_operations=("plan_r2.prepare",),
            peer_pid=os.getpid(),
        )
        race_connection = TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=race_token,
            client_id=race_id,
            process_birth_id=race_birth,
            store_id=str(capability["store_id"]),
        )
        original_grant_check = TypedStateOwnerGateway._require_active_grant
        first_grant_check = threading.Event()
        grant_check_count = 0

        def observed_grant_check(
            gateway: TypedStateOwnerGateway,
            grant: object,
            *,
            peer_identity: tuple[int, int, int],
        ) -> None:
            nonlocal grant_check_count
            if getattr(grant, "grant_id", "") == race_grant.grant_id:
                grant_check_count += 1
                first_grant_check.set()
            original_grant_check(
                gateway,
                grant,  # type: ignore[arg-type]
                peer_identity=peer_identity,
            )

        monkeypatch.setattr(
            TypedStateOwnerGateway,
            "_require_active_grant",
            observed_grant_check,
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                with server._lock:  # noqa: SLF001 - force a lock waiter
                    waiting = executor.submit(
                        race_connection.submit_eaaef_plan_r2_operation,
                        capture.envelope,
                        capture.operation_payload,
                        remote_capability_cid=(
                            plan_service.remote_capability_cid
                        ),
                        plan_r2_operational_capability_cid=(
                            plan_service.operational_capability_cid
                        ),
                        plan_r2_authorization_cid=(
                            plan_service.authorization_cid
                        ),
                    )
                    assert first_grant_check.wait(timeout=5)
                    assert server._command_gateway is not None  # noqa: SLF001
                    server._command_gateway.revoke_grant(  # noqa: SLF001
                        race_grant.grant_id
                    )
                with pytest.raises(TypedStateOwnerRemoteError) as revoked:
                    waiting.result(timeout=10)
                assert revoked.value.error_code == "authorization_denied"
            assert grant_check_count >= 2
        finally:
            race_connection.close()

        adapter.attach()

        def contend_with_all_r1_sockets(
            plan_phase: Callable[[], object],
        ) -> object:
            with ThreadPoolExecutor(max_workers=32) as executor:
                r1_reads = [
                    executor.submit(
                        connection.lookup_eaaef_authorized_operation_receipt,
                        envelope,
                        merge_admission_cid=bootstrap_service.admission_cid,
                        operational_capability_cid=(
                            bootstrap_service.operational_capability_cid
                        ),
                    )
                    for connection, envelope in zip(
                        r1_connections,
                        r1_envelopes,
                        strict=True,
                    )
                ]
                plan_future = executor.submit(plan_phase)
                assert [
                    future.result(timeout=15) for future in r1_reads
                ] == [None] * 31
                return plan_future.result(timeout=15)

        prepared = contend_with_all_r1_sockets(
            lambda: adapter.prepare_authorized_plan_r2_transition(
                authorization
            )
        )
        receipt = contend_with_all_r1_sockets(
            lambda: adapter.apply_authorized_plan_r2_transition(
                authorization,
                prepared,
            )
        )
        observation = contend_with_all_r1_sockets(
            lambda: adapter.observe_authorized_plan_r2_transition(
                authorization,
                receipt,
            )
        )
        assert prepared["schema"] == r2.PLAN_R2_PREPARED_PROJECTION_SCHEMA
        assert receipt["schema"] == r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA
        assert observation["schema"] == r2.PLAN_R2_STATE_OBSERVATION_SCHEMA
        assert plan_service.evidence()["operations"] == list(
            PLAN_R2_REMOTE_OPERATIONS
        )
        assert plan_service.evidence()["production_admitted"] is False
        with pytest.raises(
            owner_service.EAAEFPlanR2OwnerServiceError,
            match="production no-go",
        ):
            plan_service.require_production_admission()
        with pytest.raises(
            owner_service.EAAEFPlanR2OwnerServiceError,
            match="production no-go",
        ):
            plan_client.require_production_admission()
        for escaped_name in (
            "connection",
            "token",
            "socket_path",
            "database_path",
            "execute",
            "_request",
        ):
            assert not hasattr(plan_client, escaped_name)
        assert not (
            set(PLAN_R2_REMOTE_OPERATIONS)
            | set(
                bootstrap_owner_service.EAAEF_TYPED_OWNER_COMMAND_OPERATIONS
            )
        ) & set(SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS)
        bootstrap_service.close()
        with pytest.raises(
            bootstrap_owner_service.EAAEFTypedOwnerServiceError,
            match="closed",
        ):
            owner_service.bind_eaaef_plan_r2_borrowed_owner_service(
                owner_server=server,
                admission=authority["admission"],
                plan_r2_operational_capability=authority[
                    "plan_capability"
                ],
                authorization=authorization,
                trusted_capability_reviewer_dids=[
                    authority["plan_reviewer_did"]
                ],
                trusted_operator_dids=[authority["operator_did"]],
                trusted_security_reviewer_dids=[
                    authority["security_did"]
                ],
            )
    finally:
        adapter.close()
        for connection in r1_connections:
            connection.close()
        owner_connection.close()
        server.stop()
    with pytest.raises(
        owner_service.EAAEFPlanR2OwnerServiceError,
        match="closed",
    ):
        plan_service.submit_authorized_plan_r2_operation(  # type: ignore[arg-type]
            None,
            {},
        )


def test_plan_r2_owner_binding_is_monotonic_and_rejects_raw_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_admission, capability, _context, authority = (
        _joined_authorities(tmp_path / "authority")
    )
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    authorization = authority["authorization"]
    assert isinstance(authorization, dict)
    operational = tmp_path / "owner-private.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did=str(authority["principal_did"]),
        owner_status="ready",
    )
    seed = duckdb.connect(str(operational))
    try:
        _seed_bootstrap_board_lease(seed, capability)
    finally:
        seed.close()
    connection = duckdb.connect(str(operational))
    gateway = TypedStateOwnerGateway(
        connection=connection,
        socket_path=tmp_path / "typed-owner.sock",
        store_id=str(authorization["store_id"]),
        identity={
            "server_id": "server:plan-r2",
            "store_id": authorization["store_id"],
            "generation": authorization["owner_generation"],
            "fence_epoch": authorization["fencing_token"],
        },
    )
    arguments = {
        "admission": authority["admission"],
        "plan_r2_operational_capability": authority["plan_capability"],
        "authorization": authorization,
        "trusted_capability_reviewer_dids": [
            authority["plan_reviewer_did"]
        ],
        "trusted_operator_dids": [authority["operator_did"]],
        "trusted_security_reviewer_dids": [authority["security_did"]],
    }
    try:
        _bootstrap_service, service = _bind_joined_services(
            gateway,
            bootstrap_admission=bootstrap_admission,
            plan_authority=authority,
        )
        with pytest.raises(
            owner_service.EAAEFPlanR2OwnerServiceError,
            match="already bound",
        ):
            gateway._bind_eaaef_plan_r2_owner_service_from_server(  # noqa: SLF001
                **arguments
            )
        with pytest.raises(TypeError):
            owner_service.bind_eaaef_plan_r2_borrowed_owner_service(
                owner_gateway=gateway,
                **arguments,
            )
    finally:
        gateway.stop()
        connection.close()


def test_single_owner_external_review_request_remains_explicit_no_go() -> None:
    def sha(digit: str) -> str:
        return "sha256:" + digit * 64

    owner_key = Ed25519PrivateKey.generate()
    reviewer_key = Ed25519PrivateKey.generate()
    wrong_key = Ed25519PrivateKey.generate()
    owner_did = ed25519_did_key(owner_key.public_key())
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    bindings = {
        "source_head": "1" * 40,
        "source_tree": "2" * 40,
        "source_forest_root": sha("3"),
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "management_generation_id": "eaaef-review-request-001",
        "management_binding_cid": sha("4"),
        "management_snapshot_bindings_cid": sha("5"),
        "management_capsule_cid": sha("6"),
        "broker_process_birth_cid": sha("7"),
        "owner_start_receipt_cid": sha("8"),
        "owner_commit_receipt_cid": sha("9"),
        "r1_merge_admission_cid": sha("a"),
        "r1_operational_capability_cid": sha("b"),
        "plan_r2_authorization_cid": sha("c"),
        "plan_r2_operational_capability_cid": sha("d"),
        "plan_r2_remote_capability_cid": sha("e"),
        "plan_r2_authority_bundle_cid": sha("f"),
        "plan_r2_trust_bundle_cid": sha("0"),
        "owner_principal_did": owner_did,
        "shard_id": "eaaef-shard-1",
        "store_id": "eaaef-store-1",
        "owner_generation": 4,
        "plan_r2_epoch": 4,
        "fence_epoch": 4,
        "active_plan_root_cid": sha("1"),
        "active_plan_revision": 1,
        "active_plan_revision_cid": sha("2"),
        "r1_service_interface": "EAAEFTypedOwnerCommandService@1",
        "r1_service_schema": "eaaef-r1-service@1",
        "plan_r2_service_interface": "EAAEFPlanR2BorrowedOwnerService@1",
        "plan_r2_service_schema": "eaaef-plan-r2-service@1",
        "plan_r2_gateway_interface": "PlanR2OwnerGateway@1",
        "request_channel_id": "plan-r2-request-channel-1",
        "response_channel_id": "plan-r2-response-channel-1",
        "maximum_wait_ms": 60_000,
    }
    statement = single_owner_review.prepare_eaaef_single_owner_review_request(
        bindings,
        reviewer_did=reviewer_did,
        issued_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 1_000,
        issuance_nonce="single-owner-review-request-001",
    )
    assert statement["allowed"] is False
    assert statement["production_admitted"] is False
    assert statement["maximum_request_bytes"] == 786_432
    assert statement["maximum_response_bytes"] == 262_144
    assert list(statement["blockers"]) == list(
        single_owner_review.EAAEF_SINGLE_OWNER_REVIEW_BLOCKERS
    )
    assert "authenticated_lifecycle_cid_preimages_absent" in statement["blockers"]
    assert "durable_atomic_cutover_replay_journal_absent" in statement["blockers"]
    assert isinstance(statement["blockers"], tuple)
    assert isinstance(statement["proposed_operations"], tuple)

    statement_bytes = (
        single_owner_review.canonical_eaaef_single_owner_review_request_bytes(
            statement
        )
    )
    reviewer_signature = base64.b64encode(
        reviewer_key.sign(statement_bytes)
    ).decode("ascii")
    sealed = single_owner_review.seal_eaaef_single_owner_review_request(
        statement,
        reviewer_signature=reviewer_signature,
    )
    assert sealed["request_cid"].startswith("sha256:")
    assert sealed["production_admitted"] is False
    assert isinstance(sealed["blockers"], tuple)
    assert isinstance(sealed["proposed_operations"], tuple)
    assert single_owner_review.canonical_eaaef_single_owner_review_request_bytes(
        sealed
    )
    assert not hasattr(single_owner_review, "VerifiedEAAEFSingleOwnerCutover")
    with pytest.raises(AttributeError):
        sealed["blockers"].append("post-cid-mutation")

    wrong_signature = base64.b64encode(
        wrong_key.sign(statement_bytes)
    ).decode("ascii")
    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="signature authenticity differs",
    ):
        single_owner_review.seal_eaaef_single_owner_review_request(
            statement,
            reviewer_signature=wrong_signature,
        )

    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="reviewer is not independent",
    ):
        single_owner_review.prepare_eaaef_single_owner_review_request(
            {**bindings, "owner_principal_did": reviewer_did},
            reviewer_did=reviewer_did,
            issued_at_ms=NOW_MS,
            expires_at_ms=NOW_MS + 1_000,
            issuance_nonce="single-owner-review-request-colliding-reviewer",
        )

    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="signature syntax differs",
    ):
        single_owner_review.seal_eaaef_single_owner_review_request(
            statement,
            reviewer_signature="",
        )
    injected = {**bindings, "database_path": "/not/authority"}
    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="binding shape is not exact",
    ):
        single_owner_review.prepare_eaaef_single_owner_review_request(
            injected,
            reviewer_did=reviewer_did,
            issued_at_ms=NOW_MS,
            expires_at_ms=NOW_MS + 1_000,
            issuance_nonce="single-owner-review-request-002",
        )
    non_string = {**bindings, "board_namespace": 123}
    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="bounded identity differs",
    ):
        single_owner_review.prepare_eaaef_single_owner_review_request(
            non_string,
            reviewer_did=reviewer_did,
            issued_at_ms=NOW_MS,
            expires_at_ms=NOW_MS + 1_000,
            issuance_nonce="single-owner-review-request-non-string",
        )
    oversized = {**dict(statement), "maximum_request_bytes": 786_433}
    with pytest.raises(
        single_owner_review.EAAEFSingleOwnerReviewRequestError,
        match="constants differ",
    ):
        single_owner_review.seal_eaaef_single_owner_review_request(
            oversized,
            reviewer_signature=reviewer_signature,
        )


@pytest.mark.parametrize(
    ("plan_overrides", "mismatch"),
    [
        ({"source_tree": "5" * 40}, "source_tree"),
        (
            {"bootstrap_admission_cid": "sha256:" + "5" * 64},
            "bootstrap_admission_cid",
        ),
        (
            {
                "quack_command_fabric_qualification_cid": (
                    "sha256:" + "6" * 64
                )
            },
            "quack_command_fabric_qualification_cid",
        ),
        (
            {"expected_active_plan_root_cid": "sha256:" + "7" * 64},
            "expected_active_plan_root_cid",
        ),
        (
            {"expected_active_plan_cid": "sha256:" + "9" * 64},
            "expected_active_plan_cid",
        ),
    ],
)
def test_plan_r2_rejects_valid_signed_authority_divergent_from_bound_r1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_overrides: Mapping[str, object],
    mismatch: str,
) -> None:
    bootstrap_admission, capability, _context, authority = (
        _joined_authorities(
            tmp_path / "authority",
            plan_overrides=plan_overrides,
        )
    )
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    authorization = authority["authorization"]
    assert isinstance(authorization, dict)
    operational = tmp_path / "owner-private.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did=str(authority["principal_did"]),
        owner_status="ready",
    )
    seed = duckdb.connect(str(operational))
    try:
        _seed_bootstrap_board_lease(seed, capability)
    finally:
        seed.close()
    connection = duckdb.connect(str(operational))
    gateway = TypedStateOwnerGateway(
        connection=connection,
        socket_path=tmp_path / "typed-owner.sock",
        store_id=str(authorization["store_id"]),
        identity={
            "server_id": "server:plan-r2",
            "store_id": authorization["store_id"],
            "database_uuid": "123e4567-e89b-12d3-a456-426614174000",
            "generation": authorization["owner_generation"],
            "fence_epoch": authorization["fencing_token"],
        },
    )
    try:
        gateway.start()
        gateway._bind_eaaef_typed_owner_command_service_from_server(  # noqa: SLF001
            admission=bootstrap_admission,
        )
        with pytest.raises(
            owner_service.EAAEFPlanR2OwnerServiceError,
            match=(
                r"Plan-R2 authority differs from bound R1: .*"
                + mismatch
            ),
        ):
            gateway._bind_eaaef_plan_r2_owner_service_from_server(  # noqa: SLF001
                admission=authority["admission"],
                plan_r2_operational_capability=authority[
                    "plan_capability"
                ],
                authorization=authorization,
                trusted_capability_reviewer_dids=[
                    authority["plan_reviewer_did"]
                ],
                trusted_operator_dids=[authority["operator_did"]],
                trusted_security_reviewer_dids=[
                    authority["security_did"]
                ],
            )
    finally:
        gateway.stop()
        connection.close()


def test_all_31_bootstrap_names_contend_with_ordered_plan_r2_on_one_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_admission, capability, context, plan_authority = (
        _joined_authorities(tmp_path / "bootstrap-authority")
    )
    authorization = plan_authority["authorization"]
    assert isinstance(authorization, dict)
    operational = tmp_path / "combined-owner.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did=str(plan_authority["principal_did"]),
        owner_status="ready",
    )
    seed_connection = duckdb.connect(str(operational))
    try:
        _seed_bootstrap_board_lease(seed_connection, capability)
    finally:
        seed_connection.close()

    native_connect = duckdb.connect
    open_count = 0

    def counted_connect(*args: object, **kwargs: object) -> object:
        nonlocal open_count
        open_count += 1
        return native_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", counted_connect)
    raw = duckdb.connect(str(operational))
    guarded = _GuardedOwnerConnection(raw)
    gateway = TypedStateOwnerGateway(
        connection=guarded,
        socket_path=tmp_path / "typed-owner.sock",
        store_id=str(capability["store_id"]),
        identity={
            "server_id": "server:plan-r2",
            "store_id": capability["store_id"],
            "database_uuid": "123e4567-e89b-12d3-a456-426614174000",
            "generation": capability["owner_generation"],
            "fence_epoch": capability["fence_epoch"],
        },
    )
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    bootstrap_service, plan_service = _bind_joined_services(
        gateway,
        bootstrap_admission=bootstrap_admission,
        plan_authority=plan_authority,
    )
    assert bootstrap_service._connection is plan_service._connection  # noqa: SLF001
    assert (  # noqa: SLF001
        bootstrap_service._transaction_lock
        is plan_service._transaction_lock
        is gateway._transaction_lock
    )
    assert open_count == 1

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("combined EAAEF owner attempted a second database topology")

    monkeypatch.setattr(duckdb, "connect", forbidden)
    monkeypatch.setattr(QuackCommandFabric, "__init__", forbidden)
    monkeypatch.setattr(QuackCommandFabric, "start", forbidden)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_duckdb_connection",
        forbidden,
    )
    bootstrap_envelopes = []
    for serial, operation in enumerate(
        sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS),
        start=100,
    ):
        intent = dict(
            quack_daemon_operation_intent(
                gateway_binding_cid=str(capability["gateway_binding_cid"]),
                operational_capability_cid=str(capability["capability_cid"]),
                operation=operation,
                arguments={},
            )
        )
        bootstrap_envelopes.append(
            _envelope(intent, capability, context, serial=serial)
        )
    assert len(bootstrap_envelopes) == 31

    adapter = _adapter(
        authority=plan_authority,
        gateway=_TestOnlyPlanR2Gateway(  # type: ignore[arg-type]
            plan_service
        ),
        slot_allocator=iter(range(1, 20)).__next__,
        now_ms=NOW_MS,
    )
    adapter.attach()

    def contend(phase: object) -> object:
        with ThreadPoolExecutor(max_workers=32) as executor:
            reads = [
                executor.submit(
                    bootstrap_service.lookup_authorized_operation_receipt,
                    envelope,
                )
                for envelope in bootstrap_envelopes
            ]
            plan = executor.submit(phase)
            assert [future.result(timeout=15) for future in reads] == [
                None
            ] * 31
            return plan.result(timeout=15)

    try:
        prepared = contend(
            lambda: adapter.prepare_authorized_plan_r2_transition(
                authorization
            )
        )
        assert isinstance(prepared, dict)
        receipt = contend(
            lambda: adapter.apply_authorized_plan_r2_transition(
                authorization,
                prepared,
            )
        )
        assert isinstance(receipt, dict)
        observation = contend(
            lambda: adapter.observe_authorized_plan_r2_transition(
                authorization,
                receipt,
            )
        )
        assert isinstance(observation, dict)
        assert guarded.maximum_active == 1
        assert open_count == 1
        assert bootstrap_service.evidence()["operation_count"] == 31
        assert plan_service.evidence()["operation_count"] == 3
        assert sorted(path.name for path in tmp_path.glob("*.duckdb")) == [
            "combined-owner.duckdb"
        ]
    finally:
        adapter.close()
        gateway.stop()
        raw.close()
