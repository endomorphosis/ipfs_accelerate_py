"""Exact 1.5.5 local compatibility tests for the bounded Quack fabric."""

from __future__ import annotations

import base64
import json
import os
import platform
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
    verify_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandCapabilityDecision,
    QuackCommandFabric,
    QuackCommandIngressError,
    assess_quack_command_capability,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE,
    QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA,
    QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE,
    REQUIRED_QUACK_DAEMON_OPERATIONS,
    QuackDaemonGatewayCapability,
    QuackDaemonGatewayError,
    QuackDaemonRemoteCommandTransport,
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operational_capability_signing_payload,
    quack_daemon_state_command_parameters,
    seal_quack_daemon_operational_capability,
    verify_quack_daemon_operational_capability,
)

_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _ROOT / "ipfs_datasets_py" / "requirements" / "duckdb-quack.lock"
_UUID = "123e4567-e89b-12d3-a456-426614174000"
_AUTHORITY_CID = "sha256:" + "a" * 64
_INGRESS_TOKEN = "ingress-transport-token-0000000001"
_STATE_TOKEN = "state-transport-token-000000000003"


def _machine_lock_name() -> str:
    machine = platform.machine().lower()
    return {
        "aarch64": "linux_arm64",
        "arm64": "linux_arm64",
        "x86_64": "linux_amd64",
        "amd64": "linux_amd64",
    }.get(machine, f"linux_{machine}")


def _free_endpoint() -> str:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return f"quack:127.0.0.1:{port}"


def _reachable(endpoint: str) -> bool:
    port = int(endpoint.rsplit(":", 1)[1])
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
    except OSError:
        return False


def _exact_runtime():
    import duckdb

    artifact = Path(os.environ.get("QUACK_155_EXTENSION_PATH", ""))
    if duckdb.__version__ != "1.5.5" or not artifact.is_file():
        pytest.skip("requires DuckDB 1.5.5 and QUACK_155_EXTENSION_PATH exact artifact")
    return duckdb, artifact.resolve()


def _provision_operational(duckdb, path: Path) -> None:
    install_control_plane_schema(
        path,
        application_version="0.0.45",
        tool_version="1.5.5",
        owner_id="quack-command-fabric-test",
    )
    connection = duckdb.connect(str(path))
    try:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (1, 1, 1, 0, ?, 'birth:fabric-test',
                      '1970-01-01T00:00:00Z')
            """,
            [_UUID],
        )
        connection.execute("""
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES ('goal:fabric', 'G-FABRIC', 'objective:fabric', '', 1,
                      'Fabric', 'open', '1970-01-01T00:00:00Z',
                      '1970-01-01T00:00:00Z', 0, '{}')
            """)
        connection.execute("""
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES ('task:fabric:1', 'T-FABRIC-1', 'goal:fabric', '',
                      'objective:fabric', 1, 'ready', 0, 'P0',
                      '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z',
                      '{}', '{}')
            """)
    finally:
        connection.close()


def _provision_live_lease(
    duckdb,
    path: Path,
    *,
    principal_did: str,
    claim_cid: str = "lease:fabric",
    expires_at_ms: int | None = None,
    state: str = "accepted",
) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt,
                state, started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision, extension_schema,
                extension_json
            ) VALUES ('task:fabric:1', ?, 'resolution:fabric', ?, 1, 7,
                      ?, 1, ?, ?, NULL, 0, 'session:fabric', 1, 0,
                      'AuthorizedStateCommandLease@1', '{}')
            """,
            [
                claim_cid,
                principal_did,
                int(expires_at_ms or (time.time_ns() // 1_000_000 + 120_000)),
                state,
                time.time_ns() // 1_000_000,
            ],
        )
    finally:
        connection.close()


def _command(fabric: QuackCommandFabric, *, command_id: str, idem: str) -> StateCommand:
    generation = fabric.repository.load_generation()
    return StateCommand(
        command_id=command_id,
        command_kind=CommandKind.CLAIM,
        store_id=generation.store_id,
        session_id="session:remote-client",
        expected_generation=generation.generation,
        expected_revision=generation.revision,
        fence_epoch=generation.fence_epoch,
        idempotency_key=idem,
        parameters={
            "task_cid": "task:fabric:1",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )


def _authorized(
    command: StateCommand,
    *,
    policy: QuackCommandAuthorizationPolicy,
    approver_key: Ed25519PrivateKey,
    slot: int,
    submission_id: str,
    request_id: str,
    nonce: str,
):
    now_ms = time.time_ns() // 1_000_000
    prepared = authorized_state_command_signing_payload(
        request_id=request_id,
        submission_id=submission_id,
        ingress_slot=slot,
        principal_did=next(iter(policy.authorized_principal_dids)),
        approver_did=next(iter(policy.trusted_approver_dids)),
        authority_ref_cid=policy.authority_ref_cid,
        board_namespace=policy.board_namespace,
        shard_id=policy.shard_id,
        owner_principal_did=policy.owner_principal_did,
        lease_id="lease:fabric",
        scope_id="task:fabric:1",
        effect=f"control-plane/{command.command_kind.value}",
        issued_at_ms=now_ms - 100,
        expires_at_ms=now_ms + 60_000,
        deadline_ms=now_ms + 30_000,
        one_use_nonce=nonce,
        command=command,
    )
    signature = base64.b64encode(
        approver_key.sign(
            json.dumps(
                dict(prepared),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
    ).decode("ascii")
    return seal_authorized_state_command(prepared, approver_signature=signature)


def test_exact_capability_mismatch_is_typed_no_go(tmp_path: Path) -> None:
    artifact = tmp_path / "quack.duckdb_extension"
    artifact.write_bytes(b"not-the-locked-extension")
    result = assess_quack_command_capability(
        duckdb_module=SimpleNamespace(__version__="1.5.5"),
        extension_path=artifact,
        lock_path=_LOCK,
        machine=_machine_lock_name(),
    )
    assert result.decision is QuackCommandCapabilityDecision.NO_GO
    assert result.admitted is False
    assert "repository lock" in result.reason
    assert result.to_dict()["operational_database_served"] is False
    assert result.to_dict()["ingress_relation_count"] == 1


def test_remote_daemon_task_read_crosses_exact_quack_owner_transaction(
    tmp_path: Path,
) -> None:
    """Real Quack transport returns a receipt-bound canonical task read."""

    duckdb, artifact = _exact_runtime()
    operational = tmp_path / "private-operational.duckdb"
    ingress = tmp_path / "command-ingress.duckdb"
    projection = tmp_path / "read-projection.duckdb"
    _provision_operational(duckdb, operational)
    ingress_endpoint = _free_endpoint()
    state_endpoint = _free_endpoint()
    reviewer_key = Ed25519PrivateKey.generate()
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    owner_key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    policy = QuackCommandAuthorizationPolicy(
        board_namespace="EAAEF-v1",
        shard_id="shard:eaaef",
        store_id="control.duckdb",
        authority_ref_cid=_AUTHORITY_CID,
        owner_principal_did=ed25519_did_key(owner_key.public_key()),
        owner_generation=1,
        fence_epoch=1,
        trusted_approver_dids=frozenset(
            {ed25519_did_key(approver_key.public_key())}
        ),
        authorized_principal_dids=frozenset(
            {ed25519_did_key(principal_key.public_key())}
        ),
        allowed_command_kinds=frozenset(
            {CommandKind.OBSERVE, CommandKind.APPEND, CommandKind.MIGRATE}
        ),
    )
    _provision_live_lease(
        duckdb,
        operational,
        principal_did=next(iter(policy.authorized_principal_dids)),
    )
    now_ms = time.time_ns() // 1_000_000
    capability_body = {
        "schema": QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA,
        "interface": QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE,
        "board_namespace": policy.board_namespace,
        "shard_id": policy.shard_id,
        "store_id": policy.store_id,
        "control_plane_schema_version": "QuackStateRepository@1",
        "state_schema_revision": "datasets-authoritative-operational-control-plane@1",
        "command_endpoint": ingress_endpoint,
        "state_endpoint": state_endpoint,
        "owner_principal_did": policy.owner_principal_did,
        "owner_generation": policy.owner_generation,
        "fence_epoch": policy.fence_epoch,
        "authorization_policy_cid": policy.policy_cid,
        "command_fabric_qualification_cid": "sha256:" + "b" * 64,
        "authorized_state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "authorized_state_command_interface": "AuthorizedStateCommand@1",
        "dispatcher_interface": QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE,
        "operations": sorted(REQUIRED_QUACK_DAEMON_OPERATIONS),
        "guarantees": {
            name: True
            for name in (
                "one_mutable_owner",
                "operational_database_private",
                "authorized_state_command_required",
                "owner_verifies_command_signature",
                "live_lease_verified_in_transaction",
                "fencing_token_verified_in_transaction",
                "replay_claims_consumed_in_transaction",
                "cas_and_effect_applied_in_transaction",
                "durable_idempotent_receipt",
                "no_portal_fallback",
                "no_local_sidecar",
                "no_direct_database_open",
                "no_arbitrary_sql",
            )
        },
        "allowed": True,
        "blockers": [],
        "issued_at_ms": now_ms - 100,
        "expires_at_ms": now_ms + 120_000,
        "reviewer_identity_did": reviewer_did,
    }
    prepared_capability = quack_daemon_operational_capability_signing_payload(
        capability_body
    )
    capability_signature = base64.b64encode(
        reviewer_key.sign(
            json.dumps(
                dict(prepared_capability),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
    ).decode("ascii")
    operational_capability = seal_quack_daemon_operational_capability(
        prepared_capability,
        reviewer_signature=capability_signature,
    )
    fabric = QuackCommandFabric(
        duckdb_module=duckdb,
        extension_path=artifact,
        lock_path=_LOCK,
        machine=_machine_lock_name(),
        ingress_database=ingress,
        operational_database=operational,
        projection_database=projection,
        ingress_endpoint=ingress_endpoint,
        state_endpoint=state_endpoint,
        ingress_token=_INGRESS_TOKEN,
        state_token=_STATE_TOKEN,
        authorization_policy=policy,
        daemon_operational_capability=operational_capability,
        trusted_daemon_capability_reviewer_dids=(reviewer_did,),
        command_fabric_qualification_cid="sha256:" + "b" * 64,
    )
    fabric.start()
    verified = verify_quack_daemon_operational_capability(
        operational_capability,
        trusted_reviewer_dids=(reviewer_did,),
        now_ms=time.time_ns() // 1_000_000,
    )
    gateway_capability = (
        QuackDaemonGatewayCapability.from_verified_operational_capability(verified)
    )

    authorization_count = 0

    def authorize(intent):
        nonlocal authorization_count
        authorization_count += 1
        observed = time.time_ns() // 1_000_000
        generation = fabric.repository.load_generation()
        operation = str(intent["operation"])
        suffix = operation.replace(".", "-")
        command_kind = CommandKind(
            quack_daemon_operation_command_vocabulary()[operation]
        )
        request_id = f"request:remote-daemon:{authorization_count}"
        idempotency_key = f"idempotency:remote-daemon:{authorization_count}"
        deadline_ms = observed + 30_000
        command = StateCommand(
            command_id=f"{request_id}:{suffix}",
            command_kind=command_kind,
            store_id=policy.store_id,
            session_id="lease:fabric",
            expected_generation=generation.generation,
            expected_revision=generation.revision,
            fence_epoch=generation.fence_epoch,
            idempotency_key=idempotency_key,
            parameters=quack_daemon_state_command_parameters(
                intent,
                request_id=request_id,
                principal_did=next(iter(policy.authorized_principal_dids)),
                authority_ref_cid=policy.authority_ref_cid,
                lease_id="lease:fabric",
                scope_id="task:fabric:1",
                deadline_ms=deadline_ms,
                fencing_token=7,
                idempotency_key=idempotency_key,
            ),
        )
        prepared = authorized_state_command_signing_payload(
            request_id=request_id,
            submission_id=f"submission:remote-daemon:{authorization_count}",
            ingress_slot=authorization_count,
            principal_did=next(iter(policy.authorized_principal_dids)),
            approver_did=next(iter(policy.trusted_approver_dids)),
            authority_ref_cid=policy.authority_ref_cid,
            board_namespace=policy.board_namespace,
            shard_id=policy.shard_id,
            owner_principal_did=policy.owner_principal_did,
            lease_id="lease:fabric",
            scope_id="task:fabric:1",
            effect=f"control-plane/{command_kind.value}",
            issued_at_ms=observed - 10,
            expires_at_ms=observed + 60_000,
            deadline_ms=deadline_ms,
            one_use_nonce=f"nonce:remote-daemon:{authorization_count}",
            command=command,
        )
        signature = base64.b64encode(
            approver_key.sign(
                json.dumps(
                    dict(prepared),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            )
        ).decode("ascii")
        return seal_authorized_state_command(
            prepared,
            approver_signature=signature,
        )

    transport = QuackDaemonRemoteCommandTransport(
        capability=gateway_capability,
        operational_capability=operational_capability,
        authorization_policy=policy,
        authorization_provider=authorize,
        command_client=fabric.command_client(alias="remote_daemon"),
        read_client=fabric.read_client(),
        clock_ms=lambda: time.time_ns() // 1_000_000,
        maximum_wait_ms=5_000,
        poll_interval_ms=5,
    )
    owner_errors: list[BaseException] = []

    def owner_tick() -> None:
        try:
            time.sleep(0.05)
            fabric.apply_pending()
        except BaseException as exc:  # pragma: no cover - asserted below
            owner_errors.append(exc)

    worker = threading.Thread(target=owner_tick, daemon=True)
    worker.start()
    try:
        result = transport.dispatch("task.get", {"task_cid": "task:fabric:1"})
        assert result is not None
        assert result["task_cid"] == "task:fabric:1"
        assert result["task_alias"] == "T-FABRIC-1"
        assert result["status"] == "ready"
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert owner_errors == []
        second_worker = threading.Thread(target=owner_tick, daemon=True)
        second_worker.start()
        with pytest.raises(
            QuackDaemonGatewayError,
            match=(
                "daemon_operation_no_go:operation=task.materialize;"
                "reason_code=plan_r2_population_transition_required"
            ),
        ):
            transport.dispatch("task.materialize", {"population": {}})
        second_worker.join(timeout=5)
        assert not second_worker.is_alive()
        assert owner_errors == []
        assert transport.evidence()["production_admitted"] is False
        assert transport.evidence()["append_only_authorized_state_command"] is True
        task = fabric.repository.get_task("task:fabric:1")
        assert task is not None and task["status"] == "ready"
        with fabric.read_client() as reader:
            receipts = reader.list_receipts()
            assert len(receipts) == 2
            assert receipts[0]["outcome"] == CommandOutcome.ACCEPTED.value
            assert receipts[0]["error"] == ""
            receipt_result = json.loads(receipts[0]["result_json"])
            assert receipt_result["daemon_operation"] == "task.get"
            assert receipt_result["value"]["task_cid"] == "task:fabric:1"
            assert receipts[1]["outcome"] == CommandOutcome.REJECTED.value
            assert "plan_r2_population_transition_required" in receipts[1]["error"]
        assert fabric.repository.load_generation().revision == 1
        assert len(fabric.repository.list_commands()) == 1

        # Even independently signed, live-leased OBSERVE/MIGRATE commands may
        # not bypass the closed daemon operation registry by supplying generic
        # task-status parameters.  Both replay claims and mutations roll back;
        # only a rejected audit receipt is persisted.
        generation = fabric.repository.load_generation()
        bare_commands = (
            StateCommand(
                command_id="cmd:bare-observe",
                command_kind=CommandKind.OBSERVE,
                store_id=generation.store_id,
                session_id="session:remote-client",
                expected_generation=generation.generation,
                expected_revision=generation.revision,
                fence_epoch=generation.fence_epoch,
                idempotency_key="idem:bare-observe",
                parameters={
                    "task_cid": "task:fabric:1",
                    "expected_task_revision": 0,
                    "status": "claimed",
                },
            ),
            StateCommand(
                command_id="cmd:bare-migrate",
                command_kind=CommandKind.MIGRATE,
                store_id=generation.store_id,
                session_id="session:remote-client",
                expected_generation=generation.generation,
                expected_revision=generation.revision,
                fence_epoch=generation.fence_epoch,
                idempotency_key="idem:bare-migrate",
                parameters={
                    "task_cid": "task:fabric:1",
                    "expected_task_revision": 0,
                    "status": "claimed",
                },
            ),
        )
        bare_envelopes = tuple(
            _authorized(
                command,
                policy=policy,
                approver_key=approver_key,
                slot=90 + offset,
                submission_id=f"submission:bare:{command.command_kind.value}",
                request_id=f"request:bare:{command.command_kind.value}",
                nonce=f"nonce:bare:{command.command_kind.value}",
            )
            for offset, command in enumerate(bare_commands)
        )
        with fabric.command_client(alias="bare_generic_attack") as attack_client:
            for envelope in bare_envelopes:
                attack_client.append(envelope)
        rejected = fabric.apply_pending()
        assert [row["outcome"] for row in rejected] == [
            CommandOutcome.REJECTED.value,
            CommandOutcome.REJECTED.value,
        ]
        assert all(
            "specialized owner command fabric rejects bare generic StateCommand fallback"
            in row["error"]
            for row in rejected
        )
        task = fabric.repository.get_task("task:fabric:1")
        assert task is not None and task["status"] == "ready" and task["revision"] == 0
        assert fabric.repository.load_generation().revision == 1
        assert len(fabric.repository.list_commands()) == 1

        # A frozen/injected authorization clock cannot keep remote polling
        # alive.  The wait is bounded by an independent real monotonic clock,
        # and each poll uses only the fixed newest-receipt window.
        frozen_now = time.time_ns() // 1_000_000 + 5_000
        transport._clock_ms = lambda: frozen_now  # type: ignore[method-assign]  # noqa: SLF001
        transport._maximum_wait_ms = 25  # noqa: SLF001
        transport._poll_interval_ms = 5  # noqa: SLF001
        monotonic_started = time.monotonic()
        with pytest.raises(
            QuackDaemonGatewayError,
            match="remote daemon operation receipt deadline expired",
        ):
            transport.dispatch("task.get", {"task_cid": "task:fabric:1"})
        assert time.monotonic() - monotonic_started < 1.0
    finally:
        transport.close()
        fabric.stop()


def test_two_clients_duplicate_receipt_restart_and_clean_stop(tmp_path: Path) -> None:
    duckdb, artifact = _exact_runtime()
    operational = tmp_path / "private-operational.duckdb"
    ingress = tmp_path / "command-ingress.duckdb"
    projection = tmp_path / "read-projection.duckdb"
    _provision_operational(duckdb, operational)
    ingress_endpoint = _free_endpoint()
    state_endpoint = _free_endpoint()
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    owner_key = Ed25519PrivateKey.generate()
    policy = QuackCommandAuthorizationPolicy(
        board_namespace="EAAEF-v1",
        shard_id="shard:eaaef",
        store_id="control.duckdb",
        authority_ref_cid=_AUTHORITY_CID,
        owner_principal_did=ed25519_did_key(owner_key.public_key()),
        owner_generation=1,
        fence_epoch=1,
        trusted_approver_dids=frozenset({ed25519_did_key(approver_key.public_key())}),
        authorized_principal_dids=frozenset({ed25519_did_key(principal_key.public_key())}),
        allowed_command_kinds=frozenset({CommandKind.CLAIM}),
    )
    _provision_live_lease(
        duckdb,
        operational,
        principal_did=next(iter(policy.authorized_principal_dids)),
    )

    def build() -> QuackCommandFabric:
        return QuackCommandFabric(
            duckdb_module=duckdb,
            extension_path=artifact,
            lock_path=_LOCK,
            machine=_machine_lock_name(),
            ingress_database=ingress,
            operational_database=operational,
            projection_database=projection,
            ingress_endpoint=ingress_endpoint,
            state_endpoint=state_endpoint,
            ingress_token=_INGRESS_TOKEN,
            state_token=_STATE_TOKEN,
            authorization_policy=policy,
        )

    fabric = build()
    fabric.start()
    assert fabric.capability.admitted is True
    assert _reachable(ingress_endpoint)
    assert _reachable(state_endpoint)
    command = _command(fabric, command_id="cmd:fabric:claim", idem="idem:fabric")
    envelope_a = _authorized(
        command,
        policy=policy,
        approver_key=approver_key,
        slot=1,
        submission_id="submission:a",
        request_id="request:a",
        nonce="nonce:a",
    )
    envelope_b = _authorized(
        command,
        policy=policy,
        approver_key=approver_key,
        slot=2,
        submission_id="submission:b",
        request_id="request:b",
        nonce="nonce:b",
    )
    forged_envelope = _authorized(
        command,
        policy=policy,
        approver_key=Ed25519PrivateKey.generate(),
        slot=3,
        submission_id="submission:forged",
        request_id="request:forged",
        nonce="nonce:forged",
    )
    client_a = fabric.command_client(alias="client_a")
    client_b = fabric.command_client(alias="client_b")
    try:
        client_a.append(envelope_a)
        client_b.append(envelope_b)
        client_b.append(forged_envelope)
        with pytest.raises(QuackCommandIngressError, match="append was rejected"):
            duplicate_slot = _authorized(
                command,
                policy=policy,
                approver_key=approver_key,
                slot=1,
                submission_id="submission:duplicate-slot",
                request_id="request:duplicate-slot",
                nonce="nonce:duplicate-slot",
            )
            client_b.append(duplicate_slot)
        with pytest.raises(QuackCommandAuthorizationError, match="positive integer"):
            _authorized(
                command,
                policy=policy,
                approver_key=approver_key,
                slot=0,
                submission_id="submission:invalid",
                request_id="request:invalid",
                nonce="nonce:invalid",
            )
        with pytest.raises(QuackCommandIngressError, match="AuthorizedStateCommand"):
            client_a.append(  # type: ignore[arg-type]
                command,
            )

        receipts = fabric.apply_pending()
        assert [row["outcome"] for row in receipts] == [
            CommandOutcome.ACCEPTED.value,
            CommandOutcome.IDEMPOTENT_REPLAY.value,
            CommandOutcome.REJECTED.value,
        ]
        assert [row["changed"] for row in receipts] == [True, False, False]
        assert "signature is invalid" in receipts[2]["error"]
        with fabric.read_client() as reader:
            assert [(row["task_cid"], row["status"]) for row in reader.list_state()] == [
                ("task:fabric:1", "claimed")
            ]
            remote_receipts = reader.list_receipts()
            assert len(remote_receipts) == 3
            assert all(not row["error"] for row in remote_receipts[:2])
            assert "signature is invalid" in remote_receipts[2]["error"]
            assert {row["authority_ref_cid"] for row in remote_receipts} == {_AUTHORITY_CID}
            assert {row["principal_did"] for row in remote_receipts} == set(
                policy.authorized_principal_dids
            )
    finally:
        client_b.close()
        client_a.close()
        fabric.stop()

    assert not _reachable(ingress_endpoint)
    assert not _reachable(state_endpoint)
    check_ingress = duckdb.connect(str(ingress), read_only=True)
    try:
        assert check_ingress.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT internal ORDER BY 1"
        ).fetchall() == [("command_inbox",)]
        assert check_ingress.execute("SELECT count(*) FROM command_inbox").fetchone() == (0,)
    finally:
        check_ingress.close()

    restarted = build()
    restarted.start()
    restart_client = restarted.command_client(alias="client_restart")
    try:
        assert restarted.apply_pending() == ()
        task = restarted.repository.get_task("task:fabric:1")
        assert task is not None and task["status"] == "claimed"
        with restarted.read_client() as reader:
            assert len(reader.list_receipts()) == 3
    finally:
        restart_client.close()
        restarted.stop()
    assert not _reachable(ingress_endpoint)
    assert not _reachable(state_endpoint)


def test_private_atomic_receipt_rebuild_nonce_reuse_and_revoked_lease(tmp_path: Path) -> None:
    duckdb, artifact = _exact_runtime()
    operational = tmp_path / "private-operational.duckdb"
    ingress = tmp_path / "command-ingress.duckdb"
    projection = tmp_path / "read-projection.duckdb"
    _provision_operational(duckdb, operational)
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    owner_key = Ed25519PrivateKey.generate()
    principal_did = ed25519_did_key(principal_key.public_key())
    policy = QuackCommandAuthorizationPolicy(
        board_namespace="EAAEF-v1",
        shard_id="shard:eaaef",
        store_id="control.duckdb",
        authority_ref_cid=_AUTHORITY_CID,
        owner_principal_did=ed25519_did_key(owner_key.public_key()),
        owner_generation=1,
        fence_epoch=1,
        trusted_approver_dids=frozenset({ed25519_did_key(approver_key.public_key())}),
        authorized_principal_dids=frozenset({principal_did}),
        allowed_command_kinds=frozenset({CommandKind.CLAIM}),
    )
    _provision_live_lease(duckdb, operational, principal_did=principal_did)

    def build() -> QuackCommandFabric:
        return QuackCommandFabric(
            duckdb_module=duckdb,
            extension_path=artifact,
            lock_path=_LOCK,
            machine=_machine_lock_name(),
            ingress_database=ingress,
            operational_database=operational,
            projection_database=projection,
            ingress_endpoint=_free_endpoint(),
            state_endpoint=_free_endpoint(),
            ingress_token=_INGRESS_TOKEN,
            state_token=_STATE_TOKEN,
            authorization_policy=policy,
        )

    fabric = build()
    fabric.start()
    command = _command(fabric, command_id="cmd:atomic:first", idem="idem:atomic:first")
    accepted = _authorized(
        command,
        policy=policy,
        approver_key=approver_key,
        slot=1,
        submission_id="submission:atomic:first",
        request_id="request:atomic:first",
        nonce="nonce:atomic:shared",
    )
    with fabric.command_client(alias="atomic_first") as client:
        client.append(accepted)

    # Simulate a process failure after the private transaction commits but
    # before its disposable projection is refreshed.
    def fail_projection() -> None:
        raise RuntimeError("injected projection failure")

    fabric._rebuild_projection = fail_projection  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(RuntimeError, match="injected projection failure"):
        fabric.apply_pending()

    # Projection loss cannot restore authority: restart must rebuild it from
    # the private domain-event receipt and replay-suppression claims.
    fabric.stop()
    restarted = build()
    restarted.start()
    with restarted.read_client() as reader:
        assert len(reader.list_receipts()) == 1
        assert reader.list_state()[0]["status"] == "claimed"

    # A different signed command cannot reuse the consumed nonce. The durable
    # task mutation remains the first accepted result.
    live = restarted.repository.load_generation()
    divergent = StateCommand(
        command_id="cmd:atomic:divergent",
        command_kind=CommandKind.CLAIM,
        store_id=live.store_id,
        session_id="session:remote-client",
        expected_generation=live.generation,
        expected_revision=live.revision,
        fence_epoch=live.fence_epoch,
        idempotency_key="idem:atomic:divergent",
        parameters={
            "task_cid": "task:fabric:1",
            "expected_task_revision": 1,
            "status": "ready",
        },
    )
    nonce_reuse = _authorized(
        divergent,
        policy=policy,
        approver_key=approver_key,
        slot=2,
        submission_id="submission:atomic:nonce-reuse",
        request_id="request:atomic:nonce-reuse",
        nonce="nonce:atomic:shared",
    )
    with restarted.command_client(alias="atomic_nonce_reuse") as client:
        client.append(nonce_reuse)
    rejected = restarted.apply_pending()[0]
    assert rejected["outcome"] == CommandOutcome.REJECTED.value
    assert "already consumed" in rejected["error"]
    assert restarted.repository.get_task("task:fabric:1")["status"] == "claimed"

    request_reuse = _authorized(
        divergent,
        policy=policy,
        approver_key=approver_key,
        slot=3,
        submission_id="submission:atomic:request-reuse",
        request_id="request:atomic:first",
        nonce="nonce:atomic:request-reuse",
    )
    with restarted.command_client(alias="atomic_request_reuse") as client:
        client.append(request_reuse)
    request_rejected = restarted.apply_pending()[0]
    assert request_rejected["outcome"] == CommandOutcome.REJECTED.value
    assert "already consumed" in request_rejected["error"]
    assert restarted.repository.get_task("task:fabric:1")["status"] == "claimed"

    # Revocation is checked from the live private lease in the same transaction.
    connection = duckdb.connect(str(operational))
    try:
        connection.execute(
            "UPDATE leases SET state = 'released', revision = revision + 1 "
            "WHERE task_cid = 'task:fabric:1'"
        )
    finally:
        connection.close()
    revoked = _authorized(
        divergent,
        policy=policy,
        approver_key=approver_key,
        slot=4,
        submission_id="submission:atomic:revoked",
        request_id="request:atomic:revoked",
        nonce="nonce:atomic:revoked",
    )
    with restarted.command_client(alias="atomic_revoked") as client:
        client.append(revoked)
    revoked_receipt = restarted.apply_pending()[0]
    assert revoked_receipt["outcome"] == CommandOutcome.REJECTED.value
    assert "lease is revoked" in revoked_receipt["error"]

    connection = duckdb.connect(str(operational))
    try:
        connection.execute(
            "UPDATE leases SET state = 'accepted', expires_at_ms = 1, "
            "revision = revision + 1 WHERE task_cid = 'task:fabric:1'"
        )
    finally:
        connection.close()
    expired = _authorized(
        divergent,
        policy=policy,
        approver_key=approver_key,
        slot=5,
        submission_id="submission:atomic:expired",
        request_id="request:atomic:expired",
        nonce="nonce:atomic:expired",
    )
    with restarted.command_client(alias="atomic_expired") as client:
        client.append(expired)
    expired_receipt = restarted.apply_pending()[0]
    assert expired_receipt["outcome"] == CommandOutcome.REJECTED.value
    assert "lease is expired" in expired_receipt["error"]
    restarted.stop()


def test_policy_rejects_owner_or_worker_self_approval() -> None:
    key = Ed25519PrivateKey.generate()
    identity = ed25519_did_key(key.public_key())
    other = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    with pytest.raises(QuackCommandAuthorizationError, match="owner cannot approve"):
        QuackCommandAuthorizationPolicy(
            board_namespace="EAAEF-v1",
            shard_id="shard:eaaef",
            store_id="control.duckdb",
            authority_ref_cid=_AUTHORITY_CID,
            owner_principal_did=identity,
            owner_generation=1,
            fence_epoch=1,
            trusted_approver_dids=frozenset({identity}),
            authorized_principal_dids=frozenset({other}),
            allowed_command_kinds=frozenset({CommandKind.CLAIM}),
        )
    with pytest.raises(QuackCommandAuthorizationError, match="principal cannot approve"):
        QuackCommandAuthorizationPolicy(
            board_namespace="EAAEF-v1",
            shard_id="shard:eaaef",
            store_id="control.duckdb",
            authority_ref_cid=_AUTHORITY_CID,
            owner_principal_did=other,
            owner_generation=1,
            fence_epoch=1,
            trusted_approver_dids=frozenset({identity}),
            authorized_principal_dids=frozenset({identity}),
            allowed_command_kinds=frozenset({CommandKind.CLAIM}),
        )


def test_signed_command_rejects_token_only_forgery_and_expiry() -> None:
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    owner_key = Ed25519PrivateKey.generate()
    policy = QuackCommandAuthorizationPolicy(
        board_namespace="EAAEF-v1",
        shard_id="shard:eaaef",
        store_id="control.duckdb",
        authority_ref_cid=_AUTHORITY_CID,
        owner_principal_did=ed25519_did_key(owner_key.public_key()),
        owner_generation=1,
        fence_epoch=1,
        trusted_approver_dids=frozenset({ed25519_did_key(approver_key.public_key())}),
        authorized_principal_dids=frozenset({ed25519_did_key(principal_key.public_key())}),
        allowed_command_kinds=frozenset({CommandKind.CLAIM}),
    )
    command = StateCommand(
        command_id="cmd:authorization:test",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id="session:authorization:test",
        expected_generation=1,
        expected_revision=0,
        fence_epoch=1,
        idempotency_key="idem:authorization:test",
        parameters={
            "task_cid": "task:fabric:1",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )
    valid = _authorized(
        command,
        policy=policy,
        approver_key=approver_key,
        slot=1,
        submission_id="submission:authorization:valid",
        request_id="request:authorization:valid",
        nonce="nonce:authorization:valid",
    )
    verify_authorized_state_command(valid, policy=policy, now_ms=time.time_ns() // 1_000_000)

    forged = _authorized(
        command,
        policy=policy,
        approver_key=Ed25519PrivateKey.generate(),
        slot=2,
        submission_id="submission:authorization:forged",
        request_id="request:authorization:forged",
        nonce="nonce:authorization:forged",
    )
    with pytest.raises(QuackCommandAuthorizationError, match="signature is invalid"):
        verify_authorized_state_command(forged, policy=policy, now_ms=time.time_ns() // 1_000_000)
    with pytest.raises(QuackCommandAuthorizationError, match="expired"):
        verify_authorized_state_command(valid, policy=policy, now_ms=valid.expires_at_ms)
