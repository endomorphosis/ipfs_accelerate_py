"""Exact owner-transaction tests for the signed EAAEF 31-op fabric path."""

from __future__ import annotations

import base64
import json
import os
import platform
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    QuackStateRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
    EAAEF_DAEMON_LANE_BINDING_SCHEMA,
    EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
    EAAEFBootstrapBorrowedTransactionOperationHandler,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    eaaef_board_scheduler_lease_seed,
    install_eaaef_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    AuthorizedStateCommand,
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
    verify_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandAuthorizationError,
    QuackCommandFabric,
    QuackCommandFabricStateError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operation_intent,
    quack_daemon_state_command_parameters,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_bootstrap_gateway_launch as launch,
)
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _expected_bindings,
    _signed_capability,
)

_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _ROOT / "ipfs_datasets_py" / "requirements" / "duckdb-quack.lock"
_INGRESS_TOKEN = "eaaef-ingress-transport-token-0001"
_STATE_TOKEN = "eaaef-state-transport-token-0000002"


def _machine_lock_name() -> str:
    machine = platform.machine().lower()
    return {
        "aarch64": "linux_arm64",
        "arm64": "linux_arm64",
        "x86_64": "linux_amd64",
        "amd64": "linux_amd64",
    }.get(machine, f"linux_{machine}")


def _exact_runtime():
    import duckdb

    artifact = Path(os.environ.get("QUACK_155_EXTENSION_PATH", ""))
    if duckdb.__version__ != "1.5.5" or not artifact.is_file():
        pytest.skip(
            "requires DuckDB 1.5.5 and QUACK_155_EXTENSION_PATH exact artifact"
        )
    return duckdb, artifact.resolve()


def _verified_capability():
    capability, context = _signed_capability()
    verified = launch.verify_eaaef_bootstrap_operational_capability(
        capability,
        trusted_reviewer_dids=[context["operational_reviewer"]],
        trusted_authorization_service_reviewer_dids=[
            context["service_reviewer"]
        ],
        expected=_expected_bindings(capability),
        now_ms=NOW_MS,
    )
    return capability, context, verified


def _provision_operational(path: Path, capability: dict[str, object]) -> None:
    install_eaaef_operational_schema(
        path,
        application_version="eaaef-fabric-test",
        tool_version="1.5.5",
        owner_id="eaaef-fabric-test-materializer",
    )
    with DatabaseTaskSource(path, install_schema=False) as source:
        source.materialize(
            {
                "goals": [{"goal_cid": "goal:eaaef", "goal_id": "EAAEF"}],
                "tasks": [
                    {
                        "task_cid": "task:eaaef:1",
                        "task_id": "EAAEF-001",
                        "goal_cid": "goal:eaaef",
                        "status": "ready",
                        "priority": "P0",
                    }
                ],
            },
            repository_tree_id="tree:eaaef-fabric-test",
        )
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
    with open_duckdb_connection(path) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            "INSERT INTO store_generations(generation, schema_revision, fence_epoch, "
            "revision, database_uuid, birth_id, created_at) "
            "VALUES (?, 2, ?, 1, '12345678-1234-4234-8234-123456789abc', "
            "'birth:eaaef-fabric', '2026-08-18T00:00:00Z')",
            [int(capability["owner_generation"]), int(capability["fence_epoch"])],
        )
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
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_kind",
            "scope_id",
            "mode",
        )
        connection.execute(
            "INSERT INTO leases(" + ",".join(columns) + ") VALUES (" +
            ",".join("?" for _ in columns) + ")",
            [
                (
                    json.dumps(row[name], sort_keys=True, separators=(",", ":"))
                    if name == "extension_json"
                    else row[name]
                )
                for name in columns
            ],
        )


def _fabric(
    tmp_path: Path,
    *,
    capability: dict[str, object],
    context: dict[str, object],
    verified: object,
) -> QuackCommandFabric:
    duckdb, artifact = _exact_runtime()
    operational = tmp_path / "operational.duckdb"
    _provision_operational(operational, capability)
    fabric = QuackCommandFabric(
        duckdb_module=duckdb,
        extension_path=artifact,
        lock_path=_LOCK,
        machine=_machine_lock_name(),
        ingress_database=tmp_path / "ingress.duckdb",
        operational_database=operational,
        projection_database=tmp_path / "projection.duckdb",
        ingress_endpoint=str(capability["command_endpoint"]),
        state_endpoint=str(capability["state_endpoint"]),
        ingress_token=_INGRESS_TOKEN,
        state_token=_STATE_TOKEN,
        authorization_policy=context["policy"],
        eaaef_bootstrap_operational_capability=verified,
        command_fabric_qualification_cid=str(
            capability["command_fabric_qualification_cid"]
        ),
        clock_ms=lambda: NOW_MS,
    )
    repository = QuackStateRepository(
        "quack:127.0.0.1:1",
        owner_id=str(capability["owner_principal_did"]),
        store_id=str(capability["store_id"]),
        connection_factory=lambda _endpoint: duckdb.connect(str(operational)),
        seed_generation=False,
    )
    repository.attach()
    fabric._repository = repository  # noqa: SLF001 - no-network owner fixture
    return fabric


def _authorized_operation(
    fabric: QuackCommandFabric,
    capability: dict[str, object],
    context: dict[str, object],
    *,
    operation: str,
    arguments: dict[str, object],
    sequence: int,
    scope_id: str | None = None,
    fencing_token: int | None = None,
    request_id: str | None = None,
    submission_id: str | None = None,
    nonce: str | None = None,
    lease_id: str | None = None,
):
    request = request_id or f"request:eaaef-fabric:{sequence}"
    submission = submission_id or f"submission:eaaef-fabric:{sequence}"
    one_use_nonce = nonce or f"nonce:eaaef-fabric:{sequence}"
    scope = str(scope_id or capability["board_scope"])
    authorized_lease_id = str(lease_id or capability["lease_id"])
    intent = dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation=operation,
            arguments=arguments,
        )
    )
    deadline_ms = NOW_MS + 5_000
    idempotency_key = f"idempotency:eaaef-fabric:{sequence}"
    parameters = dict(
        quack_daemon_state_command_parameters(
            intent,
            request_id=request,
            principal_did=str(capability["command_principal_did"]),
            authority_ref_cid=context["policy"].authority_ref_cid,
            lease_id=authorized_lease_id,
            scope_id=scope,
            deadline_ms=deadline_ms,
            fencing_token=(
                int(capability["fencing_token"])
                if fencing_token is None
                else fencing_token
            ),
            idempotency_key=idempotency_key,
        )
    )
    parameters["authorization_request_cid"] = "sha256:" + f"{sequence:x}"[-1] * 64
    vocabulary = quack_daemon_operation_command_vocabulary()
    command_kind = CommandKind(vocabulary[operation])
    generation = fabric.repository.load_generation()
    command = StateCommand(
        command_id=f"{request}:{operation.replace('.', '-')}",
        command_kind=command_kind,
        store_id=str(capability["store_id"]),
        session_id=authorized_lease_id,
        expected_generation=int(capability["owner_generation"]),
        expected_revision=generation.revision,
        fence_epoch=int(capability["fence_epoch"]),
        idempotency_key=idempotency_key,
        parameters=parameters,
    )
    prepared = authorized_state_command_signing_payload(
        request_id=request,
        submission_id=submission,
        ingress_slot=sequence,
        principal_did=str(capability["command_principal_did"]),
        approver_did=str(context["approver_did"]),
        authority_ref_cid=context["policy"].authority_ref_cid,
        board_namespace=str(capability["board_namespace"]),
        shard_id=str(capability["shard_id"]),
        owner_principal_did=str(capability["owner_principal_did"]),
        lease_id=authorized_lease_id,
        scope_id=scope,
        effect=f"control-plane/{command_kind.value}",
        issued_at_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 10_000,
        deadline_ms=deadline_ms,
        one_use_nonce=one_use_nonce,
        command=command,
    )
    signature = base64.b64encode(
        context["approver_key"].sign(
            json.dumps(
                dict(prepared),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
    ).decode("ascii")
    return intent, seal_authorized_state_command(
        prepared,
        approver_signature=signature,
    )


class _LocalEndpoint:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    def stop(self) -> None:
        self.connection.close()


def _start_local_projection(
    fabric: QuackCommandFabric,
    duckdb: object,
) -> tuple[object, object]:
    """Start only the local test connections; never create a Quack service."""

    fabric._install_ingress()  # noqa: SLF001 - exact local owner fixture
    fabric._install_projection()  # noqa: SLF001
    ingress = duckdb.connect(str(fabric.ingress_database))
    projection = duckdb.connect(str(fabric.projection_database))
    fabric._ingress_server = _LocalEndpoint(ingress)  # noqa: SLF001
    fabric._state_server = _LocalEndpoint(projection)  # noqa: SLF001
    fabric.started = True
    return ingress, projection


def _expire_board_lease(fabric: QuackCommandFabric, scope_id: str) -> None:
    transaction = fabric.repository.transaction(
        expected_generation=fabric.repository.load_generation()
    )
    try:
        transaction.begin()
        transaction._connection.execute(  # noqa: SLF001 - owner test mutation
            "UPDATE leases SET state='expired' WHERE task_cid=?",
            [scope_id],
        )
        transaction.commit()
    except BaseException:
        if transaction.active:
            transaction.rollback()
        raise


def test_typed_eaaef_direct_owner_transaction_replay_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    try:
        assert type(fabric._daemon_operation_handler) is (  # noqa: SLF001
            EAAEFBootstrapBorrowedTransactionOperationHandler
        )
        gateway = fabric.daemon_owner_gateway()
        assert gateway.production_capability_cid == capability["capability_cid"]

        intent, envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 4},
            sequence=1,
        )
        first = gateway.submit_authorized_daemon_operation(envelope, intent)
        assert [record["task_cid"] for record in first["tasks"]] == [
            "task:eaaef:1"
        ]
        assert gateway.submit_authorized_daemon_operation(envelope, intent) == first
        assert len(fabric._private_receipts()) == 1  # noqa: SLF001

        bad_intent, bad_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 0},
            sequence=2,
        )
        with pytest.raises(Exception, match="limit must be"):
            gateway.submit_authorized_daemon_operation(bad_envelope, bad_intent)
        assert len(fabric._private_receipts()) == 1  # noqa: SLF001
        fixed_intent, fixed_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=2,
        )
        fixed = gateway.submit_authorized_daemon_operation(
            fixed_envelope, fixed_intent
        )
        assert [record["task_cid"] for record in fixed["tasks"]] == [
            "task:eaaef:1"
        ]

        stale_intent, stale_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=3,
            fencing_token=int(capability["fencing_token"]) + 1,
        )
        with pytest.raises(
            QuackCommandAuthorizationError,
            match="fencing token is stale",
        ):
            gateway.submit_authorized_daemon_operation(
                stale_envelope, stale_intent
            )
        repaired_intent, repaired_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=3,
        )
        repaired = gateway.submit_authorized_daemon_operation(
            repaired_envelope, repaired_intent
        )
        assert [record["task_cid"] for record in repaired["tasks"]] == [
            "task:eaaef:1"
        ]

        nonce_intent, nonce_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=4,
            nonce=str(repaired_envelope.one_use_nonce),
        )
        with pytest.raises(Exception, match="already consumed"):
            gateway.submit_authorized_daemon_operation(nonce_envelope, nonce_intent)
        assert len(fabric._private_receipts()) == 3  # noqa: SLF001

        mutation = fabric.repository.transaction(
            expected_generation=fabric.repository.load_generation()
        ).begin()
        try:
            mutation._connection.execute(  # noqa: SLF001 - owner test mutation
                "UPDATE leases SET state='expired' WHERE task_cid=?",
                [capability["board_scope"]],
            )
            mutation.commit()
        except BaseException:
            if mutation.active:
                mutation.rollback()
            raise
        expired_intent, expired_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=5,
        )
        with pytest.raises(Exception, match="revoked|not accepted|live"):
            gateway.submit_authorized_daemon_operation(
                expired_envelope, expired_intent
            )
        assert len(fabric._private_receipts()) == 3  # noqa: SLF001
    finally:
        fabric.stop()
        # Assert this test never starts a Quack endpoint or another process.
        assert not (tmp_path / "ingress.duckdb").exists()
        assert duckdb.__version__ == "1.5.5"


def test_typed_eaaef_inbox_uses_same_strong_verifier_and_one_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    fabric._install_ingress()  # noqa: SLF001 - local no-network inbox fixture
    fabric._install_projection()  # noqa: SLF001
    ingress = duckdb.connect(str(fabric.ingress_database))
    projection = duckdb.connect(str(fabric.projection_database))
    fabric._ingress_server = _LocalEndpoint(ingress)  # noqa: SLF001
    fabric._state_server = _LocalEndpoint(projection)  # noqa: SLF001
    fabric.started = True
    intent, envelope = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=20,
    )
    del intent
    encoded = canonical_json_bytes(envelope.to_dict()).decode("ascii")
    ingress.execute(
        "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
        [
            envelope.ingress_slot,
            envelope.submission_id,
            envelope.envelope_cid,
            encoded,
            NOW_MS,
        ],
    )
    try:
        receipts = fabric.apply_pending()
        assert len(receipts) == 1
        assert receipts[0]["outcome"] == "accepted"
        assert receipts[0]["daemon_operation"] == "task.ready"
        result = json.loads(str(receipts[0]["result_json"]))
        assert result["daemon_operation"] == "task.ready"
        assert result["value"]["tasks"][0]["task_cid"] == "task:eaaef:1"

        ingress.execute(
            "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
            [
                envelope.ingress_slot,
                envelope.submission_id,
                envelope.envelope_cid,
                encoded,
                NOW_MS,
            ],
        )
        assert fabric.apply_pending() == ()
        assert len(fabric._private_receipts()) == 1  # noqa: SLF001
        assert ingress.execute("SELECT count(*) FROM command_inbox").fetchone()[0] == 0
    finally:
        fabric.stop()


def test_inbox_exact_reappend_repairs_projection_after_committed_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    ingress, projection = _start_local_projection(fabric, duckdb)
    _intent, envelope = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=21,
    )
    encoded = canonical_json_bytes(envelope.to_dict()).decode("ascii")
    ingress.execute(
        "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
        [
            envelope.ingress_slot,
            envelope.submission_id,
            envelope.envelope_cid,
            encoded,
            NOW_MS,
        ],
    )
    rebuild_projection = fabric._rebuild_projection  # noqa: SLF001

    def fail_projection() -> None:
        raise RuntimeError("injected post-commit projection failure")

    monkeypatch.setattr(fabric, "_rebuild_projection", fail_projection)
    try:
        with pytest.raises(
            RuntimeError,
            match="injected post-commit projection failure",
        ):
            fabric.apply_pending()
        assert len(fabric._private_receipts()) == 1  # noqa: SLF001
        assert projection.execute(
            "SELECT count(*) FROM apply_receipts"
        ).fetchone()[0] == 0
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 1

        _expire_board_lease(fabric, str(capability["board_scope"]))
        fabric._clock_ms = lambda: NOW_MS + 1_000_000  # noqa: SLF001
        private_receipts = fabric._private_receipts  # noqa: SLF001
        private_receipt_reads = 0

        def miss_initial_receipt_snapshot():
            nonlocal private_receipt_reads
            private_receipt_reads += 1
            if private_receipt_reads == 1:
                return ()
            return private_receipts()

        # Exercise the race-safe second lookup: the initial receipt snapshot
        # misses a concurrently durable result, then the transaction sees it.
        # A second projection failure must retain the replay row.
        monkeypatch.setattr(
            fabric,
            "_private_receipts",
            miss_initial_receipt_snapshot,
        )
        with pytest.raises(
            RuntimeError,
            match="injected post-commit projection failure",
        ):
            fabric.apply_pending()
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 1

        monkeypatch.setattr(fabric, "_private_receipts", private_receipts)
        monkeypatch.setattr(fabric, "_rebuild_projection", rebuild_projection)
        assert fabric.apply_pending() == ()
        projected = projection.execute(
            "SELECT submission_id, envelope_cid, outcome FROM apply_receipts"
        ).fetchall()
        assert projected == [
            (envelope.submission_id, envelope.envelope_cid, "accepted")
        ]
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 0
    finally:
        fabric.stop()


def test_divergent_durable_submission_is_quarantined_without_starving_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    ingress, _projection = _start_local_projection(fabric, duckdb)
    first_intent, first = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=23,
    )
    try:
        accepted = fabric.daemon_owner_gateway().submit_authorized_daemon_operation(
            first, first_intent
        )
        assert accepted["tasks"][0]["task_cid"] == "task:eaaef:1"

        _poison_intent, poison = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 2},
            sequence=24,
            submission_id=first.submission_id,
        )
        _next_intent, next_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=25,
        )
        for envelope in (poison, next_envelope):
            ingress.execute(
                "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
                [
                    envelope.ingress_slot,
                    envelope.submission_id,
                    envelope.envelope_cid,
                    canonical_json_bytes(envelope.to_dict()).decode("ascii"),
                    NOW_MS,
                ],
            )

        receipts = fabric.apply_pending()
        assert [receipt["outcome"] for receipt in receipts] == [
            "rejected",
            "accepted",
        ]
        quarantine = receipts[0]
        assert quarantine["schema"] == (
            "ipfs_accelerate_py/agent-supervisor/"
            "divergent-authorized-command-ingress-quarantine@1"
        )
        assert quarantine["original_submission_id"] == first.submission_id
        assert quarantine["durable_envelope_cid"] == first.envelope_cid
        assert quarantine["divergent_envelope_cid"] == poison.envelope_cid
        assert quarantine["authority_reopened"] is False
        assert receipts[1]["submission_id"] == next_envelope.submission_id
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 0
        private = fabric._private_receipts()  # noqa: SLF001
        assert [
            receipt["submission_id"] for receipt in private
        ] == [first.submission_id, next_envelope.submission_id]
        transaction = fabric.repository.transaction(
            expected_generation=fabric.repository.load_generation()
        )
        try:
            transaction.begin()
            durable_quarantine = (
                transaction.lookup_authorized_command_ingress_quarantine(
                    quarantine_event_id=quarantine["quarantine_event_id"]
                )
            )
            transaction.commit()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        assert dict(durable_quarantine or {}) == dict(quarantine)

        # The audit event has a distinct domain-event namespace: an otherwise
        # valid command may use the same text as its submission ID without
        # colliding with or replacing the quarantine.
        _collision_intent, collision = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=29,
            submission_id=str(quarantine["quarantine_event_id"]),
        )
        ingress.execute(
            "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
            [
                collision.ingress_slot,
                collision.submission_id,
                collision.envelope_cid,
                canonical_json_bytes(collision.to_dict()).decode("ascii"),
                NOW_MS,
            ],
        )
        collision_receipts = fabric.apply_pending()
        assert len(collision_receipts) == 1
        assert collision_receipts[0]["outcome"] == "accepted"
        assert collision_receipts[0]["submission_id"] == quarantine[
            "quarantine_event_id"
        ]
        assert dict(durable_quarantine or {}) == dict(quarantine)

        # A response-loss retry carries a fresh transport timestamp but the
        # exact same signed poison envelope.  It adopts the one durable
        # quarantine and cannot prevent a later valid command from advancing.
        _later_intent, later = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.ready",
            arguments={"limit": 1},
            sequence=30,
        )
        for replayed in (poison, later):
            ingress.execute(
                "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
                [
                    replayed.ingress_slot,
                    replayed.submission_id,
                    replayed.envelope_cid,
                    canonical_json_bytes(replayed.to_dict()).decode("ascii"),
                    NOW_MS + 999,
                ],
            )
        replayed_receipts = fabric.apply_pending()
        assert [receipt["outcome"] for receipt in replayed_receipts] == [
            "rejected",
            "accepted",
        ]
        assert replayed_receipts[0]["quarantine_event_id"] == quarantine[
            "quarantine_event_id"
        ]
        assert replayed_receipts[1]["submission_id"] == later.submission_id
        audit = fabric.repository.transaction(
            expected_generation=fabric.repository.load_generation()
        )
        try:
            audit.begin()
            quarantine_count = audit._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM domain_events WHERE event_type="
                "'authorized_state_command_ingress_quarantine'"
            ).fetchone()[0]
            audit.commit()
        except BaseException:
            if audit.active:
                audit.rollback()
            raise
        assert quarantine_count == 1
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 0
    finally:
        fabric.stop()


@pytest.mark.parametrize(
    "recovery_path",
    ("transaction_prior", "exception_recovery"),
)
def test_divergent_receipt_recovery_race_is_quarantined_and_row_is_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_path: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
        StateTransaction,
    )

    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    ingress, _projection = _start_local_projection(fabric, duckdb)
    original_intent, original = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=26,
    )
    fabric.daemon_owner_gateway().submit_authorized_daemon_operation(
        original, original_intent
    )
    _poison_intent, poison = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 2},
        sequence=27,
        submission_id=original.submission_id,
    )
    ingress.execute(
        "INSERT INTO command_inbox VALUES (?, ?, ?, ?, ?)",
        [
            poison.ingress_slot,
            poison.submission_id,
            poison.envelope_cid,
            canonical_json_bytes(poison.to_dict()).decode("ascii"),
            NOW_MS,
        ],
    )
    private_receipts = fabric._private_receipts  # noqa: SLF001
    monkeypatch.setattr(fabric, "_private_receipts", lambda: ())
    lookup = StateTransaction.lookup_authorized_command_receipt
    lookup_calls = 0

    def miss_transaction_lookup_once(self, *, receipt_event_id):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            return None
        return lookup(self, receipt_event_id=receipt_event_id)

    if recovery_path == "exception_recovery":
        monkeypatch.setattr(
            StateTransaction,
            "lookup_authorized_command_receipt",
            miss_transaction_lookup_once,
        )
    try:
        receipts = fabric.apply_pending()
        assert len(receipts) == 1
        assert receipts[0]["outcome"] == "rejected"
        assert receipts[0]["disposition"] == "quarantined"
        if recovery_path == "exception_recovery":
            assert lookup_calls >= 2
        assert ingress.execute(
            "SELECT count(*) FROM command_inbox"
        ).fetchone()[0] == 0
        monkeypatch.setattr(fabric, "_private_receipts", private_receipts)
        assert [
            receipt["submission_id"] for receipt in fabric._private_receipts()  # noqa: SLF001
        ] == [original.submission_id]
    finally:
        fabric.stop()


def test_shared_and_direct_authority_boundaries_reject_subclass_serialization(
    tmp_path: Path,
) -> None:
    capability, context, verified = _verified_capability()
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    intent, envelope = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=28,
    )
    gateway = fabric.daemon_owner_gateway()
    gateway.submit_authorized_daemon_operation(envelope, intent)

    class VirtualEnvelope(AuthorizedStateCommand):
        def unsigned_payload(self):
            return envelope.unsigned_payload()

    virtual_envelope = VirtualEnvelope(
        **{field.name: getattr(envelope, field.name) for field in fields(envelope)}
    )
    with pytest.raises(Exception, match="envelope is untyped"):
        verify_authorized_state_command(
            virtual_envelope,
            policy=context["policy"],
            now_ms=NOW_MS,
        )
    with pytest.raises(Exception, match="envelope is untyped"):
        gateway.submit_authorized_daemon_operation(virtual_envelope, intent)

    class VirtualCommand(StateCommand):
        pass

    virtual_command = VirtualCommand(
        **{
            field.name: getattr(envelope.command, field.name)
            for field in fields(envelope.command)
        }
    )
    virtual_command_envelope = AuthorizedStateCommand(
        **{
            field.name: (
                virtual_command
                if field.name == "command"
                else getattr(envelope, field.name)
            )
            for field in fields(envelope)
        }
    )
    with pytest.raises(Exception, match="embedded command is untyped"):
        verify_authorized_state_command(
            virtual_command_envelope,
            policy=context["policy"],
            now_ms=NOW_MS,
        )
    with pytest.raises(Exception, match="embedded command is untyped"):
        gateway.submit_authorized_daemon_operation(virtual_command_envelope, intent)
    fabric.stop()


def test_direct_exact_replay_adopts_before_expiry_and_repairs_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    _ingress, projection = _start_local_projection(fabric, duckdb)
    gateway = fabric.daemon_owner_gateway()
    intent, envelope = _authorized_operation(
        fabric,
        capability,
        context,
        operation="task.ready",
        arguments={"limit": 1},
        sequence=22,
    )
    rebuild_projection = fabric._rebuild_projection  # noqa: SLF001

    def fail_projection() -> None:
        raise RuntimeError("injected direct projection failure")

    monkeypatch.setattr(fabric, "_rebuild_projection", fail_projection)
    try:
        with pytest.raises(RuntimeError, match="injected direct projection failure"):
            gateway.submit_authorized_daemon_operation(envelope, intent)
        assert len(fabric._private_receipts()) == 1  # noqa: SLF001
        assert projection.execute(
            "SELECT count(*) FROM apply_receipts"
        ).fetchone()[0] == 0

        _expire_board_lease(fabric, str(capability["board_scope"]))
        fabric._clock_ms = lambda: NOW_MS + 1_000_000  # noqa: SLF001
        monkeypatch.setattr(fabric, "_rebuild_projection", rebuild_projection)
        replay = gateway.submit_authorized_daemon_operation(envelope, intent)
        assert replay["tasks"][0]["task_cid"] == "task:eaaef:1"
        assert projection.execute(
            "SELECT submission_id, envelope_cid, outcome FROM apply_receipts"
        ).fetchall() == [
            (envelope.submission_id, envelope.envelope_cid, "accepted")
        ]

        divergent = dict(intent)
        divergent["arguments"] = {"limit": 2}
        with pytest.raises(
            QuackCommandFabricStateError,
            match="replay intent is divergent",
        ):
            gateway.submit_authorized_daemon_operation(envelope, divergent)
    finally:
        fabric.stop()


def test_typed_eaaef_task_lease_and_lane_are_cross_joined_in_owner_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _duckdb, _artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    fabric = _fabric(
        tmp_path,
        capability=capability,
        context=context,
        verified=verified,
    )
    gateway = fabric.daemon_owner_gateway()
    lane_session_id = "session:eaaef-worker-lane-1"
    process_instance_id = "process:eaaef-worker-lane-1"
    lane = {
        "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "owner_principal_did": capability["owner_principal_did"],
        "owner_session_id": capability["owner_session_id"],
        "owner_generation": capability["owner_generation"],
        "lane_session_id": lane_session_id,
        "lane_generation": 1,
        "process_instance_id": process_instance_id,
        "fence_epoch": capability["fence_epoch"],
    }
    metadata = {
        "interface": "DatabaseImplementationDaemon@1",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-implementation-daemon@1"
        ),
        "authority_mode": "quack",
        "logical_owner_session_id": lane_session_id,
        "process_instance_id": process_instance_id,
        "state_schema_revision": capability["state_schema_revision"],
        "gateway_binding_cid": capability["gateway_binding_cid"],
        "gateway_owner_principal_did": capability["owner_principal_did"],
        "gateway_owner_generation": capability["owner_generation"],
        "gateway_fence_epoch": capability["fence_epoch"],
        "gateway_control_plane_schema_version": capability[
            "control_plane_schema_version"
        ],
        "gateway_state_schema_revision": capability["state_schema_revision"],
    }
    try:
        bind_intent, bind_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="execution.bind_daemon",
            arguments={"metadata": metadata, "daemon_lane_binding": lane},
            sequence=30,
        )
        bound = gateway.submit_authorized_daemon_operation(
            bind_envelope, bind_intent
        )
        assert bound["session_id"] == lane_session_id

        claim_intent, claim_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="coordination.claim_ready",
            arguments={
                "owner_session_id": lane_session_id,
                "lease_ms": 10_000,
                "exclude_task_cids": [],
                "now_ms": NOW_MS,
                "daemon_lane_binding": lane,
            },
            sequence=31,
        )
        claim = gateway.submit_authorized_daemon_operation(
            claim_envelope, claim_intent
        )
        assert claim["task_cid"] == "task:eaaef:1"
        assert claim["owner_session_id"] == lane_session_id
        assert int(claim["fencing_token"]) > 0

        task_authority = {
            "schema": EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
            "task_cid": claim["task_cid"],
            "claim_id": claim["claim_id"],
            "attempt_id": claim["attempt_id"],
            "attempt_number": claim["attempt_number"],
            "lease_id": claim["lease_id"],
            "owner_session_id": claim["owner_session_id"],
            "fencing_token": claim["fencing_token"],
            "fence_epoch": claim["fence_epoch"],
            "daemon_lane_binding": lane,
        }
        get_intent, get_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.get",
            arguments={
                "task_cid": claim["task_cid"],
                "task_authority_binding": task_authority,
            },
            sequence=32,
            scope_id=str(claim["task_cid"]),
            lease_id=str(claim["lease_id"]),
            fencing_token=int(claim["fencing_token"]),
        )
        task = gateway.submit_authorized_daemon_operation(
            get_envelope, get_intent
        )
        assert task["task_cid"] == claim["task_cid"]

        crossed_intent, crossed_envelope = _authorized_operation(
            fabric,
            capability,
            context,
            operation="task.get",
            arguments={
                "task_cid": "task:eaaef:other",
                "task_authority_binding": task_authority,
            },
            sequence=33,
            scope_id=str(claim["task_cid"]),
            lease_id=str(claim["lease_id"]),
            fencing_token=int(claim["fencing_token"]),
        )
        with pytest.raises(Exception, match="target differs"):
            gateway.submit_authorized_daemon_operation(
                crossed_envelope, crossed_intent
            )
        assert len(fabric._private_receipts()) == 3  # noqa: SLF001
    finally:
        fabric.stop()


def test_eaaef_capability_and_handler_cannot_cross_generic_protocol(
    tmp_path: Path,
) -> None:
    duckdb, artifact = _exact_runtime()
    capability, context, verified = _verified_capability()
    operational = tmp_path / "operational.duckdb"
    _provision_operational(operational, capability)
    common = {
        "duckdb_module": duckdb,
        "extension_path": artifact,
        "lock_path": _LOCK,
        "machine": _machine_lock_name(),
        "ingress_database": tmp_path / "ingress.duckdb",
        "operational_database": operational,
        "projection_database": tmp_path / "projection.duckdb",
        "ingress_endpoint": capability["command_endpoint"],
        "state_endpoint": capability["state_endpoint"],
        "ingress_token": _INGRESS_TOKEN,
        "state_token": _STATE_TOKEN,
        "authorization_policy": context["policy"],
        "command_fabric_qualification_cid": capability[
            "command_fabric_qualification_cid"
        ],
        "clock_ms": lambda: NOW_MS,
    }
    with pytest.raises(QuackCommandFabricStateError, match="verifier-owned typed"):
        QuackCommandFabric(
            **common,
            eaaef_bootstrap_operational_capability=dict(verified),
        )
    with pytest.raises(QuackCommandFabricStateError, match="mutually exclusive"):
        QuackCommandFabric(
            **common,
            daemon_operational_capability={"schema": "generic-forgery"},
            trusted_daemon_capability_reviewer_dids=("did:key:zreviewer",),
            eaaef_bootstrap_operational_capability=verified,
        )
    with pytest.raises(QuackCommandFabricStateError, match="arbitrary EAAEF"):
        QuackCommandFabric(
            **common,
            eaaef_bootstrap_operational_capability=verified,
            daemon_operation_handler=SimpleNamespace(
                INTERFACE="EAAEFBootstrapBorrowedTransactionOperationHandler@1",
                apply_authorized_daemon_operation=lambda **_kwargs: {"value": None},
            ),
        )
