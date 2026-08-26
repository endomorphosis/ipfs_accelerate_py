from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    StateTransaction,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
    EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
    EAAEF_BOARD_SCOPED_OPERATIONS,
    EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS,
    EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA,
    EAAEF_DAEMON_LANE_BINDING_SCHEMA,
    EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
    EAAEFBootstrapBorrowedTransactionOperationHandler,
    EAAEFBorrowedTransactionAdapter,
    EAAEFBorrowedTransactionConflict,
    EAAEFBorrowedTransactionError,
    EAAEFBorrowedTransactionNotReady,
    eaaef_bootstrap_handler_source_evidence,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    eaaef_board_scheduler_lease_seed,
    install_eaaef_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    COMPLETION_EVIDENCE_SCHEMA,
    completion_evidence_projection_on_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    quack_daemon_operation_command_vocabulary,
)

_BOARD = "external-agent-autonomous-execution-fabric"
_SHARD = "control-shard-0"
_BOARD_SCOPE = f"board:{_BOARD}:{_SHARD}"
_PRINCIPAL = "did:key:z6MkEAAEFWorker"
_OWNER_PRINCIPAL = "did:key:z6MkEAAEFOwner"
_OWNER_SESSION = "session:eaaef-scheduler"
_OWNER_GENERATION = 1
_FENCE_EPOCH = 7
_GATEWAY_BINDING_CID = "sha256:" + "9" * 64
_CONTROL_PLANE_SCHEMA_VERSION = "eaaef-v2"
_STATE_SCHEMA_REVISION = "eaaef-run-v5"
_NOW = time.time_ns() // 1_000_000
_BOARD_LEASE_ID = "lease:eaaef-board-scheduler"


def _lane_process_id(lane_session_id: str) -> str:
    return f"process:{lane_session_id.removeprefix('session:')}"


def _lane_binding(
    lane_session_id: str,
    *,
    process_instance_id: str | None = None,
    lane_generation: int = 1,
) -> dict[str, object]:
    return {
        "schema": EAAEF_DAEMON_LANE_BINDING_SCHEMA,
        "gateway_binding_cid": _GATEWAY_BINDING_CID,
        "owner_principal_did": _OWNER_PRINCIPAL,
        "owner_session_id": _OWNER_SESSION,
        "owner_generation": _OWNER_GENERATION,
        "lane_session_id": lane_session_id,
        "lane_generation": lane_generation,
        "process_instance_id": (
            process_instance_id or _lane_process_id(lane_session_id)
        ),
        "fence_epoch": _FENCE_EPOCH,
    }


def _adapter() -> EAAEFBorrowedTransactionAdapter:
    return EAAEFBorrowedTransactionAdapter(
        board_namespace=_BOARD,
        shard_id=_SHARD,
        owner_principal_did=_OWNER_PRINCIPAL,
        command_principal_did=_PRINCIPAL,
        owner_session_id=_OWNER_SESSION,
        owner_generation=_OWNER_GENERATION,
        fence_epoch=_FENCE_EPOCH,
        gateway_binding_cid=_GATEWAY_BINDING_CID,
        control_plane_schema_version=_CONTROL_PLANE_SCHEMA_VERSION,
        state_schema_revision=_STATE_SCHEMA_REVISION,
    )


def _population(count: int = 8) -> dict[str, object]:
    return {
        "goals": [{"goal_cid": "goal:eaaef", "goal_id": "EAAEF"}],
        "tasks": [
            {
                "task_cid": f"task:eaaef:{index}",
                "task_id": f"EAAEF-{index:03d}",
                "goal_cid": "goal:eaaef",
                "status": "ready",
                "priority": "P1",
            }
            for index in range(1, count + 1)
        ],
    }


def _database(tmp_path: Path, *, count: int = 8) -> Path:
    database = tmp_path / "eaaef-operational.duckdb"
    install_eaaef_operational_schema(
        database,
        application_version="test",
        tool_version="test",
        owner_id="eaaef-test-materializer",
    )
    with DatabaseTaskSource(database, install_schema=False) as source:
        source.materialize(_population(count), repository_tree_id="tree:eaaef")
    seed = eaaef_board_scheduler_lease_seed(
        board_namespace=_BOARD,
        shard_id=_SHARD,
        lease_id=_BOARD_LEASE_ID,
        principal_did=_PRINCIPAL,
        owner_session_id=_OWNER_SESSION,
        owner_generation=_OWNER_GENERATION,
        fencing_token=1,
        fence_epoch=_FENCE_EPOCH,
        issued_at_ms=_NOW - 1_000,
        expires_at_ms=_NOW + 1_000_000,
    )
    row = seed["row"]
    with open_duckdb_connection(database) as connection:
        connection.execute(
            "INSERT INTO store_generations(generation, schema_revision, fence_epoch, "
            "revision, database_uuid, birth_id, created_at) "
            "VALUES (1, 2, ?, 1, '12345678-1234-4234-8234-123456789abc', "
            "'birth:eaaef-test', ?)",
            [_FENCE_EPOCH, "2026-08-18T00:00:00Z"],
        )
        connection.execute(
            "INSERT INTO leases(task_cid, claim_cid, resolution_cid, claimant_did, "
            "logical_epoch, fencing_token, expires_at_ms, attempt, state, started_at_ms, "
            "release_reason, retry_not_before_ms, owner_session_id, fence_epoch, revision, "
            "extension_schema, extension_json, claim_id, attempt_id, attempt_number, "
            "lease_kind, scope_id, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                row[name]
                if name != "extension_json"
                else json.dumps(row[name], sort_keys=True, separators=(",", ":"))
                for name in (
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
            ],
        )
    return database


def _command(
    operation: str,
    *,
    scope: str,
    idempotency_key: str,
) -> SimpleNamespace:
    vocabulary = quack_daemon_operation_command_vocabulary()
    return SimpleNamespace(
        command_kind=CommandKind(vocabulary[operation]),
        parameters={"task_cid": scope, "daemon_operation": operation},
        idempotency_key=idempotency_key,
        fence_epoch=7,
    )


def _apply(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    operation: str,
    arguments: Mapping[str, object],
    *,
    scope: str,
    sequence: int,
) -> object:
    row = transaction._connection.execute(  # noqa: SLF001 - exact test observation
        "SELECT claim_cid, claimant_did, fencing_token, fence_epoch, expires_at_ms, state, "
        "claim_id, attempt_id, attempt_number, owner_session_id "
        "FROM leases WHERE task_cid=?",
        [scope],
    ).fetchone()
    if row is None:
        authorized_lease: dict[str, object] = {"principal_did": _PRINCIPAL}
    else:
        authorized_lease = {
            "lease_id": str(row[0]),
            "principal_did": str(row[1]),
            "fencing_token": int(row[2]),
            "fence_epoch": int(row[3]),
            "expires_at_ms": int(row[4]),
            "state": str(row[5]),
        }
    bound_arguments = dict(arguments)
    if operation in {
        "execution.bind_daemon",
        "coordination.claim_ready",
        "execution.list_running_attempts",
    } and not (
        operation == "execution.list_running_attempts"
        and "recovery_authority" in bound_arguments
    ):
        if operation == "execution.bind_daemon":
            metadata = dict(bound_arguments["metadata"])
            lane_session_id = str(metadata["logical_owner_session_id"])
            process_instance_id = str(metadata["process_instance_id"])
        else:
            lane_session_id = str(bound_arguments["owner_session_id"])
            session_row = transaction._connection.execute(  # noqa: SLF001
                "SELECT daemon_id FROM daemon_sessions WHERE session_id=?",
                [lane_session_id],
            ).fetchone()
            process_instance_id = (
                str(session_row[0])
                if session_row is not None
                else _lane_process_id(lane_session_id)
            )
        bound_arguments["daemon_lane_binding"] = _lane_binding(
            lane_session_id,
            process_instance_id=process_instance_id,
        )
    if operation not in EAAEF_BOARD_SCOPED_OPERATIONS and not (
        operation == "execution.record_event"
        and not str(arguments.get("task_cid") or "")
        and not str(arguments.get("attempt_id") or "")
    ):
        if row is None or not str(row[6] or ""):
            bound_arguments["task_authority_binding"] = {}
            return adapter.apply(
                operation=operation,
                arguments=bound_arguments,
                transaction=transaction,
                command=_command(
                    operation,
                    scope=scope,
                    idempotency_key=f"idem:{operation}:{sequence}",
                ),
                lease=authorized_lease,
            )
        lane_row = transaction._connection.execute(  # noqa: SLF001
            "SELECT metadata_json FROM daemon_sessions WHERE session_id=?",
            [str(row[9])],
        ).fetchone()
        assert lane_row is not None
        lane_metadata = json.loads(str(lane_row[0]))
        bound_arguments["task_authority_binding"] = {
            "schema": EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
            "task_cid": scope,
            "claim_id": str(row[6]),
            "attempt_id": str(row[7]),
            "attempt_number": int(row[8]),
            "lease_id": str(row[0]),
            "owner_session_id": str(row[9]),
            "fencing_token": int(row[2]),
            "fence_epoch": int(row[3]),
            "daemon_lane_binding": lane_metadata["lane_binding"],
        }
    return adapter.apply(
        operation=operation,
        arguments=bound_arguments,
        transaction=transaction,
        command=_command(
            operation,
            scope=scope,
            idempotency_key=f"idem:{operation}:{sequence}",
        ),
        lease=authorized_lease,
    )


def _bind_lane(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    *,
    sequence: int,
    lane_session_id: str | None = None,
) -> str:
    lane_session_id = lane_session_id or f"session:worker:{sequence}"
    _apply(
        adapter,
        transaction,
        "execution.bind_daemon",
        {
            "metadata": {
                "interface": "DatabaseImplementationDaemon@1",
                "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
                "authority_mode": "quack",
                "logical_owner_session_id": lane_session_id,
                "process_instance_id": _lane_process_id(lane_session_id),
                "state_schema_revision": _STATE_SCHEMA_REVISION,
                "gateway_binding_cid": _GATEWAY_BINDING_CID,
                "gateway_owner_principal_did": _OWNER_PRINCIPAL,
                "gateway_owner_generation": _OWNER_GENERATION,
                "gateway_fence_epoch": _FENCE_EPOCH,
                "gateway_control_plane_schema_version": _CONTROL_PLANE_SCHEMA_VERSION,
                "gateway_state_schema_revision": _STATE_SCHEMA_REVISION,
            }
        },
        scope=_BOARD_SCOPE,
        sequence=sequence * 1_000,
    )
    return lane_session_id


def _claim(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    *,
    sequence: int,
    now_ms: int = _NOW,
) -> dict[str, object]:
    lane_session_id = _bind_lane(adapter, transaction, sequence=sequence)
    value = _apply(
        adapter,
        transaction,
        "coordination.claim_ready",
        {
            "owner_session_id": lane_session_id,
            "lease_ms": 60_000,
            "exclude_task_cids": [],
            "now_ms": now_ms,
        },
        scope=_BOARD_SCOPE,
        sequence=sequence,
    )
    assert isinstance(value, dict)
    return value


def _claim_identity(claim: Mapping[str, object]) -> dict[str, object]:
    return {
        name: claim[name]
        for name in (
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "attempt_id",
            "attempt_number",
            "lease_id",
        )
    }


def _attempt_wire(claim: Mapping[str, object]) -> dict[str, object]:
    identity = _claim_identity(claim)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-attempt@1",
        "interface": "DatabaseTaskAttempt@1",
        **identity,
        "task_alias": str(claim.get("body", {}).get("task_alias") or "EAAEF-001"),
        "committed_phase": "claimed",
        "status": "running",
        "started_at_ms": claim["claimed_at_ms"],
        "finished_at_ms": None,
        "revision": 1,
        "body": {},
    }


def _validation_phase_payload(
    claim: Mapping[str, object],
    *,
    tag: str,
    accepted_result: Mapping[str, object],
) -> dict[str, object]:
    dispatch_claim_cid = str(accepted_result["claim_cid"])
    authority_cid = "sha256:" + "a" * 64
    admission_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-host-merge-admission@1",
        "interface": "ExternalAgentContainerWorkerDispatcher@1",
        "decision": "accepted",
        "delivery_mode": "reviewed_patch",
        "task_cid": claim["task_cid"],
        "attempt_id": claim["attempt_id"],
        "claim_cid": dispatch_claim_cid,
        "accepted_result_receipt_id": accepted_result["receipt_id"],
        "patch_artifact_cid": accepted_result["patch_artifact_cid"],
        "reviewer_principal_did": "did:key:zindependentreviewer",
        "effect_authority_cid": authority_cid,
        "merge_commit": "",
    }
    admission = {
        **admission_body,
        "receipt_cid": "sha256:"
        + hashlib.sha256(
            json.dumps(
                admission_body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    return {
        "outcome": "passed",
        "evidence_digest": admission["receipt_cid"],
        "argv": ["external-agent-host-merge-admission"],
        "body": {
            "schema": EAAEF_CONTAINER_VALIDATION_EVIDENCE_SCHEMA,
            "validator": "ExternalAgentContainerWorkerDispatcher@1",
            "task_cid": claim["task_cid"],
            "attempt_id": claim["attempt_id"],
            "control_claim_id": claim["claim_id"],
            "dispatch_claim_cid": dispatch_claim_cid,
            "owner_session_id": claim["owner_session_id"],
            "fencing_token": claim["fencing_token"],
            "fence_epoch": claim["fence_epoch"],
            "authority_cid": authority_cid,
            "admission_receipt": admission,
            "delivery_mode": "reviewed_patch",
            "merge_commit": "",
            "patch_artifact_cid": accepted_result["patch_artifact_cid"],
        },
    }


def _container_dispatch_claim(
    claim: Mapping[str, object], *, tag: str
) -> dict[str, object]:
    def cid(name: str) -> str:
        return "sha256:" + hashlib.sha256(f"{tag}:{name}".encode()).hexdigest()

    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-dispatch-claim@1",
        "interface": "ExternalAgentContainerWorkerDispatcher@1",
        "packet_cid": cid("packet"),
        "task_id": "EAAEF-001",
        "task_cid": claim["task_cid"],
        "attempt_id": claim["attempt_id"],
        "attempt_number": claim["attempt_number"],
        "plan_revision_cid": cid("plan"),
        "repository_tree": "1" * 40,
        "semantic_state_root": cid("semantic"),
        "worktree_id": cid("worktree"),
        "planned_container_id": cid("container"),
        "container_profile_cid": cid("profile"),
        "image_digest": cid("image"),
        "network_authorization_cid": cid("network"),
        "lease_id": claim["lease_id"],
        "fencing_token": claim["fencing_token"],
        "fence_epoch": claim["fence_epoch"],
        "idempotency_key": f"container-dispatch:{tag}",
        "effect_scope_cid": cid("effect-scope"),
        "gateway_binding_cid": _GATEWAY_BINDING_CID,
        "worker_principal_did": _PRINCIPAL,
        "provider_principal_did": "did:key:zprovider",
        "provider": "grok",
        "model_route_cid": cid("route"),
    }
    return {
        **body,
        "claim_cid": "sha256:"
        + hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _container_accepted_result(
    claim: Mapping[str, object],
    dispatch_claim: Mapping[str, object],
    *,
    reservation_id: str,
) -> dict[str, object]:
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-accepted-result@1",
        "interface": "ExternalAgentContainerWorkerDispatcher@1",
        "status": "succeeded",
        "accepted": True,
        "task_result_accepted": False,
        "merge_admitted": False,
        "task_cid": claim["task_cid"],
        "attempt_id": claim["attempt_id"],
        "packet_cid": dispatch_claim["packet_cid"],
        "claim_cid": dispatch_claim["claim_cid"],
        "reservation_id": reservation_id,
        "proposal_receipt_cid": "sha256:" + "d" * 64,
        "verification_receipt_cid": "sha256:" + "e" * 64,
        "patch_artifact_cid": "sha256:" + "f" * 64,
        "artifact_cids": ["sha256:" + "1" * 64],
        "test_receipt_cids": ["sha256:" + "2" * 64],
        "proof_receipt_cids": ["sha256:" + "3" * 64],
        "worker_principal_did": dispatch_claim["worker_principal_did"],
        "independent_verifier_principal_did": "did:key:zverifier",
    }
    return {
        **body,
        "receipt_id": "sha256:"
        + hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _container_effect_result(
    claim: Mapping[str, object],
    dispatch_claim: Mapping[str, object],
    accepted_result: Mapping[str, object],
) -> dict[str, object]:
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-effect-receipt@1",
        "interface": "ExternalAgentContainerWorkerDispatcher@1",
        "status": "applied",
        "effect": "isolated_container_patch_proposal_recorded",
        "effect_key": "eaaef-proposal:" + str(dispatch_claim["claim_cid"]),
        "task_cid": claim["task_cid"],
        "attempt_id": claim["attempt_id"],
        "packet_cid": dispatch_claim["packet_cid"],
        "claim_cid": dispatch_claim["claim_cid"],
        "accepted_result_receipt_id": accepted_result["receipt_id"],
        "patch_artifact_cid": accepted_result["patch_artifact_cid"],
        "task_result_accepted": False,
        "merge_admitted": False,
        "host_mutation_performed": False,
    }
    return {
        **body,
        "receipt_cid": "sha256:"
        + hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _commit_container_dispatch(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    claim: Mapping[str, object],
    *,
    sequence: int,
    tag: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    dispatch_claim = _container_dispatch_claim(claim, tag=tag)
    common = {
        "kind": "external_agent_container_dispatch",
        "record_id": dispatch_claim["claim_cid"],
        "attempt_id": claim["attempt_id"],
        "task_cid": claim["task_cid"],
        "operation_key": dispatch_claim["claim_cid"],
        "idempotency_key": dispatch_claim["idempotency_key"],
        "owner_session_id": claim["owner_session_id"],
        "recorded_at_ms": _NOW,
        "fencing_token": claim["fencing_token"],
        "fence_epoch": claim["fence_epoch"],
        "claim": dispatch_claim,
    }
    reserved = _apply(
        adapter,
        transaction,
        "effect.reserve",
        common,
        scope=str(claim["task_cid"]),
        sequence=sequence,
    )
    accepted = _container_accepted_result(
        claim,
        dispatch_claim,
        reservation_id=str(reserved["reservation_id"]),
    )
    _apply(
        adapter,
        transaction,
        "effect.commit",
        {
            **common,
            "reservation_id": reserved["reservation_id"],
            "result": accepted,
        },
        scope=str(claim["task_cid"]),
        sequence=sequence + 1,
    )
    return dispatch_claim, accepted, _container_effect_result(
        claim, dispatch_claim, accepted
    )


def _ensure_attempt_record(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    claim: Mapping[str, object],
    *,
    sequence: int,
) -> dict[str, object]:
    value = _apply(
        adapter,
        transaction,
        "execution.ensure_attempt",
        {
            "attempt": _attempt_wire(claim),
            "claimed_phase": {
                "phase": "claimed",
                "committed_at_ms": int(claim["claimed_at_ms"]) + 1,
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "revision": 1,
                "body": {},
            },
        },
        scope=str(claim["task_cid"]),
        sequence=sequence,
    )
    assert isinstance(value, dict)
    return value


def _commit_attempt_through_validation(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    claim: Mapping[str, object],
    *,
    sequence: int,
    tag: str,
) -> tuple[dict[str, object], dict[str, object]]:
    timing = transaction._connection.execute(  # noqa: SLF001 - owner test fixture
        "SELECT started_at_ms, COALESCE((SELECT MAX(committed_at_ms) "
        "FROM attempt_phases WHERE attempt_id=?), 0) "
        "FROM task_attempts WHERE attempt_id=?",
        [claim["attempt_id"], claim["attempt_id"]],
    ).fetchone()
    assert timing is not None
    phase_time_base = max(
        int(claim["claimed_at_ms"]) + 10,
        int(timing[0]),
        int(timing[1]),
    )
    revision = 1
    updated: dict[str, object] | None = None
    accepted_result: dict[str, object] | None = None
    container_effect: dict[str, object] | None = None
    for offset, phase in enumerate(("context", "provider", "effect", "validation")):
        body: dict[str, object] = {}
        if phase in {"provider", "effect"}:
            if phase == "provider":
                _, accepted_result, container_effect = _commit_container_dispatch(
                    adapter,
                    transaction,
                    claim,
                    sequence=sequence + 50,
                    tag=tag,
                )
                result = accepted_result
            else:
                assert container_effect is not None
                result = container_effect
            key = f"{phase}:helper:{sequence}"
            reservation = _apply(
                adapter,
                transaction,
                f"{phase}.reserve",
                {
                    "kind": phase,
                    "attempt_id": claim["attempt_id"],
                    "idempotency_key": key,
                },
                scope=str(claim["task_cid"]),
                sequence=sequence + 100 + offset,
            )
            assert isinstance(reservation, dict)
            _apply(
                adapter,
                transaction,
                f"{phase}.commit",
                {
                    "kind": phase,
                    "record_id": reservation["record_id"],
                    "attempt_id": claim["attempt_id"],
                    "task_cid": claim["task_cid"],
                    "operation_key": (
                        "" if phase == "provider" else result["effect_key"]
                    ),
                    "idempotency_key": key,
                    "owner_session_id": claim["owner_session_id"],
                    "recorded_at_ms": int(claim["claimed_at_ms"]) + 5 + offset,
                    "result": result,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                },
                scope=str(claim["task_cid"]),
                sequence=sequence + 200 + offset,
            )
            body = {"idempotency_key": key, "result": result}
        elif phase == "validation":
            assert accepted_result is not None
            body = _validation_phase_payload(
                claim,
                tag=tag,
                accepted_result=accepted_result,
            )
        value = _apply(
            adapter,
            transaction,
            "execution.commit_phase",
            {
                "attempt_id": claim["attempt_id"],
                "expected_revision": revision,
                "expected_status": "running",
                "committed_phase": phase,
                "status": "running",
                "finished_at_ms": None,
                "revision": revision + 1,
                "committed_at_ms": phase_time_base + offset,
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "body": body,
            },
            scope=str(claim["task_cid"]),
            sequence=sequence + offset,
        )
        assert isinstance(value, dict)
        updated = value
        revision += 1
    assert updated is not None
    return updated, dict(updated["body"])


def _mark_claimed_task_in_progress(
    adapter: EAAEFBorrowedTransactionAdapter,
    transaction: StateTransaction,
    claim: Mapping[str, object],
    *,
    expected_revision: int,
    sequence: int,
) -> dict[str, object]:
    value = _apply(
        adapter,
        transaction,
        "task.cas_status",
        {
            "task_cid": claim["task_cid"],
            "expected_revision": expected_revision,
            "status": "in_progress",
            "receipt": {
                "operation": "database_claim",
                "claim_id": claim["claim_id"],
                "attempt_id": claim["attempt_id"],
                "owner_session_id": claim["owner_session_id"],
            },
            "evidence_digests": [],
        },
        scope=str(claim["task_cid"]),
        sequence=sequence,
    )
    assert isinstance(value, dict)
    return value


def test_all_29_borrowed_operations_execute_in_one_owner_transaction(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    adapter = _adapter()
    exercised: set[str] = set()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            task_cid = "task:eaaef:1"
            _apply(
                adapter,
                transaction,
                "coordination.register_task",
                {
                    "task_cid": task_cid,
                    "task_id": "EAAEF-001",
                    "dependency_task_cids": [],
                    "body": {
                        "task_alias": "EAAEF-001",
                        "status": "ready",
                        "priority": "P1",
                    },
                },
                scope=_BOARD_SCOPE,
                sequence=1,
            )
            exercised.add("coordination.register_task")
            claim = _claim(adapter, transaction, sequence=2)
            assert claim["task_cid"] == task_cid
            identity = _claim_identity(claim)
            exercised.add("coordination.claim_ready")

            observed = _apply(
                adapter,
                transaction,
                "coordination.get_claim",
                {"claim_id": claim["claim_id"]},
                scope=task_cid,
                sequence=3,
            )
            assert observed == claim
            exercised.add("coordination.get_claim")
            protected = _apply(
                adapter,
                transaction,
                "coordination.protect_claim",
                {
                    "claim": identity,
                    "expected_task_cid": task_cid,
                    "expected_attempt_id": claim["attempt_id"],
                    "expected_owner_session_id": claim["owner_session_id"],
                    "expected_fencing_token": claim["fencing_token"],
                    "expected_fence_epoch": claim["fence_epoch"],
                    "now_ms": _NOW + 1,
                },
                scope=task_cid,
                sequence=4,
            )
            assert protected["claim_id"] == claim["claim_id"]
            exercised.add("coordination.protect_claim")
            claim = _apply(
                adapter,
                transaction,
                "coordination.renew_lease",
                {
                    "lease": identity,
                    "lease_ms": 90_000,
                    "expected_fencing_token": claim["fencing_token"],
                    "expected_fence_epoch": claim["fence_epoch"],
                    "now_ms": _NOW + 2,
                },
                scope=task_cid,
                sequence=5,
            )
            assert isinstance(claim, dict)
            identity = _claim_identity(claim)
            exercised.add("coordination.renew_lease")

            claimed_task = _apply(
                adapter,
                transaction,
                "task.cas_status",
                {
                    "task_cid": task_cid,
                    "expected_revision": 1,
                    "status": "in_progress",
                    "receipt": {
                        "operation": "database_claim",
                        "claim_id": claim["claim_id"],
                        "attempt_id": claim["attempt_id"],
                        "owner_session_id": claim["owner_session_id"],
                    },
                    "evidence_digests": [],
                },
                scope=task_cid,
                sequence=6,
            )
            assert claimed_task["task"]["status"] == "in_progress"
            exercised.add("task.cas_status")

            attempt = _apply(
                adapter,
                transaction,
                "execution.ensure_attempt",
                {
                    "attempt": {
                        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-attempt@1",
                        "interface": "DatabaseTaskAttempt@1",
                        **identity,
                        "task_alias": "EAAEF-001",
                        "committed_phase": "claimed",
                        "status": "running",
                        "started_at_ms": claim["claimed_at_ms"],
                        "finished_at_ms": None,
                        "revision": 1,
                        "body": {},
                    },
                    "claimed_phase": {
                        "phase": "claimed",
                        "committed_at_ms": int(claim["claimed_at_ms"]) + 3,
                        "fencing_token": claim["fencing_token"],
                        "fence_epoch": claim["fence_epoch"],
                        "revision": 1,
                        "body": {},
                    },
                },
                scope=task_cid,
                sequence=7,
            )
            assert attempt["status"] == "running"
            exercised.add("execution.ensure_attempt")
            assert (
                _apply(
                    adapter,
                    transaction,
                    "execution.get_attempt",
                    {"attempt_id": claim["attempt_id"]},
                    scope=task_cid,
                    sequence=7,
                )["attempt_id"]
                == claim["attempt_id"]
            )
            exercised.add("execution.get_attempt")
            running = _apply(
                adapter,
                transaction,
                "execution.list_running_attempts",
                {"owner_session_id": claim["owner_session_id"]},
                scope=_BOARD_SCOPE,
                sequence=8,
            )
            assert [item["attempt_id"] for item in running] == [claim["attempt_id"]]
            exercised.add("execution.list_running_attempts")
            _apply(
                adapter,
                transaction,
                "execution.record_event",
                {
                    "event_id": "event:eaaef:1",
                    "attempt_id": claim["attempt_id"],
                    "task_cid": task_cid,
                    "event_type": "attempt_started",
                    "recorded_at_ms": _NOW + 4,
                    "body": {},
                },
                scope=task_cid,
                sequence=9,
            )
            exercised.add("execution.record_event")

            _, accepted_result, container_effect = _commit_container_dispatch(
                adapter,
                transaction,
                claim,
                sequence=910,
                tag="all-29",
            )
            for kind in ("provider", "effect"):
                reserve_operation = f"{kind}.reserve"
                commit_operation = f"{kind}.commit"
                key = f"{kind}:idem:1"
                reservation = _apply(
                    adapter,
                    transaction,
                    reserve_operation,
                    {
                        "kind": kind,
                        "attempt_id": claim["attempt_id"],
                        "idempotency_key": key,
                    },
                    scope=task_cid,
                    sequence=10,
                )
                assert reservation["state"] == "newly_reserved"
                exercised.add(reserve_operation)
                result = (
                    accepted_result if kind == "provider" else container_effect
                )
                assert (
                    _apply(
                        adapter,
                        transaction,
                        commit_operation,
                        {
                            "kind": kind,
                            "record_id": reservation["record_id"],
                            "attempt_id": claim["attempt_id"],
                            "task_cid": task_cid,
                            "operation_key": (
                                "" if kind == "provider" else result["effect_key"]
                            ),
                            "idempotency_key": key,
                            "owner_session_id": claim["owner_session_id"],
                            "recorded_at_ms": _NOW + 10,
                            "result": result,
                            "fencing_token": claim["fencing_token"],
                            "fence_epoch": claim["fence_epoch"],
                        },
                        scope=task_cid,
                        sequence=11,
                    )
                    == result
                )
                replay = _apply(
                        adapter,
                        transaction,
                        reserve_operation,
                        {
                            "kind": kind,
                            "attempt_id": claim["attempt_id"],
                            "idempotency_key": key,
                        },
                        scope=task_cid,
                        sequence=12,
                )
                assert replay["state"] == "committed"
                assert replay["result"] == result
                exercised.add(commit_operation)

            validation_payload = _validation_phase_payload(
                claim,
                tag="all-29",
                accepted_result=accepted_result,
            )
            digest = str(validation_payload["evidence_digest"])
            revision = 1
            for phase in ("context", "provider", "effect", "validation"):
                if phase in {"provider", "effect"}:
                    phase_body = {
                        "idempotency_key": f"{phase}:idem:1",
                        "result": (
                            accepted_result
                            if phase == "provider"
                            else container_effect
                        ),
                    }
                elif phase == "validation":
                    phase_body = validation_payload
                else:
                    phase_body = {}
                updated = _apply(
                    adapter,
                    transaction,
                    "execution.commit_phase",
                    {
                        "attempt_id": claim["attempt_id"],
                        "expected_revision": revision,
                        "expected_status": "running",
                        "committed_phase": phase,
                        "status": "running",
                        "finished_at_ms": None,
                        "revision": revision + 1,
                        "committed_at_ms": int(claim["claimed_at_ms"]) + 20 + revision,
                        "fencing_token": claim["fencing_token"],
                        "fence_epoch": claim["fence_epoch"],
                        "body": phase_body,
                    },
                    scope=task_cid,
                    sequence=20 + revision,
                )
                revision += 1
                assert updated["revision"] == revision
            validation_payload = dict(updated["body"])
            digest = str(validation_payload["evidence_digest"])
            exercised.add("execution.commit_phase")
            history = _apply(
                adapter,
                transaction,
                "execution.phase_history",
                {"attempt_id": claim["attempt_id"]},
                scope=task_cid,
                sequence=30,
            )
            assert [item["phase"] for item in history] == [
                "claimed",
                "context",
                "provider",
                "effect",
                "validation",
            ]
            exercised.add("execution.phase_history")

            for operation, evidence in (
                ("validation.record", digest),
                ("task.record_validation", digest),
            ):
                result = _apply(
                    adapter,
                    transaction,
                    operation,
                    {
                        "task_cid": task_cid,
                        "attempt_id": claim["attempt_id"],
                        "outcome": "passed",
                        "evidence_digest": evidence,
                        "argv": validation_payload["argv"],
                        "body": validation_payload["body"],
                    },
                    scope=task_cid,
                    sequence=31,
                )
                assert result["outcome"] == "passed"
                exercised.add(operation)

            prepared = _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": identity,
                    "control_expected_revision": 2,
                    "control_expected_status": "in_progress",
                    "evidence_digest": digest,
                    "body": {},
                    "now_ms": _NOW + 40,
                },
                scope=task_cid,
                sequence=32,
            )
            exercised.add("coordination.prepare_completion")
            assert (
                _apply(
                    adapter,
                    transaction,
                    "coordination.get_prepared_completion",
                    {"task_cid": task_cid},
                    scope=task_cid,
                    sequence=33,
                )["preparation_digest"]
                == prepared["preparation_digest"]
            )
            exercised.add("coordination.get_prepared_completion")
            completion_receipt = {
                "operation": "database_complete",
                "attempt_id": claim["attempt_id"],
                "claim_id": claim["claim_id"],
                "lease_id": claim["lease_id"],
                "owner_session_id": claim["owner_session_id"],
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "evidence_digest": digest,
                "coordination_preparation": prepared,
                "validation": {
                    **validation_payload,
                },
            }
            cas = _apply(
                adapter,
                transaction,
                "task.cas_status",
                {
                    "task_cid": task_cid,
                    "expected_revision": 2,
                    "status": "completed",
                    "receipt": completion_receipt,
                    "evidence_digests": [digest],
                },
                scope=task_cid,
                sequence=34,
            )
            assert cas["task"]["status"] == "completed"
            completion_projection = completion_evidence_projection_on_connection(
                transaction._connection,  # noqa: SLF001 - owner snapshot assertion
                task_cids=(task_cid,),
                transaction_owned_by_caller=True,
            )
            assert completion_projection["completion_receipts"] == [
                {
                    "receipt_cid": cas["receipt_cid"],
                    "task_cid": task_cid,
                    "goal_cid": "goal:eaaef",
                    "attempt_id": claim["attempt_id"],
                    "claim_cid": claim["claim_id"],
                    "fencing_token": claim["fencing_token"],
                    "completed_at": completion_projection["completion_receipts"][0][
                        "completed_at"
                    ],
                    "validation_run_id": "",
                    "evidence_digest": completion_projection[
                        "completion_receipts"
                    ][0]["evidence_digest"],
                    "body": {
                        "schema": COMPLETION_EVIDENCE_SCHEMA,
                        "receipt": completion_receipt,
                        "evidence_digests": [digest],
                        "revision": 3,
                    },
                }
            ]
            replayed_cas = _apply(
                adapter,
                transaction,
                "task.cas_status",
                {
                    "task_cid": task_cid,
                    "expected_revision": 3,
                    "status": "completed",
                    "receipt": completion_receipt,
                    "evidence_digests": [digest],
                },
                scope=task_cid,
                sequence=34,
            )
            assert replayed_cas == {
                **cas,
                "previous_status": "completed",
                "changed": False,
            }
            exercised.add("task.cas_status")
            completed = _apply(
                adapter,
                transaction,
                "coordination.complete_claim",
                {
                    "claim": identity,
                    "control_completion_receipt": cas,
                    "now_ms": _NOW + 41,
                },
                scope=task_cid,
                sequence=35,
            )
            assert completed["status"] == "succeeded"
            exercised.add("coordination.complete_claim")
            terminal_time = max(
                _NOW + 42,
                int(history[-1]["committed_at_ms"]) + 1,
            )
            terminal_attempt = _apply(
                adapter,
                transaction,
                "execution.commit_phase",
                {
                    "attempt_id": claim["attempt_id"],
                    "expected_revision": revision,
                    "expected_status": "running",
                    "committed_phase": "complete",
                    "status": "succeeded",
                    "finished_at_ms": terminal_time,
                    "revision": revision + 1,
                    "committed_at_ms": terminal_time,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                    "body": {"evidence_digest": digest},
                },
                scope=task_cid,
                sequence=35,
            )
            assert terminal_attempt["committed_phase"] == "complete"
            unsettled = _apply(
                adapter,
                transaction,
                "coordination.list_unsettled_completions",
                {"limit": 10, "now_ms": terminal_time},
                scope=_BOARD_SCOPE,
                sequence=36,
            )
            assert [item["preparation"]["task_cid"] for item in unsettled] == [
                task_cid
            ]
            exercised.add("coordination.list_unsettled_completions")
            settled = _apply(
                adapter,
                transaction,
                "coordination.settle_claim",
                {
                    "claim": identity,
                    "reason": "attempt_complete",
                    "now_ms": terminal_time + 1,
                },
                scope=task_cid,
                sequence=37,
            )
            assert settled["state"] == "released"
            assert (
                _apply(
                    adapter,
                    transaction,
                    "coordination.list_unsettled_completions",
                    {"limit": 10, "now_ms": terminal_time + 2},
                    scope=_BOARD_SCOPE,
                    sequence=37,
                )
                == []
            )
            exercised.add("coordination.settle_claim")
            _apply(
                adapter,
                transaction,
                "execution.bind_daemon",
                {
                    "metadata": {
                        "interface": "DatabaseImplementationDaemon@1",
                        "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
                        "authority_mode": "quack",
                        "logical_owner_session_id": claim["owner_session_id"],
                        "process_instance_id": _lane_process_id(
                            str(claim["owner_session_id"])
                        ),
                        "state_schema_revision": _STATE_SCHEMA_REVISION,
                        "gateway_binding_cid": _GATEWAY_BINDING_CID,
                        "gateway_owner_principal_did": _OWNER_PRINCIPAL,
                        "gateway_owner_generation": _OWNER_GENERATION,
                        "gateway_fence_epoch": _FENCE_EPOCH,
                        "gateway_control_plane_schema_version": _CONTROL_PLANE_SCHEMA_VERSION,
                        "gateway_state_schema_revision": _STATE_SCHEMA_REVISION,
                    }
                },
                scope=_BOARD_SCOPE,
                sequence=38,
            )
            exercised.add("execution.bind_daemon")
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise

    # Recovery operations are tested with independent tasks so their terminal
    # states cannot alias the normal completion trace above.
    recovery_cases = (
        ("coordination.reconcile_promoted_completion", 2, True),
        ("coordination.recover_prepared_completion", 3, True),
        ("coordination.abort_prepared_completion", 4, False),
    )
    for case_number, (operation, task_number, completed_control) in enumerate(
        recovery_cases,
        start=100,
    ):
        with open_duckdb_connection(database) as connection:
            transaction = StateTransaction(connection, store_id="eaaef-control").begin()
            try:
                claim = _claim(adapter, transaction, sequence=case_number)
                assert claim["task_cid"] == f"task:eaaef:{task_number}"
                identity = _claim_identity(claim)
                _mark_claimed_task_in_progress(
                    adapter,
                    transaction,
                    claim,
                    expected_revision=1,
                    sequence=case_number + 1,
                )
                _ensure_attempt_record(
                    adapter, transaction, claim, sequence=case_number + 2
                )
                validation_attempt, validation_payload = _commit_attempt_through_validation(
                    adapter,
                    transaction,
                    claim,
                    sequence=case_number + 3,
                    tag=f"recovery-{case_number}",
                )
                evidence = str(validation_payload["evidence_digest"])
                prepared = _apply(
                    adapter,
                    transaction,
                    "coordination.prepare_completion",
                    {
                        "claim": identity,
                        "control_expected_revision": 2,
                        "control_expected_status": "in_progress",
                        "evidence_digest": evidence,
                        "body": {},
                        "now_ms": _NOW,
                    },
                    scope=claim["task_cid"],
                    sequence=case_number + 1,
                )
                observation: dict[str, object]
                if completed_control:
                    observation = _apply(
                        adapter,
                        transaction,
                        "task.cas_status",
                        {
                            "task_cid": claim["task_cid"],
                            "expected_revision": 2,
                            "status": "completed",
                            "receipt": {
                                "operation": "database_complete",
                                "coordination_preparation": prepared,
                                "attempt_id": claim["attempt_id"],
                                "claim_id": claim["claim_id"],
                                "lease_id": claim["lease_id"],
                                "owner_session_id": claim["owner_session_id"],
                                "fencing_token": claim["fencing_token"],
                                "fence_epoch": claim["fence_epoch"],
                                "evidence_digest": evidence,
                                "validation": {
                                    **validation_payload,
                                },
                            },
                            "evidence_digests": [evidence],
                        },
                        scope=claim["task_cid"],
                        sequence=case_number + 3,
                    )
                    arguments = {
                        "task_cid": claim["task_cid"],
                        "control_completion_receipt": observation,
                        "now_ms": _NOW + 70_000,
                    }
                else:
                    observed_task = adapter._task_record(  # noqa: SLF001
                        transaction._connection,  # noqa: SLF001
                        str(claim["task_cid"]),
                    )
                    assert observed_task is not None
                    observation = observed_task
                    arguments = {
                        "task_cid": claim["task_cid"],
                        "control_task_observation": observation,
                        "reason": "expired_before_control_cas",
                        "now_ms": _NOW + 70_000,
                    }
                if operation != "coordination.reconcile_promoted_completion":
                    transaction._connection.execute(  # noqa: SLF001
                        "UPDATE leases SET expires_at_ms=? WHERE task_cid=?",
                        [_NOW - 1, claim["task_cid"]],
                    )
                result = _apply(
                    adapter,
                    transaction,
                    operation,
                    arguments,
                    scope=_BOARD_SCOPE,
                    sequence=case_number + 4,
                )
                assert result["status"] == ("succeeded" if completed_control else "aborted")
                exercised.add(operation)
                if operation != "coordination.abort_prepared_completion":
                    attempt = _apply(
                        adapter,
                        transaction,
                        "execution.get_attempt",
                        {"attempt_id": claim["attempt_id"]},
                        scope=claim["task_cid"],
                        sequence=case_number + 5,
                    )
                    reconciled = _apply(
                        adapter,
                        transaction,
                        "execution.commit_reconciled_attempt",
                        {
                            "attempt_id": claim["attempt_id"],
                            "expected_revision": validation_attempt["revision"],
                            "expected_status": attempt["status"],
                            "committed_phase": "complete",
                            "status": "succeeded",
                            "finished_at_ms": _NOW + 70_001,
                            "revision": validation_attempt["revision"] + 1,
                            "committed_at_ms": _NOW + 70_001,
                            "fencing_token": claim["fencing_token"],
                            "fence_epoch": claim["fence_epoch"],
                            "preparation": prepared,
                            "reconciliation": result,
                            "body": {},
                        },
                        scope=_BOARD_SCOPE,
                        sequence=case_number + 6,
                    )
                    assert reconciled["status"] == "succeeded"
                    exercised.add("execution.commit_reconciled_attempt")
                transaction.commit()
            except BaseException:
                transaction.rollback()
                raise

    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=200, now_ms=_NOW)
            transaction._connection.execute(  # noqa: SLF001
                "UPDATE leases SET expires_at_ms=? WHERE task_cid=?",
                [_NOW - 1, claim["task_cid"]],
            )
            transaction._connection.execute(  # noqa: SLF001
                "UPDATE task_claims SET expires_at_ms=? WHERE claim_id=?",
                [_NOW - 1, claim["claim_id"]],
            )
            expired = _apply(
                adapter,
                transaction,
                "coordination.expire_claim",
                {"claim": _claim_identity(claim), "now_ms": _NOW + 70_000},
                scope=_BOARD_SCOPE,
                sequence=201,
            )
            assert expired["state"] == "expired"
            exercised.add("coordination.expire_claim")
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise

    assert exercised == set(EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS)


def test_five_claim_lanes_are_unique_and_board_lease_is_required(tmp_path: Path) -> None:
    database = _database(tmp_path, count=5)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            with pytest.raises(Exception, match="board/shard claim lease"):
                _apply(
                    adapter,
                    transaction,
                    "coordination.claim_ready",
                    {
                        "owner_session_id": "session:wrong",
                        "lease_ms": 60_000,
                        "exclude_task_cids": [],
                        "now_ms": _NOW,
                    },
                    scope="task:eaaef:1",
                    sequence=1,
                )
            claims = [_claim(adapter, transaction, sequence=index) for index in range(1, 6)]
            assert len({item["task_cid"] for item in claims}) == 5
            assert len({item["claim_id"] for item in claims}) == 5
            assert len({item["attempt_id"] for item in claims}) == 5
            for index, claim in enumerate(claims, start=1):
                with pytest.raises(
                    Exception,
                    match="(exact task claim lease|task authority binding|bounded identifier)",
                ):
                    _apply(
                        adapter,
                        transaction,
                        "coordination.protect_claim",
                        {
                            "claim": _claim_identity(claim),
                            "expected_task_cid": claim["task_cid"],
                            "expected_attempt_id": claim["attempt_id"],
                            "expected_owner_session_id": claim["owner_session_id"],
                            "expected_fencing_token": claim["fencing_token"],
                            "expected_fence_epoch": claim["fence_epoch"],
                            "now_ms": _NOW + 1,
                        },
                        scope=_BOARD_SCOPE,
                        sequence=100 + index,
                    )
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise


def test_crash_rollback_replay_and_stale_fence_fail_closed(tmp_path: Path) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        claim = _claim(adapter, transaction, sequence=1)
        transaction.rollback()
    with open_duckdb_connection(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_claims").fetchone()[0] == 0
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        claim = _claim(adapter, transaction, sequence=1)
        identity = _claim_identity(claim)
        with pytest.raises(EAAEFBorrowedTransactionConflict, match="differs"):
            _apply(
                adapter,
                transaction,
                "coordination.protect_claim",
                {
                    "claim": {**identity, "fencing_token": claim["fencing_token"] + 1},
                    "expected_task_cid": claim["task_cid"],
                    "expected_attempt_id": claim["attempt_id"],
                    "expected_owner_session_id": claim["owner_session_id"],
                    "expected_fencing_token": claim["fencing_token"] + 1,
                    "expected_fence_epoch": claim["fence_epoch"],
                    "now_ms": _NOW + 1,
                },
                scope=claim["task_cid"],
                sequence=2,
            )
        transaction.commit()


def test_handler_is_source_implemented_but_not_self_admitted() -> None:
    handler = EAAEFBootstrapBorrowedTransactionOperationHandler(
        board_namespace=_BOARD,
        shard_id=_SHARD,
        owner_principal_did=_OWNER_PRINCIPAL,
        command_principal_did=_PRINCIPAL,
        owner_session_id=_OWNER_SESSION,
        owner_generation=_OWNER_GENERATION,
        fence_epoch=_FENCE_EPOCH,
        gateway_binding_cid=_GATEWAY_BINDING_CID,
        control_plane_schema_version=_CONTROL_PLANE_SCHEMA_VERSION,
        state_schema_revision=_STATE_SCHEMA_REVISION,
    )
    evidence = handler.evidence()
    assert evidence["qualification_status"] == (
        EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS
    )
    assert evidence["production_admitted"] is False
    assert evidence["operation_count"] == 31
    assert len(evidence["borrowed_transaction_operations"]) == 29
    assert evidence["handler_source_evidence_cid"]
    assert evidence == eaaef_bootstrap_handler_source_evidence(
        board_namespace=_BOARD,
        shard_id=_SHARD,
    )
    assert evidence["runtime_authority_fields"] == [
        "board_namespace",
        "shard_id",
        "owner_principal_did",
        "command_principal_did",
        "owner_session_id",
        "owner_generation",
        "fence_epoch",
        "gateway_binding_cid",
        "control_plane_schema_version",
        "state_schema_revision",
    ]


def test_owner_and_command_principals_are_distinct_and_exact(tmp_path: Path) -> None:
    with pytest.raises(EAAEFBorrowedTransactionError, match="must be distinct"):
        EAAEFBorrowedTransactionAdapter(
            board_namespace=_BOARD,
            shard_id=_SHARD,
            owner_principal_did=_PRINCIPAL,
            command_principal_did=_PRINCIPAL,
            owner_session_id=_OWNER_SESSION,
            owner_generation=_OWNER_GENERATION,
            fence_epoch=_FENCE_EPOCH,
            gateway_binding_cid=_GATEWAY_BINDING_CID,
            control_plane_schema_version=_CONTROL_PLANE_SCHEMA_VERSION,
            state_schema_revision=_STATE_SCHEMA_REVISION,
        )

    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            row = transaction._connection.execute(  # noqa: SLF001
                "SELECT claimant_did FROM leases WHERE task_cid=?",
                [_BOARD_SCOPE],
            ).fetchone()
            assert str(row[0]) == _PRINCIPAL
            transaction._connection.execute(  # noqa: SLF001
                "UPDATE leases SET claimant_did=? WHERE task_cid=?",
                [_OWNER_PRINCIPAL, _BOARD_SCOPE],
            )
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="scheduler lease differs",
            ):
                _bind_lane(
                    adapter,
                    transaction,
                    sequence=998,
                    lane_session_id="session:worker:principal-negative",
                )
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_adapter_source_has_no_transaction_or_path_escape() -> None:
    source = Path(
        "ipfs_accelerate_py/agent_supervisor/task_sources/"
        "eaaef_borrowed_transaction.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "BEGIN TRANSACTION",
        'execute("COMMIT',
        'execute("ROLLBACK',
        "duckdb.connect",
        "open_duckdb_connection",
        ".database_path",
        "ATTACH ",
    )
    assert not [token for token in forbidden if token in source]


def test_terminal_task_cas_and_claim_completion_require_canonical_barrier_receipt(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=300)
            _mark_claimed_task_in_progress(
                adapter,
                transaction,
                claim,
                expected_revision=1,
                sequence=301,
            )
            _ensure_attempt_record(adapter, transaction, claim, sequence=302)
            _, validation_payload = _commit_attempt_through_validation(
                adapter,
                transaction,
                claim,
                sequence=303,
                tag="terminal-receipt",
            )
            digest = str(validation_payload["evidence_digest"])
            incomplete_receipt = {
                "operation": "database_complete",
                "attempt_id": claim["attempt_id"],
                "claim_id": claim["claim_id"],
                "lease_id": claim["lease_id"],
                "owner_session_id": claim["owner_session_id"],
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "evidence_digest": digest,
                "coordination_preparation": {},
                "validation": {
                    **validation_payload,
                },
            }
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="prepared completion barrier",
            ):
                _apply(
                    adapter,
                    transaction,
                    "task.cas_status",
                    {
                        "task_cid": claim["task_cid"],
                        "expected_revision": 2,
                        "status": "completed",
                        "receipt": incomplete_receipt,
                        "evidence_digests": [digest],
                    },
                    scope=str(claim["task_cid"]),
                    sequence=304,
                )
            prepared = _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": _claim_identity(claim),
                    "control_expected_revision": 2,
                    "control_expected_status": "in_progress",
                    "evidence_digest": digest,
                    "body": {},
                    "now_ms": _NOW,
                },
                scope=str(claim["task_cid"]),
                sequence=305,
            )
            divergent_validation = json.loads(json.dumps(validation_payload))
            divergent_validation["body"]["delivery_mode"] = "merge_accepted"
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="committed phase",
            ):
                _apply(
                    adapter,
                    transaction,
                    "task.cas_status",
                    {
                        "task_cid": claim["task_cid"],
                        "expected_revision": 2,
                        "status": "completed",
                        "receipt": {
                            **incomplete_receipt,
                            "coordination_preparation": prepared,
                            "validation": divergent_validation,
                        },
                        "evidence_digests": [digest],
                    },
                    scope=str(claim["task_cid"]),
                    sequence=305,
                )
            task = adapter._task_record(  # noqa: SLF001
                transaction._connection, str(claim["task_cid"])  # noqa: SLF001
            )
            assert task is not None and task["status"] == "in_progress"
            forged = {
                "schema": "ipfs_accelerate_py/agent-supervisor/database-task-cas@1",
                "task": {**task, "status": "completed", "revision": 3},
                "previous_status": "in_progress",
                "revision": 3,
                "event_cursor": 0,
                "changed": True,
                "receipt_cid": "sha256:" + "e" * 64,
            }
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="canonical durable receipt",
            ):
                _apply(
                    adapter,
                    transaction,
                    "coordination.complete_claim",
                    {
                        "claim": _claim_identity(claim),
                        "control_completion_receipt": forged,
                        "now_ms": _NOW,
                    },
                    scope=str(claim["task_cid"]),
                    sequence=306,
                )
            assert adapter._task_record(  # noqa: SLF001
                transaction._connection, str(claim["task_cid"])  # noqa: SLF001
            )["status"] == "in_progress"
            receipt = {**incomplete_receipt, "coordination_preparation": prepared}
            cas = _apply(
                adapter,
                transaction,
                "task.cas_status",
                {
                    "task_cid": claim["task_cid"],
                    "expected_revision": 2,
                    "status": "completed",
                    "receipt": receipt,
                    "evidence_digests": [digest],
                },
                scope=str(claim["task_cid"]),
                sequence=307,
            )
            forged_cas = {**cas, "task": {**cas["task"], "revision": 999}}
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="canonical stored state",
            ):
                _apply(
                    adapter,
                    transaction,
                    "coordination.complete_claim",
                    {
                        "claim": _claim_identity(claim),
                        "control_completion_receipt": forged_cas,
                        "now_ms": _NOW,
                    },
                    scope=str(claim["task_cid"]),
                    sequence=308,
                )
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_terminal_task_cas_replay_rejects_pre_normalization_raw_receipt_row(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=350)
            _mark_claimed_task_in_progress(
                adapter,
                transaction,
                claim,
                expected_revision=1,
                sequence=351,
            )
            _ensure_attempt_record(adapter, transaction, claim, sequence=352)
            _, validation_payload = _commit_attempt_through_validation(
                adapter,
                transaction,
                claim,
                sequence=353,
                tag="legacy-raw-completion-receipt",
            )
            digest = str(validation_payload["evidence_digest"])
            prepared = _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": _claim_identity(claim),
                    "control_expected_revision": 2,
                    "control_expected_status": "in_progress",
                    "evidence_digest": digest,
                    "body": {},
                    "now_ms": _NOW,
                },
                scope=str(claim["task_cid"]),
                sequence=354,
            )
            receipt = {
                "operation": "database_complete",
                "attempt_id": claim["attempt_id"],
                "claim_id": claim["claim_id"],
                "lease_id": claim["lease_id"],
                "owner_session_id": claim["owner_session_id"],
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "evidence_digest": digest,
                "coordination_preparation": prepared,
                "validation": validation_payload,
            }
            cas = _apply(
                adapter,
                transaction,
                "task.cas_status",
                {
                    "task_cid": claim["task_cid"],
                    "expected_revision": 2,
                    "status": "completed",
                    "receipt": receipt,
                    "evidence_digests": [digest],
                },
                scope=str(claim["task_cid"]),
                sequence=355,
            )
            before = adapter._task_record(  # noqa: SLF001 - exact state assertion
                transaction._connection,  # noqa: SLF001
                str(claim["task_cid"]),
            )
            assert before is not None and before["revision"] == 3

            # This is the exact pre-normalization representation: the raw
            # control receipt occupied completion_receipts.body_json directly.
            legacy_raw_body = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            transaction._connection.execute(  # noqa: SLF001 - corruption fixture
                "UPDATE completion_receipts SET body_json=? WHERE receipt_cid=?",
                [legacy_raw_body, cas["receipt_cid"]],
            )

            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="not normalized for the current task revision",
            ):
                _apply(
                    adapter,
                    transaction,
                    "task.cas_status",
                    {
                        "task_cid": claim["task_cid"],
                        "expected_revision": 3,
                        "status": "completed",
                        "receipt": receipt,
                        "evidence_digests": [digest],
                    },
                    scope=str(claim["task_cid"]),
                    sequence=356,
                )

            assert adapter._task_record(  # noqa: SLF001
                transaction._connection,  # noqa: SLF001
                str(claim["task_cid"]),
            ) == before
            durable_raw_body = transaction._connection.execute(  # noqa: SLF001
                "SELECT body_json FROM completion_receipts WHERE receipt_cid=?",
                [cas["receipt_cid"]],
            ).fetchone()
            assert durable_raw_body is not None
            assert str(durable_raw_body[0]) == legacy_raw_body
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_aborted_barrier_allows_only_higher_fenced_retry_and_old_attempt_is_inert(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            old_claim = _claim(adapter, transaction, sequence=400)
            _mark_claimed_task_in_progress(
                adapter,
                transaction,
                old_claim,
                expected_revision=1,
                sequence=401,
            )
            old_attempt = _ensure_attempt_record(
                adapter, transaction, old_claim, sequence=402
            )
            old_attempt, old_validation_payload = _commit_attempt_through_validation(
                adapter,
                transaction,
                old_claim,
                sequence=403,
                tag="old-aborted",
            )
            old_evidence_digest = str(old_validation_payload["evidence_digest"])
            old_prepared = _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": _claim_identity(old_claim),
                    "control_expected_revision": 2,
                    "control_expected_status": "in_progress",
                    "evidence_digest": old_evidence_digest,
                    "body": {},
                    "now_ms": _NOW,
                },
                scope=str(old_claim["task_cid"]),
                sequence=403,
            )
            for table, key in (
                ("leases", "task_cid"),
                ("task_claims", "claim_id"),
            ):
                identity = old_claim["task_cid"] if table == "leases" else old_claim["claim_id"]
                transaction._connection.execute(  # noqa: SLF001
                    f"UPDATE {table} SET expires_at_ms=? WHERE {key}=?",
                    [_NOW - 1, identity],
                )
            observation = adapter._task_record(  # noqa: SLF001
                transaction._connection, str(old_claim["task_cid"])  # noqa: SLF001
            )
            aborted = _apply(
                adapter,
                transaction,
                "coordination.abort_prepared_completion",
                {
                    "task_cid": old_claim["task_cid"],
                    "control_task_observation": observation,
                    "reason": "expired_before_control_cas",
                    "now_ms": _NOW,
                },
                scope=_BOARD_SCOPE,
                sequence=404,
            )
            assert aborted["status"] == "aborted"
            reset_task = adapter._task_record(  # noqa: SLF001
                transaction._connection, str(old_claim["task_cid"])  # noqa: SLF001
            )
            assert reset_task is not None and reset_task["status"] == "ready"
            retired_attempt = _apply(
                adapter,
                transaction,
                "execution.commit_reconciled_attempt",
                {
                    "attempt_id": old_claim["attempt_id"],
                    "expected_revision": int(old_attempt["revision"]),
                    "expected_status": "running",
                    "committed_phase": "failed",
                    "status": "failed",
                    "finished_at_ms": int(old_attempt["started_at_ms"]) + 100,
                    "revision": int(old_attempt["revision"]) + 1,
                    "committed_at_ms": int(old_attempt["started_at_ms"]) + 100,
                    "fencing_token": old_claim["fencing_token"],
                    "fence_epoch": old_claim["fence_epoch"],
                    "preparation": old_prepared,
                    "reconciliation": aborted,
                    "body": {"crash_recovered": True},
                },
                scope=_BOARD_SCOPE,
                sequence=404,
            )
            assert retired_attempt["status"] == "failed"
            new_claim = _claim(adapter, transaction, sequence=405)
            assert new_claim["task_cid"] == old_claim["task_cid"]
            assert new_claim["attempt_number"] > old_claim["attempt_number"]
            assert new_claim["fencing_token"] > old_claim["fencing_token"]
            assert _apply(
                adapter,
                transaction,
                "coordination.get_claim",
                {"claim_id": old_claim["claim_id"]},
                scope=str(new_claim["task_cid"]),
                sequence=406,
            ) is None
            assert _apply(
                adapter,
                transaction,
                "execution.get_attempt",
                {"attempt_id": old_claim["attempt_id"]},
                scope=str(new_claim["task_cid"]),
                sequence=407,
            ) is None
            assert _apply(
                adapter,
                transaction,
                "execution.phase_history",
                {"attempt_id": old_claim["attempt_id"]},
                scope=str(new_claim["task_cid"]),
                sequence=408,
            ) == []
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="not the current task claim",
            ):
                _apply(
                    adapter,
                    transaction,
                    "execution.record_event",
                    {
                        "event_id": "event:stale-attempt",
                        "attempt_id": old_claim["attempt_id"],
                        "task_cid": old_claim["task_cid"],
                        "event_type": "stale_attempt",
                        "recorded_at_ms": _NOW,
                        "body": {},
                    },
                    scope=str(new_claim["task_cid"]),
                    sequence=409,
                )
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="not the current task claim",
            ):
                _apply(
                    adapter,
                    transaction,
                    "provider.reserve",
                    {
                        "kind": "provider",
                        "attempt_id": old_claim["attempt_id"],
                        "idempotency_key": "provider:stale",
                    },
                    scope=str(new_claim["task_cid"]),
                    sequence=410,
                )
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="validation attempt",
            ):
                _apply(
                    adapter,
                    transaction,
                    "validation.record",
                    {
                        "task_cid": old_claim["task_cid"],
                        "attempt_id": old_claim["attempt_id"],
                        "outcome": "passed",
                        "evidence_digest": "sha256:" + "1" * 64,
                        "argv": ["pytest"],
                        "body": {},
                    },
                    scope=str(new_claim["task_cid"]),
                    sequence=411,
                )
            _mark_claimed_task_in_progress(
                adapter,
                transaction,
                new_claim,
                expected_revision=int(reset_task["revision"]),
                sequence=412,
            )
            _ensure_attempt_record(adapter, transaction, new_claim, sequence=413)
            new_validation_attempt, new_validation_payload = _commit_attempt_through_validation(
                adapter,
                transaction,
                new_claim,
                sequence=414,
                tag="new-retry",
            )
            new_evidence_digest = str(new_validation_payload["evidence_digest"])
            new_prepared = _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": _claim_identity(new_claim),
                    "control_expected_revision": int(reset_task["revision"]) + 1,
                    "control_expected_status": "in_progress",
                    "evidence_digest": new_evidence_digest,
                    "body": {},
                    "now_ms": _NOW,
                },
                scope=str(new_claim["task_cid"]),
                sequence=414,
            )
            assert new_prepared["preparation_digest"] != old_prepared["preparation_digest"]
            assert transaction._connection.execute(  # noqa: SLF001
                "SELECT COUNT(*) FROM eaaef_completion_barrier_history "
                "WHERE preparation_digest=?",
                [old_prepared["preparation_digest"]],
            ).fetchone()[0] == 1
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="historical identity differs",
            ):
                _apply(
                    adapter,
                    transaction,
                    "coordination.expire_claim",
                    {
                        "claim": {
                            **_claim_identity(old_claim),
                            "owner_session_id": new_claim["owner_session_id"],
                        },
                        "now_ms": _NOW,
                    },
                    scope=_BOARD_SCOPE,
                    sequence=415,
                )
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="exact barrier",
            ):
                _apply(
                    adapter,
                    transaction,
                    "execution.commit_reconciled_attempt",
                    {
                        "attempt_id": new_claim["attempt_id"],
                        "expected_revision": new_validation_attempt["revision"],
                        "expected_status": "running",
                        "committed_phase": "failed",
                        "status": "failed",
                        "finished_at_ms": _NOW + 100,
                        "revision": new_validation_attempt["revision"] + 1,
                        "committed_at_ms": _NOW + 100,
                        "fencing_token": new_claim["fencing_token"],
                        "fence_epoch": new_claim["fence_epoch"],
                        "preparation": old_prepared,
                        "reconciliation": {
                            "operation": "coordination.abort_prepared_completion",
                            "task_cid": new_claim["task_cid"],
                            "claim_id": new_claim["claim_id"],
                            "attempt_id": new_claim["attempt_id"],
                            "status": "aborted",
                            "observed_at_ms": _NOW,
                            "lease_state": "expired",
                            "replayed": False,
                        },
                        "body": {},
                    },
                    scope=_BOARD_SCOPE,
                    sequence=416,
                )
            assert old_attempt["attempt_id"] != new_claim["attempt_id"]
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_container_dispatch_uses_its_closed_reservation_contract(tmp_path: Path) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=500)
            _mark_claimed_task_in_progress(
                adapter, transaction, claim, expected_revision=1, sequence=501
            )
            _ensure_attempt_record(adapter, transaction, claim, sequence=502)
            _apply(
                adapter,
                transaction,
                "execution.commit_phase",
                {
                    "attempt_id": claim["attempt_id"],
                    "expected_revision": 1,
                    "expected_status": "running",
                    "committed_phase": "context",
                    "status": "running",
                    "finished_at_ms": None,
                    "revision": 2,
                    "committed_at_ms": int(claim["claimed_at_ms"]) + 2,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                    "body": {},
                },
                scope=str(claim["task_cid"]),
                sequence=503,
            )
            dispatch_claim = _container_dispatch_claim(claim, tag="adapter")
            common = {
                "kind": "external_agent_container_dispatch",
                "record_id": dispatch_claim["claim_cid"],
                "attempt_id": claim["attempt_id"],
                "task_cid": claim["task_cid"],
                "operation_key": dispatch_claim["claim_cid"],
                "idempotency_key": dispatch_claim["idempotency_key"],
                "owner_session_id": claim["owner_session_id"],
                "recorded_at_ms": _NOW,
                "fencing_token": claim["fencing_token"],
                "fence_epoch": claim["fence_epoch"],
                "claim": dispatch_claim,
            }
            reserved = _apply(
                adapter,
                transaction,
                "effect.reserve",
                common,
                scope=str(claim["task_cid"]),
                sequence=504,
            )
            assert reserved["outcome"] == "reserved_new"
            ambiguous = _apply(
                adapter,
                transaction,
                "effect.reserve",
                common,
                scope=str(claim["task_cid"]),
                sequence=505,
            )
            assert ambiguous["outcome"] == "in_flight_ambiguous"
            result_body = {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-container-accepted-result@1",
                "interface": "ExternalAgentContainerWorkerDispatcher@1",
                "status": "succeeded",
                "accepted": True,
                "task_result_accepted": False,
                "merge_admitted": False,
                "task_cid": claim["task_cid"],
                "attempt_id": claim["attempt_id"],
                "packet_cid": dispatch_claim["packet_cid"],
                "claim_cid": dispatch_claim["claim_cid"],
                "reservation_id": reserved["reservation_id"],
                "proposal_receipt_cid": "sha256:" + "d" * 64,
                "verification_receipt_cid": "sha256:" + "e" * 64,
                "patch_artifact_cid": "sha256:" + "f" * 64,
                "artifact_cids": ["sha256:" + "1" * 64],
                "test_receipt_cids": ["sha256:" + "2" * 64],
                "proof_receipt_cids": ["sha256:" + "3" * 64],
                "worker_principal_did": dispatch_claim["worker_principal_did"],
                "independent_verifier_principal_did": "did:key:zverifier",
            }
            result = {
                **result_body,
                "receipt_id": "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        result_body,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
            assert _apply(
                adapter,
                transaction,
                "effect.commit",
                {
                    **common,
                    "reservation_id": reserved["reservation_id"],
                    "result": result,
                },
                scope=str(claim["task_cid"]),
                sequence=506,
            ) == result
            replay = _apply(
                adapter,
                transaction,
                "effect.reserve",
                common,
                scope=str(claim["task_cid"]),
                sequence=507,
            )
            assert replay["outcome"] == "accepted_replay"
            assert replay["accepted_result"] == result
            with pytest.raises(EAAEFBorrowedTransactionConflict):
                _apply(
                    adapter,
                    transaction,
                    "effect.reserve",
                    {
                        **common,
                        "claim": {
                            **dispatch_claim,
                            "fencing_token": int(claim["fencing_token"]) + 1,
                        },
                    },
                    scope=str(claim["task_cid"]),
                    sequence=508,
                )
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_phase_commits_require_exact_durable_execution_and_validation_evidence(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=550)
            _mark_claimed_task_in_progress(
                adapter, transaction, claim, expected_revision=1, sequence=551
            )
            _ensure_attempt_record(adapter, transaction, claim, sequence=552)
            context = _apply(
                adapter,
                transaction,
                "execution.commit_phase",
                {
                    "attempt_id": claim["attempt_id"],
                    "expected_revision": 1,
                    "expected_status": "running",
                    "committed_phase": "context",
                    "status": "running",
                    "finished_at_ms": None,
                    "revision": 2,
                    "committed_at_ms": int(claim["claimed_at_ms"]) + 2,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                    "body": {},
                },
                scope=str(claim["task_cid"]),
                sequence=553,
            )
            assert context["committed_phase"] == "context"
            dispatch_claim, accepted, container_effect = _commit_container_dispatch(
                adapter,
                transaction,
                claim,
                sequence=554,
                tag="phase-evidence",
            )

            def commit_attempt_phase(
                phase: str,
                *,
                expected_revision: int,
                body: Mapping[str, object],
                sequence: int,
            ) -> object:
                return _apply(
                    adapter,
                    transaction,
                    "execution.commit_phase",
                    {
                        "attempt_id": claim["attempt_id"],
                        "expected_revision": expected_revision,
                        "expected_status": "running",
                        "committed_phase": phase,
                        "status": "running",
                        "finished_at_ms": None,
                        "revision": expected_revision + 1,
                        "committed_at_ms": int(claim["claimed_at_ms"])
                        + expected_revision
                        + 2,
                        "fencing_token": claim["fencing_token"],
                        "fence_epoch": claim["fence_epoch"],
                        "body": dict(body),
                    },
                    scope=str(claim["task_cid"]),
                    sequence=sequence,
                )

            provider_key = "provider:phase-evidence"
            provider_body = {"idempotency_key": provider_key, "result": accepted}
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="exact committed reservation",
            ):
                commit_attempt_phase(
                    "provider",
                    expected_revision=2,
                    body=provider_body,
                    sequence=556,
                )
            provider_reservation = _apply(
                adapter,
                transaction,
                "provider.reserve",
                {
                    "kind": "provider",
                    "attempt_id": claim["attempt_id"],
                    "idempotency_key": provider_key,
                },
                scope=str(claim["task_cid"]),
                sequence=557,
            )
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="exact committed reservation",
            ):
                commit_attempt_phase(
                    "provider",
                    expected_revision=2,
                    body=provider_body,
                    sequence=558,
                )
            _apply(
                adapter,
                transaction,
                "provider.commit",
                {
                    "kind": "provider",
                    "record_id": provider_reservation["record_id"],
                    "attempt_id": claim["attempt_id"],
                    "task_cid": claim["task_cid"],
                    "operation_key": "",
                    "idempotency_key": provider_key,
                    "owner_session_id": claim["owner_session_id"],
                    "recorded_at_ms": _NOW,
                    "result": accepted,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                },
                scope=str(claim["task_cid"]),
                sequence=559,
            )
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="exact committed reservation",
            ):
                commit_attempt_phase(
                    "provider",
                    expected_revision=2,
                    body={
                        "idempotency_key": provider_key,
                        "result": {**accepted, "accepted": False},
                    },
                    sequence=560,
                )
            provider_attempt = commit_attempt_phase(
                "provider",
                expected_revision=2,
                body=provider_body,
                sequence=561,
            )
            assert provider_attempt["committed_phase"] == "provider"

            effect_key = "effect:phase-evidence"
            effect_body = {
                "idempotency_key": effect_key,
                "result": container_effect,
            }
            effect_reservation = _apply(
                adapter,
                transaction,
                "effect.reserve",
                {
                    "kind": "effect",
                    "attempt_id": claim["attempt_id"],
                    "idempotency_key": effect_key,
                },
                scope=str(claim["task_cid"]),
                sequence=562,
            )
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="exact committed reservation",
            ):
                commit_attempt_phase(
                    "effect",
                    expected_revision=3,
                    body=effect_body,
                    sequence=563,
                )
            _apply(
                adapter,
                transaction,
                "effect.commit",
                {
                    "kind": "effect",
                    "record_id": effect_reservation["record_id"],
                    "attempt_id": claim["attempt_id"],
                    "task_cid": claim["task_cid"],
                    "operation_key": container_effect["effect_key"],
                    "idempotency_key": effect_key,
                    "owner_session_id": claim["owner_session_id"],
                    "recorded_at_ms": _NOW,
                    "result": container_effect,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                },
                scope=str(claim["task_cid"]),
                sequence=564,
            )
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="exact committed reservation",
            ):
                commit_attempt_phase(
                    "effect",
                    expected_revision=3,
                    body={
                        "idempotency_key": effect_key,
                        "result": {**container_effect, "status": "failed"},
                    },
                    sequence=565,
                )
            effect_attempt = commit_attempt_phase(
                "effect",
                expected_revision=3,
                body=effect_body,
                sequence=566,
            )
            assert effect_attempt["committed_phase"] == "effect"

            validation = _validation_phase_payload(
                claim,
                tag="phase-evidence",
                accepted_result=accepted,
            )
            with pytest.raises(
                EAAEFBorrowedTransactionNotReady,
                match="created only by its atomic validation phase",
            ):
                _apply(
                    adapter,
                    transaction,
                    "validation.record",
                    {
                        "task_cid": claim["task_cid"],
                        "attempt_id": claim["attempt_id"],
                        **validation,
                    },
                    scope=str(claim["task_cid"]),
                    sequence=567,
                )

            def reseal_admission(
                payload: Mapping[str, object],
                **changes: object,
            ) -> dict[str, object]:
                changed = json.loads(json.dumps(payload))
                admission = dict(changed["body"]["admission_receipt"])
                admission.update(changes)
                admission_body = {
                    key: value
                    for key, value in admission.items()
                    if key != "receipt_cid"
                }
                admission["receipt_cid"] = "sha256:" + hashlib.sha256(
                    json.dumps(
                        admission_body,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                changed["body"]["admission_receipt"] = admission
                changed["evidence_digest"] = admission["receipt_cid"]
                return changed

            unrelated_claim = "sha256:" + "4" * 64
            forged = reseal_admission(validation, claim_cid=unrelated_claim)
            forged["body"]["dispatch_claim_cid"] = unrelated_claim
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="exact durable container execution",
            ):
                commit_attempt_phase(
                    "validation",
                    expected_revision=4,
                    body=forged,
                    sequence=568,
                )
            for sequence, reviewer in enumerate(
                (
                    dispatch_claim["worker_principal_did"],
                    dispatch_claim["provider_principal_did"],
                ),
                start=569,
            ):
                self_approved = reseal_admission(
                    validation,
                    reviewer_principal_did=reviewer,
                )
                with pytest.raises(
                    EAAEFBorrowedTransactionConflict,
                    match="exact durable container execution",
                ):
                    commit_attempt_phase(
                        "validation",
                        expected_revision=4,
                        body=self_approved,
                        sequence=sequence,
                    )

            validated = commit_attempt_phase(
                "validation",
                expected_revision=4,
                body=validation,
                sequence=571,
            )
            canonical_validation = dict(validated["body"])
            replay = _apply(
                adapter,
                transaction,
                "validation.record",
                {
                    "task_cid": claim["task_cid"],
                    "attempt_id": claim["attempt_id"],
                    "outcome": canonical_validation["outcome"],
                    "evidence_digest": canonical_validation["evidence_digest"],
                    "argv": canonical_validation["argv"],
                    "body": canonical_validation["body"],
                },
                scope=str(claim["task_cid"]),
                sequence=572,
            )
            assert replay["replayed"] is True
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_complete_cannot_skip_and_retryable_failure_requeues_atomically(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=580)
            _mark_claimed_task_in_progress(
                adapter, transaction, claim, expected_revision=1, sequence=581
            )
            attempt = _ensure_attempt_record(
                adapter, transaction, claim, sequence=582
            )
            terminal_ms = int(claim["claimed_at_ms"]) + 10
            with pytest.raises(
                EAAEFBorrowedTransactionConflict,
                match="skips a boundary",
            ):
                _apply(
                    adapter,
                    transaction,
                    "execution.commit_phase",
                    {
                        "attempt_id": claim["attempt_id"],
                        "expected_revision": attempt["revision"],
                        "expected_status": "running",
                        "committed_phase": "complete",
                        "status": "succeeded",
                        "finished_at_ms": terminal_ms,
                        "revision": attempt["revision"] + 1,
                        "committed_at_ms": terminal_ms,
                        "fencing_token": claim["fencing_token"],
                        "fence_epoch": claim["fence_epoch"],
                        "body": {},
                    },
                    scope=str(claim["task_cid"]),
                    sequence=583,
                )
            failed = _apply(
                adapter,
                transaction,
                "execution.commit_phase",
                {
                    "attempt_id": claim["attempt_id"],
                    "expected_revision": attempt["revision"],
                    "expected_status": "running",
                    "committed_phase": "failed",
                    "status": "failed",
                    "finished_at_ms": terminal_ms,
                    "revision": attempt["revision"] + 1,
                    "committed_at_ms": terminal_ms,
                    "fencing_token": claim["fencing_token"],
                    "fence_epoch": claim["fence_epoch"],
                    "body": {
                        "reason": "retryable portal transport failure",
                        "portal_retryable_failure": True,
                    },
                },
                scope=str(claim["task_cid"]),
                sequence=584,
            )
            assert failed["status"] == "failed"
            task = adapter._task_record(  # noqa: SLF001
                transaction._connection,  # noqa: SLF001
                str(claim["task_cid"]),
            )
            assert task is not None
            assert task["status"] == "ready"
            claim_row = adapter._claim_record(  # noqa: SLF001
                transaction._connection,  # noqa: SLF001
                str(claim["claim_id"]),
            )
            assert claim_row is not None
            assert claim_row["state"] == "expired"
            lease_state = transaction._connection.execute(  # noqa: SLF001
                "SELECT state, release_reason FROM leases WHERE task_cid=?",
                [claim["task_cid"]],
            ).fetchone()
            assert tuple(lease_state[index] for index in range(len(lease_state))) == (
                "expired",
                "retryable_portal_failure",
            )
            failure_event = transaction._connection.execute(  # noqa: SLF001
                "SELECT event_type, task_cid, attempt_id, body_json "
                "FROM domain_events WHERE event_type='attempt_phase_committed' "
                "AND attempt_id=?",
                [claim["attempt_id"]],
            ).fetchone()
            assert failure_event is not None
            assert tuple(failure_event[index] for index in range(3)) == (
                "attempt_phase_committed",
                claim["task_cid"],
                claim["attempt_id"],
            )
            assert json.loads(str(failure_event[3])) == {
                "phase": "failed",
                "portal_retryable_failure": True,
                "reason": "retryable portal transport failure",
                "revision": attempt["revision"] + 1,
            }
            replacement = _apply(
                adapter,
                transaction,
                "coordination.claim_ready",
                {
                    "owner_session_id": _bind_lane(
                        adapter,
                        transaction,
                        sequence=585,
                        lane_session_id="session:worker:retry",
                    ),
                    "lease_ms": 60_000,
                    "exclude_task_cids": [],
                    "now_ms": _NOW,
                },
                scope=_BOARD_SCOPE,
                sequence=586,
            )
            assert replacement["task_cid"] == claim["task_cid"]
            assert replacement["attempt_number"] > claim["attempt_number"]
            assert replacement["fencing_token"] > claim["fencing_token"]
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_canonical_reads_require_exact_board_or_task_lane_scope(tmp_path: Path) -> None:
    database = _database(tmp_path, count=2)
    handler = EAAEFBootstrapBorrowedTransactionOperationHandler(
        board_namespace=_BOARD,
        shard_id=_SHARD,
        owner_principal_did=_OWNER_PRINCIPAL,
        command_principal_did=_PRINCIPAL,
        owner_session_id=_OWNER_SESSION,
        owner_generation=_OWNER_GENERATION,
        fence_epoch=_FENCE_EPOCH,
        gateway_binding_cid=_GATEWAY_BINDING_CID,
        control_plane_schema_version=_CONTROL_PLANE_SCHEMA_VERSION,
        state_schema_revision=_STATE_SCHEMA_REVISION,
    )
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            board = transaction._connection.execute(  # noqa: SLF001
                "SELECT claim_cid, claimant_did, fencing_token, fence_epoch, "
                "expires_at_ms, state FROM leases WHERE task_cid=?",
                [_BOARD_SCOPE],
            ).fetchone()
            board_lease = {
                "lease_id": str(board[0]),
                "principal_did": str(board[1]),
                "fencing_token": int(board[2]),
                "fence_epoch": int(board[3]),
                "expires_at_ms": int(board[4]),
                "state": str(board[5]),
            }
            ready = handler.apply_authorized_daemon_operation(
                operation="task.ready",
                arguments={"limit": 2},
                transaction=transaction,
                command=_command("task.ready", scope=_BOARD_SCOPE, idempotency_key="read:ready"),
                lease=board_lease,
            )
            assert isinstance(ready["value"]["tasks"], list)

            claim = _claim(adapter, transaction, sequence=600)
            row = transaction._connection.execute(  # noqa: SLF001
                "SELECT claim_cid, claimant_did, fencing_token, fence_epoch, "
                "expires_at_ms, state, claim_id, attempt_id, attempt_number, "
                "owner_session_id FROM leases WHERE task_cid=?",
                [claim["task_cid"]],
            ).fetchone()
            task_lease = {
                "lease_id": str(row[0]),
                "principal_did": str(row[1]),
                "fencing_token": int(row[2]),
                "fence_epoch": int(row[3]),
                "expires_at_ms": int(row[4]),
                "state": str(row[5]),
            }
            lane_row = transaction._connection.execute(  # noqa: SLF001
                "SELECT metadata_json FROM daemon_sessions WHERE session_id=?",
                [claim["owner_session_id"]],
            ).fetchone()
            lane = json.loads(str(lane_row[0]))["lane_binding"]
            authority = {
                "schema": EAAEF_TASK_OPERATION_AUTHORITY_SCHEMA,
                "task_cid": claim["task_cid"],
                "claim_id": row[6],
                "attempt_id": row[7],
                "attempt_number": int(row[8]),
                "lease_id": row[0],
                "owner_session_id": row[9],
                "fencing_token": int(row[2]),
                "fence_epoch": int(row[3]),
                "daemon_lane_binding": lane,
            }
            got = handler.apply_authorized_daemon_operation(
                operation="task.get",
                arguments={
                    "task_cid": claim["task_cid"],
                    "task_authority_binding": authority,
                },
                transaction=transaction,
                command=_command(
                    "task.get",
                    scope=str(claim["task_cid"]),
                    idempotency_key="read:get",
                ),
                lease=task_lease,
            )
            assert got["value"]["task_cid"] == claim["task_cid"]
            with pytest.raises(Exception, match="target differs"):
                handler.apply_authorized_daemon_operation(
                    operation="task.get",
                    arguments={
                        "task_cid": "task:eaaef:2",
                        "task_authority_binding": authority,
                    },
                    transaction=transaction,
                    command=_command(
                        "task.get",
                        scope=str(claim["task_cid"]),
                        idempotency_key="read:cross-task",
                    ),
                    lease=task_lease,
                )
            with pytest.raises(Exception, match="board/shard claim lease"):
                handler.apply_authorized_daemon_operation(
                    operation="task.ready",
                    arguments={"limit": 2},
                    transaction=transaction,
                    command=_command(
                        "task.ready",
                        scope=str(claim["task_cid"]),
                        idempotency_key="read:wrong-scope",
                    ),
                    lease=task_lease,
                )
            with pytest.raises(Exception, match="task authority binding"):
                handler.apply_authorized_daemon_operation(
                    operation="task.get",
                    arguments={"task_cid": claim["task_cid"]},
                    transaction=transaction,
                    command=_command(
                        "task.get",
                        scope=_BOARD_SCOPE,
                        idempotency_key="read:board-bypass",
                    ),
                    lease=board_lease,
                )
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_dead_lane_recovery_precedes_higher_fence_reclaim(tmp_path: Path) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            old_claim = _claim(adapter, transaction, sequence=700)
            old_lane = str(old_claim["owner_session_id"])
            for table, key, identity in (
                ("leases", "task_cid", old_claim["task_cid"]),
                ("task_claims", "claim_id", old_claim["claim_id"]),
            ):
                transaction._connection.execute(  # noqa: SLF001
                    f"UPDATE {table} SET expires_at_ms=? "
                    f"WHERE {key}=?",
                    [_NOW - 1, identity],
                )

            def authority_rows() -> tuple[tuple[object, ...], tuple[object, ...]]:
                lease_row = transaction._connection.execute(  # noqa: SLF001
                    "SELECT state, revision, expires_at_ms, release_reason "
                    "FROM leases WHERE task_cid=?",
                    [old_claim["task_cid"]],
                ).fetchone()
                claim_row = transaction._connection.execute(  # noqa: SLF001
                    "SELECT state, revision, expires_at_ms, released_at_ms "
                    "FROM task_claims WHERE claim_id=?",
                    [old_claim["claim_id"]],
                ).fetchone()
                assert lease_row is not None and claim_row is not None
                return (
                    tuple(lease_row[index] for index in range(len(lease_row))),
                    tuple(claim_row[index] for index in range(len(claim_row))),
                )

            new_lane = _bind_lane(
                adapter,
                transaction,
                sequence=701,
                lane_session_id="session:worker:replacement",
            )
            assert _apply(
                adapter,
                transaction,
                "execution.list_running_attempts",
                {"owner_session_id": new_lane},
                scope=_BOARD_SCOPE,
                sequence=703,
            ) == []
            before_observe = authority_rows()
            recovered = _apply(
                adapter,
                transaction,
                "execution.list_running_attempts",
                {
                    "recovery_authority": {
                        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-dead-lane-recovery-authority@1",
                        "purpose": "expired_lane_retirement",
                        "lane_bindings": [_lane_binding(old_lane)],
                        "limit": 5,
                        "now_ms": _NOW,
                    }
                },
                scope=_BOARD_SCOPE,
                sequence=704,
            )
            assert len(recovered) == 1
            assert authority_rows() == before_observe
            assert before_observe[0][0] == "accepted"
            assert before_observe[1][0] == "accepted"
            snapshot = recovered[0]["eaaef_recovery_snapshot"]
            assert snapshot["preparation"] is None
            historical_claim = snapshot["claim"]
            expired = _apply(
                adapter,
                transaction,
                "coordination.expire_claim",
                {"claim": historical_claim, "now_ms": _NOW},
                scope=_BOARD_SCOPE,
                sequence=705,
            )
            assert expired["state"] == "expired"
            assert _apply(
                adapter,
                transaction,
                "coordination.claim_ready",
                {
                    "owner_session_id": new_lane,
                    "lease_ms": 60_000,
                    "exclude_task_cids": [],
                    "now_ms": _NOW,
                },
                scope=_BOARD_SCOPE,
                sequence=706,
            ) is None
            attempt = recovered[0]
            reconciliation = {
                "task_cid": old_claim["task_cid"],
                "claim_id": old_claim["claim_id"],
                "attempt_id": old_claim["attempt_id"],
                "status": "expired",
                "lease_state": "expired",
                "retry_required": True,
                "provider_evidence_reused": False,
                "effect_evidence_reused": False,
                "reason": "coordination_lease_expired_before_completion",
            }
            terminal_ms = max(_NOW + 1, int(attempt["started_at_ms"]) + 1)
            retired = _apply(
                adapter,
                transaction,
                "execution.commit_reconciled_attempt",
                {
                    "attempt_id": old_claim["attempt_id"],
                    "expected_revision": attempt["revision"],
                    "expected_status": "running",
                    "committed_phase": "failed",
                    "status": "failed",
                    "finished_at_ms": terminal_ms,
                    "revision": attempt["revision"] + 1,
                    "committed_at_ms": terminal_ms,
                    "fencing_token": old_claim["fencing_token"],
                    "fence_epoch": old_claim["fence_epoch"],
                    "preparation": historical_claim,
                    "reconciliation": reconciliation,
                    "body": {"crash_recovered": True},
                },
                scope=_BOARD_SCOPE,
                sequence=707,
            )
            assert retired["status"] == "failed"
            replacement = _apply(
                adapter,
                transaction,
                "coordination.claim_ready",
                {
                    "owner_session_id": new_lane,
                    "lease_ms": 60_000,
                    "exclude_task_cids": [],
                    "now_ms": _NOW,
                },
                scope=_BOARD_SCOPE,
                sequence=708,
            )
            assert replacement["task_cid"] == old_claim["task_cid"]
            assert replacement["attempt_number"] > old_claim["attempt_number"]
            assert replacement["fencing_token"] > old_claim["fencing_token"]
            assert replacement["owner_session_id"] == new_lane
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


def test_unsettled_completion_observation_does_not_expire_authority(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, count=1)
    adapter = _adapter()
    with open_duckdb_connection(database) as connection:
        transaction = StateTransaction(connection, store_id="eaaef-control").begin()
        try:
            claim = _claim(adapter, transaction, sequence=800)
            _mark_claimed_task_in_progress(
                adapter,
                transaction,
                claim,
                expected_revision=1,
                sequence=801,
            )
            _ensure_attempt_record(adapter, transaction, claim, sequence=802)
            _, validation = _commit_attempt_through_validation(
                adapter,
                transaction,
                claim,
                sequence=803,
                tag="pure-unsettled-observe",
            )
            _apply(
                adapter,
                transaction,
                "coordination.prepare_completion",
                {
                    "claim": _claim_identity(claim),
                    "control_expected_revision": 2,
                    "control_expected_status": "in_progress",
                    "evidence_digest": validation["evidence_digest"],
                    "body": {},
                    "now_ms": _NOW,
                },
                scope=str(claim["task_cid"]),
                sequence=804,
            )
            for table, key, identity in (
                ("leases", "task_cid", claim["task_cid"]),
                ("task_claims", "claim_id", claim["claim_id"]),
            ):
                transaction._connection.execute(  # noqa: SLF001
                    f"UPDATE {table} SET expires_at_ms=? WHERE {key}=?",
                    [_NOW - 1, identity],
                )

            def authority_rows() -> tuple[tuple[object, ...], ...]:
                rows = []
                for query, argument in (
                    (
                        "SELECT state, revision, expires_at_ms, release_reason "
                        "FROM leases WHERE task_cid=?",
                        claim["task_cid"],
                    ),
                    (
                        "SELECT state, revision, expires_at_ms, released_at_ms "
                        "FROM task_claims WHERE claim_id=?",
                        claim["claim_id"],
                    ),
                    (
                        "SELECT generation, revision, fence_epoch "
                        "FROM store_generations",
                        None,
                    ),
                ):
                    row = transaction._connection.execute(  # noqa: SLF001
                        query,
                        [] if argument is None else [argument],
                    ).fetchone()
                    assert row is not None
                    rows.append(tuple(row[index] for index in range(len(row))))
                return tuple(rows)

            before = authority_rows()
            snapshots = _apply(
                adapter,
                transaction,
                "coordination.list_unsettled_completions",
                {"limit": 10, "now_ms": _NOW},
                scope=_BOARD_SCOPE,
                sequence=805,
            )
            assert len(snapshots) == 1
            assert snapshots[0]["claim"]["state"] == "accepted"
            assert authority_rows() == before
            assert before[0][0] == "accepted"
            assert before[1][0] == "accepted"
            transaction.rollback()
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
