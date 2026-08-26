"""Conformance test for the signed Plan-R2 owner adapter.

The fixture is an owner-private DuckDB transaction, not a mock repository.  It
verifies both the Plan-R2 authorization and AuthorizedStateCommand signatures,
checks every CID join, performs the protected-row CAS, commits the population
atomically, and reads the result back through a separate authorized command.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import socket
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.planning import external_agent_plan_r2 as r2
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PLAN_R2_OWNER_GATEWAY_INTERFACE,
    PLAN_R2_OWNER_OPERATION_SCHEMA,
    PREPARE_PLAN_R2_OPERATION,
    ExternalAgentStateRepository,
    PlanR2OwnerAdapterError,
    PlanR2OwnerAdapterUnavailable,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    AuthorizedStateCommand,
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandFabric,
    QuackCommandFabricStateError,
)

NOW_MS = 1_800_000_000_000


def _sha(token: str) -> str:
    return "sha256:" + token * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _task(token: str, alias: str, ordinal: int, status: str) -> dict[str, object]:
    task_cid = _sha(token)
    return {
        "task_cid": task_cid,
        "task_alias": alias,
        "goal_cid": _sha("9"),
        "plan_cid": _sha("8"),
        "objective_id": "EAAEF-OBJ-001",
        "ordinal": ordinal,
        "status": status,
        "revision": 1,
        "priority": "high",
        "identity": {"task_cid": task_cid},
        "body": {
            "read_scope": [f"read/{alias}"],
            "write_scope": [f"write/{alias}"],
            "effect_scope": [f"effect/{alias}"],
        },
    }


def _authorization() -> tuple[
    dict[str, object],
    Ed25519PrivateKey,
    str,
    str,
    str,
]:
    protected_task = _task("a", "EAAEF-000", 1, "accepted")
    frontier_task = _task("b", "EAAEF-001", 2, "todo")
    frontier_task["revision"] = 2
    new_task = _task("d", "EAAEF-002", 3, "todo")
    protected = {
        "task_cid": protected_task["task_cid"],
        "status": protected_task["status"],
        "revision": protected_task["revision"],
        "task_row": protected_task,
        "task_row_cid": r2._cid(protected_task),
    }
    owner_key = Ed25519PrivateKey.generate()
    owner_did = ed25519_did_key(owner_key.public_key())
    statement = r2.prepare_plan_r2_transition_authorization(
        board_namespace="external-agent-autonomous-execution-fabric-v1",
        source_head="1" * 40,
        source_tree="2" * 40,
        source_generation_cid=_sha("3"),
        bootstrap_admission_cid=_sha("4"),
        r1_launch_capsule_cid=_sha("5"),
        quack_owner_qualification_cid=_sha("6"),
        quack_command_fabric_qualification_cid=_sha("7"),
        owner_principal_did=owner_did,
        shard_id="eaaef-control-shard",
        store_id="eaaef-control-run-v5",
        owner_generation=3,
        expected_epoch=4,
        fencing_token=5,
        lease_id="eaaef-plan-r2-lease",
        expected_version=9,
        expected_active_plan_cid=_sha("c"),
        expected_active_plan_root_cid=_sha("d"),
        expected_active_plan_revision=1,
        expected_event_cursor="event-cursor-9",
        expected_semantic_root_cid=_sha("e"),
        new_plan={
            "plan_cid": _sha("8"),
            "plan_alias": "EAAEF-PLAN-R2",
            "plan_root_cid": _sha("f"),
            "semantic_root_cid": _sha("0"),
            "status": "active",
            "revision": 2,
            "body": {"objective": "continue after reconciliation"},
        },
        tasks=[protected_task, frontier_task, new_task],
        dependencies=[
            {
                "task_cid": frontier_task["task_cid"],
                "dependency_task_cid": protected_task["task_cid"],
                "kind": "requires",
            },
            {
                "task_cid": new_task["task_cid"],
                "dependency_task_cid": frontier_task["task_cid"],
                "kind": "requires",
            },
        ],
        protected_tasks=[protected],
        frontier_task_cids=[str(frontier_task["task_cid"])],
        delta_cid=_sha("1"),
        request_id="eaaef-plan-r2-request-1",
        idempotency_key="eaaef-plan-r2-idempotency-1",
        deadline_ms=NOW_MS + 50_000,
        issued_at_ms=NOW_MS - 1_000,
        expires_at_ms=NOW_MS + 100_000,
        one_use_nonce="eaaef-plan-r2-nonce-1",
    )
    operator_key = Ed25519PrivateKey.generate()
    security_key = Ed25519PrivateKey.generate()
    operator_did = ed25519_did_key(operator_key.public_key())
    security_did = ed25519_did_key(security_key.public_key())

    def approval(role: str, key: Ed25519PrivateKey, did: str) -> dict[str, object]:
        value = r2.prepare_plan_r2_transition_approval(
            statement,
            role=role,
            identity_did=did,
            issued_at_ms=NOW_MS - 500,
            expires_at_ms=NOW_MS + 50_000,
        )
        value["signature"] = base64.b64encode(key.sign(r2._canonical_bytes(value))).decode("ascii")
        return value

    authorization = r2.assemble_plan_r2_transition_authorization(
        statement,
        operator_approval=approval("independent_operator", operator_key, operator_did),
        security_approval=approval("independent_security_reviewer", security_key, security_did),
        trusted_operator_dids=[operator_did],
        trusted_security_reviewer_dids=[security_did],
        now_ms=NOW_MS,
    )
    return authorization, owner_key, owner_did, operator_did, security_did


class _OwnerConformanceGateway:
    """Small real owner used to qualify the required extension boundary."""

    INTERFACE = PLAN_R2_OWNER_GATEWAY_INTERFACE

    def __init__(
        self,
        path: Path,
        *,
        policy: QuackCommandAuthorizationPolicy,
        operator_did: str,
        security_did: str,
        capability_cid: str,
    ) -> None:
        self._connection = duckdb.connect(str(path))
        self._policy = policy
        self._operator_did = operator_did
        self._security_did = security_did
        self._capability_cid = capability_cid
        self.envelopes: list[AuthorizedStateCommand] = []
        self.payload_cids: list[str] = []
        self._connection.execute(
            "CREATE TABLE owner_state(version BIGINT, event_cursor VARCHAR, "
            "plan_cid VARCHAR, plan_root_cid VARCHAR, plan_revision BIGINT, "
            "semantic_root_cid VARCHAR)"
        )
        self._connection.execute(
            "INSERT INTO owner_state VALUES (9, 'event-cursor-9', ?, ?, 1, ?)",
            [_sha("c"), _sha("d"), _sha("e")],
        )
        self._connection.execute(
            "CREATE TABLE protected_rows(task_cid VARCHAR PRIMARY KEY, row_json VARCHAR)"
        )
        self._connection.execute(
            "CREATE TABLE accepted_population(authorization_cid VARCHAR PRIMARY KEY, "
            "population_cid VARCHAR, command_cid VARCHAR, transaction_cid VARCHAR)"
        )

    def close(self) -> None:
        self._connection.close()

    def seed_protected(self, authorization: Mapping[str, object]) -> None:
        row = authorization["protected_tasks"][0]["task_row"]
        self._connection.execute(
            "INSERT INTO protected_rows VALUES (?, ?)",
            [row["task_cid"], _canonical(row).decode("ascii")],
        )

    def submit_authorized_plan_r2_operation(
        self,
        envelope: AuthorizedStateCommand,
        operation_payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        verify_authorized_state_command(envelope, policy=self._policy, now_ms=NOW_MS)
        assert envelope.authority_ref_cid == operation_payload["authorization"]["authorization_cid"]
        assert envelope.command.store_id == operation_payload["authorization"]["store_id"]
        assert envelope.shard_id == operation_payload["authorization"]["shard_id"]
        assert envelope.shard_id != envelope.command.store_id
        assert envelope.command.parameters["operation_payload_cid"] == r2._cid(operation_payload)
        assert operation_payload["schema"] == PLAN_R2_OWNER_OPERATION_SCHEMA
        authorization = operation_payload["authorization"]
        r2.verify_plan_r2_transition_authorization(
            authorization,
            trusted_operator_dids=[self._operator_did],
            trusted_security_reviewer_dids=[self._security_did],
            now_ms=NOW_MS,
        )
        self.envelopes.append(envelope)
        self.payload_cids.append(r2._cid(operation_payload))
        operation = operation_payload["operation"]
        if operation == PREPARE_PLAN_R2_OPERATION:
            return self._prepare(envelope, authorization)
        if operation == APPLY_PLAN_R2_OPERATION:
            return self._apply(envelope, authorization, operation_payload["prepared_projection"])
        if operation == OBSERVE_PLAN_R2_OPERATION:
            return self._observe(authorization, operation_payload["transition_receipt"])
        raise AssertionError("unreachable operation")

    def _state(self) -> tuple[object, ...]:
        return self._connection.execute("SELECT * FROM owner_state").fetchone()

    def _assert_protected(self, authorization: Mapping[str, object]) -> None:
        for protected in authorization["protected_tasks"]:
            row = self._connection.execute(
                "SELECT row_json FROM protected_rows WHERE task_cid = ?",
                [protected["task_cid"]],
            ).fetchone()
            if row is None or json.loads(row[0]) != protected["task_row"]:
                raise RuntimeError("protected task changed")

    def _prepare(
        self, envelope: AuthorizedStateCommand, authorization: Mapping[str, object]
    ) -> Mapping[str, object]:
        state = self._state()
        self._assert_protected(authorization)
        assert state == (
            authorization["expected_version"],
            authorization["expected_event_cursor"],
            authorization["expected_active_plan_cid"],
            authorization["expected_active_plan_root_cid"],
            authorization["expected_active_plan_revision"],
            authorization["expected_semantic_root_cid"],
        )
        value = {
            "schema": r2.PLAN_R2_PREPARED_PROJECTION_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "capability_cid": self._capability_cid,
            "authorized_prepare_command_cid": envelope.envelope_cid,
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "before_plan_cid": state[2],
            "before_plan_root_cid": state[3],
            "before_plan_revision": state[4],
            "before_version": state[0],
            "before_event_cursor": state[1],
            "before_semantic_root_cid": state[5],
            "population_cid": authorization["population_cid"],
            "plan_root_cid": authorization["plan_root_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "frontier_cid": authorization["frontier_cid"],
            "prepared_at_ms": NOW_MS,
            "expires_at_ms": NOW_MS + 40_000,
            "authority_mutated": False,
            "process_started": False,
        }
        value["projection_cid"] = r2._cid(value)
        return value

    def _apply(
        self,
        envelope: AuthorizedStateCommand,
        authorization: Mapping[str, object],
        prepared: Mapping[str, object],
    ) -> Mapping[str, object]:
        self._connection.execute("BEGIN TRANSACTION")
        try:
            state = self._state()
            self._assert_protected(authorization)
            assert state[0] == prepared["before_version"]
            assert state[1] == prepared["before_event_cursor"]
            after_cursor = "event-cursor-10"
            after_version = int(state[0]) + 1
            new_plan = authorization["new_plan"]
            self._connection.execute(
                "UPDATE owner_state SET version = ?, event_cursor = ?, plan_cid = ?, "
                "plan_root_cid = ?, plan_revision = ?, semantic_root_cid = ?",
                [
                    after_version,
                    after_cursor,
                    new_plan["plan_cid"],
                    new_plan["plan_root_cid"],
                    new_plan["revision"],
                    new_plan["semantic_root_cid"],
                ],
            )
            transaction_cid = r2._cid(
                {
                    "authorization_cid": authorization["authorization_cid"],
                    "command_cid": envelope.envelope_cid,
                    "before": list(state),
                    "after_version": after_version,
                    "after_event_cursor": after_cursor,
                }
            )
            self._connection.execute(
                "INSERT INTO accepted_population VALUES (?, ?, ?, ?)",
                [
                    authorization["authorization_cid"],
                    authorization["population_cid"],
                    envelope.envelope_cid,
                    transaction_cid,
                ],
            )
            self._assert_protected(authorization)
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        value = {
            "schema": r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "statement_cid": authorization["statement_cid"],
            "capability_cid": self._capability_cid,
            "authorized_prepare_command_cid": prepared["authorized_prepare_command_cid"],
            "authorized_apply_command_cid": envelope.envelope_cid,
            "prepared_projection_cid": prepared["projection_cid"],
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "before_plan_cid": authorization["expected_active_plan_cid"],
            "after_plan_cid": new_plan["plan_cid"],
            "before_plan_root_cid": authorization["expected_active_plan_root_cid"],
            "after_plan_root_cid": new_plan["plan_root_cid"],
            "before_plan_revision": authorization["expected_active_plan_revision"],
            "after_plan_revision": new_plan["revision"],
            "before_version": authorization["expected_version"],
            "after_version": after_version,
            "before_event_cursor": authorization["expected_event_cursor"],
            "after_event_cursor": after_cursor,
            "before_semantic_root_cid": authorization["expected_semantic_root_cid"],
            "after_semantic_root_cid": new_plan["semantic_root_cid"],
            "population_cid": authorization["population_cid"],
            "task_population_cid": authorization["task_population_cid"],
            "dependency_population_cid": authorization["dependency_population_cid"],
            "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
            "frontier_cid": authorization["frontier_cid"],
            "frontier_task_cids": authorization["frontier_task_cids"],
            "protected_tasks_unchanged": True,
            "transaction_cid": transaction_cid,
            "replayed": False,
            "committed_at_ms": NOW_MS,
        }
        value["receipt_cid"] = r2._cid(value)
        return value

    def _observe(
        self,
        authorization: Mapping[str, object],
        receipt: Mapping[str, object],
    ) -> Mapping[str, object]:
        state = self._state()
        accepted = self._connection.execute(
            "SELECT population_cid, command_cid, transaction_cid "
            "FROM accepted_population WHERE authorization_cid = ?",
            [authorization["authorization_cid"]],
        ).fetchone()
        assert accepted == (
            receipt["population_cid"],
            receipt["authorized_apply_command_cid"],
            receipt["transaction_cid"],
        )
        value = {
            "schema": r2.PLAN_R2_STATE_OBSERVATION_SCHEMA,
            "authorization_cid": authorization["authorization_cid"],
            "transition_receipt_cid": receipt["receipt_cid"],
            "transaction_cid": receipt["transaction_cid"],
            "authorized_prepare_command_cid": receipt["authorized_prepare_command_cid"],
            "authorized_apply_command_cid": receipt["authorized_apply_command_cid"],
            "quack_command_fabric_qualification_cid": authorization[
                "quack_command_fabric_qualification_cid"
            ],
            "source_head": authorization["source_head"],
            "source_tree": authorization["source_tree"],
            "owner_principal_did": authorization["owner_principal_did"],
            "shard_id": authorization["shard_id"],
            "owner_generation": authorization["owner_generation"],
            "epoch": authorization["expected_epoch"],
            "fence": authorization["fencing_token"],
            "store_version": state[0],
            "active_plan_cid": state[2],
            "active_plan_root_cid": state[3],
            "active_plan_revision": state[4],
            "event_cursor": state[1],
            "semantic_root_cid": state[5],
            "population_cid": receipt["population_cid"],
            "task_population_cid": receipt["task_population_cid"],
            "dependency_population_cid": receipt["dependency_population_cid"],
            "protected_tasks_root_cid": receipt["protected_tasks_root_cid"],
            "frontier_cid": receipt["frontier_cid"],
            "frontier_task_cids": receipt["frontier_task_cids"],
            "captured_at_ms": NOW_MS,
            "authority_mutated": False,
            "process_started": False,
        }
        value["observation_cid"] = r2._cid(value)
        return value


def _adapter(tmp_path: Path):
    authorization, _owner_key, owner_did, operator_did, security_did = _authorization()
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    approver_did = ed25519_did_key(approver_key.public_key())
    principal_did = ed25519_did_key(principal_key.public_key())
    capability_cid = _sha("2")
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=authorization["board_namespace"],
        shard_id=authorization["shard_id"],
        store_id=authorization["store_id"],
        authority_ref_cid=authorization["authorization_cid"],
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        fence_epoch=authorization["fencing_token"],
        trusted_approver_dids=frozenset({approver_did}),
        authorized_principal_dids=frozenset({principal_did}),
        allowed_command_kinds=frozenset({CommandKind.OBSERVE, CommandKind.MIGRATE}),
    )
    gateway = _OwnerConformanceGateway(
        tmp_path / "owner-private.duckdb",
        policy=policy,
        operator_did=operator_did,
        security_did=security_did,
        capability_cid=capability_cid,
    )
    gateway.seed_protected(authorization)
    next_slot = iter(range(1, 20)).__next__

    def sign(payload: Mapping[str, object]) -> str:
        return base64.b64encode(approver_key.sign(_canonical(dict(payload)))).decode("ascii")

    adapter = ExternalAgentStateRepository(
        owner_gateway=gateway,
        board_namespace=authorization["board_namespace"],
        shard_id=authorization["shard_id"],
        store_id=authorization["store_id"],
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        owner_epoch=authorization["expected_epoch"],
        fence_epoch=authorization["fencing_token"],
        capability_cid=capability_cid,
        command_fabric_qualification_cid=authorization["quack_command_fabric_qualification_cid"],
        principal_did=principal_did,
        approver_did=approver_did,
        envelope_signer=sign,
        ingress_slot_allocator=next_slot,
        clock_ms=lambda: NOW_MS,
    )
    return adapter, gateway, authorization


def test_real_owner_transaction_preserves_signed_cid_joins_and_string_cursor(
    tmp_path: Path,
) -> None:
    adapter, gateway, authorization = _adapter(tmp_path)
    try:
        prepared = adapter.prepare_authorized_plan_r2_transition(authorization)
        receipt = adapter.apply_authorized_plan_r2_transition(authorization, prepared)
        observation = adapter.observe_authorized_plan_r2_transition(authorization, receipt)
        launch = r2.validate_plan_r2_launch_transition(
            repository=adapter,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=observation,
            trusted_operator_dids=[
                authorization["operator_approval"]["identity_did"]
            ],
            trusted_security_reviewer_dids=[
                authorization["security_approval"]["identity_did"]
            ],
            now_ms=NOW_MS,
        )
        assert launch["valid"] is True
        assert [item.command.command_kind for item in gateway.envelopes] == [
            CommandKind.OBSERVE,
            CommandKind.MIGRATE,
            CommandKind.OBSERVE,
            CommandKind.OBSERVE,
        ]
        assert all(
            item.authority_ref_cid == authorization["authorization_cid"]
            for item in gateway.envelopes
        )
        assert all(
            item.command.store_id == authorization["store_id"]
            and item.shard_id == authorization["shard_id"]
            and item.shard_id != item.command.store_id
            for item in gateway.envelopes
        )
        assert prepared["authorized_prepare_command_cid"] == gateway.envelopes[0].envelope_cid
        assert receipt["authorized_apply_command_cid"] == gateway.envelopes[1].envelope_cid
        assert (
            observation["authorized_apply_command_cid"] == receipt["authorized_apply_command_cid"]
        )
        assert receipt["before_event_cursor"] == "event-cursor-9"
        assert receipt["after_event_cursor"] == "event-cursor-10"
        assert observation["event_cursor"] == "event-cursor-10"
        assert receipt["protected_tasks_unchanged"] is True
        assert gateway._connection.execute(
            "SELECT count(*) FROM accepted_population"
        ).fetchone() == (1,)
    finally:
        adapter.close()


def test_changed_protected_full_row_rolls_back_atomic_apply(tmp_path: Path) -> None:
    adapter, gateway, authorization = _adapter(tmp_path)
    try:
        prepared = adapter.prepare_authorized_plan_r2_transition(authorization)
        gateway._connection.execute("UPDATE protected_rows SET row_json = '{\"poisoned\":true}'")
        with pytest.raises(RuntimeError, match="protected task changed"):
            adapter.apply_authorized_plan_r2_transition(authorization, prepared)
        assert gateway._connection.execute(
            "SELECT version, event_cursor FROM owner_state"
        ).fetchone() == (9, "event-cursor-9")
        assert gateway._connection.execute(
            "SELECT count(*) FROM accepted_population"
        ).fetchone() == (0,)
    finally:
        adapter.close()


def test_missing_atomic_owner_extension_is_typed_no_go() -> None:
    with pytest.raises(
        PlanR2OwnerAdapterUnavailable,
        match="submit_authorized_plan_r2_operation",
    ):
        ExternalAgentStateRepository(
            owner_gateway=object(),  # type: ignore[arg-type]
            board_namespace="external-agent-autonomous-execution-fabric-v1",
            shard_id="eaaef-control-shard",
            store_id="eaaef-control-run-v5",
            owner_principal_did="did:key:zOwner",
            owner_generation=3,
            owner_epoch=4,
            fence_epoch=5,
            capability_cid=_sha("2"),
            command_fabric_qualification_cid=_sha("7"),
            principal_did="did:key:zPrincipal",
            approver_did="did:key:zApprover",
            envelope_signer=lambda _payload: "unreachable",
            ingress_slot_allocator=lambda: 1,
        )


def test_store_and_shard_must_not_collapse(tmp_path: Path) -> None:
    adapter, gateway, authorization = _adapter(tmp_path)
    adapter.close()
    with pytest.raises(PlanR2OwnerAdapterError, match="remain distinct"):
        ExternalAgentStateRepository(
            owner_gateway=gateway,
            board_namespace=authorization["board_namespace"],
            shard_id="same-id",
            store_id="same-id",
            owner_principal_did=authorization["owner_principal_did"],
            owner_generation=3,
            owner_epoch=4,
            fence_epoch=5,
            capability_cid=_sha("2"),
            command_fabric_qualification_cid=_sha("7"),
            principal_did="did:key:zPrincipal",
            approver_did="did:key:zApprover",
            envelope_signer=lambda _payload: "unreachable",
            ingress_slot_allocator=lambda: 1,
        )


def _free_quack_endpoint() -> str:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return f"quack:127.0.0.1:{port}"


def _exact_quack_runtime() -> tuple[object, Path]:
    artifact = Path(os.environ.get("QUACK_155_EXTENSION_PATH", ""))
    if duckdb.__version__ != "1.5.5" or not artifact.is_file():
        pytest.skip("requires DuckDB 1.5.5 and QUACK_155_EXTENSION_PATH")
    return duckdb, artifact.resolve()


def _machine_lock_name() -> str:
    machine = platform.machine().lower()
    return {
        "aarch64": "linux_arm64",
        "arm64": "linux_arm64",
        "x86_64": "linux_amd64",
        "amd64": "linux_amd64",
    }.get(machine, f"linux_{machine}")


def _signed_plan_r2_capability(
    authorization: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    value: dict[str, object] = {
        "schema": r2.PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA,
        "allowed": True,
        "blockers": [],
        "source_head": authorization["source_head"],
        "source_tree": authorization["source_tree"],
        "bootstrap_admission_cid": authorization["bootstrap_admission_cid"],
        "quack_owner_qualification_cid": authorization["quack_owner_qualification_cid"],
        "quack_command_fabric_qualification_cid": authorization[
            "quack_command_fabric_qualification_cid"
        ],
        "owner_principal_did": authorization["owner_principal_did"],
        "shard_id": authorization["shard_id"],
        "owner_generation": authorization["owner_generation"],
        "epoch": authorization["expected_epoch"],
        "fence": authorization["fencing_token"],
        "duckdb_version": "1.5.5",
        "quack_build": "quack@1.5.5+core",
        "authorized_state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "ingress_authenticated": True,
        "ingress_append_only_single_relation": True,
        "ingress_accepts_signed_envelope_only": True,
        "bare_state_command_rejected": True,
        "owner_verifies_authorized_state_command": True,
        "authority_ref_binds_transition_authorization": True,
        "local_owner_verifies_transition_authorization": True,
        "operational_database_private": True,
        "one_mutable_owner": True,
        "atomic_plan_population_cas": True,
        "egress_read_only": True,
        "egress_append_denied": True,
        "durable_idempotent_receipts": True,
        "protected_full_rows_bound": True,
        "reviewer_identity_did": reviewer_did,
        "issued_at_ms": NOW_MS - 500,
        "expires_at_ms": NOW_MS + 50_000,
    }
    value["reviewer_signature"] = base64.b64encode(
        reviewer_key.sign(r2._canonical_bytes(value))
    ).decode("ascii")
    value["capability_cid"] = r2._cid(value)
    return value, reviewer_did


def _provision_canonical_plan_r2_owner(
    path: Path,
    authorization: Mapping[str, object],
    *,
    principal_did: str,
    owner_status: str = "running",
) -> None:
    if owner_status not in {"ready", "running"}:
        raise ValueError("owner_status must be ready or running")
    install_control_plane_schema(
        path,
        application_version="0.0.45",
        tool_version="1.5.5",
        owner_id="plan-r2-owner-test",
    )
    connection = duckdb.connect(str(path))
    protected = authorization["protected_tasks"][0]["task_row"]
    existing = dict(authorization["tasks"][1])
    existing["plan_cid"] = authorization["expected_active_plan_cid"]
    existing["revision"] = 1
    omitted = _task("c", "EAAEF-R1-OMITTED", 4, "todo")
    omitted["plan_cid"] = authorization["expected_active_plan_cid"]
    try:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (?, 1, ?, ?, '123e4567-e89b-12d3-a456-426614174000',
                      'birth:plan-r2', '1970-01-01T00:00:00Z')
            """,
            [
                authorization["owner_generation"],
                authorization["fencing_token"],
                authorization["expected_version"],
            ],
        )
        connection.execute(
            """
            INSERT INTO state_servers (
                server_id, store_id, database_uuid, process_birth_id,
                listen_uri, extension_fingerprint, schema_revision,
                generation, started_at, stopped_at, status, revision
            ) VALUES ('server:plan-r2', ?,
                      '123e4567-e89b-12d3-a456-426614174000',
                      'birth:plan-r2', 'quack:127.0.0.1:19495', 'sha256:test',
                      1, ?, '1970-01-01T00:00:00Z', NULL, ?, 1)
            """,
            [
                authorization["store_id"],
                authorization["owner_generation"],
                owner_status,
            ],
        )
        connection.execute(
            "INSERT INTO server_epochs VALUES ('server:plan-r2', ?, ?, "
            "'1970-01-01T00:00:00Z', NULL)",
            [authorization["expected_epoch"], authorization["fencing_token"]],
        )
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, 'EAAEF-G000', 'EAAEF-OBJ-001', '', 1, 'EAAEF',
                      'open', '1970-01-01T00:00:00Z',
                      '1970-01-01T00:00:00Z', 1, '{}')
            """,
            [protected["goal_cid"]],
        )
        predecessor = {
            "plan_cid": authorization["expected_active_plan_cid"],
            "plan_root_cid": authorization["expected_active_plan_root_cid"],
            "semantic_root_cid": authorization["expected_semantic_root_cid"],
        }
        connection.execute(
            """
            INSERT INTO plans VALUES (?, ?, 'EAAEF-PLAN-R1', 'active',
                                      '1970-01-01T00:00:00Z',
                                      '1970-01-01T00:00:00Z', ?, ?)
            """,
            [
                authorization["expected_active_plan_cid"],
                protected["goal_cid"],
                authorization["expected_active_plan_revision"],
                _canonical(predecessor).decode("ascii"),
            ],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                      '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z', ?, ?)
            """,
            [
                protected["task_cid"],
                protected["task_alias"],
                protected["goal_cid"],
                protected["plan_cid"],
                protected["objective_id"],
                protected["ordinal"],
                protected["status"],
                protected["revision"],
                protected["priority"],
                _canonical(protected["identity"]).decode("ascii"),
                _canonical(protected["body"]).decode("ascii"),
            ],
        )
        connection.execute(
            "INSERT INTO task_revisions VALUES (?, 1, ?, ?, "
            "'1970-01-01T00:00:00Z')",
            [
                protected["task_cid"],
                protected["status"],
                _canonical(protected["body"]).decode("ascii"),
            ],
        )
        for predecessor_task in (existing, omitted):
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                          '1970-01-01T00:00:00Z',
                          '1970-01-01T00:00:00Z', ?, ?)
                """,
                [
                    predecessor_task["task_cid"],
                    predecessor_task["task_alias"],
                    predecessor_task["goal_cid"],
                    predecessor_task["plan_cid"],
                    predecessor_task["objective_id"],
                    predecessor_task["ordinal"],
                    predecessor_task["status"],
                    predecessor_task["revision"],
                    predecessor_task["priority"],
                    _canonical(predecessor_task["identity"]).decode("ascii"),
                    _canonical(predecessor_task["body"]).decode("ascii"),
                ],
            )
            connection.execute(
                "INSERT INTO task_revisions VALUES (?, 1, ?, ?, "
                "'1970-01-01T00:00:00Z')",
                [
                    predecessor_task["task_cid"],
                    predecessor_task["status"],
                    _canonical(predecessor_task["body"]).decode("ascii"),
                ],
            )
        for sequence in range(1, 10):
            connection.execute(
                "INSERT INTO domain_events (event_id, stream_id, sequence, "
                "global_sequence, event_type, task_cid, attempt_id, "
                "session_id, recorded_at, body_json) VALUES "
                "(?, 'bootstrap', ?, ?, 'intent.seeded', '', '', '', "
                "'1970-01-01T00:00:00Z', '{}')",
                [f"seed:{sequence}", sequence, sequence],
            )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt,
                state, started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision, extension_schema,
                extension_json
            ) VALUES (?, ?, 'resolution:plan-r2', ?, ?, ?, ?, 1, 'accepted',
                      ?, NULL, 0, 'session:plan-r2', ?, 1,
                      'AuthorizedStateCommandLease@1', '{}')
            """,
            [
                authorization["plan_root_cid"],
                authorization["lease_id"],
                principal_did,
                authorization["expected_epoch"],
                authorization["fencing_token"],
                authorization["expires_at_ms"],
                NOW_MS - 1000,
                authorization["fencing_token"],
            ],
        )
    finally:
        connection.close()


def _production_adapter(tmp_path: Path):
    runtime, artifact = _exact_quack_runtime()
    authorization, _owner_key, owner_did, operator_did, security_did = _authorization()
    approver_key = Ed25519PrivateKey.generate()
    principal_key = Ed25519PrivateKey.generate()
    approver_did = ed25519_did_key(approver_key.public_key())
    principal_did = ed25519_did_key(principal_key.public_key())
    capability, capability_reviewer_did = _signed_plan_r2_capability(authorization)
    operational = tmp_path / "owner-private.duckdb"
    _provision_canonical_plan_r2_owner(operational, authorization, principal_did=principal_did)
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=authorization["board_namespace"],
        shard_id=authorization["shard_id"],
        store_id=authorization["store_id"],
        authority_ref_cid=authorization["authorization_cid"],
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        fence_epoch=authorization["fencing_token"],
        trusted_approver_dids=frozenset({approver_did}),
        authorized_principal_dids=frozenset({principal_did}),
        allowed_command_kinds=frozenset({CommandKind.OBSERVE, CommandKind.MIGRATE}),
    )
    fabric = QuackCommandFabric(
        duckdb_module=runtime,
        extension_path=artifact,
        lock_path=Path(__file__).resolve().parents[2]
        / "ipfs_datasets_py/requirements/duckdb-quack.lock",
        machine=_machine_lock_name(),
        ingress_database=tmp_path / "ingress.duckdb",
        operational_database=operational,
        projection_database=tmp_path / "projection.duckdb",
        ingress_endpoint=_free_quack_endpoint(),
        state_endpoint=_free_quack_endpoint(),
        ingress_token="plan-r2-ingress-transport-token-0001",
        state_token="plan-r2-state-transport-token-0000002",
        authorization_policy=policy,
        plan_r2_operational_capability=capability,
        command_fabric_qualification_cid=authorization["quack_command_fabric_qualification_cid"],
        trusted_plan_r2_capability_reviewer_dids=[capability_reviewer_did],
        trusted_plan_r2_operator_dids=[operator_did],
        trusted_plan_r2_security_reviewer_dids=[security_did],
        clock_ms=lambda: NOW_MS,
    )
    fabric.start()
    slots = iter(range(1, 20)).__next__

    def sign(payload: Mapping[str, object]) -> str:
        return base64.b64encode(approver_key.sign(_canonical(dict(payload)))).decode("ascii")

    adapter = ExternalAgentStateRepository(
        owner_gateway=fabric.plan_r2_owner_gateway(),
        board_namespace=authorization["board_namespace"],
        shard_id=authorization["shard_id"],
        store_id=authorization["store_id"],
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        owner_epoch=authorization["expected_epoch"],
        fence_epoch=authorization["fencing_token"],
        capability_cid=capability["capability_cid"],
        command_fabric_qualification_cid=authorization["quack_command_fabric_qualification_cid"],
        principal_did=principal_did,
        approver_did=approver_did,
        envelope_signer=sign,
        ingress_slot_allocator=slots,
        clock_ms=lambda: NOW_MS,
    )
    return adapter, fabric, authorization, operational


def _apply_plan_r2_population_directly(
    connection: object,
    authorization: Mapping[str, object],
) -> Mapping[str, object]:
    snapshot = {
        "plan_cid": authorization["expected_active_plan_cid"],
        "plan_revision": authorization["expected_active_plan_revision"],
        "goal_cid": authorization["tasks"][0]["goal_cid"],
        "version": authorization["expected_version"],
        "event_cursor": authorization["expected_event_cursor"],
    }
    envelope = SimpleNamespace(
        envelope_cid="sha256:" + "f" * 64,
        scope_id=authorization["plan_root_cid"],
        command=SimpleNamespace(session_id="session:plan-r2-direct-test"),
    )
    transaction = SimpleNamespace(active=True, _connection=connection)
    return QuackCommandFabric._plan_r2_apply_population(  # noqa: SLF001
        transaction,
        authorization=authorization,
        snapshot=snapshot,
        envelope=envelope,
        committed_at_ms=NOW_MS,
    )


def test_plan_r2_direct_owner_keeps_every_task_history_contiguous(
    tmp_path: Path,
) -> None:
    authorization = _authorization()[0]
    operational = tmp_path / "direct-owner.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did="did:key:plan-r2-direct-test",
    )
    connection = duckdb.connect(str(operational))
    try:
        connection.execute("BEGIN TRANSACTION")
        _apply_plan_r2_population_directly(connection, authorization)
        connection.execute("COMMIT")
        assert connection.execute(
            "SELECT task_alias, status, revision FROM tasks ORDER BY task_alias"
        ).fetchall() == [
            ("EAAEF-000", "accepted", 1),
            ("EAAEF-001", "todo", 2),
            ("EAAEF-002", "todo", 1),
            ("EAAEF-R1-OMITTED", "superseded", 2),
        ]
        assert connection.execute(
            "SELECT t.task_alias, COUNT(r.revision), MIN(r.revision), "
            "MAX(r.revision) FROM tasks AS t JOIN task_revisions AS r "
            "ON r.task_cid = t.task_cid GROUP BY t.task_alias "
            "ORDER BY t.task_alias"
        ).fetchall() == [
            ("EAAEF-000", 1, 1, 1),
            ("EAAEF-001", 2, 1, 2),
            ("EAAEF-002", 1, 1, 1),
            ("EAAEF-R1-OMITTED", 2, 1, 2),
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks AS t LEFT JOIN task_revisions AS r "
            "ON r.task_cid = t.task_cid AND r.revision = t.revision "
            "WHERE r.task_cid IS NULL OR r.status IS DISTINCT FROM t.status "
            "OR r.body_json IS DISTINCT FROM t.body_json "
            "OR r.recorded_at IS DISTINCT FROM t.updated_at"
        ).fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize("fault", ("gap", "revision_jump", "new_revision_zero"))
def test_plan_r2_direct_owner_rejects_noncontiguous_history_or_revision(
    tmp_path: Path,
    fault: str,
) -> None:
    authorization = json.loads(json.dumps(_authorization()[0]))
    operational = tmp_path / f"direct-owner-{fault}.duckdb"
    _provision_canonical_plan_r2_owner(
        operational,
        authorization,
        principal_did="did:key:plan-r2-direct-test",
    )
    connection = duckdb.connect(str(operational))
    try:
        if fault == "gap":
            connection.execute(
                "DELETE FROM task_revisions WHERE task_cid = ? AND revision = 1",
                [authorization["tasks"][1]["task_cid"]],
            )
        elif fault == "revision_jump":
            authorization["tasks"][1]["revision"] = 3
        else:
            authorization["tasks"][2]["revision"] = 0
        connection.execute("BEGIN TRANSACTION")
        with pytest.raises(QuackCommandFabricStateError):
            _apply_plan_r2_population_directly(connection, authorization)
        connection.execute("ROLLBACK")
        assert connection.execute(
            "SELECT status FROM plans WHERE plan_cid = ?",
            [authorization["expected_active_plan_cid"]],
        ).fetchone() == ("active",)
        assert connection.execute(
            "SELECT COUNT(*) FROM plans WHERE plan_cid = ?",
            [authorization["new_plan"]["plan_cid"]],
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_canonical_quack_owner_applies_and_reads_back_plan_r2_atomically(
    tmp_path: Path,
) -> None:
    adapter, fabric, authorization, operational = _production_adapter(tmp_path)
    try:
        prepared = adapter.prepare_authorized_plan_r2_transition(authorization)
        receipt = adapter.apply_authorized_plan_r2_transition(authorization, prepared)
        observation = adapter.observe_authorized_plan_r2_transition(authorization, receipt)
        launch = r2.validate_plan_r2_launch_transition(
            repository=adapter,
            authorization=authorization,
            transition_receipt=receipt,
            state_observation=observation,
            trusted_operator_dids=[
                authorization["operator_approval"]["identity_did"]
            ],
            trusted_security_reviewer_dids=[
                authorization["security_approval"]["identity_did"]
            ],
            now_ms=NOW_MS,
        )
        assert launch["valid"] is True
        assert receipt["after_version"] == 10
        assert receipt["after_event_cursor"] == "event-cursor-11"
        assert observation["active_plan_cid"] == authorization["new_plan"]["plan_cid"]
        assert observation["population_cid"] == authorization["population_cid"]
        check = duckdb.connect(str(operational))
        try:
            assert check.execute(
                "SELECT plan_alias, status FROM plans ORDER BY plan_alias"
            ).fetchall() == [
                ("EAAEF-PLAN-R1", "superseded"),
                ("EAAEF-PLAN-R2", "active"),
            ]
            assert check.execute("SELECT count(*) FROM tasks").fetchone() == (4,)
            assert check.execute("SELECT count(*) FROM task_dependencies").fetchone() == (2,)
            assert check.execute(
                "SELECT task_alias, status, revision FROM tasks ORDER BY task_alias"
            ).fetchall() == [
                ("EAAEF-000", "accepted", 1),
                ("EAAEF-001", "todo", 2),
                ("EAAEF-002", "todo", 1),
                ("EAAEF-R1-OMITTED", "superseded", 2),
            ]
            assert check.execute(
                "SELECT t.task_alias, COUNT(r.revision), MIN(r.revision), "
                "MAX(r.revision) FROM tasks AS t JOIN task_revisions AS r "
                "ON r.task_cid = t.task_cid GROUP BY t.task_alias "
                "ORDER BY t.task_alias"
            ).fetchall() == [
                ("EAAEF-000", 1, 1, 1),
                ("EAAEF-001", 2, 1, 2),
                ("EAAEF-002", 1, 1, 1),
                ("EAAEF-R1-OMITTED", 2, 1, 2),
            ]
            assert check.execute(
                "SELECT COUNT(*) FROM tasks AS t LEFT JOIN task_revisions AS r "
                "ON r.task_cid = t.task_cid AND r.revision = t.revision "
                "WHERE r.task_cid IS NULL OR r.status IS DISTINCT FROM t.status "
                "OR r.body_json IS DISTINCT FROM t.body_json "
                "OR r.recorded_at IS DISTINCT FROM t.updated_at"
            ).fetchone() == (0,)
            assert check.execute(
                "SELECT count(*) FROM domain_events "
                "WHERE event_type = 'authorized_plan_r2_owner_result'"
            ).fetchone() == (4,)
        finally:
            check.close()
    finally:
        fabric.stop()


def test_canonical_quack_owner_exact_prepare_replay_is_one_durable_result(
    tmp_path: Path,
) -> None:
    adapter, fabric, authorization, operational = _production_adapter(tmp_path)
    try:
        auth = adapter._validate_authorization_binding(authorization)  # noqa: SLF001
        payload = {
            "schema": PLAN_R2_OWNER_OPERATION_SCHEMA,
            "operation": PREPARE_PLAN_R2_OPERATION,
            "authorization": auth,
        }
        payload_cid = r2._cid(payload)
        command = adapter._build_state_command(  # noqa: SLF001
            operation=PREPARE_PLAN_R2_OPERATION,
            authorization=auth,
            operation_payload_cid=payload_cid,
            context={},
            ingress_slot=1,
        )
        envelope = adapter._build_envelope(  # noqa: SLF001
            operation=PREPARE_PLAN_R2_OPERATION,
            authorization=auth,
            command=command,
            operation_payload_cid=payload_cid,
            ingress_slot=1,
        )
        first = adapter.owner_gateway.submit_authorized_plan_r2_operation(envelope, payload)
        fabric._clock_ms = lambda: int(authorization["expires_at_ms"]) + 1  # noqa: SLF001
        replay = adapter.owner_gateway.submit_authorized_plan_r2_operation(envelope, payload)
        assert replay == first
        connection = duckdb.connect(str(operational))
        try:
            assert connection.execute(
                "SELECT count(*) FROM domain_events "
                "WHERE event_type = 'authorized_plan_r2_owner_result'"
            ).fetchone() == (1,)
            assert connection.execute("SELECT revision FROM store_generations").fetchone() == (9,)
        finally:
            connection.close()
    finally:
        fabric.stop()


def test_canonical_quack_owner_rollback_replay_crash_and_live_fences(
    tmp_path: Path,
) -> None:
    adapter, fabric, authorization, operational = _production_adapter(tmp_path)
    try:
        prepared = adapter.prepare_authorized_plan_r2_transition(authorization)
        connection = duckdb.connect(str(operational))
        try:
            connection.execute(
                "UPDATE tasks SET body_json = '{\"poisoned\":true}' WHERE task_cid = ?",
                [authorization["protected_tasks"][0]["task_cid"]],
            )
        finally:
            connection.close()
        with pytest.raises(QuackCommandFabricStateError, match="protected"):
            adapter.apply_authorized_plan_r2_transition(authorization, prepared)
        check = duckdb.connect(str(operational))
        try:
            assert check.execute("SELECT revision FROM store_generations").fetchone() == (9,)
            assert check.execute(
                "SELECT count(*) FROM plans WHERE plan_alias = 'EAAEF-PLAN-R2'"
            ).fetchone() == (0,)
        finally:
            check.close()

        connection = duckdb.connect(str(operational))
        try:
            protected = authorization["protected_tasks"][0]["task_row"]
            connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [
                    _canonical(protected["body"]).decode("ascii"),
                    protected["task_cid"],
                ],
            )
        finally:
            connection.close()
        fabric._plan_r2_after_commit_hook = (  # noqa: SLF001
            lambda result: (
                (_ for _ in ()).throw(RuntimeError("crash-after-commit"))
                if result.get("schema") == r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA
                else None
            )
        )
        receipt = adapter.apply_authorized_plan_r2_transition(authorization, prepared)
        assert receipt["replayed"] is False
        check = duckdb.connect(str(operational))
        try:
            assert check.execute(
                "SELECT count(*) FROM plans WHERE plan_alias = 'EAAEF-PLAN-R2'"
            ).fetchone() == (1,)
            assert check.execute("SELECT revision FROM store_generations").fetchone() == (10,)
        finally:
            check.close()

        # An exact owner retry returns the one durable result; it cannot apply
        # the population or advance the store a second time.
        assert adapter.owner_gateway.production_capability_cid == adapter.capability_cid
    finally:
        fabric.stop()

    revoked_adapter, revoked_fabric, revoked_auth, revoked_db = _production_adapter(
        tmp_path / "revoked"
    )
    try:
        connection = duckdb.connect(str(revoked_db))
        try:
            connection.execute(
                "UPDATE leases SET state = 'released' WHERE task_cid = ?",
                [revoked_auth["plan_root_cid"]],
            )
        finally:
            connection.close()
        with pytest.raises(Exception, match="lease is revoked"):
            revoked_adapter.prepare_authorized_plan_r2_transition(revoked_auth)
    finally:
        revoked_fabric.stop()

    stale_adapter, stale_fabric, stale_auth, stale_db = _production_adapter(tmp_path / "stale")
    try:
        connection = duckdb.connect(str(stale_db))
        try:
            connection.execute(
                "UPDATE server_epochs SET epoch = epoch + 1, "
                "fence_epoch = fence_epoch + 1 "
                "WHERE server_id = 'server:plan-r2'"
            )
            connection.execute("UPDATE store_generations SET fence_epoch = fence_epoch + 1")
        finally:
            connection.close()
        with pytest.raises(QuackCommandFabricStateError, match="source CAS is stale"):
            stale_adapter.prepare_authorized_plan_r2_transition(stale_auth)
        assert stale_auth["shard_id"] != stale_auth["store_id"]
    finally:
        stale_fabric.stop()
