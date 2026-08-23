from __future__ import annotations

import base64
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.planning import external_agent_plan_r2 as r2
from ipfs_accelerate_py.agent_supervisor.runtime.plan_r2_remote_owner import (
    PlanR2ProcessRemoteOwnerGateway,
    PlanR2RemoteOwnerError,
    PlanR2RemoteOwnerService,
    PlanR2RemoteReplayDiverged,
    PlanR2RemoteResponseUnavailable,
    bind_plan_r2_process_remote_owner_gateway,
    bind_plan_r2_remote_exact_envelope_journal,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.external_agent_state_repository import (
    APPLY_PLAN_R2_OPERATION,
    OBSERVE_PLAN_R2_OPERATION,
    PLAN_R2_OWNER_OPERATION_SCHEMA,
    PREPARE_PLAN_R2_OPERATION,
    ExternalAgentStateRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionStore,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandFabric,
    QuackPlanR2OwnerGateway,
)
from ipfs_accelerate_py.agent_supervisor.validation.plan_r2_remote_owner_admission import (
    PLAN_R2_REMOTE_OPERATIONS,
    PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
    PlanR2RemoteOwnerAdmissionError,
    VerifiedPlanR2RemoteOwnerAdmission,
    plan_r2_remote_owner_capability_signing_payload,
    seal_plan_r2_remote_owner_capability,
    verify_plan_r2_remote_owner_admission,
)


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


def _signed_authority(now_ms: int) -> dict[str, object]:
    completed = _task("a", "EAAEF-000", 1, "accepted")
    frontier = _task("b", "EAAEF-001", 2, "todo")
    protected = {
        "task_cid": completed["task_cid"],
        "status": completed["status"],
        "revision": completed["revision"],
        "task_row": completed,
        "task_row_cid": r2._cid(completed),
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
        tasks=[completed, frontier],
        dependencies=[
            {
                "task_cid": frontier["task_cid"],
                "dependency_task_cid": completed["task_cid"],
                "kind": "requires",
            }
        ],
        protected_tasks=[protected],
        frontier_task_cids=[str(frontier["task_cid"])],
        delta_cid=_sha("1"),
        request_id="eaaef-plan-r2-request-remote",
        idempotency_key="eaaef-plan-r2-idempotency-remote",
        deadline_ms=now_ms + 80_000,
        issued_at_ms=now_ms - 1_000,
        expires_at_ms=now_ms + 120_000,
        one_use_nonce="eaaef-plan-r2-nonce-remote",
    )
    operator_key = Ed25519PrivateKey.generate()
    security_key = Ed25519PrivateKey.generate()
    operator_did = ed25519_did_key(operator_key.public_key())
    security_did = ed25519_did_key(security_key.public_key())

    def approval(role: str, key: Ed25519PrivateKey, identity: str) -> dict[str, object]:
        value = r2.prepare_plan_r2_transition_approval(
            statement,
            role=role,
            identity_did=identity,
            issued_at_ms=now_ms - 500,
            expires_at_ms=now_ms + 100_000,
        )
        value["signature"] = base64.b64encode(key.sign(r2._canonical_bytes(value))).decode("ascii")
        return value

    authorization = r2.assemble_plan_r2_transition_authorization(
        statement,
        operator_approval=approval("independent_operator", operator_key, operator_did),
        security_approval=approval("independent_security_reviewer", security_key, security_did),
        trusted_operator_dids=[operator_did],
        trusted_security_reviewer_dids=[security_did],
        now_ms=now_ms,
    )
    return {
        "authorization": authorization,
        "owner_did": owner_did,
        "operator_did": operator_did,
        "operator_key": operator_key,
        "security_did": security_did,
        "security_key": security_key,
    }


def _signed_plan_capability(
    authorization: Mapping[str, object], now_ms: int
) -> tuple[Mapping[str, object], str]:
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
        "issued_at_ms": now_ms - 500,
        "expires_at_ms": now_ms + 100_000,
    }
    value["reviewer_signature"] = base64.b64encode(
        reviewer_key.sign(r2._canonical_bytes(value))
    ).decode("ascii")
    value["capability_cid"] = r2._cid(value)
    return value, reviewer_did


def _admitted_authority(now_ms: int) -> dict[str, object]:
    authority = _signed_authority(now_ms)
    authorization = authority["authorization"]
    assert isinstance(authorization, Mapping)
    plan_capability, plan_reviewer_did = _signed_plan_capability(authorization, now_ms)
    principal_key = Ed25519PrivateKey.generate()
    approver_key = Ed25519PrivateKey.generate()
    remote_reviewer_key = Ed25519PrivateKey.generate()
    principal_did = ed25519_did_key(principal_key.public_key())
    approver_did = ed25519_did_key(approver_key.public_key())
    remote_reviewer_did = ed25519_did_key(remote_reviewer_key.public_key())
    statement = plan_r2_remote_owner_capability_signing_payload(
        source_head=str(authorization["source_head"]),
        source_tree=str(authorization["source_tree"]),
        board_namespace=str(authorization["board_namespace"]),
        plan_root_cid=str(authorization["plan_root_cid"]),
        population_cid=str(authorization["population_cid"]),
        plan_r2_authorization_cid=str(authorization["authorization_cid"]),
        plan_r2_operational_capability_cid=str(plan_capability["capability_cid"]),
        quack_command_fabric_qualification_cid=str(
            authorization["quack_command_fabric_qualification_cid"]
        ),
        owner_principal_did=str(authorization["owner_principal_did"]),
        shard_id=str(authorization["shard_id"]),
        store_id=str(authorization["store_id"]),
        owner_generation=int(authorization["owner_generation"]),
        epoch=int(authorization["expected_epoch"]),
        fence=int(authorization["fencing_token"]),
        authorized_principal_did=principal_did,
        independent_approver_did=approver_did,
        request_channel_id="plan-r2-request-channel-1",
        response_channel_id="plan-r2-response-channel-1",
        reviewer_did=remote_reviewer_did,
        issued_at_ms=now_ms - 100,
        expires_at_ms=now_ms + 90_000,
        issuance_nonce="plan-r2-remote-capability-nonce-1",
    )
    remote_capability = seal_plan_r2_remote_owner_capability(
        statement,
        reviewer_signature=base64.b64encode(
            remote_reviewer_key.sign(_canonical(dict(statement)))
        ).decode("ascii"),
    )
    admission = verify_plan_r2_remote_owner_admission(
        remote_capability,
        plan_r2_operational_capability=plan_capability,
        authorization=authorization,
        trusted_remote_reviewer_dids=[remote_reviewer_did],
        trusted_plan_r2_capability_reviewer_dids=[plan_reviewer_did],
        trusted_operator_dids=[authority["operator_did"]],
        trusted_security_reviewer_dids=[authority["security_did"]],
        now_ms=now_ms,
    )
    return {
        **authority,
        "plan_capability": plan_capability,
        "plan_reviewer_did": plan_reviewer_did,
        "principal_did": principal_did,
        "approver_did": approver_did,
        "approver_key": approver_key,
        "remote_reviewer_did": remote_reviewer_did,
        "remote_capability": remote_capability,
        "admission": admission,
    }


class _OwnerFabric:
    def __init__(
        self,
        *,
        plan_capability: Mapping[str, object],
        policy: QuackCommandAuthorizationPolicy,
        now_ms: int,
    ) -> None:
        self._plan_capability = dict(plan_capability)
        self._policy = policy
        self._now_ms = now_ms
        self._durable: dict[str, tuple[dict[str, object], Mapping[str, object]]] = {}
        self._prepared: Mapping[str, object] | None = None
        self._receipt: Mapping[str, object] | None = None
        self.effect_count = 0

    @property
    def plan_r2_production_capability_cid(self) -> str:
        return str(self._plan_capability["capability_cid"])

    def _require_plan_r2_capability(self) -> Mapping[str, object]:
        return self._plan_capability

    def _submit_authorized_plan_r2_operation(
        self, envelope, operation_payload
    ) -> Mapping[str, object]:
        verify_authorized_state_command(envelope, policy=self._policy, now_ms=self._now_ms)
        prior = self._durable.get(envelope.envelope_cid)
        if prior is not None:
            if prior[0] != dict(operation_payload):
                raise RuntimeError("divergent exact owner replay")
            return prior[1]
        assert operation_payload["schema"] == PLAN_R2_OWNER_OPERATION_SCHEMA
        operation = operation_payload["operation"]
        authorization = operation_payload["authorization"]
        if operation == PREPARE_PLAN_R2_OPERATION:
            result = QuackCommandFabric._prepare_plan_r2_result(
                envelope=envelope,
                authorization=authorization,
                capability=self._plan_capability,
                snapshot={
                    "epoch": authorization["expected_epoch"],
                    "fence": authorization["fencing_token"],
                    "plan_cid": authorization["expected_active_plan_cid"],
                    "plan_root_cid": authorization["expected_active_plan_root_cid"],
                    "plan_revision": authorization["expected_active_plan_revision"],
                    "version": authorization["expected_version"],
                    "event_cursor": authorization["expected_event_cursor"],
                    "semantic_root_cid": authorization["expected_semantic_root_cid"],
                },
                now_ms=self._now_ms,
            )
            self._prepared = result
        elif operation == APPLY_PLAN_R2_OPERATION:
            prepared = operation_payload["prepared_projection"]
            assert prepared == self._prepared
            new_plan = authorization["new_plan"]
            result = {
                "schema": r2.PLAN_R2_TRANSITION_RECEIPT_SCHEMA,
                "authorization_cid": authorization["authorization_cid"],
                "statement_cid": authorization["statement_cid"],
                "capability_cid": self._plan_capability["capability_cid"],
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
                "after_version": int(authorization["expected_version"]) + 1,
                "before_event_cursor": authorization["expected_event_cursor"],
                "after_event_cursor": "event-cursor-10",
                "before_semantic_root_cid": authorization["expected_semantic_root_cid"],
                "after_semantic_root_cid": new_plan["semantic_root_cid"],
                "population_cid": authorization["population_cid"],
                "task_population_cid": authorization["task_population_cid"],
                "dependency_population_cid": authorization["dependency_population_cid"],
                "protected_tasks_root_cid": authorization["protected_tasks_root_cid"],
                "frontier_cid": authorization["frontier_cid"],
                "frontier_task_cids": authorization["frontier_task_cids"],
                "protected_tasks_unchanged": True,
                "transaction_cid": r2._cid(
                    {
                        "authorization_cid": authorization["authorization_cid"],
                        "envelope_cid": envelope.envelope_cid,
                    }
                ),
                "replayed": False,
                "committed_at_ms": self._now_ms,
            }
            result["receipt_cid"] = r2._cid(result)
            self._receipt = result
        elif operation == OBSERVE_PLAN_R2_OPERATION:
            receipt = operation_payload["transition_receipt"]
            assert receipt == self._receipt
            new_plan = authorization["new_plan"]
            result = {
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
                "store_version": receipt["after_version"],
                "active_plan_cid": new_plan["plan_cid"],
                "active_plan_root_cid": new_plan["plan_root_cid"],
                "active_plan_revision": new_plan["revision"],
                "event_cursor": receipt["after_event_cursor"],
                "semantic_root_cid": new_plan["semantic_root_cid"],
                "population_cid": receipt["population_cid"],
                "task_population_cid": receipt["task_population_cid"],
                "dependency_population_cid": receipt["dependency_population_cid"],
                "protected_tasks_root_cid": receipt["protected_tasks_root_cid"],
                "frontier_cid": receipt["frontier_cid"],
                "frontier_task_cids": receipt["frontier_task_cids"],
                "captured_at_ms": self._now_ms,
                "authority_mutated": False,
                "process_started": False,
            }
            result["observation_cid"] = r2._cid(result)
        else:  # pragma: no cover - the service rejects this first
            raise AssertionError("operation escaped the Plan-R2 vocabulary")
        self.effect_count += 1
        self._durable[envelope.envelope_cid] = (
            dict(operation_payload),
            result,
        )
        return result


class _WireChannel:
    INTERFACE = PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE

    def __init__(
        self,
        service: PlanR2RemoteOwnerService,
        *,
        lose_first_response: bool,
    ) -> None:
        self.request_channel_id = "plan-r2-request-channel-1"
        self.response_channel_id = "plan-r2-response-channel-1"
        self._service = service
        self._lose_first_response = lose_first_response
        self._attached = False
        self.requests: list[bytes] = []

    def attach(self) -> None:
        self._attached = True

    def exchange(
        self,
        request_bytes: bytes,
        *,
        request_cid: str,
        maximum_wait_ms: int,
    ) -> bytes:
        assert self._attached
        assert request_cid == json.loads(request_bytes)["request_cid"]
        assert 0 < maximum_wait_ms <= 60_000
        self.requests.append(request_bytes)
        response = self._service.handle_exchange(request_bytes)
        if self._lose_first_response:
            self._lose_first_response = False
            raise TimeoutError("simulated response loss")
        return response

    def close(self) -> None:
        self._attached = False


def _runtime(tmp_path: Path, authority: Mapping[str, object], now_ms: int):
    authorization = authority["authorization"]
    assert isinstance(authorization, Mapping)
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=str(authorization["board_namespace"]),
        shard_id=str(authorization["shard_id"]),
        store_id=str(authorization["store_id"]),
        authority_ref_cid=str(authorization["authorization_cid"]),
        owner_principal_did=str(authority["owner_did"]),
        owner_generation=int(authorization["owner_generation"]),
        fence_epoch=int(authorization["fencing_token"]),
        trusted_approver_dids=frozenset({str(authority["approver_did"])}),
        authorized_principal_dids=frozenset({str(authority["principal_did"])}),
        allowed_command_kinds=frozenset({CommandKind.OBSERVE, CommandKind.MIGRATE}),
    )
    fabric = _OwnerFabric(
        plan_capability=authority["plan_capability"],
        policy=policy,
        now_ms=now_ms,
    )
    owner_gateway = QuackPlanR2OwnerGateway(fabric)  # type: ignore[arg-type]
    service = PlanR2RemoteOwnerService(
        admission=authority["admission"],
        owner_gateway=owner_gateway,
    )
    store = PlanRevisionStore(tmp_path / "plan-r2-remote-journal")
    return fabric, service, store


def _adapter(
    *,
    authority: Mapping[str, object],
    gateway: PlanR2ProcessRemoteOwnerGateway,
    slot_allocator,
    now_ms: int,
) -> ExternalAgentStateRepository:
    authorization = authority["authorization"]
    assert isinstance(authorization, Mapping)
    approver_key = authority["approver_key"]
    assert isinstance(approver_key, Ed25519PrivateKey)

    def sign(payload: Mapping[str, object]) -> str:
        return base64.b64encode(approver_key.sign(_canonical(dict(payload)))).decode("ascii")

    return ExternalAgentStateRepository(
        owner_gateway=gateway,
        board_namespace=str(authorization["board_namespace"]),
        shard_id=str(authorization["shard_id"]),
        store_id=str(authorization["store_id"]),
        owner_principal_did=str(authority["owner_did"]),
        owner_generation=int(authorization["owner_generation"]),
        owner_epoch=int(authorization["expected_epoch"]),
        fence_epoch=int(authorization["fencing_token"]),
        capability_cid=str(authority["plan_capability"]["capability_cid"]),
        command_fabric_qualification_cid=str(
            authorization["quack_command_fabric_qualification_cid"]
        ),
        principal_did=str(authority["principal_did"]),
        approver_did=str(authority["approver_did"]),
        envelope_signer=sign,
        ingress_slot_allocator=slot_allocator,
        clock_ms=lambda: now_ms,
    )


def test_remote_admission_is_signed_and_mutually_exclusive() -> None:
    now_ms = time.time_ns() // 1_000_000
    authority = _admitted_authority(now_ms)
    admission = authority["admission"]
    assert list(admission["allowed_operations"]) == list(PLAN_R2_REMOTE_OPERATIONS)
    assert admission["r1_operations_allowed"] is False
    assert admission["merge_operations_allowed"] is False
    assert admission["generic_state_command_allowed"] is False
    assert admission["process_birth_allowed"] is False
    assert admission["database_authority_crossing_allowed"] is False
    assert admission["transport_token_authority_crossing_allowed"] is False

    with pytest.raises(TypeError, match="come from the signature verifier"):
        VerifiedPlanR2RemoteOwnerAdmission(
            object(),
            dict(admission),
        )

    colliding_statement = {
        key: value
        for key, value in authority["remote_capability"].items()
        if key not in {"reviewer_signature", "capability_cid"}
    }
    colliding_statement["reviewer_did"] = authority["operator_did"]
    colliding_capability = seal_plan_r2_remote_owner_capability(
        colliding_statement,
        reviewer_signature=base64.b64encode(
            authority["operator_key"].sign(_canonical(colliding_statement))
        ).decode("ascii"),
    )
    with pytest.raises(
        PlanR2RemoteOwnerAdmissionError,
        match="reviewers and command principals are not independent",
    ):
        verify_plan_r2_remote_owner_admission(
            colliding_capability,
            plan_r2_operational_capability=authority["plan_capability"],
            authorization=authority["authorization"],
            trusted_remote_reviewer_dids=[authority["operator_did"]],
            trusted_plan_r2_capability_reviewer_dids=[
                authority["plan_reviewer_did"]
            ],
            trusted_operator_dids=[authority["operator_did"]],
            trusted_security_reviewer_dids=[authority["security_did"]],
            now_ms=now_ms,
        )

    forged = dict(authority["remote_capability"])
    forged["r1_operations_allowed"] = True
    with pytest.raises(
        PlanR2RemoteOwnerAdmissionError,
        match="authority isolation",
    ):
        verify_plan_r2_remote_owner_admission(
            forged,
            plan_r2_operational_capability=authority["plan_capability"],
            authorization=authority["authorization"],
            trusted_remote_reviewer_dids=[authority["remote_reviewer_did"]],
            trusted_plan_r2_capability_reviewer_dids=[authority["plan_reviewer_did"]],
            trusted_operator_dids=[authority["operator_did"]],
            trusted_security_reviewer_dids=[authority["security_did"]],
            now_ms=now_ms,
        )


def test_prepare_response_loss_restart_adopts_exact_envelope(
    tmp_path: Path,
) -> None:
    now_ms = time.time_ns() // 1_000_000
    authority = _admitted_authority(now_ms)
    fabric, service, store = _runtime(tmp_path, authority, now_ms)
    first_channel = _WireChannel(service, lose_first_response=True)
    first_gateway = bind_plan_r2_process_remote_owner_gateway(
        admission=authority["admission"],
        channel=first_channel,
        journal=bind_plan_r2_remote_exact_envelope_journal(
            store=store,
            admission=authority["admission"],
        ),
    )
    first_adapter = _adapter(
        authority=authority,
        gateway=first_gateway,
        slot_allocator=iter([7]).__next__,
        now_ms=now_ms,
    )
    first_adapter.attach()
    with pytest.raises(
        PlanR2RemoteResponseUnavailable,
        match="retry the exact envelope",
    ):
        first_adapter.prepare_authorized_plan_r2_transition(authority["authorization"])
    assert fabric.effect_count == 1
    original_request = first_channel.requests[0]
    original_envelope_cid = json.loads(original_request)["envelope"]["envelope_cid"]
    first_adapter.close()

    allocation_count = 0

    def must_not_allocate() -> int:
        nonlocal allocation_count
        allocation_count += 1
        return 99

    restarted_store = PlanRevisionStore(tmp_path / "plan-r2-remote-journal")
    restarted_channel = _WireChannel(service, lose_first_response=False)
    restarted_gateway = bind_plan_r2_process_remote_owner_gateway(
        admission=authority["admission"],
        channel=restarted_channel,
        journal=bind_plan_r2_remote_exact_envelope_journal(
            store=restarted_store,
            admission=authority["admission"],
        ),
    )
    restarted_adapter = _adapter(
        authority=authority,
        gateway=restarted_gateway,
        slot_allocator=must_not_allocate,
        now_ms=now_ms,
    )
    restarted_adapter.attach()
    prepared = restarted_adapter.prepare_authorized_plan_r2_transition(authority["authorization"])
    assert allocation_count == 0
    assert restarted_channel.requests == [original_request]
    assert prepared["authorized_prepare_command_cid"] == original_envelope_cid
    assert fabric.effect_count == 1

    # A later exact logical prepare is adopted from the committed client
    # journal without another process exchange or envelope allocation.
    replay = restarted_adapter.prepare_authorized_plan_r2_transition(authority["authorization"])
    assert replay == prepared
    assert restarted_channel.requests == [original_request]
    assert allocation_count == 0
    restarted_adapter.close()


def test_prepare_apply_and_fresh_observe_use_only_three_operation_wire(
    tmp_path: Path,
) -> None:
    now_ms = time.time_ns() // 1_000_000
    authority = _admitted_authority(now_ms)
    fabric, service, store = _runtime(tmp_path, authority, now_ms)
    channel = _WireChannel(service, lose_first_response=False)
    gateway = bind_plan_r2_process_remote_owner_gateway(
        admission=authority["admission"],
        channel=channel,
        journal=bind_plan_r2_remote_exact_envelope_journal(
            store=store,
            admission=authority["admission"],
        ),
    )
    adapter = _adapter(
        authority=authority,
        gateway=gateway,
        slot_allocator=iter([1, 2, 3, 4]).__next__,
        now_ms=now_ms,
    )
    adapter.attach()
    prepared = adapter.prepare_authorized_plan_r2_transition(authority["authorization"])
    receipt = adapter.apply_authorized_plan_r2_transition(authority["authorization"], prepared)
    first_observation = adapter.observe_authorized_plan_r2_transition(
        authority["authorization"], receipt
    )
    second_observation = adapter.observe_authorized_plan_r2_transition(
        authority["authorization"], receipt
    )
    requests = [json.loads(item) for item in channel.requests]
    assert [item["operation"] for item in requests] == [
        PREPARE_PLAN_R2_OPERATION,
        APPLY_PLAN_R2_OPERATION,
        OBSERVE_PLAN_R2_OPERATION,
        OBSERVE_PLAN_R2_OPERATION,
    ]
    assert all(item["operation"] in PLAN_R2_REMOTE_OPERATIONS for item in requests)
    assert requests[2]["envelope"]["envelope_cid"] != requests[3]["envelope"]["envelope_cid"]
    assert first_observation == second_observation
    assert receipt["after_plan_cid"] == authority["authorization"]["new_plan"]["plan_cid"]
    assert fabric.effect_count == 4
    adapter.close()


def test_journal_and_wire_reject_divergent_or_noncanonical_replay(
    tmp_path: Path,
) -> None:
    now_ms = time.time_ns() // 1_000_000
    authority = _admitted_authority(now_ms)
    _fabric, service, store = _runtime(tmp_path, authority, now_ms)
    channel = _WireChannel(service, lose_first_response=True)
    journal = bind_plan_r2_remote_exact_envelope_journal(
        store=store,
        admission=authority["admission"],
    )
    gateway = bind_plan_r2_process_remote_owner_gateway(
        admission=authority["admission"], channel=channel, journal=journal
    )
    adapter = _adapter(
        authority=authority,
        gateway=gateway,
        slot_allocator=iter([11]).__next__,
        now_ms=now_ms,
    )
    adapter.attach()
    with pytest.raises(PlanR2RemoteResponseUnavailable):
        adapter.prepare_authorized_plan_r2_transition(authority["authorization"])
    request = json.loads(channel.requests[0])
    divergent = deepcopy(request)
    divergent["operation_payload"]["authorization"]["delta_cid"] = _sha("d")
    with pytest.raises(PlanR2RemoteReplayDiverged, match="bytes diverged"):
        journal.lookup(divergent)

    noncanonical = json.dumps(request, indent=2, sort_keys=False).encode("ascii")
    with pytest.raises(PlanR2RemoteOwnerError, match="not canonical"):
        service.handle_exchange(noncanonical)
    adapter.close()


def test_channel_cannot_expose_token_path_portal_or_database_authority(
    tmp_path: Path,
) -> None:
    now_ms = time.time_ns() // 1_000_000
    authority = _admitted_authority(now_ms)
    _fabric, service, store = _runtime(tmp_path, authority, now_ms)

    class _LeakyChannel(_WireChannel):
        token = "not-authority"

    with pytest.raises(PlanR2RemoteOwnerError, match="forbidden authority: token"):
        bind_plan_r2_process_remote_owner_gateway(
            admission=authority["admission"],
            channel=_LeakyChannel(service, lose_first_response=False),
            journal=bind_plan_r2_remote_exact_envelope_journal(
                store=store,
                admission=authority["admission"],
            ),
        )
