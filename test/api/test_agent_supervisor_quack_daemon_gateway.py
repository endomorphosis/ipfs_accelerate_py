from __future__ import annotations

import base64
import json
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_agent_control_plane_promotion import (
    exact_plan_r2_operation_vocabulary,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateAuthorityClass,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    AUTHORIZED_STATE_COMMAND_INTERFACE,
    AUTHORIZED_STATE_COMMAND_SCHEMA,
    QuackCommandAuthorizationPolicy,
    authorized_state_command_signing_payload,
    seal_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackDaemonOwnerGateway,
    QuackPlanR2OwnerGateway,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE,
    QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE,
    QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA,
    QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE,
    REQUIRED_QUACK_DAEMON_OPERATIONS,
    QuackDaemonCanonicalOwnerOperationHandler,
    QuackDaemonCommandGateway,
    QuackDaemonGatewayCapability,
    QuackDaemonGatewayError,
    QuackDaemonOwnerOperationNoGo,
    quack_daemon_operation_command_vocabulary,
    quack_daemon_operational_capability_signing_payload,
    quack_daemon_owner_operation_dispositions,
    quack_daemon_state_command_parameters,
    seal_quack_daemon_operational_capability,
    verify_quack_daemon_operational_capability,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_CONTEXT,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)


def _cid(char: str) -> str:
    return "sha256:" + char * 64


def _capability() -> QuackDaemonGatewayCapability:
    return QuackDaemonGatewayCapability(
        board_namespace="board:eaaef",
        shard_id="control-shard-0",
        store_id="eaaef-control",
        control_plane_schema_version="QuackStateRepository@1",
        state_schema_revision="datasets-authoritative-operational-control-plane@1",
        command_endpoint="quack:127.0.0.1:19495",
        state_endpoint="quack:127.0.0.1:19496",
        owner_principal_did="did:key:z6Mkowner",
        owner_generation=1,
        fence_epoch=7,
        authorization_policy_cid=_cid("a"),
        command_fabric_qualification_cid=_cid("b"),
    )


def _canonical_signature(key: Ed25519PrivateKey, payload: Any) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return base64.b64encode(key.sign(encoded)).decode("ascii")


class _SignedOperationalAuthority:
    """Ephemeral protocol fixture; never a production qualification artifact."""

    def __init__(self, *, now_ms: int = 10_000) -> None:
        self.now_ms = now_ms
        self.reviewer_key = Ed25519PrivateKey.generate()
        self.approver_key = Ed25519PrivateKey.generate()
        self.principal_key = Ed25519PrivateKey.generate()
        self.owner_key = Ed25519PrivateKey.generate()
        self.reviewer_did = ed25519_did_key(self.reviewer_key.public_key())
        self.approver_did = ed25519_did_key(self.approver_key.public_key())
        self.principal_did = ed25519_did_key(self.principal_key.public_key())
        self.owner_did = ed25519_did_key(self.owner_key.public_key())
        self.policy = QuackCommandAuthorizationPolicy(
            board_namespace="board:eaaef",
            shard_id="control-shard-0",
            store_id="eaaef-control",
            authority_ref_cid=_cid("a"),
            owner_principal_did=self.owner_did,
            owner_generation=1,
            fence_epoch=7,
            trusted_approver_dids=frozenset({self.approver_did}),
            authorized_principal_dids=frozenset({self.principal_did}),
            allowed_command_kinds=frozenset(CommandKind),
        )
        body = {
            "schema": QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA,
            "interface": QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE,
            "board_namespace": self.policy.board_namespace,
            "shard_id": self.policy.shard_id,
            "store_id": self.policy.store_id,
            "control_plane_schema_version": "QuackStateRepository@1",
            "state_schema_revision": (
                "datasets-authoritative-operational-control-plane@1"
            ),
            "command_endpoint": "quack:127.0.0.1:19495",
            "state_endpoint": "quack:127.0.0.1:19496",
            "owner_principal_did": self.owner_did,
            "owner_generation": self.policy.owner_generation,
            "fence_epoch": self.policy.fence_epoch,
            "authorization_policy_cid": self.policy.policy_cid,
            "command_fabric_qualification_cid": _cid("b"),
            "authorized_state_command_schema": AUTHORIZED_STATE_COMMAND_SCHEMA,
            "authorized_state_command_interface": AUTHORIZED_STATE_COMMAND_INTERFACE,
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
            "expires_at_ms": now_ms + 100_000,
            "reviewer_identity_did": self.reviewer_did,
        }
        prepared = quack_daemon_operational_capability_signing_payload(body)
        self.capability = seal_quack_daemon_operational_capability(
            prepared,
            reviewer_signature=_canonical_signature(self.reviewer_key, prepared),
        )
        self.leases = {
            "board:eaaef": {
                "lease_id": "lease:gateway",
                "principal_did": self.principal_did,
                "fencing_token": 11,
                "fence_epoch": 7,
                "state": "accepted",
            }
        }
        self.tasks = {
            f"task:{index}": {
                "task_cid": f"task:{index}",
                "task_alias": f"EAAEF-{index:03d}",
                "dependencies": [],
                "status": "ready",
                "revision": 1,
                "body": {},
            }
            for index in (1, 2)
        }
        self.claimed: set[str] = set()
        self.receipts: dict[str, tuple[str, str, Any]] = {}
        self.requests: set[str] = set()
        self.nonces: set[str] = set()
        self.accepted_results = 0
        self.sequence = 0
        self.force_nonce = ""
        self.force_fencing_token: int | None = None
        self.replay_envelope: Any = None
        self.lock = threading.RLock()
        self.owner_gateway = QuackDaemonOwnerGateway(self)

    @property
    def daemon_production_capability_cid(self) -> str:
        return str(self.capability["capability_cid"])

    def _require_daemon_capability(self) -> Mapping[str, Any]:
        return self.capability

    def _scope_for_intent(self, intent: Mapping[str, Any]) -> str:
        arguments = dict(intent["arguments"])
        task_cid = str(arguments.get("task_cid") or "")
        return task_cid if task_cid in self.leases else "board:eaaef"

    def authorize(self, intent: Mapping[str, Any]) -> Any:
        if self.replay_envelope is not None:
            return self.replay_envelope
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
        scope_id = self._scope_for_intent(intent)
        lease = self.leases[scope_id]
        operation = str(intent["operation"])
        kind_by_prefix = {
            "task.list": CommandKind.OBSERVE,
            "task.ready": CommandKind.OBSERVE,
            "coordination.claim_ready": CommandKind.CLAIM,
        }
        command_kind = kind_by_prefix[operation]
        request_id = f"request:{sequence}"
        nonce = self.force_nonce or f"nonce:{sequence}"
        idempotency_key = f"idempotency:{sequence}"
        deadline = self.now_ms + 10_000
        fencing_token = (
            lease["fencing_token"]
            if self.force_fencing_token is None
            else self.force_fencing_token
        )
        parameters = quack_daemon_state_command_parameters(
            intent,
            request_id=request_id,
            principal_did=self.principal_did,
            authority_ref_cid=self.policy.authority_ref_cid,
            lease_id=str(lease["lease_id"]),
            scope_id=scope_id,
            deadline_ms=deadline,
            fencing_token=int(fencing_token),
            idempotency_key=idempotency_key,
        )
        command = StateCommand(
            command_id=f"{request_id}:{operation.replace('.', '-')}",
            command_kind=command_kind,
            store_id=self.policy.store_id,
            session_id=str(lease["lease_id"]),
            expected_generation=self.policy.owner_generation,
            expected_revision=1,
            fence_epoch=self.policy.fence_epoch,
            idempotency_key=idempotency_key,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters=parameters,
        )
        unsigned = authorized_state_command_signing_payload(
            request_id=request_id,
            submission_id=f"submission:{sequence}",
            ingress_slot=sequence,
            principal_did=self.principal_did,
            approver_did=self.approver_did,
            authority_ref_cid=self.policy.authority_ref_cid,
            board_namespace=self.policy.board_namespace,
            shard_id=self.policy.shard_id,
            owner_principal_did=self.owner_did,
            lease_id=str(lease["lease_id"]),
            scope_id=scope_id,
            effect=f"control-plane/{command_kind.value}",
            issued_at_ms=self.now_ms - 1,
            expires_at_ms=self.now_ms + 20_000,
            deadline_ms=deadline,
            one_use_nonce=nonce,
            command=command,
        )
        return seal_authorized_state_command(
            unsigned,
            approver_signature=_canonical_signature(self.approver_key, unsigned),
        )

    def _submit_authorized_daemon_operation(
        self, envelope: Any, intent: Mapping[str, Any]
    ) -> Any:
        with self.lock:
            prior = self.receipts.get(envelope.submission_id)
            if prior is not None:
                prior_envelope, prior_intent, result = prior
                if (
                    prior_envelope != envelope.envelope_cid
                    or prior_intent != intent["intent_cid"]
                ):
                    raise RuntimeError("divergent replay")
                return result
            if envelope.request_id in self.requests or envelope.one_use_nonce in self.nonces:
                raise RuntimeError("request or nonce already consumed")
            lease = self.leases[envelope.scope_id]
            if lease["state"] != "accepted":
                raise RuntimeError("lease revoked")
            if (
                envelope.lease_id != lease["lease_id"]
                or envelope.principal_did != lease["principal_did"]
                or int(envelope.command.parameters["fencing_token"])
                != int(lease["fencing_token"])
                or envelope.command.fence_epoch != lease["fence_epoch"]
            ):
                raise RuntimeError("stale fence")
            operation = str(intent["operation"])
            if operation == "task.ready":
                result: Any = {
                    "tasks": [
                        dict(task)
                        for task_cid, task in sorted(self.tasks.items())
                        if task_cid not in self.claimed
                    ]
                }
            elif operation == "task.list":
                result = {"tasks": [dict(task) for task in self.tasks.values()]}
            elif operation == "coordination.claim_ready":
                candidates = sorted(set(self.tasks) - self.claimed)
                if not candidates:
                    result = None
                else:
                    task_cid = candidates[0]
                    self.claimed.add(task_cid)
                    result = {
                        "claim_id": f"claim:{task_cid}",
                        "task_cid": task_cid,
                        "owner_session_id": intent["arguments"]["owner_session_id"],
                        "fencing_token": 11 + len(self.claimed),
                        "fence_epoch": 7,
                        "attempt_id": f"attempt:{task_cid}",
                        "attempt_number": 1,
                        "lease_id": f"lease:{task_cid}",
                        "worktree_id": f"worktree:{task_cid}",
                        "claimed_at_ms": self.now_ms,
                        "expires_at_ms": self.now_ms + 60_000,
                        "state": "accepted",
                        "revision": 1,
                    }
            else:
                raise AssertionError(operation)
            self.requests.add(envelope.request_id)
            self.nonces.add(envelope.one_use_nonce)
            self.receipts[envelope.submission_id] = (
                envelope.envelope_cid,
                str(intent["intent_cid"]),
                result,
            )
            self.accepted_results += 1
            return result

    def gateway(self) -> QuackDaemonCommandGateway:
        return QuackDaemonCommandGateway.from_operational_capability(
            self.capability,
            trusted_reviewer_dids=(self.reviewer_did,),
            authorization_policy=self.policy,
            authorization_provider=self.authorize,
            clock_ms=lambda: self.now_ms,
        )


class _SharedAuthority:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tasks = {
            f"task:{index}": {
                "task_cid": f"task:{index}",
                "task_alias": f"EAAEF-{index:03d}",
                "dependencies": (),
                "status": "ready",
                "revision": 1,
                "body": {},
            }
            for index in (1, 2)
        }
        self.claims: dict[str, SimpleNamespace] = {}
        self.claimed_tasks: set[str] = set()
        self.attempts: dict[str, dict[str, Any]] = {}
        self.phases: dict[str, list[dict[str, Any]]] = {}
        self.results: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []


class _Component:
    GATEWAY_COMPONENT_INTERFACE = QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE

    def __init__(self, binding: str, shared: _SharedAuthority) -> None:
        self.gateway_binding_cid = binding
        self.shared = shared
        self.attached = False

    def attach(self) -> None:
        self.attached = True

    def close(self) -> None:
        self.attached = False


class _TaskSource(_Component):
    @staticmethod
    def _task(record: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**record)

    def materialize(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"materialized": True}

    def list_tasks(self, *, limit: int) -> SimpleNamespace:
        del limit
        with self.shared.lock:
            return SimpleNamespace(
                tasks=[self._task(dict(item)) for item in self.shared.tasks.values()]
            )

    def ready_tasks(self, *, limit: int) -> SimpleNamespace:
        del limit
        with self.shared.lock:
            return SimpleNamespace(
                tasks=[
                    self._task(dict(item))
                    for item in self.shared.tasks.values()
                    if item["status"] == "ready"
                ]
            )

    def get(self, task_cid: str) -> SimpleNamespace | None:
        with self.shared.lock:
            item = self.shared.tasks.get(task_cid)
            return None if item is None else self._task(dict(item))

    def compare_and_set_status(
        self,
        task_cid: str,
        *,
        expected_revision: int,
        status: str,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        with self.shared.lock:
            task = self.shared.tasks[task_cid]
            if task["revision"] != expected_revision:
                raise RuntimeError("stale task revision")
            task["status"] = status
            task["revision"] += 1
            return SimpleNamespace(to_dict=lambda: dict(task))

    def record_validation_result(self, **_kwargs: Any) -> dict[str, Any]:
        return {"recorded": True}


class _Coordinator(_Component):
    def register_task(self, **_kwargs: Any) -> dict[str, Any]:
        return {"registered": True}

    def claim_ready_task(
        self,
        *,
        owner_session_id: str,
        lease_ms: int,
        exclude_task_cids: set[str],
        now_ms: int,
    ) -> SimpleNamespace | None:
        with self.shared.lock:
            candidates = sorted(
                task_cid
                for task_cid, task in self.shared.tasks.items()
                if task["status"] == "ready"
                and task_cid not in self.shared.claimed_tasks
                and task_cid not in exclude_task_cids
            )
            if not candidates:
                return None
            task_cid = candidates[0]
            self.shared.claimed_tasks.add(task_cid)
            attempt_number = 1
            claim_id = f"claim:{task_cid}:{attempt_number}"
            claim = SimpleNamespace(
                claim_id=claim_id,
                task_cid=task_cid,
                owner_session_id=owner_session_id,
                fencing_token=len(self.shared.claims) + 1,
                fence_epoch=7,
                claimed_at_ms=now_ms,
                expires_at_ms=now_ms + lease_ms,
                state="accepted",
                revision=1,
                attempt_id=f"attempt:{task_cid}:{attempt_number}",
                attempt_number=attempt_number,
                lease_id=f"lease:{task_cid}:{attempt_number}",
                worktree_id=f"worktree:{task_cid}",
                to_dict=lambda: {},
            )
            claim.to_dict = lambda claim=claim: dict(vars(claim), to_dict=None)
            self.shared.claims[claim_id] = claim
            return claim

    def get_task_claim(self, claim_id: str) -> SimpleNamespace | None:
        return self.shared.claims.get(claim_id)

    def protect_task_claim(self, claim: SimpleNamespace, **expected: Any) -> SimpleNamespace:
        current = self.shared.claims.get(claim.claim_id)
        if current is None:
            raise RuntimeError("claim missing")
        checks = {
            "task_cid": "expected_task_cid",
            "attempt_id": "expected_attempt_id",
            "owner_session_id": "expected_owner_session_id",
            "fencing_token": "expected_fencing_token",
            "fence_epoch": "expected_fence_epoch",
        }
        for field, parameter in checks.items():
            supplied = expected.get(parameter)
            if supplied is not None and getattr(current, field) != supplied:
                raise RuntimeError("stale fence")
        return current

    def renew(self, lease: SimpleNamespace, **_kwargs: Any) -> SimpleNamespace:
        return self.protect_task_claim(lease)

    def prepare_task_completion(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "prepared"}

    def get_prepared_task_completion(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def complete_task_claim(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "completed"}

    def settle_task_claim(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "settled"}

    def list_unsettled_task_completions(self, **_kwargs: Any) -> list[Any]:
        return []

    def reconcile_promoted_task_completion(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "reconciled"}

    def recover_prepared_task_completion(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "recovered"}

    def abort_prepared_task_completion(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "aborted"}

    def expire_task_claim(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "expired"}


class _ExecutionRepository(_Component):
    def bind_daemon(self, metadata: dict[str, Any]) -> None:
        self.shared.metadata.append(dict(metadata))

    def record_event(self, **event: Any) -> None:
        self.shared.events.append(dict(event))

    def ensure_attempt(
        self, *, attempt: dict[str, Any], claimed_phase: dict[str, Any]
    ) -> dict[str, Any]:
        with self.shared.lock:
            current = self.shared.attempts.setdefault(attempt["attempt_id"], dict(attempt))
            if current != attempt:
                raise RuntimeError("divergent attempt replay")
            self.shared.phases.setdefault(attempt["attempt_id"], [dict(claimed_phase)])
            return dict(current)

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        record = self.shared.attempts.get(attempt_id)
        return None if record is None else dict(record)

    def list_running_attempts(self, *, owner_session_id: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.shared.attempts.values()
            if item["owner_session_id"] == owner_session_id and item["status"] == "running"
        ]

    def commit_phase(self, **operation: Any) -> dict[str, Any] | None:
        with self.shared.lock:
            record = self.shared.attempts[operation["attempt_id"]]
            if (
                record["revision"] != operation["expected_revision"]
                or record["status"] != operation["expected_status"]
                or record["fencing_token"] != operation["fencing_token"]
                or record["fence_epoch"] != operation["fence_epoch"]
            ):
                return None
            record.update(
                committed_phase=operation["committed_phase"],
                status=operation["status"],
                finished_at_ms=operation["finished_at_ms"],
                revision=operation["revision"],
            )
            self.shared.phases[operation["attempt_id"]].append(
                {
                    "phase": operation["committed_phase"],
                    "committed_at_ms": operation["committed_at_ms"],
                    "fencing_token": operation["fencing_token"],
                    "fence_epoch": operation["fence_epoch"],
                    "revision": operation["revision"],
                    "body": dict(operation["body"]),
                }
            )
            return dict(record)

    def commit_reconciled_attempt(self, **operation: Any) -> dict[str, Any] | None:
        return self.commit_phase(**operation)

    def phase_history(self, attempt_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self.shared.phases.get(attempt_id, ())]

    def get_idempotent_result(
        self, *, kind: str, attempt_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        result = self.shared.results.get((kind, attempt_id, idempotency_key))
        return None if result is None else dict(result)

    def record_idempotent_result(self, **record: Any) -> dict[str, Any]:
        key = (record["kind"], record["attempt_id"], record["idempotency_key"])
        result = dict(record["result"])
        with self.shared.lock:
            prior = self.shared.results.setdefault(key, result)
            if prior != result:
                raise RuntimeError("divergent idempotency replay")
            return dict(prior)

    def reserve_provider(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def commit_provider(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def reserve_effect(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def commit_effect(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def record_validation(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)


class _MergeRepository(_Component):
    def enqueue(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def observe(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def accept(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)


class _PlanRepository(_Component):
    def prepare(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def apply(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)

    def observe(self, **operation: Any) -> dict[str, Any]:
        return dict(operation)


class _ProductionPlanRepository(_PlanRepository):
    INTERFACE = "AuthorizedPlanR2TransitionRepository@1"

    def __init__(self, binding: str, shared: _SharedAuthority, owner_gateway: Any) -> None:
        super().__init__(binding, shared)
        self.owner_gateway = owner_gateway


def _gateway(
    capability: QuackDaemonGatewayCapability, shared: _SharedAuthority
) -> QuackDaemonCommandGateway:
    binding = capability.content_id
    return QuackDaemonCommandGateway(
        capability=capability,
        task_source=_TaskSource(binding, shared),
        coordinator=_Coordinator(binding, shared),
        execution_repository=_ExecutionRepository(binding, shared),
        merge_repository=_MergeRepository(binding, shared),
        plan_repository=_PlanRepository(binding, shared),
    )


def test_quack_daemon_refuses_legacy_sidecars_before_file_access(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="QuackDaemonCommandGateway@1",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri="quack:127.0.0.1:19495",
        )
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="independently injected",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri="quack:127.0.0.1:19495",
            task_source=object(),
        )


def test_gateway_rejects_incomplete_or_unreviewed_operation_vocabulary() -> None:
    assert len(REQUIRED_QUACK_DAEMON_OPERATIONS) == 39
    with pytest.raises(QuackDaemonGatewayError, match="not exact"):
        replace(
            _capability(),
            operations=REQUIRED_QUACK_DAEMON_OPERATIONS - {"effect.commit"},
        )
    with pytest.raises(QuackDaemonGatewayError, match="not exact"):
        replace(
            _capability(),
            operations=REQUIRED_QUACK_DAEMON_OPERATIONS | {"sql.execute"},
        )
    with pytest.raises(QuackDaemonGatewayError, match="structural gateway"):
        replace(_capability(), production_admitted=True)


def test_external_operational_capability_is_signed_closed_and_not_self_promoting() -> None:
    authority = _SignedOperationalAuthority()
    verified = verify_quack_daemon_operational_capability(
        authority.capability,
        trusted_reviewer_dids=(authority.reviewer_did,),
        now_ms=authority.now_ms,
    )
    assert verified["operations"] == sorted(REQUIRED_QUACK_DAEMON_OPERATIONS)
    assert verified["reviewer_identity_did"] != verified["owner_principal_did"]

    tampered = dict(authority.capability)
    tampered["operations"] = tampered["operations"][:-1]
    with pytest.raises(QuackDaemonGatewayError, match="operation vocabulary|CID mismatch"):
        verify_quack_daemon_operational_capability(
            tampered,
            trusted_reviewer_dids=(authority.reviewer_did,),
            now_ms=authority.now_ms,
        )

    with pytest.raises(
        QuackDaemonGatewayError,
        match="canonical_39_operation_owner_handler_unqualified",
    ):
        authority.gateway()


def test_daemon_registry_reuses_exact_promoted_plan_r2_vocabulary() -> None:
    daemon_vocabulary = quack_daemon_operation_command_vocabulary()
    promoted = {
        str(item["operation"]): str(item["command_kind"])
        for item in exact_plan_r2_operation_vocabulary()
    }
    assert {
        operation: daemon_vocabulary[operation] for operation in promoted
    } == promoted


def test_canonical_owner_handler_has_exact_fail_closed_operation_parity() -> None:
    dispositions = quack_daemon_owner_operation_dispositions()
    assert set(dispositions) == REQUIRED_QUACK_DAEMON_OPERATIONS
    assert len(dispositions) == 39
    assert {
        operation
        for operation, record in dispositions.items()
        if record["disposition"] == "admitted_owner_transaction"
    } == {"task.get", "task.list", "task.ready"}
    assert all(
        record["reason_code"]
        for record in dispositions.values()
        if record["disposition"] == "typed_no_go"
    )
    handler = QuackDaemonCanonicalOwnerOperationHandler()
    evidence = handler.evidence()
    assert evidence["operation_count"] == 39
    assert evidence["all_operations_recognized"] is True
    assert evidence["production_admitted"] is False
    assert evidence["opens_database"] is False
    assert evidence["owns_transaction_lifecycle"] is False
    with pytest.raises(
        QuackDaemonOwnerOperationNoGo,
        match=(
            "operation=effect.reserve;"
            "reason_code=effect_reservation_before_external_effect_unqualified"
        ),
    ):
        handler.require_operation("effect.reserve")
    with pytest.raises(QuackDaemonGatewayError, match="outside the closed"):
        handler.require_operation("sql.execute")


def test_plan_r2_component_requires_explicit_signed_owner_dispatch_capability() -> None:
    capability = _capability()
    shared = _SharedAuthority()
    binding = capability.content_id
    common = {
        "capability": capability,
        "task_source": _TaskSource(binding, shared),
        "coordinator": _Coordinator(binding, shared),
        "execution_repository": _ExecutionRepository(binding, shared),
        "merge_repository": _MergeRepository(binding, shared),
    }
    with pytest.raises(QuackDaemonGatewayError, match="production owner-dispatch"):
        QuackDaemonCommandGateway(
            **common,
            plan_repository=_ProductionPlanRepository(binding, shared, object()),
        )
    forged_owner_gateway = SimpleNamespace(
        INTERFACE="AuthorizedStateCommandPlanR2OwnerGateway@1",
        production_capability_cid=_cid("c"),
        command_fabric_qualification_cid=(capability.command_fabric_qualification_cid),
    )
    with pytest.raises(QuackDaemonGatewayError, match="production owner-dispatch"):
        QuackDaemonCommandGateway(
            **common,
            plan_repository=_ProductionPlanRepository(binding, shared, forged_owner_gateway),
        )

    class _ForgedSubclass(QuackPlanR2OwnerGateway):
        def __init__(self) -> None:
            pass

        @property
        def production_capability_cid(self) -> str:
            return _cid("c")

        @property
        def command_fabric_qualification_cid(self) -> str:
            return capability.command_fabric_qualification_cid

    with pytest.raises(QuackDaemonGatewayError, match="production owner-dispatch"):
        QuackDaemonCommandGateway(
            **common,
            plan_repository=_ProductionPlanRepository(binding, shared, _ForgedSubclass()),
        )


def test_structural_gateway_cannot_start_real_execution(tmp_path: Path) -> None:
    capability = _capability()
    shared = _SharedAuthority()
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not production-admitted",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "must-not-open.duckdb",
            owner_session_id="lane:production",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri=capability.command_endpoint,
            quack_command_gateway=_gateway(capability, shared),
            require_real_execution=True,
        )
    assert shared.metadata == []
    assert list(tmp_path.iterdir()) == []


def test_two_daemons_claim_distinct_tasks_through_one_gateway_authority(
    tmp_path: Path,
) -> None:
    capability = _capability()
    shared = _SharedAuthority()
    daemons = [
        DatabaseImplementationDaemon(
            database_path=tmp_path / "must-not-open.duckdb",
            owner_session_id=f"lane:{index}",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri=capability.command_endpoint,
            quack_command_gateway=_gateway(capability, shared),
        )
        for index in (1, 2)
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(lambda daemon: daemon.claim_next(), daemons))
        assert all(attempt is not None for attempt in attempts)
        assert len({attempt.task_cid for attempt in attempts if attempt is not None}) == 2
        assert len(shared.attempts) == 2
        assert not (tmp_path / "must-not-open.duckdb").exists()
        assert not list(tmp_path.glob("*.execution.duckdb"))
        assert not list(tmp_path.glob("*.coordination.duckdb"))
    finally:
        for daemon in daemons:
            daemon.close()


def test_stale_fence_and_divergent_idempotency_replays_fail_closed(
    tmp_path: Path,
) -> None:
    capability = _capability()
    shared = _SharedAuthority()
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "must-not-open.duckdb",
        owner_session_id="lane:1",
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_uri=capability.command_endpoint,
        quack_command_gateway=_gateway(capability, shared),
    )
    try:
        attempt = daemon.claim_next()
        assert attempt is not None
        claim = shared.claims[attempt.claim_id]
        claim.fencing_token += 1
        with pytest.raises(RuntimeError, match="stale fence"):
            daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)

        repository = daemon._require_execution_repository()  # noqa: SLF001
        common = {
            "kind": "provider",
            "record_id": "provider:1",
            "attempt_id": attempt.attempt_id,
            "task_cid": attempt.task_cid,
            "operation_key": "",
            "idempotency_key": "provider:once",
            "owner_session_id": "lane:1",
            "recorded_at_ms": 1,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
        }
        assert repository.record_idempotent_result(**common, result={"accepted": True}) == {
            "accepted": True
        }
        assert repository.record_idempotent_result(**common, result={"accepted": True}) == {
            "accepted": True
        }
        with pytest.raises(RuntimeError, match="divergent idempotency"):
            repository.record_idempotent_result(**common, result={"accepted": False})
    finally:
        daemon.close()
