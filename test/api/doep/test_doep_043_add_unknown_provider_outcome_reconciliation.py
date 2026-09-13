"""Independent current-tree checks for DOEP-043 unknown-outcome reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_BINDING,
    UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_CONSUMES,
    UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_SCHEMA,
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionPhase,
    ProviderExecutionRequest,
    SideEffectBoundary,
    UnknownProviderOutcomeDisposition,
    UnknownProviderOutcomeReconciliation,
    assert_unknown_provider_outcome_transition,
    blind_retry_forbidden,
    build_execution_request,
    is_unknown_provider_outcome,
    new_attempt_idempotency_key,
    task_state_for_execution_result,
    unknown_provider_outcome_retry_allowed,
    unknown_provider_outcome_successors,
)
from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageScope,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    STATE_TRANSACTION_INTERFACE,
    FenceMismatchError,
    OptimisticConflictError,
)
from ipfs_accelerate_py.endpoint_usage import LimitWindow, UsageVector, WindowKind


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/provider_execution.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-043.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-043.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_execution.py",
    "test/api/doep/test_doep_043_add_unknown_provider_outcome_reconciliation.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-043.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-043.json",
)
TASK_CID = "sha256:e9719e129e9c3aba17241c5362ccbdd3d49b3f7fffabea564c6d044f9fa1fb38"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _budget() -> SupervisorUsageBudget:
    return SupervisorUsageBudget.of(
        window=LimitWindow(kind=WindowKind.LIFETIME),
        currency="USD",
        requests=1,
        input_tokens=100,
        output_tokens=50,
        cost_micros=10_000,
    )


def _request(
    *,
    request_id: str = "request:doep-043-1",
    idempotency_key: str = "idem:doep-043",
    attempt: int = 1,
    lease_id: str = "lease:1",
    fence_id: str = "1",
    side_effect_boundary: SideEffectBoundary = SideEffectBoundary.IDEMPOTENT,
    catalog_revision: str = "catalog:rev-1",
    usage_revision: str = "usage:rev-1",
) -> ProviderExecutionRequest:
    scope = SupervisorUsageScope(
        level=SupervisorUsageLevel.REQUEST,
        repository_id="repository:supervisor",
        state_id="state:lane-0",
        tree_id="tree:doep-043",
        policy_id="policy:implementation-daemon",
        policy_revision="policy:implementation-daemon@1",
        supervisor_run_id="run:doep-043",
        goal_id="DOEP-G050",
        objective_id="DOEP-043",
        objective_revision="sha256:e9719e129e9c3aba17241c5362ccbdd3d49b3f7fffabea564c6d044f9fa1fb38",
        task_id="DOEP-043",
        attempt=attempt,
        stage="implementation",
        lane="lane-0",
        request_id=request_id,
        catalog_revision=catalog_revision,
        usage_revision=usage_revision,
        endpoint_scope_id="endpoint-scope:doep-043",
        caller_id="caller:supervisor",
        deadline_at="2026-09-13T00:00:00Z",
        idempotency_key=idempotency_key,
        lease_id=lease_id,
        fence_id=fence_id,
        parent_scope_id="scope:parent-lane",
    )
    envelope = SupervisorUsageEnvelope(scope=scope, budget=_budget())
    bridge = SupervisorToEndpointRequest(
        scope=scope,
        envelope_id=envelope.envelope_id,
        endpoint_scope_id=scope.endpoint_scope_id,
        catalog_revision=scope.catalog_revision,
        usage_revision=scope.usage_revision,
        estimated=UsageVector.of(requests=1),
        request_id=scope.request_id,
        attempt=scope.attempt,
        idempotency_key=scope.idempotency_key,
        caller_id=scope.caller_id,
        deadline_at=scope.deadline_at,
        lease_id=scope.lease_id,
        fence_id=scope.fence_id,
    )
    return build_execution_request(
        bridge=bridge,
        envelope=envelope,
        provider_id="provider:example",
        modality="text",
        side_effect_boundary=side_effect_boundary,
        operation="text.generate",
    )


def _snapshot(**updates: object) -> TaskStateSnapshot:
    values: dict[str, object] = {
        "task_cid": TASK_CID,
        "state": TaskState.PROVIDER_OUTCOME_UNKNOWN,
        "revision": 2,
        "lease_id": "lease:1",
        "fence_epoch": 1,
        "policy_cid": "policy:current",
        "repository_tree_id": "tree:current",
        "plan_cid": PLAN_CID,
        "plan_epoch": 1,
    }
    values.update(updates)
    return TaskStateSnapshot(**values)


class _FakeDecision:
    def __init__(self, reservation_id: str = "reservation:1") -> None:
        self.granted = True
        self.reservation_id = reservation_id
        self.usage_revision = "usage:rev-1"
        self.reason_codes: tuple[str, ...] = ()


class _FakeSettlement:
    def __init__(self, *, charged: UsageVector | None = None, event_id: str = "event:1") -> None:
        self.charged = charged if charged is not None else UsageVector.of(requests=1)
        self.event_id = event_id
        self.usage_revision = "usage:rev-1"
        self.state = "committed"


class FakeCoordinator:
    def __init__(self) -> None:
        self.reserve_calls = 0
        self.commit_calls = 0
        self.cancel_calls = 0
        self.release_calls = 0
        self.dispatch_calls = 0

    def reserve(self, scope_id: str, requested: Any, **kwargs: Any) -> _FakeDecision:
        del scope_id, requested, kwargs
        self.reserve_calls += 1
        return _FakeDecision(reservation_id=f"reservation:{self.reserve_calls}")

    def mark_dispatched(self, reservation_id: str) -> None:
        del reservation_id
        self.dispatch_calls += 1

    def cancel(self, reservation_id: str, *, reason: str = "cancelled") -> _FakeSettlement:
        del reservation_id, reason
        self.cancel_calls += 1
        return _FakeSettlement(event_id="cancel-event:1")

    def release(self, reservation_id: str, *, reason: str = "released") -> _FakeSettlement:
        del reservation_id, reason
        self.release_calls += 1
        return _FakeSettlement(charged=UsageVector(), event_id="release-event:1")

    def commit(
        self,
        reservation_id: str,
        actual: Any = None,
        *,
        observation_id: str | None = None,
        release_unused: bool = True,
    ) -> _FakeSettlement:
        del reservation_id, actual, observation_id, release_unused
        self.commit_calls += 1
        return _FakeSettlement()


def _failing_invoker(calls: dict[str, int]):
    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        raise TimeoutError("provider response lost")

    return invoker


def _unknown_status_invoker(calls: dict[str, int]):
    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"status": "provider_outcome_unknown", "units": {"requests": 1}}

    return invoker


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_unknown_effects_enter_reconciliation_and_never_blind_retry() -> None:
    assert UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_BINDING == (
        "UnknownProviderOutcomeReconciliation@1"
    )
    assert UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_SCHEMA.endswith(
        "unknown-provider-outcome-reconciliation@1"
    )
    assert UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_CONSUMES == (
        CANONICAL_TASK_STATE_MACHINE_INTERFACE,
        STATE_TRANSACTION_INTERFACE,
    )
    assert unknown_provider_outcome_successors() == (TaskState.RECONCILING,)
    assert not unknown_provider_outcome_retry_allowed(
        task_state=TaskState.PROVIDER_OUTCOME_UNKNOWN, same_attempt=True
    )
    assert not unknown_provider_outcome_retry_allowed(
        task_state=TaskState.PROVIDER_OUTCOME_UNKNOWN,
        same_attempt=False,
        reconciled=False,
    )
    assert unknown_provider_outcome_retry_allowed(
        task_state=TaskState.RETRYING, same_attempt=False, reconciled=True
    )
    assert_unknown_provider_outcome_transition(
        TaskState.PROVIDER_OUTCOME_UNKNOWN, TaskState.RECONCILING
    )
    with pytest.raises(ControlPlaneContractError, match="blindly retried"):
        assert_unknown_provider_outcome_transition(
            TaskState.PROVIDER_OUTCOME_UNKNOWN, TaskState.RETRYING
        )
    with pytest.raises(ControlPlaneContractError):
        assert_unknown_provider_outcome_transition(
            TaskState.PROVIDER_OUTCOME_UNKNOWN, TaskState.COMPLETED
        )

    coordinator = FakeCoordinator()
    calls = {"n": 0}
    gateway = ProviderExecutionGateway(
        coordinator=coordinator, invoker=_failing_invoker(calls)
    )
    request = _request()
    result = gateway.execute(request)
    assert result.phase is ProviderExecutionPhase.PROVIDER_OUTCOME_UNKNOWN
    assert result.final_status is SupervisorUsageFinalStatus.UNKNOWN
    assert is_unknown_provider_outcome(result)
    assert blind_retry_forbidden(result)
    assert task_state_for_execution_result(result) is TaskState.PROVIDER_OUTCOME_UNKNOWN
    assert result.success is False
    assert result.is_completion_evidence is False
    assert "never_blind_retry" in result.reason_codes
    assert "provider_outcome_unknown" in result.reason_codes
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 1
    assert coordinator.dispatch_calls == 1
    assert coordinator.cancel_calls == 1

    replayed = gateway.execute(request)
    assert replayed.phase is ProviderExecutionPhase.PROVIDER_OUTCOME_UNKNOWN
    assert replayed.replayed is True
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 1

    retry_request = _request(
        request_id="request:doep-043-2",
        idempotency_key=new_attempt_idempotency_key("idem:doep-043", 2),
        attempt=2,
    )
    with pytest.raises(ProviderExecutionError) as retry_exc:
        gateway.execute(retry_request)
    assert "blind_retry_forbidden" in retry_exc.value.reason_codes
    assert calls["n"] == 1

    entered = gateway.reconcile(request)
    assert entered.phase is ProviderExecutionPhase.RECONCILING
    assert task_state_for_execution_result(entered) is TaskState.RECONCILING
    assert entered.success is False
    assert calls["n"] == 1

    failed = gateway.reconcile(
        request,
        disposition=UnknownProviderOutcomeDisposition.FAILED,
        evidence_ids=("event:reconcile-1",),
        expected_task=_snapshot(state=TaskState.RECONCILING, revision=3),
        proposed_task=_snapshot(state=TaskState.FAILED, revision=4),
    )
    assert failed.phase is ProviderExecutionPhase.FAILED
    assert task_state_for_execution_result(failed) is TaskState.FAILED
    assert "reconciled" in failed.reason_codes
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 1


def test_inconclusive_observation_and_read_only_failure_are_classified() -> None:
    unknown_calls = {"n": 0}
    gateway = ProviderExecutionGateway(
        coordinator=FakeCoordinator(), invoker=_unknown_status_invoker(unknown_calls)
    )
    unknown = gateway.execute(_request())
    assert is_unknown_provider_outcome(unknown)
    assert unknown.phase is ProviderExecutionPhase.PROVIDER_OUTCOME_UNKNOWN
    assert unknown_calls["n"] == 1

    known_calls = {"n": 0}
    known_gateway = ProviderExecutionGateway(
        coordinator=FakeCoordinator(), invoker=_failing_invoker(known_calls)
    )
    known = known_gateway.execute(
        _request(side_effect_boundary=SideEffectBoundary.READ_ONLY)
    )
    assert known.phase is ProviderExecutionPhase.FAILED
    assert not is_unknown_provider_outcome(known)
    assert known_calls["n"] == 1


def test_reconciliation_consumes_cas_and_rejects_stale_or_skipped_edges() -> None:
    calls = {"n": 0}
    gateway = ProviderExecutionGateway(
        coordinator=FakeCoordinator(), invoker=_failing_invoker(calls)
    )
    request = _request()
    gateway.execute(request)
    with pytest.raises(ControlPlaneContractError, match="must enter reconciliation"):
        gateway.reconcile(
            request,
            disposition=UnknownProviderOutcomeDisposition.FAILED,
            expected_task=_snapshot(),
            proposed_task=_snapshot(state=TaskState.FAILED, revision=3),
        )
    entered = gateway.reconcile(
        request,
        expected_task=_snapshot(),
        proposed_task=_snapshot(state=TaskState.RECONCILING, revision=3),
    )
    assert entered.phase is ProviderExecutionPhase.RECONCILING
    with pytest.raises(OptimisticConflictError):
        gateway.reconcile(
            request,
            disposition=UnknownProviderOutcomeDisposition.FAILED,
            expected_task=_snapshot(state=TaskState.RECONCILING, revision=3),
            proposed_task=_snapshot(state=TaskState.FAILED, revision=5),
        )
    stale_lease = _snapshot(
        state=TaskState.FAILED, revision=4, lease_id="lease:replaced"
    )
    with pytest.raises(OptimisticConflictError):
        gateway.reconcile(
            request,
            disposition=UnknownProviderOutcomeDisposition.FAILED,
            expected_task=_snapshot(state=TaskState.RECONCILING, revision=3),
            proposed_task=stale_lease,
        )
    record = UnknownProviderOutcomeReconciliation(
        attempt_key=request.attempt_key,
        request_key=request.request_key,
        request_id=request.bridge.request_id,
        reservation_id="reservation:1",
        source_task_state=TaskState.PROVIDER_OUTCOME_UNKNOWN,
        target_task_state=TaskState.RECONCILING,
        lease_id="lease:1",
        fence_id="1",
    )
    assert record.same_attempt_retry_allowed is False
    assert record.is_completion_evidence is False
    assert record.authorizes_completion is False
    with pytest.raises(ControlPlaneContractError):
        UnknownProviderOutcomeReconciliation(
            attempt_key=request.attempt_key,
            request_key=request.request_key,
            request_id=request.bridge.request_id,
            reservation_id="reservation:1",
            source_task_state=TaskState.PROVIDER_OUTCOME_UNKNOWN,
            target_task_state=TaskState.RETRYING,
        )


def test_worker_assertion_is_not_reconciliation_authority() -> None:
    calls = {"n": 0}
    gateway = ProviderExecutionGateway(
        coordinator=FakeCoordinator(), invoker=_failing_invoker(calls)
    )
    request = _request()
    gateway.execute(request)
    stale = _request(lease_id="lease:replaced")
    with pytest.raises(ProviderExecutionError) as lease_exc:
        gateway.reconcile(
            stale,
            disposition=UnknownProviderOutcomeDisposition.RETRYING,
            worker_assertion=True,
        )
    assert "stale_lease" in lease_exc.value.reason_codes
    fenced = _request(fence_id="9")
    with pytest.raises(FenceMismatchError):
        gateway.reconcile(
            fenced,
            disposition=UnknownProviderOutcomeDisposition.COMPLETED,
            worker_assertion=True,
        )
    with pytest.raises(ProviderExecutionError) as complete_exc:
        gateway.reconcile(
            request,
            disposition=UnknownProviderOutcomeDisposition.COMPLETED,
            worker_assertion=True,
        )
    assert "completion_not_authorized" in complete_exc.value.reason_codes

    retry_calls = {"n": 0}

    def selective_invoker(current: ProviderExecutionRequest) -> Mapping[str, Any]:
        retry_calls["n"] += 1
        if current.bridge.attempt == 1:
            raise TimeoutError("provider response lost")
        return {"status": "ok", "units": {"requests": 1}}

    retry_gateway = ProviderExecutionGateway(
        coordinator=FakeCoordinator(), invoker=selective_invoker
    )
    first = retry_gateway.execute(request)
    assert is_unknown_provider_outcome(first)
    retrying = retry_gateway.reconcile(
        request,
        disposition=UnknownProviderOutcomeDisposition.RETRYING,
        evidence_ids=("event:retry",),
        worker_assertion=True,
    )
    assert "worker_assertion_insufficient" in retrying.reason_codes
    assert retrying.is_completion_evidence is False
    assert not unknown_provider_outcome_retry_allowed(
        task_state=task_state_for_execution_result(retrying),
        same_attempt=True,
        reconciled=True,
    )
    assert unknown_provider_outcome_retry_allowed(
        task_state=TaskState.RETRYING, same_attempt=False, reconciled=True
    )
    assert retry_calls["n"] == 1
    same_attempt = retry_gateway.execute(request)
    assert same_attempt.replayed is True
    assert retry_calls["n"] == 1
    new_attempt = _request(
        request_id="request:doep-043-3",
        idempotency_key=new_attempt_idempotency_key("idem:doep-043", 3),
        attempt=3,
    )
    settled = retry_gateway.execute(new_attempt)
    assert settled.phase is ProviderExecutionPhase.SETTLED
    assert retry_calls["n"] == 2


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-043"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["binding"] == (
        UNKNOWN_PROVIDER_OUTCOME_RECONCILIATION_BINDING
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(EXECUTION_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
