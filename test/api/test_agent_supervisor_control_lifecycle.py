from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlContractError,
    EffectKind,
    ControlMutationGuardEvidence,
    ExpectedEffect,
    IdempotencyKey,
    LifecycleAction,
    LifecycleCommand,
    MutationGuardRejection,
    MutationGuardExecutionObservation,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": "ASI-G103",
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    dry_run: bool,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    effect = ExpectedEffect(
        effect_id=f"{operation.value}:supervisor",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:fixture",
        paths=("supervisor.json",),
        description=f"Transition with {operation.value}",
    )
    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "parameters": {
            "target_id": "supervisor:fixture",
            "reason": "operator maintenance",
            "requested_state": operation.value,
        },
        "expected_effects": (effect,),
        "dry_run": dry_run,
    }
    if not dry_run:
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=f"lifecycle:{operation.value}:1",
                    operation=operation,
                    caller=binding["caller"],
                    repository_id=binding["repository_id"],
                    objective_id=binding["objective_id"],
                ),
                "authorization": AuthorizationDecision(
                    verdict=AuthorizationVerdict.PERMIT,
                    operation=operation,
                    granted_authority=OperationAuthority.MUTATION,
                    **binding,
                    lease_id="lease:7",
                    fencing_epoch=7,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("grant:operator",),
                    evaluated_at_ms=1_000,
                    expires_at_ms=2_000,
                ),
                "lease_id": "lease:7",
                "fencing_epoch": 7,
            }
        )
    return OperationRequest(**values)


def _command(operation: Operation, *, dry_run: bool) -> LifecycleCommand:
    return LifecycleCommand(
        action=LifecycleAction(operation.value),
        target_id="supervisor:fixture",
        reason="operator maintenance",
        requested_state=operation.value,
        dry_run=dry_run,
    )


def _service(
    repo_root: Path,
    state_root: Path,
    calls: list[str],
) -> SupervisorControlService:
    def transition(request: OperationRequest) -> BackendResponse:
        calls.append(request.request_id)
        effect_id = request.expected_effects[0].effect_id
        return BackendResponse(
            data={
                "previous_state": "healthy",
                "state": request.operation.value,
            },
            changed=True,
            applied_effect_ids=(effect_id,),
        )

    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            operation: transition
            for operation in (
                Operation.START,
                Operation.PAUSE,
                Operation.RESUME,
                Operation.DRAIN,
                Operation.STOP,
            )
        },
        lease_validator=lambda request: (
            request.lease_id == "lease:7" and request.fencing_epoch == 7
        ),
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )


def test_lifecycle_dry_run_binds_typed_command_without_dispatch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=True
    )

    result = service.lifecycle(request, _command(Operation.PAUSE, dry_run=True))

    assert result.succeeded
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.would_change is True
    assert calls == []


def test_lifecycle_command_binds_reason_and_requested_state(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root, [])
    request = _request(
        repo_root, state_root, Operation.DRAIN, dry_run=True
    )

    with pytest.raises(ValueError, match="reason"):
        service.lifecycle(
            request,
            LifecycleCommand(
                action=LifecycleAction.DRAIN,
                target_id="supervisor:fixture",
                reason="different reason",
                requested_state="drain",
                dry_run=True,
            ),
        )
    with pytest.raises(ValueError, match="requested_state"):
        service.lifecycle(
            request,
            LifecycleCommand(
                action=LifecycleAction.DRAIN,
                target_id="supervisor:fixture",
                reason="operator maintenance",
                requested_state="paused",
                dry_run=True,
            ),
        )


def test_authorized_lifecycle_mutation_is_fenced_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=False
    )
    command = _command(Operation.PAUSE, dry_run=False)

    first = service.lifecycle(request, command)
    replay = service.lifecycle(request, command)

    assert first is replay
    assert first.status is OperationStatus.SUCCEEDED
    assert first.data == {
        "previous_state": "healthy",
        "state": "pause",
    }
    assert first.effects[0].applied is True
    assert first.audit_receipt_id
    assert calls == [request.request_id]


def test_mutation_guard_evidence_replays_all_required_fail_closed_cases(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=False
    )
    before = service.mutation_runtime_state()
    result = service.execute(request)
    after_result = service.mutation_runtime_state()
    replay = service.execute(request)
    after_replay = service.mutation_runtime_state()
    canonical = request.to_record()

    def rejected(
        scenario: str, *removed: str, error_type: str
    ) -> MutationGuardRejection:
        payload = dict(canonical)
        payload.pop("content_id", None)
        for name in removed:
            payload.pop(name, None)
        return MutationGuardRejection(
            scenario=scenario,
            request_payload=payload,
            error_type=error_type,
        )

    evidence = ControlMutationGuardEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        request=request,
        result=result,
        replay_result=replay,
        execution=MutationGuardExecutionObservation(
            request_id=request.request_id,
            result_id=result.result_id,
            audit_receipt_id=result.audit_receipt_id,
            before=before,
            after_result=after_result,
            after_replay=after_replay,
        ),
        rejections=(
            rejected(
                "missing_authorization",
                "authorization",
                error_type="AuthorizationBindingError",
            ),
            rejected(
                "missing_idempotency",
                "idempotency",
                error_type="MissingIdempotencyError",
            ),
            rejected(
                "missing_lease_or_fence",
                "lease_id",
                "fencing_epoch",
                error_type="AuthorizationBindingError",
            ),
        ),
    )

    assert evidence.proved_requirement_ids == (
        CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
    )
    assert calls == [request.request_id]
    assert ControlMutationGuardEvidence.from_dict(evidence.to_record()) == evidence

    for field, message in (
        ("result_id", "detached"),
        ("audit_receipt_id", "bound audit receipt"),
    ):
        tampered = evidence.to_record()
        tampered.pop("content_id")
        execution = dict(tampered["execution"])
        execution.pop("content_id")
        execution[field] = "sha256:" + ("0" * 64)
        tampered["execution"] = execution
        with pytest.raises(ControlContractError, match=message):
            ControlMutationGuardEvidence.from_dict(tampered)

    duplicate_dispatch = evidence.to_record()
    duplicate_dispatch.pop("content_id")
    execution = dict(duplicate_dispatch["execution"])
    execution.pop("content_id")
    after_replay_record = dict(execution["after_replay"])
    after_replay_record.pop("content_id")
    after_replay_record["dispatch_count"] = 2
    execution["after_replay"] = after_replay_record
    duplicate_dispatch["execution"] = execution
    with pytest.raises(ControlContractError, match="must not dispatch"):
        ControlMutationGuardEvidence.from_dict(duplicate_dispatch)

    mismatched_receipt = result.to_record()
    mismatched_receipt.pop("content_id")
    effect = dict(mismatched_receipt["effects"][0])
    effect.pop("content_id")
    effect["receipt_id"] = "sha256:" + ("f" * 64)
    mismatched_receipt["effects"] = [effect]
    with pytest.raises(ControlContractError, match="must match"):
        OperationResult.from_dict(mismatched_receipt)


def test_lifecycle_rejects_non_lifecycle_operation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root, [])
    request = OperationRequest(
        operation=Operation.STATUS,
        **_binding(repo_root, state_root),
    )

    with pytest.raises(ValueError, match="not a lifecycle"):
        service.lifecycle(request)
