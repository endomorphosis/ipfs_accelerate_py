from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.authorization_logic import (
    ControlMutationAuthorizer,
    ControlMutationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    JsonlControlStateStore,
    MutationRecoveryAction,
    MutationTransactionPhase,
    PartialMutationError,
    SupervisorControlService,
    TransactionConflictError,
)

NOW = 1_500


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:test",
        "tree_id": "tree:current",
        "objective_id": "ASI-116",
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _request(
    repository_root: Path,
    state_root: Path,
    *,
    key: str = "pause:one",
    effect_id: str = "pause:supervisor",
    dry_run: bool = False,
    tree_id: str = "tree:current",
    lease_id: str = "lease:7",
    fencing_epoch: int = 7,
) -> OperationRequest:
    binding = _binding(repository_root, state_root)
    binding["tree_id"] = tree_id
    effect = ExpectedEffect(
        effect_id=effect_id,
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:test",
        paths=("supervisor/status.json",),
        description="Pause the test supervisor",
    )
    values: dict[str, Any] = {
        "operation": Operation.PAUSE,
        **binding,
        "parameters": {"target_id": "supervisor:test"},
        "expected_effects": (effect,),
        "dry_run": dry_run,
    }
    if not dry_run:
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=key,
                    operation=Operation.PAUSE,
                    caller=binding["caller"],
                    repository_id=binding["repository_id"],
                    objective_id=binding["objective_id"],
                ),
                "authorization": AuthorizationDecision(
                    verdict=AuthorizationVerdict.PERMIT,
                    operation=Operation.PAUSE,
                    granted_authority=OperationAuthority.MUTATION,
                    **binding,
                    lease_id=lease_id,
                    fencing_epoch=fencing_epoch,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("policy-grant:pause",),
                    evaluated_at_ms=1_000,
                    expires_at_ms=2_000,
                ),
                "lease_id": lease_id,
                "fencing_epoch": fencing_epoch,
            }
        )
    return OperationRequest(**values)


def _policy(*requests: OperationRequest) -> ControlMutationPolicy:
    decisions = tuple(
        request.authorization
        for request in requests
        if request.authorization is not None
    )
    return ControlMutationPolicy(
        policy_id="policy:control",
        policy_revision="policy:1",
        permits=decisions,
        current_tree_ids={"repository:test": "tree:current"},
        current_objective_revisions={"ASI-116": "objective:1"},
        active_lease_fences={"lease:7": 7},
    )


class _Backend:
    registered_operations = (Operation.PAUSE,)

    def __init__(self, *, partial: bool = False) -> None:
        self.calls = 0
        self.recoveries: list[str] = []
        self.partial = partial

    def execute(self, request: OperationRequest) -> BackendResponse:
        self.calls += 1
        effect_id = request.expected_effects[0].effect_id
        if self.partial:
            raise PartialMutationError(
                "second mutation step failed",
                applied_effect_ids=(effect_id,),
                recovery=MutationRecoveryAction.COMPENSATE,
            )
        return BackendResponse(
            data={"state": "paused"},
            changed=True,
            applied_effect_ids=(effect_id,),
        )

    def compensate(
        self,
        request: OperationRequest,
        transaction: Any,
    ) -> bool:
        self.recoveries.append(
            f"{request.request_id}:{transaction.transaction_id}"
        )
        return True

    def repair(
        self,
        request: OperationRequest,
        transaction: Any,
    ) -> bool:
        self.recoveries.append(
            f"repair:{request.request_id}:{transaction.transaction_id}"
        )
        return True


def _service(
    repository_root: Path,
    state_root: Path,
    request: OperationRequest,
    backend: _Backend,
    *,
    store: InMemoryControlStateStore | None = None,
    policy: ControlMutationPolicy | None = None,
) -> SupervisorControlService:
    policy = policy or _policy(request)
    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        backend=backend,
        authorization_validator=ControlMutationAuthorizer(
            policy, clock_ms=lambda: NOW
        ),
        identity_validator=lambda candidate: (
            policy.current_tree_ids.get(candidate.repository_id)
            == candidate.tree_id
            and policy.current_objective_revisions.get(candidate.objective_id)
            == candidate.objective_revision
        ),
        lease_validator=lambda candidate: (
            policy.active_lease_fences.get(candidate.lease_id)
            == candidate.fencing_epoch
        ),
        state_store=store or InMemoryControlStateStore(),
        clock_ms=lambda: NOW,
    )


def test_real_mutation_requires_a_policy_issued_exact_permit(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    authorized = _request(repository_root, state_root)
    counterfeit = _request(
        repository_root,
        state_root,
        key="pause:counterfeit",
        effect_id="pause:counterfeit",
    )
    backend = _Backend()
    service = _service(
        repository_root,
        state_root,
        authorized,
        backend,
        policy=_policy(authorized),
    )

    denied = service.execute(counterfeit)
    accepted = service.execute(authorized)

    assert denied.status is OperationStatus.DENIED
    assert denied.error and denied.error.code is ErrorCode.UNAUTHORIZED
    assert accepted.status is OperationStatus.SUCCEEDED
    assert backend.calls == 1
    transaction = service.mutation_transaction(authorized)
    assert transaction is not None
    assert transaction.phase is MutationTransactionPhase.COMMITTED
    assert transaction.revision == 2
    assert transaction.result == accepted


def test_dry_run_has_proposal_authority_and_never_dispatches_or_reserves(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    dry_run = _request(repository_root, state_root, dry_run=True)
    real = _request(repository_root, state_root)
    backend = _Backend()
    service = _service(repository_root, state_root, real, backend)

    result = service.execute(dry_run)

    assert result.status is OperationStatus.SUCCEEDED
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.expected_effects == dry_run.expected_effects
    assert not result.effects
    assert backend.calls == 0
    with pytest.raises(ValueError, match="real mutations"):
        service.mutation_transaction(dry_run)


def test_durable_result_replay_survives_restart_and_changed_effects_conflict(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(repository_root, state_root)
    first_backend = _Backend()
    first = _service(
        repository_root,
        state_root,
        request,
        first_backend,
        store=JsonlControlStateStore(),
    ).execute(request)

    second_backend = _Backend()
    restarted = _service(
        repository_root,
        state_root,
        request,
        second_backend,
        store=JsonlControlStateStore(),
    )
    replay = restarted.execute(request)
    changed = _request(
        repository_root,
        state_root,
        key=request.idempotency_key,
        effect_id="pause:different-effects",
    )
    changed_service = _service(
        repository_root,
        state_root,
        changed,
        second_backend,
        store=JsonlControlStateStore(),
        policy=_policy(request, changed),
    )
    conflict = changed_service.execute(changed)

    assert replay == first
    assert second_backend.calls == 0
    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error
    assert conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert second_backend.calls == 0


def test_transaction_compare_and_swap_rejects_stale_revision(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(repository_root, state_root)
    store = InMemoryControlStateStore()
    prepared = store.begin_mutation(request, now_ms=NOW)
    dispatching = store.compare_and_swap_mutation(
        request,
        expected_revision=prepared.revision,
        phase=MutationTransactionPhase.DISPATCHING,
        now_ms=NOW,
    )

    with pytest.raises(TransactionConflictError, match="stale transaction revision"):
        store.compare_and_swap_mutation(
            request,
            expected_revision=prepared.revision,
            phase=MutationTransactionPhase.REPAIR_REQUIRED,
            failure_code="conflict",
            now_ms=NOW,
        )

    assert dispatching.revision == 1
    assert store.get_mutation(request) == dispatching


def test_partial_failure_is_durable_replayable_and_typed_for_compensation(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(repository_root, state_root)
    backend = _Backend(partial=True)
    service = _service(repository_root, state_root, request, backend)

    failed = service.execute(request)
    replay = service.execute(request)
    transaction = service.mutation_transaction(request)

    assert failed.status is OperationStatus.CONFLICT
    assert replay == failed
    assert backend.calls == 1
    assert transaction is not None
    assert transaction.phase is MutationTransactionPhase.COMPENSATION_REQUIRED
    assert transaction.recovery_action is MutationRecoveryAction.COMPENSATE
    assert transaction.applied_effect_ids == ("pause:supervisor",)
    assert transaction.result == failed
    assert failed.data["transaction"]["recovery_action"] == "compensate"

    recovered = service.recover_mutation(
        request,
        expected_revision=transaction.revision,
        action=MutationRecoveryAction.COMPENSATE,
    )

    assert recovered.phase is MutationTransactionPhase.COMPENSATED
    assert recovered.revision == transaction.revision + 1
    assert recovered.applied_effect_ids == ()
    assert len(backend.recoveries) == 1
    with pytest.raises(TransactionConflictError, match="stale transaction revision"):
        service.recover_mutation(
            request,
            expected_revision=transaction.revision,
            action=MutationRecoveryAction.COMPENSATE,
        )


def test_restart_turns_an_unknown_dispatch_outcome_into_typed_repair(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(repository_root, state_root)
    initial_store = JsonlControlStateStore()
    with initial_store.transaction(request):
        prepared = initial_store.begin_mutation(request, now_ms=NOW)
        initial_store.compare_and_swap_mutation(
            request,
            expected_revision=prepared.revision,
            phase=MutationTransactionPhase.DISPATCHING,
            now_ms=NOW,
        )

    backend = _Backend()
    restarted = _service(
        repository_root,
        state_root,
        request,
        backend,
        store=JsonlControlStateStore(),
    )
    failed = restarted.execute(request)
    transaction = restarted.mutation_transaction(request)

    assert failed.status is OperationStatus.CONFLICT
    assert failed.error and failed.error.code is ErrorCode.CONFLICT
    assert backend.calls == 0
    assert transaction is not None
    assert transaction.phase is MutationTransactionPhase.REPAIR_REQUIRED
    assert transaction.recovery_action is MutationRecoveryAction.REPAIR
    repaired = restarted.recover_mutation(
        request,
        expected_revision=transaction.revision,
        action=MutationRecoveryAction.REPAIR,
    )
    assert repaired.phase is MutationTransactionPhase.REPAIRED
    assert backend.recoveries == [
        f"repair:{request.request_id}:{transaction.transaction_id}"
    ]


def test_stale_policy_targets_and_lease_loss_reject_before_dispatch(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(repository_root, state_root)
    backend = _Backend()
    stale_tree_policy = replace(
        _policy(request),
        current_tree_ids={"repository:test": "tree:new"},
    )
    stale_lease_policy = replace(
        _policy(request),
        active_lease_fences={"lease:7": 8},
    )

    stale_tree = _service(
        repository_root,
        state_root,
        request,
        backend,
        policy=stale_tree_policy,
    ).execute(request)
    stale_lease = _service(
        repository_root,
        state_root,
        request,
        backend,
        policy=stale_lease_policy,
    ).execute(request)

    assert stale_tree.status is OperationStatus.CONFLICT
    assert stale_tree.error and stale_tree.error.code is ErrorCode.STALE_TREE
    assert stale_lease.status is OperationStatus.DENIED
    assert stale_lease.error and stale_lease.error.code is ErrorCode.UNAUTHORIZED
    assert backend.calls == 0
