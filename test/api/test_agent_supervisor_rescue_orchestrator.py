from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    IncidentKind,
    ProgrammaticRecoveryExhaustionReceipt,
    PromptWorkflowBudget,
    RecoveryAttempt,
    RecoveryAttemptOutcome,
    RescueAction,
    RescueOperation,
    RescuePlan,
    SupervisorIncident,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.rescue_orchestrator import (
    REQUIRED_AUTHORIZATION_DOMAINS,
    RESCUE_ORCHESTRATION_REQUIREMENT_ID,
    RescueAuthorizationDomain,
    RescueAuthorizationReceipt,
    RescueAuthorizationVerdict,
    RescueEffectObservation,
    RescueExecutionBudget,
    RescueExecutionRequest,
    RescueHealthReceipt,
    RescueHealthState,
    RescueOrchestrationError,
    RescueOrchestrator,
    RescuePermitUseReceipt,
    RescueReceiptDisposition,
    RescueRootBinding,
    RescueRuntimeSnapshot,
    RescueSimulatedEffect,
    RescueSimulationReceipt,
    RescueStopReason,
)
from ipfs_accelerate_py.agent_supervisor.rescue_planner import (
    DEFAULT_RESCUE_OPERATION_CATALOG,
)


NOW = 50_000


def _cid(name: str) -> str:
    return prompt_workflow_cid({"rescue-orchestrator-fixture": name})


def _incident(
    *,
    target_ids: tuple[str, ...] = ("lane:implementation",),
    repository_root_cid: str | None = None,
) -> SupervisorIncident:
    return SupervisorIncident(
        repository_root="/workspace/repository",
        state_root="/workspace/repository/state/supervisor",
        repository_root_cid=repository_root_cid or _cid("repository"),
        policy_root=_cid("policy"),
        run_cid=_cid("run"),
        kind=IncidentKind.STALE_HEARTBEAT,
        failure_fingerprint="sha256:" + "a" * 64,
        target_ids=target_ids,
        evidence_cids=(_cid("health"),),
        health={"state": "unhealthy"},
        cooldown_key="rescue/lane",
        observed_at_ms=NOW,
        updated_at_ms=NOW,
    )


def _exhaustion(
    incident: SupervisorIncident,
) -> ProgrammaticRecoveryExhaustionReceipt:
    return ProgrammaticRecoveryExhaustionReceipt(
        incident_cid=incident.incident_cid,
        repository_root_cid=incident.repository_root_cid,
        policy_root=incident.policy_root,
        run_cid=incident.run_cid,
        attempts=(
            RecoveryAttempt(
                operation=RescueOperation.RESTART_LANE,
                target_id=incident.target_ids[0],
                attempt=1,
                outcome=RecoveryAttemptOutcome.FAILED,
                receipt_cid=_cid("failed-restart"),
                failure_fingerprint="sha256:" + "b" * 64,
            ),
        ),
        inapplicable_operations=(RescueOperation.REPAIR_ORPHANED_LOCK,),
        exhaustion_reason="deterministic recovery exhausted",
        budget=PromptWorkflowBudget(
            max_prompt_tokens=16_384,
            max_provider_tokens=4_096,
            max_latency_ms=60_000,
            max_rescue_actions=4,
        ),
        circuit_open=False,
        created_at_ms=NOW,
        updated_at_ms=NOW,
    )


def _action(
    operation: RescueOperation = RescueOperation.RESTART_LANE,
    *,
    target_id: str = "lane:implementation",
) -> RescueAction:
    spec = DEFAULT_RESCUE_OPERATION_CATALOG[operation]
    parameters: dict[str, Any] = {}
    if operation is RescueOperation.RESTART_LANE:
        parameters["grace_period_ms"] = 1_000
    elif operation is RescueOperation.RETRY:
        parameters["attempt_limit"] = 1
    return RescueAction(
        operation=operation,
        target_id=target_id,
        parameters=parameters,
        precondition_cids=(_cid("precondition"),),
        expected_effects=spec.expected_effects,
        success_test=spec.success_test,
        stop_condition=spec.stop_condition,
    )


def _roots(
    incident: SupervisorIncident,
    exhaustion: ProgrammaticRecoveryExhaustionReceipt,
    **changes: str,
) -> RescueRootBinding:
    values = {
        "incident_cid": incident.incident_cid,
        "exhaustion_receipt_cid": exhaustion.receipt_cid,
        "request_root": _cid("request"),
        "program_root": _cid("program"),
        "repository_root_cid": incident.repository_root_cid,
        "tree_id": _cid("tree"),
        "run_cid": incident.run_cid,
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "policy_root": incident.policy_root,
        "catalog_root": _cid("catalog"),
    }
    values.update(changes)
    return RescueRootBinding(**values)


def _request(
    *,
    actions: tuple[RescueAction, ...] | None = None,
    budget: RescueExecutionBudget | None = None,
    model_tokens: int = 10,
) -> RescueExecutionRequest:
    incident = _incident()
    exhaustion = _exhaustion(incident)
    roots = _roots(incident, exhaustion)
    selected = actions or (_action(),)
    plan = RescuePlan(
        incident_cid=incident.incident_cid,
        exhaustion_receipt_cid=exhaustion.receipt_cid,
        repository_root_cid=incident.repository_root_cid,
        run_cid=incident.run_cid,
        policy_root=incident.policy_root,
        actions=selected,
        rationale_reference_cids=incident.evidence_cids,
        unresolved_risks=("The lane may remain unhealthy.",),
        max_actions=max(1, len(selected)),
    )
    return RescueExecutionRequest(
        plan=plan,
        incident=incident,
        exhaustion_receipt=exhaustion,
        roots=roots,
        lease_id="lease:active",
        fencing_epoch=7,
        idempotency_scope="rescue:" + plan.rescue_plan_cid,
        budget=budget or RescueExecutionBudget(),
        model_tokens=model_tokens,
    )


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> int:
        self.now += 1
        return self.now


class State:
    def __init__(self, request: RescueExecutionRequest) -> None:
        self.snapshots = [
            RescueRuntimeSnapshot(
                roots=request.roots,
                lease_id=request.lease_id,
                fencing_epoch=request.fencing_epoch,
                revision=1,
                observed_at_ms=NOW,
            )
        ]
        self.calls = 0

    def snapshot(self) -> RescueRuntimeSnapshot:
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return self.snapshots[index]


class Simulator:
    def __init__(self) -> None:
        self.calls = 0

    def simulate(
        self, action: RescueAction, roots: RescueRootBinding, now_ms: int
    ) -> RescueSimulationReceipt:
        self.calls += 1
        return RescueSimulationReceipt(
            action_content_id=action.content_id,
            root_binding_id=roots.content_id,
            effects=tuple(
                RescueSimulatedEffect(
                    effect_id=f"effect:{self.calls}:{index}",
                    effect=effect,
                    target_id=action.target_id,
                )
                for index, effect in enumerate(action.expected_effects)
            ),
            simulator_id="deterministic-recovery-simulator",
            simulated_at_ms=now_ms,
        )


class Authorizer:
    def __init__(
        self,
        domain: RescueAuthorizationDomain,
        verdict: RescueAuthorizationVerdict = (
            RescueAuthorizationVerdict.PERMIT
        ),
    ) -> None:
        self.domain = domain
        self.verdict = verdict
        self.calls = 0

    def authorize(self, binding, now_ms: int) -> RescueAuthorizationReceipt:
        self.calls += 1
        return RescueAuthorizationReceipt(
            domain=self.domain,
            verdict=self.verdict,
            binding_id=binding.binding_id,
            root_binding_id=binding.roots.content_id,
            authority_id=f"authority:{self.domain.value}",
            reason_code=(
                "" if self.verdict is RescueAuthorizationVerdict.PERMIT
                else "explicit_denial"
            ),
            evaluated_at_ms=now_ms,
            expires_at_ms=now_ms + 1_000,
        )


def _authorizers(
    *,
    denied: RescueAuthorizationDomain | None = None,
) -> dict[RescueAuthorizationDomain, Authorizer]:
    return {
        domain: Authorizer(
            domain,
            (
                RescueAuthorizationVerdict.DENY
                if domain is denied
                else RescueAuthorizationVerdict.PERMIT
            ),
        )
        for domain in REQUIRED_AUTHORIZATION_DOMAINS
    }


class PermitBoundary:
    def __init__(self, *, changed_incident: bool = False) -> None:
        self.calls = 0
        self.changed_incident = changed_incident

    def issue_and_consume(
        self,
        binding,
        authorizations,
        snapshot,
        issued_at_ms: int,
        expires_at_ms: int,
    ) -> RescuePermitUseReceipt:
        self.calls += 1
        assert {item.domain for item in authorizations} == set(
            REQUIRED_AUTHORIZATION_DOMAINS
        )
        return RescuePermitUseReceipt(
            permit_id=f"permit:{self.calls}:{binding.binding_id}",
            binding_id=binding.binding_id,
            root_binding_id=binding.roots.content_id,
            incident_cid=(
                _cid("other-incident")
                if self.changed_incident
                else binding.roots.incident_cid
            ),
            lease_id=binding.lease_id,
            fencing_epoch=binding.fencing_epoch,
            idempotency_key=binding.idempotency_key,
            issued_at_ms=issued_at_ms,
            expires_at_ms=expires_at_ms,
            consumed_at_ms=issued_at_ms,
        )


class Transaction:
    def __init__(self, *, partial: bool = False, unexpected: bool = False) -> None:
        self.calls = 0
        self.partial = partial
        self.unexpected = unexpected

    def execute(self, binding, permit, control_request):
        self.calls += 1
        effects = binding.simulation.effects
        if self.partial:
            effects = ()
        elif self.unexpected:
            effects = (
                RescueSimulatedEffect(
                    effect_id="effect:unexpected",
                    effect="undeclared_effect",
                    target_id=binding.action.target_id,
                ),
            )
        return RescueEffectObservation(
            effects=effects,
            transaction_receipt_id=f"control-transaction:{self.calls}",
            complete=not self.partial,
        )


class Health:
    def __init__(self, request: RescueExecutionRequest, *states) -> None:
        self.request = request
        self.states = list(states or (RescueHealthState.UNHEALTHY,))
        self.calls = 0

    def test(self, binding, now_ms: int) -> RescueHealthReceipt:
        index = min(self.calls, len(self.states) - 1)
        self.calls += 1
        return RescueHealthReceipt(
            state=self.states[index],
            incident_cid=self.request.roots.incident_cid,
            root_binding_id=self.request.roots.content_id,
            health_test_id=f"health-test:{self.calls}",
            evidence_ids=(_cid(f"health-{self.calls}"),),
            checked_at_ms=now_ms,
        )


def _orchestrator(
    request: RescueExecutionRequest,
    *,
    state: State | None = None,
    authorizers=None,
    permit=None,
    transaction=None,
    health=None,
):
    dependencies = {
        "state_provider": state or State(request),
        "simulator": Simulator(),
        "authorizers": authorizers or _authorizers(),
        "permit_boundary": permit or PermitBoundary(),
        "control_transaction": transaction or Transaction(),
        "health_tester": health
        or Health(
            request,
            RescueHealthState.UNHEALTHY,
            RescueHealthState.HEALTHY,
        ),
        "clock_ms": Clock(),
    }
    return RescueOrchestrator(**dependencies), dependencies


def test_rebinds_all_roots_and_executes_with_five_independent_checks() -> None:
    request = _request()
    orchestrator, deps = _orchestrator(request)

    receipt = orchestrator.execute(request)

    assert receipt.recovered
    assert receipt.stop_reason is RescueStopReason.HEALTH_RESTORED
    assert receipt.disposition is RescueReceiptDisposition.RECOVERED
    assert len(receipt.action_receipts) == 1
    action = receipt.action_receipts[0]
    assert {item.domain for item in action.authorization_receipts} == set(
        REQUIRED_AUTHORIZATION_DOMAINS
    )
    assert all(item.admitted for item in action.authorization_receipts)
    assert action.permit_use_receipt is not None
    assert action.permit_use_receipt.remaining_uses == 0
    assert not action.permit_use_receipt.grants_completion_authority
    assert action.transaction_receipt_id == "control-transaction:1"
    assert action.health_receipt is not None
    assert action.health_receipt.state is RescueHealthState.HEALTHY
    assert receipt.to_dict()["requirement_id"] == (
        RESCUE_ORCHESTRATION_REQUIREMENT_ID
    )
    assert deps["permit_boundary"].calls == 1
    assert deps["control_transaction"].calls == 1


def test_each_action_gets_one_exact_permit_and_stops_as_soon_as_healthy() -> None:
    request = _request(actions=(_action(), _action()))
    health = Health(
        request,
        RescueHealthState.UNHEALTHY,
        RescueHealthState.UNHEALTHY,
        RescueHealthState.UNHEALTHY,
        RescueHealthState.HEALTHY,
    )
    permit = PermitBoundary()
    transaction = Transaction()
    orchestrator, _deps = _orchestrator(
        request,
        permit=permit,
        transaction=transaction,
        health=health,
    )

    receipt = orchestrator.execute(request)

    assert receipt.stop_reason is RescueStopReason.HEALTH_RESTORED
    assert [item.stop_reason for item in receipt.action_receipts] == [
        RescueStopReason.ACTION_APPLIED,
        RescueStopReason.HEALTH_RESTORED,
    ]
    assert permit.calls == transaction.calls == 2
    permit_ids = {
        item.permit_use_receipt.permit_id
        for item in receipt.action_receipts
        if item.permit_use_receipt
    }
    assert len(permit_ids) == 2


def test_root_incident_lease_fence_cooldown_and_model_budgets_fail_closed() -> None:
    request = _request()
    cases: list[tuple[RescueRuntimeSnapshot, RescueStopReason]] = [
        (
            RescueRuntimeSnapshot(
                roots=replace(
                    request.roots, program_root=_cid("changed-program")
                ),
                lease_id=request.lease_id,
                fencing_epoch=request.fencing_epoch,
            ),
            RescueStopReason.ROOT_DRIFT,
        ),
        (
            RescueRuntimeSnapshot(
                roots=replace(
                    request.roots, incident_cid=_cid("changed-incident")
                ),
                lease_id=request.lease_id,
                fencing_epoch=request.fencing_epoch,
            ),
            RescueStopReason.INCIDENT_DRIFT,
        ),
        (
            RescueRuntimeSnapshot(
                roots=request.roots,
                lease_id="lease:other",
                fencing_epoch=request.fencing_epoch,
            ),
            RescueStopReason.LEASE_LOST,
        ),
        (
            RescueRuntimeSnapshot(
                roots=request.roots,
                lease_id=request.lease_id,
                fencing_epoch=request.fencing_epoch + 1,
            ),
            RescueStopReason.FENCE_LOST,
        ),
        (
            RescueRuntimeSnapshot(
                roots=request.roots,
                lease_id=request.lease_id,
                fencing_epoch=request.fencing_epoch,
                cooldown_until_ms=NOW + 100_000,
            ),
            RescueStopReason.COOLDOWN_ACTIVE,
        ),
    ]
    for snapshot, expected in cases:
        state = State(request)
        state.snapshots = [snapshot]
        permit = PermitBoundary()
        transaction = Transaction()
        orchestrator, _ = _orchestrator(
            request,
            state=state,
            permit=permit,
            transaction=transaction,
        )
        receipt = orchestrator.execute(request)
        assert receipt.stop_reason is expected
        assert permit.calls == transaction.calls == 0

    over_budget = _request(
        budget=RescueExecutionBudget(max_model_tokens=10),
        model_tokens=11,
    )
    orchestrator, deps = _orchestrator(over_budget)
    result = orchestrator.execute(over_budget)
    assert result.stop_reason is RescueStopReason.MODEL_BUDGET
    assert deps["permit_boundary"].calls == 0


def test_drift_between_authorization_and_effect_prevents_permit_and_dispatch() -> None:
    request = _request()
    state = State(request)
    state.snapshots = [
        state.snapshots[0],
        replace(
            state.snapshots[0],
            roots=replace(
                request.roots, security_ir_root=_cid("changed-security")
            ),
        ),
    ]
    permit = PermitBoundary()
    transaction = Transaction()
    orchestrator, _ = _orchestrator(
        request,
        state=state,
        permit=permit,
        transaction=transaction,
    )

    receipt = orchestrator.execute(request)

    assert receipt.stop_reason is RescueStopReason.ROOT_DRIFT
    assert permit.calls == transaction.calls == 0


def test_any_independent_denial_stops_before_permit() -> None:
    request = _request()
    authorizers = _authorizers(denied=RescueAuthorizationDomain.LEGAL)
    permit = PermitBoundary()
    transaction = Transaction()
    orchestrator, _ = _orchestrator(
        request,
        authorizers=authorizers,
        permit=permit,
        transaction=transaction,
    )

    receipt = orchestrator.execute(request)

    assert receipt.stop_reason is RescueStopReason.AUTHORIZATION_DENIED
    assert receipt.disposition is RescueReceiptDisposition.DENIED
    assert permit.calls == transaction.calls == 0
    assert authorizers[RescueAuthorizationDomain.INTENT].calls == 1
    assert authorizers[RescueAuthorizationDomain.LEGAL].calls == 1
    assert authorizers[RescueAuthorizationDomain.SECURITY].calls == 0


@pytest.mark.parametrize(
    ("transaction", "reason", "disposition"),
    [
        (
            Transaction(partial=True),
            RescueStopReason.PARTIAL_EFFECT,
            RescueReceiptDisposition.PARTIAL,
        ),
        (
            Transaction(unexpected=True),
            RescueStopReason.UNEXPECTED_EFFECT,
            RescueReceiptDisposition.QUARANTINED,
        ),
    ],
)
def test_partial_and_unexpected_effects_have_exact_recovery_receipts(
    transaction: Transaction,
    reason: RescueStopReason,
    disposition: RescueReceiptDisposition,
) -> None:
    request = _request()
    orchestrator, _ = _orchestrator(request, transaction=transaction)

    receipt = orchestrator.execute(request)

    action = receipt.action_receipts[0]
    assert action.stop_reason is reason
    assert action.disposition is disposition
    assert action.permit_use_receipt is not None
    assert action.transaction_receipt_id
    assert action.health_receipt is not None
    assert "quarantine_exact_incident_scope" in action.recovery_steps


def test_unknown_control_outcome_is_non_replayable_partial_with_health() -> None:
    request = _request()

    class FailingTransaction:
        calls = 0

        def execute(self, binding, permit, control_request):
            self.calls += 1
            raise RuntimeError("injected unknown outcome")

    transaction = FailingTransaction()
    orchestrator, _ = _orchestrator(request, transaction=transaction)

    first = orchestrator.execute(request)
    replay = orchestrator.execute(request)

    action = first.action_receipts[0]
    assert action.stop_reason is RescueStopReason.PARTIAL_EFFECT
    assert action.disposition is RescueReceiptDisposition.PARTIAL
    assert action.health_receipt is not None
    assert action.transaction_receipt_id.startswith("unknown-control-outcome:")
    assert replay.stop_reason is RescueStopReason.IDEMPOTENCY_REPLAY
    assert transaction.calls == 1


def test_replay_changed_permit_and_schema_drift_fail_closed() -> None:
    request = _request()
    orchestrator, deps = _orchestrator(request)
    first = orchestrator.execute(request)
    replay = orchestrator.execute(request)
    assert first.recovered
    assert replay.stop_reason is RescueStopReason.IDEMPOTENCY_REPLAY
    assert deps["control_transaction"].calls == 1

    cross_incident = PermitBoundary(changed_incident=True)
    second_orchestrator, second_deps = _orchestrator(
        request, permit=cross_incident
    )
    result = second_orchestrator.execute(request)
    assert result.stop_reason is RescueStopReason.PERMIT_DENIED
    assert second_deps["control_transaction"].calls == 0

    spec = DEFAULT_RESCUE_OPERATION_CATALOG[RescueOperation.RESTART_LANE]
    changed_action = replace(
        _action(),
        expected_effects=("model_invented_effect",),
        success_test=spec.success_test,
    )
    changed = _request(actions=(changed_action,))
    changed_orchestrator, changed_deps = _orchestrator(changed)
    denied = changed_orchestrator.execute(changed)
    assert denied.stop_reason is RescueStopReason.SCHEMA_DENIED
    assert changed_deps["permit_boundary"].calls == 0


def test_authorizers_must_be_complete_independent_and_non_model_authoritative() -> None:
    request = _request()
    incomplete = _authorizers()
    incomplete.pop(RescueAuthorizationDomain.PROOF)
    with pytest.raises(RescueOrchestrationError, match="exactly one"):
        _orchestrator(request, authorizers=incomplete)

    shared = Authorizer(RescueAuthorizationDomain.INTENT)
    not_independent = {
        domain: shared for domain in REQUIRED_AUTHORIZATION_DOMAINS
    }
    with pytest.raises(RescueOrchestrationError, match="independent"):
        _orchestrator(request, authorizers=not_independent)

    with pytest.raises(RescueOrchestrationError, match="cannot authorize"):
        RescueAuthorizationReceipt(
            domain=RescueAuthorizationDomain.CONTROL,
            verdict=RescueAuthorizationVerdict.PERMIT,
            binding_id=_cid("binding"),
            root_binding_id=_cid("roots"),
            authority_id="rescue-model:self-authorized",
            evaluated_at_ms=NOW,
            expires_at_ms=NOW + 1,
        )
