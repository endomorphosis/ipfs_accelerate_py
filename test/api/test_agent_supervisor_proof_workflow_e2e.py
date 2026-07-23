from __future__ import annotations

import threading
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    AttemptStatus,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofPlan,
    ProofPlanStep,
    ProofReceipt,
    ProofStage,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    MergeProofGateReceipt,
    ProofOutcome,
    ProofPolicyRule,
    ProofResultStatus,
    RiskLevel,
    RolloutMode,
)
from ipfs_accelerate_py.agent_supervisor.proof_scheduler import (
    ProofNodeState,
    ProofScheduler,
    ProofSchedulerConfig,
    ProofStepResult,
)


TREE = "git-tree:workflow-e2e"
BUDGET = ResourceBudget(
    wall_time_ms=30_000,
    cpu_time_ms=20_000,
    memory_bytes=512 * 1024 * 1024,
    max_processes=3,
)


def _step(
    step_id: str,
    stage: ProofStage,
    *,
    obligation: str | None = None,
    depends_on: tuple[str, ...] = (),
    resource_class: str = "",
) -> ProofPlanStep:
    return ProofPlanStep(
        step_id=step_id,
        obligation_id=obligation or f"obligation:{step_id}",
        stage=stage,
        provider_id=f"provider:{step_id}",
        depends_on=depends_on,
        resource_class=resource_class or stage.value,
    )


def _plan(
    *steps: ProofPlanStep,
    max_parallel: int = 3,
    policy_id: str = "policy:workflow",
) -> ProofPlan:
    return ProofPlan(
        repository_tree_id=TREE,
        obligation_ids=tuple(sorted({step.obligation_id for step in steps})),
        steps=steps,
        policy_id=policy_id,
        resource_budget=BUDGET,
        max_parallel=max_parallel,
    )


def _receipt(
    plan: ProofPlan,
    step: ProofPlanStep,
    *,
    kind: str = "proved",
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
) -> ProofReceipt:
    if kind == "proved":
        evidence = ProofEvidence(
            kind=EvidenceKind.KERNEL_VERIFICATION,
            authority=EvidenceAuthority.KERNEL,
            verdict=EvidenceVerdict.ACCEPTED,
            artifact_id=f"artifact:{step.step_id}",
            subject_id=step.obligation_id,
            verifier_id="kernel:lean",
            freshness=freshness,
            independent=True,
        )
        claimed = ProofVerdict.PROVED
    elif kind == "counterexample":
        evidence = ProofEvidence(
            kind=EvidenceKind.SOLVER_RESULT,
            authority=EvidenceAuthority.SOLVER,
            verdict=EvidenceVerdict.REJECTED,
            artifact_id=f"counterexample:{step.step_id}",
            subject_id=step.obligation_id,
            verifier_id="solver:z3",
            freshness=freshness,
            independent=True,
            metadata={"counterexample_verified": True},
        )
        claimed = ProofVerdict.DISPROVED
    else:
        evidence = ProofEvidence(
            kind=EvidenceKind.KERNEL_VERIFICATION,
            authority=EvidenceAuthority.KERNEL,
            verdict=EvidenceVerdict.REJECTED,
            artifact_id=f"rejection:{step.step_id}",
            subject_id=step.obligation_id,
            verifier_id="kernel:lean",
            freshness=freshness,
            independent=True,
            metadata={"failure_code": "proof_rejected"},
        )
        claimed = ProofVerdict.INCONCLUSIVE
    return ProofReceipt(
        obligation_id=step.obligation_id,
        plan_id=plan.plan_id,
        attempt_id=f"attempt:{step.step_id}",
        repository_id="repository:test",
        repository_tree_id=TREE,
        ast_scope_ids=(f"scope:{step.step_id}",),
        premise_ids=(),
        translator_id="translator:reviewed",
        solver_id="solver:z3",
        kernel_id="kernel:lean",
        toolchain_id="toolchain:locked",
        theorem_registry_id="registry:reviewed",
        policy_id=plan.policy_id,
        resource_budget=BUDGET,
        verdict=claimed,
        evidence=(evidence,),
        freshness=freshness,
    )


def test_workflow_fixture_matrix_preserves_truthful_outcomes(tmp_path: Path) -> None:
    steps = (
        _step("cache-hit", ProofStage.KERNEL_VERIFY),
        _step("proof-success", ProofStage.KERNEL_VERIFY),
        _step("counterexample", ProofStage.SOLVE),
        _step("unsupported-fallback", ProofStage.VALIDATE),
        _step("kernel-rejection", ProofStage.KERNEL_VERIFY),
        _step("provider-outage", ProofStage.SOLVE),
        _step("stale-evidence", ProofStage.KERNEL_VERIFY),
    )
    plan = _plan(*steps)
    by_id = {step.step_id: step for step in steps}

    def execute(context):
        step = context.step
        if step.step_id == "cache-hit":
            return ProofStepResult(
                receipts=(_receipt(plan, step),),
                metadata={"cache_outcome": "hit"},
            )
        if step.step_id == "proof-success":
            return _receipt(plan, step)
        if step.step_id == "counterexample":
            return _receipt(plan, step, kind="counterexample")
        if step.step_id == "unsupported-fallback":
            return ProofStepResult(
                status=AttemptStatus.UNSUPPORTED,
                error_code="explicit_fallback_required",
            )
        if step.step_id == "kernel-rejection":
            return ProofStepResult(
                status=AttemptStatus.FAILED,
                receipts=(_receipt(plan, step, kind="rejected"),),
                error_code="kernel_rejected",
            )
        if step.step_id == "provider-outage":
            return ProofStepResult(
                status=AttemptStatus.UNAVAILABLE,
                error_code="provider_outage",
            )
        return _receipt(
            plan,
            by_id["stale-evidence"],
            freshness=EvidenceFreshness.STALE,
        )

    result = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        max_parallel=3,
    ).run()

    assert result.states["cache-hit"] is ProofNodeState.SUCCEEDED
    assert result.states["proof-success"] is ProofNodeState.SUCCEEDED
    assert result.states["counterexample"] is ProofNodeState.SUCCEEDED
    assert result.states["unsupported-fallback"] is ProofNodeState.UNSUPPORTED
    assert result.states["kernel-rejection"] is ProofNodeState.FAILED
    assert result.states["provider-outage"] is ProofNodeState.UNSUPPORTED
    assert result.states["stale-evidence"] is ProofNodeState.SUCCEEDED
    authoritative = {
        receipt.authoritative_verdict
        for receipt in result.authoritative_receipts
    }
    assert authoritative == {ProofVerdict.PROVED, ProofVerdict.DISPROVED}
    stale = next(
        item for item in result.receipts
        if item.obligation_id == by_id["stale-evidence"].obligation_id
    )
    assert stale.authoritative_verdict is ProofVerdict.INCONCLUSIVE


def test_independent_lanes_share_global_budget_without_duplicate_work(
    tmp_path: Path,
) -> None:
    classes = (
        "cpu-proof-solver",
        "cpu-proof-kernel",
        "cpu-validation",
        "llm-proof-draft",
        "io-artifact",
    )
    steps = tuple(
        _step(
            f"lane-{index}",
            ProofStage.SOLVE,
            resource_class=resource_class,
        )
        for index, resource_class in enumerate(classes)
    )
    plan = _plan(*steps, max_parallel=3)
    lock = threading.Lock()
    active = 0
    peak = 0
    calls: dict[str, int] = {}

    def execute(context) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls[context.step_id] = calls.get(context.step_id, 0) + 1
        time.sleep(0.03)
        with lock:
            active -= 1

    result = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        config=ProofSchedulerConfig(max_parallel=3, lease_seconds=2),
    ).run()

    assert result.complete
    assert 1 < peak <= 3
    assert calls == {step.step_id: 1 for step in steps}
    assert result.snapshot.active_leases == 0
    assert len({attempt.attempt_id for attempt in result.attempts}) == len(
        result.attempts
    )


def test_restart_preserves_dependencies_receipts_and_exactly_once_ownership(
    tmp_path: Path,
) -> None:
    translate = _step("translate", ProofStage.TRANSLATE)
    solve = _step("solve", ProofStage.SOLVE, depends_on=("translate",))
    kernel = _step("kernel", ProofStage.KERNEL_VERIFY, depends_on=("solve",))
    plan = _plan(translate, solve, kernel)
    calls: dict[str, int] = {}

    def execute(context):
        calls[context.step_id] = calls.get(context.step_id, 0) + 1
        if context.step_id == "kernel":
            assert {
                item.step_id for item in context.dependency_attempts
            } == {"solve"}
            assert any(
                item.status is AttemptStatus.SUCCEEDED
                for item in context.dependency_attempts
            )
            return _receipt(plan, kernel)
        return None

    first = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
    ).run(stages=(ProofStage.TRANSLATE,))
    assert first.states["translate"] is ProofNodeState.SUCCEEDED
    assert first.states["solve"] in {ProofNodeState.PENDING, ProofNodeState.READY}

    restarted = ProofScheduler(
        executor=execute,
        state_path=tmp_path,
        plan_id=plan.plan_id,
    ).run()

    assert restarted.complete
    assert calls == {"translate": 1, "solve": 1, "kernel": 1}
    assert len(restarted.authoritative_receipts) == 1
    assert restarted.authoritative_receipts[0].plan_id == plan.plan_id
    assert restarted.snapshot.plan_id == first.snapshot.plan_id


def test_scheduler_receipt_flows_into_fail_closed_merge_gate(tmp_path: Path) -> None:
    step = _step("kernel", ProofStage.KERNEL_VERIFY, obligation="obligation:lease")
    policy = FormalVerificationPolicy(
        name="workflow-gate",
        version="1",
        rollout_mode=RolloutMode.ENFORCEMENT,
        rules=(
            ProofPolicyRule(
                rule_id="lease",
                path_patterns=("src/agent_supervisor/**",),
                minimum_risk=RiskLevel.HIGH,
                invariant_classes=("lease_safety",),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            ),
        ),
    )
    plan = _plan(step, policy_id=policy.policy_id)
    proof = _receipt(plan, step)
    scheduled = ProofScheduler(
        plan,
        lambda context: proof,
        state_path=tmp_path,
    ).run()
    selection = policy.select(
        (
            ChangedScope(
                path="src/agent_supervisor/lease.py",
                ast_scope_ids=("scope:kernel",),
                risk=RiskLevel.CRITICAL,
                invariant_classes=("lease_safety",),
            ),
        ),
        repository_tree_id=TREE,
    )
    outcome = ProofOutcome.from_receipt(
        selection.requirements[0].requirement_id,
        scheduled.authoritative_receipts[0],
    )
    merge = MergeProofGateReceipt.build(
        policy=policy,
        selection=selection,
        repository_tree_id=TREE,
        proof_plan=plan,
        outcomes=(outcome,),
        proof_receipts=scheduled.authoritative_receipts,
        provider_status={"status": "healthy"},
    )
    outage = MergeProofGateReceipt.build(
        policy=policy,
        selection=selection,
        repository_tree_id=TREE,
        proof_plan=plan,
        outcomes=(
            ProofOutcome(
                requirement_id=selection.requirements[0].requirement_id,
                status=ProofResultStatus.UNAVAILABLE,
                reason_code="provider_outage",
            ),
        ),
        provider_status={"status": "unavailable"},
        provider_error="provider_outage",
    )

    assert merge.allowed is True
    assert merge.proof_receipt_ids == (proof.receipt_id,)
    assert outage.allowed is False
    assert outage.decision.rollout_mode is RolloutMode.ENFORCEMENT
