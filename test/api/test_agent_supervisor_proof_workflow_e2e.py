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
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import resource_pool
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
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
    active_by_pool: dict[str, int] = {}
    peak_by_pool: dict[str, int] = {}
    calls: dict[str, int] = {}

    def execute(context) -> None:
        nonlocal active, peak
        pool = resource_pool(context.resource_class)
        with lock:
            active += 1
            peak = max(peak, active)
            active_by_pool[pool] = active_by_pool.get(pool, 0) + 1
            peak_by_pool[pool] = max(
                peak_by_pool.get(pool, 0),
                active_by_pool[pool],
            )
            calls[context.step_id] = calls.get(context.step_id, 0) + 1
        time.sleep(0.03)
        with lock:
            active -= 1
            active_by_pool[pool] -= 1

    result = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        config=ProofSchedulerConfig(
            max_parallel=3,
            lease_seconds=2,
            max_cpu_proof_concurrency=1,
            max_model_concurrency=1,
            max_artifact_concurrency=1,
        ),
    ).run()

    assert result.complete
    assert 1 < peak <= 3
    assert peak_by_pool == {"cpu-proof": 1, "model": 1, "artifact": 1}
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
    assert {
        lease.step_id: (lease.fencing_token, lease.active)
        for lease in restarted.snapshot.leases
    } == {
        "translate": (1, False),
        "solve": (1, False),
        "kernel": (1, False),
    }
    serialized = restarted.to_dict()
    assert serialized["receipt_lineage"] == [
        {
            "receipt_id": restarted.authoritative_receipts[0].receipt_id,
            "attempt_id": restarted.authoritative_receipts[0].attempt_id,
            "obligation_id": kernel.obligation_id,
            "authoritative_verdict": "proved",
            "freshness": "current",
        }
    ]
    assert next(
        node for node in serialized["nodes"] if node["step_id"] == "kernel"
    )["dependency_step_ids"] == ["solve"]


def test_parallel_daemon_lanes_resume_one_proof_workflow_and_report_truth(
    tmp_path: Path,
) -> None:
    translate = _step("daemon-translate", ProofStage.TRANSLATE)
    solve = _step(
        "daemon-solve",
        ProofStage.SOLVE,
        depends_on=(translate.step_id,),
    )
    kernel = _step(
        "daemon-kernel",
        ProofStage.KERNEL_VERIFY,
        depends_on=(solve.step_id,),
    )
    plan = _plan(translate, solve, kernel, max_parallel=2)
    task = PortalTask(
        task_id="REF-E2E",
        title="proof-aware implementation",
        status="todo",
        completion="manual",
        priority="P0",
        track="G11",
        validation=[],
        acceptance="proof workflow completes exactly once",
        board_namespace="proof-e2e",
    )
    calls: dict[str, int] = {}
    lock = threading.Lock()
    translate_entered = threading.Event()

    def execute(context):
        with lock:
            calls[context.step_id] = calls.get(context.step_id, 0) + 1
        if context.step_id == translate.step_id:
            translate_entered.set()
            time.sleep(0.04)
        if context.step_id == kernel.step_id:
            assert {
                item.step_id for item in context.dependency_attempts
            } == {solve.step_id}
            return _receipt(plan, kernel)
        return None

    todo_path = tmp_path / "todo.md"
    todo_path.write_text(
        "- [ ] Task checkbox-999: REF-E2E proof-aware implementation\n\n"
        "## REF-E2E proof-aware implementation\n\n"
        "- Status: todo\n",
        encoding="utf-8",
    )
    daemons = [
        PortalImplementationDaemon(
            todo_path=todo_path,
            state_path=tmp_path / f"lane-{lane}" / "state.json",
            strategy_path=tmp_path / f"lane-{lane}" / "strategy.json",
            events_path=tmp_path / f"lane-{lane}" / "events.jsonl",
            repo_root=tmp_path,
            merge_queue_dir=tmp_path / "merge-queue",
            validation_cache_dir=tmp_path / "validation-cache",
            proof_workflow={
                "proof_plan": plan,
                "proof_executor": execute,
            },
            task_header_prefix="## REF-",
            worktree_pool_enabled=False,
        )
        for lane in ("a", "b")
    ]
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def run_lane(index: int) -> None:
        try:
            state = PortalTaskState(active_task_id=task.task_id)
            result = daemons[index]._run_validation_commands(
                tmp_path,
                task,
                tmp_path / f"lane-{index}.log",
                state=state,
            )
            with lock:
                results.append(result)
        except BaseException as exc:  # pragma: no cover - asserted below
            with lock:
                failures.append(exc)

    first = threading.Thread(target=run_lane, args=(0,))
    second = threading.Thread(target=run_lane, args=(1,))
    first.start()
    assert translate_entered.wait(timeout=5)
    second.start()
    first.join(timeout=15)
    second.join(timeout=15)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert len(results) == 2
    assert calls == {
        translate.step_id: 1,
        solve.step_id: 1,
        kernel.step_id: 1,
    }
    assert all(result["passed"] is True for result in results), [
        {
            "passed": result.get("passed"),
            "error": result.get("error"),
            "proof": result.get("proof"),
            "stages": result.get("stages"),
        }
        for result in results
    ]
    assert all(
        result["shared_resource_scheduler"] is True
        and result["shared_resource_lease_budget"] is True
        for result in results
    )
    operator_states = [
        result["proof_operator_state"]
        for result in results
    ]
    assert all(state["complete"] is True for state in operator_states)
    assert all(state["active_leases"] == 0 for state in operator_states)
    assert all(
        state["dependencies"][kernel.step_id] == [solve.step_id]
        for state in operator_states
    )
    assert all(
        len(state["authoritative_receipt_ids"]) == 1
        for state in operator_states
    )
    assert len(
        {
            tuple(state["authoritative_receipt_ids"])
            for state in operator_states
        }
    ) == 1
    assert len(operator_states[0]["receipt_lineage"]) == 1

    restarted = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "lane-restarted" / "state.json",
        strategy_path=tmp_path / "lane-restarted" / "strategy.json",
        events_path=tmp_path / "lane-restarted" / "events.jsonl",
        repo_root=tmp_path,
        merge_queue_dir=tmp_path / "merge-queue",
        validation_cache_dir=tmp_path / "validation-cache",
        proof_workflow={
            "proof_plan": plan,
            "proof_executor": execute,
        },
        task_header_prefix="## REF-",
        worktree_pool_enabled=False,
    )
    resumed = restarted._run_validation_commands(
        tmp_path,
        task,
        tmp_path / "restarted.log",
    )
    assert resumed["passed"] is True
    assert calls == {
        translate.step_id: 1,
        solve.step_id: 1,
        kernel.step_id: 1,
    }
    durable = PortalTaskState.load(restarted.state_path).last_proof_workflow
    assert durable["plan_id"] == plan.plan_id
    assert durable["receipt_lineage"] == operator_states[0]["receipt_lineage"]

    merge_requests = [
        daemon._enqueue_merge_candidate(
            branch_name=f"implementation/ref-e2e-{index}",
            implementation_commit="commit:proof-e2e",
            baseline_ref="baseline:proof-e2e",
            worktree_path=None,
            task=task,
            attempt=1,
            validation_result=resumed,
        )[0]
        for index, daemon in enumerate(daemons)
    ]
    assert merge_requests[0].request_id == merge_requests[1].request_id
    assert daemons[0].merge_queue.pending_count() == 1

    first_transition = daemons[0]._mark_tasks_completed_in_todo(
        (task.task_id,),
        primary_task_id=task.task_id,
        completion_reason="proof_e2e",
    )
    duplicate_transition = daemons[1]._mark_tasks_completed_in_todo(
        (task.task_id,),
        primary_task_id=task.task_id,
        completion_reason="proof_e2e",
    )
    assert first_transition["updated_task_ids"] == [task.task_id]
    assert duplicate_transition["updated"] is False
    assert duplicate_transition["reason"] == "already_completed"
    assert todo_path.read_text(encoding="utf-8").count(
        "- Status: completed"
    ) == 1


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
