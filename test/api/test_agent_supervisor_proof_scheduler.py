from __future__ import annotations

import threading
import time
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
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
from ipfs_accelerate_py.agent_supervisor.proof_scheduler import (
    ProofNodeState,
    ProofScheduler,
    ProofSchedulerConfig,
    ProofStepResult,
)


TREE = "git-tree:scheduler"
OBLIGATION = "obligation:scheduler"


def _step(
    step_id: str,
    stage: ProofStage,
    *,
    depends_on: tuple[str, ...] = (),
    obligation_id: str = OBLIGATION,
    portfolio_id: str = "",
) -> ProofPlanStep:
    metadata = {"portfolio_id": portfolio_id} if portfolio_id else {}
    return ProofPlanStep(
        step_id=step_id,
        obligation_id=obligation_id,
        stage=stage,
        provider_id=f"provider:{step_id}",
        depends_on=depends_on,
        resource_class=stage.value,
        metadata=metadata,
    )


def _plan(
    *steps: ProofPlanStep,
    max_parallel: int = 4,
    obligation_ids: tuple[str, ...] = (OBLIGATION,),
) -> ProofPlan:
    return ProofPlan(
        repository_tree_id=TREE,
        obligation_ids=obligation_ids,
        steps=steps,
        policy_id="policy:scheduler",
        resource_budget=ResourceBudget(
            wall_time_ms=30_000,
            cpu_time_ms=30_000,
            memory_bytes=256 * 1024 * 1024,
            max_processes=max_parallel,
        ),
        max_parallel=max_parallel,
    )


def _receipt(plan: ProofPlan, step: ProofPlanStep, suffix: str) -> ProofReceipt:
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id=f"artifact:{suffix}",
        subject_id=step.obligation_id,
        verifier_id="kernel:lean",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    return ProofReceipt(
        obligation_id=step.obligation_id,
        plan_id=plan.plan_id,
        attempt_id=f"attempt:{suffix}",
        repository_id="repo:test",
        repository_tree_id=plan.repository_tree_id,
        ast_scope_ids=("scope:test",),
        premise_ids=(),
        translator_id="translator:test",
        solver_id="solver:test",
        kernel_id="kernel:lean",
        toolchain_id="toolchain:test",
        theorem_registry_id="registry:test",
        policy_id=plan.policy_id,
        resource_budget=plan.resource_budget,
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
    )


def test_priorities_expose_critical_path_and_downstream_unlocks(tmp_path: Path) -> None:
    plan = _plan(
        _step("long-root", ProofStage.TRANSLATE),
        _step("short-root", ProofStage.SOLVE),
        _step("middle", ProofStage.SOLVE, depends_on=("long-root",)),
        _step("leaf", ProofStage.KERNEL_VERIFY, depends_on=("middle",)),
    )
    scheduler = ProofScheduler(plan, lambda step: None, state_path=tmp_path)

    ready = scheduler.ready_steps()

    assert [item.step_id for item in ready] == ["long-root", "short-root"]
    priority = scheduler.priorities()
    assert priority["long-root"].critical_path_length == 3
    assert priority["long-root"].downstream_unlock_count == 2
    assert priority["short-root"].critical_path_length == 1
    assert plan.topological_step_ids.index("middle") < plan.topological_step_ids.index("leaf")


def test_dependency_order_and_bounded_independent_stage_overlap(tmp_path: Path) -> None:
    plan = _plan(
        _step("translate-a", ProofStage.TRANSLATE),
        _step("translate-b", ProofStage.TRANSLATE),
        _step("solver", ProofStage.SOLVE, depends_on=("translate-a",)),
        _step("validation", ProofStage.VALIDATE, depends_on=("translate-b",)),
        _step("kernel", ProofStage.KERNEL_VERIFY, depends_on=("solver",)),
        _step(
            "artifact",
            ProofStage.PERSIST,
            depends_on=("kernel", "validation"),
        ),
        max_parallel=2,
    )
    lock = threading.Lock()
    active = 0
    peak = 0
    started: dict[str, float] = {}
    finished: dict[str, float] = {}

    def execute(step: ProofPlanStep) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started[step.step_id] = time.monotonic()
        time.sleep(0.025)
        with lock:
            finished[step.step_id] = time.monotonic()
            active -= 1

    result = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        config=ProofSchedulerConfig(max_parallel=2, lease_seconds=2),
    ).run()

    assert result.complete
    assert set(result.states.values()) == {ProofNodeState.SUCCEEDED}
    assert peak == 2
    assert started["solver"] >= finished["translate-a"]
    assert started["validation"] >= finished["translate-b"]
    assert started["kernel"] >= finished["solver"]
    assert started["artifact"] >= max(finished["kernel"], finished["validation"])


def test_stage_and_resource_class_limits_refine_global_parallelism(tmp_path: Path) -> None:
    plan = _plan(
        _step("solve-a", ProofStage.SOLVE),
        _step("solve-b", ProofStage.SOLVE),
        _step("validate", ProofStage.VALIDATE),
        max_parallel=3,
    )
    lock = threading.Lock()
    solve_active = 0
    solve_peak = 0
    global_active = 0
    global_peak = 0

    def execute(step: ProofPlanStep) -> None:
        nonlocal solve_active, solve_peak, global_active, global_peak
        with lock:
            global_active += 1
            global_peak = max(global_peak, global_active)
            if step.stage is ProofStage.SOLVE:
                solve_active += 1
                solve_peak = max(solve_peak, solve_active)
        time.sleep(0.035)
        with lock:
            global_active -= 1
            if step.stage is ProofStage.SOLVE:
                solve_active -= 1

    ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        config=ProofSchedulerConfig(
            max_parallel=3,
            lease_seconds=2,
            stage_limits={ProofStage.SOLVE: 1},
        ),
    ).run()

    assert solve_peak == 1
    assert global_peak == 2


def test_preparation_failure_releases_entire_admission_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(
        _step("translate-a", ProofStage.TRANSLATE),
        _step("translate-b", ProofStage.TRANSLATE),
        max_parallel=2,
    )
    scheduler = ProofScheduler(
        plan,
        lambda step: None,
        state_path=tmp_path,
        max_parallel=2,
    )
    original_prepare = scheduler._prepare_invocation
    preparation_count = 0

    def fail_second_preparation(*args, **kwargs):
        nonlocal preparation_count
        preparation_count += 1
        if preparation_count == 2:
            raise RuntimeError("preparation failed")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_prepare_invocation", fail_second_preparation)

    with pytest.raises(RuntimeError, match="preparation failed"):
        scheduler.run()

    snapshot = scheduler.snapshot()
    assert snapshot.active_leases == 0
    assert scheduler.resource_scheduler.active_leases == ()
    assert {node.state for node in snapshot.nodes} == {ProofNodeState.READY}


def test_conclusive_portfolio_receipt_cancels_redundant_attempt(tmp_path: Path) -> None:
    fast = _step(
        "fast-kernel",
        ProofStage.KERNEL_VERIFY,
        portfolio_id="portfolio:one",
    )
    slow = _step(
        "slow-kernel",
        ProofStage.KERNEL_VERIFY,
        portfolio_id="portfolio:one",
    )
    plan = _plan(fast, slow, max_parallel=2)
    both_started = threading.Barrier(2)
    slow_observed_cancellation = threading.Event()

    def fast_executor(context) -> ProofReceipt:
        both_started.wait(timeout=2)
        return _receipt(plan, fast, "fast")

    def slow_executor(context) -> ProofStepResult:
        both_started.wait(timeout=2)
        assert context.cancellation_token.wait(timeout=2)
        slow_observed_cancellation.set()
        return ProofStepResult(
            status=AttemptStatus.CANCELLED,
            error_code="portfolio_cancelled",
        )

    result = ProofScheduler(
        plan,
        executors={
            "fast-kernel": fast_executor,
            "slow-kernel": slow_executor,
        },
        state_path=tmp_path,
        max_parallel=2,
        lease_seconds=2,
    ).run()

    assert slow_observed_cancellation.is_set()
    assert result.succeeded
    assert result.states["fast-kernel"] is ProofNodeState.SUCCEEDED
    assert result.states["slow-kernel"] is ProofNodeState.CANCELLED
    assert len(result.authoritative_receipts) == 1
    assert result.authoritative_receipts[0].authoritative_verdict is ProofVerdict.PROVED

    connection = duckdb.connect(str(tmp_path / "proof_scheduler.duckdb"))
    try:
        authoritative_count = connection.execute(
            "SELECT COUNT(*) FROM proof_receipts WHERE authoritative=1"
        ).fetchone()[0]
    finally:
        connection.close()
    assert authoritative_count == 1


def test_unsupported_and_failed_dependencies_propagate_explicitly(tmp_path: Path) -> None:
    plan = _plan(
        _step("unsupported", ProofStage.TRANSLATE),
        _step("failed", ProofStage.SOLVE),
        _step("unsupported-child", ProofStage.VALIDATE, depends_on=("unsupported",)),
        _step("failed-child", ProofStage.PERSIST, depends_on=("failed",)),
    )

    def execute(step: ProofPlanStep) -> AttemptStatus:
        if step.step_id == "unsupported":
            return AttemptStatus.UNSUPPORTED
        if step.step_id == "failed":
            return AttemptStatus.FAILED
        raise AssertionError("blocked dependency was executed")

    scheduler = ProofScheduler(plan, execute, state_path=tmp_path)
    result = scheduler.run()
    nodes = {node.step_id: node for node in result.snapshot.nodes}

    assert result.states["unsupported"] is ProofNodeState.UNSUPPORTED
    assert result.states["unsupported-child"] is ProofNodeState.UNSUPPORTED
    assert nodes["unsupported-child"].reason_code == "unsupported_dependency:unsupported"
    assert result.states["failed"] is ProofNodeState.FAILED
    assert result.states["failed-child"] is ProofNodeState.BLOCKED
    assert nodes["failed-child"].reason_code == "blocked_dependency:failed"


def test_restart_restores_plan_attempts_receipts_and_does_not_reexecute(tmp_path: Path) -> None:
    kernel = _step("kernel", ProofStage.KERNEL_VERIFY)
    plan = _plan(kernel)
    calls = 0

    def execute(step: ProofPlanStep) -> ProofReceipt:
        nonlocal calls
        calls += 1
        return _receipt(plan, step, f"run-{calls}")

    first = ProofScheduler(plan, execute, state_path=tmp_path).run()
    assert calls == 1
    assert len(first.authoritative_receipts) == 1

    def must_not_run(step: ProofPlanStep) -> None:
        raise AssertionError("a completed durable node was executed twice")

    restarted = ProofScheduler(
        executor=must_not_run,
        state_path=tmp_path,
        plan_id=plan.plan_id,
    ).run()

    assert restarted.states == first.states
    assert len(restarted.authoritative_receipts) == 1
    assert restarted.authoritative_receipts[0].receipt_id == first.authoritative_receipts[0].receipt_id
    assert any(attempt.status is AttemptStatus.SUCCEEDED for attempt in restarted.attempts)


def test_restart_requeues_an_expired_fenced_lease(tmp_path: Path) -> None:
    plan = _plan(_step("translate", ProofStage.TRANSLATE))
    initial = ProofScheduler(plan, lambda step: None, state_path=tmp_path)
    db_path = initial.db_path
    now_ms = int(time.time() * 1000)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            UPDATE proof_nodes SET state='running'
            WHERE plan_id=? AND step_id='translate'
            """,
            (plan.plan_id,),
        )
        connection.execute(
            """
            INSERT INTO proof_leases(
                plan_id, step_id, owner_id, token, fencing_token,
                acquired_at_ms, heartbeat_at_ms, expires_at_ms,
                released_at_ms, release_reason
            ) VALUES (?, 'translate', 'dead-owner', 'old-token', 7, ?, ?, ?, NULL, '')
            """,
            (plan.plan_id, now_ms - 10_000, now_ms - 10_000, now_ms - 1),
        )
        connection.commit()
    finally:
        connection.close()

    calls = 0

    def execute(step: ProofPlanStep) -> None:
        nonlocal calls
        calls += 1

    restarted = ProofScheduler(
        executor=execute,
        state_path=tmp_path,
        plan_id=plan.plan_id,
        lease_seconds=2,
    )
    result = restarted.run()

    assert calls == 1
    assert result.states["translate"] is ProofNodeState.SUCCEEDED
    connection = duckdb.connect(str(db_path))
    try:
        lease = connection.execute(
            "SELECT fencing_token, release_reason FROM proof_leases "
            "WHERE plan_id=? AND step_id='translate'",
            (plan.plan_id,),
        ).fetchone()
    finally:
        connection.close()
    assert lease[0] == 8
    assert lease[1] == "succeeded"
