from __future__ import annotations

import threading
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AttemptStatus,
    ProofPlan,
    ProofPlanStep,
    ProofStage,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    DEFAULT_RESOURCE_CLASSES,
    HostResourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    DeclaredValidation,
    ValidationCommand,
    ValidationRequirementKind,
    ValidationStage,
    select_validation_commands,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ValidationScheduler,
)


TREE = "git-tree:proof-validation"
OBLIGATION = "obligation:proof-validation"


def _step(
    step_id: str,
    stage: ProofStage,
    *,
    depends_on: tuple[str, ...] = (),
) -> ProofPlanStep:
    return ProofPlanStep(
        step_id=step_id,
        obligation_id=OBLIGATION,
        stage=stage,
        provider_id=f"provider:{step_id}",
        depends_on=depends_on,
        resource_class=stage.value,
    )


def _plan(*steps: ProofPlanStep, max_parallel: int = 2) -> ProofPlan:
    return ProofPlan(
        repository_tree_id=TREE,
        obligation_ids=(OBLIGATION,),
        steps=steps,
        policy_id="policy:proof-validation",
        resource_budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=10_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=max_parallel,
        ),
        max_parallel=max_parallel,
    )


def _host(capacity: int = 2) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        observed_at_ms=1,
        cpu_percent=10,
        memory_percent=10,
        disk_percent=10,
        memory_total_bytes=1_000_000_000,
        memory_available_bytes=900_000_000,
        disk_total_bytes=1_000_000_000,
        disk_available_bytes=900_000_000,
        worker_limit=capacity,
        available_worker_capacity=capacity,
        resource_classes=DEFAULT_RESOURCE_CLASSES,
    )


def _command_result(spec: ValidationCommand, returncode: int = 0) -> dict[str, object]:
    return {
        "command": spec.command,
        "returncode": returncode,
        "output": "",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
    }


def test_staged_pipeline_orders_proof_tests_and_attestation(tmp_path: Path) -> None:
    calls: list[str] = []
    plan = _plan(
        _step("translate", ProofStage.TRANSLATE),
        _step("solve", ProofStage.SOLVE, depends_on=("translate",)),
        _step("reconstruct", ProofStage.RECONSTRUCT, depends_on=("solve",)),
        _step("kernel", ProofStage.KERNEL_VERIFY, depends_on=("reconstruct",)),
        _step("proof-validation", ProofStage.VALIDATE, depends_on=("kernel",)),
        _step("attest", ProofStage.ATTEST, depends_on=("proof-validation",)),
        _step("persist", ProofStage.PERSIST, depends_on=("attest",)),
    )

    def proof_executor(context) -> None:
        calls.append(context.stage.value)

    def runner(*, spec, **_kwargs):
        calls.append(spec.stage.label)
        return _command_result(spec)

    report = ValidationScheduler(
        max_workers=2,
        resource_budget=2,
        host_resource_source=_host(),
        runner=runner,
    ).run(
        [
            "pytest tests/test_alpha.py",
            "custom-broad-validation --all",
            "git diff --check",
        ],
        workspace_path=tmp_path,
        proof_plan=plan,
        proof_executor=proof_executor,
        proof_scheduler_options={"state_path": tmp_path / "proof-state"},
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
    )

    assert calls == [
        "cheap",
        "translate",
        "solve",
        "reconstruct",
        "kernel_verify",
        "validate",
        "targeted",
        "broad",
        "attest",
        "persist",
    ]
    assert report["passed"] is True
    assert [stage["stage"] for stage in report["stages"]] == [
        "deterministic",
        "translation",
        "solver",
        "kernel",
        "proof_validation",
        "focused",
        "broad",
        "attestation",
        "persist",
    ]
    assert report["shared_resource_scheduler"] is True
    assert report["shared_resource_lease_budget"] is True
    assert all(
        report["verdicts"][kind]["verdict"] == "passed"
        for kind in (
            "deterministic",
            "translation",
            "solver",
            "kernel",
            "test",
            "attestation",
        )
    )


def test_independent_proof_and_test_checks_share_one_bounded_budget(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    proof_active = 0
    test_active = 0
    proof_peak = 0
    test_peak = 0
    lease_peak = 0
    plan = _plan(
        _step("solve-a", ProofStage.SOLVE),
        _step("solve-b", ProofStage.SOLVE),
    )
    scheduler = ValidationScheduler(
        max_workers=3,
        resource_budget=2,
        host_resource_source=_host(2),
    )

    def observe_leases() -> None:
        nonlocal lease_peak
        lease_peak = max(
            lease_peak, len(scheduler.resource_scheduler.active_leases)
        )

    def proof_executor(_context) -> None:
        nonlocal proof_active, proof_peak
        with lock:
            proof_active += 1
            proof_peak = max(proof_peak, proof_active)
            observe_leases()
        time.sleep(0.04)
        with lock:
            proof_active -= 1

    def runner(*, spec, **_kwargs):
        nonlocal test_active, test_peak
        with lock:
            test_active += 1
            test_peak = max(test_peak, test_active)
            observe_leases()
        time.sleep(0.04)
        with lock:
            test_active -= 1
        return _command_result(spec)

    report = scheduler.run_staged(
        ["pytest tests/test_alpha.py", "pytest tests/test_beta.py"],
        workspace_path=tmp_path,
        proof_plan=plan,
        proof_executor=proof_executor,
        proof_scheduler_options={"state_path": tmp_path / "proof-state"},
        changed_files=["pyproject.toml"],
        target_commit="commit:test",
        dependency_state="deps",
        runner=runner,
    )

    assert report["passed"] is True
    assert proof_peak == 2
    assert test_peak == 2
    assert lease_peak == 2
    assert scheduler.resource_scheduler.active_leases == ()


def test_selection_explains_omission_escalation_and_fallbacks(
    tmp_path: Path,
) -> None:
    narrow = select_validation_commands(
        [
            "git diff --check",
            "pytest tests/test_alpha.py",
            "pytest tests/test_beta.py",
        ],
        ["src/alpha.py"],
    )
    decisions = {item.spec.command: item for item in narrow.items if item.spec}
    assert decisions["git diff --check"].decision.value == "included"
    assert decisions["pytest tests/test_alpha.py"].reason == (
        "changed_path_matches_command_target"
    )
    assert decisions["pytest tests/test_beta.py"].decision.value == "omitted"

    escalated = select_validation_commands(
        ["pytest tests/test_beta.py"],
        ["src/alpha.py"],
        require_full_validation=True,
    )
    assert escalated.items[0].decision.value == "escalated"
    assert escalated.items[0].spec is not None
    assert escalated.items[0].spec.stage is ValidationStage.BROAD

    executable = DeclaredValidation(
        validation_id="focused:fallback",
        kind=ValidationRequirementKind.FOCUSED_TEST,
        command=ValidationCommand(
            "pytest tests/test_fallback.py",
            stage=ValidationStage.TARGETED,
            validation_id="focused:fallback",
        ),
    )
    manual = DeclaredValidation(
        validation_id="manual:proof-review",
        kind=ValidationRequirementKind.MANUAL_REVIEW,
        reason="declared_manual_review",
    )
    unresolved = DeclaredValidation(
        validation_id="static:unresolved",
        kind=ValidationRequirementKind.STATIC_CHECK,
        reason="declared_validation_requires_catalog_resolution",
    )
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _command_result(spec)

    report = ValidationScheduler(runner=runner).run_staged(
        workspace_path=tmp_path,
        fallback_plan={
            "plan_id": "fallback:one",
            "obligation_id": OBLIGATION,
            "can_continue": True,
            "blocking": False,
            "validations": (executable, manual, unresolved),
        },
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
    )

    assert calls == ["pytest tests/test_fallback.py"]
    fallback_decisions = [
        item
        for item in report["selection"]["decisions"]
        if item["fallback"]
    ]
    assert {item["validation_id"] for item in fallback_decisions} == {
        "focused:fallback",
        "manual:proof-review",
        "static:unresolved",
    }
    assert next(
        item
        for item in fallback_decisions
        if item["validation_id"] == "manual:proof-review"
    )["executable"] is False
    assert report["selection"]["unresolved_fallback_count"] == 2


def test_solver_kernel_test_and_attestation_verdicts_do_not_collapse(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _step("translate", ProofStage.TRANSLATE),
        _step("solve", ProofStage.SOLVE, depends_on=("translate",)),
        _step("kernel", ProofStage.KERNEL_VERIFY, depends_on=("solve",)),
        _step("attest", ProofStage.ATTEST, depends_on=("kernel",)),
    )

    def proof_executor(context):
        if context.stage is ProofStage.KERNEL_VERIFY:
            return AttemptStatus.FAILED
        return AttemptStatus.SUCCEEDED

    report = ValidationScheduler(
        host_resource_source=_host(),
    ).run_staged(
        ["git diff --check"],
        workspace_path=tmp_path,
        proof_plan=plan,
        proof_executor=proof_executor,
        proof_scheduler_options={"state_path": tmp_path / "proof-state"},
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
        runner=lambda spec, **_kwargs: _command_result(spec),
    )

    assert report["passed"] is False
    assert report["verdicts"]["deterministic"]["verdict"] == "passed"
    assert report["verdicts"]["solver"]["verdict"] == "passed"
    assert report["verdicts"]["kernel"]["verdict"] == "failed"
    assert report["verdicts"]["test"]["verdict"] == "not_run"
    assert report["verdicts"]["attestation"]["verdict"] == "failed"


def test_deterministic_failure_prevents_all_proof_and_test_work(
    tmp_path: Path,
) -> None:
    plan = _plan(_step("translate", ProofStage.TRANSLATE))
    proof_calls = 0

    def proof_executor(_context) -> None:
        nonlocal proof_calls
        proof_calls += 1

    def runner(*, spec, **_kwargs):
        return _command_result(
            spec, returncode=9 if spec.stage is ValidationStage.CHEAP else 0
        )

    report = ValidationScheduler(runner=runner).run_staged(
        ["git diff --check", "pytest tests/test_alpha.py"],
        workspace_path=tmp_path,
        proof_plan=plan,
        proof_executor=proof_executor,
        proof_scheduler_options={"state_path": tmp_path / "proof-state"},
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
    )

    assert proof_calls == 0
    assert report["passed"] is False
    assert report["failed_command"] == "git diff --check"
    assert report["verdicts"]["deterministic"]["verdict"] == "failed"
    assert report["verdicts"]["translation"]["verdict"] == "not_run"
    assert report["verdicts"]["solver"]["verdict"] == "not_run"
    assert report["verdicts"]["kernel"]["verdict"] == "not_run"
    assert report["verdicts"]["test"]["verdict"] == "not_run"
    assert report["verdicts"]["attestation"]["verdict"] == "not_run"


def test_shadow_fallback_continues_without_upgrading_failed_proof(
    tmp_path: Path,
) -> None:
    fallback = DeclaredValidation(
        validation_id="focused:shadow-regression",
        kind=ValidationRequirementKind.FOCUSED_TEST,
        command=ValidationCommand(
            "pytest tests/test_shadow_regression.py",
            stage=ValidationStage.TARGETED,
            validation_id="focused:shadow-regression",
        ),
    )
    plan = _plan(_step("solve", ProofStage.SOLVE))

    report = ValidationScheduler().run_staged(
        workspace_path=tmp_path,
        proof_plan=plan,
        proof_executor=lambda _context: AttemptStatus.FAILED,
        proof_scheduler_options={"state_path": tmp_path / "proof-state"},
        fallback_plan={
            "plan_id": "fallback:shadow",
            "obligation_id": OBLIGATION,
            "can_continue": True,
            "blocking": False,
            "validations": (fallback,),
        },
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
        runner=lambda spec, **_kwargs: _command_result(spec),
    )

    assert report["passed"] is True
    assert report["verdicts"]["solver"]["verdict"] == "failed"
    assert report["verdicts"]["test"]["verdict"] == "passed"
    assert report["verdicts"]["kernel"]["verdict"] == "not_run"
    assert report["verdicts"]["attestation"]["verdict"] == "not_run"
    assert report["fallbacks"][0]["can_continue"] is True


def test_blocking_fallback_runs_focused_check_but_stops_broad_gate(
    tmp_path: Path,
) -> None:
    fallback = DeclaredValidation(
        validation_id="focused:enforcement-regression",
        kind=ValidationRequirementKind.FOCUSED_TEST,
        command=ValidationCommand(
            "pytest tests/test_enforcement_regression.py",
            stage=ValidationStage.TARGETED,
            validation_id="focused:enforcement-regression",
        ),
    )
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _command_result(spec)

    report = ValidationScheduler(runner=runner).run_staged(
        ["custom-broad-validation --all"],
        workspace_path=tmp_path,
        fallback_plan={
            "plan_id": "fallback:enforcement",
            "obligation_id": OBLIGATION,
            "can_continue": False,
            "blocking": True,
            "validations": (fallback,),
        },
        changed_files=["src/alpha.py"],
        target_commit="commit:test",
        dependency_state="deps",
    )

    assert calls == ["pytest tests/test_enforcement_regression.py"]
    assert report["passed"] is False
    assert report["error"] == "proof_gate_failed"
    assert report["verdicts"]["test"]["verdict"] == "passed"
    broad_stage = next(
        item for item in report["stages"] if item["stage"] == "broad"
    )
    assert broad_stage["attempted"] is False
    assert broad_stage["reason"] == "prior_stage_failed"
