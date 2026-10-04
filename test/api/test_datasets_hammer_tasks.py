"""Native checked SMT collection through the datasets shared resource owner."""
from dataclasses import replace
import shutil
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.datasets_hammer_tasks import make_datasets_smt_task
from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import open_datasets_prover_lease
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget, MultiProverResourceLease,
)
from ipfs_datasets_py.logic.hammers import semantic_routing as routing
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


def target(name="fixture", text="p and not p", solvers=("z3", "cvc5")):
    parsed = routing.modal.parse_modal(text, routing.modal.profile_k())
    assert parsed.ok
    replay = routing.modal.parse_modal(parsed.printed, routing.modal.profile_k())
    return dict(request_id=name, source_construct="authored-propositional-fixture",
        logic_family="propositional", ast_format="shared_logic", printed=parsed.printed,
        native_ast=replay.root.to_dict(), solver_names=solvers)


def budget(width=2):
    return MultiProverResourceBudget(cpu_slots=width, process_slots=width,
        thread_slots=width, memory_bytes=width * 512 * 1024**2, max_portfolio_width=width)


@pytest.fixture
def native(tmp_path):
    assert shutil.which("z3"), "native qualification requires z3"
    assert shutil.which("cvc5"), "native qualification requires cvc5"
    healthy = ProofHostResources(8, 8192, 8192)
    current = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", proof_resource_sampler=lambda: current[0],
        total_cpu_slots=4, total_memory_mb=4096, total_child_process_slots=4,
        lane_reservations={}, proof_backoff_seconds=.04, poll_interval_seconds=.005,
    ))
    yield scheduler, current, healthy
    deadline = time.monotonic() + 3
    while scheduler.snapshot()["active_lease_count"] and time.monotonic() < deadline:
        time.sleep(.01)
    assert scheduler.snapshot()["active_lease_count"] == 0
    assert scheduler.snapshot()["waiting_request_count"] == 0


def test_native_dependency_bundle_retains_every_solver_without_proof_authority(native):
    scheduler, _, _ = native
    tasks = [make_datasets_smt_task(target=target("contradiction"), expected_verdict="unsat"),
        make_datasets_smt_task(target=target("satisfiable", "p or q"), expected_verdict="sat"),
        make_datasets_smt_task(target=target("after-both"), expected_verdict="unsat",
            dependencies=("contradiction", "satisfiable"), max_parallel_processes=2)]
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        receipt = BundleProverSupervisor(lease).execute(tasks)
        assert scheduler.snapshot()["active_root_lease_count"] == 1
        assert receipt.successful_task_ids == ("contradiction", "satisfiable", "after-both")
    first, second, last = receipt.receipts
    assert last.started_at_ms >= max(first.finished_at_ms, second.finished_at_ms)
    for row in receipt.receipts:
        result = row.result
        assert result["all_expected_verdicts_observed"]
        assert {item["solver_name"] for item in result["attempts"]} == {"z3", "cvc5"}
        assert all(item["solver_version"] for item in result["attempts"])
        assert not row.cache_bypassed_execution
        assert result["input_semantics"] == "assert_formula"
        assert all(result[field] is False for field in ("proof_authority", "source_semantics_verified",
            "behavior_authority", "execution_authority", "completion_authority"))


def test_native_wrong_expectation_retains_counter_verdict_and_blocks_dependency(native):
    scheduler, _, _ = native
    mismatch = make_datasets_smt_task(target=target("mismatch"), expected_verdict="sat")
    dependent = make_datasets_smt_task(target=target("dependent"), expected_verdict="unsat",
        dependencies=("mismatch",))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((mismatch, dependent))
    failed, blocked = result.receipts
    assert failed.status == ExecutionStatus.FAILED
    assert failed.reasons == ("smt_expected_verdict_not_observed",)
    assert not failed.result["all_expected_verdicts_observed"]
    assert [item["verdict"] for item in failed.result["attempts"]] == ["unsat", "unsat"]
    assert blocked.status == ExecutionStatus.BLOCKED
    assert blocked.reasons == ("dependency:mismatch",)


def test_native_pressure_waits_before_process_launch_then_resumes_same_task(native, monkeypatch):
    scheduler, current, healthy = native
    from ipfs_datasets_py.logic.hammers import process_lifecycle
    native_run = process_lifecycle.subprocess.Popen
    launches = []
    def observed(*args, **kwargs):
        launches.append(args[0])
        return native_run(*args, **kwargs)
    monkeypatch.setattr(process_lifecycle.subprocess, "Popen", observed)
    task = make_datasets_smt_task(target=target(), expected_verdict="unsat")
    results, failures = [], []
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(),
            admission_timeout_seconds=2) as lease:
        current[0] = replace(healthy, memory_stall_percent=5)
        def execute():
            try:
                results.append(BundleProverSupervisor(lease).execute((task,)))
            except BaseException as error:
                failures.append(error)
        worker = threading.Thread(target=execute)
        worker.start()
        try:
            deadline = time.monotonic() + 1
            while not scheduler.snapshot()["proof_backoff"] and time.monotonic() < deadline:
                time.sleep(.005)
            assert scheduler.snapshot()["proof_backoff"]["reason"] == "proof_memory_stall"
            assert not launches and not results
            current[0] = healthy
            worker.join(10)
            assert not worker.is_alive() and not failures
        finally:
            current[0] = healthy
            lease.cancel() if worker.is_alive() else None
            worker.join(10)
    assert launches
    assert results[0].successful_task_ids == (task.task_id,)


def test_task_requires_native_bridge_even_with_sufficient_local_budget():
    task = make_datasets_smt_task(target=target(), expected_verdict="unsat")
    with MultiProverResourceLease(budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((task,)).receipts[0]
    assert result.status == ExecutionStatus.FAILED
    assert "datasets-backed" in result.diagnostics


def test_mutating_original_target_does_not_change_sealed_native_task(native):
    scheduler, _, _ = native
    original = target()
    task = make_datasets_smt_task(target=original, expected_verdict="unsat")
    original.update(target(text="p or q"))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((task,)).receipts[0]
    assert result.successful
    assert [item["verdict"] for item in result.result["attempts"]] == ["unsat", "unsat"]


@pytest.mark.parametrize("defect", ["exit-code", "unknown", "missing-version", "missing-attempt",
    "duplicate-attempt", "denied", "cancelled", "request-binding", "translation-binding", "input-binding"])
def test_incomplete_or_contradictory_native_observation_cannot_succeed(native, monkeypatch, defect):
    scheduler, _, _ = native
    execute = routing.run_family_portfolio
    def damaged(**kwargs):
        result = execute(**kwargs)
        first = result.attempts[0]
        if defect == "exit-code":
            first.exit_code = 1
        elif defect == "unknown":
            first.verdict = routing.models.SolverVerdict.UNKNOWN
        elif defect == "missing-version":
            first.solver_version = None
        elif defect == "missing-attempt":
            result.attempts.pop()
        elif defect == "duplicate-attempt":
            result.attempts[1] = first
        elif defect == "denied":
            result.denied.append({"solver_name": "cvc5", "reason": "unavailable"})
        elif defect == "cancelled":
            result.cancelled_attempt_ids.append(first.attempt_id)
        elif defect == "request-binding":
            first.request_id = "unrelated-request"
        elif defect == "translation-binding":
            first.translation_id = "unrelated-input"
        else:
            result.evidence[first.attempt_id].input_digest = "unrelated-input"
        return result
    monkeypatch.setattr(routing, "run_family_portfolio", damaged)
    request = make_datasets_smt_task(target=target(), expected_verdict="unsat")
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        result = BundleProverSupervisor(lease).execute((request,)).receipts[0]
    assert result.status == ExecutionStatus.FAILED
    assert result.reasons == ("smt_expected_verdict_not_observed",)
    assert not result.result["all_expected_verdicts_observed"]


@pytest.mark.parametrize("override", [
    {"expected_verdict": "proved"}, {"timeout_seconds": float("inf")}, {"timeout_seconds": True},
    {"memory_mb_per_solver": 0}, {"memory_mb_per_solver": True},
    {"max_parallel_processes": 3}, {"max_parallel_processes": True},
    {"dependencies": "opaque"}, {"dependencies": ["a"] * 65},
])
def test_bad_controls_reject_before_execution(override):
    options = dict(target=target(), expected_verdict="unsat")
    options.update(override)
    with pytest.raises((TypeError, ValueError)):
        make_datasets_smt_task(**options)


def test_atp_conjecture_verdict_cannot_enter_smt_task():
    with pytest.raises(ValueError, match="only native Z3/CVC5"):
        make_datasets_smt_task(target=target(solvers=("vampire",)), expected_verdict="unsat")


def test_unsupported_family_never_reaches_task_execution():
    value = target()
    value["logic_family"] = "temporal"
    with pytest.raises(ValueError, match="unsupported family"):
        make_datasets_smt_task(target=value, expected_verdict="unsat")
