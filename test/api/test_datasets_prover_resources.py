"""Cross-package admission, backoff, cancellation and retained ownership."""
from dataclasses import replace
import threading
import time
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import open_datasets_prover_lease
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget,
    ProverResourceRequest, ProverTask, ProverTaskFailure,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


@pytest.fixture
def owner(tmp_path):
    healthy = ProofHostResources(8, 8192, 8192)
    current = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", proof_resource_sampler=lambda: current[0],
        total_cpu_slots=2, total_memory_mb=2048, total_child_process_slots=2,
        lane_reservations={}, proof_backoff_seconds=.03, poll_interval_seconds=.005))
    yield scheduler, current, healthy
    until(lambda: scheduler.snapshot()["active_lease_count"] == 0)
    assert scheduler.snapshot()["waiting_request_count"] == 0


def budget(width=1):
    return MultiProverResourceBudget(cpu_slots=width, thread_slots=width,
        process_slots=width, memory_bytes=512 * 1024**2 * width, max_portfolio_width=width)


def task(runner, *, timeout_ms=1000):
    return ProverTask("fixture", ProverResourceRequest.for_family("fixture", "smt"),
        runner=runner, timeout_ms=timeout_ms)


def test_defaults_use_actual_global_owner_with_safety_and_finite_ram(owner, monkeypatch):
    scheduler, _, _ = owner
    from ipfs_accelerate_py.agent_supervisor.proof import datasets_prover_resources as module
    monkeypatch.setattr(module, "get_global_resource_scheduler", lambda: scheduler)
    with open_datasets_prover_lease() as lease:
        assert lease.datasets_scheduler is scheduler and scheduler.config.proof_safety_enabled
        assert lease.budget.cpu_slots == 2 and lease.budget.memory_bytes == 1024 * 1024**2
        admission, child = lease.try_acquire(ProverResourceRequest.for_family("default-ram", "smt"))
        assert admission.admitted and child.datasets_parent_lease.memory_mb == 512
        assert scheduler.snapshot()["active_root_lease_count"] == 1
        assert scheduler.snapshot()["allocated"] == {"cpu_slots": 2, "memory_mb": 1024}
        child.release()


def test_bridge_is_one_child_of_existing_parent_and_does_not_release_parent(owner):
    scheduler, _, _ = owner
    with scheduler.acquire("orchestration", cpu_slots=2, memory_mb=1024,
            child_process_slots=2, timeout=0) as parent:
        with open_datasets_prover_lease(parent_lease=parent, budget=budget()) as lease:
            assert lease.datasets_parent_lease.parent_lease_id == parent.lease_id
            assert scheduler.snapshot()["active_root_lease_count"] == 1
        assert not parent.released and not parent.cancelled
        assert scheduler.snapshot()["active_lease_count"] == 1


@pytest.mark.parametrize("pressure", [
    {"available_memory_mb": 128}, {"memory_stall_percent": 5},
    {"cpu_stall_percent": 90}, {"io_stall_percent": 25},
])
def test_pressure_defers_task_then_recovers_without_rebuilding_it(owner, pressure):
    scheduler, current, healthy = owner
    ran = threading.Event()
    results = []
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=2) as lease:
        current[0] = replace(healthy, **pressure)
        worker = threading.Thread(target=lambda: results.append(BundleProverSupervisor(lease).execute(
            (task(lambda context: ran.set(), timeout_ms=2000),))))
        worker.start()
        try:
            until(lambda: bool(scheduler.snapshot()["proof_backoff"]))
            assert not ran.is_set()
            current[0] = healthy
            worker.join(3)
            assert not worker.is_alive() and ran.is_set()
            assert results[0].receipts[0].status == ExecutionStatus.SUCCEEDED
        finally:
            current[0] = healthy
            lease.cancel()
            worker.join(3)


@pytest.mark.parametrize("interrupt", ["cancel", "task-timeout"])
def test_pending_native_admission_observes_task_cancel_and_task_deadline(owner, interrupt):
    scheduler, current, healthy = owner
    cancel = threading.Event()
    called = threading.Event()
    results = []
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=10) as lease:
        current[0] = replace(healthy, cpu_stall_percent=90)
        worker = threading.Thread(target=lambda: results.append(BundleProverSupervisor(lease).execute(
            (task(lambda _: called.set(), timeout_ms=150 if interrupt == "task-timeout" else 5000),),
            cancellation=cancel)))
        started = time.monotonic()
        worker.start()
        try:
            until(lambda: bool(scheduler.snapshot()["proof_backoff"]))
            if interrupt == "cancel":
                cancel.set()
            worker.join(2)
            assert not worker.is_alive() and not called.is_set()
            assert time.monotonic() - started < 2
            expected = ExecutionStatus.CANCELLED if interrupt == "cancel" else ExecutionStatus.TIMED_OUT
            assert results[0].receipts[0].status == expected
            assert scheduler.snapshot()["active_lease_count"] == 1
        finally:
            current[0] = healthy
            lease.cancel()
            worker.join(3)


def test_external_parent_release_does_not_reclaim_timed_out_callable(owner):
    scheduler, _, _ = owner
    entered, finish = threading.Event(), threading.Event()
    parent = scheduler.acquire("orchestration", cpu_slots=2, memory_mb=1024, child_process_slots=2, timeout=0)
    lease = open_datasets_prover_lease(parent_lease=parent, budget=budget())
    def lingering(context):
        entered.set()
        finish.wait(3)
        return "late result"
    try:
        result = BundleProverSupervisor(lease).execute((task(lingering, timeout_ms=60),))
        assert entered.is_set() and result.receipts[0].status == ExecutionStatus.TIMED_OUT
        parent.release()
        lease.close()
        state = scheduler.snapshot()
        assert state["allocated"] == {"cpu_slots": 2, "memory_mb": 1024}
        assert state["active_lease_count"] == 3
        assert scheduler.try_acquire("orchestration", memory_mb=128) is None
    finally:
        finish.set()
        lease.close()
        parent.release()
    until(lambda: scheduler.snapshot()["active_lease_count"] == 0)


def test_structured_failure_never_unlocks_dependency_or_populates_cache(owner):
    scheduler, _, _ = owner
    def fail(_):
        raise ProverTaskFailure("candidate rejected", result={"verdict": "unknown"}, reasons=("unknown",))
    first = replace(task(fail), deterministic=True, cache_key="candidate", deterministic_identity="digest")
    next_task = ProverTask("next", ProverResourceRequest.for_family("next", "smt"),
        runner=lambda _: pytest.fail("failed candidate unlocked dependent task"), dependencies=("fixture",))
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        supervisor = BundleProverSupervisor(lease)
        result = supervisor.execute((first, next_task))
        assert result.receipts[0].result == {"verdict": "unknown"}
        assert result.receipts[0].status == ExecutionStatus.FAILED
        assert result.receipts[1].status == ExecutionStatus.BLOCKED
        assert len(supervisor.executor.cache) == 0


def test_byte_rounding_and_thread_count_cannot_undercharge_native_work(owner):
    scheduler, _, _ = owner
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(2)) as lease:
        request = ProverResourceRequest.for_family("rounded", "smt", cpu_slots=1, thread_slots=2,
            memory_bytes=1024**2 + 1)
        admission, child = lease.try_acquire(request)
        assert admission.admitted
        assert child.datasets_parent_lease.memory_mb == 2
        assert child.datasets_parent_lease.cpu_slots == 2
        child.release()


def test_router_exposes_same_native_child_to_runner(owner):
    scheduler, _, _ = owner
    from test.api.test_agent_supervisor_multi_prover_router import _one_lane_router, _obligation
    from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import PropertyKind, ProverOutput, AttemptOutcome
    seen = []
    def run(request, cancellation):
        native = request.resource_lease.datasets_parent_lease
        assert request.resource_lease.datasets_scheduler is scheduler
        with native.acquire_child(memory_mb=128, child_process_slots=1, timeout=0):
            assert scheduler.snapshot()["active_root_lease_count"] == 1
            seen.append(native.lease_id)
        return ProverOutput(AttemptOutcome.UNKNOWN, "resource fixture only")
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget()) as lease:
        _one_lane_router(resource_lease=lease).execute(_obligation(PropertyKind.KERNEL_CHECK), run)
        assert len(seen) == 1 and not lease.active_children
        assert scheduler.snapshot()["active_lease_count"] == 1


def test_router_timeout_cancels_waiting_native_admission_without_late_launch(owner):
    scheduler, current, healthy = owner
    from test.api.test_agent_supervisor_multi_prover_router import _one_lane_router, _obligation
    from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import PropertyKind
    called = threading.Event()
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(), admission_timeout_seconds=10) as lease:
        current[0] = replace(healthy, cpu_stall_percent=90)
        try:
            result = _one_lane_router(timeout_seconds=.1, resource_lease=lease).execute(
                _obligation(PropertyKind.KERNEL_CHECK), lambda *_: called.set())
            until(lambda: not lease.active_children)
            assert not called.is_set() and result.verdict.value != "proved"
            assert not result.authority_attempt_ids
            assert scheduler.snapshot()["waiting_request_count"] == 0
        finally:
            current[0] = healthy


def test_command_admission_stays_charged_during_close_spawn_race(owner, monkeypatch):
    scheduler, _, _ = owner
    from ipfs_accelerate_py.agent_supervisor.proof import multi_prover_resources as resources
    entered, proceed = threading.Event(), threading.Event()
    native_popen = resources.subprocess.Popen
    processes, results = [], []
    def gated_popen(*args, **kwargs):
        entered.set()
        assert proceed.wait(3)
        process = native_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(resources.subprocess, "Popen", gated_popen)
    command = ProverTask("native-command", ProverResourceRequest.for_family("native-command", "smt"),
        command=(sys.executable, "-c", "import time; time.sleep(30)"), timeout_ms=5000)
    lease = open_datasets_prover_lease(scheduler=scheduler, budget=budget())
    worker = threading.Thread(target=lambda: results.append(BundleProverSupervisor(lease).execute((command,))))
    worker.start()
    try:
        assert entered.wait(3)
        lease.close()
        assert scheduler.snapshot()["active_lease_count"] == 2
        assert lease.usage.process_slots == 1
    finally:
        proceed.set()
        worker.join(5)
        lease.close()
    assert not worker.is_alive() and processes and all(process.poll() is not None for process in processes)
    assert results[0].receipts[0].status == ExecutionStatus.CANCELLED


def test_repeated_bundles_drain_every_child_without_reallocating_root(owner):
    scheduler, _, _ = owner
    with open_datasets_prover_lease(scheduler=scheduler, budget=budget(2)) as lease:
        supervisor = BundleProverSupervisor(lease)
        for iteration in range(16):
            jobs = [ProverTask(f"{iteration}:{n}", ProverResourceRequest.for_family(f"{iteration}:{n}", "smt"),
                runner=lambda context: time.sleep(.005)) for n in range(4)]
            result = supervisor.execute(jobs)
            assert len(result.successful_task_ids) == 4
            assert not lease.active_children and lease.usage.cpu_slots == 0
            assert scheduler.snapshot()["active_lease_count"] == 1


@pytest.mark.parametrize("arguments", [
    {"parent_lease": "opaque"}, {"scheduler": object()}, {"cancel_event": object()},
    {"timeout_seconds": float("nan")}, {"admission_timeout_seconds": 0},
    {"default_child_memory_mb": True}, {"budget": MultiProverResourceBudget()},
    {"budget": MultiProverResourceBudget(thread_slots=2, memory_bytes=1024)},
])
def test_bad_authority_and_budgets_fail_before_launch(arguments):
    with pytest.raises((TypeError, ValueError)):
        open_datasets_prover_lease(**arguments)
