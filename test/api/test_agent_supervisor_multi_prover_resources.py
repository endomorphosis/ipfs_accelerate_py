from __future__ import annotations

import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.multi_prover_resources import (
    PROVER_RESOURCE_CLASSES,
    BundleProverSupervisor,
    DeterministicResultCache,
    ExecutionStatus,
    MultiProverResourceBudget,
    MultiProverResourceClass,
    MultiProverResourceManager,
    ProverResourceRequest,
    ProverTask,
    ProverTaskExecutor,
    SerialProverSupervisor,
    adaptive_portfolio_width,
    dependency_closed_ready_slice,
    normalize_prover_resource_class,
)
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    HostResourceSnapshot,
    ProviderCapacity,
    ResourceLeaseBudget,
)


def _host(pressure: int = 20, *, workers: int = 8) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        observed_at_ms=1,
        cpu_percent=pressure,
        memory_percent=pressure,
        disk_percent=pressure,
        memory_total_bytes=32_000,
        memory_available_bytes=24_000,
        disk_total_bytes=64_000,
        disk_available_bytes=48_000,
        active_workers=0,
        worker_limit=workers,
        available_worker_capacity=workers,
    )


def _budget(**overrides: int) -> MultiProverResourceBudget:
    values = {
        "cpu_slots": 4,
        "process_slots": 4,
        "thread_slots": 4,
        "memory_bytes": 8_000,
        "disk_bytes": 8_000,
        "provider_quota": 4,
        "model_concurrency": 2,
        "artifact_concurrency": 2,
        "max_portfolio_width": 4,
        "wall_time_ms": 2_000,
        "max_diagnostic_bytes": 512,
    }
    values.update(overrides)
    return MultiProverResourceBudget(**values)


def _callable_task(
    task_id: str,
    family: MultiProverResourceClass = MultiProverResourceClass.SMT,
    *,
    runner=None,
    dependencies: tuple[str, ...] = (),
    critical_path_value: int = 0,
    resources: ProverResourceRequest | None = None,
    **kwargs,
) -> ProverTask:
    return ProverTask(
        task_id,
        resources
        or ProverResourceRequest.for_family(
            task_id, family, critical_path_value=critical_path_value
        ),
        runner=runner or (lambda _context: {"task": task_id}),
        dependencies=dependencies,
        **kwargs,
    )


def test_resource_taxonomy_covers_every_formal_execution_family() -> None:
    assert set(PROVER_RESOURCE_CLASSES) == {
        "translation",
        "smt",
        "atp",
        "itp_kernel",
        "jvm_model_checking",
        "protocol_verification",
        "hyperproperty_checking",
        "runtime_monitor",
        "llm_inference",
        "artifact_io",
    }
    assert normalize_prover_resource_class("z3") is MultiProverResourceClass.SMT
    assert normalize_prover_resource_class("hammer") is MultiProverResourceClass.ATP
    assert normalize_prover_resource_class("lean") is MultiProverResourceClass.ITP_KERNEL
    assert normalize_prover_resource_class("tlc") is MultiProverResourceClass.JVM_MODEL_CHECKING
    assert normalize_prover_resource_class("tamarin") is (
        MultiProverResourceClass.PROTOCOL_VERIFICATION
    )
    assert normalize_prover_resource_class("autohyper") is (
        MultiProverResourceClass.HYPERPROPERTY_CHECKING
    )
    assert normalize_prover_resource_class("mtl") is (
        MultiProverResourceClass.RUNTIME_MONITOR
    )
    assert normalize_prover_resource_class("llm") is MultiProverResourceClass.LLM_INFERENCE
    assert normalize_prover_resource_class("artifact") is MultiProverResourceClass.ARTIFACT_IO


def test_one_root_lease_accounts_process_thread_memory_disk_model_and_quota() -> None:
    manager = MultiProverResourceManager()
    lease = manager.open_lease(
        _budget(
            cpu_slots=3,
            process_slots=2,
            thread_slots=3,
            memory_bytes=1_500,
            disk_bytes=1_500,
            provider_quota=2,
            model_concurrency=1,
        ),
        host=_host(),
    )
    first_request = ProverResourceRequest(
        "model",
        MultiProverResourceClass.LLM_INFERENCE,
        cpu_slots=1,
        process_slots=0,
        thread_slots=1,
        memory_bytes=1_000,
        disk_bytes=1_000,
        provider_quota=2,
    )
    admitted, first = lease.try_acquire(first_request)
    assert admitted.admitted and first is not None
    assert lease.usage.model_slots == 1
    assert lease.usage.provider_quota == 2

    blocked, no_lease = lease.try_acquire(
        ProverResourceRequest(
            "kernel",
            MultiProverResourceClass.ITP_KERNEL,
            cpu_slots=2,
            process_slots=2,
            thread_slots=2,
            memory_bytes=600,
            disk_bytes=600,
        )
    )
    assert no_lease is None
    assert set(blocked.reasons) >= {"memory_bytes", "disk_bytes"}

    assert first.release()
    reclaimed, kernel = lease.try_acquire(
        ProverResourceRequest(
            "kernel",
            MultiProverResourceClass.ITP_KERNEL,
            cpu_slots=2,
            process_slots=2,
            thread_slots=2,
            memory_bytes=600,
            disk_bytes=600,
        )
    )
    assert reclaimed.admitted and kernel is not None
    assert kernel.release()
    assert lease.usage.process_slots == 0
    assert lease.usage.thread_slots == 0


def test_ref258_supervisor_budget_adapts_into_the_shared_prover_lease() -> None:
    upstream = ResourceLeaseBudget(
        max_parallel=3,
        max_cpu_proof_concurrency=2,
        max_model_concurrency=1,
        max_artifact_concurrency=1,
        max_processes=3,
        wall_time_ms=1_000,
        memory_bytes=4_000,
        disk_bytes=5_000,
        provider_quota=2,
    )
    lease = MultiProverResourceManager().open_lease(upstream, host=_host())
    assert lease.budget.cpu_slots == 2
    assert lease.budget.process_slots == 3
    assert lease.budget.thread_slots == 2
    assert lease.budget.max_portfolio_width == 3
    assert lease.budget.model_concurrency == 1
    assert lease.budget.memory_bytes == 4_000
    assert lease.budget.disk_bytes == 5_000


def test_child_limits_prevent_native_and_jvm_nested_pool_expansion() -> None:
    lease = MultiProverResourceManager().open_lease(_budget(), host=_host())
    _, child = lease.try_acquire(
        ProverResourceRequest(
            "tlc",
            MultiProverResourceClass.JVM_MODEL_CHECKING,
            cpu_slots=2,
            process_slots=2,
            thread_slots=2,
        )
    )
    assert child is not None
    env = child.child_environment()
    assert env["PROVER_MAX_PROCESSES"] == "2"
    assert env["PROVER_PORTFOLIO_WIDTH"] == "2"
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["RAYON_NUM_THREADS"] == "2"
    assert env["JAVA_TOOL_OPTIONS"] == "-XX:ActiveProcessorCount=2"
    child.release()


def test_serial_and_bundle_supervisors_share_the_same_live_capacity() -> None:
    lease = MultiProverResourceManager().open_lease(
        _budget(cpu_slots=1, process_slots=1, thread_slots=1),
        host=_host(),
    )
    started = threading.Event()
    release = threading.Event()

    def long_runner(_context):
        started.set()
        assert release.wait(2)
        return "serial"

    serial = SerialProverSupervisor(lease)
    bundle = BundleProverSupervisor(lease)
    serial_result: list[object] = []
    thread = threading.Thread(
        target=lambda: serial_result.append(
            serial.execute([_callable_task("serial", runner=long_runner)])
        )
    )
    thread.start()
    assert started.wait(1)

    blocked = bundle.execute([_callable_task("bundle")])
    assert blocked.receipts[0].status is ExecutionStatus.ADMISSION_REJECTED
    assert "cpu_slots" in blocked.receipts[0].reasons

    release.set()
    thread.join(2)
    assert serial_result
    assert lease.usage.cpu_slots == 0


def test_dependency_closed_slices_do_not_let_a_late_blocked_member_idle_ready_work() -> None:
    order: list[str] = []
    lock = threading.Lock()

    def record(name: str):
        def runner(_context):
            with lock:
                order.append(name)
            return name

        return runner

    tasks = [
        _callable_task("ready-first", runner=record("ready-first"), critical_path_value=5),
        _callable_task(
            "blocked-later",
            runner=record("must-not-run"),
            dependencies=("external-missing",),
            critical_path_value=100,
        ),
        _callable_task(
            "ready-after-first",
            runner=record("ready-after-first"),
            dependencies=("ready-first",),
            critical_path_value=10,
        ),
        _callable_task("independent", runner=record("independent")),
    ]
    initial = dependency_closed_ready_slice(tasks)
    assert [item.task_id for item in initial] == ["ready-first", "independent"]

    lease = MultiProverResourceManager().open_lease(_budget(), host=_host())
    result = BundleProverSupervisor(lease).execute(tasks)

    assert set(order) == {"ready-first", "ready-after-first", "independent"}
    assert order.index("ready-after-first") > order.index("ready-first")
    assert result.status_by_task_id["blocked-later"] is ExecutionStatus.BLOCKED
    assert result.status_by_task_id["ready-after-first"] is ExecutionStatus.SUCCEEDED


def test_failed_dependency_blocks_only_its_lane_while_independent_lane_completes() -> None:
    def fail(_context):
        raise RuntimeError("solver failed")

    tasks = [
        _callable_task("fail", runner=fail),
        _callable_task("dependent", dependencies=("fail",)),
        _callable_task("independent"),
    ]
    lease = MultiProverResourceManager().open_lease(_budget(), host=_host())
    result = BundleProverSupervisor(lease).execute(tasks)

    assert result.status_by_task_id["fail"] is ExecutionStatus.FAILED
    assert result.status_by_task_id["dependent"] is ExecutionStatus.BLOCKED
    assert result.status_by_task_id["independent"] is ExecutionStatus.SUCCEEDED


def test_bundle_width_never_exceeds_shared_process_or_thread_capacity() -> None:
    lease = MultiProverResourceManager().open_lease(
        _budget(
            cpu_slots=2,
            process_slots=2,
            thread_slots=2,
            max_portfolio_width=8,
        ),
        host=_host(),
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def runner(_context):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return True

    result = BundleProverSupervisor(lease).execute(
        [_callable_task(f"task-{index}", runner=runner) for index in range(6)]
    )
    assert all(item.successful for item in result.receipts)
    assert maximum <= 2
    assert lease.usage.process_slots == 0


def test_adaptive_width_contracts_under_pressure_and_values_critical_work() -> None:
    budget = _budget(max_portfolio_width=4)
    low_value = [_callable_task(f"l-{index}") for index in range(4)]
    critical = [
        _callable_task(
            f"c-{index}", critical_path_value=10 if index == 0 else 0
        )
        for index in range(4)
    ]

    low_pressure = adaptive_portfolio_width(budget, _host(10), low_value)
    high_pressure = adaptive_portfolio_width(budget, _host(85), low_value)
    critical_pressure = adaptive_portfolio_width(budget, _host(70), critical)

    assert low_pressure > high_pressure >= 1
    assert critical_pressure > adaptive_portfolio_width(
        budget, _host(70), low_value
    )
    assert adaptive_portfolio_width(budget, _host(95), critical) == 0


def test_exact_deterministic_cache_hit_bypasses_execution_and_host_pressure() -> None:
    cache = DeterministicResultCache()
    manager = MultiProverResourceManager()
    seed_lease = manager.open_lease(_budget(), host=_host())
    calls = 0

    def runner(_context):
        nonlocal calls
        calls += 1
        return {"proof": "checked"}

    seed = _callable_task(
        "cached",
        runner=runner,
        deterministic=True,
        cache_key="proof:key",
        deterministic_identity="tree+tool:v1",
    )
    assert ProverTaskExecutor(seed_lease, cache=cache).execute(seed).successful
    assert calls == 1

    pressured = manager.open_lease(_budget(), host=_host(99))
    hit = ProverTaskExecutor(pressured, cache=cache).execute(seed)
    assert hit.status is ExecutionStatus.CACHE_HIT
    assert hit.cache_bypassed_execution
    assert hit.child_lease_id == ""
    assert calls == 1

    stale = _callable_task(
        "cached",
        runner=runner,
        deterministic=True,
        cache_key="proof:key",
        deterministic_identity="tree+tool:v2",
    )
    rejected = ProverTaskExecutor(pressured, cache=cache).execute(stale)
    assert rejected.status is ExecutionStatus.ADMISSION_REJECTED
    assert calls == 1


def test_timeout_terminates_process_group_releases_capacity_and_bounds_diagnostics() -> None:
    lease = MultiProverResourceManager().open_lease(
        _budget(wall_time_ms=500, max_diagnostic_bytes=128),
        host=_host(),
    )
    script = (
        "import subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "print(p.pid, flush=True);"
        "print('x'*10000, flush=True);"
        "time.sleep(30)"
    )
    task = ProverTask.command_task(
        "timeout",
        MultiProverResourceClass.JVM_MODEL_CHECKING,
        [sys.executable, "-c", script],
        timeout_ms=100,
    )
    receipt = ProverTaskExecutor(lease).execute(task)

    assert receipt.status is ExecutionStatus.TIMED_OUT
    assert receipt.partial
    assert receipt.process_group_id is not None
    assert len(receipt.diagnostics.encode("utf-8")) <= 128
    assert lease.active_children == ()
    assert lease.usage.process_slots == 0

    again = ProverTaskExecutor(lease).execute(_callable_task("reclaimed"))
    assert again.status is ExecutionStatus.SUCCEEDED


def test_root_cancellation_terminates_running_command_and_returns_partial_receipt() -> None:
    lease = MultiProverResourceManager().open_lease(
        _budget(wall_time_ms=5_000), host=_host()
    )
    task = ProverTask.command_task(
        "cancel",
        MultiProverResourceClass.PROTOCOL_VERIFICATION,
        [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
    )
    receipts = []
    thread = threading.Thread(
        target=lambda: receipts.append(ProverTaskExecutor(lease).execute(task))
    )
    thread.start()
    deadline = time.monotonic() + 2
    while not lease.active_children and time.monotonic() < deadline:
        time.sleep(0.01)
    assert lease.active_children
    lease.cancel()
    thread.join(3)

    assert not thread.is_alive()
    assert receipts[0].status is ExecutionStatus.CANCELLED
    assert receipts[0].partial
    assert lease.active_children == ()
    assert lease.usage.process_slots == 0


def test_provider_and_model_capacity_are_reclaimed_after_inference() -> None:
    provider = ProviderCapacity(
        provider_id="model-a",
        healthy=True,
        quota_remaining=10,
        max_concurrency=1,
        active_requests=0,
    )
    lease = MultiProverResourceManager().open_lease(
        _budget(model_concurrency=1),
        host=_host(),
        providers=[provider],
    )
    request = ProverResourceRequest.for_family(
        "llm-1",
        MultiProverResourceClass.LLM_INFERENCE,
        provider_id="model-a",
    )
    _, first = lease.try_acquire(request)
    assert first is not None
    blocked, second = lease.try_acquire(
        ProverResourceRequest.for_family(
            "llm-2",
            MultiProverResourceClass.LLM_INFERENCE,
            provider_id="model-a",
        )
    )
    assert second is None
    assert set(blocked.reasons) >= {"model_concurrency", "provider_concurrency"}
    first.release()

    admitted, reclaimed = lease.try_acquire(
        ProverResourceRequest.for_family(
            "llm-2",
            MultiProverResourceClass.LLM_INFERENCE,
            provider_id="model-a",
        )
    )
    assert admitted.admitted and reclaimed is not None
    reclaimed.release()


def test_provider_telemetry_quota_is_reserved_cumulatively() -> None:
    provider = ProviderCapacity(
        provider_id="model-a",
        healthy=True,
        quota_remaining=3,
        max_concurrency=3,
        active_requests=0,
    )
    lease = MultiProverResourceManager().open_lease(
        _budget(provider_quota=10, model_concurrency=3),
        host=_host(),
        providers=[provider],
    )
    _, first = lease.try_acquire(
        ProverResourceRequest.for_family(
            "first",
            MultiProverResourceClass.LLM_INFERENCE,
            provider_id="model-a",
            provider_quota=2,
        )
    )
    assert first is not None
    blocked, second = lease.try_acquire(
        ProverResourceRequest.for_family(
            "second",
            MultiProverResourceClass.LLM_INFERENCE,
            provider_id="model-a",
            provider_quota=2,
        )
    )
    assert second is None
    assert "provider_quota_remaining" in blocked.reasons
    first.release()


def test_wall_time_is_one_root_deadline_not_a_fresh_timeout_per_child() -> None:
    lease = MultiProverResourceManager().open_lease(
        _budget(wall_time_ms=80), host=_host()
    )

    def work(_context):
        time.sleep(0.055)
        return True

    result = SerialProverSupervisor(lease).execute(
        [
            _callable_task("first", runner=work),
            _callable_task("second", runner=work),
        ]
    )
    assert result.status_by_task_id["first"] is ExecutionStatus.SUCCEEDED
    assert result.status_by_task_id["second"] in {
        ExecutionStatus.TIMED_OUT,
        ExecutionStatus.ADMISSION_REJECTED,
    }
    assert result.receipts[1].partial
    assert lease.usage.cpu_slots == 0


def test_invalid_or_nondeterministic_cache_claims_fail_closed() -> None:
    with pytest.raises(ValueError, match="require both"):
        _callable_task(
            "missing-identity",
            deterministic=True,
            cache_key="key",
        )
    with pytest.raises(ValueError, match="only valid"):
        _callable_task(
            "not-deterministic",
            cache_key="key",
            deterministic_identity="identity",
        )


def test_manager_closes_root_and_releases_all_outstanding_grants() -> None:
    manager = MultiProverResourceManager()
    lease = manager.open_lease(_budget(), host=_host())
    _, first = lease.try_acquire(
        ProverResourceRequest.for_family("one", MultiProverResourceClass.SMT)
    )
    _, second = lease.try_acquire(
        ProverResourceRequest.for_family("two", MultiProverResourceClass.ATP)
    )
    assert first is not None and second is not None
    assert manager.close(lease)
    assert lease.closed
    assert lease.active_children == ()
    assert lease.usage.cpu_slots == 0
    assert manager.active_leases == ()
