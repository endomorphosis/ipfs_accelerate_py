"""Default native bundle sizing, shared pressure and weighted execution."""
from dataclasses import replace
import json
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import datasets_native_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    ProverTask, ProverResourceRequest, ExecutionStatus,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, LeaseCancelledError,
)

MIB = 1024**2


def task(name, *, cpu=1, processes=1, memory_mb=512, threads=None, runner=None, dependencies=()):
    return ProverTask(name, ProverResourceRequest.for_family(name, "smt", cpu_slots=cpu,
        process_slots=processes, thread_slots=cpu if threads is None else threads, memory_bytes=memory_mb * MIB),
        runner=runner or (lambda _: "checked fixture"), dependencies=dependencies, timeout_ms=3000)


@pytest.fixture
def native(tmp_path):
    healthy = ProofHostResources(16, 16384, 16384)
    current = [healthy]
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "leases.json", proof_resource_sampler=lambda: current[0],
        total_cpu_slots=8, total_memory_mb=8192, total_child_process_slots=16,
        lane_reservations={"validation": {"cpu_slots": 1, "memory_mb": 256}},
        proof_backoff_seconds=.03, poll_interval_seconds=.005))
    yield scheduler, current, healthy
    deadline = time.monotonic() + 4
    while scheduler.snapshot()["active_lease_count"] and time.monotonic() < deadline:
        time.sleep(.005)
    assert scheduler.snapshot()["active_lease_count"] == 0
    assert scheduler.snapshot()["waiting_request_count"] == 0


def test_default_entrypoint_sizes_large_native_cost_and_preserves_one_safe_owner(native, monkeypatch):
    scheduler, _, _ = native
    monkeypatch.setattr(bundles, "get_global_resource_scheduler", lambda: scheduler)
    seen = []
    def observe(context):
        lease = context.lease.datasets_parent_lease
        seen.append((scheduler.snapshot()["active_root_lease_count"], lease.cpu_slots,
                     lease.child_process_slots, lease.memory_mb))
        return "native cost fixture"
    result = bundles.execute_datasets_native_bundle([
        task("large", cpu=2, processes=8, memory_mb=2304, runner=observe)])
    assert result.receipt.successful_task_ids == ("large",)
    assert seen == [(1, 2, 8, 2304)]
    assert result.plan.proof_safety_enabled
    assert result.plan.budget.memory_bytes == 2304 * MIB
    assert result.to_dict()["proof_composition_authority"] is False
    assert result.plan.to_dict()["resource_authority"] is False


def test_largest_weighted_subset_caps_parallelism_by_real_process_capacity(native):
    scheduler, _, _ = native
    tasks = [task(f"large-{n}", cpu=2, processes=8, memory_mb=2304) for n in range(4)]
    plan = bundles.plan_datasets_native_bundle(tasks, scheduler=scheduler)
    assert plan.requested_max_workers == 4 and plan.budget.max_portfolio_width == 2
    assert plan.budget.cpu_slots == 4 and plan.budget.process_slots == 16
    assert plan.budget.memory_bytes == 4608 * MIB
    assert scheduler.snapshot()["active_lease_count"] == 0


def test_mixed_costs_use_largest_subsets_instead_of_multiplying_largest_task(native):
    scheduler, _, _ = native
    tasks = [task("large", cpu=2, processes=8, memory_mb=2304),
             *[task(f"small-{n}", memory_mb=640) for n in range(3)]]
    plan = bundles.plan_datasets_native_bundle(tasks, scheduler=scheduler)
    assert plan.budget.max_portfolio_width == 4
    assert plan.budget.cpu_slots == 5 and plan.budget.process_slots == 11
    assert plan.budget.memory_bytes == (2304 + 3 * 640) * MIB


def test_actual_isabelle_profile_headroom_limits_width_and_allows_light_siblings(native):
    from ipfs_accelerate_py.agent_supervisor.proof.datasets_isabelle_tasks import make_datasets_isabelle_nat_task
    from ipfs_accelerate_py.agent_supervisor.proof.datasets_kernel_tasks import make_datasets_lean_nat_task
    from ipfs_accelerate_py.agent_supervisor.proof.datasets_tla_tasks import make_datasets_tlc_counter_task
    scheduler, _, _ = native
    heavy = [make_datasets_isabelle_nat_task(task_id=f"isabelle-{n}", offset=n) for n in range(4)]
    plan = bundles.plan_datasets_native_bundle(heavy, scheduler=scheduler)
    assert plan.budget.max_portfolio_width == 1
    assert (plan.budget.cpu_slots, plan.budget.process_slots, plan.budget.memory_bytes) == (3, 12, 2304 * MIB)
    mixed = bundles.plan_datasets_native_bundle([
        heavy[0], make_datasets_lean_nat_task(task_id="lean", offset=7),
        make_datasets_tlc_counter_task(task_id="tlc", counter_bound=2, invariant_max=2),
    ], scheduler=scheduler)
    assert mixed.budget.max_portfolio_width == 3
    assert (mixed.budget.cpu_slots, mixed.budget.process_slots, mixed.budget.memory_bytes) == (5, 14, 3584 * MIB)
    assert scheduler.snapshot()["active_lease_count"] == 0


def test_actual_execution_obeys_planned_width_without_serializing_siblings(native):
    scheduler, _, _ = native
    rendezvous = threading.Barrier(2)
    entered = []
    def run(context):
        entered.append(context.task.task_id)
        assert scheduler.snapshot()["active_root_lease_count"] == 1
        rendezvous.wait(2)
    tasks = [task(f"large-{n}", cpu=2, processes=8, memory_mb=2304, runner=run) for n in range(4)]
    result = bundles.execute_datasets_native_bundle(tasks, scheduler=scheduler)
    assert result.receipt.successful_task_ids == tuple(item.task_id for item in tasks), json.dumps(result.to_dict(), indent=2)
    assert len(entered) == 4 and result.plan.budget.max_portfolio_width == 2


def test_admission_completion_refills_capacity_before_running_callable_finishes(native, monkeypatch):
    """A conservative admission forecast must wake when its real grant arrives."""
    from ipfs_accelerate_py.agent_supervisor.proof.datasets_prover_resources import DatasetsProverResourceLease
    scheduler, _, _ = native
    first_sibling_running = threading.Event()
    third_reserved = threading.Event()
    conservative_forecast_seen = threading.Event()
    admission_callable_started = threading.Event()
    final_sibling_running = threading.Event()
    snapshots = []
    acquire = DatasetsProverResourceLease.acquire_for_execution
    ready_tasks = DatasetsProverResourceLease.ready_tasks

    def delayed_admission(owner, request, **options):
        result = acquire(owner, request, **options)
        if request.task_id == "large-2" and result[1] is not None:
            third_reserved.set()
            # Hold after local/native reservation but before the executor's
            # admission-complete event. The earlier sibling can now finish.
            assert conservative_forecast_seen.wait(1)
        return result

    def observe_forecast(owner, candidates, **options):
        selected = ready_tasks(owner, candidates, **options)
        pending_ids = {row.task_id for row in options.get("pending_requests", ())}
        if [row.task_id for row in candidates] == ["large-3"] and "large-2" in pending_ids:
            assert not selected  # real reservation + pending forecast overlap
            snapshots.append((owner.usage.cpu_slots, owner.usage.process_slots))
            conservative_forecast_seen.set()
            # Force admission to complete before returning the already-made
            # conservative decision. A fresh event check only at wait() would
            # miss this transition and still wait for callable completion.
            assert admission_callable_started.wait(1)
        return selected

    monkeypatch.setattr(DatasetsProverResourceLease, "acquire_for_execution", delayed_admission)
    monkeypatch.setattr(DatasetsProverResourceLease, "ready_tasks", observe_forecast)

    def run(context):
        name = context.task.task_id
        if name == "large-0":
            assert first_sibling_running.wait(1)
        elif name == "large-1":
            first_sibling_running.set()
            assert third_reserved.wait(1)
        elif name == "large-2":
            admission_callable_started.set()
            # This task cannot finish until readiness notices that its native
            # admission completed and schedules the final sibling.
            assert final_sibling_running.wait(.5), "admission completion did not refill the ready pool"
        else:
            final_sibling_running.set()

    tasks = [task(f"large-{n}", cpu=2, processes=8, memory_mb=2304, runner=run) for n in range(4)]
    result = bundles.execute_datasets_native_bundle(tasks, scheduler=scheduler)
    assert conservative_forecast_seen.is_set() and snapshots
    assert result.receipt.successful_task_ids == tuple(item.task_id for item in tasks), json.dumps(result.to_dict(), indent=2)


def test_existing_parent_is_authority_and_remains_live_after_bundle(native):
    scheduler, _, _ = native
    with scheduler.acquire("orchestration", cpu_slots=2, memory_mb=3000,
                           child_process_slots=8, timeout=0) as parent:
        def observe(context):
            assert scheduler.snapshot()["active_root_lease_count"] == 1
            assert context.lease.datasets_parent_lease.parent_lease_id != parent.lease_id
        result = bundles.execute_datasets_native_bundle([
            task("nested", cpu=2, processes=8, memory_mb=2304, runner=observe)], parent_lease=parent)
        assert result.receipt.successful_task_ids == ("nested",)
        assert not parent.cancelled and not parent.released
        assert scheduler.snapshot()["active_lease_count"] == 1


@pytest.mark.parametrize("pressure", [{"available_memory_mb": 100}, {"cpu_stall_percent": 90}, {"io_stall_percent": 25}])
def test_default_entrypoint_waits_for_external_pressure_and_resumes(native, monkeypatch, pressure):
    scheduler, current, healthy = native
    monkeypatch.setattr(bundles, "get_global_resource_scheduler", lambda: scheduler)
    current[0] = replace(healthy, **pressure)
    called, outcomes = threading.Event(), []
    worker = threading.Thread(target=lambda: outcomes.append(bundles.execute_datasets_native_bundle(
        [task("after-pressure", runner=lambda _: called.set())], admission_timeout_seconds=2)))
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while not scheduler.snapshot()["proof_backoff"] and time.monotonic() < deadline:
            time.sleep(.005)
        assert scheduler.snapshot()["proof_backoff"] and not called.is_set()
        current[0] = healthy
        worker.join(3)
        assert not worker.is_alive() and called.is_set()
        assert outcomes[0].receipt.successful_task_ids == ("after-pressure",)
    finally:
        current[0] = healthy
        worker.join(3)


def test_precancelled_entrypoint_never_launches_or_leaks(native):
    scheduler, _, _ = native
    signal = threading.Event()
    signal.set()
    with pytest.raises(LeaseCancelledError):
        bundles.execute_datasets_native_bundle([task("cancelled", runner=lambda _: pytest.fail("launched"))],
                                               scheduler=scheduler, cancel_event=signal)


def test_too_large_task_fails_before_admission_and_does_not_drop_it(native):
    scheduler, _, _ = native
    with pytest.raises(ValueError, match="large.*cannot fit"):
        bundles.execute_datasets_native_bundle([task("small"), task("large", memory_mb=9000)], scheduler=scheduler)
    assert scheduler.snapshot()["active_lease_count"] == 0


def test_normalized_byte_rounding_and_thread_cost_match_actual_child(native):
    scheduler, _, _ = native
    seen = []
    prepared = task("round", cpu=1, threads=2,
                    runner=lambda context: seen.append(context.lease.datasets_parent_lease.memory_mb))
    prepared = replace(prepared, resources=replace(prepared.resources, memory_bytes=MIB + 1))
    result = bundles.execute_datasets_native_bundle([prepared], scheduler=scheduler)
    assert result.plan.budget.cpu_slots == 2 and result.plan.budget.thread_slots == 2
    assert result.plan.budget.memory_bytes == 2 * MIB and seen == [2]


def test_missing_memory_uses_explicit_bounded_default(native):
    scheduler, _, _ = native
    plan = bundles.plan_datasets_native_bundle([task("default", memory_mb=0)], scheduler=scheduler,
                                              default_child_memory_mb=640)
    assert plan.budget.memory_bytes == 640 * MIB


def test_provider_model_requests_require_explicit_accounting(native):
    scheduler, _, _ = native
    prepared = ProverTask("provider", ProverResourceRequest.for_family("provider", "llm_inference"), runner=lambda _: None)
    with pytest.raises(ValueError, match="provider/model"):
        bundles.plan_datasets_native_bundle([prepared], scheduler=scheduler)


def test_failures_and_missing_dependencies_keep_complete_input_ordered_ledger(native):
    scheduler, _, _ = native
    def failed(_):
        raise ValueError("negative check")
    tasks = [task("failure", runner=failed), task("dependent", dependencies=("failure",)),
             task("missing", dependencies=("not-in-bundle",)), task("independent")]
    result = bundles.execute_datasets_native_bundle(tasks, scheduler=scheduler)
    assert [row.task_id for row in result.receipt.receipts] == [item.task_id for item in tasks]
    assert [row.status for row in result.receipt.receipts] == [ExecutionStatus.FAILED,
        ExecutionStatus.BLOCKED, ExecutionStatus.BLOCKED, ExecutionStatus.SUCCEEDED]


@pytest.mark.parametrize("tasks,options", [
    ([], {}), ([task("x")] * 2, {}), ([task(str(n)) for n in range(65)], {}),
    ([task("x")], {"max_workers": True}), ([task("x")], {"max_workers": 0}),
    ([task("x")], {"max_workers": 33}), ([task("x")], {"parent_lease": "token"}),
    ([task("x")], {"default_child_memory_mb": 0}),
])
def test_invalid_bundles_refuse_without_admission(native, tasks, options):
    scheduler, _, _ = native
    with pytest.raises((ValueError, TypeError)):
        bundles.execute_datasets_native_bundle(tasks, scheduler=scheduler, **options)
