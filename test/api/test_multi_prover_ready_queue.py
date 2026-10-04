"""Scheduling follows individual completions and fits real child costs."""
import threading
import time

from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget, MultiProverResourceLease,
    ProverResourceRequest, ProverTask,
)


def lease(**changes):
    values = dict(cpu_slots=2, process_slots=2, thread_slots=2,
                  memory_bytes=1024, max_portfolio_width=2)
    values.update(changes)
    return MultiProverResourceLease(MultiProverResourceBudget(**values))


def task(name, runner, *, dependencies=(), slots=1, memory=512, **kwargs):
    return ProverTask(name, ProverResourceRequest.for_family(name, "smt", cpu_slots=slots,
        process_slots=slots, thread_slots=slots, memory_bytes=memory), runner=runner,
        dependencies=dependencies, timeout_ms=2000, **kwargs)


def test_dependent_starts_while_unrelated_sibling_is_still_running():
    entered, dependent = threading.Event(), threading.Event()
    def slow(_):
        entered.set()
        assert dependent.wait(1), "ready task stalled behind unrelated sibling"
    def fast(_):
        assert entered.wait(1)
    def unlocked(_):
        dependent.set()
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute((
            task("a-fast", fast), task("b-slow", slow),
            task("c-dependent", unlocked, dependencies=("a-fast",))))
        assert result.successful_task_ids == ("a-fast", "b-slow", "c-dependent")
        assert not owner.active_children


def test_new_independent_work_refills_free_slot_without_batch_barrier():
    started, third = threading.Event(), threading.Event()
    def slow(_):
        started.set()
        assert third.wait(1)
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute((
            task("a-slow", slow), task("b-fast", lambda _: started.wait(1)),
            task("c-next", lambda _: third.set())))
    assert len(result.successful_task_ids) == 3


def test_wide_task_waits_for_sibling_capacity_instead_of_false_rejection():
    entered = threading.Event()
    def slow(_):
        entered.set()
        time.sleep(.06)
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute((
            task("a-fast", lambda _: entered.wait(1)), task("b-slow", slow),
            task("c-wide", lambda _: "complete", dependencies=("a-fast",), slots=2, memory=1024)))
    assert len(result.successful_task_ids) == 3
    assert result.receipts[2].started_at_ms >= result.receipts[1].finished_at_ms


def test_ready_packing_does_not_submit_two_full_envelope_requests():
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute(tuple(
            task(f"large-{n}", lambda _: time.sleep(.005), memory=1024) for n in range(4)))
    assert len(result.successful_task_ids) == 4


def test_failed_branch_stays_blocked_while_independent_chain_finishes():
    def fail(_):
        raise RuntimeError("fixture failure")
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute((
            task("failed", fail), task("healthy", lambda _: True),
            task("blocked", lambda _: False, dependencies=("failed",)),
            task("healthy-child", lambda _: True, dependencies=("healthy",))))
    assert result.status_by_task_id == {"failed": ExecutionStatus.FAILED,
        "healthy": ExecutionStatus.SUCCEEDED, "blocked": ExecutionStatus.BLOCKED,
        "healthy-child": ExecutionStatus.SUCCEEDED}


def test_cache_hit_unlocks_child_without_waiting_for_running_sibling():
    event = threading.Event()
    def held(_):
        assert event.wait(1)
    cached = task("cached", lambda _: True, deterministic=True, cache_key="key", deterministic_identity="identity")
    with lease() as owner:
        supervisor = BundleProverSupervisor(owner)
        assert supervisor.execute((cached,)).successful_task_ids == ("cached",)
        result = supervisor.execute((task("a-held", held), cached,
            task("z-child", lambda _: event.set(), dependencies=("cached",))))
    assert result.receipts[1].status == ExecutionStatus.CACHE_HIT
    assert len(result.successful_task_ids) == 3


def test_precancelled_queue_has_complete_ordered_ledger_without_launch():
    cancellation = threading.Event()
    cancellation.set()
    called = []
    jobs = tuple(task(f"node-{n}", lambda _: called.append(True)) for n in range(8))
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute(jobs, cancellation=cancellation)
    assert not called and result.cancelled
    assert [row.task_id for row in result.receipts] == [job.task_id for job in jobs]
    assert all(row.status == ExecutionStatus.CANCELLED for row in result.receipts)


def test_intrinsically_oversized_task_retains_real_admission_reason():
    with lease() as owner:
        result = BundleProverSupervisor(owner).execute((task("large", lambda _: True, memory=2048),))
    assert result.receipts[0].status == ExecutionStatus.ADMISSION_REJECTED
    assert "memory_bytes" in result.receipts[0].reasons


def test_empty_and_missing_dependency_bundles_terminate_with_complete_dispositions():
    with lease() as owner:
        supervisor = BundleProverSupervisor(owner)
        assert supervisor.execute(()).receipts == ()
        result = supervisor.execute((task("unready", lambda _: True, dependencies=("missing",)),))
    assert result.receipts[0].status == ExecutionStatus.BLOCKED
    assert result.receipts[0].reasons == ("dependency:missing",)
