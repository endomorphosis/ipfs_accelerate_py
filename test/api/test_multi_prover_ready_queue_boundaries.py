"""Adversarial readiness tests spanning submitted, admitted and retained work."""
import threading
import time

from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    BundleProverSupervisor, ExecutionStatus, MultiProverResourceBudget,
    MultiProverResourceLease, ProverResourceRequest, ProverTask,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import ProviderCapacity


def owner(**changes):
    budget = dict(cpu_slots=2, process_slots=2, thread_slots=2,
                  memory_bytes=200, max_portfolio_width=2)
    budget.update(changes)
    return MultiProverResourceLease(MultiProverResourceBudget(**budget))


def task(name, function, *, cpu=1, processes=1, memory=100, family="smt", **changes):
    request = ProverResourceRequest.for_family(name, family, cpu_slots=cpu,
        process_slots=processes, thread_slots=cpu, memory_bytes=memory, **changes)
    return ProverTask(name, request, runner=function, timeout_ms=2000)


def test_submitted_but_not_admitted_sibling_retains_its_scheduling_cost(monkeypatch):
    sibling_waiting, allow_sibling = threading.Event(), threading.Event()
    sibling_finished, wide_entered = threading.Event(), threading.Event()
    with owner() as lease:
        supervisor = BundleProverSupervisor(lease)
        execute = supervisor.executor.execute
        def delayed(job, **options):
            if job.task_id == "b-delayed-admission":
                sibling_waiting.set()
                assert allow_sibling.wait(1)
                try:
                    return execute(job, **options)
                finally:
                    sibling_finished.set()
            return execute(job, **options)
        monkeypatch.setattr(supervisor.executor, "execute", delayed)
        def fast(_):
            assert sibling_waiting.wait(1)
        def wide(_):
            wide_entered.set()
            assert sibling_finished.wait(1)
        # If a wide task improperly overtakes the already-submitted sibling,
        # allow that sibling to attempt admission while wide owns the capacity.
        # With correct accounting, the sibling instead starts after this short
        # deadline, drains, and only then allows the wide task to start.
        def resume():
            wide_entered.wait(.2)
            allow_sibling.set()
        controller = threading.Thread(target=resume)
        controller.start()
        try:
            receipt = supervisor.execute((task("a-fast", fast),
                task("b-delayed-admission", lambda _: None),
                task("c-wide", wide, cpu=2, processes=2, memory=200)))
        finally:
            allow_sibling.set()
            controller.join(2)
    assert receipt.successful_task_ids == ("a-fast", "b-delayed-admission", "c-wide")


def test_queued_process_task_does_not_block_ready_inprocess_work():
    monitor_entered = threading.Event()
    def solver(_):
        assert monitor_entered.wait(.3), "free thread stalled behind process-slot demand"
    with owner(process_slots=1) as lease:
        receipt = BundleProverSupervisor(lease).execute((
            task("a-running-solver", solver),
            task("b-next-solver", lambda _: None),
            task("z-monitor", lambda _: monitor_entered.set(), processes=0, family="runtime_monitor"),
        ))
    assert len(receipt.successful_task_ids) == 3


def test_provider_quota_weighting_is_reserved_across_the_ready_subset():
    provider = ProviderCapacity("model", quota_remaining=3, max_concurrency=3)
    budget = MultiProverResourceBudget(cpu_slots=3, process_slots=3, thread_slots=3,
        model_concurrency=3, max_portfolio_width=3, memory_bytes=300)
    active = peak = 0
    lock = threading.Lock()
    def infer(_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.01)
        finally:
            with lock:
                active -= 1
    jobs = tuple(task(f"inference-{n}", infer, processes=0, family="llm_inference",
        provider_id="model", provider_quota=2) for n in range(4))
    with MultiProverResourceLease(budget, providers=(provider,)) as lease:
        result = BundleProverSupervisor(lease).execute(jobs)
        assert not lease.active_children
    assert len(result.successful_task_ids) == 4
    assert peak == 1 and active == 0


def test_cancelled_completion_does_not_unlock_its_dependent():
    cancellation = threading.Event()
    launched = []
    def cancel(_):
        cancellation.set()
        return "late successful value"
    original = task("first", cancel)
    child = ProverTask("dependent", ProverResourceRequest.for_family("dependent", "smt",
        memory_bytes=100), runner=lambda _: launched.append(True), dependencies=("first",))
    with owner() as lease:
        result = BundleProverSupervisor(lease).execute((original, child), cancellation=cancellation)
    assert result.receipts[0].status == ExecutionStatus.CANCELLED
    assert result.receipts[1].status == ExecutionStatus.CANCELLED
    assert not launched


def test_retained_timed_out_callable_does_not_free_capacity_for_pending_task():
    stop = threading.Event()
    entered = threading.Event()
    launched = []
    def held(_):
        entered.set()
        stop.wait(2)
    first = ProverTask("a-held", ProverResourceRequest.for_family("a-held", "smt",
        cpu_slots=2, process_slots=2, thread_slots=2, memory_bytes=200), runner=held, timeout_ms=30)
    with owner() as lease:
        try:
            result = BundleProverSupervisor(lease).execute((first,
                task("b-next", lambda _: launched.append(True))))
            assert entered.is_set()
            assert result.receipts[0].status == ExecutionStatus.TIMED_OUT
            # A timed-out callable sets the bundle cancellation signal while
            # its retained execution drains; pending work must not launch.
            assert result.receipts[1].status == ExecutionStatus.CANCELLED
            assert result.cancelled
            assert lease.usage.cpu_slots == 2
            assert not launched
        finally:
            stop.set()
            deadline = time.monotonic() + 2
            while lease.active_children and time.monotonic() < deadline:
                time.sleep(.005)
        assert not lease.active_children
