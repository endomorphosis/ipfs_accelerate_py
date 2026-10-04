"""Live callable work remains accounted after timeout, cancellation and close."""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_resources import (
    DeterministicResultCache,
    ExecutionStatus,
    MultiProverResourceBudget,
    MultiProverResourceManager,
    ProverResourceRequest,
    ProverTask,
    ProverTaskExecutor,
)


def _lease(*, wall_time_ms=5_000):
    manager = MultiProverResourceManager()
    lease = manager.open_lease(MultiProverResourceBudget(
        cpu_slots=1, process_slots=1, thread_slots=1, memory_bytes=1024,
        wall_time_ms=wall_time_ms, max_portfolio_width=1,
    ))
    return manager, lease


def _task(task_id, runner, *, timeout_ms=0, deterministic=False):
    return ProverTask(
        task_id,
        ProverResourceRequest.for_family(task_id, "smt", memory_bytes=1024),
        runner=runner, timeout_ms=timeout_ms, deterministic=deterministic,
        cache_key="callable-key" if deterministic else "",
        deterministic_identity="callable-identity" if deterministic else "",
    )


def _drained(lease):
    deadline = time.monotonic() + 3
    while lease.active_children and time.monotonic() < deadline:
        time.sleep(0.005)
    assert lease.active_children == ()
    assert lease.usage.cpu_slots == 0
    assert lease.usage.process_slots == 0
    assert lease.usage.memory_bytes == 0


@pytest.mark.parametrize("root_deadline", [False, True])
def test_timeout_retains_capacity_until_worker_exit_and_discards_late_cache_result(root_deadline):
    manager, lease = _lease(wall_time_ms=80 if root_deadline else 5_000)
    entered, release = threading.Event(), threading.Event()
    cache = DeterministicResultCache()
    executor = ProverTaskExecutor(lease, cache=cache)

    def ignores_cancellation(context):
        entered.set()
        assert release.wait(3)
        assert context.cancelled
        return {"late": True}

    task = _task("timed-out", ignores_cancellation,
                 timeout_ms=0 if root_deadline else 80, deterministic=True)
    started = time.monotonic()
    try:
        receipt = executor.execute(task)
        assert entered.is_set()
        assert time.monotonic() - started < 1
        assert receipt.status is ExecutionStatus.TIMED_OUT
        assert receipt.partial and receipt.result is None
        assert "callable_cleanup_pending" in receipt.reasons
        assert "lease retained" in receipt.diagnostics
        assert lease.usage.cpu_slots == 1
        assert lease.usage.process_slots == 1
        assert lease.usage.memory_bytes == 1024
        assert not lease.active_children[0].release()
        assert lease.usage.cpu_slots == 1
        assert manager.get(lease.lease_id) is lease
        assert manager.active_leases == (lease,)
        blocked = executor.execute(_task("retry-too-early", lambda _: True))
        assert blocked.status is ExecutionStatus.ADMISSION_REJECTED
        assert ("lease_wall_time" if root_deadline else "cpu_slots") in blocked.reasons
        assert len(cache) == 0
    finally:
        release.set()
        _drained(lease)
    assert len(cache) == 0
    if not root_deadline:
        fresh = executor.execute(_task("fresh", lambda _: {"fresh": True}, deterministic=True))
        assert fresh.status is ExecutionStatus.SUCCEEDED
        assert fresh.result == {"fresh": True}
        assert len(cache) == 1
    manager.close(lease)


@pytest.mark.parametrize("interruption", ["timeout", "external_cancel", "close_root"])
def test_interruption_keeps_real_native_work_accounted_until_callable_joins_it(interruption):
    manager, lease = _lease()
    entered, release, cancellation = threading.Event(), threading.Event(), threading.Event()
    processes, receipts = [], []

    def native_work(_context):
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        processes.append(process)
        entered.set()
        try:
            assert release.wait(3)
            process.communicate(b"x", timeout=2)
            return {"native_exit_code": process.returncode}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    executor = ProverTaskExecutor(lease)
    caller = threading.Thread(target=lambda: receipts.append(executor.execute(
        _task("native", native_work, timeout_ms=120 if interruption == "timeout" else 0),
        cancellation=cancellation,
    )))
    caller.start()
    try:
        assert entered.wait(2)
        if interruption == "close_root":
            assert manager.close(lease)
            assert lease.closed
        elif interruption == "external_cancel":
            cancellation.set()
        caller.join(1)
        assert not caller.is_alive()
        assert len(receipts) == 1
        assert receipts[0].status is (
            ExecutionStatus.TIMED_OUT if interruption == "timeout" else ExecutionStatus.CANCELLED
        )
        assert "callable_cleanup_pending" in receipts[0].reasons
        assert processes[0].poll() is None
        assert lease.usage.cpu_slots == lease.usage.process_slots == 1
        assert lease.usage.memory_bytes == 1024
        assert manager.get(lease.lease_id) is lease
        assert manager.active_leases == (lease,)
        if interruption == "close_root":
            # Repeat close must not treat a live held callable as stale work.
            assert manager.close(lease)
            assert lease.usage.cpu_slots == 1
        blocked = executor.execute(_task("overlap", lambda _: True))
        assert blocked.status in {ExecutionStatus.CANCELLED, ExecutionStatus.ADMISSION_REJECTED}
    finally:
        release.set()
        caller.join(3)
        _drained(lease)
        manager.close(lease)
    assert processes[0].poll() == 0
    assert manager.get(lease.lease_id) is None
    assert manager.active_leases == ()


def test_cancellation_during_worker_completion_cannot_publish_success_or_cache():
    manager, lease = _lease()
    cancellation = threading.Event()
    cache = DeterministicResultCache()

    def finishes_after_cancelling(_context):
        cancellation.set()
        return {"late": True}

    receipt = ProverTaskExecutor(lease, cache=cache).execute(
        _task("cancel-at-return", finishes_after_cancelling, deterministic=True),
        cancellation=cancellation,
    )
    assert receipt.status is ExecutionStatus.CANCELLED
    assert receipt.result is None and receipt.partial
    _drained(lease)
    assert len(cache) == 0
    manager.close(lease)


def test_callable_start_failure_releases_execution_hold(monkeypatch):
    manager, lease = _lease()
    original = threading.Thread.start

    def fail_start(thread):
        if thread.name == "prover-start-failure":
            raise RuntimeError("thread start rejected")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start rejected"):
        ProverTaskExecutor(lease).execute(_task("start-failure", lambda _: True))
    _drained(lease)
    manager.close(lease)


def test_callable_failure_releases_capacity_without_waiting_for_root_close():
    manager, lease = _lease()

    def failure(_context):
        raise ValueError("runner failure")

    receipt = ProverTaskExecutor(lease).execute(_task("failure", failure))
    assert receipt.status is ExecutionStatus.FAILED
    assert "runner_error" in receipt.reasons
    _drained(lease)
    assert ProverTaskExecutor(lease).execute(_task("next", lambda _: True)).successful
    manager.close(lease)
