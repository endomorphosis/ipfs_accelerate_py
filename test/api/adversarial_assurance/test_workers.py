"""Tests for parallel mutation workers, resource admission, timeout, and cancellation (AAE-042)."""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.workers import (
    AAE_MUTATION_WORKERS_EVIDENCE,
    MUTATION_WORKER_POOL_INTERFACE,
    MutationWorkerBoundsError,
    MutationWorkerBudget,
    MutationWorkerCancellation,
    MutationWorkerCheckpointStore,
    MutationWorkerContext,
    MutationWorkerDisposition,
    MutationWorkerError,
    MutationWorkerPolicyError,
    MutationWorkerPool,
    MutationWorkerTask,
    mutation_worker_pool_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_TREE_CANCELLATION_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _host(*, worker_limit: int = 8) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=worker_limit,
        available_worker_capacity=worker_limit,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-large",
            "cpu-small",
            "cpu-medium",
            "cpu-validation",
            "cpu-proof-solver",
        ),
    )


def _pool(
    tmp_path: Path | None = None,
    *,
    max_concurrency: int = 2,
    default_timeout_seconds: float = 5.0,
    host: HostResourceSnapshot | None = None,
    scheduler: ResourceScheduler | None = None,
    **kwargs: Any,
) -> MutationWorkerPool:
    checkpoint = None
    if tmp_path is not None:
        checkpoint = tmp_path / "worker-checkpoints"
    return MutationWorkerPool.create(
        max_concurrency=max_concurrency,
        default_timeout_seconds=default_timeout_seconds,
        checkpoint_dir=checkpoint,
        resource_scheduler=scheduler
        or ResourceScheduler(ResourcePolicy(max_lanes=max_concurrency)),
        host_snapshot=host or _host(worker_limit=max(max_concurrency, 4)),
        **kwargs,
    )


def _ok_runner(context: MutationWorkerContext) -> dict[str, Any]:
    context.check_cancelled()
    return {
        "ok": True,
        "task_id": context.task.task_id,
        "lease_id": context.lease_id,
        "network_policy": context.network_policy,
    }


def _fail_runner(context: MutationWorkerContext) -> dict[str, Any]:
    context.check_cancelled()
    return {"ok": False, "reason": "synthetic_failure"}


def _slow_runner(
    context: MutationWorkerContext, *, seconds: float = 10.0
) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        context.check_cancelled()
        time.sleep(0.05)
    return {"ok": True, "slow": True}


# ---------------------------------------------------------------------------
# Cold import / descriptor / contracts
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.workers"
    )
    assert module.MUTATION_WORKER_POOL_INTERFACE == "MutationWorkerPool@1"
    assert module.AAE_MUTATION_WORKERS_EVIDENCE == "aae/mutation-workers@1"


def test_descriptor_declares_reuse_and_invariants() -> None:
    descriptor = mutation_worker_pool_descriptor()
    assert descriptor["interface"] == MUTATION_WORKER_POOL_INTERFACE
    assert descriptor["evidence"] == AAE_MUTATION_WORKERS_EVIDENCE
    assert (
        descriptor["process_tree_cancellation_evidence"]
        == PROCESS_TREE_CANCELLATION_EVIDENCE
    )
    assert descriptor["network_policy"] == NETWORK_POLICY_DENY_ALL
    invariants = set(descriptor["invariants"])
    assert "reuses_resource_scheduler_for_admission" in invariants
    assert "reuses_process_tree_cancellation" in invariants
    assert "records_infrastructure_separately" in invariants
    assert "restartable_via_checkpoint_journal" in invariants
    assert "leak_free_shutdown_releases_leases_and_fences_trees" in invariants
    reuses = set(descriptor["reuses"])
    assert "ResourceScheduler@1" in reuses
    assert "ivp/process-tree-cancellation@1" in reuses


def test_budget_rejects_network_policy_widen() -> None:
    with pytest.raises(MutationWorkerPolicyError) as excinfo:
        MutationWorkerBudget(network_policy="allow_all")
    assert excinfo.value.reason_code == "network_policy_denied"


def test_budget_rejects_invalid_concurrency() -> None:
    with pytest.raises(MutationWorkerBoundsError):
        MutationWorkerBudget(max_concurrency=0)
    with pytest.raises(MutationWorkerBoundsError):
        MutationWorkerBudget(max_concurrency=-1)


def test_task_requires_exactly_one_of_runner_or_command() -> None:
    with pytest.raises(MutationWorkerError) as excinfo:
        MutationWorkerTask(task_id="t0")
    assert excinfo.value.reason_code == "invalid_task"
    with pytest.raises(MutationWorkerError):
        MutationWorkerTask(
            task_id="t1",
            runner=_ok_runner,
            command=[sys.executable, "-c", "print(1)"],
        )


def test_task_rejects_network_policy_widen() -> None:
    with pytest.raises(MutationWorkerPolicyError):
        MutationWorkerTask(
            task_id="t-net",
            runner=_ok_runner,
            network_policy="allow_egress",
        )


# ---------------------------------------------------------------------------
# Happy path / concurrency / resource admission
# ---------------------------------------------------------------------------


def test_callable_happy_path_records_infrastructure_separately(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    try:
        result = pool.run(
            MutationWorkerTask(
                task_id="happy-1",
                runner=_ok_runner,
                candidate_id="cand-1",
                candidate_cid="cid-1",
                metadata={"phase": "unit"},
            )
        )
        assert result.disposition is MutationWorkerDisposition.COMPLETED
        assert result.is_infrastructure is False
        assert result.publication_allowed is True
        assert result.payload is not None
        assert result.payload["ok"] is True
        assert result.payload["network_policy"] == NETWORK_POLICY_DENY_ALL
        assert result.infrastructure.disposition is MutationWorkerDisposition.COMPLETED
        assert result.infrastructure.lease_id
        assert result.infrastructure.admission_admitted is True
        assert result.infrastructure.network_policy == NETWORK_POLICY_DENY_ALL
        assert result.infrastructure.is_infrastructure is False
        body = result.to_dict()
        assert "infrastructure" in body
        assert body["payload"]["ok"] is True
        assert body["is_infrastructure"] is False
        assert body["evidence"] == AAE_MUTATION_WORKERS_EVIDENCE
    finally:
        pool.shutdown(wait=True)


def test_runner_reported_failure_is_semantic_failed(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        result = pool.run(
            MutationWorkerTask(task_id="fail-1", runner=_fail_runner)
        )
        assert result.disposition is MutationWorkerDisposition.FAILED
        assert result.is_infrastructure is False
        assert result.publication_allowed is True
        assert result.payload is not None
        assert result.payload["ok"] is False
        assert "runner_reported_failure" in result.infrastructure.reason_codes
    finally:
        pool.shutdown(wait=True)


def test_map_runs_in_parallel_under_concurrency_cap(tmp_path: Path) -> None:
    pool = _pool(tmp_path, max_concurrency=2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def gated(context: MutationWorkerContext) -> dict[str, Any]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.15)
            context.check_cancelled()
            return {"ok": True, "task_id": context.task.task_id}
        finally:
            with lock:
                active -= 1

    tasks = [
        MutationWorkerTask(task_id=f"p-{index}", runner=gated)
        for index in range(4)
    ]
    try:
        results = pool.map(tasks)
        assert len(results) == 4
        assert all(
            item.disposition is MutationWorkerDisposition.COMPLETED
            for item in results
        )
        assert peak <= 2
        assert peak >= 2  # exercised parallel capacity
    finally:
        pool.shutdown(wait=True)


def test_resource_lease_acquired_and_released(tmp_path: Path) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=2))
    pool = _pool(tmp_path, max_concurrency=2, scheduler=scheduler)
    try:
        result = pool.run(
            MutationWorkerTask(task_id="lease-1", runner=_ok_runner)
        )
        assert result.disposition is MutationWorkerDisposition.COMPLETED
        assert result.infrastructure.lease_id
        # Lease must be released after completion so capacity is reusable.
        assert scheduler.active_leases == ()
        result2 = pool.run(
            MutationWorkerTask(task_id="lease-2", runner=_ok_runner)
        )
        assert result2.disposition is MutationWorkerDisposition.COMPLETED
        assert scheduler.active_leases == ()
    finally:
        pool.shutdown(wait=True)


def test_resource_lease_denial_is_resource_denied(tmp_path: Path) -> None:
    # Exhaust capacity: host with zero free workers.
    exhausted = HostResourceSnapshot(
        worker_limit=1,
        available_worker_capacity=0,
        active_workers=1,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=("cpu-large", "cpu-small"),
    )
    pool = _pool(
        tmp_path,
        max_concurrency=1,
        host=exhausted,
        scheduler=ResourceScheduler(ResourcePolicy(max_lanes=1)),
    )
    try:
        result = pool.run(
            MutationWorkerTask(task_id="denied-1", runner=_ok_runner)
        )
        assert result.disposition is MutationWorkerDisposition.RESOURCE_DENIED
        assert result.is_infrastructure is True
        assert result.publication_allowed is False
        assert result.payload is None
        assert "resource_lease_denied" in result.infrastructure.reason_codes
        assert result.infrastructure.admission_admitted is False
    finally:
        pool.shutdown(wait=True)


def test_concurrency_budget_via_scheduler_max_lanes(tmp_path: Path) -> None:
    """Second simultaneous slot is denied when max_lanes=1 and first holds lease."""

    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=1))
    host = _host(worker_limit=4)
    pool = MutationWorkerPool(
        MutationWorkerBudget(max_concurrency=2, default_timeout_seconds=5.0),
        resource_scheduler=scheduler,
        host_snapshot=host,
        checkpoint_dir=tmp_path / "ck",
    )
    hold = threading.Event()
    entered = threading.Event()

    def blocker(context: MutationWorkerContext) -> dict[str, Any]:
        entered.set()
        # Hold the lease until released.
        while not hold.is_set():
            context.check_cancelled()
            time.sleep(0.02)
        return {"ok": True}

    try:
        future1 = pool.submit(
            MutationWorkerTask(task_id="block-1", runner=blocker)
        )
        assert entered.wait(timeout=2.0)
        # While first lease is held, second admission should be denied.
        result2 = pool.run(
            MutationWorkerTask(task_id="block-2", runner=_ok_runner)
        )
        assert result2.disposition is MutationWorkerDisposition.RESOURCE_DENIED
        hold.set()
        result1 = future1.result(timeout=5.0)
        assert result1.disposition is MutationWorkerDisposition.COMPLETED
    finally:
        hold.set()
        pool.shutdown(wait=True, cancel=True)


# ---------------------------------------------------------------------------
# Timeout / cancellation / process-tree fencing
# ---------------------------------------------------------------------------


def test_callable_timeout_is_infrastructure(tmp_path: Path) -> None:
    pool = _pool(tmp_path, default_timeout_seconds=0.2)
    try:
        result = pool.run(
            MutationWorkerTask(
                task_id="timeout-1",
                runner=lambda ctx: _slow_runner(ctx, seconds=5.0),
                timeout_seconds=0.2,
            )
        )
        assert result.disposition is MutationWorkerDisposition.TIMEOUT
        assert result.is_infrastructure is True
        assert result.publication_allowed is False
        assert result.payload is None
        assert result.infrastructure.timed_out is True
        assert "timeout" in result.infrastructure.reason_codes
    finally:
        pool.shutdown(wait=True, cancel=True)


def test_pre_admit_cancellation(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cancel = MutationWorkerCancellation()
    cancel.cancel(reason="stop_now")
    try:
        result = pool.run(
            MutationWorkerTask(task_id="pre-cancel", runner=_ok_runner),
            cancellation=cancel,
        )
        assert result.disposition is MutationWorkerDisposition.CANCELLED
        assert result.is_infrastructure is True
        assert result.publication_allowed is False
        assert "cancelled_before_admit" in result.infrastructure.reason_codes
    finally:
        pool.shutdown(wait=True)


def test_cancellation_identity_mismatch_is_ignored(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    cancel = MutationWorkerCancellation(cancellation_id="owner-token")
    assert cancel.cancel(cancellation_id="wrong-token", reason="nope") is False
    assert cancel.is_cancelled() is False
    try:
        result = pool.run(
            MutationWorkerTask(task_id="id-cancel", runner=_ok_runner),
            cancellation=cancel,
        )
        assert result.disposition is MutationWorkerDisposition.COMPLETED
    finally:
        pool.shutdown(wait=True)


def test_command_success_and_nonzero_exit(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        ok = pool.run(
            MutationWorkerTask(
                task_id="cmd-ok",
                command=[sys.executable, "-c", "print('hello-worker')"],
                cwd=str(tmp_path),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        )
        assert ok.disposition is MutationWorkerDisposition.COMPLETED
        assert ok.payload is not None
        assert ok.payload["exit_code"] == 0
        assert ok.infrastructure.process_started is True
        assert ok.infrastructure.pid is not None

        bad = pool.run(
            MutationWorkerTask(
                task_id="cmd-fail",
                command=[sys.executable, "-c", "import sys; sys.exit(7)"],
                cwd=str(tmp_path),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        )
        assert bad.disposition is MutationWorkerDisposition.FAILED
        assert bad.is_infrastructure is False
        assert bad.payload is not None
        assert bad.payload["exit_code"] == 7
        assert "nonzero_exit" in bad.infrastructure.reason_codes
    finally:
        pool.shutdown(wait=True)


def test_command_timeout_fences_process_tree(tmp_path: Path) -> None:
    pool = _pool(tmp_path, default_timeout_seconds=0.3)
    try:
        result = pool.run(
            MutationWorkerTask(
                task_id="cmd-timeout",
                command=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ],
                cwd=str(tmp_path),
                timeout_seconds=0.3,
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        )
        assert result.disposition is MutationWorkerDisposition.TIMEOUT
        assert result.is_infrastructure is True
        assert result.publication_allowed is False
        assert result.infrastructure.process_started is True
        assert result.infrastructure.process_tree_fenced is True
        pid = result.infrastructure.pid
        if pid is not None:
            # Process tree must be gone after fence.
            deadline = time.time() + 2.0
            while time.time() < deadline and pid_alive(pid):
                time.sleep(0.05)
            assert not pid_alive(pid)
    finally:
        pool.shutdown(wait=True, cancel=True)


def test_command_cancellation_fences_and_blocks_late_success(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path, default_timeout_seconds=10.0)
    cancel = MutationWorkerCancellation()

    # Use a process that signals readiness via a file, then sleeps.
    ready = tmp_path / "ready.flag"
    script = tmp_path / "slow.py"
    script.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(ready)!r}).write_text('1')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    try:
        future = pool.submit(
            MutationWorkerTask(
                task_id="cmd-cancel",
                command=[sys.executable, str(script)],
                cwd=str(tmp_path),
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            ),
            cancellation=cancel,
        )
        deadline = time.time() + 5.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists(), "child never became ready"
        cancel.cancel(reason="test_cancel")
        result = future.result(timeout=10.0)
        assert result.disposition is MutationWorkerDisposition.CANCELLED
        assert result.publication_allowed is False
        assert result.payload is None
        assert result.infrastructure.process_tree_fenced is True
        assert result.infrastructure.cancelled is True
        pid = result.infrastructure.pid
        if pid is not None:
            deadline = time.time() + 2.0
            while time.time() < deadline and pid_alive(pid):
                time.sleep(0.05)
            assert not pid_alive(pid)
    finally:
        pool.shutdown(wait=True, cancel=True)


def test_missing_executable_is_infrastructure_failure(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        result = pool.run(
            MutationWorkerTask(
                task_id="missing-bin",
                command=[str(tmp_path / "no-such-binary-aae042")],
                cwd=str(tmp_path),
            )
        )
        assert (
            result.disposition
            is MutationWorkerDisposition.INFRASTRUCTURE_FAILURE
        )
        assert result.is_infrastructure is True
        assert result.payload is None
        assert "executable_missing" in result.infrastructure.reason_codes
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Restartability / leak freedom / pool lifecycle
# ---------------------------------------------------------------------------


def test_checkpoint_restart_recovers_incomplete_as_infrastructure(
    tmp_path: Path,
) -> None:
    store = MutationWorkerCheckpointStore(tmp_path / "ck")
    store.mark_running(
        "orphan-task",
        lease_id="lease-x",
        pool_id="pool-old",
        attempt=1,
    )
    incomplete = store.list_incomplete()
    assert len(incomplete) == 1
    assert incomplete[0]["task_id"] == "orphan-task"

    # Point recovery at the existing store by reconstructing with same dir.
    pool = MutationWorkerPool(
        MutationWorkerBudget(max_concurrency=1),
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=1)),
        host_snapshot=_host(),
        checkpoint_dir=tmp_path / "ck",
        pool_id="pool-new",
    )
    try:
        recovered = pool.recover()
        assert len(recovered) == 1
        result = recovered[0]
        assert result.task_id == "orphan-task"
        assert (
            result.disposition
            is MutationWorkerDisposition.INFRASTRUCTURE_FAILURE
        )
        assert result.infrastructure.restart_recovered is True
        assert "restart_recovered_incomplete" in result.infrastructure.reason_codes
        assert store.list_incomplete() == ()
        # Complete journal is durable.
        sealed = store.read("orphan-task")
        assert sealed is not None
        assert sealed["phase"] == "complete"
    finally:
        pool.shutdown(wait=True)


def test_shutdown_releases_leases_and_is_idempotent(tmp_path: Path) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=2))
    pool = _pool(tmp_path, scheduler=scheduler, max_concurrency=2)
    hold = threading.Event()
    entered = threading.Event()

    def blocker(context: MutationWorkerContext) -> dict[str, Any]:
        entered.set()
        while not hold.is_set():
            time.sleep(0.02)
            if context.cancelled:
                return {"ok": False, "reason": "cancelled"}
        return {"ok": True}

    future = pool.submit(
        MutationWorkerTask(task_id="shut-1", runner=blocker)
    )
    assert entered.wait(timeout=2.0)
    pool.shutdown(wait=False, cancel=True, reason="test_shutdown")
    hold.set()
    # Future should settle without leaking the lease.
    try:
        future.result(timeout=5.0)
    except Exception:
        pass
    deadline = time.time() + 2.0
    while time.time() < deadline and scheduler.active_leases:
        time.sleep(0.05)
    assert scheduler.active_leases == ()
    pool.shutdown(wait=True)  # idempotent
    assert pool.closed is True


def test_context_manager_cleans_up(tmp_path: Path) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=2))
    with MutationWorkerPool(
        MutationWorkerBudget(max_concurrency=1),
        resource_scheduler=scheduler,
        host_snapshot=_host(),
        checkpoint_dir=tmp_path / "ck",
    ) as pool:
        result = pool.run(
            MutationWorkerTask(task_id="cm-1", runner=_ok_runner)
        )
        assert result.disposition is MutationWorkerDisposition.COMPLETED
    assert pool.closed is True
    assert scheduler.active_leases == ()


def test_pool_to_dict_and_infrastructure_log(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        pool.run(MutationWorkerTask(task_id="log-1", runner=_ok_runner))
        body = pool.to_dict()
        assert body["interface"] == MUTATION_WORKER_POOL_INTERFACE
        assert body["evidence"] == AAE_MUTATION_WORKERS_EVIDENCE
        assert body["completed_count"] == 1
        assert body["budget"]["network_policy"] == NETWORK_POLICY_DENY_ALL
        records = pool.infrastructure_records
        assert len(records) >= 1
        # Infrastructure log is a separate surface from semantic payloads.
        assert all("disposition" in item or "event" in item for item in records)
    finally:
        pool.shutdown(wait=True)


def test_duplicate_inflight_task_id_rejected(tmp_path: Path) -> None:
    pool = _pool(tmp_path, max_concurrency=2)
    hold = threading.Event()
    entered = threading.Event()

    def blocker(context: MutationWorkerContext) -> dict[str, Any]:
        entered.set()
        while not hold.is_set():
            time.sleep(0.02)
        return {"ok": True}

    try:
        pool.submit(MutationWorkerTask(task_id="dup", runner=blocker))
        assert entered.wait(timeout=2.0)
        with pytest.raises(MutationWorkerError) as excinfo:
            pool.submit(MutationWorkerTask(task_id="dup", runner=_ok_runner))
        assert excinfo.value.reason_code == "duplicate_task"
    finally:
        hold.set()
        pool.shutdown(wait=True, cancel=True)


def test_map_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        with pytest.raises(MutationWorkerError) as excinfo:
            pool.map(
                [
                    MutationWorkerTask(task_id="same", runner=_ok_runner),
                    MutationWorkerTask(task_id="same", runner=_ok_runner),
                ]
            )
        assert excinfo.value.reason_code == "duplicate_task"
    finally:
        pool.shutdown(wait=True)


def test_closed_pool_rejects_submit(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.shutdown(wait=True)
    with pytest.raises(MutationWorkerError) as excinfo:
        pool.submit(MutationWorkerTask(task_id="after-close", runner=_ok_runner))
    assert excinfo.value.reason_code == "pool_closed"


def test_infrastructure_payload_never_attached_to_infra_disposition(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path, default_timeout_seconds=0.15)
    try:
        result = pool.run(
            MutationWorkerTask(
                task_id="no-payload",
                runner=lambda ctx: _slow_runner(ctx, seconds=5.0),
                timeout_seconds=0.15,
            )
        )
        assert result.disposition.is_infrastructure
        assert result.payload is None
        body = result.to_dict()
        assert body["payload"] is None
        assert body["infrastructure"]["is_infrastructure"] is True
    finally:
        pool.shutdown(wait=True, cancel=True)


def test_pool_cancel_propagates_to_inflight_callables(tmp_path: Path) -> None:
    pool = _pool(tmp_path, max_concurrency=2, default_timeout_seconds=10.0)
    entered = threading.Event()

    def blocker(context: MutationWorkerContext) -> dict[str, Any]:
        entered.set()
        while True:
            context.check_cancelled()
            time.sleep(0.05)

    try:
        future = pool.submit(
            MutationWorkerTask(task_id="pool-cancel-1", runner=blocker)
        )
        assert entered.wait(timeout=2.0)
        pool.cancel(reason="operator_abort")
        result = future.result(timeout=5.0)
        assert result.disposition is MutationWorkerDisposition.CANCELLED
        assert result.is_infrastructure is True
    finally:
        pool.shutdown(wait=True, cancel=True)


def test_result_get_after_completion(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    try:
        result = pool.run(
            MutationWorkerTask(task_id="get-1", runner=_ok_runner)
        )
        cached = pool.get_result("get-1")
        assert cached is not None
        assert cached.task_id == result.task_id
        assert cached.disposition is result.disposition
        assert pool.get_result("missing") is None
    finally:
        pool.shutdown(wait=True)
