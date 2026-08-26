"""Regression coverage for durable verified merge-lock deferrals."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    CheckoutMutationLease,
    acquire_checkout_mutation_lease,
    board_scoped_checkout_mutation_lock_path,
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    read_checkout_mutation_lease,
    release_checkout_mutation_lease,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon import core as todo_core
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Merge Deferral Test")
    _git(
        repo,
        "config",
        "user.email",
        "merge-deferral@example.invalid",
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    return repo


def _lease(
    repo: Path,
    *,
    pid: int | None = None,
    protected_recovery_required: bool = False,
) -> CheckoutMutationLease:
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="EXTERNAL-OWNER",
        branch="external/owner",
        owner_script="pytest",
        extra={
            "operation": "merge_branch_to_main",
            "protected_recovery_required": protected_recovery_required,
        },
    )
    if pid is not None:
        metadata["pid"] = pid
    lease, reason, _existing, _waited = acquire_checkout_mutation_lease(
        checkout_mutation_lock_path(repo),
        metadata,
        owner_active=lambda _metadata: True,
    )
    assert reason == "acquired"
    assert lease is not None
    return lease


def _transaction_daemon(
    repo: Path,
    *,
    state_name: str,
) -> TodoImplementationDaemon:
    state_dir = repo / state_name
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        board_namespace="checkout-transaction-wait-test",
        merge_target_branch="main",
    )


def _transaction_owner_lease(
    daemon: TodoImplementationDaemon,
) -> CheckoutMutationLease:
    lease, reason, _existing, _waited = (
        daemon._acquire_checkout_mutation_lease(
            task_id="TRANSIENT-OWNER",
            branch="",
            operation="generated_dirty_repair",
        )
    )
    assert reason == "acquired"
    assert lease is not None
    return lease


def _foreign_supervisor_recovery_lease(
    daemon: TodoImplementationDaemon,
) -> CheckoutMutationLease:
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=daemon.repo_root,
        task_id="TRANSIENT-SUPERVISOR-OWNER",
        owner_script=Path(sys.argv[0]).name,
        extra={
            "operation": "generated_dirty_repair",
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
        },
    )
    lease, reason, _existing, _waited = acquire_checkout_mutation_lease(
        daemon._repo_merge_lock_path(),
        metadata,
        owner_active=lambda _metadata: True,
    )
    assert reason == "acquired"
    assert lease is not None
    return lease


def _keep_checkout_owner_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_is_running",
        lambda pid: pid == os.getpid(),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_command_line",
        lambda pid: (
            f"python {Path(sys.argv[0]).name}"
            if pid == os.getpid()
            else ""
        ),
    )


def _contention_result(
    lease: CheckoutMutationLease,
    *,
    lease_id: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": False,
        "merged": False,
        "reason": "checkout_mutation_lock_exists",
        "lock_path": str(lease.lock_path),
        "lock_owner_pid": int(lease.metadata["pid"]),
        "lock_owner_task_id": str(lease.metadata.get("task_id") or ""),
        "lock_owner_branch": str(lease.metadata.get("branch") or ""),
    }
    if lease_id is not None:
        result["lock_owner_lease_id"] = lease_id
    return result


def test_verified_live_lock_wait_remains_pending_beyond_recorded_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    now = [100.0]
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: now[0],
        max_attempts=1,
    )
    request = queue.enqueue(
        branch_name="implementation/waits-for-external-owner",
        task_id="WAIT-FOR-EXTERNAL-OWNER",
        canonical_task_id="canonical-wait-for-external-owner",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
    )
    lease = _lease(repo)
    # Canonical checkout leases bind sibling worktrees by repository identity;
    # the legacy exact-path ``repo_root`` field is intentionally blank.
    assert lease.metadata["repo_root"] == ""
    assert lease.metadata["worktree_root"] == str(repo.resolve())
    waiting = [True]
    monkeypatch.setattr(
        todo_core,
        "pid_alive",
        lambda pid: pid == os.getpid(),
    )
    monkeypatch.setattr(
        todo_core,
        "process_args",
        lambda pid: "python -m pytest" if pid == os.getpid() else "",
    )

    def merge_callback(_request: object) -> dict[str, Any]:
        if waiting[0]:
            return _contention_result(lease, lease_id=lease.lease_id)
        return {"attempted": True, "merged": True}

    train = MergeTrain(
        repo,
        queue,
        max_attempts=1,
        merge_lock_deferral_seconds=1,
        max_merge_lock_deferrals=2,
        merge_callback=merge_callback,
    )
    released = False
    try:
        results = []
        for _index in range(40):
            result = train.run_once()
            assert result is not None
            results.append(result)
            assert result["status"] == "deferred"
            assert result["failure_count"] == 0
            now[0] = float(result["retry_not_before"])

        stored = queue.get(request.request_id)
        assert stored is not None
        assert stored.status == "pending"
        assert stored.attempt == 1
        assert stored.failure_count == 0
        assert len(stored.metadata["deferrals"]) == 32
        assert results[-1]["prolonged_live_owner_wait"] is True
        assert not (queue.quarantine_dir / f"{request.request_id}.json").exists()

        assert release_checkout_mutation_lease(lease) is True
        released = True
        waiting[0] = False
        merged = train.run_once()

        assert merged is not None
        assert merged["status"] == "merged"
        accepted = queue.get(request.request_id)
        assert accepted is not None
        assert accepted.status == "completed"
        assert accepted.attempt == 1
        assert accepted.failure_count == 0
    finally:
        if not released:
            release_checkout_mutation_lease(lease)


def test_daemon_board_scoped_live_lock_wait_does_not_consume_queue_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    sibling = tmp_path / "board-worktree"
    _git(repo, "worktree", "add", "-b", "board-worktree", str(sibling))
    todo_path = sibling / "todo.md"
    todo_path.write_text("# Board-scoped lock test\n", encoding="utf-8")
    owner_daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=sibling / "owner-state" / "task-state.json",
        strategy_path=sibling / "owner-state" / "strategy.json",
        events_path=sibling / "owner-state" / "events.jsonl",
        repo_root=sibling,
        board_namespace="merge-lock-deferral-test",
        merge_target_branch="board-worktree",
    )
    consumer_daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=sibling / "consumer-state" / "task-state.json",
        strategy_path=sibling / "consumer-state" / "strategy.json",
        events_path=sibling / "consumer-state" / "events.jsonl",
        repo_root=sibling,
        board_namespace="merge-lock-deferral-test",
        merge_target_branch="board-worktree",
    )
    lock_path = owner_daemon._repo_merge_lock_path()
    assert consumer_daemon._repo_merge_lock_path() == lock_path
    assert lock_path == board_scoped_checkout_mutation_lock_path(
        sibling,
        "merge-lock-deferral-test",
    )
    assert lock_path.parent.resolve() == checkout_mutation_lock_path(
        repo
    ).parent.resolve()

    now = [100.0]
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: now[0],
        max_attempts=1,
    )
    request = queue.enqueue(
        branch_name="implementation/waits-for-board-owner",
        task_id="WAIT-FOR-BOARD-OWNER",
        canonical_task_id="canonical-wait-for-board-owner",
        commit_sha=_git(sibling, "rev-parse", "HEAD"),
    )
    lease, reason, _existing, _waited = (
        owner_daemon._acquire_checkout_mutation_lease(
            task_id="EXTERNAL-BOARD-OWNER",
            branch="implementation/external-board-owner",
            operation="merge_branch_to_main",
        )
    )
    assert reason == "acquired"
    assert lease is not None
    monkeypatch.setattr(
        todo_core,
        "pid_alive",
        lambda pid: pid == os.getpid(),
    )
    monkeypatch.setattr(
        todo_core,
        "process_args",
        lambda pid: (
            f"python {Path(sys.argv[0]).name}"
            if pid == os.getpid()
            else ""
        ),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_is_running",
        lambda pid: pid == os.getpid(),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_command_line",
        lambda pid: (
            f"python {Path(sys.argv[0]).name}"
            if pid == os.getpid()
            else ""
        ),
    )

    callback_results: list[dict[str, Any]] = []
    incumbent = read_checkout_mutation_lease(lock_path)
    assert incumbent is not None
    assert incumbent.lease_id == lease.lease_id
    assert consumer_daemon._merge_lock_owner_is_active(
        dict(incumbent.metadata)
    )

    def merge_callback(_request: object) -> dict[str, Any]:
        callback_result = consumer_daemon._run_checkout_mutation_transaction(
            task_id="WAIT-FOR-BOARD-OWNER",
            branch="implementation/waits-for-board-owner",
            operation="merge_branch_to_main",
            callback=lambda: {"attempted": True, "merged": True},
            failure_fields={"attempted": False, "merged": False},
        )
        callback_results.append(callback_result)
        return callback_result

    try:
        result = MergeTrain(
            sibling,
            queue,
            target_branch="board-worktree",
            max_attempts=1,
            merge_lock_path=lock_path,
            merge_lock_deferral_seconds=1,
            merge_callback=merge_callback,
        ).run_once()

        assert result is not None
        assert result["status"] == "deferred", {
            "train_result": result,
            "callback_results": callback_results,
        }
        assert result["failure_count"] == 0
        assert result["attempt"] == 1
        assert result["merge_result"]["lock_path"] == str(lock_path)
        assert result["merge_result"]["lock_owner_lease_id"] == lease.lease_id
        assert result["merge_lock_contention"]["lock_owner_lease_id"] == (
            lease.lease_id
        )
        stored = queue.get(request.request_id)
        assert stored is not None
        assert stored.status == "pending"
        assert stored.attempt == 1
        assert stored.failure_count == 0
    finally:
        assert owner_daemon._release_checkout_mutation_lease(lease) is True


def test_checkout_transaction_waits_through_transient_foreign_protected_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    _keep_checkout_owner_live(monkeypatch)
    consumer = _transaction_daemon(repo, state_name="consumer-state")
    lease = _foreign_supervisor_recovery_lease(consumer)
    acquisition_started = threading.Event()
    observed_timeouts: list[float] = []
    original_acquire = consumer._acquire_checkout_mutation_lease

    def tracking_acquire(**kwargs: Any):
        observed_timeouts.append(float(kwargs["timeout_seconds"]))
        acquisition_started.set()
        return original_acquire(**kwargs)

    monkeypatch.setattr(
        consumer,
        "_acquire_checkout_mutation_lease",
        tracking_acquire,
    )

    def release_transient_owner() -> bool:
        assert acquisition_started.wait(timeout=1.0)
        time.sleep(0.02)
        return release_checkout_mutation_lease(lease)

    with ThreadPoolExecutor(max_workers=1) as executor:
        released = executor.submit(release_transient_owner)
        result = consumer._run_checkout_mutation_transaction(
            task_id="WAITING-CONSUMER",
            operation="requalify_post_merge_callback_integration",
            callback=lambda: {"passed": True},
            failure_fields={"passed": False},
            timeout_seconds=0.5,
        )
        assert released.result(timeout=1.0) is True

    assert result == {"passed": True}
    assert len(observed_timeouts) == 1
    assert 0.0 < observed_timeouts[0] <= 0.5
    assert consumer._current_checkout_mutation_lease() is None
    assert not consumer._repo_merge_lock_path().exists()


def test_checkout_transaction_times_out_on_persistent_foreign_protected_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    _keep_checkout_owner_live(monkeypatch)
    consumer = _transaction_daemon(repo, state_name="consumer-state")
    lease = _foreign_supervisor_recovery_lease(consumer)
    original_lease = lease.lock_path.read_bytes()
    callback_calls: list[str] = []

    started = time.monotonic()
    try:
        result = consumer._run_checkout_mutation_transaction(
            task_id="WAITING-CONSUMER",
            operation="requalify_post_merge_callback_integration",
            callback=lambda: (
                callback_calls.append("called") or {"passed": True}
            ),
            failure_fields={"passed": False},
            timeout_seconds=0.06,
        )
    finally:
        assert lease.lock_path.read_bytes() == original_lease
        assert release_checkout_mutation_lease(lease) is True
    elapsed = time.monotonic() - started

    assert result["passed"] is False
    assert result["reason"] == "checkout_mutation_lock_exists"
    assert result["lock_owner_lease_id"] == lease.lease_id
    assert callback_calls == []
    assert 0.04 <= elapsed < 0.5
    assert consumer._current_checkout_mutation_lease() is None


def test_other_checkout_operation_does_not_wait_through_protected_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    _keep_checkout_owner_live(monkeypatch)
    consumer = _transaction_daemon(repo, state_name="consumer-state")
    lease = _foreign_supervisor_recovery_lease(consumer)
    callback_calls: list[str] = []
    monkeypatch.setattr(
        consumer,
        "_acquire_checkout_mutation_lease",
        lambda **_kwargs: pytest.fail(
            "ordinary operation bypassed protected recovery deferral"
        ),
    )

    started = time.monotonic()
    try:
        result = consumer._run_checkout_mutation_transaction(
            task_id="NON-CALLBACK-CONSUMER",
            operation="merge_branch_to_main",
            callback=lambda: (
                callback_calls.append("called") or {"merged": True}
            ),
            failure_fields={"merged": False},
            timeout_seconds=0.5,
        )
    finally:
        assert release_checkout_mutation_lease(lease) is True
    elapsed = time.monotonic() - started

    assert result["merged"] is False
    assert result["reason"] == (
        "external_protected_checkout_recovery_required"
    )
    assert result["foreign_owner_liveness"] == "verified_live"
    assert callback_calls == []
    assert elapsed < 0.1


@pytest.mark.parametrize(
    "timeout_seconds",
    (True, "0", None, -0.01, float("nan"), float("inf"), 5.01),
)
def test_checkout_transaction_rejects_invalid_wait_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: object,
) -> None:
    consumer = _transaction_daemon(
        _repo(tmp_path),
        state_name="consumer-state",
    )
    monkeypatch.setattr(
        consumer,
        "_recover_protected_checkout_mutation",
        lambda: pytest.fail("invalid timeout reached lease recovery"),
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        consumer._run_checkout_mutation_transaction(
            task_id="INVALID-WAIT",
            operation="requalify_post_merge_callback_integration",
            callback=lambda: pytest.fail("invalid timeout reached callback"),
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        )


def test_checkout_transaction_default_remains_zero_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    _keep_checkout_owner_live(monkeypatch)
    owner = _transaction_daemon(repo, state_name="owner-state")
    consumer = _transaction_daemon(repo, state_name="consumer-state")
    lease = _transaction_owner_lease(owner)
    observed_acquisitions: list[tuple[float, float]] = []
    original_acquire = consumer._acquire_checkout_mutation_lease

    def tracking_acquire(**kwargs: Any):
        result = original_acquire(**kwargs)
        observed_acquisitions.append(
            (float(kwargs["timeout_seconds"]), float(result[3]))
        )
        return result

    monkeypatch.setattr(
        consumer,
        "_acquire_checkout_mutation_lease",
        tracking_acquire,
    )
    try:
        result = consumer._run_checkout_mutation_transaction(
            task_id="NONBLOCKING-CONSUMER",
            operation="merge_branch_to_main",
            callback=lambda: pytest.fail(
                "default checkout transaction waited through its owner"
            ),
            failure_fields={"merged": False},
        )
    finally:
        assert owner._release_checkout_mutation_lease(lease) is True

    assert result["merged"] is False
    assert result["reason"] == "checkout_mutation_lock_exists"
    assert len(observed_acquisitions) == 1
    timeout_seconds, waited_seconds = observed_acquisitions[0]
    assert timeout_seconds == 0.0
    assert waited_seconds < 0.1


def test_merge_train_rejects_merge_lock_outside_git_common_dir(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue")

    with pytest.raises(ValueError, match="repository Git common directory"):
        MergeTrain(
            repo,
            queue,
            merge_lock_path=tmp_path / "implementation-main-merge.lock",
        )


def test_dead_protected_owner_does_not_authorize_non_consuming_wait(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue", max_attempts=1)
    request = queue.enqueue(
        branch_name="implementation/dead-protected-owner",
        task_id="DEAD-PROTECTED-OWNER",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
    )
    lease = _lease(
        repo,
        pid=2_147_483_647,
        protected_recovery_required=True,
    )
    try:
        result = MergeTrain(
            repo,
            queue,
            max_attempts=1,
            merge_callback=lambda _request: _contention_result(
                lease,
                lease_id=lease.lease_id,
            ),
        ).run_once()
    finally:
        release_checkout_mutation_lease(lease)

    assert result is not None
    assert result["status"] == "quarantined"
    assert result.get("deferred", False) is False
    stored = queue.get(request.request_id)
    assert stored is not None
    assert stored.status == "quarantined"
    assert stored.failure_count == 1


@pytest.mark.parametrize("callback_lease_id", [None, "forged-lease-id"])
def test_unbound_live_lock_result_fails_closed(
    tmp_path: Path,
    callback_lease_id: str | None,
) -> None:
    repo = _repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue", max_attempts=1)
    request = queue.enqueue(
        branch_name="implementation/unbound-live-owner",
        task_id="UNBOUND-LIVE-OWNER",
        commit_sha=_git(repo, "rev-parse", "HEAD"),
    )
    lease = _lease(repo, pid=os.getpid())
    try:
        result = MergeTrain(
            repo,
            queue,
            max_attempts=1,
            merge_callback=lambda _request: _contention_result(
                lease,
                lease_id=callback_lease_id,
            ),
        ).run_once()
    finally:
        assert release_checkout_mutation_lease(lease) is True

    assert result is not None
    assert result["status"] == "quarantined"
    assert result.get("deferred", False) is False
    stored = queue.get(request.request_id)
    assert stored is not None
    assert stored.status == "quarantined"
    assert stored.failure_count == 1
