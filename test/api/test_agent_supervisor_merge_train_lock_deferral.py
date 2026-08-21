"""Regression coverage for durable verified merge-lock deferrals."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    CheckoutMutationLease,
    acquire_checkout_mutation_lease,
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    release_checkout_mutation_lease,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.todo_daemon import core as todo_core


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
