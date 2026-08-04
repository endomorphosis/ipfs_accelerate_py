"""Supervisor auto-unblock: task claims must not outlive implementation work."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
    PortalImplementationDaemon,
)


def _daemon(repo: Path) -> PortalImplementationDaemon:
    obj = object.__new__(PortalImplementationDaemon)
    obj.repo_root = repo.resolve()
    return obj


def test_claim_inactive_when_implementation_finished(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "lane_task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_task_id": "",
                "active_task_cid": "",
                "active_attempt": 0,
                "implementation_in_progress": False,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
        "task_id": "UIR-033",
        "canonical_task_cid": "cid:033",
        "attempt": 1,
        "pid": 1,
        "state_path": str(state_path),
        "state_dir": str(state_dir),
    }
    daemon = _daemon(tmp_path)
    assert (
        daemon._implementation_task_claim_still_bound_to_active_work(metadata)
        is False
    )


def test_claim_active_while_implementation_in_progress(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "lane_task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_task_id": "UIR-033",
                "active_task_cid": "cid:033",
                "active_attempt": 2,
                "implementation_in_progress": True,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
        "task_id": "UIR-033",
        "canonical_task_cid": "cid:033",
        "attempt": 2,
        "state_path": str(state_path),
    }
    daemon = _daemon(tmp_path)
    assert (
        daemon._implementation_task_claim_still_bound_to_active_work(metadata)
        is True
    )


def test_claim_inactive_on_task_id_mismatch(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "lane_task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_task_id": "UIR-034",
                "active_task_cid": "cid:034",
                "active_attempt": 1,
                "implementation_in_progress": True,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
        "task_id": "UIR-033",
        "canonical_task_cid": "cid:033",
        "attempt": 1,
        "state_path": str(state_path),
    }
    daemon = _daemon(tmp_path)
    assert (
        daemon._implementation_task_claim_still_bound_to_active_work(metadata)
        is False
    )


def test_reclaim_stale_task_claims_unlinks_finished(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    # Place claim dir under fake git common dir via worktree-style .git file? 
    # checkout_mutation_lock_path uses git common dir; for bare .git dir:
    claim_dir = repo / ".git" / "implementation-task-claims"
    claim_dir.mkdir(parents=True)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "uiir_lane_0_task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_task_id": "",
                "implementation_in_progress": False,
                "active_attempt": 0,
            }
        ),
        encoding="utf-8",
    )
    claim_path = claim_dir / "canonical-task-test.lock"
    claim_path.write_text(
        json.dumps(
            {
                "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
                "task_id": "UIR-033",
                "canonical_task_cid": "cid:033",
                "attempt": 1,
                "pid": 999999,  # not running
                "state_path": str(state_path),
                "state_dir": str(state_dir),
                "lease_id": "lease-test",
                "repository_id": "",
                "worktree_root": str(repo),
                "repo_root": "",
            }
        ),
        encoding="utf-8",
    )

    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = repo.resolve()
    daemon._last_implementation_task_claim_reclaim_monotonic = 0.0
    daemon._worktree_lifecycle_reclaim_interval_seconds = 1.0

    result = daemon._reclaim_stale_implementation_task_claims(
        reason="test",
        force=True,
    )
    assert result["reclaimed_count"] == 1
    assert "UIR-033" in result["task_ids"]
    assert not claim_path.exists()
