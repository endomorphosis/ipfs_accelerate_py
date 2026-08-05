"""Auto-finish residual merge stalls: non-overlapping dirt and concurrency peers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    BundleLaneSpec,
    _lanes_conflict,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "agent@example.invalid")
    _git(repo, "config", "user.name", "Agent")


def _daemon(repo: Path) -> TodoImplementationDaemon:
    state = repo / "state"
    state.mkdir(exist_ok=True)
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state / "task_state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        repo_root=repo,
    )


def test_nonoverlapping_tracked_dirt_does_not_block_reconciliation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    board = repo / "data" / "agent_supervisor" / "bundles"
    board.mkdir(parents=True)
    (board / "index.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "README.md", "data")
    _git(repo, "commit", "-m", "Initial")

    _git(repo, "checkout", "-b", "implementation/fvt-103")
    residual = (
        repo
        / "data"
        / "agent_supervisor"
        / "managed-residuals"
        / "managed-tlc"
        / "install-receipt.json"
    )
    residual.parent.mkdir(parents=True)
    residual.write_text('{"status":"ok"}\n', encoding="utf-8")
    _git(repo, "add", "data")
    _git(repo, "commit", "-m", "FVT-103 residual")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "main")
    (board / "index.json").write_text('{"bundles":{}}\n', encoding="utf-8")
    daemon = _daemon(repo)
    candidates = [
        {
            "task_id": "FVT-103",
            "branch": "implementation/fvt-103",
            "implementation_commit": implementation_commit,
        }
    ]
    blocking, nonblocking = daemon._reconciliation_blocking_dirty_paths(
        candidates,
        target_branch="main",
    )
    assert blocking == []
    assert "data/agent_supervisor/bundles/index.json" in nonblocking


def test_overlapping_tracked_dirt_still_blocks_reconciliation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    target = repo / "feature.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "Initial")

    _git(repo, "checkout", "-b", "implementation/fvt-x")
    target.write_text("candidate\n", encoding="utf-8")
    _git(repo, "commit", "-am", "candidate")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "main")
    target.write_text("dirty main\n", encoding="utf-8")
    daemon = _daemon(repo)
    candidates = [
        {
            "task_id": "FVT-X",
            "branch": "implementation/fvt-x",
            "implementation_commit": implementation_commit,
        }
    ]
    blocking, nonblocking = daemon._reconciliation_blocking_dirty_paths(
        candidates,
        target_branch="main",
    )
    assert blocking == ["feature.txt"]
    assert nonblocking == []


def _lane(
    *,
    bundle_key: str,
    task_ids: list[str],
    conflicting: list[str] | None = None,
    allow_concurrent: list[str] | None = None,
) -> BundleLaneSpec:
    payload = {
        "bundle_key": bundle_key,
        "allow_concurrent_with": list(allow_concurrent or ()),
        "tasks": [
            {
                "task_id": task_ids[0],
                "allow_concurrent_with": list(allow_concurrent or ()),
            }
        ],
    }
    return BundleLaneSpec(
        bundle_key=bundle_key,
        parallel_lane=bundle_key,
        todo_path=Path("todo.md"),
        state_dir=Path("state"),
        worktree_root=Path("worktrees"),
        state_prefix="test",
        task_ids=list(task_ids),
        conflict_policy="",
        command=["true"],
        log_path=Path("lane.log"),
        queue_payload=payload,
        conflicting_task_ids=list(conflicting or ()),
    )


def test_lanes_conflict_respects_allow_concurrent_with() -> None:
    left = _lane(
        bundle_key="formal-verification-tactician/managed-tlc-apalache",
        task_ids=["FVT-103"],
        conflicting=["formal-verification-tactician/toolchain-release-candidate"],
        allow_concurrent=["formal-verification-tactician/toolchain-release-candidate"],
    )
    right = _lane(
        bundle_key="formal-verification-tactician/toolchain-release-candidate",
        task_ids=["FVT-081"],
        conflicting=["formal-verification-tactician/managed-tlc-apalache"],
    )
    assert _lanes_conflict(left, right) is False
    # Without allow_concurrent, the same edge still blocks.
    blocked = _lane(
        bundle_key="formal-verification-tactician/managed-tlc-apalache",
        task_ids=["FVT-103"],
        conflicting=["formal-verification-tactician/toolchain-release-candidate"],
    )
    assert _lanes_conflict(blocked, right) is True
