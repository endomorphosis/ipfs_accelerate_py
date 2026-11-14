"""Reconciliation must bind an opportunistic train receipt to its request."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def _scenario(tmp_path: Path) -> tuple[
    TodoImplementationDaemon, dict[str, Any], str, str, str,
]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Train Receipt Test")
    _git(repo, "config", "user.email", "train-receipt@example.invalid")
    (repo / "todo.md").write_text(
        "# Tasks\n\n"
        "## BIND-001 Produce this lane's evidence\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P0\n"
        "- Track: verification\n"
        "- Outputs: evidence.txt\n"
        "- Validation: python -m pytest\n"
        "- Acceptance: The exact candidate and output are integrated.\n",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-qm", "base task")
    baseline = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-qb", "implementation/other")
    (repo / "other.txt").write_text("other lane\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-qm", "other candidate")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "--no-ff", "-m", "integrate other lane", "implementation/other")
    other_integration = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-qb", "implementation/own", baseline)
    (repo / "evidence.txt").write_text("own lane\n", encoding="utf-8")
    _git(repo, "add", "evidence.txt")
    _git(repo, "commit", "-qm", "own candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "--no-ff", "-m", "integrate own lane", "implementation/own")
    own_integration = _git(repo, "rev-parse", "HEAD")
    (repo / "later.txt").write_text("later target\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-qm", "advance target")
    target = _git(repo, "rev-parse", "HEAD")

    state = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state / "task_state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        repo_root=repo,
        task_header_prefix="BIND-",
        worktree_pool_enabled=False,
    )
    task = daemon._load_tasks()[0]
    identity = daemon._identity_for_task(task)
    event: dict[str, Any] = {
        "task_id": task.task_id,
        "task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "attempt": 1,
        "branch": "implementation/own",
        "baseline_ref": baseline,
        "implementation_commit": candidate,
        "merge_result": {
            "queued": True,
            "request_id": "own-request",
            "merged": False,
            "train_result": {
                "request_id": "other-request",
                "merge_result": {
                    "merge_commit": other_integration,
                    "target_commit": other_integration,
                },
            },
        },
    }
    daemon._failed_merge_candidates = (  # type: ignore[method-assign]
        lambda skip_task_ids=None: [event]
    )
    # Cleanup is unrelated to receipt selection; retain the real Git graph,
    # declared-output checks, board write and durable completion verification.
    daemon._cleanup_merged_worktree = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: {"cleaned": True}
    )
    return daemon, event, other_integration, own_integration, target


@pytest.mark.parametrize("identity_shape", ["different", "nested", "missing_train", "missing_own"])
def test_unrelated_train_receipt_cannot_override_landed_candidate(
    tmp_path: Path, identity_shape: str,
) -> None:
    daemon, event, other_integration, _, target = _scenario(tmp_path)
    handoff = event["merge_result"]
    train = handoff["train_result"]
    if identity_shape == "nested":
        train["request"] = {"request_id": train.pop("request_id")}
    elif identity_shape == "missing_train":
        train.pop("request_id")
    elif identity_shape == "missing_own":
        handoff.pop("request_id")

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is True, result
    assert result["merge_commit"] == target
    assert result["merge_commit"] != other_integration
    assert result["integration_commit_proof"]["passed"] is True
    assert result["post_merge_declared_output_invariant"]["passed"] is True
    assert result["completion_persistence"]["passed"] is True
    assert daemon._load_tasks()[0].status == "completed"


def test_same_request_receipt_keeps_its_immutable_integration_tree(
    tmp_path: Path,
) -> None:
    daemon, event, _, own_integration, target = _scenario(tmp_path)
    event["merge_result"]["train_result"] = {
        "request": {"request_id": "own-request"},
        "merge_result": {"merge_commit": own_integration},
    }

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is True, result
    assert result["merge_commit"] == own_integration
    assert result["merge_commit"] != target


def test_same_request_wrong_integration_stays_unresolved(tmp_path: Path) -> None:
    daemon, event, other_integration, _, _ = _scenario(tmp_path)
    event["merge_result"]["train_result"]["request_id"] = "own-request"

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is False
    assert result["merge_commit"] == other_integration
    assert result["integration_commit_proof"]["passed"] is False
    assert "implementation_not_ancestor_of_integration_commit" in (
        result["integration_commit_proof"]["reasons"]
    )
    assert daemon._load_tasks()[0].status == "todo"


def test_landed_target_fallback_still_requires_declared_outputs(tmp_path: Path) -> None:
    daemon, _, _, _, _ = _scenario(tmp_path)
    _git(daemon.repo_root, "rm", "evidence.txt")
    _git(daemon.repo_root, "commit", "-qm", "remove output after merge")

    result = daemon._reconcile_failed_merges()[0]

    assert result["resolved"] is False
    assert result["integration_commit_proof"]["passed"] is True
    assert result["post_merge_declared_output_invariant"]["passed"] is False
    assert daemon._load_tasks()[0].status == "todo"
