from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    build_post_merge_validation_evidence,
    verify_post_merge_validation_evidence,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _runtime(
    tmp_path: Path,
    *,
    scheduler: Any,
) -> tuple[TodoImplementationDaemon, Path, PortalTask, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "checkout", "-qb", "main")
    _git(repo, "config", "user.name", "Post-merge Runtime Test")
    _git(repo, "config", "user.email", "post-merge@example.invalid")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")
    commit = _git(repo, "rev-parse", "HEAD")
    tree_id = f"git-tree:{_git(repo, 'rev-parse', 'HEAD^{tree}')}"

    state_dir = tmp_path / "state"
    todo_path = tmp_path / "tasks.todo.md"
    todo_path.write_text(
        "## PMV-001 Validate landed commit\n\n"
        "- Status: todo\n"
        "- Completion: manual\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        merge_target_branch="main",
        validation_scheduler=scheduler,
        validation_cache_dir=state_dir / "validation-cache",
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
        merge_queue_dir=state_dir / "merge-queue",
        worktree_submodule_paths=(),
    )
    task = PortalTask(
        task_id="PMV-001",
        title="Validate landed commit",
        status="todo",
        completion="manual",
        priority="P0",
        track="quality",
        validation=["git diff --check"],
        metadata={"Provider role": "deterministic-only"},
    )
    return daemon, repo, task, commit, tree_id


class _RecordingScheduler:
    def __init__(
        self,
        during_run: Callable[[Path], None] | None = None,
    ) -> None:
        self.during_run = during_run
        self.calls: list[dict[str, Any]] = []

    def run(self, commands: Any, **kwargs: Any) -> dict[str, Any]:
        command_specs = tuple(commands)
        workspace = Path(kwargs["workspace_path"])
        self.calls.append(
            {
                "commands": command_specs,
                "workspace": workspace,
                **kwargs,
            }
        )
        if self.during_run is not None:
            self.during_run(workspace)
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [
                {
                    "command": command_specs[0].command,
                    "returncode": 0,
                    "stage": "cheap",
                    "output": "passed\n",
                }
            ],
            "selection": {
                "scope": kwargs["scope"],
                "changed_files": [],
                "escalated": True,
            },
            "elapsed_seconds": 0.125,
        }


def test_validation_runner_binds_uncached_post_merge_scope_and_target(
    tmp_path: Path,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, repo, task, commit, _tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    result = daemon._run_validation_commands(
        repo,
        task,
        tmp_path / "post-merge.log",
        scope="post_merge",
        target_commit=commit,
    )

    assert result["passed"] is True
    assert result["validation_scope"] == "post_merge"
    assert result["target_commit"] == commit
    assert result["validated_commit"] == commit
    call = scheduler.calls[0]
    assert call["scope"] == "post_merge"
    assert call["target_commit"] == commit
    assert call["require_full_validation"] is True
    assert all(spec.cacheable is False for spec in call["commands"])


def test_exact_post_merge_runtime_builds_receipt_in_detached_clean_worktree(
    tmp_path: Path,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is True
    assert evidence["attempted"] is True
    assert evidence["target_commit"] == commit
    assert evidence["validated_commit"] == commit
    assert evidence["repository_tree_id"] == tree_id
    assert evidence["validation_scope"] == "post_merge"
    assert verify_post_merge_validation_evidence(
        evidence,
        expected_task_id=task.task_id,
        expected_target_commit=commit,
        expected_repository_tree_id=tree_id,
    ) == (True, ())
    workspace = scheduler.calls[0]["workspace"]
    assert workspace != repo
    assert not workspace.exists()
    assert str(workspace) not in _git(repo, "worktree", "list", "--porcelain")
    assert evidence["validation_result"]["elapsed_seconds"] == "0.125"


@pytest.mark.parametrize("fence", ("target", "workspace"))
def test_exact_post_merge_runtime_rejects_changed_fence(
    tmp_path: Path,
    fence: str,
) -> None:
    future_commit = ""

    def mutate(workspace: Path) -> None:
        if fence == "target":
            _git(
                workspace,
                "update-ref",
                "refs/heads/main",
                future_commit,
            )
        else:
            (workspace / "tracked.txt").write_text(
                "changed during validation\n",
                encoding="utf-8",
            )

    scheduler = _RecordingScheduler(mutate)
    daemon, repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    if fence == "target":
        _git(repo, "checkout", "-qb", "future")
        (repo / "future.txt").write_text("future\n", encoding="utf-8")
        _git(repo, "add", "future.txt")
        _git(repo, "commit", "-qm", "future")
        future_commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-q", "main")

    evidence = daemon._validate_exact_post_merge_commit(
        task,
        target_commit=commit,
        repository_tree_id=tree_id,
    )

    assert evidence["passed"] is False
    assert evidence["stale"] is True
    nested = evidence["validation_result"]
    expected_reason = (
        "post_merge_validation_target_changed_after_execution"
        if fence == "target"
        else "post_merge_validation_workspace_dirty_after_execution"
    )
    assert nested["reason"] == expected_reason
    verified, reasons = verify_post_merge_validation_evidence(evidence)
    assert verified is True
    assert reasons == ()


def test_merge_callback_uses_fresh_post_merge_receipt_for_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    evidence = build_post_merge_validation_evidence(
        task_id=task.task_id,
        target_commit=commit,
        repository_tree_id=tree_id,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        },
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        daemon,
        "_reject_protected_merge_candidate",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: {"ready": True, "rehydrated": False},
    )
    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": commit,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_validate_exact_post_merge_commit",
        lambda selected_task, **kwargs: observed.update(
            {"validated_task": selected_task, "validation_kwargs": kwargs}
        )
        or evidence,
    )
    monkeypatch.setattr(
        daemon,
        "apply_post_merge_authoritative_acceptance",
        lambda selected_task, **kwargs: observed.update(
            {"accepted_task": selected_task, "acceptance_kwargs": kwargs}
        )
        or {
            "updated": False,
            "authoritatively_completed": False,
            "completion_authoritative": False,
            "pending_gates": ["provider_review"],
            "todo_update_result": {},
        },
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    request = SimpleNamespace(
        branch_name="implementation/pmv-001",
        commit_sha=commit,
        task_id=task.task_id,
        priority=task.priority,
        attempt=1,
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        metadata={
            "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
            "validation_proof": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "selection": {"scope": "pre_merge"},
            },
            "task": {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "completion": task.completion,
                "priority": task.priority,
                "track": task.track,
                "validation": list(task.validation),
                "metadata": dict(task.metadata),
            },
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["post_merge_validation"] == evidence
    assert result["completion_authoritative"] is False
    assert result["acceptance_pending"] is True
    assert observed["validation_kwargs"] == {
        "target_commit": commit,
        "repository_tree_id": tree_id,
    }
    assert observed["acceptance_kwargs"]["validation_result"] == evidence


def test_reconciliation_reruns_exact_validation_instead_of_replaying_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    daemon, _repo, task, commit, tree_id = _runtime(
        tmp_path,
        scheduler=scheduler,
    )
    fresh = build_post_merge_validation_evidence(
        task_id=task.task_id,
        target_commit=commit,
        repository_tree_id=tree_id,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        },
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        daemon,
        "_validate_exact_post_merge_commit",
        lambda selected_task, **kwargs: observed.update(
            {"validated_task": selected_task, "validation_kwargs": kwargs}
        )
        or fresh,
    )
    monkeypatch.setattr(
        daemon,
        "apply_post_merge_authoritative_acceptance",
        lambda selected_task, **kwargs: observed.update(
            {"accepted_task": selected_task, "acceptance_kwargs": kwargs}
        )
        or {
            "updated": False,
            "authoritatively_completed": False,
            "completion_authoritative": False,
        },
    )
    stale_event = {
        "task_id": task.task_id,
        "merge_result": {
            "validation_result": {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "validation_scope": "pre_merge",
            }
        },
    }

    result = daemon._apply_reconciled_merge_authoritative_acceptance(
        task,
        stale_event,
        implementation_commit=commit,
    )

    assert result["acceptance_attempted"] is True
    assert result["validation_result"] == fresh
    assert observed["validation_kwargs"] == {
        "target_commit": commit,
        "repository_tree_id": tree_id,
    }
    assert observed["acceptance_kwargs"]["validation_result"] == fresh
