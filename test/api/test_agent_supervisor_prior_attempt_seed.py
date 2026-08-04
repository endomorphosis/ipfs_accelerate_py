"""Tests for prior-attempt worktree seeding and board completion decisions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


def _daemon(tmp_path: Path) -> PortalImplementationDaemon:
    # Leave merge_target unset so init does not require a real git ref in tmp_path.
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )


def _task(*outputs: str) -> PortalTask:
    return PortalTask(
        task_id="SEED-001",
        title="Seed retry",
        status="todo",
        completion="manual",
        priority="P1",
        track="test",
        outputs=list(outputs),
        validation=["python -m pytest"],
    )


def _authorize_seed(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
    *changed_paths: str,
) -> None:
    identity = daemon._identity_for_task(task)
    daemon._record_event(
        "implementation_proposal_validated",
        {
            "task_id": task.task_id,
            "canonical_task_cid": identity.canonical_task_cid,
            "canonical_task_key": identity.canonical_task_key,
            "accepted": True,
            "proposal_id": "accepted-seed-proposal",
            "receipt_id": "accepted-seed-receipt",
            "changed_paths": list(changed_paths),
        },
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


def _is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _seed_prior_child_delta(
    tmp_path: Path,
    *,
    divergent_baseline: bool,
) -> tuple[Path, str, str, str, str]:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    (child_source / "base.txt").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "base.txt")
    _git(child_source, "commit", "-m", "child base")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init")
    _git(parent, "checkout", "-b", "main")
    _git(parent, "config", "user.name", "Test User")
    _git(parent, "config", "user.email", "test@example.invalid")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "deps/child",
    )
    _git(parent, "commit", "-am", "parent base")
    baseline_parent = _git(parent, "rev-parse", "HEAD")
    child = parent / "deps" / "child"
    baseline_child = _git(child, "rev-parse", "HEAD")
    _git(child, "config", "user.name", "Test User")
    _git(child, "config", "user.email", "test@example.invalid")

    _git(parent, "checkout", "-b", "prior-attempt")
    _git(child, "checkout", "-b", "prior-child")
    for sequence in ("one", "two"):
        (child / f"prior-{sequence}.txt").write_text(
            f"prior {sequence}\n",
            encoding="utf-8",
        )
        _git(child, "add", f"prior-{sequence}.txt")
        _git(child, "commit", "-m", f"prior child {sequence}")
        (parent / f"prior-root-{sequence}.txt").write_text(
            f"prior root {sequence}\n",
            encoding="utf-8",
        )
        _git(parent, "add", f"prior-root-{sequence}.txt", "deps/child")
        _git(parent, "commit", "-m", f"prior parent {sequence}")
    seed_parent = _git(parent, "rev-parse", "HEAD")
    seed_child = _git(child, "rev-parse", "HEAD")
    _git(child, "push", "origin", "prior-child")

    if not divergent_baseline:
        return parent, baseline_parent, baseline_child, seed_parent, seed_child

    _git(parent, "checkout", "main")
    _git(child, "checkout", "main")
    (child / "current.txt").write_text("current baseline\n", encoding="utf-8")
    _git(child, "add", "current.txt")
    _git(child, "commit", "-m", "current child")
    baseline_child = _git(child, "rev-parse", "HEAD")
    _git(child, "push", "origin", "main:refs/heads/current-child")
    (parent / "current-root.txt").write_text(
        "current root baseline\n",
        encoding="utf-8",
    )
    _git(parent, "add", "current-root.txt", "deps/child")
    _git(parent, "commit", "-m", "current parent")
    baseline_parent = _git(parent, "rev-parse", "HEAD")
    return parent, baseline_parent, baseline_child, seed_parent, seed_child


def _retry_worktree(parent: Path, baseline: str, path: Path) -> Path:
    _git(parent, "worktree", "add", "-b", f"retry-{path.name}", str(path), baseline)
    _git(
        path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--",
        "deps/child",
    )
    return path


def test_board_completion_merge_queued_is_not_complete() -> None:
    decision = PortalImplementationDaemon._board_completion_decision(
        returncode=0,
        merge_result={"queued": True, "merged": False, "reason": "merge_queued"},
        no_change_completion=False,
    )
    assert decision["complete"] is False
    assert decision["pending_merge"] is True
    assert decision["reason"] == "merge_queued_awaiting_integration"


def test_board_completion_merged_is_complete() -> None:
    decision = PortalImplementationDaemon._board_completion_decision(
        returncode=0,
        merge_result={"queued": False, "merged": True, "reason": "merged"},
        no_change_completion=False,
    )
    assert decision["complete"] is True
    assert decision["pending_merge"] is False


def test_board_completion_validation_failure_is_incomplete() -> None:
    decision = PortalImplementationDaemon._board_completion_decision(
        returncode=78,
        merge_result={"queued": False, "merged": False},
        no_change_completion=False,
    )
    assert decision["complete"] is False
    assert decision["pending_merge"] is False


def test_prior_attempt_seed_plan_reuses_unmerged_commit(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    identity = daemon._identity_for_task(task)
    monkeypatch.setattr(
        daemon, "_main_branch_name", lambda: "feature/logic-intent-legal-gate"
    )
    monkeypatch.setattr(
        daemon,
        "_git_commit_exists_in_repo",
        lambda _repo, ref: ref == "abc123prior",
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor",
        lambda ancestor, descendant: False,
    )
    monkeypatch.setattr(
        daemon,
        "_branch_changed_paths_in_repo",
        lambda _repo, _ref, base_ref: {"src/output.py"},
    )
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_commit="abc123prior",
        last_implementation_branch="implementation/lig-016-attempt-1",
        last_implementation_returncode=78,
    )
    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=2)
    assert plan["reuse_prior_attempt"] is True
    assert plan["seed_ref"] == "abc123prior"
    assert plan["reason"] == "prior_failed_attempt_commit"


def test_prior_attempt_seed_plan_skips_when_already_on_target(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    identity = daemon._identity_for_task(task)
    monkeypatch.setattr(
        daemon, "_main_branch_name", lambda: "feature/logic-intent-legal-gate"
    )
    monkeypatch.setattr(
        daemon,
        "_git_commit_exists_in_repo",
        lambda _repo, ref: ref == "abc123prior",
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor",
        lambda ancestor, descendant: ancestor == "abc123prior",
    )
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_commit="abc123prior",
    )
    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=3)
    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] == "prior_already_on_merge_target"


def test_prior_attempt_seed_plan_first_attempt_uses_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/x")
    state = PortalTaskState(last_implementation_commit="abc123prior")
    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=1)
    assert plan["reuse_prior_attempt"] is False
    assert plan["seed_ref"] == "feature/x"
    assert plan["reason"] == "merge_target_baseline"


def test_prior_attempt_seed_plan_rejects_cross_task_last_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/x")
    state = PortalTaskState(
        last_implementation_task_id="OTHER-001",
        last_implementation_task_key="task/v1/other-001",
        last_implementation_task_cid="cid-other-001",
        last_implementation_commit="other-commit",
    )

    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=2)

    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] == "prior_attempt_task_identity_mismatch"
    assert plan["seed_ref"] == "feature/x"
    assert plan["prior_task_identity"]["task_id"] == "OTHER-001"


def test_prior_attempt_seed_plan_rejects_revised_canonical_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/x")
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key="task/v1/obsolete-seed-001",
        last_implementation_task_cid="cid-obsolete-seed-001",
        last_implementation_commit="obsolete-commit",
    )

    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=2)

    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] == "prior_attempt_task_identity_mismatch"
    assert plan["prior_task_identity"]["canonical_task_cid"] == (
        "cid-obsolete-seed-001"
    )


def test_prior_attempt_seed_plan_rejects_out_of_scope_prior_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    task = _task("src/output.py")
    identity = daemon._identity_for_task(task)
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/x")
    monkeypatch.setattr(
        daemon,
        "_git_commit_exists_in_repo",
        lambda _repo, ref: ref == "prior-commit",
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor",
        lambda _ancestor, _descendant: False,
    )
    monkeypatch.setattr(
        daemon,
        "_branch_changed_paths_in_repo",
        lambda _repo, _ref, base_ref: {
            "src/output.py",
            "faiss_index/generated.index",
        },
    )
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_commit="prior-commit",
    )

    plan = daemon._prior_attempt_seed_plan(task=task, state=state, attempt=2)

    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] == "prior_attempt_paths_outside_task_scope"
    assert plan["prior_out_of_scope_paths"] == [
        "faiss_index/generated.index"
    ]


def test_prior_seed_requires_accepted_same_identity_proposal(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    task = _task("allowed.txt")
    daemon._record_event(
        "implementation_proposal_validated",
        {
            "task_id": task.task_id,
            "canonical_task_cid": "wrong-task-revision",
            "accepted": True,
            "changed_paths": ["allowed.txt"],
        },
    )

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": "seed"},
        baseline_ref="baseline",
    )

    assert result["applied"] is False
    assert result["reason"] == "prior_seed_accepted_proposal_missing"


def test_prior_seed_authority_filters_protected_paths_and_matches_globs(
    tmp_path: Path,
) -> None:
    protected = "docs/private/protected.json"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        implementation_protected_paths=[protected],
    )
    task = _task("docs/**/*.json")
    _authorize_seed(
        daemon,
        task,
        "docs/public/report.json",
        protected,
    )

    authority = daemon._prior_seed_proposal_authority(task)

    assert authority["ok"] is True
    assert authority["authorized_paths"] == ["docs/public/report.json"]
    assert authority["dropped_protected_paths"] == [protected]
    assert authority["dropped_receipt_paths"] == [protected]


def test_apply_prior_attempt_seed_replays_without_moving_head(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[list[str]] = []
    task = _task("allowed.txt")
    _authorize_seed(daemon, task, "allowed.txt")

    def fake_run(
        cmd,
        cwd=None,
        text=True,
        capture_output=True,
        check=False,
        input=None,
    ):
        command = list(cmd)
        calls.append(command)

        class Result:
            returncode = 0
            stderr = ""
            stdout = b"synthetic patch" if "--binary" in command else ""

        return Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        daemon,
        "_git_merge_bases_in_repo",
        lambda *_args: ["baseline"],
    )
    monkeypatch.setattr(
        daemon,
        "_resolve_git_commit_in_repo",
        lambda _repo, ref: "baseline" if ref in {"baseline", "HEAD"} else "",
    )
    monkeypatch.setattr(
        daemon,
        "_prior_seed_changed_paths",
        lambda *_args: ("allowed.txt",),
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=True,
            findings=(),
            proposal=SimpleNamespace(
                proposal_id="pre-dispatch-proposal",
                repository_tree_id="baseline",
                changed_paths=(),
            ),
            policy=SimpleNamespace(policy_id="pre-dispatch-policy"),
            receipt=SimpleNamespace(receipt_id="pre-dispatch-receipt"),
        ),
    )
    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={
            "reuse_prior_attempt": True,
            "seed_ref": "abc123prior",
            "reason": "prior_failed_attempt_commit",
        },
        baseline_ref="baseline",
    )
    assert result["applied"] is True, result
    assert result["reason"] == "replayed_prior_delta"
    assert not any(cmd[:2] in (["git", "reset"], ["git", "merge"]) for cmd in calls)


def test_prior_seed_root_replay_failure_rolls_back_to_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[list[str]] = []
    task = _task("allowed.txt")
    _authorize_seed(daemon, task, "allowed.txt")

    def fake_run(
        cmd,
        cwd=None,
        text=True,
        capture_output=True,
        check=False,
        input=None,
    ):
        command = list(cmd)
        calls.append(command)

        class Result:
            returncode = int(command[:3] == ["git", "apply", "--3way"])
            stderr = b"apply failed" if returncode else ""
            stdout = b"synthetic patch" if "--binary" in command else ""

        return Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        daemon,
        "_git_merge_bases_in_repo",
        lambda *_args: ["baseline"],
    )
    monkeypatch.setattr(
        daemon,
        "_resolve_git_commit_in_repo",
        lambda _repo, ref: "baseline" if ref in {"baseline", "HEAD"} else "",
    )
    monkeypatch.setattr(
        daemon,
        "_prior_seed_changed_paths",
        lambda *_args: ("allowed.txt",),
    )

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": "seed"},
        baseline_ref="baseline",
    )

    assert result["applied"] is False
    assert result["reason"] == "prior_seed_apply_failed"
    assert result["rollback"]["reset"] is True
    assert ["git", "reset", "--hard", "baseline"] in calls


def test_prior_seed_fast_forward_replays_without_rejected_ancestry(
    tmp_path: Path,
) -> None:
    parent, baseline, base_child, _seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=False,
    )
    source_child = parent / "deps" / "child"
    (source_child / "unrelated-child.txt").write_text(
        "must not be replayed\n",
        encoding="utf-8",
    )
    _git(source_child, "add", "unrelated-child.txt")
    _git(source_child, "commit", "-m", "polluted child seed")
    _git(source_child, "push", "origin", "prior-child")
    polluted_daemon = (
        parent
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "todo_daemon"
        / "implementation_daemon.py"
    )
    polluted_daemon.parent.mkdir(parents=True)
    polluted_daemon.write_text("must not be replayed\n", encoding="utf-8")
    polluted_test = parent / "test" / "api" / "test_prior_seed_pollution.py"
    polluted_test.parent.mkdir(parents=True)
    polluted_test.write_text("must not be replayed\n", encoding="utf-8")
    _git(
        parent,
        "add",
        "deps/child",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_prior_seed_pollution.py",
    )
    _git(parent, "commit", "-m", "polluted rejected seed")
    seed = _git(parent, "rev-parse", "HEAD")
    seed_child = _git(source_child, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-fast-forward")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task(
        "prior-root-one.txt",
        "prior-root-two.txt",
        "deps/child/prior-one.txt",
        "deps/child/prior-two.txt",
    )
    _authorize_seed(daemon, task, *task.outputs)

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is True, result
    assert result["reason"] == "replayed_prior_delta"
    assert result["replayed_root_paths"] == [
        "prior-root-one.txt",
        "prior-root-two.txt",
    ]
    assert result["skipped_root_paths"] == [
        "deps/child",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_prior_seed_pollution.py",
    ]
    assert result["submodule_reconciliation"]["results"][0]["mode"] == "replay"
    assert result["submodule_reconciliation"]["results"][0]["skipped_paths"] == [
        "unrelated-child.txt"
    ]
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(child, "rev-parse", "HEAD") == base_child
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == base_child
    assert set(_git(child, "diff", "--cached", "--name-only").splitlines()) == {
        "prior-one.txt",
        "prior-two.txt",
    }
    assert not (child / "unrelated-child.txt").exists()
    assert not (
        worktree
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "todo_daemon"
        / "implementation_daemon.py"
    ).exists()
    assert not (worktree / "test" / "api" / "test_prior_seed_pollution.py").exists()
    assert not _is_ancestor(parent, seed, _git(worktree, "rev-parse", "HEAD"))
    assert not _is_ancestor(child, seed_child)
    seed_events = daemon._iter_events()
    assert sum(
        event["type"] == "implementation_proposal_validated"
        for event in seed_events
    ) == 1
    assert sum(
        event["type"]
        == "implementation_prior_attempt_seed_pre_dispatch_validated"
        for event in seed_events
    ) == 1
    assert not any(
        event["type"]
        in {
            "implementation_proposal_rejected",
            "implementation_scope_adjudicated",
            "implementation_secret_change_scope_examined",
        }
        for event in seed_events
    )
    assert daemon._implementation_scope_adjudications == {}

    later_validation = daemon._validate_implementation_patch(
        worktree,
        task,
        baseline_ref=baseline,
        replayable_consumed_proposal_ids=(
            result["proposal_authority"]["proposal_id"],
        ),
    )
    assert later_validation.accepted is True

    _git(child, "config", "user.name", "Test User")
    _git(child, "config", "user.email", "test@example.invalid")
    _git(child, "commit", "-m", "retry child delta")
    final_child = _git(child, "rev-parse", "HEAD")
    _git(worktree, "add", "deps/child")
    _git(worktree, "commit", "-m", "retry parent delta")
    final_parent = _git(worktree, "rev-parse", "HEAD")
    assert _is_ancestor(child, base_child, final_child)
    assert not _is_ancestor(child, seed_child, final_child)
    assert _is_ancestor(worktree, baseline, final_parent)
    assert not _is_ancestor(worktree, seed, final_parent)


def test_prior_seed_root_only_replay_keeps_configured_child_at_baseline(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = (
        _seed_prior_child_delta(tmp_path, divergent_baseline=False)
    )
    _git(parent, "checkout", "main")
    child = parent / "deps" / "child"
    _git(child, "checkout", "main")
    (parent / "root-only.txt").write_text("root-only prior delta\n", encoding="utf-8")
    _git(parent, "add", "root-only.txt")
    _git(parent, "commit", "-m", "rejected root-only seed")
    root_only_seed = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-root-only")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("root-only.txt")
    _authorize_seed(daemon, task, "root-only.txt")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": root_only_seed},
        baseline_ref=baseline,
    )

    retry_child = worktree / "deps" / "child"
    assert result["applied"] is True, result
    assert result["reason"] == "replayed_prior_delta"
    assert result["no_change"] is False
    assert result["submodule_reconciliation"]["results"][0]["replayed"] is False
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(retry_child, "rev-parse", "HEAD") == baseline_child
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(worktree, "diff", "--cached", "--name-only") == "root-only.txt"
    assert _git(retry_child, "diff", "--cached", "--name-only") == ""
    assert not _is_ancestor(worktree, root_only_seed)

    _git(worktree, "commit", "-m", "retry root-only delta")
    final_parent = _git(worktree, "rev-parse", "HEAD")
    assert _is_ancestor(worktree, baseline, final_parent)
    assert not _is_ancestor(worktree, root_only_seed, final_parent)


def test_prior_seed_empty_tree_delta_is_explicit_no_change(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = (
        _seed_prior_child_delta(tmp_path, divergent_baseline=False)
    )
    _git(parent, "checkout", "main")
    child = parent / "deps" / "child"
    _git(child, "checkout", "main")
    _git(parent, "commit", "--allow-empty", "-m", "rejected empty-tree seed")
    empty_seed = _git(parent, "rev-parse", "HEAD")
    assert empty_seed != baseline

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-empty-tree")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("empty-authorized.txt")
    _authorize_seed(daemon, task, "empty-authorized.txt")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": empty_seed},
        baseline_ref=baseline,
    )

    retry_child = worktree / "deps" / "child"
    assert result["applied"] is False, result
    assert result["reason"] == "prior_seed_no_authorized_change"
    assert result["no_change"] is True
    assert result["no_change_certified"] is True
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(retry_child, "rev-parse", "HEAD") == baseline_child
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(worktree, "diff", "--cached", "--name-only") == ""
    assert _git(retry_child, "diff", "--cached", "--name-only") == ""
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(retry_child, "status", "--porcelain") == ""
    assert not _is_ancestor(worktree, empty_seed)


def test_prior_seed_root_secret_fails_pre_dispatch_and_rolls_back(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = (
        _seed_prior_child_delta(tmp_path, divergent_baseline=False)
    )
    _git(parent, "checkout", "main")
    source_child = parent / "deps" / "child"
    _git(source_child, "checkout", "main")
    (parent / "secret.py").write_text(
        'api_key = "sk-live-concrete-credential-value"\n',
        encoding="utf-8",
    )
    _git(parent, "add", "secret.py")
    _git(parent, "commit", "-m", "rejected root secret")
    secret_seed = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-root-secret")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("secret.py")
    _authorize_seed(daemon, task, "secret.py")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": secret_seed},
        baseline_ref=baseline,
    )

    retry_child = worktree / "deps" / "child"
    assert result["applied"] is False
    assert result["reason"] == "prior_seed_pre_dispatch_validation_failed"
    assert "secret_change_forbidden" in (
        result["pre_dispatch_proposal_gate"]["reason_codes"]
    )
    assert result["rollback"]["reset"] is True
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(retry_child, "rev-parse", "HEAD") == baseline_child
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(retry_child, "status", "--porcelain") == ""
    assert not (worktree / "secret.py").exists()
    assert daemon._implementation_scope_adjudications == {}
    assert not any(
        event["type"] == "implementation_secret_change_scope_examined"
        for event in daemon._iter_events()
    )


def test_prior_seed_child_secret_fails_pre_dispatch_and_rolls_back(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = (
        _seed_prior_child_delta(tmp_path, divergent_baseline=False)
    )
    source_child = parent / "deps" / "child"
    (source_child / "secret.py").write_text(
        'api_key = "sk-live-concrete-credential-value"\n',
        encoding="utf-8",
    )
    _git(source_child, "add", "secret.py")
    _git(source_child, "commit", "-m", "rejected child secret")
    _git(source_child, "push", "origin", "prior-child")
    _git(parent, "add", "deps/child")
    _git(parent, "commit", "-m", "seed rejected child secret")
    secret_seed = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-child-secret")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("deps/child/secret.py")
    _authorize_seed(daemon, task, "deps/child/secret.py")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": secret_seed},
        baseline_ref=baseline,
    )

    retry_child = worktree / "deps" / "child"
    assert result["applied"] is False
    assert result["reason"] == "prior_seed_pre_dispatch_validation_failed"
    assert "secret_change_forbidden" in (
        result["pre_dispatch_proposal_gate"]["reason_codes"]
    )
    assert result["rollback"]["reset"] is True
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(retry_child, "rev-parse", "HEAD") == baseline_child
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(retry_child, "status", "--porcelain") == ""
    assert not (retry_child / "secret.py").exists()
    assert daemon._implementation_scope_adjudications == {}
    assert not any(
        event["type"] == "implementation_secret_change_scope_examined"
        for event in daemon._iter_events()
    )


def test_prior_seed_rejects_unreconciled_nested_gitlink_delta(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=False,
    )
    leaf_source = tmp_path / "leaf-source"
    leaf_source.mkdir()
    _git(leaf_source, "init")
    _git(leaf_source, "checkout", "-b", "main")
    _git(leaf_source, "config", "user.name", "Test User")
    _git(leaf_source, "config", "user.email", "test@example.invalid")
    (leaf_source / "leaf.txt").write_text("leaf\n", encoding="utf-8")
    _git(leaf_source, "add", "leaf.txt")
    _git(leaf_source, "commit", "-m", "leaf base")

    child = parent / "deps" / "child"
    _git(
        child,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf_source),
        "vendor/leaf",
    )
    _git(child, "commit", "-am", "add nested leaf")
    _git(child, "push", "origin", "prior-child")
    _git(parent, "add", "deps/child")
    _git(parent, "commit", "-m", "seed nested leaf")
    nested_seed = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-nested")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("deps/child")
    _authorize_seed(daemon, task, "deps/child")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": nested_seed},
        baseline_ref=baseline,
    )

    assert result["applied"] is False
    assert result["reason"] == "prior_seed_submodule_preflight_failed"
    assert (
        result["submodule_reconciliation"]["reason"]
        == "prior_seed_nested_gitlink_changed"
    )
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(worktree / "deps" / "child", "rev-parse", "HEAD") == baseline_child


def test_prior_seed_rejects_unconfigured_root_gitlink_delta(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=False,
    )
    other_source = tmp_path / "other-source"
    other_source.mkdir()
    _git(other_source, "init")
    _git(other_source, "checkout", "-b", "main")
    _git(other_source, "config", "user.name", "Test User")
    _git(other_source, "config", "user.email", "test@example.invalid")
    (other_source / "other.txt").write_text("other\n", encoding="utf-8")
    _git(other_source, "add", "other.txt")
    _git(other_source, "commit", "-m", "other base")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(other_source),
        "vendor/other",
    )
    _git(parent, "commit", "-am", "seed unconfigured submodule")
    unsafe_seed = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-unconfigured")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("vendor/other")
    _authorize_seed(daemon, task, "vendor/other")
    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": unsafe_seed},
        baseline_ref=baseline,
    )

    assert result["applied"] is False
    assert (
        result["submodule_reconciliation"]["reason"]
        == "unconfigured_prior_seed_gitlink_changed"
    )
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert not (worktree / "vendor" / "other").exists()


def test_prior_seed_ignores_unconfigured_gitlink_advanced_only_on_main(
    tmp_path: Path,
) -> None:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    (child_source / "child.txt").write_text("child base\n", encoding="utf-8")
    _git(child_source, "add", "child.txt")
    _git(child_source, "commit", "-m", "child base")

    other_source = tmp_path / "other-source"
    other_source.mkdir()
    _git(other_source, "init")
    _git(other_source, "checkout", "-b", "main")
    _git(other_source, "config", "user.name", "Test User")
    _git(other_source, "config", "user.email", "test@example.invalid")
    (other_source / "other.txt").write_text("other base\n", encoding="utf-8")
    _git(other_source, "add", "other.txt")
    _git(other_source, "commit", "-m", "other base")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init")
    _git(parent, "checkout", "-b", "main")
    _git(parent, "config", "user.name", "Test User")
    _git(parent, "config", "user.email", "test@example.invalid")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "deps/child",
    )
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(other_source),
        "vendor/other",
    )
    _git(parent, "commit", "-am", "common parent")

    _git(parent, "checkout", "-b", "prior-attempt")
    (parent / "allowed.txt").write_text("authorized retry work\n", encoding="utf-8")
    _git(parent, "add", "allowed.txt")
    _git(parent, "commit", "-m", "rejected authorized seed")
    seed = _git(parent, "rev-parse", "HEAD")

    _git(parent, "checkout", "main")
    other = parent / "vendor" / "other"
    _git(other, "config", "user.name", "Test User")
    _git(other, "config", "user.email", "test@example.invalid")
    (other / "main-only.txt").write_text("main-only advance\n", encoding="utf-8")
    _git(other, "add", "main-only.txt")
    _git(other, "commit", "-m", "advance other only on main")
    main_other = _git(other, "rev-parse", "HEAD")
    _git(parent, "add", "vendor/other")
    _git(parent, "commit", "-m", "advance unconfigured dependency on main")
    baseline = _git(parent, "rev-parse", "HEAD")

    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-main-only-other")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task("allowed.txt")
    _authorize_seed(daemon, task, "allowed.txt")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    assert result["applied"] is True, result
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(worktree, "diff", "--cached", "--name-only") == "allowed.txt"
    assert daemon._proposal_index_gitlink_ref(worktree, "vendor/other") == main_other
    assert not _is_ancestor(worktree, seed)


@pytest.mark.parametrize("main_change", ["add", "advance"])
def test_prior_seed_preserves_configured_dependency_changed_only_on_main(
    tmp_path: Path,
    main_change: str,
) -> None:
    dependency_source = tmp_path / "main-only-source"
    dependency_source.mkdir()
    _git(dependency_source, "init")
    _git(dependency_source, "checkout", "-b", "main")
    _git(dependency_source, "config", "user.name", "Test User")
    _git(dependency_source, "config", "user.email", "test@example.invalid")
    (dependency_source / "dependency.txt").write_text(
        "dependency base\n",
        encoding="utf-8",
    )
    _git(dependency_source, "add", "dependency.txt")
    _git(dependency_source, "commit", "-m", "dependency base")

    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init")
    _git(parent, "checkout", "-b", "main")
    _git(parent, "config", "user.name", "Test User")
    _git(parent, "config", "user.email", "test@example.invalid")
    (parent / "base.txt").write_text("parent base\n", encoding="utf-8")
    _git(parent, "add", "base.txt")
    _git(parent, "commit", "-m", "parent base")
    if main_change == "advance":
        _git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(dependency_source),
            "deps/main-only",
        )
        _git(parent, "commit", "-am", "common configured dependency")

    _git(parent, "checkout", "-b", "prior-attempt")
    (parent / "allowed.txt").write_text("authorized retry work\n", encoding="utf-8")
    _git(parent, "add", "allowed.txt")
    _git(parent, "commit", "-m", "rejected authorized seed")
    seed = _git(parent, "rev-parse", "HEAD")

    _git(parent, "checkout", "main")
    if main_change == "add":
        _git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(dependency_source),
            "deps/main-only",
        )
        _git(parent, "commit", "-am", "add configured dependency on main")
    else:
        dependency = parent / "deps" / "main-only"
        _git(dependency, "config", "user.name", "Test User")
        _git(dependency, "config", "user.email", "test@example.invalid")
        (dependency / "main-only.txt").write_text(
            "main-only advance\n",
            encoding="utf-8",
        )
        _git(dependency, "add", "main-only.txt")
        _git(dependency, "commit", "-m", "advance dependency only on main")
        _git(dependency, "push", "origin", "HEAD:refs/heads/main-advance")
        _git(parent, "add", "deps/main-only")
        _git(parent, "commit", "-m", "advance configured dependency on main")
    baseline = _git(parent, "rev-parse", "HEAD")
    main_dependency = _git(parent / "deps" / "main-only", "rev-parse", "HEAD")

    worktree = tmp_path / f"retry-configured-{main_change}"
    _git(
        parent,
        "worktree",
        "add",
        "-b",
        f"retry-{main_change}",
        str(worktree),
        baseline,
    )
    _git(
        worktree,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--",
        "deps/main-only",
    )
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/main-only"],
    )
    task = _task("allowed.txt")
    _authorize_seed(daemon, task, "allowed.txt")

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    dependency = worktree / "deps" / "main-only"
    assert result["applied"] is True, result
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert _git(dependency, "rev-parse", "HEAD") == main_dependency
    assert daemon._proposal_index_gitlink_ref(
        worktree,
        "deps/main-only",
    ) == main_dependency
    assert _git(worktree, "diff", "--cached", "--name-only") == "allowed.txt"
    assert _git(dependency, "diff", "--cached", "--name-only") == ""
    assert not _is_ancestor(worktree, seed)


def test_prior_seed_divergence_preserves_baseline_and_replays_full_child_delta(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=True,
    )
    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-divergent")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task(
        "prior-root-one.txt",
        "prior-root-two.txt",
        "deps/child/prior-one.txt",
        "deps/child/prior-two.txt",
    )
    _authorize_seed(daemon, task, *task.outputs)

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is True
    assert result["submodule_reconciliation"]["results"][0]["mode"] == "replay"
    assert result["submodule_reconciliation"]["results"][0]["replayed"] is True
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(child, "rev-parse", "HEAD") == baseline_child
    assert (child / "current.txt").read_text(encoding="utf-8") == "current baseline\n"
    assert (child / "prior-one.txt").read_text(encoding="utf-8") == "prior one\n"
    assert (child / "prior-two.txt").read_text(encoding="utf-8") == "prior two\n"
    assert (worktree / "current-root.txt").read_text(encoding="utf-8") == (
        "current root baseline\n"
    )
    assert set(_git(child, "diff", "--cached", "--name-only").splitlines()) == {
        "prior-one.txt",
        "prior-two.txt",
    }

    entries, expansions = daemon._collect_proposal_candidate_diff(
        worktree,
        baseline_ref=baseline,
        scope_paths=["prior-root-one.txt", "deps/child/prior-one.txt"],
    )
    assert {entry.path for entry in entries} >= {
        "prior-root-one.txt",
        "deps/child/prior-one.txt",
        "deps/child/prior-two.txt",
    }
    assert expansions[0]["base_revision"] == baseline_child


def test_prior_seed_replays_child_only_divergent_delta(
    tmp_path: Path,
) -> None:
    parent, baseline, baseline_child, _seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=True,
    )
    _git(parent, "checkout", "prior-attempt")
    _git(parent, "rm", "prior-root-one.txt", "prior-root-two.txt")
    _git(parent, "commit", "-m", "retain only prior child delta")
    child_only_seed = _git(parent, "rev-parse", "HEAD")
    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-child-only")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    task = _task(
        "deps/child/prior-one.txt",
        "deps/child/prior-two.txt",
    )
    _authorize_seed(daemon, task, *task.outputs)

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": child_only_seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is True, result
    assert result["reason"] == "replayed_prior_delta"
    assert result["submodule_reconciliation"]["results"][0]["replayed"] is True
    assert _git(child, "rev-parse", "HEAD") == baseline_child
    assert (child / "prior-one.txt").is_file()
    assert (child / "prior-two.txt").is_file()
    assert (worktree / "current-root.txt").is_file()


def test_prior_seed_reconciliation_failure_removes_staged_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent, baseline, baseline_child, seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=True,
    )
    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-rollback")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    real_index_gitlink = daemon._proposal_index_gitlink_ref
    inspections = 0

    def fail_first_postcondition(workspace: Path, relative: str) -> str:
        nonlocal inspections
        inspections += 1
        if inspections == 1:
            return "f" * 40
        return real_index_gitlink(workspace, relative)

    monkeypatch.setattr(
        daemon,
        "_proposal_index_gitlink_ref",
        fail_first_postcondition,
    )
    task = _task(
        "prior-root-one.txt",
        "prior-root-two.txt",
        "deps/child/prior-one.txt",
        "deps/child/prior-two.txt",
    )
    _authorize_seed(daemon, task, *task.outputs)

    result = daemon._apply_prior_attempt_seed(
        worktree,
        task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is False
    assert result["reason"] == "prior_seed_submodule_reconciliation_failed"
    assert result["rollback"]["reset"] is True
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(child, "rev-parse", "HEAD") == baseline_child
    assert _git(child, "status", "--porcelain") == ""
    assert not (child / "prior-one.txt").exists()
    assert not (child / "prior-two.txt").exists()


def test_daemon_configured_merge_target_used_for_seed_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    target = "feature/logic-intent-legal-gate"
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_git_ref_exists",
        lambda _self, ref: ref == target,
    )
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        merge_target_branch=target,
    )
    assert daemon._main_branch_name() == target
    assert daemon.resolved_merge_target_branch == target


def test_consume_merge_rejects_queue_target_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    daemon.resolved_merge_target_branch = "feature/logic-intent-legal-gate"
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/logic-intent-legal-gate")

    class _Queue:
        target_branch = "main"
        max_attempts = 3

    daemon.merge_queue = _Queue()  # type: ignore[assignment]
    try:
        daemon._consume_one_merge_candidate()
    except RuntimeError as error:
        assert "differs from daemon merge target" in str(error)
    else:
        raise AssertionError("mismatched merge queue target must fail closed")


def test_record_prior_attempt_seed_failure_writes_guidance(
    tmp_path: Path,
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalTask,
    )

    task = PortalTask(
        task_id="LIG-016",
        title="t",
        status="pending",
        completion="manual",
        priority="P0",
        track="gate",
        outputs=["tests/fixtures/logic/admissibility"],
        canonical_task_cid="cid-lig-016",
    )
    daemon._record_prior_attempt_seed_failure(
        task=task,
        attempt=2,
        seed_plan={
            "reuse_prior_attempt": True,
            "prior_commit": "abc123",
            "prior_branch": "implementation/lig-016-attempt-1",
            "seed_ref": "abc123",
        },
        seed_apply={"applied": False, "reason": "prior_seed_apply_failed"},
        worktree_path=worktree,
        branch_name="implementation/lig-016-attempt-2",
    )
    key = daemon._canonical_ref(task)
    assert key in daemon._implementation_seed_failure_guidance
    assert "abc123" in daemon._implementation_seed_failure_guidance[key]
    assert (
        "recover preserved work"
        in daemon._implementation_seed_failure_guidance[key]
    )
    guide = (
        daemon.implementation_log_dir
        / "seed_recovery"
        / "lig-016-attempt-2-seed-recovery.md"
    )
    assert guide.is_file()
    assert "compactly" in guide.read_text(encoding="utf-8")
    assert not (worktree / "docs" / "agent-supervisor" / "rescue").exists()
    event = json.loads(
        daemon.events_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert event["type"] == "implementation_prior_attempt_seed_failed"
    assert event["guidance_path"] == str(guide)


def test_rejected_prior_seed_guidance_forbids_replay_and_reaches_retry_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    task = _task("src/repair.py")
    prior_commit = "a" * 40

    daemon._record_prior_attempt_seed_failure(
        task=task,
        attempt=2,
        seed_plan={
            "reuse_prior_attempt": True,
            "prior_commit": prior_commit,
            "prior_branch": "implementation/seed-001-attempt-1",
            "seed_ref": prior_commit,
        },
        seed_apply={
            "applied": False,
            "reason": "prior_seed_accepted_proposal_missing",
        },
        worktree_path=worktree,
        branch_name="implementation/seed-001-attempt-2",
    )

    key = daemon._canonical_ref(task)
    guidance = daemon._implementation_seed_failure_guidance[key]
    assert "read-only diagnostic evidence only" in guidance
    assert "MUST NOT cherry-pick, merge, apply, or replay it" in guidance
    assert (
        "remove every reported proposal, security, and validation finding"
        in guidance
    )
    assert "recover preserved work" not in guidance

    monkeypatch.setattr(
        daemon,
        "_compile_implementation_context",
        lambda _task, _attempt: SimpleNamespace(capsule=object()),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "render_context_capsule",
        lambda _capsule: "base implementation prompt",
    )

    prompt = daemon._build_implementation_prompt(task, attempt=2)

    assert "## Prior attempt seed recovery" in prompt
    assert guidance in prompt
    assert key not in daemon._implementation_seed_failure_guidance
