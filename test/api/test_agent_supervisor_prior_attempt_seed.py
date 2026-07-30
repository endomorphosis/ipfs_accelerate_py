"""Tests for prior-attempt worktree seeding and board completion decisions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
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
    state = PortalTaskState(
        last_implementation_commit="abc123prior",
        last_implementation_branch="implementation/lig-016-attempt-1",
        last_implementation_returncode=78,
    )
    plan = daemon._prior_attempt_seed_plan(state=state, attempt=2)
    assert plan["reuse_prior_attempt"] is True
    assert plan["seed_ref"] == "abc123prior"
    assert plan["reason"] == "prior_failed_attempt_commit"


def test_prior_attempt_seed_plan_skips_when_already_on_target(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
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
    state = PortalTaskState(last_implementation_commit="abc123prior")
    plan = daemon._prior_attempt_seed_plan(state=state, attempt=3)
    assert plan["reuse_prior_attempt"] is False
    assert plan["reason"] == "prior_already_on_merge_target"


def test_prior_attempt_seed_plan_first_attempt_uses_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "feature/x")
    state = PortalTaskState(last_implementation_commit="abc123prior")
    plan = daemon._prior_attempt_seed_plan(state=state, attempt=1)
    assert plan["reuse_prior_attempt"] is False
    assert plan["seed_ref"] == "feature/x"
    assert plan["reason"] == "merge_target_baseline"


def test_apply_prior_attempt_seed_fast_forward(
    tmp_path: Path, monkeypatch
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, text=True, capture_output=True, check=False):
        calls.append(list(cmd))

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda cwd, ancestor, descendant: True,
    )
    result = daemon._apply_prior_attempt_seed(
        worktree,
        seed_plan={
            "reuse_prior_attempt": True,
            "seed_ref": "abc123prior",
            "reason": "prior_failed_attempt_commit",
        },
        baseline_ref="baseline",
    )
    assert result["applied"] is True, result
    assert result["reason"] == "fast_forward_reset"
    assert any(cmd[:3] == ["git", "reset", "--hard"] for cmd in calls)


def test_prior_seed_reset_failure_rolls_back_to_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, text=True, capture_output=True, check=False):
        calls.append(list(cmd))

        class Result:
            returncode = int(list(cmd) == ["git", "reset", "--hard", "seed"])
            stderr = "reset failed" if returncode else ""
            stdout = (
                "baseline\n"
                if list(cmd)[-1:] in (["baseline^{commit}"], ["HEAD^{commit}"])
                else ""
            )

        return Result()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda *_args: True,
    )

    result = daemon._apply_prior_attempt_seed(
        worktree,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": "seed"},
        baseline_ref="baseline",
    )

    assert result["applied"] is False
    assert result["reason"] == "fast_forward_reset_failed"
    assert result["rollback"]["reset"] is True
    assert ["git", "reset", "--hard", "baseline"] in calls


def test_prior_seed_fast_forward_aligns_clean_child_to_seed_gitlink(
    tmp_path: Path,
) -> None:
    parent, baseline, base_child, seed, seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=False,
    )
    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-fast-forward")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )

    result = daemon._apply_prior_attempt_seed(
        worktree,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is True, result
    assert result["reason"] == "fast_forward_reset"
    assert result["submodule_reconciliation"]["results"][0]["mode"] == "align"
    assert _git(child, "rev-parse", "HEAD") == seed_child
    assert _git(worktree, "rev-parse", "HEAD:deps/child") == seed_child
    assert _git(child, "status", "--porcelain") == ""
    assert _git(child, "merge-base", "--is-ancestor", base_child, seed_child) == ""


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

    result = daemon._apply_prior_attempt_seed(
        worktree,
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
    result = daemon._apply_prior_attempt_seed(
        worktree,
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

    result = daemon._apply_prior_attempt_seed(
        worktree,
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

    result = daemon._apply_prior_attempt_seed(
        worktree,
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


def test_prior_seed_abort_failure_rolls_back_without_fallback_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent, baseline, baseline_child, seed, _seed_child = _seed_prior_child_delta(
        tmp_path,
        divergent_baseline=True,
    )
    worktree = _retry_worktree(parent, baseline, tmp_path / "retry-abort")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=["deps/child"],
    )
    real_run = subprocess.run

    def fail_merge_abort(command, *args, **kwargs):
        if list(command) == ["git", "merge", "--abort"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="forced abort failure",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fail_merge_abort,
    )

    result = daemon._apply_prior_attempt_seed(
        worktree,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": seed},
        baseline_ref=baseline,
    )

    child = worktree / "deps" / "child"
    assert result["applied"] is False
    assert result["reason"] == "prior_seed_merge_abort_failed"
    assert result["rollback"]["reset"] is True
    assert _git(worktree, "rev-parse", "HEAD") == baseline
    assert daemon._proposal_index_gitlink_ref(worktree, "deps/child") == baseline_child
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(child, "rev-parse", "HEAD") == baseline_child
    assert _git(child, "status", "--porcelain") == ""


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
    guide = (
        worktree
        / "docs"
        / "agent-supervisor"
        / "rescue"
        / "lig-016-attempt-2-seed-recovery.md"
    )
    assert guide.is_file()
    assert "compactly" in guide.read_text(encoding="utf-8")
