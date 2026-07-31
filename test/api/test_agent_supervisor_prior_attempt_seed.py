"""Tests for prior-attempt worktree seeding and board completion decisions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")


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
    assert result["applied"] is True
    assert result["reason"] == "fast_forward_reset"
    assert any(cmd[:3] == ["git", "reset", "--hard"] for cmd in calls)


def test_prior_root_seed_fast_forwards_clean_task_owned_child_before_proposal(
    tmp_path: Path,
) -> None:
    child_source = tmp_path / "child-source"
    _init_repo(child_source)
    (child_source / "README.md").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "README.md")
    _git(child_source, "commit", "-m", "child baseline")
    child_baseline = _git(child_source, "rev-parse", "HEAD")

    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "external/child",
    )
    _git(repo, "commit", "-am", "root baseline")
    root_baseline = _git(repo, "rev-parse", "HEAD")

    canonical_child = repo / "external/child"
    output = canonical_child / "docs" / "contract.md"
    output.parent.mkdir(parents=True)
    output.write_text("preserved attempt\n", encoding="utf-8")
    _git(canonical_child, "add", "docs/contract.md")
    _git(canonical_child, "commit", "-m", "preserved child attempt")
    seeded_child = _git(canonical_child, "rev-parse", "HEAD")
    _git(repo, "add", "external/child")
    _git(repo, "commit", "-m", "preserved root attempt")
    prior_root = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "target-with-unrelated-change", root_baseline)
    (repo / "target-only.txt").write_text("target moved\n", encoding="utf-8")
    _git(repo, "add", "target-only.txt")
    _git(repo, "commit", "-m", "advance target independently")
    root_baseline = _git(repo, "rev-parse", "HEAD")

    workspace = tmp_path / "workspace"
    branch_name = "implementation/test-attempt-2"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        branch_name,
        str(workspace),
        root_baseline,
    )
    child_workspace = workspace / "external/child"
    child_workspace.rmdir()
    child_branch = f"{branch_name}-submodule-external-child"
    _git(
        canonical_child,
        "worktree",
        "add",
        "-b",
        child_branch,
        str(child_workspace),
        child_baseline,
    )

    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/child",),
    )
    task = PortalTask(
        task_id="UIR-001",
        title="Define the UI/UX IR contract",
        status="pending",
        completion="manual",
        priority="P0",
        track="architecture",
        outputs=["external/child/docs/contract.md"],
        validation=["python -m pytest"],
    )
    seed_result = daemon._apply_prior_attempt_seed(
        workspace,
        seed_plan={
            "reuse_prior_attempt": True,
            "seed_ref": prior_root,
            "reason": "prior_failed_attempt_commit",
        },
        baseline_ref=root_baseline,
    )
    assert seed_result["applied"] is True
    assert seed_result["reason"] == "merged_prior_seed"
    assert _git(workspace, "rev-parse", ":external/child") == seeded_child
    assert _git(child_workspace, "rev-parse", "HEAD") == child_baseline

    sync_result = daemon._synchronize_prior_attempt_seed_submodules(
        workspace,
        task=task,
        branch_name=branch_name,
        baseline_ref=root_baseline,
        seed_reason=str(seed_result["reason"]),
        seed_ref=prior_root,
    )

    assert sync_result["synchronized_count"] == 1
    assert _git(child_workspace, "rev-parse", "HEAD") == seeded_child
    assert _git(child_workspace, "branch", "--show-current") == child_branch
    assert _git(child_workspace, "status", "--porcelain") == ""

    output = child_workspace / "docs" / "contract.md"
    output.write_text("corrected provider output\n", encoding="utf-8")
    entries, expansions = daemon._collect_proposal_candidate_diff(
        workspace,
        baseline_ref=root_baseline,
        scope_paths=daemon._proposal_scope_paths(task),
    )
    assert [item["path"] for item in expansions] == ["external/child"]
    contract_entry = next(
        entry
        for entry in entries
        if entry.new_path == "external/child/docs/contract.md"
    )
    assert contract_entry.after_source == "corrected provider output\n"
    assert all(
        "Subproject commit" not in (entry.after_source or "")
        for entry in entries
    )
    proposal_validation = daemon._validate_implementation_patch(
        workspace,
        task,
        baseline_ref=root_baseline,
    )
    assert proposal_validation.accepted is True
    assert proposal_validation.proposal.changed_paths == (
        "external/child/docs/contract.md",
    )
    assert proposal_validation.proposal.candidate_diff[0].after_source == (
        "corrected provider output\n"
    )
    assert "Subproject commit" not in proposal_validation.proposal.patch_text


def test_prior_seed_submodule_sync_rejects_dirty_child_without_mutation(
    tmp_path: Path,
) -> None:
    child_source = tmp_path / "child-source"
    _init_repo(child_source)
    (child_source / "README.md").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "README.md")
    _git(child_source, "commit", "-m", "child baseline")
    child_baseline = _git(child_source, "rev-parse", "HEAD")

    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "external/child",
    )
    _git(repo, "commit", "-am", "root baseline")
    root_baseline = _git(repo, "rev-parse", "HEAD")
    canonical_child = repo / "external/child"
    (canonical_child / "seeded.txt").write_text(
        "preserved seed\n",
        encoding="utf-8",
    )
    _git(canonical_child, "add", "seeded.txt")
    _git(canonical_child, "commit", "-m", "advance preserved child")
    seeded_child = _git(canonical_child, "rev-parse", "HEAD")
    _git(repo, "add", "external/child")
    _git(repo, "commit", "-m", "advance preserved root gitlink")
    prior_root = _git(repo, "rev-parse", "HEAD")

    workspace = tmp_path / "workspace"
    branch_name = "implementation/dirty-attempt-2"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        branch_name,
        str(workspace),
        root_baseline,
    )
    child_workspace = workspace / "external/child"
    child_workspace.rmdir()
    child_branch = f"{branch_name}-submodule-external-child"
    _git(
        repo / "external/child",
        "worktree",
        "add",
        "-b",
        child_branch,
        str(child_workspace),
        child_baseline,
    )
    dirty_path = child_workspace / "operator-context.txt"
    dirty_path.write_text("must be preserved\n", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/child",),
    )
    task = PortalTask(
        task_id="UIR-001",
        title="Define the UI/UX IR contract",
        status="pending",
        completion="manual",
        priority="P0",
        track="architecture",
        outputs=["external/child/docs/contract.md"],
    )
    seed_result = daemon._apply_prior_attempt_seed(
        workspace,
        seed_plan={
            "reuse_prior_attempt": True,
            "seed_ref": prior_root,
            "reason": "prior_failed_attempt_commit",
        },
        baseline_ref=root_baseline,
    )
    assert seed_result["applied"] is True
    assert _git(workspace, "rev-parse", ":external/child") == seeded_child

    with pytest.raises(RuntimeError, match="checkout is dirty"):
        daemon._synchronize_prior_attempt_seed_submodules(
            workspace,
            task=task,
            branch_name=branch_name,
            baseline_ref=root_baseline,
            seed_reason=str(seed_result["reason"]),
            seed_ref=prior_root,
        )

    assert _git(child_workspace, "rev-parse", "HEAD") == child_baseline
    assert _git(child_workspace, "branch", "--show-current") == child_branch
    assert dirty_path.read_text(encoding="utf-8") == "must be preserved\n"


def test_checked_out_prior_tree_aligns_child_against_seed_tree(
    tmp_path: Path,
) -> None:
    child_source = tmp_path / "child-source"
    _init_repo(child_source)
    (child_source / "README.md").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "README.md")
    _git(child_source, "commit", "-m", "child baseline")
    child_baseline = _git(child_source, "rev-parse", "HEAD")

    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "external/child",
    )
    _git(repo, "commit", "-am", "root baseline")
    root_baseline = _git(repo, "rev-parse", "HEAD")
    canonical_child = repo / "external/child"
    output = canonical_child / "docs" / "contract.md"
    output.parent.mkdir(parents=True)
    output.write_text("preserved attempt\n", encoding="utf-8")
    _git(canonical_child, "add", "docs/contract.md")
    _git(canonical_child, "commit", "-m", "preserved child attempt")
    seeded_child = _git(canonical_child, "rev-parse", "HEAD")
    _git(repo, "add", "external/child")
    _git(repo, "commit", "-m", "preserved root attempt")
    prior_root = _git(repo, "rev-parse", "HEAD")

    workspace = tmp_path / "workspace"
    branch_name = "implementation/checkout-fallback-attempt-2"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        branch_name,
        str(workspace),
        root_baseline,
    )
    child_workspace = workspace / "external/child"
    child_workspace.rmdir()
    child_branch = f"{branch_name}-submodule-external-child"
    _git(
        canonical_child,
        "worktree",
        "add",
        "-b",
        child_branch,
        str(child_workspace),
        child_baseline,
    )
    _git(workspace, "checkout", prior_root, "--", ".")
    assert _git(workspace, "rev-parse", "HEAD") == root_baseline
    assert _git(workspace, "rev-parse", ":external/child") == seeded_child
    assert _git(child_workspace, "rev-parse", "HEAD") == child_baseline

    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/child",),
    )
    task = PortalTask(
        task_id="UIR-001",
        title="Define the UI/UX IR contract",
        status="pending",
        completion="manual",
        priority="P0",
        track="architecture",
        outputs=["external/child/docs/contract.md"],
        validation=["python -m pytest"],
    )
    sync_result = daemon._synchronize_prior_attempt_seed_submodules(
        workspace,
        task=task,
        branch_name=branch_name,
        baseline_ref=root_baseline,
        seed_reason="checked_out_prior_tree",
        seed_ref=prior_root,
    )

    assert sync_result["synchronized_count"] == 1
    assert _git(child_workspace, "rev-parse", "HEAD") == seeded_child
    assert _git(child_workspace, "branch", "--show-current") == child_branch
    (child_workspace / "docs" / "contract.md").write_text(
        "corrected fallback output\n",
        encoding="utf-8",
    )
    proposal_validation = daemon._validate_implementation_patch(
        workspace,
        task,
        baseline_ref=root_baseline,
    )
    assert proposal_validation.accepted is True
    assert proposal_validation.proposal.changed_paths == (
        "external/child/docs/contract.md",
    )
    assert "Subproject commit" not in proposal_validation.proposal.patch_text


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
