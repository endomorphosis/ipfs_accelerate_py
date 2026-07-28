"""Tests for prior-attempt worktree seeding and board completion decisions."""

from __future__ import annotations

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
