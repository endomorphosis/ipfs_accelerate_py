from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ProposalValidationResult,
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


def _seed_candidate(tmp_path: Path) -> dict[str, Path | str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "# Tasks\n\n"
        "## ACCEL-010J Recover an orphaned implementation candidate\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P0\n"
        "- Track: ops\n"
        "- Outputs: feature.py\n"
        "- Validation: python -m py_compile feature.py\n"
        "- Acceptance: The recovered candidate is independently validated.\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md", "todo.md")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")
    branch_name = "implementation/accel-010j-json-safe-attempt-2-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "feature")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    worktree_path = tmp_path / "candidate"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    return {
        "repo": repo,
        "todo_path": todo_path,
        "baseline": baseline,
        "branch_name": branch_name,
        "candidate": candidate,
        "worktree_path": worktree_path,
    }


def test_reconciliation_projects_live_proposal_before_json_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _seed_candidate(tmp_path)
    state_dir = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=Path(fixture["todo_path"]),
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=Path(fixture["repo"]),
        task_header_prefix="## ACCEL-",
        worktree_root=tmp_path / "worktrees",
        merge_target_branch="main",
        worktree_submodule_paths=[],
    )
    task = daemon._load_tasks()[0]
    captured: dict[str, object] = {}

    def validation_with_live_proposal(
        _workspace_path,
        _task,
        _log_path,
        *,
        proposal_validation=None,
        **_kwargs,
    ):
        assert proposal_validation is not None
        captured["proposal_validation"] = proposal_validation
        return {
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "results": [],
            "reason": "declared_validation_failed",
            "proposal_gate": daemon._compact_proposal_validation(
                proposal_validation
            ),
            # In-process candidate binding may retain this typed object.
            "proposal_validation": proposal_validation,
        }

    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        validation_with_live_proposal,
    )
    monkeypatch.setattr(
        daemon,
        "_apply_implementation_failure_review",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )
    monkeypatch.setattr(
        daemon,
        "_automatic_implementation_rescue",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )

    result = daemon.reconcile_validated_worktree_candidate(
        worktree_path=Path(fixture["worktree_path"]),
        branch_name=str(fixture["branch_name"]),
        task=task,
        baseline_ref=str(fixture["baseline"]),
        candidate_commit=str(fixture["candidate"]),
        recovery_key="json-safe-proposal-validation",
    )

    typed = captured["proposal_validation"]
    assert isinstance(typed, ProposalValidationResult)
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(typed)
    json.dumps(result, sort_keys=True)
    persisted = result["validation_result"]
    assert "proposal_validation" not in persisted
    assert persisted["proposal_gate"]["accepted"] is True
    assert persisted["proposal_gate"]["proposal_id"] == (
        typed.proposal.proposal_id
    )
    assert persisted["proposal_gate"]["receipt_id"] == (
        typed.receipt.receipt_id
    )
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    terminal = next(
        event
        for event in reversed(events)
        if event.get("type")
        == "worktree_reconciliation_validation_finished"
    )
    assert terminal["validation_result"] == persisted
