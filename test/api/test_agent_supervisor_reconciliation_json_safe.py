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


def _seed_candidate(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "# Tasks\n\n"
        "## ACCEL-010K Recover an orphaned implementation candidate\n\n"
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
    branch_name = "implementation/accel-010k-json-safe-attempt-2-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "feature")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    worktree_path = tmp_path / "candidate"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        worktree_root=tmp_path / "worktrees",
        merge_target_branch="main",
        worktree_submodule_paths=[],
    )
    return {
        "daemon": daemon,
        "task": daemon._load_tasks()[0],
        "baseline": baseline,
        "branch_name": branch_name,
        "candidate": candidate,
        "worktree_path": worktree_path,
    }


@pytest.mark.parametrize("mode", ["queued", "rescue_exception"])
def test_reconciliation_detaches_live_proposal_before_json_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    fixture = _seed_candidate(tmp_path)
    daemon = fixture["daemon"]
    task = fixture["task"]
    captured: dict[str, object] = {}

    def validation_with_live_proposal(
        _workspace_path,
        _task,
        _log_path,
        *,
        proposal_validation=None,
        **_kwargs,
    ):
        assert isinstance(proposal_validation, ProposalValidationResult)
        captured["live_proposal"] = proposal_validation
        passed = mode == "queued"
        return {
            "attempted": True,
            "passed": passed,
            "returncode": 0 if passed else 1,
            "results": [],
            "reason": (
                "declared_validation_passed"
                if passed
                else "declared_validation_failed"
            ),
            "proposal_gate": daemon._compact_proposal_validation(
                proposal_validation
            ),
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

    if mode == "rescue_exception":

        def raise_during_rescue(**kwargs):
            assert kwargs["validation_result"]["proposal_validation"] is (
                captured["live_proposal"]
            )
            raise RuntimeError("injected reconciliation rescue failure")

        monkeypatch.setattr(
            daemon,
            "_automatic_implementation_rescue",
            raise_during_rescue,
        )
    else:
        monkeypatch.setattr(
            daemon,
            "_automatic_implementation_rescue",
            lambda **_kwargs: pytest.fail(
                "passing validation unexpectedly invoked rescue"
            ),
        )

    def assert_json_safe_enqueue(**kwargs):
        validation_result = kwargs["validation_result"]
        json.dumps(validation_result, sort_keys=True)
        captured["queued_validation"] = dict(validation_result)
        return {"merged": False, "queued": True}

    monkeypatch.setattr(
        daemon,
        "_enqueue_validated_worktree",
        assert_json_safe_enqueue,
    )

    result = daemon.reconcile_validated_worktree_candidate(
        worktree_path=fixture["worktree_path"],
        branch_name=fixture["branch_name"],
        task=task,
        baseline_ref=fixture["baseline"],
        candidate_commit=fixture["candidate"],
        recovery_key=f"json-safe-{mode}",
    )

    live_proposal = captured["live_proposal"]
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(live_proposal)
    json.dumps(result, sort_keys=True)
    persisted = result["validation_result"]
    assert "proposal_validation" not in persisted
    assert persisted["proposal_gate"]["accepted"] is True
    assert persisted["proposal_gate"]["proposal_id"] == (
        live_proposal.proposal.proposal_id
    )
    if mode == "queued":
        assert result["merge_result"]["queued"] is True
        assert captured["queued_validation"] == persisted
    else:
        assert "queued_validation" not in captured
        assert persisted["reason"] == "reconciliation_validation_exception"
        assert persisted["error_type"] == "RuntimeError"

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
        in {
            "worktree_reconciliation_candidate_queued",
            "worktree_reconciliation_validation_finished",
        }
    )
    assert terminal["validation_result"] == persisted
