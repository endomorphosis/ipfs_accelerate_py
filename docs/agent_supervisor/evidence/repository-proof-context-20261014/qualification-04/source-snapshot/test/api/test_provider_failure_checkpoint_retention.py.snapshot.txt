"""Real failed child edits remain recoverable without granting retry authority."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTask, PortalTaskState,
)


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repository, text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def failed_child_case(tmp_path: Path, *, pooled=False):
    """Author a tiny repository and a real child, without any model adapter."""
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Checkpoint fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "src").mkdir()
    (repository / "src/value.py").write_text("VALUE = 1\n")
    git(repository, "add", "src/value.py")
    git(repository, "commit", "-m", "fixture baseline")
    baseline = git(repository, "rev-parse", "HEAD")
    child = tmp_path / "failed_child.py"
    child.write_text("from pathlib import Path\n"
                     "Path('src/value.py').write_text('VALUE = 2\\n')\n"
                     "raise SystemExit(17)\n")
    daemon = PortalImplementationDaemon(
        todo_path=repository / "tasks.todo.md", repo_root=repository,
        state_path=tmp_path / "state/task-state.json",
        strategy_path=tmp_path / "state/strategy.json",
        events_path=tmp_path / "state/events.jsonl",
        implement=True, implementation_command=shlex.join([sys.executable, str(child)]),
        use_ephemeral_worktree=True, worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=pooled,
    )
    task = PortalTask(task_id="RET-001", title="Revise the declared value",
        status="ready", completion="manual", priority="P1", track="test",
        outputs=["src/value.py"], acceptance="The declared value is revised and checked.",
        validation=["python -c 'from src.value import VALUE; assert VALUE == 2'"],
        metadata={"requires provider": "true"})
    state = PortalTaskState()
    return daemon, task, state, baseline


def run_failed_child(daemon, task, state):
    prompt = daemon._build_implementation_prompt(task, attempt=1)
    daemon._persist_implementation_context_receipt(task, attempt=1)
    return daemon._run_implementation_in_ephemeral_worktree(
        task=task, state=state, attempt=1, started_at="2026-10-14T00:00:00+00:00",
        log_path=daemon.state_path.parent / "failed-child.log", prompt=prompt)


@pytest.mark.parametrize("pooled", [False, True])
def test_nonzero_child_preserves_native_commit_without_completion_or_seed_authority(tmp_path, pooled):
    daemon, task, state, baseline = failed_child_case(tmp_path, pooled=pooled)
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] == 17
    assert result["provider_dispatched"] is True
    assert result["attempt_consumed"] is True
    assert result["validation_result"]["attempted"] is False
    assert result["validation_result"]["passed"] is False
    assert result["validation_result"]["reason"] == "implementation_provider_failed"
    assert result["board_completion"]["complete"] is False
    assert result["merge_result"]["merged"] is False
    retained = result["failed_preservation_result"]
    assert retained["preserved"] is True
    assert retained["provider_failure"]["returncode"] == 17
    assert retained["provider_failure"]["completion_authority"] is False
    assert retained["provider_failure"]["retry_authority"] is False
    commit = retained["preserved_commit"]
    assert git(daemon.repo_root, "rev-parse", retained["rescue_branch"]) == commit
    assert git(daemon.repo_root, "show", commit + ":src/value.py") == "VALUE = 2"
    assert git(daemon.repo_root, "rev-parse", "HEAD") == baseline
    assert state.last_implementation_commit == commit
    assert task.task_id not in state.completed_task_ids
    events = [json.loads(row) for row in daemon.events_path.read_text().splitlines()]
    assert any(event["type"] == "provider_failed_worktree_preserved" for event in events)
    assert not any(event["type"] == "implementation_proposal_validated"
                   and event.get("accepted") is True for event in events)
    target = tmp_path / "fresh"
    git(daemon.repo_root, "worktree", "add", "-b", "fresh", str(target), baseline)
    seed = daemon._apply_prior_attempt_seed(target, task=task,
        seed_plan={"reuse_prior_attempt": True, "seed_ref": commit}, baseline_ref=baseline)
    assert seed["applied"] is False
    assert seed["reason"] == "prior_seed_accepted_proposal_missing"
    assert (target / "src/value.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("failure", ["commit_refused", "commit_raised", "ref_raised", "ref_stale", "event_raised"])
def test_preservation_failure_retains_real_edits_and_lifecycle(tmp_path, monkeypatch, failure, pooled):
    daemon, task, state, baseline = failed_child_case(tmp_path, pooled=pooled)
    if failure == "commit_refused":
        monkeypatch.setattr(daemon, "_commit_worktree_changes", lambda *_a, **_k:
                            {"committed": False, "reason": "fixture_commit_refused"})
    elif failure == "commit_raised":
        def refused(*_a, **_k):
            raise OSError("fixture_commit_error")
        monkeypatch.setattr(daemon, "_commit_worktree_changes", refused)
    elif failure in {"ref_raised", "ref_stale"}:
        original = daemon._run_git
        def refused(args, **kwargs):
            if args[:2] == ["branch", "-f"]:
                if failure == "ref_stale":
                    original(["branch", "-f", args[2], baseline], **kwargs)
                raise OSError("fixture_ref_error")
            return original(args, **kwargs)
        monkeypatch.setattr(daemon, "_run_git", refused)
    else:
        original = daemon._record_event
        def refused(name, *args, **kwargs):
            if name.startswith("provider_failed_worktree_"):
                raise OSError("fixture_event_error")
            return original(name, *args, **kwargs)
        monkeypatch.setattr(daemon, "_record_event", refused)
    monkeypatch.setattr(daemon, "_cleanup_failed_setup_worktree", lambda *_a, **_k:
                        pytest.fail("preservation must not enter setup cleanup"))
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", lambda *_a, **_k:
                        pytest.fail("uncheckpointed candidate must not be cleaned"))
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] == 17
    assert result["board_completion"]["complete"] is False
    assert result["failed_preservation_result"]["preserved"] is False
    assert result["failed_preservation_result"]["retained"] is True
    assert result["cleanup_result"]["cleaned"] is False
    assert result["lifecycle_finalize"]["finalized"] is False
    workspace = Path(result["worktree_path"])
    assert (workspace / "src/value.py").read_text() == "VALUE = 2\n"
    assert not daemon.worktree_lifecycle.load_workspace(workspace).is_terminal
    assert state.last_implementation_commit == ""
    assert git(daemon.repo_root, "rev-parse", "HEAD") == baseline


@pytest.mark.parametrize("pooled", [False, True])
def test_failed_child_without_changes_keeps_existing_release_behavior(tmp_path, monkeypatch, pooled):
    daemon, task, state, _baseline = failed_child_case(tmp_path, pooled=pooled)
    (tmp_path / "failed_child.py").write_text("raise SystemExit(17)\n")
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *_a, **_k:
                        pytest.fail("a clean baseline needs no checkpoint"))
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] == 17
    assert result["failed_preservation_result"] == {}
    assert result["cleanup_result"]["reason"] == (
        "failed_implementation_pool_lease_released" if pooled else "failed_implementation_worktree_preserved")
    assert result["lifecycle_finalize"]["finalized"] is True
    assert state.last_implementation_commit == ""


def test_unknown_database_child_custody_keeps_workspace_and_claim_without_checkpoint(tmp_path, monkeypatch):
    from test.api.test_native_provider_failure_settlement import _native_chain
    database, _bridge, portals = _native_chain(tmp_path, monkeypatch, partial_edit=True)
    monkeypatch.setattr(PortalImplementationDaemon, "_database_provider_children_quiesced",
                        staticmethod(lambda _custody: False))
    monkeypatch.setattr(PortalImplementationDaemon, "_preserve_interrupted_worktree", lambda *_a, **_k:
                        pytest.fail("unknown child custody must not commit"))
    try:
        result = database.run_once()
        attempt = database.get_attempt(result["attempt_id"])
        assert database.coordinator.get_task_claim(attempt.claim_id).state.value != "released"
        callback = database.provider_invocation_recorded(attempt.attempt_id,
            idempotency_key="provider:" + attempt.attempt_id)
        assert callback["callback_state"] == "started_outcome_unknown"
        assert len(portals) == 1
        portal = portals[0]
        events = [json.loads(row) for row in portal.events_path.read_text().splitlines()]
        finished = [event for event in events if event["type"] == "implementation_finished"][-1]
        assert finished["returncode"] != 0
        assert finished["implementation_commit"] == ""
        assert finished["cleanup_result"]["cleaned"] is False
        assert finished["lifecycle_finalize"] == {
            "finalized": False, "reason": "provider_child_custody_unverified"}
        workspace = Path(finished["worktree_path"])
        assert (workspace / "result.py").read_text() == "VALUE = 1\n"
        assert not portal.worktree_lifecycle.load_workspace(workspace).is_terminal
    finally:
        database.close()


def test_missing_declared_output_preserves_worktree_without_bypassing_commit_guard(tmp_path):
    daemon, task, state, baseline = failed_child_case(tmp_path)
    task.outputs.append("src/missing.py")
    result = run_failed_child(daemon, task, state)
    retained = result["failed_preservation_result"]
    assert result["returncode"] == 17
    assert retained["preserved"] is False
    assert retained["commit_result"]["committed"] is False
    assert retained["commit_result"]["declared_output_invariant"]["passed"] is False
    assert result["cleanup_result"]["cleaned"] is False
    assert result["lifecycle_finalize"]["finalized"] is False
    assert state.last_implementation_commit == ""
    assert (Path(result["worktree_path"]) / "src/value.py").read_text() == "VALUE = 2\n"
    assert git(daemon.repo_root, "rev-parse", "HEAD") == baseline


@pytest.mark.parametrize("pooled", [False, True])
def test_cleanup_response_loss_keeps_verified_checkpoint_and_does_not_relookup_lease(tmp_path, monkeypatch, pooled):
    daemon, task, state, baseline = failed_child_case(tmp_path, pooled=pooled)
    original_cleanup = daemon._cleanup_merged_worktree
    original_finalize = daemon._finalize_worktree_lifecycle
    cleanup_finished = False
    def cleanup_then_fail(*args, **kwargs):
        nonlocal cleanup_finished
        value = original_cleanup(*args, **kwargs)
        assert value["cleaned"] is True
        cleanup_finished = True
        raise OSError("fixture_cleanup_response_lost")
    def finalize(*args, **kwargs):
        assert not cleanup_finished, "an unknown cleanup cannot relookup a released lease"
        return original_finalize(*args, **kwargs)
    monkeypatch.setattr(daemon, "_cleanup_merged_worktree", cleanup_then_fail)
    monkeypatch.setattr(daemon, "_finalize_worktree_lifecycle", finalize)
    result = run_failed_child(daemon, task, state)
    retained = result["failed_preservation_result"]
    assert result["returncode"] == 17
    assert retained["preserved"] is True
    assert retained["cleanup_uncertain"] is True
    assert result["lifecycle_finalize"]["finalized"] is False
    assert result["cleanup_result"]["cleaned"] is False
    commit = retained["preserved_commit"]
    assert commit and state.last_implementation_commit == commit
    assert git(daemon.repo_root, "rev-parse", retained["rescue_branch"]) == commit
    assert git(daemon.repo_root, "show", commit + ":src/value.py") == "VALUE = 2"
    assert git(daemon.repo_root, "rev-parse", "HEAD") == baseline
    assert result["board_completion"]["complete"] is False
