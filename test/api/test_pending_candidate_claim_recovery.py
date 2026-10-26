"""Real Git/lifecycle preservation with explicitly synthetic proposal history."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as recovery
from test.api.test_pending_task_claim_recovery import pending

pytestmark = pytest.mark.parametrize("pending", ["candidate"], indirect=True)


def test_native_preserves_candidate_retires_exact_fence_releases_claim_and_replays(pending):
    daemon, task, workspace, claim = pending
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    board = daemon.todo_path.read_bytes()
    code = (workspace / "src/example.py").read_bytes()
    with pytest.raises(ValueError, match="controlled_terminal_transition_unproved"):
        recovery.review(daemon, task_id=task.task_id)
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    assert request["schema"] == recovery.CANDIDATE_SCHEMA
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["reconciled"] and not result["blocked"], result
    assert result["reason"] == "preserved_candidate_pending_task_claim_released"
    assert all(result[key] is False for key in recovery._FALSE)
    assert result["counters_preserved"] and result["task_status_unchanged"]
    assert not claim.exists() and not workspace.exists()
    assert daemon.worktree_lifecycle.load_workspace(workspace) is None
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert daemon.todo_path.read_bytes() == board
    candidate = result["preserved_candidate"]
    assert subprocess.check_output(["git", "show", candidate["commit"] + ":src/example.py"], cwd=daemon.repo_root) == code
    assert Path(result["receipt_path"]).stat().st_mode & 0o777 == 0o600
    again_request = recovery.review_recovery(daemon, task_id=task.task_id)
    assert again_request == request
    again = recovery.reconcile_recovery(daemon, request=again_request)
    assert again == result
    assert not any(event["type"] in recovery._OUTCOMES for event in daemon._iter_events())


def test_proposal_identity_rebuild_matches_real_native_gate(pending):
    daemon, task, workspace, _ = pending
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    actual = daemon._validate_implementation_patch(workspace, task, baseline_ref=baseline, record_event=False)
    assert actual.proposal.proposal_id == recovery._candidate_proposal_id(daemon, task, workspace, baseline)


@pytest.mark.parametrize("damage", ["unstaged", "outside", "after_accept", "branch", "rejected"])
def test_unadmitted_source_never_reaches_native_preservation(pending, monkeypatch, damage):
    daemon, task, workspace, claim = pending
    source = workspace / "src/example.py"
    if damage in {"unstaged", "after_accept"}:
        source.write_text("VALUE = 35\n")
        if damage == "after_accept":
            subprocess.run(["git", "add", "--", "src/example.py"], cwd=workspace, check=True)
    elif damage == "outside": (workspace / "foreign.py").write_text("VALUE = 1\n")
    elif damage == "branch": subprocess.run(["git", "branch", "-m", "foreign-branch"], cwd=workspace, check=True)
    else: daemon._record_event("implementation_proposal_rejected", {"task_id": task.task_id, "accepted": False})
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: pytest.fail("unadmitted preservation"))
    with pytest.raises(ValueError): recovery.review_recovery(daemon, task_id=task.task_id)
    assert claim.exists() and source.exists() and asdict(native.PortalTaskState.load(daemon.state_path)) == before


@pytest.mark.parametrize("stage", ["prepared", "preserved", "retired", "released"])
def test_native_checkpoint_and_claim_crashes_replay_without_new_attempt_or_source_reset(pending, monkeypatch, stage):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    actual = daemon._write_private_task_claim_release_receipt
    def interrupted(path, value):
        if value.get("phase") == stage: raise OSError("synthetic durable publication interruption")
        return actual(path, value)
    monkeypatch.setattr(daemon, "_write_private_task_claim_release_receipt", interrupted)
    first = recovery.reconcile_recovery(daemon, request=request)
    assert first["blocked"] and not first["reconciled"], first
    monkeypatch.setattr(daemon, "_write_private_task_claim_release_receipt", actual)
    fresh = recovery.review_recovery(daemon, task_id=task.task_id)
    assert fresh == request
    resumed = recovery.reconcile_recovery(daemon, request=fresh)
    assert resumed["reconciled"] and not claim.exists() and not workspace.exists(), resumed
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert len([e for e in daemon._iter_events() if e["type"] == "implementation_pending_candidate_preserved"]) == 1


@pytest.mark.parametrize("damage", ["bytes", "branch", "claim", "protected", "state", "queue", "life"])
def test_prepared_receipt_cannot_commit_or_cleanup_changed_custody(pending, monkeypatch, damage):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    actual = daemon._preserve_interrupted_worktree
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic before commit")))
    assert recovery.reconcile_recovery(daemon, request=request)["blocked"]
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", actual)
    if damage == "bytes":
        (workspace / "src/example.py").write_text("VALUE = 99\n")
        subprocess.run(["git", "add", "--", "src/example.py"], cwd=workspace, check=True)
    elif damage == "branch": subprocess.run(["git", "branch", "-m", "reused-branch"], cwd=workspace, check=True)
    elif damage == "claim":
        value = json.loads(claim.read_text()); value["lease_id"] = "replacement"; claim.write_text(json.dumps(value))
    elif damage == "protected": daemon._implementation_protected_incident_path().write_text("{}")
    elif damage == "state":
        state = native.PortalTaskState.load(daemon.state_path); state.implementation_attempts[task.task_id] = 2; state.save(daemon.state_path)
    elif damage == "queue":
        daemon.merge_queue.enqueue(branch_name="unrelated", task_id="OTHER", commit_sha=request["base"]["baseline_ref"],
            target_repository_id=daemon.merge_target_repository_id, target_branch=daemon.resolved_merge_target_branch)
    else:
        path = daemon.worktree_lifecycle.workspace_path_for(workspace)
        value = json.loads(path.read_text()); value["fence"] += 1; path.write_text(json.dumps(value))
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    saved = (workspace / "src/example.py").read_bytes()
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: pytest.fail("changed custody preservation"))
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["blocked"] and claim.exists() and (workspace / "src/example.py").read_bytes() == saved, result
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before


def test_existing_foreign_rescue_ref_is_never_overwritten(pending):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    operation = recovery._sha(recovery._bytes(request))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=workspace, text=True).strip()
    rescue = daemon._interrupted_worktree_rescue_branch_name(branch, "pending-candidate-" + operation[:20])
    subprocess.run(["git", "branch", rescue, request["base"]["baseline_ref"]], cwd=daemon.repo_root, check=True)
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["reason"] == "candidate_rescue_ref_already_exists" and claim.exists(), result
    assert subprocess.check_output(["git", "rev-parse", rescue], cwd=daemon.repo_root, text=True).strip() == request["base"]["baseline_ref"]


def test_commit_before_native_checkpoint_event_replays_exact_preserved_bytes(pending, monkeypatch):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    actual = daemon._record_event
    def interrupted(kind, value):
        if kind == "implementation_pending_candidate_preserved": raise OSError("synthetic checkpoint publication interruption")
        return actual(kind, value)
    monkeypatch.setattr(daemon, "_record_event", interrupted)
    first = recovery.reconcile_recovery(daemon, request=request)
    assert first["blocked"] and claim.exists() and workspace.exists(), first
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() != request["base"]["baseline_ref"]
    monkeypatch.setattr(daemon, "_record_event", actual)
    second = recovery.reconcile_recovery(daemon, request=recovery.review_recovery(daemon, task_id=task.task_id))
    assert second["reconciled"] and not claim.exists(), second


def test_released_checkpoint_refuses_replacement_claim_and_changed_source(pending):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    raw = claim.read_bytes()
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["reconciled"], result
    claim.write_bytes(raw); claim.chmod(0o600)
    denied = recovery.reconcile_recovery(daemon, request=request)
    assert denied["blocked"] and claim.read_bytes() == raw


def test_candidate_with_held_real_queue_settlement_guard_does_not_reacquire_writer(pending):
    daemon, task, _, claim = pending
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import hold_merge_queue_settlement
    with hold_merge_queue_settlement(daemon.merge_queue.queue_dir, target_repository_id=daemon.merge_target_repository_id,
            target_branch=daemon.resolved_merge_target_branch, lock_timeout_seconds=0.01):
        result = recovery.reconcile_recovery(daemon, request=recovery.review_recovery(daemon, task_id=task.task_id))
        assert result["reconciled"] and not claim.exists(), result


@pytest.fixture
def dependency_marker():
    """Ask the shared real Git fixture to seed a pinned proper dependency."""


@pytest.mark.parametrize("damage", ["tracked", "untracked", "revision"])
def test_dirty_initialized_dependency_is_never_committed_or_discarded(pending, dependency_marker, monkeypatch, damage):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    assert request["candidate"]["dependencies"]
    child = workspace / "external/runtime"
    target = child / ("local.py" if damage == "untracked" else "runtime.py")
    target.write_text("DEPENDENCY = 99\n")
    if damage == "revision":
        subprocess.run(["git", "add", "."], cwd=child, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "local child change"], cwd=child, check=True)
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: pytest.fail("dirty dependency preservation"))
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["blocked"] and claim.exists() and target.read_text() == "DEPENDENCY = 99\n", result
    with pytest.raises(ValueError, match="candidate_dependency"):
        recovery.review_recovery(daemon, task_id=task.task_id)


def test_handoff_replays_after_normal_native_implementation_lease_cleanup(pending):
    daemon, task, _, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    first = recovery.reconcile_recovery(daemon, request=request)
    assert first["reconciled"] and not claim.exists(), first
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    cleanup = daemon.reconcile_quiesced_active_attempt()
    assert cleanup["reconciled"] and not cleanup["blocked"], cleanup
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    again = recovery.reconcile_recovery(daemon, request=recovery.review_recovery(daemon, task_id=task.task_id))
    assert again == first


def test_prepared_review_itself_refuses_changed_bytes_before_any_mutation(pending, monkeypatch):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic pause")))
    assert recovery.reconcile_recovery(daemon, request=request)["blocked"]
    (workspace / "src/example.py").write_text("VALUE = 999\n")
    subprocess.run(["git", "add", "--", "src/example.py"], cwd=workspace, check=True)
    with pytest.raises(ValueError): recovery.review_recovery(daemon, task_id=task.task_id)
    assert claim.exists() and (workspace / "src/example.py").read_text() == "VALUE = 999\n"


@pytest.mark.parametrize("after_checkpoint", [False, True])
def test_native_mutation_runtime_failure_is_typed_blocked_and_exactly_replayable(pending, monkeypatch, after_checkpoint):
    daemon, task, workspace, claim = pending
    request = recovery.review_recovery(daemon, task_id=task.task_id)
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    claim_bytes = claim.read_bytes()
    actual = daemon._preserve_interrupted_worktree
    def interrupted(*args, **kwargs):
        if after_checkpoint: actual(*args, **kwargs)
        raise RuntimeError("private diagnostic must not enter public result")
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", interrupted)
    result = recovery.reconcile_recovery(daemon, request=request)
    assert result["blocked"] and result["reason"] == "RuntimeError" and not result["reconciled"]
    assert all(result[key] is False for key in recovery._FALSE)
    assert claim.read_bytes() == claim_bytes and asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert "private diagnostic" not in json.dumps(result)
    monkeypatch.setattr(daemon, "_preserve_interrupted_worktree", actual)
    resumed = recovery.reconcile_recovery(daemon, request=recovery.review_recovery(daemon, task_id=task.task_id))
    assert resumed["reconciled"] and not claim.exists(), resumed
