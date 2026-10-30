"""Real native candidate validation/merge and continuation refusal boundaries."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_continuation import (
    CandidateContinuationAdmission, build_candidate_request, candidate_manifest,
    reconcile_candidate,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon, PortalImplementationDaemon, PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge


def git(root, *args):
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def seed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Native fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "README.md").write_text("Native continuation fixture\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "Fixture baseline")
    baseline = git(repo, "rev-parse", "HEAD")
    branch = "implementation/wwl-001-retained-attempt-1"
    worktree = tmp_path / "worktrees" / "candidate"
    git(repo, "worktree", "add", "-b", branch, str(worktree), baseline)
    (worktree / "feature.py").write_text("VALUE = 1\n")
    git(worktree, "add", "feature.py")
    git(worktree, "commit", "-m", "Preserve candidate")
    return repo, worktree, branch, baseline, git(worktree, "rev-parse", "HEAD")


def test_actual_public_native_reconciliation_validates_merges_without_paid_attempt(tmp_path, monkeypatch):
    """No validator, proposal, Git, merge or provider method is substituted."""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path / "agent-home"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    repo, candidate, branch, baseline, commit = seed(tmp_path)
    todo = repo / "todo.md"
    todo.write_text("# Tasks\n\n## WWL-001 Retained fixture\n\n- Status: todo\n"
        "- Completion: auto\n- Outputs: feature.py\n"
        "- Validation: python -m py_compile feature.py\n"
        "- Acceptance: The real validator compiles the retained candidate.\n")
    # Commit the board on main only; source candidate identity remains fixed.
    git(repo, "add", "todo.md"); git(repo, "commit", "-m", "Declare exact task")
    state = tmp_path / "native"
    portal = PortalImplementationDaemon(todo_path=todo, state_path=state / "state.json",
        strategy_path=state / "strategy.json", events_path=state / "events.jsonl",
        repo_root=repo, task_header_prefix="## WWL-", worktree_root=tmp_path / "worktrees",
        merge_target_branch="main", worktree_submodule_paths=[], implement=False)
    before = PortalTaskState.load(portal.state_path)
    try:
        task = portal._load_tasks()[0]
        result = portal.reconcile_validated_worktree_candidate(worktree_path=candidate,
            branch_name=branch, task=task, baseline_ref=baseline, candidate_commit=commit,
            recovery_key="authored-isolated-native-proof")
        assert result["returncode"] == 0
        assert result["provider_dispatched"] is False and result["attempt_consumed"] is False
        assert result["validation_result"]["proposal_gate"]["accepted"] is True
        assert result["validation_result"]["validation_dag_receipt"]["passed"] is True
        assert result["merge_result"]["merged"] is True
        assert result["merge_result"]["integration_commit_proof"]
        assert result["board_completion"] == {"complete": True, "pending_merge": False, "reason": "merged_into_target"}
        assert (repo / "feature.py").read_text() == "VALUE = 1\n"
        after = PortalTaskState.load(portal.state_path)
        assert before.implementation_attempts == after.implementation_attempts
        assert before.implementation_attempts_by_cid == after.implementation_attempts_by_cid
        events = [json.loads(line) for line in portal.events_path.read_text().splitlines()]
        assert any(e["type"] == "merge_finished" and e["merged"] is True for e in events)
        terminal = next(e for e in reversed(events) if e["type"] == "implementation_finished")
        assert terminal["board_completion"] == result["board_completion"]
        identity = portal._identity_for_task(task)
        assert terminal["canonical_task_cid"] == identity.canonical_task_cid
        assert terminal["canonical_task_key"] == identity.canonical_task_key
        assert terminal["board_namespace"] == identity.board_namespace
        assert not any(e["type"] == "implementation_started" for e in events)
    finally:
        portal.close_event_runtime()


@pytest.fixture
def native_continuation(tmp_path, monkeypatch, request):
    """Real DB leases; only the predecessor is explicitly synthetic/no-provider."""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path / "agent-home"))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))
    repo, candidate, branch, baseline, commit = seed(tmp_path)
    now = [int(time.time() * 1000)]
    native = DatabaseImplementationDaemon(database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb", execution_path=tmp_path / "execution.duckdb",
        state_dir=tmp_path / "database-native", state_prefix="wwl", owner_session_id="fixture-continuation",
        authority_mode="embedded_exclusive", lease_ms=5000, clock_ms=lambda: now[0], max_task_attempts=3).open()
    native.materialize_population({"repository_tree_id": baseline,
        "objectives": [{"objective_id": "fixture-objective", "goal_cid": "fixture-goal", "title": "Fixture"}],
        "tasks": [{"task_cid": "sha256:" + hashlib.sha256(b"fixture-task").hexdigest(),
            "task_id": "WWL-001", "goal_cid": "fixture-goal", "status": "ready", "ordinal": 1,
            "title": "Retained fixture", "outputs": [{"path": "feature.py"}],
            "validation": ["python -m py_compile feature.py"],
            "acceptance": "The real validator compiles the retained candidate.", "completion": "auto"}]})
    (tmp_path / "database-native").mkdir(exist_ok=True)
    bridge = DatabasePortalExecutionBridge(task_source=native.task_source,
        attempt_root=tmp_path / "database-native" / "attempts", portal_factory=lambda *_: None,
        repo_root=repo, board_namespace="fixture-continuation", merge_target_branch="main",
        prior_attempt_authority=native.authorize_superseded_portal_attempt_binding)
    prior = native.claim_next()
    assert prior is not None
    old_paths, old_binding = bridge._ensure_attempt_projection(prior, native.task_source.get(prior.task_cid))
    # The existing public synthetic-only expiry path proves failed predecessor
    # custody, without fabricating any provider terminal or completion receipt.
    now[0] += 6000
    native.reconcile_expired_running_attempts()
    current = native.claim_next()
    assert current is not None and current.attempt_number == 2
    paths, binding = bridge._ensure_attempt_projection(current, native.task_source.get(current.task_cid))
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import bind_task_state_control_plane
    bind_task_state_control_plane(paths.state, str(paths.root / "portal-state.duckdb"))
    execution_pool = tmp_path / "worktrees"
    if getattr(request, "param", None) == "routed":
        execution_pool /= "repository-pool"
        execution_pool.mkdir()
    portal = PortalImplementationDaemon(todo_path=paths.task_projection, state_path=paths.state,
        strategy_path=paths.strategy, events_path=paths.events, repo_root=repo,
        task_header_prefix="## WWL-", board_namespace="fixture-continuation",
        worktree_root=execution_pool, merge_target_branch="main", worktree_submodule_paths=[], implement=False)
    portal.bind_database_attempt_authority(task_id=current.task_alias, database_task_cid=current.task_cid,
        database_attempt_id=current.attempt_id, database_claim_id=current.claim_id,
        database_attempt_number=current.attempt_number, database_binding_id=binding["binding_id"])
    task = portal._load_tasks()[0]
    request = build_candidate_request(run_id="fixture-continuation", task=task,
        predecessor_binding=old_binding, predecessor_attempt_number=prior.attempt_number,
        repository_root=repo, worktree_path=candidate, branch_name=branch,
        baseline_commit=baseline, candidate_commit=commit)
    admission = CandidateContinuationAdmission(request, old_binding, native.authorize_superseded_portal_attempt_binding)
    try:
        yield portal, current, paths, binding, admission, native
    finally:
        portal.close_event_runtime()
        native.close()


def test_helper_uses_real_native_fence_fresh_workspace_and_merge(native_continuation):
    portal, attempt, paths, binding, admission, native = native_continuation
    result = reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths,
        binding=binding, admission=admission)
    assert result["provider_dispatched"] is False
    assert result["implementation_result"]["merge_result"]["merged"] is True
    assert result["implementation_result"]["validation_result"]["passed"] is True
    assert result["implementation_result"]["merge_result"]["worktree_lifecycle_handoff"]["finalized"] is True
    assert Path(admission.request["worktree_path"]).is_dir()
    assert git(Path(admission.request["worktree_path"]), "rev-parse", "HEAD") == admission.request["candidate_commit"]
    assert result["continuation_workspace"] != admission.request["worktree_path"]
    assert re.fullmatch(r"implementation/wwl-[0-9]{3}-[0-9a-f]+-attempt-[0-9]+-[0-9]+", result["continuation_branch"])
    assert DatabasePortalExecutionBridge._has_completion_event(paths, "WWL-001")
    assert native.get_attempt(admission.request["predecessor_attempt_id"]).status == "failed"
    # The bridge/native completion CAS is deliberately outside this helper.
    assert native.task_source.get(attempt.task_cid).status == "in_progress"
    assert native.get_attempt(attempt.attempt_id).attempt_number == 2


@pytest.mark.parametrize("native_continuation", ["routed"], indirect=True)
def test_read_only_source_can_precede_repository_routing_but_validation_uses_its_pool(native_continuation):
    portal, attempt, paths, binding, admission, native = native_continuation
    source = Path(admission.request["worktree_path"])
    assert not source.is_relative_to(portal.worktree_root)
    result = reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths,
        binding=binding, admission=admission)
    assert result["provider_dispatched"] is False
    assert result["implementation_result"]["validation_result"]["passed"] is True
    assert result["implementation_result"]["merge_result"]["merged"] is True
    assert Path(result["continuation_workspace"]).is_relative_to(portal.worktree_root)
    assert git(source, "rev-parse", "HEAD") == admission.request["candidate_commit"]
    assert native.get_attempt(attempt.attempt_id).attempt_number == 2


def test_helper_refuses_fake_native_authority(native_continuation):
    portal, attempt, paths, binding, admission, _native = native_continuation
    with pytest.raises(ValueError, match="genuine_native_attempt_authority_required"):
        reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths, binding=binding,
            admission=replace(admission, prior_attempt_authority=lambda *_: {"authorized": True}))
    assert not paths.events.exists()


def test_helper_refuses_changed_committed_candidate(native_continuation):
    portal, attempt, paths, binding, admission, _native = native_continuation
    candidate = Path(admission.request["worktree_path"])
    (candidate / "feature.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="candidate_worktree_not_clean"):
        reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths, binding=binding, admission=admission)
    assert not paths.events.exists()


def test_manifest_refuses_symlink_output(tmp_path):
    _repo, candidate, _branch, baseline, _commit = seed(tmp_path)
    (candidate / "feature.py").unlink()
    (candidate / "feature.py").symlink_to("README.md")
    git(candidate, "add", "feature.py"); git(candidate, "commit", "-m", "Authored invalid symlink")
    with pytest.raises(ValueError, match="candidate_requires_root_regular_files"):
        candidate_manifest(candidate, baseline, git(candidate, "rev-parse", "HEAD"))


def test_helper_refuses_existing_fresh_workspace(native_continuation):
    portal, attempt, paths, binding, admission, _native = native_continuation
    fresh_id = hashlib.sha256((attempt.attempt_id + admission.request["receipt_id"]).encode()).hexdigest()[:24]
    (portal.worktree_root / ("workspace_continuation_" + fresh_id)).mkdir()
    with pytest.raises(ValueError, match="candidate_fresh_workspace_already_exists"):
        reconcile_candidate(portal_daemon=portal, attempt=attempt, paths=paths, binding=binding, admission=admission)
    assert not paths.events.exists()
