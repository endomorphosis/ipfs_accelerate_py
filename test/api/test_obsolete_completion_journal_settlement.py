"""Retire superseded journals only after real Native successor publication.

The fixture runs an owned child through candidate validation and the actual
merge train.  Its sole producer fault recreates the historical generated Git
identity bug.  A second, explicit lost-journal fault hides that exact journal
while the ordinary idempotent queue callback publishes its successor.  It then
restores the unchanged Native journal, modelling a delayed old publisher.  No
accepted queue, callback, validation, or publication record is manufactured.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_mutation_lock_path, read_checkout_mutation_lease,
    update_checkout_mutation_lease,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native


TASK = "EX-007"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, (args, result.stderr)
    return result.stdout.strip()


def daemon_for(directory: Path):
    repo = directory / "repo"
    return native.PortalImplementationDaemon(
        todo_path=repo / "todo.md", state_path=directory / "state/tasks.json",
        strategy_path=directory / "state/strategy.json",
        events_path=directory / "state/fixture_events.jsonl", repo_root=repo,
        task_header_prefix="## EX-", implement=False,
        implementation_command="forbidden-model-provider",
        max_task_attempts=3, worktree_pool_enabled=False,
        worktree_root=directory / "worktrees", worktree_submodule_paths=[],
        merge_target_branch="main", implementation_protected_paths=("todo.md",),
    )


def _produce_bad_completion(directory: Path) -> None:
    """An actual child owns the old journal and then exits normally."""
    daemon = daemon_for(directory)
    task = daemon._load_tasks()[0]
    identity = daemon._identity_for_task(task)
    state = native.PortalTaskState(
        implementation_attempts={TASK: 3},
        implementation_attempts_by_cid={identity.canonical_task_cid: 3},
        task_identities={TASK: identity.to_dict()},
    )
    state.save(daemon.state_path)
    original_run = native.subprocess.run
    original_validate = daemon._run_validation_commands
    injected = []

    def observe_actual_active_state(*args, **kwargs):
        active = native.PortalTaskState.load(daemon.state_path)
        assert active.implementation_in_progress
        assert active.active_task_id == TASK
        (directory / "actual-producer-active-state.json").write_bytes(daemon.state_path.read_bytes())
        return original_validate(*args, **kwargs)

    daemon._run_validation_commands = observe_actual_active_state

    def historical_identity(command, *args, **kwargs):
        if (isinstance(command, list) and "commit" in command
                and "user.name=Implementation Daemon" in command):
            env = dict(kwargs.get("env") or os.environ)
            env.update(GIT_AUTHOR_NAME="Historical CI Actor",
                       GIT_AUTHOR_EMAIL="ci@example.invalid",
                       GIT_COMMITTER_NAME="Historical CI Actor",
                       GIT_COMMITTER_EMAIL="ci@example.invalid")
            kwargs["env"] = env
            injected.append(tuple(command))
        return original_run(command, *args, **kwargs)

    native.subprocess.run = historical_identity
    try:
        seed = json.loads((directory / "seed.json").read_text())
        result = daemon.reconcile_validated_worktree_candidate(
            worktree_path=directory / "worktrees/candidate",
            branch_name=seed["branch"], task=task,
            baseline_ref=seed["baseline"], candidate_commit=seed["candidate"],
            recovery_key="real-obsolete-completion-producer",
        )
    finally:
        native.subprocess.run = original_run
    assert injected, result
    assert result["provider_dispatched"] is False
    assert result["attempt_consumed"] is False
    assert result["validation_result"]["passed"] is True, result
    assert result["merge_result"]["merged"] is False, result
    assert checkout_mutation_lock_path(directory / "repo").is_file(), result
    (directory / "producer.json").write_text(json.dumps(result, sort_keys=True))


def _try_baseline_recovery(directory: Path) -> None:
    daemon = daemon_for(directory)
    # Exercise the pre-existing recovery body without the new settlement
    # admission hook.  It must still refuse the old CI-authored history.
    daemon.reconcile_obsolete_protected_completion_journal = lambda: {"required": False}
    result = daemon._recover_protected_checkout_mutation()
    assert not result.get("checkout_mutation_lease_recovered"), result
    assert checkout_mutation_lock_path(directory / "repo").is_file()
    journal = json.loads(checkout_mutation_lock_path(directory / "repo").read_text())
    proof = daemon._protected_checkout_release_proof(journal["protected_release_guard"])
    assert proof["trusted"] is False, proof
    assert proof["reason"] == "protected_release_history_untrusted", proof
    result["original_guard_release_proof"] = proof
    (directory / "baseline-recovery.json").write_text(json.dumps(result, default=str))


def _hold_actual_implementation_lease(directory: Path) -> None:
    """Publish the Native attempt-zero lease from its real owning child."""
    daemon = daemon_for(directory)
    task = daemon._load_tasks()[0]
    metadata = daemon._build_implementation_lock_metadata(task, 0, native.utc_now())
    path = daemon._implementation_lock_path()
    acquired, reason, _ = daemon._try_acquire_implementation_lock(path, metadata)
    assert acquired, reason
    (directory / "live-implementation-ready.json").write_text(json.dumps(metadata))
    try:
        assert sys.stdin.read(1) == "x"
    finally:
        assert daemon._release_implementation_lock(path, metadata)


@pytest.fixture
def successor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real failed train request followed by a real accepted idempotent retry."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "todo.md").write_text(
        "# Tasks\n\n## EX-007 Preserve exact completion authority\n\n"
        "- Status: todo\n- Completion: manual\n- Priority: P0\n- Track: ops\n"
        "- Outputs: feature.py\n- Validation: python -m py_compile feature.py\n"
        "- Acceptance: The real candidate validates before completion.\n"
    )
    (repo / "README.md").write_text("fixture\n")
    git(repo, "add", "todo.md", "README.md")
    git(repo, "commit", "-m", "seed")
    baseline = git(repo, "rev-parse", "HEAD")
    branch = "implementation/ex-007-completion-journal"
    git(repo, "checkout", "-b", branch)
    (repo / "feature.py").write_text("VALUE = 7\n")
    git(repo, "add", "feature.py")
    git(repo, "commit", "-m", "real implementation candidate")
    candidate = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "main")
    (tmp_path / "worktrees").mkdir()
    git(repo, "worktree", "add", str(tmp_path / "worktrees/candidate"), branch)
    (tmp_path / "seed.json").write_text(json.dumps(
        {"baseline": baseline, "candidate": candidate, "branch": branch}))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(native.__file__).parents[3]), env.get("PYTHONPATH", ""))))
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--produce-bad-completion", str(tmp_path)],
                           env=env, capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, (child.stdout[-3000:], child.stderr[-6000:])
    lock_path = checkout_mutation_lock_path(repo)
    old_bytes = lock_path.read_bytes()
    old = json.loads(old_bytes)
    assert old["pid"] != os.getpid()
    assert not Path("/proc", str(old["pid"])).exists()
    assert old["protected_recovery_required"] is True
    assert old["operation"] == "mark_tasks_completed"
    daemon = daemon_for(tmp_path)
    state_before = asdict(native.PortalTaskState.load(daemon.state_path))
    # The only second fault models the already-observed missing/delayed old
    # journal at callback admission.  Its actual Native bytes survive intact.
    hidden = tmp_path / "original-native-journal.json"
    lock_path.rename(hidden)
    try:
        progress = daemon._consume_one_merge_candidate()
    finally:
        assert not lock_path.exists(), "successor retained its own fence"
        hidden.rename(lock_path)
    assert progress is not None, "the actual failed request was not retried"
    request = daemon.merge_queue.get(progress["request_id"])
    assert request is not None and request.status == "completed", progress
    assert request.failure_count == 1
    assert lock_path.read_bytes() == old_bytes
    assert git(repo, "status", "--porcelain", "--", "todo.md") == ""
    assert daemon._load_tasks()[0].status == "completed"
    publications = list((tmp_path / "state/completion-publications").glob("*.json"))
    assert len(publications) == 1
    publication = json.loads(publications[0].read_text())
    assert publication["phase"] == "published"
    assert publication["queue_recorded"] is True
    assert publication["publication"]["published"] is True
    assert publication["intent_id"] != old["completion_intent"]["intent_id"]
    # Baseline proof is deliberately still untrusted: the new API must not
    # manufacture a trust approval for the historical CI commit.
    baseline_child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     "--baseline-recover", str(tmp_path)],
                                    env=env, capture_output=True, text=True, timeout=90)
    assert baseline_child.returncode == 0, (baseline_child.stdout[-3000:], baseline_child.stderr[-6000:])
    baseline_result = json.loads((tmp_path / "baseline-recovery.json").read_text())
    assert not baseline_result.get("checkout_mutation_lease_recovered"), baseline_result
    assert lock_path.exists()
    daemon._clear_checkout_mutation_context()
    # Native baseline recovery may CAS-adopt the journal, so carry its actual
    # current bytes into the new exact-lease test, never reconstruct metadata.
    current_bytes = lock_path.read_bytes()
    return {"daemon": daemon, "root": repo, "lock": lock_path,
            "old_bytes": current_bytes, "state_before": state_before,
            "publication": publications[0], "request": request,
            "directory": tmp_path, "baseline_result": baseline_result}


def _assert_preserved(case):
    assert case["lock"].read_bytes() == case["old_bytes"]
    after = asdict(native.PortalTaskState.load(case["daemon"].state_path))
    assert after["implementation_attempts"] == case["state_before"]["implementation_attempts"]
    assert after["implementation_attempts_by_cid"] == case["state_before"]["implementation_attempts_by_cid"]


def test_real_superseded_completion_journal_retires_without_completion_replay(successor, monkeypatch):
    daemon = successor["daemon"]
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    board = (successor["root"] / "todo.md").read_bytes()
    publication = successor["publication"].read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("journal settlement attempted a new callback or paid attempt")
    monkeypatch.setattr(daemon, "_mark_tasks_completed_in_todo", forbidden)
    monkeypatch.setattr(daemon, "_record_task_attempt", forbidden)
    monkeypatch.setattr(daemon, "_run_implementation_in_ephemeral_worktree", forbidden)
    result = daemon.reconcile_obsolete_protected_completion_journal()
    assert result.get("checkout_mutation_lease_recovered") is True, result
    assert not successor["lock"].exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    assert (successor["root"] / "todo.md").read_bytes() == board
    assert successor["publication"].read_bytes() == publication
    assert result.get("settlement_path"), result
    settlement = json.loads(Path(result["settlement_path"]).read_text())
    assert settlement["retired"] is True
    assert settlement["proof"]["old_guard_trusted"] is False


@pytest.mark.parametrize("damage", ["missing_publication", "foreign_publication", "board_edit", "committed_board_history"])
def test_superseded_journal_refuses_incomplete_or_changed_successor(successor, damage):
    lease = read_checkout_mutation_lease(successor["lock"])
    # Confirm this real fixture is admissible before changing one bound input;
    # an unrelated earlier refusal must not make a negative pass vacuously.
    successor["daemon"]._obsolete_protected_completion_proof(lease)
    if damage == "missing_publication":
        successor["publication"].rename(successor["directory"] / "preserved-publication.json")
    elif damage == "foreign_publication":
        d = json.loads(successor["publication"].read_text())
        d["publication"]["queue_task_cid"] = "foreign-task-cid"
        successor["publication"].write_text(json.dumps(d))
    else:
        p = successor["root"] / "todo.md"
        p.write_bytes(p.read_bytes() + b"\nUnaccepted protected board edit.\n")
        if damage == "committed_board_history":
            git(successor["root"], "add", "todo.md")
            git(successor["root"], "commit", "-m", "foreign protected board history")
    result = successor["daemon"].reconcile_obsolete_protected_completion_journal()
    assert result.get("checkout_mutation_lease_recovered") is not True, result
    _assert_preserved(successor)


def test_superseded_journal_preserves_a_genuine_live_owner(successor):
    lease = read_checkout_mutation_lease(successor["lock"])
    assert lease is not None
    successor["daemon"]._obsolete_protected_completion_proof(lease)
    live = update_checkout_mutation_lease(lease, {
        **dict(lease.metadata), "pid": os.getpid(),
        "owner_script": Path(sys.argv[0]).name,
    })
    assert live is not None
    successor["old_bytes"] = successor["lock"].read_bytes()
    result = successor["daemon"].reconcile_obsolete_protected_completion_journal()
    assert result.get("checkout_mutation_lease_recovered") is not True, result
    assert result.get("error") == "journal_owner_active", result
    _assert_preserved(successor)


@pytest.mark.parametrize("change", ["new_inode", "same_inode_changed_body"])
def test_superseded_journal_rechecks_exact_lease_at_release(successor, monkeypatch, change):
    lease = read_checkout_mutation_lease(successor["lock"])
    successor["daemon"]._obsolete_protected_completion_proof(lease)
    original = native.release_checkout_mutation_lease
    changed = []

    def race(lease, **kwargs):
        assert kwargs.get("require_exact_metadata") is True
        path = successor["lock"]
        if change == "new_inode":
            replacement = path.with_name(path.name + ".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            body = json.loads(path.read_text())
            body["operation"] = "foreign_replacement_operation"
            path.write_text(json.dumps(body, sort_keys=True))
        changed.append(path.read_bytes())
        return original(lease, **kwargs)

    monkeypatch.setattr(native, "release_checkout_mutation_lease", race)
    result = successor["daemon"].reconcile_obsolete_protected_completion_journal()
    assert changed, result
    assert result.get("checkout_mutation_lease_recovered") is not True, result
    assert successor["lock"].read_bytes() == changed[0]
    after = asdict(native.PortalTaskState.load(successor["daemon"].state_path))
    assert after["implementation_attempts"] == successor["state_before"]["implementation_attempts"]


def test_superseded_journal_cannot_replay_after_retirement(successor):
    daemon = successor["daemon"]
    first = daemon.reconcile_obsolete_protected_completion_journal()
    assert first.get("checkout_mutation_lease_recovered") is True, first
    successor["lock"].write_bytes(successor["old_bytes"])
    result = daemon.reconcile_obsolete_protected_completion_journal()
    assert result.get("checkout_mutation_lease_recovered") is not True, result
    _assert_preserved(successor)


def test_quiesced_public_recovery_settles_before_protected_path_admission(successor):
    daemon = successor["daemon"]
    # Recreate a delayed save of actual producer state, whose real process
    # has exited.  These are observed Native bytes, not a fabricated outcome.
    stale = successor["directory"] / "actual-producer-active-state.json"
    daemon.state_path.write_bytes(stale.read_bytes())
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    assert before["implementation_in_progress"] is True
    assert before["active_task_id"] == TASK
    result = daemon.reconcile_quiesced_active_attempt()
    assert result.get("reconciled") is True and not result.get("blocked"), result
    assert not successor["lock"].exists()
    after = asdict(native.PortalTaskState.load(daemon.state_path))
    assert after["implementation_attempts"] == before["implementation_attempts"]
    assert after["implementation_attempts_by_cid"] == before["implementation_attempts_by_cid"]
    assert after["completed_task_ids"] == before["completed_task_ids"]


def supervisor_for(successor):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor, PortalSupervisorConfig,
    )
    daemon = successor["daemon"]
    return PortalImplementationSupervisor(PortalSupervisorConfig(
        todo_path=daemon.todo_path, state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=successor["directory"] / "state/supervisor_events.jsonl",
        state_dir=successor["directory"] / "state", state_prefix="fixture",
        repo_root=successor["root"], task_prefix="EX-",
        implementation_command="forbidden-model-provider", max_task_attempts=3,
        worktree_root=successor["directory"] / "worktrees",
        worktree_submodule_paths=[], merge_target_branch="main",
        merge_queue_dir=daemon.merge_queue_dir,
        implementation_protected_paths=("todo.md",),
    ))


def test_supervisor_preflight_delegates_real_native_owned_journal(successor):
    daemon = successor["daemon"]
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    supervisor = supervisor_for(successor)
    result = supervisor._recover_retained_generated_checkout_lease()
    assert result.get("recovered") is True and not result.get("retained_lease"), result
    assert result["native_checkout_recovery"]["checkout_mutation_lease_recovered"] is True
    assert not successor["lock"].exists()
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == before


def test_settlement_refuses_a_real_live_attempt_zero_implementation_lease(successor):
    daemon = successor["daemon"]
    daemon._obsolete_protected_completion_proof(read_checkout_mutation_lease(successor["lock"]))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(native.__file__).parents[3]), env.get("PYTHONPATH", ""))))
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              "--hold-implementation-lease", str(successor["directory"])],
                             env=env, text=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready = successor["directory"] / "live-implementation-ready.json"
        deadline = time.monotonic() + 25
        while not ready.exists():
            assert child.poll() is None, child.communicate(timeout=5)
            assert time.monotonic() < deadline, "owned child did not publish its lease"
            time.sleep(.02)
        metadata = json.loads(ready.read_text())
        assert metadata["pid"] == child.pid and metadata["attempt"] == 0
        result = daemon.reconcile_obsolete_protected_completion_journal()
        assert result.get("checkout_mutation_lease_recovered") is not True, result
        _assert_preserved(successor)
    finally:
        if child.poll() is None:
            out, err = child.communicate("x", timeout=15)
            assert child.returncode == 0, (out[-2000:], err[-3000:])


def test_settlement_accepts_its_real_current_supervisor_maintenance_lease(successor):
    daemon = successor["daemon"]
    supervisor = supervisor_for(successor)
    lease, guard = supervisor._acquire_implementation_maintenance_lease()
    assert lease is not None, guard
    assert lease["pid"] == os.getpid() and lease["lease_role"] == "supervisor_maintenance"
    before = asdict(native.PortalTaskState.load(daemon.state_path))
    try:
        result = daemon.reconcile_obsolete_protected_completion_journal()
        assert result.get("checkout_mutation_lease_recovered") is True, result
        assert not successor["lock"].exists()
        assert asdict(native.PortalTaskState.load(daemon.state_path)) == before
    finally:
        supervisor._release_implementation_maintenance_lease(lease)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--produce-bad-completion":
    _produce_bad_completion(Path(sys.argv[2]))
elif __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--baseline-recover":
    _try_baseline_recovery(Path(sys.argv[2]))
elif __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--hold-implementation-lease":
    _hold_actual_implementation_lease(Path(sys.argv[2]))
