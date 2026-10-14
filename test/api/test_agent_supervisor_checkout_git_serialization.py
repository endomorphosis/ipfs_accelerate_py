"""Real checkout concurrency and fail-closed Git contention regressions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import git_checkout_transaction as guard
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor as ImplementationSupervisor,
)


def _git(repo, *args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Checkout concurrency fixture")
    _git(repo, "config", "user.email", "concurrency@example.invalid")
    (repo / "seed.txt").write_text("seed\n")
    (repo / "todo.md").write_text("## ACCEL-001 Fixture\n\n- Status: todo\n- Completion: auto\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "candidate")
    (repo / "result.txt").write_text("native result\n")
    _git(repo, "add", "result.txt")
    _git(repo, "commit", "-qm", "candidate output")
    candidate = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    _git(repo, "checkout", "-q", "main")
    return repo, baseline, candidate, tree


def _daemon(repo, state, namespace="fixture-board"):
    daemon = native.TodoImplementationDaemon(
        todo_path=repo / "todo.md", state_path=state / "state.json",
        strategy_path=state / "strategy.json", events_path=state / "events.jsonl",
        repo_root=repo, board_namespace=namespace, merge_target_branch="main",
        task_header_prefix="## ACCEL-",
        worktree_root=state / "worktrees", worktree_pool_enabled=False,
        worktree_submodule_paths=(), merge_queue_dir=state / "merge-queue",
    )
    daemon._invoke_llm_merge_resolver_for_failed_merge = lambda **kwargs: pytest.fail(
        "Git infrastructure contention must never invoke a model resolver"
    )
    return daemon


def _task():
    return native.PortalTask(task_id="ACCEL-001", title="Fixture", status="todo",
        completion="auto", priority="P1", track="quality", outputs=["result.txt"],
        metadata={"Provider role": "deterministic-only"})


def _merge(daemon, baseline, candidate, tree):
    return daemon._merge_branch_to_main("candidate", _task(), 1,
        baseline_ref=baseline, expected_candidate_commit=candidate,
        expected_candidate_tree=tree)


def _wait(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"child exited before rendezvous: {stdout[-1500:]} {stderr[-1500:]}")
        if time.monotonic() >= deadline:
            pytest.fail("bounded child rendezvous timed out")
        time.sleep(0.02)


HOLDER = """
import pathlib,sys,time
from ipfs_accelerate_py.agent_supervisor.merge.git_checkout_transaction import checkout_git_mutation
repo,ready,release=map(pathlib.Path,sys.argv[1:])
with checkout_git_mutation(repo):
 ready.touch()
 end=time.monotonic()+15
 while not release.exists():
  if time.monotonic()>end: raise RuntimeError('release timeout')
  time.sleep(.02)
"""


def test_same_checkout_cross_process_guard_times_out_without_inode_replacement(tmp_path):
    repo, *_ = _repo(tmp_path)
    ready, release = tmp_path / "ready", tmp_path / "release"
    child = subprocess.Popen([sys.executable, "-B", "-c", HOLDER, str(repo), str(ready), str(release)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(ready, child)
        path = guard.git_checkout_directory(repo) / guard.LOCK_NAME
        inode = path.stat().st_ino
        with pytest.raises(guard.CheckoutGitMutationBlocked, match="checkout_git_mutation_busy"):
            with guard.checkout_git_mutation(repo, timeout_seconds=0.05):
                pytest.fail("entered another process's transaction")
        assert path.stat().st_ino == inode
        release.touch()
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stdout + stderr
        with guard.checkout_git_mutation(repo, timeout_seconds=0):
            assert path.stat().st_ino == inode
    finally:
        release.touch()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def test_sibling_worktree_keeps_independent_index_mutation_capacity(tmp_path):
    repo, *_ = _repo(tmp_path)
    sibling = tmp_path / "sibling"
    _git(repo, "worktree", "add", "-b", "sibling", str(sibling))
    ready, release = tmp_path / "ready", tmp_path / "release"
    child = subprocess.Popen([sys.executable, "-B", "-c", HOLDER, str(repo), str(ready), str(release)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(ready, child)
        assert guard.git_checkout_directory(repo) != guard.git_checkout_directory(sibling)
        with guard.checkout_git_mutation(sibling, timeout_seconds=0):
            (sibling / "independent.txt").write_text("independent\n")
            _git(sibling, "add", "independent.txt")
            _git(sibling, "commit", "-qm", "independent checkout")
        assert not (repo / "independent.txt").exists()
    finally:
        release.touch()
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stdout + stderr


def test_nested_checkout_guard_is_reentrant(tmp_path):
    repo, *_ = _repo(tmp_path)
    with guard.checkout_git_mutation(repo, timeout_seconds=0):
        with guard.checkout_git_mutation(repo, timeout_seconds=0):
            _git(repo, "status", "--porcelain")


MERGER = """
import json,pathlib,sys
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon,PortalTask
repo,state=map(pathlib.Path,sys.argv[1:3]);baseline,candidate,tree=sys.argv[3:]
d=TodoImplementationDaemon(todo_path=repo/'todo.md',state_path=state/'state.json',strategy_path=state/'strategy.json',events_path=state/'events.jsonl',repo_root=repo,board_namespace='merge-board',merge_target_branch='main',task_header_prefix='## ACCEL-',worktree_root=state/'worktrees',worktree_pool_enabled=False,worktree_submodule_paths=(),merge_queue_dir=state/'merge-queue')
def no_provider(**kwargs): raise RuntimeError('provider must not be invoked')
d._invoke_llm_merge_resolver_for_failed_merge=no_provider
t=PortalTask(task_id='ACCEL-001',title='Fixture',status='todo',completion='auto',priority='P1',track='quality',outputs=['result.txt'],metadata={'Provider role':'deterministic-only'})
r=d._merge_branch_to_main('candidate',t,1,baseline_ref=baseline,expected_candidate_commit=candidate,expected_candidate_tree=tree)
print(json.dumps(r))
if not r.get('merged'): raise RuntimeError('native merge failed')
"""


def test_real_native_merge_serializes_other_board_completion_rewrite_and_commit(tmp_path):
    repo, baseline, candidate, tree = _repo(tmp_path)
    ready, release = tmp_path / "hook-ready", tmp_path / "hook-release"
    hook = guard.git_checkout_directory(repo) / "hooks/pre-merge-commit"
    hook_code = ("import pathlib,time;ready=pathlib.Path(" + repr(str(ready)) + ");"
                 "release=pathlib.Path(" + repr(str(release)) + ");ready.touch();"
                 "end=time.monotonic()+15\nwhile not release.exists():\n"
                 " if time.monotonic()>end: raise RuntimeError('hook timeout')\n"
                 " time.sleep(.02)\n")
    hook.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -B -c " + shlex.quote(hook_code) + "\n")
    hook.chmod(0o700)
    daemon = _daemon(repo, tmp_path / "completion-state", namespace="different-completion-board")
    child = subprocess.Popen([sys.executable, "-B", "-c", MERGER, str(repo), str(tmp_path / "merge-state"), baseline, candidate, tree],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(ready, child)
        before_todo = (repo / "todo.md").read_bytes()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(daemon._mark_tasks_completed_in_todo,
                                     ["ACCEL-001"], primary_task_id="ACCEL-001", completion_reason="merged")
            time.sleep(0.15)
            assert not future.done()
            assert (repo / "todo.md").read_bytes() == before_todo
            release.touch()
            stdout, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stdout[-3000:] + stderr[-3000:]
            completion = future.result(timeout=15)
        assert completion["updated"] is True
        assert completion["commit_result"]["committed"] is True
        assert "- Status: completed" in (repo / "todo.md").read_text()
        assert (repo / "result.txt").read_text() == "native result\n"
        assert _git(repo, "status", "--porcelain") == ""
        _git(repo, "merge-base", "--is-ancestor", candidate, "HEAD")
        assert not (guard.git_checkout_directory(repo) / "index.lock").exists()
        assert "llm_merge_resolver_invoked" not in daemon.events_path.read_text()
    finally:
        release.touch()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


@pytest.mark.parametrize("name", ["index.lock", "HEAD.lock"])
def test_foreign_git_lock_preserves_dirty_data_index_and_ownership(tmp_path, name):
    repo, baseline, candidate, tree = _repo(tmp_path)
    directory = guard.git_checkout_directory(repo)
    (repo / "foreign.txt").write_text("foreign staged\n")
    _git(repo, "add", "foreign.txt")
    (repo / "seed.txt").write_text("foreign unstaged\n")
    lock = directory / name
    lock.write_bytes(b"unknown foreign lock\n")
    before = (directory / "index").read_bytes(), lock.read_bytes(), lock.stat().st_ino
    daemon = _daemon(repo, tmp_path / "state")
    result = _merge(daemon, baseline, candidate, tree)
    assert result["reason"] == "checkout_git_foreign_lock_present"
    assert result["provider_dispatched"] is False
    assert daemon._merge_result_needs_reconciliation(result) is True
    assert daemon._merge_result_is_transient_lock_deferral(result) is True
    assert (directory / "index").read_bytes() == before[0]
    assert (lock.read_bytes(), lock.stat().st_ino) == before[1:]
    assert (repo / "seed.txt").read_text() == "foreign unstaged\n"
    assert (repo / "foreign.txt").read_text() == "foreign staged\n"
    assert not (repo / "result.txt").exists()
    assert _git(repo, "rev-parse", "HEAD") == baseline


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_untrusted_advisory_guard_fails_closed_without_replacing_inode(tmp_path, kind):
    repo, *_ = _repo(tmp_path)
    foreign = tmp_path / "foreign-guard"
    foreign.write_text("foreign\n")
    path = guard.git_checkout_directory(repo) / guard.LOCK_NAME
    if kind == "symlink":
        path.symlink_to(foreign)
    else:
        os.link(foreign, path)
    before = foreign.read_bytes(), foreign.stat().st_ino
    with pytest.raises(guard.CheckoutGitMutationBlocked):
        with guard.checkout_git_mutation(repo, timeout_seconds=0):
            pytest.fail("entered untrusted guard")
    assert (foreign.read_bytes(), foreign.stat().st_ino) == before
    assert os.path.lexists(path)


def test_readonly_native_status_scans_never_refresh_index(tmp_path, monkeypatch):
    repo, *_ = _repo(tmp_path)
    daemon = _daemon(repo, tmp_path / "state")
    monkeypatch.delenv("GIT_OPTIONAL_LOCKS", raising=False)
    directory = guard.git_checkout_directory(repo)
    path = directory / "index"
    (repo / "seed.txt").touch()
    before = path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino
    assert daemon._path_status(repo, "seed.txt") == ""
    assert daemon._dirty_worktree_paths(repo) == set()
    assert daemon._run_git(["status", "--porcelain"], cwd=repo).stdout == ""
    assert ImplementationSupervisor._git_status_short(repo) == []
    assert ImplementationSupervisor._git_status_short_strict(repo) == []
    assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) == before


def test_mid_merge_lock_failure_retains_owned_stages_and_never_routes_to_resolver(tmp_path, monkeypatch):
    repo, baseline, candidate, tree = _repo(tmp_path)
    daemon = _daemon(repo, tmp_path / "state")
    directory = guard.git_checkout_directory(repo)
    before = (directory / "index").read_bytes()
    run = subprocess.run

    def lock_failure(command, *args, **kwargs):
        if command[:4] == ["git", "merge", "--no-ff", "--no-edit"]:
            # Model Git's observed partial staging before a subsequent lock
            # failure. No MERGE_HEAD exists, so merge --abort cannot recover it.
            (repo / "result.txt").write_text("native result\n")
            run(["git", "add", "result.txt"], cwd=repo, check=True, capture_output=True)
            (directory / "index.lock").write_bytes(b"foreign writer\n")
            return subprocess.CompletedProcess(command, 128, "", f"fatal: Unable to create '{directory}/index.lock': File exists.\n")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", lock_failure)
    monkeypatch.setattr(daemon, "_abort_failed_merge", lambda *args: pytest.fail("foreign contention must not reset/abort"))
    result = _merge(daemon, baseline, candidate, tree)
    assert result["reason"] == "git_lock_contention" and result["infrastructure_failure"]
    assert result["provider_dispatched"] is False
    custody = result["git_state_custody"]
    path = Path(custody["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == custody["sha256"]
    receipt = json.loads(path.read_bytes())
    assert path.with_name("before.index").read_bytes() == before
    assert path.with_name("after.index").read_bytes() == (directory / "index").read_bytes()
    assert custody["index_changed"] is True
    assert receipt["completion_authority"] is False and receipt["data_restored_or_deleted"] is False
    assert (directory / "index.lock").read_bytes() == b"foreign writer\n"
    assert "result.txt" in _git(repo, "diff", "--cached", "--name-only")
    # Simulate only the foreign owner's release. Native retained custody must
    # continue blocking content/model recovery of the staged partial output.
    (directory / "index.lock").unlink()
    retry = _merge(daemon, baseline, candidate, tree)
    assert retry["reason"] == "checkout_git_retained_contention_requires_recovery"
    assert retry["checkout_mutation_recovery_required"] is True
    assert daemon._merge_result_needs_reconciliation(retry) is True
    assert daemon._merge_result_is_transient_lock_deferral(retry) is False
    assert (directory / guard.PENDING_CUSTODY_NAME).exists()
    assert _git(repo, "rev-parse", "HEAD") == baseline


def test_generated_commit_contention_retains_index_without_resetting_foreign_data(tmp_path, monkeypatch):
    repo, *_ = _repo(tmp_path)
    daemon = _daemon(repo, tmp_path / "state")
    directory = guard.git_checkout_directory(repo)
    (repo / "todo.md").write_text("updated generated board\n")
    before = (directory / "index").read_bytes()
    run = subprocess.run

    def lock_failure(command, *args, **kwargs):
        if command[0] == "git" and "commit" in command:
            (directory / "HEAD.lock").write_bytes(b"foreign HEAD owner\n")
            return subprocess.CompletedProcess(command, 128, "", f"fatal: Unable to create '{directory}/HEAD.lock': File exists.\n")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", lock_failure)
    result = daemon._commit_specific_path(repo, "todo.md", subject="Fixture completed")
    assert result["reason"] == "git_lock_contention"
    receipt = Path(result["git_state_custody"]["path"])
    assert receipt.with_name("before.index").read_bytes() == before
    assert receipt.with_name("after.index").read_bytes() == (directory / "index").read_bytes()
    assert (directory / "HEAD.lock").read_bytes() == b"foreign HEAD owner\n"
    assert (repo / "todo.md").read_text() == "updated generated board\n"


def test_failed_custody_publication_never_force_removes_partial_ephemeral_merge(tmp_path, monkeypatch):
    repo, baseline, candidate, tree = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "controller")
    target = tmp_path / "ephemeral-target"
    _git(repo, "worktree", "add", str(target), "main")
    directory = guard.git_checkout_directory(target)
    daemon = _daemon(repo, tmp_path / "state")
    monkeypatch.setattr(daemon, "_prepare_main_merge_workspace", lambda *args: {
        "available": True, "path": str(target), "ephemeral": True,
    })
    run = subprocess.run

    def lock_failure(command, *args, **kwargs):
        if command[:4] == ["git", "merge", "--no-ff", "--no-edit"]:
            assert Path(kwargs["cwd"]) == target
            (target / "result.txt").write_text("native result\n")
            run(["git", "add", "result.txt"], cwd=target, check=True, capture_output=True)
            (directory / "index.lock").write_bytes(b"foreign writer\n")
            return subprocess.CompletedProcess(command, 128, "", f"fatal: Unable to create '{directory}/index.lock': File exists.\n")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", lock_failure)
    monkeypatch.setattr(native, "retain_git_contention_custody", lambda *args, **kwargs: (
        _ for _ in ()).throw(OSError("simulated custody fsync failure")))
    result = _merge(daemon, baseline, candidate, tree)
    assert result["reason"] == "git_lock_contention_custody_failed"
    assert result["used_ephemeral_main_worktree"] is True
    assert result["provider_dispatched"] is False
    assert "fsync failure" in result["custody_error"]
    assert target.is_dir()
    assert (target / "result.txt").read_text() == "native result\n"
    assert (directory / "index.lock").read_bytes() == b"foreign writer\n"
    assert "result.txt" in _git(target, "diff", "--cached", "--name-only")
    assert _git(target, "rev-parse", "HEAD") == baseline


def test_failed_post_contention_capture_still_publishes_pending_barrier(tmp_path, monkeypatch):
    repo, *_ = _repo(tmp_path)
    before = guard.capture_checkout_index(repo)
    monkeypatch.setattr(guard, "capture_checkout_index", lambda repo: (_ for _ in ()).throw(
        guard.CheckoutGitMutationBlocked("checkout_git_index_changed_during_custody")))
    result = guard.retain_git_contention_custody(repo, before_index=before, task_id="fixture",
        branch="candidate", before_head=_git(repo, "rev-parse", "HEAD"),
        result=subprocess.CompletedProcess([], 128, "", "index.lock: File exists"))
    receipt = json.loads(Path(result["path"]).read_bytes())
    assert receipt["index_snapshots"]["after"]["present"] is None
    assert "capture_error" in receipt["index_snapshots"]["after"]
    with pytest.raises(guard.CheckoutGitMutationBlocked, match="retained_contention_requires_recovery"):
        with guard.checkout_git_mutation(repo, timeout_seconds=0):
            pytest.fail("entered retained failed custody")


@pytest.mark.parametrize("message,expected", [
    ("fatal: Unable to create '/repo/index.lock': File exists.", True),
    ("error: cannot lock ref 'refs/heads/main': Unable to create '/repo/main.lock': File exists.", True),
    ("CONFLICT (content): Merge conflict in file.txt", False),
    ("fatal: Unable to create '/repo/index.lock': Permission denied", False),
    ("fatal: index.lock is corrupt", False),
])
def test_only_explicit_git_lock_contention_is_infrastructure(message, expected):
    assert guard.git_lock_contention(subprocess.CompletedProcess([], 128, "", message)) is expected


def test_index_custody_bound_refuses_before_reading_oversized_index(tmp_path):
    repo, *_ = _repo(tmp_path)
    path = guard.git_checkout_directory(repo) / "index"
    with path.open("r+b") as stream:
        stream.truncate(guard.MAX_INDEX_BYTES + 1)
    with pytest.raises(guard.CheckoutGitMutationBlocked, match="index_custody_bound"):
        guard.capture_checkout_index(repo)


@pytest.mark.parametrize("reason", ["checkout_git_mutation_busy", "checkout_mutation_lock_exists"])
def test_busy_result_preserves_validated_candidate_for_native_retry(reason):
    result = {"attempted": False, "merged": False, "reason": reason}
    assert native.TodoImplementationDaemon._merge_result_needs_reconciliation(result) is True
    assert native.TodoImplementationDaemon._merge_result_is_transient_lock_deferral(result) is True
