"""Actual Git checkouts and kernel exclusion for physical storage admission."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import checkout_storage as storage
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import WorktreePool
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTask, PortalTaskState, WorktreeStorageAdmissionDeferred,
)


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.PIPE).decode().strip()


def seed(path, payload=b"hello\n"):
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Storage Test")
    git(path, "config", "user.email", "storage@example.invalid")
    (path / "payload").write_bytes(payload)
    git(path, "add", "payload")
    git(path, "commit", "-m", "seed")
    return path


def add_submodule(parent, source, relative):
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(source), relative)
    git(parent, "commit", "-am", "dependency")


@pytest.fixture(autouse=True)
def isolated_allocation_inode(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_LOCK_ROOT", tmp_path / "allocator")
    monkeypatch.setenv("IPFS_ACCELERATE_WORKTREE_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("IPFS_ACCELERATE_WORKTREE_MIN_FREE_INODES", "0")


@pytest.fixture
def policy():
    return storage.CheckoutStoragePolicy(0, 0, 0.4)


def daemon_for(repo, tmp_path, policy, *, pool=False, dependencies=()):
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=repo, use_ephemeral_worktree=True, worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=pool, worktree_submodule_paths=dependencies,
        worktree_storage_policy=policy,
    )


def test_exact_sizes_nested_pins_and_nonexistent_destination(tmp_path):
    grand = seed(tmp_path / "grand", b"grand bytes\n")
    child = seed(tmp_path / "child", b"child\n")
    add_submodule(child, grand, "nested")
    parent = seed(tmp_path / "parent", b"parent payload\n")
    add_submodule(parent, child, "vendor/child")
    # Initialize the actual local nested object store; estimates never fetch.
    git(parent / "vendor/child", "-c", "protocol.file.allow=always", "submodule", "update", "--init")
    target = tmp_path / "future/checkout"
    estimates = storage.estimate_configured_checkouts(parent, target, "main", ("vendor/child/nested",))
    assert len(estimates) == 3
    assert [item.blob_bytes for item in estimates] == [
        len(b"parent payload\n") + (parent / ".gitmodules").stat().st_size,
        len(b"child\n") + (child / ".gitmodules").stat().st_size,
        len(b"grand bytes\n"),
    ]
    assert [item.file_count for item in estimates] == [2, 2, 1]
    assert estimates[1].commit == git(child, "rev-parse", "HEAD")
    assert estimates[2].commit == git(grand, "rev-parse", "HEAD")
    assert not target.parent.exists()


@pytest.mark.parametrize("resource", ["bytes", "inodes", "unknown"])
def test_pool_refusal_precedes_state_and_branch_mutation(tmp_path, monkeypatch, policy, resource):
    repo = seed(tmp_path / "repo")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool", storage_policy=policy)
    sample = storage._sample
    def limited(path):
        if resource == "unknown":
            raise storage.CheckoutStorageDeferred("checkout_storage_unavailable", path=str(path))
        return {**sample(path), "available_" + resource: 0}
    monkeypatch.setattr(storage, "_sample", limited)
    before = git(repo, "worktree", "list", "--porcelain")
    with pytest.raises(storage.CheckoutStorageDeferred, match="checkout_storage"):
        pool.acquire(cache_key="cold", branch_name="task/new")
    assert git(repo, "worktree", "list", "--porcelain") == before
    assert git(repo, "branch", "--list", "task/new") == ""
    assert list(pool.state_root.iterdir()) == []
    assert list(pool.worktree_root.iterdir()) == [pool.state_root]


@pytest.mark.parametrize("resource", ["bytes", "inodes"])
def test_source_git_filesystem_capacity_is_separately_checked(tmp_path, monkeypatch, policy, resource):
    repo = seed(tmp_path / "repo")
    estimate = storage.estimate_checkout(repo, tmp_path / "target", "main")
    sample = storage._sample
    def separate(path):
        value = sample(path)
        if path == estimate.common_directory:
            return {**value, "device": 9002, "available_" + resource: 0}
        return {**value, "device": 9001}
    monkeypatch.setattr(storage, "_sample", separate)
    with pytest.raises(storage.CheckoutStorageDeferred) as caught:
        storage.require_capacity((estimate,), policy)
    assert caught.value.details["device"] == 9002
    assert [item["role"] for item in caught.value.details["paths"]] == ["git_store"]


def test_unknown_configured_pin_refuses_without_fetch_or_branch(tmp_path, policy):
    child = seed(tmp_path / "child")
    repo = seed(tmp_path / "repo")
    add_submodule(repo, child, "child")
    missing = "f" * 40
    git(repo, "update-index", "--cacheinfo", "160000," + missing + ",child")
    git(repo, "commit", "-m", "unavailable pin")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool", storage_policy=policy)
    with pytest.raises(storage.CheckoutStorageDeferred, match="dependency_source_unavailable"):
        pool.acquire(cache_key="pin", branch_name="task/new", dependency_paths=("child",))
    assert git(repo, "branch", "--list", "task/new") == ""
    assert not (repo / ".git/FETCH_HEAD").exists()
    assert list(pool.state_root.iterdir()) == []


def test_reentrant_preparation_retains_original_kernel_lock(tmp_path, policy):
    repo = seed(tmp_path / "repo")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool", storage_policy=policy)
    observations = []
    def prepare(path):
        inode = os.fstat(storage._LOCK_FD).st_ino
        with storage.checkout_allocation(repo_root=repo, destination=tmp_path / "nested", ref="main", policy=policy):
            observations.append((inode, os.fstat(storage._LOCK_FD).st_ino, storage._LOCK_DEPTH))
            fd = os.open(storage._LOCK_ROOT / "allocation.lock", os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
    lease = pool.acquire(cache_key="nested", prepare=prepare)
    assert observations[0][0] == observations[0][1] and observations[0][2] == 2
    assert lease.path.joinpath("payload").read_text() == "hello\n"
    assert storage._LOCK_FD is None and storage._LOCK_DEPTH == 0


def test_body_exception_preserves_exception_and_releases_kernel_lock(policy):
    with pytest.raises(OSError, match="prepare failure"):
        with storage.allocation_exclusion(policy):
            raise OSError("prepare failure")
    with storage.allocation_exclusion(policy):
        assert storage._LOCK_DEPTH == 1


@pytest.mark.parametrize("tamper", ["symlink", "hardlink", "permissions"])
def test_allocation_lock_identity_refuses_aliases(tmp_path, policy, tamper):
    directory = storage._LOCK_ROOT
    directory.mkdir(mode=0o700)
    lock = directory / "allocation.lock"
    original = tmp_path / "original"
    original.touch(mode=0o600)
    if tamper == "symlink":
        lock.symlink_to(original)
    elif tamper == "hardlink":
        os.link(original, lock)
    else:
        lock.touch(mode=0o644)
    before = original.read_bytes()
    with pytest.raises(storage.CheckoutStorageDeferred, match="allocation_lock"):
        with storage.allocation_exclusion(policy):
            pytest.fail("entered")
    assert original.read_bytes() == before


@pytest.mark.parametrize("pooled", [False, True])
def test_native_parent_creates_exact_parent_and_dependency(tmp_path, policy, pooled):
    child = seed(tmp_path / "child", b"dependency\n")
    repo = seed(tmp_path / "repo")
    add_submodule(repo, child, "child")
    daemon = daemon_for(repo, tmp_path, policy, pool=pooled, dependencies=("child",))
    target = tmp_path / "worktrees/attempt"
    baseline = daemon._create_seeded_worktree(target, "task/attempt", seed_context=False, offline_local_only=True)
    actual = daemon._effective_pooled_worktree_path(target) if pooled else target
    assert baseline == git(repo, "rev-parse", "HEAD")
    assert actual.joinpath("child/payload").read_text() == "dependency\n"
    assert git(actual / "child", "rev-parse", "HEAD") == git(child, "rev-parse", "HEAD")


@pytest.mark.parametrize("pooled", [False, True])
def test_native_parent_storage_failure_is_typed_retry(tmp_path, monkeypatch, policy, pooled):
    repo = seed(tmp_path / "repo")
    daemon = daemon_for(repo, tmp_path, policy, pool=pooled)
    sample = storage._sample
    monkeypatch.setattr(storage, "_sample", lambda path: {**sample(path), "available_bytes": 0})
    with pytest.raises(WorktreeStorageAdmissionDeferred) as caught:
        daemon._create_seeded_worktree(tmp_path / "worktrees/attempt", "task/attempt", seed_context=False)
    assert caught.value.reason == "checkout_storage_bytes_low" and caught.value.backoff_seconds == 30
    assert git(repo, "branch", "--list", "task/attempt") == ""


def test_nested_refusal_leaves_existing_target_bytes_and_registration(tmp_path, monkeypatch, policy):
    child = seed(tmp_path / "child")
    repo = seed(tmp_path / "repo")
    add_submodule(repo, child, "child")
    target = tmp_path / "target"
    git(repo, "worktree", "add", "--detach", str(target), "main")
    (target / "child/retained").write_text("unknown artifacts")
    daemon = daemon_for(repo, tmp_path, policy, dependencies=("child",))
    sample = storage._sample
    monkeypatch.setattr(storage, "_sample", lambda path: {**sample(path), "available_bytes": 0})
    prior = git(repo / "child", "worktree", "list", "--porcelain")
    with pytest.raises(WorktreeStorageAdmissionDeferred):
        daemon._create_local_submodule_worktree(target, "child", branch_name="task/attempt")
    assert target.joinpath("child/retained").read_text() == "unknown artifacts"
    assert git(repo / "child", "worktree", "list", "--porcelain") == prior


@pytest.mark.parametrize("mode", ["filter", "encoding", "hook"])
def test_unbounded_checkout_expansion_refuses_without_executing_it(tmp_path, policy, mode):
    repo = seed(tmp_path / "repo")
    marker = tmp_path / "executed"
    if mode == "hook":
        hook = repo / ".git/hooks/post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        hook.chmod(0o700)
    else:
        attribute = "filter=unbounded" if mode == "filter" else "working-tree-encoding=UTF-32"
        (repo / ".gitattributes").write_text("payload " + attribute + "\n")
        git(repo, "add", ".gitattributes")
        git(repo, "commit", "-m", "checkout expansion")
    with pytest.raises(storage.CheckoutStorageDeferred, match="checkout_expansion_unbounded"):
        with storage.checkout_allocation(repo_root=repo, destination=tmp_path / "checkout", ref="HEAD", policy=policy):
            pytest.fail("unbounded checkout admitted")
    assert not marker.exists() and not (tmp_path / "checkout").exists()


def test_crlf_allowance_preserves_exact_stored_blob_total(tmp_path):
    repo = seed(tmp_path / "repo", b"a\n" * 4096)
    git(repo, "config", "core.autocrlf", "true")
    estimate = storage.estimate_checkout(repo, tmp_path / "checkout", "HEAD")
    assert estimate.blob_bytes == 8192 and estimate.checkout_expansion == 2


@pytest.mark.parametrize("pooled", [False, True])
def test_actual_attempt_storage_refusal_never_dispatches_or_consumes_attempt(tmp_path, monkeypatch, policy, pooled):
    repo = seed(tmp_path / "repo")
    daemon = daemon_for(repo, tmp_path, policy, pool=pooled)
    daemon.implement = True
    marker = tmp_path / "provider-invoked"
    daemon.implementation_command = f"{sys.executable} -c \"open('{marker}', 'w').write('called')\""
    sample = storage._sample
    monkeypatch.setattr(storage, "_sample", lambda path: {**sample(path), "available_bytes": 0})
    task = PortalTask(task_id="STORAGE-001", title="Storage admission", status="todo",
                      completion="manual", priority="P1", track="runtime")
    result = daemon._run_implementation(task, PortalTaskState())
    assert result["reason"] == "checkout_storage_bytes_low", result
    assert result["deferred"] is True and result["attempt_consumed"] is False
    assert not marker.exists()
    assert result["provider_call_allowed"] is False


def test_kernel_exclusion_times_out_without_signalling_holder(tmp_path, policy):
    with storage.allocation_exclusion(policy):
        lock = storage._LOCK_ROOT / "allocation.lock"
    code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.readline()"
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(lock)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        started = time.monotonic()
        with pytest.raises(storage.CheckoutStorageDeferred, match="checkout_allocation_busy"):
            with storage.allocation_exclusion(policy):
                pytest.fail("competing writer entered")
        assert 0.35 <= time.monotonic() - started < 2
        assert child.poll() is None and lock.exists()
        child.communicate("release\n", timeout=5)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_fork_child_cannot_inherit_or_unlock_parent_admission(policy):
    with storage.allocation_exclusion(policy):
        pid = os.fork()
        if pid == 0:
            try:
                with storage.allocation_exclusion(policy):
                    os._exit(2)
            except storage.CheckoutStorageDeferred as exc:
                os._exit(0 if exc.reason == "checkout_allocation_busy" else 3)
            except BaseException:
                os._exit(4)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        fd = os.open(storage._LOCK_ROOT / "allocation.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


@pytest.mark.parametrize("failure", [None, "checkout", "read_tree"])
def test_existing_nested_branch_reference_is_bound_and_released(tmp_path, monkeypatch, policy, failure):
    child = seed(tmp_path / "child")
    repo = seed(tmp_path / "repo")
    add_submodule(repo, child, "child")
    target = tmp_path / "target"
    git(repo, "worktree", "add", "--detach", str(target), "main")
    daemon = daemon_for(repo, tmp_path, policy, dependencies=("child",))
    branch = daemon._submodule_worktree_branch_name("task/attempt", "child")
    source = repo / "child"
    git(source, "branch", branch)
    original = git(source, "rev-parse", branch)
    native_git = daemon._run_git
    observed = []
    def command(args, *, cwd):
        if args[0] in {"worktree", "read-tree"} and ("add" in args or args[0] == "read-tree"):
            reference_lock = storage._common_dir(source) / "refs/heads" / (branch + ".lock")
            assert reference_lock.exists()
            contender = subprocess.run(["git", "update-ref", "refs/heads/" + branch, original],
                                       cwd=source, capture_output=True)
            assert contender.returncode != 0
            observed.append(args[0])
            if (failure == "checkout" and args[0] == "worktree") or (failure == "read_tree" and args[0] == "read-tree"):
                raise RuntimeError("injected native operation failure")
        return native_git(args, cwd=cwd)
    monkeypatch.setattr(daemon, "_run_git", command)
    if failure:
        with pytest.raises(RuntimeError, match="injected native operation failure"):
            daemon._create_local_submodule_worktree(target, "child", branch_name="task/attempt")
    else:
        assert daemon._create_local_submodule_worktree(target, "child", branch_name="task/attempt")
        assert target.joinpath("child/payload").read_text() == "hello\n"
        assert git(target / "child", "branch", "--show-current") == branch
        assert git(target / "child", "status", "--porcelain") == ""
    assert observed
    assert git(source, "rev-parse", branch) == original
    assert not (storage._common_dir(source) / "refs/heads" / (branch + ".lock")).exists()
    # A failed read-tree leaves its partial registration for the existing
    # native lifecycle cleanup; reference release never declares it settled.
    if failure == "read_tree":
        assert target.joinpath("child/.git").exists()
        assert str(target / "child") in git(source, "worktree", "list", "--porcelain")


def test_moved_branch_refuses_before_existing_branch_checkout(tmp_path, policy):
    repo = seed(tmp_path / "repo")
    git(repo, "branch", "retained")
    old = git(repo, "rev-parse", "HEAD")
    (repo / "payload").write_text("newer content")
    git(repo, "commit", "-am", "new")
    new = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "-f", "retained", new)
    with pytest.raises(storage.CheckoutStorageDeferred, match="checkout_branch_reference_unavailable"):
        with storage.verified_branch_reference(repo, "retained", old):
            pytest.fail("moved ref admitted")
    assert git(repo, "rev-parse", "retained") == new
    assert not (repo / ".git/refs/heads/retained.lock").exists()


def test_existing_branch_owned_by_another_worktree_remains_untouched(tmp_path, policy):
    child = seed(tmp_path / "child")
    repo = seed(tmp_path / "repo")
    add_submodule(repo, child, "child")
    target = tmp_path / "target"
    git(repo, "worktree", "add", "--detach", str(target), "main")
    daemon = daemon_for(repo, tmp_path, policy, dependencies=("child",))
    branch = daemon._submodule_worktree_branch_name("task/attempt", "child")
    other = tmp_path / "other-owner"
    source = repo / "child"
    git(source, "worktree", "add", "-b", branch, str(other), "HEAD")
    (other / "unknown-callback").write_text("retain")
    before = git(source, "worktree", "list", "--porcelain")
    with pytest.raises(RuntimeError, match="already (checked out|used by worktree)"):
        daemon._create_local_submodule_worktree(target, "child", branch_name="task/attempt")
    assert (other / "unknown-callback").read_text() == "retain"
    assert git(source, "worktree", "list", "--porcelain") == before
    assert not (storage._common_dir(source) / "refs/heads" / (branch + ".lock")).exists()


def test_global_attribute_source_is_included_in_exact_tree_check(tmp_path, policy):
    repo = seed(tmp_path / "repo")
    attributes = tmp_path / "global-attributes"
    attributes.write_text("payload filter=external-growth\n")
    git(repo, "config", "core.attributesFile", str(attributes))
    with pytest.raises(storage.CheckoutStorageDeferred, match="checkout_expansion_unbounded"):
        with storage.checkout_allocation(repo_root=repo, destination=tmp_path / "target", ref="HEAD", policy=policy):
            pytest.fail("external attribute expansion admitted")
    assert not (tmp_path / "target").exists()


def test_warm_pool_lease_does_not_wait_for_unrelated_cold_allocation(tmp_path, policy):
    repo = seed(tmp_path / "repo")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool", storage_policy=policy)
    first = pool.acquire(cache_key="same", branch_name="task/first")
    assert first.release()["pooled"] is True
    code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.readline()"
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(storage._LOCK_ROOT / "allocation.lock")],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        lease = pool.acquire(cache_key="same", branch_name="task/second")
        assert lease.reused and lease.path == first.path and child.poll() is None
        assert lease.release()["pooled"] is True
        child.communicate("release\n", timeout=5)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


PROCESS_SCRIPT = r'''
import json, os, subprocess, sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime import checkout_storage as s
repo, target, lock, first, budget = sys.argv[1:]
s._LOCK_ROOT = Path(lock)
real_sample = s._sample
def measured(path):
    used = (Path(first) / 'payload').stat().st_size if (Path(first) / 'payload').exists() else 0
    return {**real_sample(path), 'available_bytes': int(budget) - used}
s._sample = measured
try:
    with s.checkout_allocation(repo_root=Path(repo), destination=Path(target), ref='main',
                               policy=s.CheckoutStoragePolicy(0, 0, 4)) as estimates:
        print('entered', flush=True)
        if target == first:
            sys.stdin.readline()
        subprocess.run(['git','worktree','add','--detach',target,estimates[0].commit],cwd=repo,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print('created', flush=True)
except s.CheckoutStorageDeferred as e:
    print(e.reason, flush=True)
'''


def test_separate_process_allocators_resample_after_real_checkout(tmp_path, policy):
    repo = seed(tmp_path / "repo", b"x" * 1024 ** 2)
    first = tmp_path / "first"
    estimate = storage.estimate_checkout(repo, first, "main")
    sample = storage._sample
    # Capture the actual computed requirement against a deliberately low
    # observation, then expose only half one checkout's growth as spare room.
    try:
        storage._sample = lambda path: {**sample(path), "available_bytes": 0}
        with pytest.raises(storage.CheckoutStorageDeferred) as caught:
            storage.require_capacity((estimate,), policy)
        budget = caught.value.details["required_bytes"] + 512 * 1024
    finally:
        storage._sample = sample
    processes = []
    def launch(target):
        proc = subprocess.Popen([sys.executable, "-B", "-c", PROCESS_SCRIPT,
            str(repo), str(target), str(storage._LOCK_ROOT), str(first), str(budget)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        return proc
    try:
        one = launch(first)
        assert one.stdout.readline().strip() == "entered"
        two = launch(tmp_path / "second")
        time.sleep(0.15)
        assert two.poll() is None and not (tmp_path / "second").exists()
        one.stdin.write("continue\n"); one.stdin.flush()
        assert one.communicate(timeout=10)[0].strip() == "created"
        assert two.communicate(timeout=10)[0].strip() == "checkout_storage_bytes_low"
        assert first.joinpath("payload").stat().st_size == 1024 ** 2
        assert not (tmp_path / "second").exists()
        assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 2
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
