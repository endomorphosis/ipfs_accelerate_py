"""Private fresh Git checkout creation without changing the owner's umask.

The factory, decorated workspace guard, Git commands and subsequent descriptor
provisioning are real. Only deployment boundary authentication is authored.
The isolated fixture selects no pool, context seeding or submodules. Shared Git
configuration, ACL-bearing trees and populated seed outputs require their own
admission; this test does not infer protected modes for those paths.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as implementation


DRIVER = r'''
import hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
sys.path[:0] = CONFIG["runtime_paths"]
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.runtime import owner_writable_worktree as provisioner
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
assert "IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_FD" not in os.environ
assert "IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_SHA256" not in os.environ
assert "IPFS_ACCELERATE_TEST_UNFORWARDED_PRIVATE_KEY" not in os.environ

def mask():
    rows = [line.split(":",1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Umask:")]
    assert len(rows) == 1
    return int(rows[0],8)

def mode(path):
    return stat.S_IMODE(path.stat().st_mode)

def git(repo,*args):
    result=subprocess.run(["git","-c","core.hooksPath=/dev/null",*args],cwd=repo,
        text=True,capture_output=True,check=False,umask=0o077)
    assert result.returncode == 0,result.stderr
    return result.stdout.strip()

base=Path(CONFIG["base"])
repo=base/"repository"
repo.mkdir(mode=0o700)
git(repo,"init","-b","main")
git(repo,"config","user.name","Protected checkout fixture")
git(repo,"config","user.email","checkout@example.invalid")
git(repo,"config","core.filemode","true")
(repo/"sub/bin").mkdir(mode=0o700,parents=True)
(repo/"plain.txt").write_bytes(b"unchanged original source\n")
(repo/"sub/bin/run").write_bytes(b"#!/bin/sh\nexit 7\n")
(repo/"sub/bin/run").chmod(0o755)
git(repo,"add","plain.txt","sub/bin/run")
git(repo,"commit","-m","inert protected checkout source")
expected_head=git(repo,"rev-parse","HEAD")
root=base/"worktrees"
root.mkdir(mode=0o700)
owner=base/"owner"
owner.mkdir(mode=0o700)
boundary=owner/"boundary.json"
boundary.write_bytes(b"authored isolated deployment boundary\n")
boundary.chmod(0o444)
private=owner/"private-state"
private.write_bytes(b"private authority untouched")
private.chmod(0o600)
private_before=(private.stat().st_ino,mode(private),private.read_bytes())
def owner_boundary(path):
    assert Path(path)==boundary
    return dict(owner_uid=os.getuid(),worker_uid=os.getuid()+1,
        allowed_worktree_roots=[str(root)],owner_private_paths=[str(private)]),hashlib.sha256(boundary.read_bytes()).hexdigest()
boundary_guard._owner_boundary=owner_boundary
daemon=PortalImplementationDaemon(todo_path=base/"tasks.todo.md",state_path=owner/"state.json",
    strategy_path=owner/"strategy.json",events_path=owner/"events.jsonl",repo_root=repo,
    use_ephemeral_worktree=True,worktree_root=root,worktree_pool_enabled=False,
    worktree_submodule_paths=(),merge_target_branch="main")
assert daemon.worktree_pool is None
assert daemon._effective_worktree_submodule_paths() == ()
workspace=root/"candidate"
before_mask=mask()
assert before_mask==CONFIG["ambient_umask"]
ambient_before=base/"ambient-before"
ambient_before.write_bytes(b"before")
before_sentinel=(ambient_before.stat().st_ino,mode(ambient_before),ambient_before.read_bytes())
calls=[]
original_git=daemon._run_git
def recorded_git(args,*,cwd,child_umask=None):
    calls.append(dict(args=list(args),child_umask=child_umask))
    return original_git(args,cwd=cwd,child_umask=child_umask)
daemon._run_git=recorded_git
submodule_calls=[]
original_submodules=daemon._initialize_worktree_submodules
def submodules(path,**kwargs):
    submodule_calls.append(dict(path=str(path),**kwargs))
    return original_submodules(path,**kwargs)
daemon._initialize_worktree_submodules=submodules
def unselected(*args,**kwargs):
    raise AssertionError("no-seed factory reached context linking or copying")
daemon._link_shared_worktree_paths=unselected
daemon._seed_untracked_worktree_context=unselected
def forbidden_umask(*args):
    raise AssertionError("production path changed process-wide owner umask")
os.umask=forbidden_umask
actual_head=daemon._create_seeded_worktree(workspace,"implementation/protected-child",
    allow_pool=False,seed_context=False,offline_local_only=True)
assert actual_head==expected_head
assert len(submodule_calls)==1
assert submodule_calls[0]==dict(path=str(workspace),branch_name="implementation/protected-child",
    offline_local_only=True,task=None)
assert calls[0]["args"][:2]==["worktree","add"]
assert calls[0]["child_umask"]==0o077
assert all(call["child_umask"] is None for call in calls[1:])
paths=(workspace,workspace/"sub",workspace/"sub/bin",workspace/"plain.txt",
    workspace/"sub/bin/run",workspace/".git")
initial_modes={str(path.relative_to(workspace)):mode(path) for path in paths}
assert initial_modes=={".":0o700,"sub":0o700,"sub/bin":0o700,"plain.txt":0o600,
    "sub/bin/run":0o700,".git":0o600}
initial_identity={str(path.relative_to(workspace)):(path.stat().st_dev,path.stat().st_ino)
    for path in paths}
assert git(workspace,"ls-files","--stage","sub/bin/run").startswith("100755 ")
assert (workspace/"plain.txt").read_bytes()==b"unchanged original source\n"
assert (workspace/"sub/bin/run").read_bytes()==b"#!/bin/sh\nexit 7\n"
# There is no fixture chmod on this checkout. The strict helper consumes the
# actual output of the production Git factory before changing any permissions.
receipt=provisioner.provision_owner_writable_worktree(boundary=boundary,workspace=workspace,
    absolute_deadline_ns=time.monotonic_ns()+30_000_000_000)
assert receipt["descriptor_permission_transition_verified"] is True
assert mode(workspace)==0o3770 and mode(workspace/".git")==0o440
assert mode(workspace/"plain.txt")==0o660 and mode(workspace/"sub/bin/run")==0o770
assert all((path.stat().st_dev,path.stat().st_ino)==initial_identity[str(path.relative_to(workspace))]
    for path in paths)
ambient_after=base/"ambient-after"
ambient_after.write_bytes(b"after")
assert mask()==before_mask
assert mode(ambient_before)==mode(ambient_after)==(0o666&~before_mask)
assert (ambient_before.stat().st_ino,mode(ambient_before),ambient_before.read_bytes())==before_sentinel
assert (private.stat().st_ino,mode(private),private.read_bytes())==private_before
print("[protected-checkout] "+json.dumps(dict(ambient_umask=before_mask,parent_umask_unchanged=True,
    initial_modes=initial_modes,executable_git_mode="100755",provisioned=True,
    submodule_count=0,context_seeded=False,pool_selected=False,
    process_global_umask_called=False,provider_calls=0),sort_keys=True))
'''


def current_umask():
    values = [line.split(":", 1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Umask:")]
    assert len(values) == 1
    return int(values[0], 8)


@pytest.mark.parametrize("ambient_umask", [0o002, 0o000])
def test_actual_factory_creates_provisionable_private_checkout_without_parent_umask_change(
        tmp_path, monkeypatch, ambient_umask):
    parent_mask = current_umask()
    driver = tmp_path / "private-checkout-driver.py"
    public = dict(base=str(tmp_path), ambient_umask=ambient_umask,
        runtime_paths=[path for path in sys.path if type(path) is str and path])
    source = "CONFIG = " + repr(public) + "\n" + DRIVER
    ast.parse(source, filename=str(driver))
    driver.write_text(source)
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir(mode=0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_TEST_UNFORWARDED_PRIVATE_KEY", "authored-private-environment-sentinel")
    # Forward only explicit public runtime controls, never provider credentials,
    # native authorization tokens, or the sealed resource-parent capability.
    public_environment = ("PATH", "LANG", "LC_ALL", "TMPDIR", "TZ",
        "IPFS_ACCELERATE_TEST_PINNED_OWNER_RUNNER", "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH")
    environment = {name: os.environ[name] for name in public_environment if name in os.environ}
    environment.update(PYTHONPATH=os.pathsep.join(public["runtime_paths"]),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        RAYON_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
        IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false",
        IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS="0", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1",
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(orchestration),
        IPFS_ACCEL_SKIP_CORE="1", IPFS_KIT_DISABLE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", GIT_OPTIONAL_LOCKS="0")
    pinned = environment.get("IPFS_ACCELERATE_TEST_PINNED_OWNER_RUNNER")
    command = ([sys.executable, pinned, sys.executable, str(driver)] if pinned
        else [sys.executable, str(driver)])
    result = subprocess.run(command, cwd=Path(implementation.__file__).parents[3],
        env=environment, text=True, capture_output=True, check=False,
        timeout=120, close_fds=True, umask=ambient_umask)
    assert result.returncode == 0, result.stderr
    observations = [line.removeprefix("[protected-checkout] ") for line in result.stdout.splitlines()
        if line.startswith("[protected-checkout] ")]
    assert len(observations) == 1
    observation = json.loads(observations[0])
    assert observation["ambient_umask"] == ambient_umask
    assert observation["parent_umask_unchanged"] is True
    assert observation["provisioned"] is True
    assert observation["provider_calls"] == 0
    assert current_umask() == parent_mask


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir(mode=0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(orchestration))
    return implementation.PortalImplementationDaemon(todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json", strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl", repo_root=tmp_path,
        worktree_pool_enabled=False, worktree_submodule_paths=())


@pytest.mark.parametrize("child_umask", [True, False, 0, 0o002, 0o022, -1, 0o777, "077", 0.077])
def test_nonexact_or_nonprivate_child_mask_refuses_before_any_subprocess(
        daemon, tmp_path, monkeypatch, child_umask):
    monkeypatch.setattr(implementation.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("invalid child mask reached subprocess"))
    with pytest.raises(ValueError):
        daemon._run_git(["worktree", "add", "--detach", "literal-candidate"],
            cwd=tmp_path, child_umask=child_umask)


@pytest.mark.parametrize("args", [["status", "--porcelain"], ["rev-parse", "HEAD"],
    ["worktree", "remove", "literal-candidate"], ["worktree"]])
def test_private_child_mask_cannot_be_applied_to_unrelated_git_commands(
        daemon, tmp_path, monkeypatch, args):
    monkeypatch.setattr(implementation.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("noncreation command reached subprocess with private mask"))
    with pytest.raises(ValueError):
        daemon._run_git(args, cwd=tmp_path, child_umask=0o077)


@pytest.mark.parametrize("options", [{}, {"child_umask": None}, {"child_umask": 0o077}])
def test_git_runner_preserves_default_arguments_and_scopes_private_mask(
        daemon, tmp_path, monkeypatch, options):
    args = ["worktree", "add", "--detach", "literal-candidate"]
    returned = subprocess.CompletedProcess(["git"], 0, "unchanged output\n", "")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return returned
    monkeypatch.setattr(implementation.subprocess, "run", run)
    monkeypatch.setattr(implementation.os, "umask",
        lambda *_args: pytest.fail("Git runner changed its parent's process-wide umask"))
    assert daemon._run_git(args, cwd=tmp_path, **options) is returned
    expected = dict(cwd=tmp_path, text=True, capture_output=True, check=False)
    if options.get("child_umask") is not None:
        expected["umask"] = 0o077
    assert calls == [(["git", "--no-optional-locks", *args], expected)]


@pytest.mark.parametrize("child_umask", [None, 0o077])
def test_git_failure_keeps_existing_error_and_never_changes_owner_mask(
        daemon, tmp_path, monkeypatch, child_umask):
    args = ["worktree", "add", "--detach", "literal-candidate"]
    monkeypatch.setattr(implementation.subprocess, "run", lambda *_args, **_kwargs:
        subprocess.CompletedProcess(["git"], 7, "", "  authored Git failure\n"))
    monkeypatch.setattr(implementation.os, "umask",
        lambda *_args: pytest.fail("Git failure changed its parent's process-wide umask"))
    with pytest.raises(RuntimeError, match="git worktree add --detach literal-candidate failed: authored Git failure"):
        daemon._run_git(args, cwd=tmp_path, child_umask=child_umask)
