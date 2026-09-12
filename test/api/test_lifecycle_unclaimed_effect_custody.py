"""Main integration keeps native absence custody through the whole effect."""

import json
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError,
    WorktreeLifecycleStore,
)


def fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = repo / "worktrees" / "candidate"
    workspace.mkdir(parents=True)
    return WorktreeLifecycleStore(repo), workspace


@pytest.mark.parametrize("kind", ["malformed", "dangling_symlink"])
def test_existing_unreadable_record_never_confers_absence(tmp_path, kind):
    store, workspace = fixture(tmp_path)
    record = store.workspace_path_for(workspace)
    record.parent.mkdir(parents=True, exist_ok=True)
    if kind == "malformed":
        record.write_bytes(b"{incomplete native record")
    else:
        record.symlink_to(record.parent / "missing-original")
    before = record.lstat()
    effects = []
    with pytest.raises(OwnershipError, match="appeared before effect"):
        store.run_effect_if_unclaimed(workspace, effect=lambda: effects.append(True))
    assert effects == []
    after = record.lstat()
    assert (before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_mtime_ns, after.st_ctime_ns
    )
    if kind == "malformed":
        assert record.read_bytes() == b"{incomplete native record"
    else:
        assert record.is_symlink() and not record.exists()


CLAIM = """import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
s=WorktreeLifecycleStore(Path(sys.argv[1]));w=Path(sys.argv[2])
print('ready',flush=True)
r=s.begin_preparing(task_id='TEST-001',canonical_task_cid='cid:test-001',attempt=1,
    lane_id='fixture-lane',workspace_path=w,branch='implementation/test-001',merge_target='main')
print(json.dumps(r.to_dict()),flush=True)
"""


def test_native_claim_waits_until_absence_effect_releases(tmp_path):
    store, workspace = fixture(tmp_path)
    child = None

    def effect():
        nonlocal child
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", CLAIM, store.repo_root, str(workspace)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.25)
        assert store.load_workspace(workspace) is None
        return "effect completed"

    try:
        assert store.run_effect_if_unclaimed(workspace, effect=effect) == "effect completed"
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        assert store.load_workspace(workspace).to_dict() == json.loads(output)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_effect_exception_releases_absence_lock_without_publishing_record(tmp_path):
    store, workspace = fixture(tmp_path)
    error = RuntimeError("effect refused")

    def refuse():
        raise error

    with pytest.raises(RuntimeError) as caught:
        store.run_effect_if_unclaimed(workspace, effect=refuse)
    assert caught.value is error
    assert store.load_workspace(workspace) is None
    assert store.run_effect_if_unclaimed(workspace, effect=lambda: "retry") == "retry"
