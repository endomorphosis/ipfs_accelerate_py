"""Real native lifecycle inodes and independent process crash/replay tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import (
    worktree_lifecycle_delete_journal as journal,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
)

HANDOFF = "sha256:" + "a" * 64
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir(mode=0o775)
    repo.chmod(0o775)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    store = WorktreeLifecycleStore(repo)
    prior = store.begin_preparing(
        task_id="TASK-1",
        canonical_task_cid="task:exact:one",
        attempt=1,
        lane_id="lane-0",
        workspace_path=tmp_path / "workspace",
        branch="implementation/exact",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        lease_id="lease:exact",
    )
    store.store_dir.chmod(0o775)
    terminal = store.mark_terminal(
        prior.workspace_path,
        lease_id=prior.lease_id,
        expected_fence=prior.fence,
        reason="worktree_cleaned",
    )
    record = store.workspace_path_for(terminal.workspace_path)
    index = store.task_index_path_for(
        canonical_task_cid=terminal.canonical_task_cid,
        task_id=terminal.task_id,
        attempt=terminal.attempt,
    )
    return store, terminal, record, index


def _journal(store):
    return store.store_dir / journal.JOURNAL_DIR


def _snapshot(directory):
    return {
        str(p.relative_to(directory)): (
            p.stat().st_ino,
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            p.read_bytes(),
        )
        for p in directory.rglob("*")
        if p.is_file()
    }


def _write_private(path, raw):
    path.write_bytes(raw)
    path.chmod(0o600)


def _child_crash(native, tmp_path, stage):
    store, terminal, _, _ = native
    config = tmp_path / "terminal.json"
    config.write_text(
        json.dumps(
            {
                "repo": str(store.repo_root),
                "store": str(store.store_dir),
                "terminal": terminal.to_dict(),
            }
        )
    )
    script = r"""
import json,os,sys
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle_delete_journal as j
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore,WorkspaceLifecycleRecord
v=json.load(open(sys.argv[2])); stage=sys.argv[3]
s=WorktreeLifecycleStore(v['repo'],store_dir=v['store']); record=WorkspaceLifecycleRecord.from_dict(v['terminal'])
original_publish=j._publish; original_move=j._move_without_replace; original_directory_publish=j._publish_without_replace_at

def publish(fd,name,value):
    identity=original_publish(fd,name,value)
    if (stage=='prepared' and name.endswith('.prepared.json')) or (stage=='committed' and name.endswith('.committed.json')):
        os._exit(73)
    return identity

def move(fd,source,target_fd,target):
    original_move(fd,source,target_fd,target)
    if (stage=='first_move' and target.endswith('.record')) or (stage=='second_move' and target.endswith('.index')):
        os._exit(73)

def directory_publish(fd,source,target):
    original_directory_publish(fd,source,target)
    if stage=='provision' and target==j.JOURNAL_DIR:
        os._exit(73)

j._publish=publish;j._move_without_replace=move;j._publish_without_replace_at=directory_publish
s.delete_candidate_observed(record,handoff_receipt_id='sha256:'+'a'*64)
raise AssertionError('crash boundary not reached')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT), str(config), stage],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 73, result.stderr


def test_actual_delete_retains_original_inodes_and_observer_never_writes(
    native, monkeypatch
):
    store, terminal, record, index = native
    original = {"record": record.stat().st_ino, "index": index.stat().st_ino}
    assert store.repo_root.stat().st_mode & 0o777 == 0o775
    assert store.store_dir.stat().st_mode & 0o777 == 0o775
    observed = store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    evidence = observed.to_dict()
    assert store.repo_root.stat().st_mode & 0o777 == 0o775
    assert store.store_dir.stat().st_mode & 0o777 == 0o775
    assert _journal(store).stat().st_mode & 0o777 == 0o700
    assert evidence["committed"]["completion_authority"] is False
    assert not record.exists() and not index.exists()
    for role, inode in original.items():
        assert evidence["committed"]["retained"][role]["identity"]["ino"] == inode
        retained = next(_journal(store).glob(f"candidate-delete-*.{role}"))
        assert retained.stat().st_ino == inode
    before = _snapshot(store.store_dir)
    with monkeypatch.context() as m:
        m.setattr(journal.os, "fsync", lambda *_: pytest.fail("observer may not fsync"))
        m.setattr(
            journal, "_publish", lambda *_: pytest.fail("observer may not publish")
        )
        m.setattr(
            journal,
            "_move_without_replace",
            lambda *_: pytest.fail("observer may not move"),
        )
        assert (
            store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF)
            == observed
        )
    assert _snapshot(store.store_dir) == before
    assert (
        store.resume_candidate_observed_delete(terminal, handoff_receipt_id=HANDOFF)
        == observed
    )
    assert (
        store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
        == observed
    )
    assert _snapshot(store.store_dir) == before


@pytest.mark.parametrize(
    "stage", ["prepared", "first_move", "second_move", "committed"]
)
def test_real_process_crash_recovers_only_prepared_exact_native_inodes(
    native, tmp_path, stage
):
    store, terminal, record, index = native
    original = record.stat().st_ino, index.stat().st_ino
    _child_crash(native, tmp_path, stage)
    restarted = WorktreeLifecycleStore(store.repo_root)
    if stage != "committed":
        assert (
            restarted.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF)
            is None
        )
    result = restarted.resume_candidate_observed_delete(
        terminal, handoff_receipt_id=HANDOFF
    )
    assert not record.exists() and not index.exists()
    receipt = result.to_dict()["committed"]
    assert (
        tuple(receipt["retained"][r]["identity"]["ino"] for r in ("record", "index"))
        == original
    )
    assert (
        restarted.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF)
        == result
    )
    assert (
        restarted.resume_candidate_observed_delete(terminal, handoff_receipt_id=HANDOFF)
        == result
    )
    assert len(list(_journal(store).glob("*.committed.json"))) == 1


def test_absence_has_no_migration_or_delete_authority(native):
    store, terminal, record, index = native
    record.unlink()
    index.unlink()
    before = _snapshot(store.store_dir)
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) is None
    )
    for method in (
        store.delete_candidate_observed,
        store.resume_candidate_observed_delete,
    ):
        with pytest.raises(journal.CandidateDeletionUnverified):
            method(terminal, handoff_receipt_id=HANDOFF)
    assert _snapshot(store.store_dir) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("lease_id", "lease:foreign"),
        ("fence", 99),
        ("canonical_task_cid", "task:foreign"),
        ("repo_root", "/foreign/repository"),
    ],
)
def test_foreign_binding_never_reuses_committed_native_receipt(native, field, value):
    store, terminal, _, _ = native
    store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    foreign = replace(terminal, **{field: value}, record_id="")
    before = _snapshot(store.store_dir)
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.resume_candidate_observed_delete(foreign, handoff_receipt_id=HANDOFF)
    assert _snapshot(store.store_dir) == before


def test_foreign_handoff_cannot_rebind(native):
    store, terminal, _, _ = native
    store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    before = _snapshot(store.store_dir)
    assert (
        store.observe_candidate_deletion(
            terminal, handoff_receipt_id="sha256:" + "b" * 64
        )
        is None
    )
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.resume_candidate_observed_delete(
            terminal, handoff_receipt_id="sha256:" + "b" * 64
        )
    assert _snapshot(store.store_dir) == before


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "foreign_inode",
        "changed_bytes",
        "changed_mode",
        "canonical_replacement",
        "symlink",
        "fifo",
        "oversized",
    ],
)
def test_committed_evidence_never_accepts_missing_or_foreign_state(native, mode):
    store, terminal, record, _ = native
    store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    retained = next(_journal(store).glob("candidate-delete-*.record"))
    raw = retained.read_bytes()
    if mode == "missing":
        retained.unlink()
    elif mode == "foreign_inode":
        foreign = retained.with_suffix(".foreign")
        _write_private(foreign, raw)
        os.replace(foreign, retained)
    elif mode == "changed_bytes":
        _write_private(retained, raw + b" ")
    elif mode == "changed_mode":
        retained.chmod(0o640)
    elif mode == "canonical_replacement":
        _write_private(record, raw)
    elif mode == "symlink":
        retained.unlink()
        retained.symlink_to(record)
    elif mode == "fifo":
        retained.unlink()
        os.mkfifo(retained, 0o600)
    else:
        _write_private(retained, b"x" * (journal.MAX_BYTES + 1))
    for method in (
        store.observe_candidate_deletion,
        store.resume_candidate_observed_delete,
    ):
        with pytest.raises(journal.CandidateDeletionUnverified):
            method(terminal, handoff_receipt_id=HANDOFF)
    if mode == "canonical_replacement":
        assert record.read_bytes() == raw


@pytest.mark.parametrize("stage", ["prepared", "first_move", "second_move"])
def test_partial_journal_missing_native_inode_does_not_infer_own_removal(
    native, tmp_path, stage
):
    store, terminal, record, _ = native
    _child_crash(native, tmp_path, stage)
    target = (
        record
        if stage == "prepared"
        else next(_journal(store).glob("candidate-delete-*.record"))
    )
    target.unlink()
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.resume_candidate_observed_delete(terminal, handoff_receipt_id=HANDOFF)
    assert not list(_journal(store).glob("*.committed.json"))


@pytest.mark.parametrize(
    "boundary",
    [
        "prepared_file",
        "prepared_directory",
        "record_directory",
        "index_directory",
        "commit_file",
        "commit_directory",
    ],
)
def test_fsync_failure_never_returns_success_and_exact_retry_can_recover(
    native, monkeypatch, boundary
):
    store, terminal, _, _ = native
    original = journal.os.fsync
    failed = []

    def fsync(fd):
        name = os.readlink(f"/proc/self/fd/{fd}")
        prepared = list(_journal(store).glob("*.prepared.json"))
        record = list(_journal(store).glob("candidate-delete-*.record"))
        index = list(_journal(store).glob("candidate-delete-*.index"))
        committed = list(_journal(store).glob("*.committed.json"))
        is_directory = Path(name) in (store.store_dir, _journal(store))
        matches = {
            "prepared_file": not is_directory
            and Path(name).parent == _journal(store)
            and not prepared,
            "prepared_directory": is_directory and prepared and not record,
            "record_directory": is_directory and record and not index,
            "index_directory": is_directory and index and not committed,
            "commit_file": not is_directory
            and Path(name).parent == _journal(store)
            and index
            and not committed,
            "commit_directory": is_directory and committed,
        }
        if not failed and matches[boundary]:
            failed.append(boundary)
            raise OSError("fixture fsync failed")
        return original(fd)

    with monkeypatch.context() as m:
        m.setattr(journal.os, "fsync", fsync)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert failed == [boundary]
    # The creating call is required if publication failed before an intent existed.
    result = store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) == result
    )


def test_native_rename_collision_preserves_foreign_entry_and_original(
    native, monkeypatch
):
    store, terminal, record, _ = native
    original = journal._move_without_replace
    captured = {}

    def collide(fd, source, target_fd, target):
        if target.endswith(".record"):
            path = _journal(store) / target
            _write_private(path, b'{"foreign":true}\n')
            captured["path"] = path
            captured["inode"] = path.stat().st_ino
        return original(fd, source, target_fd, target)

    inode, raw = record.stat().st_ino, record.read_bytes()
    with monkeypatch.context() as m:
        m.setattr(journal, "_move_without_replace", collide)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert record.stat().st_ino == inode and record.read_bytes() == raw
    assert captured["path"].stat().st_ino == captured["inode"]
    assert captured["path"].read_bytes() == b'{"foreign":true}\n'
    assert not list(_journal(store).glob("*.committed.json"))


def test_observer_is_noncreating_before_any_deletion(native):
    store, terminal, _, _ = native
    before = _snapshot(store.store_dir)
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) is None
    )
    assert _snapshot(store.store_dir) == before


def test_foreign_empty_private_child_is_not_implicitly_adopted(native):
    store, terminal, record, index = native
    _journal(store).mkdir(mode=0o700)
    before = _snapshot(store.store_dir)
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert record.exists() and index.exists()
    assert _snapshot(store.store_dir) == before


@pytest.mark.parametrize("kind", ["writable", "symlink", "file", "fifo"])
def test_unsafe_existing_private_child_preserves_native_rows(native, kind):
    store, terminal, record, index = native
    child = _journal(store)
    original = {p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)}
    if kind == "writable":
        child.mkdir(mode=0o770)
        child.chmod(0o770)
    elif kind == "symlink":
        child.symlink_to(store.repo_root, target_is_directory=True)
    elif kind == "file":
        _write_private(child, b'{"foreign":true}\n')
    else:
        os.mkfifo(child, 0o600)
    for method in (store.observe_candidate_deletion, store.delete_candidate_observed):
        with pytest.raises(journal.CandidateDeletionUnverified):
            method(terminal, handoff_receipt_id=HANDOFF)
    assert {
        p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)
    } == original


@pytest.mark.parametrize("boundary", ["store", "journal"])
def test_directory_replacement_after_preparation_cannot_redirect_native_moves(
    native, monkeypatch, boundary
):
    store, terminal, record, index = native
    originals = {p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)}
    original = journal._publish
    moved = []

    def publish(fd, name, value):
        original(fd, name, value)
        if name.endswith(".prepared.json") and not moved:
            target = store.store_dir if boundary == "store" else _journal(store)
            displaced = target.with_name(target.name + ".displaced")
            target.rename(displaced)
            target.mkdir(mode=0o775 if boundary == "store" else 0o700)
            moved.append(displaced)

    with monkeypatch.context() as m:
        m.setattr(journal, "_publish", publish)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert moved
    original_dir = moved[0] if boundary == "store" else store.store_dir
    assert {
        name: ((original_dir / name).stat().st_ino, (original_dir / name).read_bytes())
        for name in originals
    } == originals
    assert not list(store.store_dir.rglob("*.committed.json"))


def test_scope_record_cannot_be_rebound_after_private_child_replacement(native):
    store, terminal, _, _ = native
    store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    child = _journal(store)
    displaced = child.with_name(child.name + ".old")
    child.rename(displaced)
    child.mkdir(mode=0o700)
    for source in displaced.iterdir():
        _write_private(child / source.name, source.read_bytes())
    before = _snapshot(store.store_dir)
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF)
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.resume_candidate_observed_delete(terminal, handoff_receipt_id=HANDOFF)
    assert _snapshot(store.store_dir) == before


def test_resume_reasserts_durable_intent_before_any_remaining_move(
    native, tmp_path, monkeypatch
):
    store, terminal, _, _ = native
    _child_crash(native, tmp_path, "prepared")
    original_sync = journal.os.fsync
    original_move = journal._move_without_replace
    synced = []

    def fsync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return original_sync(fd)

    def move(source_fd, source, target_fd, target):
        assert any(name.endswith(".prepared.json") for name in synced)
        assert str(store.store_dir) in synced and str(_journal(store)) in synced
        return original_move(source_fd, source, target_fd, target)

    with monkeypatch.context() as m:
        m.setattr(journal.os, "fsync", fsync)
        m.setattr(journal, "_move_without_replace", move)
        result = store.resume_candidate_observed_delete(
            terminal, handoff_receipt_id=HANDOFF
        )
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) == result
    )


def test_actual_child_crash_during_private_child_publication_never_moves_originals(
    native, tmp_path
):
    store, terminal, record, index = native
    original = {p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)}
    _child_crash(native, tmp_path, "provision")
    assert {
        p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)
    } == original
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) is None
    )
    with pytest.raises(journal.CandidateDeletionUnverified):
        store.resume_candidate_observed_delete(terminal, handoff_receipt_id=HANDOFF)
    result = store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) == result
    )


def test_child_publication_collision_preserves_foreign_child_and_originals(
    native, monkeypatch
):
    store, terminal, record, index = native
    original = journal._publish_without_replace_at
    captured = []

    def collision(fd, source, target):
        if target == journal.JOURNAL_DIR:
            _journal(store).mkdir(mode=0o700)
            marker = _journal(store) / "foreign.json"
            _write_private(marker, b'{"foreign":true}\n')
            captured.append((marker.stat().st_ino, marker.read_bytes()))
        return original(fd, source, target)

    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)}
    with monkeypatch.context() as m:
        m.setattr(journal, "_publish_without_replace_at", collision)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert {
        p.name: (p.stat().st_ino, p.read_bytes()) for p in (record, index)
    } == before
    marker = _journal(store) / "foreign.json"
    assert (marker.stat().st_ino, marker.read_bytes()) == captured[0]


def test_mutating_committed_retry_reasserts_receipt_and_directory_durability(
    native, monkeypatch
):
    store, terminal, _, _ = native
    original_sync = journal.os.fsync
    failed = []

    def fail_commit_directory(fd):
        if (
            not failed
            and os.readlink(f"/proc/self/fd/{fd}") == str(_journal(store))
            and list(_journal(store).glob("*.committed.json"))
        ):
            failed.append(True)
            raise OSError("fixture committed directory fsync unavailable")
        return original_sync(fd)

    with monkeypatch.context() as m:
        m.setattr(journal.os, "fsync", fail_commit_directory)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert failed
    committed = next(_journal(store).glob("*.committed.json"))
    synced = []

    def sync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return original_sync(fd)

    with monkeypatch.context() as m:
        m.setattr(journal.os, "fsync", sync)
        result = store.resume_candidate_observed_delete(
            terminal, handoff_receipt_id=HANDOFF
        )
    assert str(committed) in synced
    assert str(_journal(store)) in synced and str(store.store_dir) in synced
    assert synced.index(str(committed)) < synced.index(str(_journal(store)))
    assert (
        store.observe_candidate_deletion(terminal, handoff_receipt_id=HANDOFF) == result
    )


@pytest.mark.parametrize("mode", ["removed", "foreign_inode", "foreign_payload"])
def test_new_commit_publication_must_reobserve_exact_file_before_return(
    native, monkeypatch, mode
):
    store, terminal, _, _ = native
    original = journal._publish
    changed = []

    def publish(fd, name, value):
        identity = original(fd, name, value)
        if name.endswith(".committed.json"):
            target = _journal(store) / name
            raw = target.read_bytes()
            if mode == "removed":
                target.unlink()
            else:
                replacement = target.with_suffix(".replacement")
                _write_private(
                    replacement,
                    raw if mode == "foreign_inode" else b'{"foreign":true}\n',
                )
                os.replace(replacement, target)
            changed.append(True)
        return identity

    with monkeypatch.context() as m:
        m.setattr(journal, "_publish", publish)
        with pytest.raises(journal.CandidateDeletionUnverified):
            store.delete_candidate_observed(terminal, handoff_receipt_id=HANDOFF)
    assert changed
