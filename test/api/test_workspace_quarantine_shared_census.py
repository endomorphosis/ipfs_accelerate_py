"""A board's complete retained population is separate from shared Git traffic."""

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as q
from ipfs_accelerate_py.agent_supervisor.task_sources.owner_task_quarantine import (
    QuarantineDenied,
)
from test.api.test_workspace_root_quarantine import seed


def lifecycle_record(lifecycle, original, workspace, *, ordinal=1):
    """Write the native record shape without starting thousands of Git worktrees."""
    value = replace(
        original,
        workspace_path=str(workspace),
        task_id=f"foreign-{ordinal}",
        record_id=original.compute_record_id_for(
            canonical_task_cid=original.canonical_task_cid,
            task_id=f"foreign-{ordinal}",
            attempt=original.attempt,
            workspace_path=str(workspace),
        ),
    )
    path = lifecycle.workspace_path_for(workspace)
    path.write_text(json.dumps(value.to_dict()))
    return path


def test_more_than_8192_foreign_native_records_preserve_exact_board_custody(tmp_path):
    repo, root, _, lease, lifecycle, original = seed(tmp_path)
    before = q.census(repo, root)
    foreign = tmp_path / "other-board-workspaces"
    for index in range(8193):
        lifecycle_record(lifecycle, original, foreign / str(index), ordinal=index)
    # Foreign metadata exceeds the retained-root budget but does not disappear
    # from the scan. The final installed fence remains exactly this board's.
    assert q.census(repo, root) == before
    frozen = q.freeze(repo, root, expected=before)
    assert frozen["snapshot"] == before
    assert q.verify(repo, root) == frozen
    assert lease.path.exists()
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        with q.mutation(repo, lease.path):
            pytest.fail("old workspace became writable")


def test_relevant_population_still_refuses_overflow(tmp_path, monkeypatch):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    baseline = q.census(repo, root)
    monkeypatch.setattr(q, "MAX_FILES", len(baseline["files"]) + 1)
    lifecycle_record(lifecycle, original, root / "extra-1", ordinal=1)
    lifecycle_record(lifecycle, original, root / "extra-2", ordinal=2)
    with pytest.raises(QuarantineDenied, match="workspace_quarantine_population_bound"):
        q.plan(repo, root)


def test_foreign_bytes_do_not_consume_retained_byte_budget(tmp_path, monkeypatch):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    baseline = q.census(repo, root)
    monkeypatch.setattr(q, "MAX_BYTES", sum(row["size"] for row in baseline["files"]))
    for index in range(8):
        lifecycle_record(
            lifecycle, original, tmp_path / "foreign" / str(index), ordinal=index
        )
    assert q.census(repo, root) == baseline
    lifecycle_record(lifecycle, original, root / "relevant-overflow")
    with pytest.raises(QuarantineDenied, match="workspace_quarantine_snapshot_bound"):
        q.census(repo, root)


@pytest.mark.parametrize(
    "damage",
    ["malformed", "missing_scope", "relative_scope", "bad_record_id", "symlink"],
)
def test_foreign_looking_shared_metadata_cannot_hide_ambiguous_scope(tmp_path, damage):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    path = lifecycle_record(lifecycle, original, tmp_path / "foreign")
    value = json.loads(path.read_text())
    if damage == "malformed":
        path.write_text("{incomplete")
    elif damage == "symlink":
        other = tmp_path / "metadata-target"
        path.rename(other)
        path.symlink_to(other)
    else:
        if damage == "missing_scope":
            value.pop("workspace_path")
        elif damage == "relative_scope":
            value["workspace_path"] = "../possibly-this-board"
        else:
            value["record_id"] = "wrong"
        path.write_text(json.dumps(value))
    with pytest.raises((QuarantineDenied, ValueError, OSError)):
        q.census(repo, root)


@pytest.mark.parametrize(
    "change",
    [
        "foreign_becomes_relevant",
        "foreign_replaced",
        "new_relevant_record",
        "new_pool_entry",
    ],
)
def test_changes_after_scope_read_refuse_the_complete_census(
    tmp_path, monkeypatch, change
):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    foreign = lifecycle_record(lifecycle, original, tmp_path / "foreign")
    read_done, write_done = threading.Event(), threading.Event()
    writer_errors = []
    native_read = q.read_regular

    def paused_read(path, **kwargs):
        result = native_read(path, **kwargs)
        if path == foreign:
            read_done.set()
            assert write_done.wait(5), "writer did not finish"
        return result

    def writer():
        try:
            assert read_done.wait(5), "reader did not reach foreign metadata"
            if change == "foreign_becomes_relevant":
                value = json.loads(foreign.read_text())
                value["workspace_path"] = str(root / "newly-relevant")
                foreign.write_text(json.dumps(value))
            elif change == "foreign_replaced":
                replacement = foreign.with_suffix(".replacement")
                replacement.write_bytes(foreign.read_bytes())
                replacement.replace(foreign)
            elif change == "new_relevant_record":
                lifecycle_record(lifecycle, original, root / "new-record")
            else:
                # This directory's enumeration finished before the scope read.
                pool = root / ".pool-state"
                (pool / "late-entry.json").write_text("{}")
        except BaseException as error:
            writer_errors.append(error)
        finally:
            write_done.set()

    monkeypatch.setattr(q, "read_regular", paused_read)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        with pytest.raises(
            QuarantineDenied, match="workspace_quarantine_census_changed"
        ):
            q.census(repo, root)
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert writer_errors == []


def test_shared_scanner_has_its_own_strict_entry_bound(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    repo, root, _, _, lifecycle, _ = seed(tmp_path)
    pool_entries = len(list((root / ".pool-state").iterdir()))
    monkeypatch.setattr(q, "MAX_CENSUS_SCAN_FILES", pool_entries + 3)
    original_scandir = q.os.scandir
    consumed, closed = [], []

    @contextmanager
    def observed_scandir(directory):
        if Path(directory) != lifecycle.store_dir:
            with original_scandir(directory) as entries:
                yield entries
            return

        def endless():
            for index in range(10000):
                assert index <= 3, "scanner read beyond its first excess entry"
                consumed.append(index)
                yield SimpleNamespace(
                    path=str(lifecycle.store_dir / f"ignored-{index}.other")
                )

        try:
            yield endless()
        finally:
            closed.append(True)

    monkeypatch.setattr(q.os, "scandir", observed_scandir)
    with pytest.raises(
        QuarantineDenied, match="workspace_quarantine_scan_population_bound"
    ):
        q.census(repo, root)
    assert consumed == [0, 1, 2, 3]
    assert closed == [True]


def test_foreign_scope_symlink_cannot_move_into_board_after_classification(
    tmp_path, monkeypatch
):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    foreign_directory = tmp_path / "foreign-directory"
    foreign_directory.mkdir()
    scope_link = tmp_path / "scope-link"
    scope_link.symlink_to(foreign_directory, target_is_directory=True)
    foreign_workspace = scope_link / "workspace"
    lifecycle_record(lifecycle, original, foreign_workspace)
    native_scope = q._census_scope
    read_done, write_done = threading.Event(), threading.Event()
    writer_errors = []

    def paused_scope(path, observed):
        resolved = native_scope(path, observed)
        if path == foreign_workspace:
            read_done.set()
            assert write_done.wait(5)
        return resolved

    def writer():
        try:
            assert read_done.wait(5)
            replacement = tmp_path / "replacement-link"
            replacement.symlink_to(root, target_is_directory=True)
            replacement.replace(scope_link)
        except BaseException as error:
            writer_errors.append(error)
        finally:
            write_done.set()

    monkeypatch.setattr(q, "_census_scope", paused_scope)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        with pytest.raises(
            QuarantineDenied, match="workspace_quarantine_census_changed"
        ):
            q.census(repo, root)
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert writer_errors == []


def test_shared_scanner_has_its_own_byte_bound(tmp_path, monkeypatch):
    repo, root, _, _, lifecycle, original = seed(tmp_path)
    baseline = q.census(repo, root)
    foreign = lifecycle_record(lifecycle, original, tmp_path / "foreign")
    monkeypatch.setattr(
        q,
        "MAX_CENSUS_SCAN_BYTES",
        sum(row["size"] for row in baseline["files"]) + foreign.stat().st_size - 1,
    )
    with pytest.raises(QuarantineDenied):
        q.census(repo, root)
