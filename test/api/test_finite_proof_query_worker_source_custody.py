"""Native linked-allocation source custody at the owner broker boundary.

Claims, signed proof scope, Portal projection, lifecycle allocation, Git and
kernel owner birth are genuine. Deployment launcher and sudo cleanup remain
explicit host fixtures; these tests do not launch an isolated coding child.
"""
from dataclasses import replace
import hashlib
import json
import stat
import time
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_source_custody as original
from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_source_custody as custody
from test.api.test_finite_integer_codebase import finite_tools, finite_git  # noqa: F401
from test.api.test_finite_proof_query_worktree_allocation import _native_proof_query_allocation_case


@pytest.fixture(scope="module")
def native_worker_source_case(tmp_path_factory, finite_tools):
    patch = pytest.MonkeyPatch()
    # Git creates an ordinary owner single-link marker at 0664 under the host
    # group-sharing umask. Capture that real mode before testing delegation.
    original_umask = os.umask(0o002)
    try:
        with _native_proof_query_allocation_case(tmp_path_factory.mktemp("native-worker-source"),
                finite_tools, patch, gc_before_allocation=True) as case:
            os.umask(original_umask)
            yield case
    finally:
        os.umask(original_umask)
        patch.undo()


def _check(case):
    with case["native"].server._lock, case["closure"]._catalog.store._lock:
        return custody.FrozenWorkerSourceCustody.require_current(
            case["worker_source_custody"], lambda: 90.0)


def _close(case):
    return execution.FrozenFiniteProofQueryExecutionClosure.require_worker_dispatch_current(
        case["closure"], scope=case["scope"], worker_source_custody=case["worker_source_custody"])


# Finish the three independent native constructors before the shared live
# capability exists; their proof setup must not age its authored 300s policy.
@pytest.mark.parametrize("control", ["canonical-source", "canonical-config", "linked-common-dir"])
def test_original_source_and_bidirectional_git_controls_stay_strict_with_genuine_stop(
        tmp_path, finite_tools, monkeypatch, control):
    with _native_proof_query_allocation_case(tmp_path / control, finite_tools, monkeypatch) as case:
        if control == "canonical-source":
            path, changed = case["repository"] / "calc.py", b"def f(n):\n    return n + 7\n"
            expected = "canonical worker source or selected CAS physical witness"
        elif control == "canonical-config":
            path = case["repository"] / ".git/config"
            changed = path.read_bytes() + b"\n[core]\n\tignorecase = true\n"
            expected = "canonical Git identity control bytes/mode/inode"
        else:
            path = Path(_check(case)["git_administration_path"]) / "commondir"
            changed, expected = b"../../foreign\n", "linked Git common directory"
        before, mode = path.read_bytes(), path.stat().st_mode & 0o777
        path.chmod(0o600); path.write_bytes(changed); path.chmod(mode)
        try:
            with pytest.raises(custody.WorkerSourceCustodyError, match=expected):
                _close(case)
            stopped = case["runtime"].stop()
            assert stopped.succeeded is True
            assert case["scope"]._spawned is False
            assert case["runtime"]._context_refresh_stopped() is True
        finally:
            # Restore Git bytes for genuine native teardown. The original
            # physical witness stays stale; STOP does not consult it.
            path.chmod(0o600); path.write_bytes(before); path.chmod(mode)


def test_real_gc_then_native_allocation_has_worker_only_full_proof_closure(native_worker_source_case):
    case = native_worker_source_case
    cap, body = case["worker_source_custody"], _check(case)
    assert type(cap) is custody.FrozenWorkerSourceCustody and cap._scope is case["scope"]
    assert body["profile"] == custody.PROFILE
    assert body["allocation_cid"] == case["allocation"].material_binding["allocation_cid"]
    assert body["branch"] == "refs/heads/" + case["branch"]
    assert body["baseline_source_commit"] == case["manifest"]["payload"]["baseline_commit"]
    assert (case["repository"] / ".git/info/refs").is_file()
    assert body["model_off"] is True and body["training_steps"] == 0
    assert all(body[name] is False for name in custody._FALSE)
    _close(case)
    with pytest.raises(original.SourceCustodyError, match="worktrees"):
        case["scope"]._physical_source()


def test_real_git_group_sharing_umask_marker_keeps_owner_single_link_and_exact_custody(native_worker_source_case):
    case = native_worker_source_case
    marker = case["workspace"] / ".git"
    info = marker.lstat()
    assert info.st_uid == os.geteuid() and info.st_nlink == 1
    assert info.st_mode & 0o7777 == 0o664
    body = _check(case)
    assert body["git_marker"] == case["worker_source_custody"]._marker.material()
    _close(case)


def test_genuine_owner_revalidation_preserves_equal_path_and_worker_source_material(native_worker_source_case):
    case, owner = native_worker_source_case, native_worker_source_case["scope"]._owner
    before = _check(case)
    original_path = owner.repository
    # This is the real native producer's normalization hook; its full old Git
    # custody remains strict and rejects a linked checkout after allocation.
    owner.__post_init__()
    assert owner.repository == original_path and owner.repository is not original_path
    custody._assert_baseline(case["source_baseline"])
    assert _check(case) == before
    _close(case)


def test_different_existing_repository_path_cannot_rebind_native_worker_source_scope(native_worker_source_case):
    case, owner = native_worker_source_case, native_worker_source_case["scope"]._owner
    original_path = owner.repository
    assert original_path.parent.is_dir() and original_path.parent != original_path
    try:
        object.__setattr__(owner, "repository", original_path.parent)
        with pytest.raises(custody.WorkerSourceCustodyError, match="scope was rebound: canonical_repository"):
            _check(case)
    finally:
        object.__setattr__(owner, "repository", original_path)
    _check(case)
    _close(case)


def _retain_live_reflog_stage(case, destination, phase):
    """Bounded current physical log between exact native class guards."""
    path = case["worker_source_custody"]._admin / "logs/HEAD"
    checkpoint = lambda: 90.0
    def witness(info):
        return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink]
    with case["native"].server._lock, case["closure"]._catalog.store._lock:
        material = custody.FrozenWorkerSourceCustody.require_current(
            case["worker_source_custody"], checkpoint)
        closure_return = execution.FrozenFiniteProofQueryExecutionClosure.require_worker_dispatch_current(
            case["closure"], scope=case["scope"], worker_source_custody=case["worker_source_custody"])
        before_stat = path.lstat()
        raw = path.read_bytes()
        after_stat = path.lstat()
        assert witness(before_stat) == witness(after_stat) and stat.S_ISREG(after_stat.st_mode)
        assert len(raw) <= 64 * 1024 and after_stat.st_uid == os.geteuid() and after_stat.st_nlink == 1
        retained = destination / (phase + ".bin")
        retained.write_bytes(raw); retained.chmod(0o400)
        assert retained.read_bytes() == raw
        repeated_material = custody.FrozenWorkerSourceCustody.require_current(
            case["worker_source_custody"], checkpoint)
        repeated_closure_return = execution.FrozenFiniteProofQueryExecutionClosure.require_worker_dispatch_current(
            case["closure"], scope=case["scope"], worker_source_custody=case["worker_source_custody"])
        assert repeated_material == material and closure_return is repeated_closure_return is None
        return raw, {"phase": phase, "captured_utc_ns": time.time_ns(), "source_path": str(path),
            "retained_path": str(retained), "full_stat": witness(after_stat), "owner_euid": os.geteuid(),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "source_current_class_return": material, "source_repeat_class_return": repeated_material,
            "proof_query_closure_class_return": closure_return,
            "proof_query_closure_repeat_class_return": repeated_closure_return,
            "genuine_native_guards_closed_before_and_after_physical_retention": True}


def test_real_ref_packing_and_gc_preserve_original_and_one_allocated_ref(native_worker_source_case, tmp_path):
    case = native_worker_source_case
    before = _check(case)
    finite_git(case["repository"], "pack-refs", "--all", "--prune")
    assert _check(case) == before
    destination = tmp_path / "actual-live-linked-reflog-gc"
    destination.mkdir(mode=0o700)
    before_raw, before_stage = _retain_live_reflog_stage(case, destination, "before")
    finite_git(case["repository"], "gc", "--prune=now")
    assert _check(case) == before
    after_raw, after_stage = _retain_live_reflog_stage(case, destination, "after")
    assert before_stage["full_stat"][1] != after_stage["full_stat"][1]
    assert before_stage["full_stat"][2:5] == after_stage["full_stat"][2:5]
    assert custody._reflog_rows(before_raw, case["source_baseline"]._commit) == custody._reflog_rows(after_raw, case["source_baseline"]._commit)
    receipt = {"schema": "host-native-actual-linked-reflog-gc-retention@1",
        "scope": str(case["scope"]._output / "execution-scope.json"),
        "native_initial_custody_receipt": str(case["initial_live_custody_receipt"]),
        "before": before_stage, "after": after_stage, "model_off": True, "training_steps": 0,
        "historical_only": True, "current_dispatch_authority": False,
        "isolated_worker_birth_qualified": False, "proof_authority": False,
        "completion_authority": False, "global_convergence_proved": False}
    receipt_raw = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert len(receipt_raw) <= 256 * 1024
    receipt_path = destination / "receipt.json"
    receipt_path.write_bytes(receipt_raw); receipt_path.chmod(0o400)
    finite_git(case["repository"], "update-server-info")
    assert _check(case) == before
    _close(case)


def test_complete_ref_map_refuses_extra_ref_and_wrong_allocated_branch_oid(native_worker_source_case):
    case = native_worker_source_case
    repository = case["repository"]
    extra = "refs/heads/foreign-source"
    finite_git(repository, "update-ref", extra, case["manifest"]["payload"]["baseline_commit"])
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="ref meaning"):
            _check(case)
    finally:
        finite_git(repository, "update-ref", "-d", extra)
    assert _check(case)["allocation_cid"] == case["allocation"].material_binding["allocation_cid"]
    branch = "refs/heads/" + case["branch"]
    path = repository / ".git" / branch
    had_loose = path.exists()
    before = path.read_bytes() if had_loose else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0" * len(case["manifest"]["payload"]["baseline_commit"]) + "\n")
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="references disagree|ref meaning"):
            _check(case)
    finally:
        if had_loose:
            path.write_bytes(before)
        else:
            path.unlink()
    _close(case)


def test_advertisement_cache_cannot_drop_original_refs(native_worker_source_case):
    case = native_worker_source_case
    path = case["repository"] / ".git/info/refs"
    before = path.read_bytes()
    path.write_bytes(b"")
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="advertised refs"):
            _check(case)
    finally:
        path.write_bytes(before)
    _close(case)


def test_actual_staged_index_change_refuses_without_relying_on_index_stat_cache(native_worker_source_case, tmp_path):
    case = native_worker_source_case
    path = case["repository"] / ".git/index"
    before, mode = path.read_bytes(), path.stat().st_mode & 0o777
    changed = tmp_path / "different-source.py"
    changed.write_text("def f(n):\n    return n + 9\n")
    blob = finite_git(case["repository"], "hash-object", "-w", str(changed)).strip()
    finite_git(case["repository"], "update-index", "--cacheinfo", "100644," + blob + ",calc.py")
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="staged Git identity"):
            _check(case)
    finally:
        path.write_bytes(before); path.chmod(mode)
    _close(case)


def test_allocated_bytes_marker_inode_and_extra_allocation_are_closed(native_worker_source_case):
    case = native_worker_source_case
    path = case["workspace"] / "calc.py"
    before = path.read_bytes()
    path.write_text("def f(n):\n    return n + 8\n")
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="allocated tracked source bytes"):
            _check(case)
    finally:
        path.write_bytes(before)
    marker = case["workspace"] / ".git"
    saved = marker.with_name(".git-saved")
    before, mode = marker.read_bytes(), marker.stat().st_mode & 0o777
    marker.rename(saved); marker.write_bytes(before); marker.chmod(mode)
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="marker bytes, inode"):
            _check(case)
    finally:
        marker.unlink(); saved.rename(marker)
    extra = Path(_check(case)["git_administration_path"]).parent / "unregistered"
    extra.mkdir()
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="additional linked worktree"):
            _check(case)
    finally:
        extra.rmdir()
    _close(case)


def test_delegated_marker_mode_has_exact_owner_inode_bytes_and_no_executable_or_world_write(native_worker_source_case):
    case = native_worker_source_case
    path = case["workspace"] / ".git"
    before_mode = path.stat().st_mode & 0o777
    path.chmod(0o440)
    try:
        _close(case)
        path.chmod(0o777)
        with pytest.raises(custody.WorkerSourceCustodyError, match="owner single-link"):
            _check(case)
    finally:
        path.chmod(before_mode)
    _close(case)


@pytest.mark.parametrize("field", ["_branch", "_admin", "_files", "_allocation_material", "_reflog"])
def test_genuine_owned_cached_custody_material_cannot_rebind(native_worker_source_case, field):
    case = native_worker_source_case
    cap = case["worker_source_custody"]
    before = getattr(cap, field)
    if field == "_files":
        changed = ((before[0][0], replace(before[0][1], sha="0" * 64)), *before[1:])
    elif field == "_allocation_material":
        changed = b"{}"
    elif field == "_admin":
        changed = before.with_name("foreign-administration")
    elif field == "_reflog":
        changed = b"\n".join(line if b"\t" in line else line + b"\t"
            for line in before[:-1].split(b"\n")) + b"\n"
        assert changed != before
        assert custody._reflog_rows(changed, case["source_baseline"]._commit) == custody._reflog_rows(before, case["source_baseline"]._commit)
    else:
        changed = "refs/heads/foreign"
    object.__setattr__(cap, field, changed)
    try:
        with pytest.raises(custody.WorkerSourceCustodyError, match="material changed|closed material|registered branch|cached original linked Git reflog"):
            _check(case)
    finally:
        object.__setattr__(cap, field, before)
    _close(case)


@pytest.mark.parametrize("control", ["message", "mode", "extra-row", "omitted-row", "hard-link"])
def test_complete_reflog_rows_and_native_owner_single_link_mode_stay_strict(native_worker_source_case, tmp_path, control):
    case = native_worker_source_case
    path = Path(_check(case)["git_administration_path"]) / "logs/HEAD"
    before, info = path.read_bytes(), path.lstat()
    mode, inode = info.st_mode & 0o7777, info.st_ino
    rows = before.splitlines(keepends=True)
    assert len(rows) == 2
    # Keep the real extra inode link outside the exact Git admin inventory;
    # this control must reach the native nlink check rather than layout refusal.
    assert tmp_path.lstat().st_dev == info.st_dev
    alias = tmp_path / "extra-owner-link"
    try:
        if control == "message":
            changed = before.replace(b"reset: moving to HEAD", b"reset: moving to foreign")
            assert changed != before
            path.write_bytes(changed)
        elif control == "mode":
            path.chmod(0o600 if mode != 0o600 else 0o640)
        elif control == "extra-row":
            path.write_bytes(before + rows[-1])
        elif control == "omitted-row":
            path.write_bytes(rows[0])
        else:
            alias.hardlink_to(path)
            assert path.lstat().st_uid == os.geteuid() and path.lstat().st_nlink == 2
        with pytest.raises(custody.WorkerSourceCustodyError,
                match="reflog row meaning|Git control owner, link count or mode"):
            _check(case)
    finally:
        if alias.exists():
            alias.unlink()
        path.write_bytes(before)
        path.chmod(mode)
    assert path.lstat().st_ino == inode and path.lstat().st_uid == os.geteuid() and path.lstat().st_nlink == 1
    _check(case)
    _close(case)


