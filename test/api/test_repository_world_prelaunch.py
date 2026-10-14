"""Actual native owner gates, with authored Quack transport/deployment seams.

These checks qualify neither a worker UID transition nor a provider/container
launch. The inherited allocation fixture retains real native claims, grants,
Portal controls, Git/CAS source and worktree lifecycle custody.
"""
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_prelaunch as prelaunch
from ipfs_accelerate_py.agent_supervisor.runtime.current_source_task_admission import NativeSourceTaskAdmission
from ipfs_accelerate_py.agent_supervisor.runtime.current_source_worktree_allocation import NativeSourceWorktreeAllocation
from test.api.test_current_source_task_admission import SOURCE, _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_native_nominated_source_allocation import bind, nominated_allocation  # noqa: F401


PROMPT = "Implement the exact nominated assignment. Unicode: λ."
REFUSALS = (ValueError, RuntimeError, AttributeError, OSError)


def provision_authored_prelaunch_sources(case):
    """Trusted fixture setup before first capture, never a production repair.

    The legacy native allocation fixture follows the host umask for authored
    program files. Only its exact already-bound selected population is made
    owner-controlled here; Git/control files and native guards are untouched.
    """
    source = case.native.handle._source
    paths = source._paths
    assert type(paths) is tuple and 0 < len(paths) <= prelaunch.MAX_SOURCE_PATHS
    assert case.guard._lookup is source and source._native_source_worktree_allocation is case.guard
    count = 0
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def digest(fd, size):
        os.lseek(fd, 0, os.SEEK_SET)
        sha, remaining = hashlib.sha256(), size
        while remaining:
            block = os.read(fd, min(65536, remaining))
            assert block
            remaining -= len(block)
            sha.update(block)
        assert not os.read(fd, 1)
        return sha.digest()
    for root in (case.native.canonical, case.native.workspace):
        for raw_path in paths:
            parts = bytes.fromhex(raw_path).split(b"/")
            assert all(part and part not in {b".", b"..", b".git"} and b"\0" not in part for part in parts)
            held, fresh, fd = [prelaunch._root_fd(root)], [], None
            try:
                for part in parts[:-1]:
                    held.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=held[-1]))
                    directory = os.fstat(held[-1])
                    assert directory.st_uid == os.geteuid() and not directory.st_mode & 0o022
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held[-1])
                before = os.fstat(fd)
                assert stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1
                assert 0 <= before.st_size <= prelaunch.MAX_SELECTED_SOURCE_BYTES
                before_sha = digest(fd, before.st_size)
                assert identity(before) == identity(os.fstat(fd)) == identity(os.stat(parts[-1], dir_fd=held[-1], follow_symlinks=False))
                expected_mode = stat.S_IMODE(before.st_mode) & ~0o022
                os.fchmod(fd, expected_mode)
                after = os.fstat(fd)
                assert (after.st_dev, after.st_ino, after.st_uid, after.st_nlink, after.st_size, after.st_mtime_ns) == (
                    before.st_dev, before.st_ino, before.st_uid, before.st_nlink, before.st_size, before.st_mtime_ns)
                assert stat.S_IMODE(after.st_mode) == expected_mode
                assert after.st_mode & 0o111 == before.st_mode & 0o111
                assert digest(fd, after.st_size) == before_sha
                fresh.append(prelaunch._root_fd(root))
                for index in range(len(held)):
                    if index:
                        fresh.append(os.open(parts[index - 1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fresh[-1]))
                    old, named = os.fstat(held[index]), os.fstat(fresh[-1])
                    assert (old.st_dev, old.st_ino, old.st_uid, old.st_mode) == (named.st_dev, named.st_ino, named.st_uid, named.st_mode)
                assert identity(after) == identity(os.fstat(fd)) == identity(os.stat(parts[-1], dir_fd=fresh[-1], follow_symlinks=False))
                count += 1
            finally:
                if fd is not None:
                    os.close(fd)
                for directory in reversed(fresh):
                    os.close(directory)
                for directory in reversed(held):
                    os.close(directory)
    return {"selected_files": count, "bytes_unchanged": True, "executable_bits_unchanged": True}


@pytest.fixture
def allocated(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch) as case:
        provision_authored_prelaunch_sources(case)
        yield case


def capture(case, prompt=PROMPT):
    return prelaunch.capture_repository_world_prelaunch_assignment(
        task_admission=case.native.handle, worktree_admission=case.guard, prompt=prompt)


def broker_args(case, prompt=PROMPT):
    return dict(server=case.native.server, canonical_repository=case.native.canonical,
        workspace=case.native.workspace, prompt=prompt,
        task_admission=case.native.handle, worktree_admission=case.guard)


def protected_claim(case):
    coordinator = case.native.daemon.coordinator
    claim = coordinator.get_task_claim(case.native.attempt.claim_id)
    return coordinator, claim, coordinator.protect_task_claim(claim)


def test_capture_binds_actual_native_task_allocation_and_literal_without_dispatch(allocated):
    gate = capture(allocated)
    observed = gate.require_current()
    task = allocated.native.handle.require_current()
    allocation = allocated.guard.require_current()
    assert observed["schema"] == prelaunch.SCHEMA
    assert observed["readiness_scope"] == "owner_prelaunch_assignment_only"
    assert observed["task_cid"] == task["task_cid"] != allocated.portal_identity["canonical_task_cid"]
    assert observed["task_revision"] == task["task_revision"]
    assert observed["expected_task_body_cid"] == task["expected_task_body_cid"]
    assert observed["claim_identity"] == task["claim_identity"]
    assert observed["allocation_cid"] == allocation["allocation_cid"]
    assert observed["source_binding_cid"] == allocated.native.binding["binding_cid"]
    assert observed["selected_source_count"] == 1
    assert observed["selected_source_bytes"] == len(SOURCE)
    assert observed["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert observed["prompt_bytes"] == len(PROMPT.encode())
    assert observed["prompt_binding_scope"] == "caller_supplied_literal_only"
    assert observed["nominated_scope"] is None
    assert all(observed[field] is False for field in prelaunch.FALSE_FIELDS)
    wire = json.dumps(observed, sort_keys=True)
    assert len(wire.encode()) <= prelaunch.MAX_OBSERVATION_BYTES
    assert str(allocated.native.canonical) not in wire and str(allocated.native.workspace) not in wire
    assert PROMPT not in wire and SOURCE.decode() not in wire
    assert allocated.native.daemon.provider_invocation_recorded(allocated.native.attempt.attempt_id,
        idempotency_key="provider:" + allocated.native.attempt.attempt_id) is None


def test_detached_observation_mutation_does_not_change_live_gate(allocated):
    gate = capture(allocated)
    before = gate.require_current()
    altered = gate.require_current()
    altered["claim_identity"]["claim_id"] = "changed"
    altered["world_dispatch_ready"] = True
    assert gate.require_current() == before


def test_original_broker_tuple_and_utf8_bytes_prompt_recheck_same_gate(allocated):
    gate = capture(allocated)
    assert gate.require_for_broker(**broker_args(allocated, PROMPT.encode())) == gate.require_current()


@pytest.mark.parametrize("field", ["server", "canonical_repository", "workspace", "prompt", "task_admission", "worktree_admission"])
def test_broker_refuses_foreign_assignment_even_when_gate_is_current(allocated, field):
    gate = capture(allocated)
    args = broker_args(allocated)
    args[field] = (Path("/different-owner-root") if field in {"canonical_repository", "workspace"}
                   else "changed prompt" if field == "prompt" else object())
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="broker_assignment_mismatch"):
        gate.require_for_broker(**args)


@pytest.mark.parametrize("handle", ["task", "allocation"])
@pytest.mark.parametrize("replacement", ["serialized", "subclass", "unsealed"])
def test_exact_native_objects_required_before_capture(allocated, handle, replacement):
    original = allocated.native.handle if handle == "task" else allocated.guard
    if replacement == "serialized":
        forged = original.require_current()
    elif replacement == "subclass":
        forged = object.__new__(type("AuthoredSubclass", (type(original),), {}))
    else:
        forged = object.__new__(type(original))
    args = dict(task_admission=allocated.native.handle, worktree_admission=allocated.guard, prompt=PROMPT)
    args["task_admission" if handle == "task" else "worktree_admission"] = forged
    with pytest.raises(REFUSALS):
        prelaunch.capture_repository_world_prelaunch_assignment(**args)


def test_same_class_unissued_and_shallow_copied_gates_cannot_rehydrate_authority(allocated):
    gate = capture(allocated)
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError):
        prelaunch.NativeRepositoryWorldPrelaunchAssignment.require_current(
            object.__new__(prelaunch.NativeRepositoryWorldPrelaunchAssignment))
    duplicate = copy(gate)
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="not_issued"):
        duplicate.require_current()
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="host_capture_required"):
        prelaunch.NativeRepositoryWorldPrelaunchAssignment(object(), task_admission=allocated.native.handle,
            worktree_admission=allocated.guard, prompt=PROMPT)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("method", ["require_current", "require_for_broker"])
def test_empty_or_partially_populated_exact_gate_fails_with_named_value_error(allocated, partial, method):
    gate = object.__new__(prelaunch.NativeRepositoryWorldPrelaunchAssignment)
    if partial:
        gate._seal, gate._pid, gate._uid = prelaunch._SEAL, os.getpid(), os.geteuid()
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError):
        if method == "require_current":
            gate.require_current()
        else:
            gate.require_for_broker(**broker_args(allocated))


@pytest.mark.parametrize("field", ["_observation", "_source_rows", "_source_witnesses", "_original", "_prompt"])
def test_byte_equal_private_snapshot_replacement_is_not_original_custody(allocated, field):
    gate = capture(allocated)
    old = getattr(gate, field)
    replacement = bytes(bytearray(old)) if type(old) is bytes else tuple(list(old))
    assert replacement == old and replacement is not old
    setattr(gate, field, replacement)
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="not_issued"):
        gate.require_current()


@pytest.mark.parametrize("field", ["_task", "_allocation", "_source", "_server", "_canonical", "_workspace", "_nominated"])
def test_original_assignment_reference_replacement_refuses(allocated, field):
    gate = capture(allocated)
    setattr(gate, field, object())
    with pytest.raises(REFUSALS):
        gate.require_current()


@pytest.mark.parametrize("method", ["require_current", "_observe", "_retained", "_rows"])
def test_issued_gate_rejects_instance_callback_shadowing_and_stale_claim(allocated, method):
    gate = capture(allocated)
    before = gate.require_current()
    with pytest.raises(AttributeError):
        setattr(gate, method, lambda *_args, **_kwargs: before)
    coordinator, _claim, lease = protected_claim(allocated)
    coordinator.release(lease, reason="authored shadowing refusal")
    with pytest.raises(REFUSALS):
        prelaunch.NativeRepositoryWorldPrelaunchAssignment.require_for_broker(gate, **broker_args(allocated))


@pytest.mark.parametrize("change", ["revision", "body", "phase", "claim"])
def test_native_task_revision_full_body_and_claim_drift_refuses(allocated, change):
    gate = capture(allocated)
    native = allocated.native
    if change == "revision":
        with native.server._lock:
            native.server._connection.execute("UPDATE tasks SET revision = revision + 1 WHERE task_cid = ?", [native.task_cid])
    else:
        body = deepcopy(dict(native.native_task.body))
        if change == "body":
            body["previously_unknown_field"] = {"changed": True}
        elif change == "phase":
            body["completion_receipt"]["attempt_execution_phase"] = "context"
        else:
            body["completion_receipt"]["claim_id"] = "foreign-claim"
        _update_body(native, body)
    with pytest.raises(REFUSALS):
        gate.require_current()


@pytest.mark.parametrize("field", ["task_id", "lease_id", "branch", "state", "expires_at"])
def test_self_consistent_lifecycle_drift_refuses(allocated, field):
    gate = capture(allocated)
    value = time.time() - 1 if field == "expires_at" else "terminal" if field == "state" else "foreign-owner"
    with _record_change(allocated, lambda row: row.update({field: value})):
        with pytest.raises(REFUSALS):
            gate.require_current()


@pytest.mark.parametrize("transition", ["released", "expired", "revoked_grant"])
def test_real_native_claim_or_owner_grant_loss_refuses(allocated, transition):
    gate = capture(allocated)
    coordinator, claim, lease = protected_claim(allocated)
    if transition == "released":
        coordinator.release(lease, reason="authored prelaunch lease release")
    elif transition == "expired":
        coordinator.expire_task_claim(claim, now_ms=lease.expires_at_ms)
    else:
        allocated.native.server._command_gateway.revoke_grant(allocated.native.daemon_grant_id)
    with pytest.raises(REFUSALS):
        gate.require_current()


@pytest.mark.parametrize("issued", [False, True])
def test_last_retained_cas_callback_releases_real_coordinator_lease_and_refuses(allocated, monkeypatch, issued):
    gate = capture(allocated) if issued else None
    coordinator, _claim, lease = protected_claim(allocated)
    original = allocated.native.retained.cas.get_bytes
    fired = []
    def release_after_native_cas_read(cid):
        value = original(cid)
        if not fired:
            fired.append(cid)
            coordinator.release(lease, reason="authored last CAS callback release")
        return value
    monkeypatch.setattr(allocated.native.retained.cas, "get_bytes", release_after_native_cas_read)
    with pytest.raises(REFUSALS):
        gate.require_current() if issued else capture(allocated)
    assert len(fired) == 1
    # The release is real even though its SQL row/body and daemon grant survive.
    with allocated.native.server._lock:
        observed, _grant = allocated.native.handle._native()
        assert json.loads(observed[0])["task_cid"] == allocated.native.task_cid


@pytest.mark.parametrize("name", ["binding", "task_projection", "git_marker"])
def test_byte_identical_control_inode_replacement_refuses(allocated, name):
    gate = capture(allocated)
    path = allocated.native.workspace / ".git" if name == "git_marker" else getattr(allocated.paths, name)
    before, mode = path.read_bytes(), path.stat().st_mode & 0o777
    original = path.with_name(path.name + ".prelaunch-original")
    replacement = path.with_name(path.name + ".prelaunch-replacement")
    path.rename(original)
    replacement.write_bytes(before)
    replacement.chmod(mode)
    replacement.rename(path)
    try:
        with pytest.raises(REFUSALS):
            gate.require_current()
    finally:
        path.rename(replacement)
        original.rename(path)


@pytest.mark.parametrize("which", ["canonical", "workspace"])
def test_group_write_permission_transition_refuses_without_normalizing(allocated, which):
    gate = capture(allocated)
    path = getattr(allocated.native, which)
    mode = path.stat().st_mode & 0o777
    path.chmod(0o770)
    try:
        with pytest.raises(REFUSALS):
            gate.require_current()
        assert path.stat().st_mode & 0o777 == 0o770
    finally:
        path.chmod(mode)


def test_authored_uid_mismatch_refuses_without_claiming_cross_uid_launch(allocated, monkeypatch):
    gate = capture(allocated)
    uid = os.geteuid()
    with monkeypatch.context() as controlled:
        controlled.setattr(prelaunch.os, "geteuid", lambda: uid + 1)
        with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="owner_process_required"):
            gate.require_current()


@pytest.mark.parametrize("which", ["canonical", "workspace"])
def test_changed_selected_source_bytes_refuse(allocated, which):
    gate = capture(allocated)
    path = getattr(allocated.native, which) / "module.py"
    before = path.read_bytes()
    path.write_bytes(before.replace(b"VALUE", b"OTHER", 1))
    try:
        with pytest.raises(REFUSALS):
            gate.require_current()
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize("which", ["canonical", "workspace"])
def test_root_namespace_swap_after_selected_fd_read_refuses_identical_bytes(allocated, monkeypatch, which):
    gate = capture(allocated)
    root = getattr(allocated.native, which)
    displaced, replacement = root.with_name(root.name + "-displaced"), root.with_name(root.name + "-replacement")
    inode = (root / "module.py").stat().st_ino
    original_read, fired = os.read, []
    def swap_after_open(fd, amount):
        value = original_read(fd, amount)
        if not fired and os.fstat(fd).st_ino == inode and value:
            fired.append(True)
            root.rename(displaced)
            shutil.copytree(displaced, root, symlinks=True)
        return value
    monkeypatch.setattr(prelaunch.os, "read", swap_after_open)
    try:
        with pytest.raises(REFUSALS):
            gate.require_current()
        assert fired == [True]
        assert (root / "module.py").read_bytes() == (displaced / "module.py").read_bytes()
    finally:
        if fired:
            root.rename(replacement)
            displaced.rename(root)
    assert replacement.exists()  # Preserve both namespaces; no deletion/reset.


@pytest.mark.parametrize("issued", [False, True])
def test_real_coordinator_clock_source_edit_after_native_read_refuses(allocated, monkeypatch, issued):
    gate = capture(allocated) if issued else None
    native = allocated.native
    original_native = NativeSourceTaskAdmission._native
    armed = []
    def arm_after_actual_native_read(handle):
        value = original_native(handle)
        if handle is native.handle and not armed:
            armed.append(True)
            # Preserve the actual coordinator clock object and its call path.
            native.clock.reads_until_edit = 1
        return value
    monkeypatch.setattr(NativeSourceTaskAdmission, "_native", arm_after_actual_native_read)
    with pytest.raises(REFUSALS):
        gate.require_current() if issued else capture(allocated)
    assert armed == [True] and native.clock.edited is True


def test_nested_directory_namespace_swap_after_selected_fd_read_refuses(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files={b"nested/item.py": SOURCE}) as case:
        args = case.source_binding | {"allowed_paths": (b"nested/item.py".hex(),)}
        case.native.binding = case.guard.bind_current_source_lookup(**args)
        # Only this new authored nested fixture needs directory provisioning;
        # the inherited repository/Git writer follows umask for new parents.
        for root in (case.native.canonical, case.native.workspace):
            root_fd = prelaunch._root_fd(root)
            directory_fd = file_fd = fresh_fd = None
            try:
                directory_fd = os.open(b"nested", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
                before = os.fstat(directory_fd)
                assert stat.S_ISDIR(before.st_mode) and before.st_uid == os.geteuid()
                file_fd = os.open(b"item.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                file_before = os.fstat(file_fd)
                assert stat.S_ISREG(file_before.st_mode) and file_before.st_uid == os.geteuid() and file_before.st_nlink == 1
                assert os.read(file_fd, len(SOURCE) + 1) == SOURCE
                os.fchmod(directory_fd, stat.S_IMODE(before.st_mode) & ~0o022)
                after = os.fstat(directory_fd)
                assert (after.st_dev, after.st_ino, after.st_uid) == (before.st_dev, before.st_ino, before.st_uid)
                assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) & ~0o022
                assert after.st_mode & 0o111 == before.st_mode & 0o111
                os.lseek(file_fd, 0, os.SEEK_SET)
                assert os.read(file_fd, len(SOURCE) + 1) == SOURCE
                assert prelaunch._file_identity(file_before) == prelaunch._file_identity(os.fstat(file_fd))
                fresh_fd = prelaunch._root_fd(root)
                assert (os.fstat(fresh_fd).st_dev, os.fstat(fresh_fd).st_ino) == (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino)
                named = os.stat(b"nested", dir_fd=fresh_fd, follow_symlinks=False)
                assert (named.st_dev, named.st_ino, named.st_uid, named.st_mode) == (after.st_dev, after.st_ino, after.st_uid, after.st_mode)
            finally:
                for fd in (file_fd, directory_fd, fresh_fd, root_fd):
                    if fd is not None:
                        os.close(fd)
        provision_authored_prelaunch_sources(case)
        gate = capture(case)
        parent = case.native.workspace / "nested"
        displaced, replacement = parent.with_name("nested-displaced"), parent.with_name("nested-replacement")
        inode = (parent / "item.py").stat().st_ino
        original_read, fired = os.read, []
        def swap_after_open(fd, amount):
            value = original_read(fd, amount)
            if not fired and os.fstat(fd).st_ino == inode and value:
                fired.append(True)
                parent.rename(displaced)
                shutil.copytree(displaced, parent)
            return value
        with monkeypatch.context() as controlled:
            controlled.setattr(prelaunch.os, "read", swap_after_open)
            try:
                with pytest.raises(REFUSALS):
                    gate.require_current()
                assert fired == [True]
            finally:
                if fired:
                    parent.rename(replacement)
                    displaced.rename(parent)
        assert replacement.exists()


def test_real_native_and_lifecycle_renewal_preserve_assignment_identity(allocated):
    gate = capture(allocated)
    before = gate.require_current()
    coordinator, _claim, lease = protected_claim(allocated)
    renewed = coordinator.renew(lease, lease_ms=120000)
    assert renewed.lease_id == lease.lease_id
    record = allocated.store.load_workspace(allocated.native.workspace)
    renewed_record = allocated.store.renew_lease(allocated.native.workspace,
        lease_id=record.lease_id, expected_fence=record.fence)
    assert renewed_record.fence > record.fence
    assert gate.require_current() == before


def test_original_genuine_nomination_replayed_and_observed_without_compiler_claim(nominated_allocation):
    case = nominated_allocation
    bind(case)
    provision_authored_prelaunch_sources(case)
    gate = capture(case)
    observed = gate.require_current()
    nominated = case.guard.require_nominated_source_scope()
    assert observed["selected_source_count"] == 2
    assert observed["nominated_scope"] == {key: nominated[key] for key in (
        "source_scope_cid", "frontier_cid", "population_cid", "native_snapshot_cid",
        "selected_count", "selected_bytes", "frontier_count", "allowed_paths_cid")}
    assert observed["compiled_context_verified"] is False


@pytest.mark.parametrize("prompt", [None, {}, "", b"", b"\xff"])
def test_invalid_prompt_refuses_before_native_observation(allocated, monkeypatch, prompt):
    calls = []
    def forbidden(*_args, **_kwargs):
        calls.append(True)
        pytest.fail("invalid literal reached native observation")
    monkeypatch.setattr(NativeSourceTaskAdmission, "require_current", forbidden)
    with pytest.raises((ValueError, UnicodeError)):
        capture(allocated, prompt)
    assert not calls


def test_utf8_prompt_limit_refuses_before_native_observation(allocated, monkeypatch):
    monkeypatch.setattr(prelaunch, "MAX_PROMPT_BYTES", 3)
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError, match="prompt_byte_limit"):
        capture(allocated, "λλ")


@pytest.mark.parametrize("limit", ["MAX_SOURCE_PATHS", "MAX_SELECTED_SOURCE_BYTES", "MAX_OBSERVATION_BYTES"])
def test_lowered_gate_bounds_refuse_complete_population_without_truncation(allocated, monkeypatch, limit):
    monkeypatch.setattr(prelaunch, limit, 0 if limit != "MAX_OBSERVATION_BYTES" else 64)
    with pytest.raises(prelaunch.RepositoryWorldPrelaunchError):
        capture(allocated)
