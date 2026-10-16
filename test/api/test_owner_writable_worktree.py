"""Descriptor-based permission provisioning before any owner worker census.

Temporary files, file descriptors, modes, inode identities and chmod operations
are real. Deployment boundary authentication is the existing authored host
seam; ownership, ACL and mount refusal cases narrowly author those observations.
No test qualifies a different UID, container deployment or provider execution.
"""
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_writable_worktree as provisioner
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
from test.api.candidate_resource_test_support import resource_parent
from test.api.test_current_source_worktree_allocation import _protect_control_tree
from test.api.test_native_candidate_continuation import children, prepared, run_options
from test.api.test_native_delegated_allocation import delegated_case


@pytest.fixture
def protected_tree(tmp_path, monkeypatch):
    root = tmp_path / "allocated"
    root.mkdir(mode=0o700)
    workspace = root / "candidate"
    workspace.mkdir(mode=0o750)
    (workspace / "sub").mkdir(mode=0o700)
    (workspace / "sub/bin").mkdir(mode=0o750)
    files = {".git": (b"gitdir: owner-private-admin\n", 0o640),
        "plain.txt": (b"Preserve the complete original source.\n", 0o640),
        "sub/data.bin": (b"\x00\xffunchanged bytes", 0o600),
        "sub/bin/run": (b"#!/bin/sh\nexit 7\n", 0o750)}
    for name, (raw, mode) in files.items():
        path = workspace / name
        path.write_bytes(raw)
        path.chmod(mode)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    sentinel = outside / "private-state"
    sentinel.write_bytes(b"private owner state must remain unchanged")
    sentinel.chmod(0o600)
    boundary = tmp_path / "boundary.json"
    boundary.write_bytes(b"authored host deployment boundary\n")
    boundary.chmod(0o444)
    def owner_boundary(path):
        if Path(path) != boundary:
            raise ValueError("authored host boundary differs")
        return dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1,
            allowed_worktree_roots=[str(root)], owner_private_paths=[]), hashlib.sha256(
                boundary.read_bytes()).hexdigest()
    monkeypatch.setattr(boundary_guard, "_owner_boundary", owner_boundary)
    return SimpleNamespace(root=root, workspace=workspace, outside=outside,
        sentinel=sentinel, boundary=boundary, files=files)


def provision(case, **changes):
    return provisioner.provision_owner_writable_worktree(**dict(
        boundary=case.boundary, workspace=case.workspace,
        absolute_deadline_ns=time.monotonic_ns() + 30_000_000_000) | changes)


def identity(path):
    info = path.lstat()
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_uid", "st_gid",
        "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def snapshot(root):
    result = {".": identity(root)}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in sorted((*directories, *files)):
            path = Path(base) / name
            info = path.lstat()
            extra = (path.read_bytes() if stat.S_ISREG(info.st_mode)
                else os.readlink(path) if stat.S_ISLNK(info.st_mode) else None)
            result[path.relative_to(root).as_posix()] = identity(path), extra
    return result


@contextmanager
def tracked_descriptors(monkeypatch):
    actual_open, actual_dup = os.open, os.dup
    opened = set()
    def tracked_open(*args, **kwargs):
        descriptor = actual_open(*args, **kwargs)
        opened.add(descriptor)
        return descriptor
    def tracked_dup(descriptor):
        copied = actual_dup(descriptor)
        opened.add(copied)
        return copied
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "open", tracked_open)
        patch.setattr(provisioner.os, "dup", tracked_dup)
        yield opened
    for descriptor in opened:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def refuses_without_chmod(case, monkeypatch, **changes):
    before, outside = snapshot(case.workspace), snapshot(case.outside)
    chmod_calls = []
    def forbidden(descriptor, mode):
        chmod_calls.append((descriptor, mode))
        pytest.fail("preflight refusal changed a permission")
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", forbidden)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case, **changes)
    assert chmod_calls == []
    assert refused.value.mutation_started is False
    assert refused.value.changed_entries == 0
    assert snapshot(case.workspace) == before
    assert snapshot(case.outside) == outside
    return refused.value


def test_real_safe_tree_is_provisioned_by_fds_without_rewriting_bytes_or_inodes(
        protected_tree, monkeypatch):
    case = protected_tree
    before, outside = snapshot(case.workspace), snapshot(case.outside)
    actual_chmod = os.fchmod
    calls = []
    def chmod(descriptor, mode):
        info = os.fstat(descriptor)
        calls.append(((info.st_dev, info.st_ino), mode))
        return actual_chmod(descriptor, mode)
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", chmod)
        with tracked_descriptors(patch):
            receipt = provision(case)
    root_id = before["."][0:2]
    assert calls[0] == (root_id, 0o700)
    assert calls[-1] == (root_id, 0o3770)
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o3770
    for name in ("sub", "sub/bin"):
        assert stat.S_IMODE((case.workspace / name).stat().st_mode) == 0o2770
    for name, (raw, _mode) in case.files.items():
        path = case.workspace / name
        assert path.read_bytes() == raw
        assert identity(path)[:4] == before[name][0][:4]
        expected = 0o440 if name == ".git" else 0o770 if name == "sub/bin/run" else 0o660
        assert stat.S_IMODE(path.stat().st_mode) == expected
    assert snapshot(case.outside) == outside
    assert type(receipt) is dict
    assert receipt["completion_authority"] is receipt["execution_authority"] is False
    assert receipt["descriptor_permission_transition_verified"] is True
    assert receipt["named_root_policy_verified"] is True
    assert receipt["worker_isolation_qualified"] is False
    assert receipt["filesystem_writer_lease_verified"] is False


@pytest.mark.parametrize("kind", ["directory_symlink", "file_symlink", "file_hardlink",
    "marker_hardlink", "fifo"])
def test_unsafe_entries_are_all_rejected_before_any_permission_mutation(
        protected_tree, monkeypatch, kind):
    case = protected_tree
    if kind == "directory_symlink":
        (case.workspace / "unsafe").symlink_to(case.outside, target_is_directory=True)
    elif kind == "file_symlink":
        (case.workspace / "unsafe").symlink_to(case.sentinel)
    elif kind == "file_hardlink":
        os.link(case.sentinel, case.workspace / "unsafe")
    elif kind == "marker_hardlink":
        os.link(case.workspace / ".git", case.outside / "marker-alias")
    else:
        os.mkfifo(case.workspace / "unsafe")
    refuses_without_chmod(case, monkeypatch)


def test_a_symlink_workspace_root_is_refused_without_affecting_its_target(
        protected_tree, monkeypatch):
    case = protected_tree
    original = case.workspace
    link = case.root / "candidate-link"
    link.symlink_to(original, target_is_directory=True)
    case.workspace = link
    refuses_without_chmod(case, monkeypatch)


@pytest.mark.parametrize("name,mode", [(".", 0o770), ("sub", 0o770), ("plain.txt", 0o664)])
def test_existing_group_writes_are_not_adopted_as_provisioning_input(
        protected_tree, monkeypatch, name, mode):
    (protected_tree.workspace / name).chmod(mode)
    refuses_without_chmod(protected_tree, monkeypatch)


@pytest.mark.parametrize("mode", [0o4750, 0o2750, 0o1750])
def test_regular_file_privilege_or_sticky_bits_are_refused_before_mutation(
        protected_tree, monkeypatch, mode):
    (protected_tree.workspace / "sub/bin/run").chmod(mode)
    refuses_without_chmod(protected_tree, monkeypatch)


@pytest.mark.parametrize("name", ["system.posix_acl_access", "system.posix_acl_default"])
def test_access_or_default_acl_on_a_nested_inode_refuses_full_preflight(
        protected_tree, monkeypatch, name):
    case = protected_tree
    expected = (case.workspace / "sub").stat()
    actual_list = os.listxattr
    actual_acl = provisioner._acl_state
    def authored_acl(path, *, descriptor):
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
            def listed(selected):
                values = actual_list(selected)
                return list(values) + [name] if selected == descriptor else values
            with monkeypatch.context() as patch:
                patch.setattr(os, "listxattr", listed)
                return actual_acl(path, descriptor=descriptor)
        return actual_acl(path, descriptor=descriptor)
    monkeypatch.setattr(provisioner, "_acl_state", authored_acl)
    refuses_without_chmod(case, monkeypatch)


@pytest.mark.parametrize("field", ["st_uid", "st_gid"])
def test_foreign_owner_or_group_on_an_opened_file_refuses_without_chmod(
        protected_tree, monkeypatch, field):
    case = protected_tree
    expected = (case.workspace / "plain.txt").stat()
    actual_fstat, actual_stat = os.fstat, os.stat
    def foreign(info):
        if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values[field] += 1
            return SimpleNamespace(**values)
        return info
    monkeypatch.setattr(provisioner.os, "fstat", lambda descriptor: foreign(actual_fstat(descriptor)))
    monkeypatch.setattr(provisioner.os, "stat", lambda *args, **kwargs: foreign(actual_stat(*args, **kwargs)))
    refuses_without_chmod(case, monkeypatch)


def test_a_nested_mount_identity_cannot_join_the_owner_root(protected_tree, monkeypatch):
    case = protected_tree
    expected = (case.workspace / "sub").stat()
    actual = provisioner._mount_identity
    def different(descriptor):
        observed = actual(descriptor)
        info = os.fstat(descriptor)
        return ((observed[0] + 1, observed[1])
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino) else observed)
    monkeypatch.setattr(provisioner, "_mount_identity", different)
    refuses_without_chmod(case, monkeypatch)


def test_a_workspace_root_mount_cannot_differ_from_its_protected_parent(
        protected_tree, monkeypatch):
    case = protected_tree
    expected = case.workspace.stat()
    actual = provisioner._mount_identity
    def different(descriptor):
        observed = actual(descriptor)
        info = os.fstat(descriptor)
        return ((observed[0] + 1, observed[1])
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino) else observed)
    monkeypatch.setattr(provisioner, "_mount_identity", different)
    refuses_without_chmod(case, monkeypatch)


@pytest.mark.parametrize("scope", ["missing", "private_ancestor", "private_workspace", "private_child"])
def test_private_authority_scope_must_be_explicit_and_disjoint_from_workspace(
        protected_tree, monkeypatch, scope):
    case = protected_tree
    actual = boundary_guard._owner_boundary
    def unsafe_boundary(path):
        value, digest = actual(path)
        if scope == "missing":
            del value["owner_private_paths"]
        else:
            private = {"private_ancestor": case.root, "private_workspace": case.workspace,
                "private_child": case.workspace / "sub"}[scope]
            value["owner_private_paths"] = [str(private)]
        return value, digest
    monkeypatch.setattr(boundary_guard, "_owner_boundary", unsafe_boundary)
    refuses_without_chmod(case, monkeypatch)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_named_replacement_after_preflight_refuses_before_chmod(
        protected_tree, monkeypatch, kind):
    case = protected_tree
    actual = provisioner._preflight
    original = case.workspace / ("plain.txt" if kind == "file" else "sub")
    retained = case.outside / "retained-original"
    original_identity = identity(original)
    calls = []
    def changed(root_fd, **kwargs):
        manifest = actual(root_fd, **kwargs)
        original.rename(retained)
        if kind == "file":
            original.symlink_to(case.sentinel)
        else:
            original.mkdir(mode=0o700)
        return manifest
    monkeypatch.setattr(provisioner, "_preflight", changed)
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", lambda *_args: calls.append(True))
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case)
    assert calls == []
    assert refused.value.mutation_started is False
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o750
    assert identity(retained)[:6] == original_identity[:6]
    assert case.sentinel.read_bytes() == b"private owner state must remain unchanged"
    assert stat.S_IMODE(case.sentinel.stat().st_mode) == 0o600


@pytest.mark.parametrize("constant,value", [("MAX_ENTRIES", 2), ("MAX_DEPTH", 1),
    ("MAX_RELATIVE_PATH_BYTES", 3), ("MAX_TOTAL_PATH_BYTES", 8)])
def test_preflight_bounds_refuse_a_complete_tree_before_permission_mutation(
        protected_tree, monkeypatch, constant, value):
    actual = provisioner._preflight
    def bounded(root_fd, **kwargs):
        # Lower inventory limits only after real ancestry admission, so the
        # depth case exercises the worktree walk rather than its parent path.
        with monkeypatch.context() as patch:
            patch.setattr(provisioner, constant, value)
            return actual(root_fd, **kwargs)
    monkeypatch.setattr(provisioner, "_preflight", bounded)
    refuses_without_chmod(protected_tree, monkeypatch)


def test_expired_original_deadline_refuses_before_chmod(protected_tree, monkeypatch):
    refuses_without_chmod(protected_tree, monkeypatch,
        absolute_deadline_ns=time.monotonic_ns() - 1)


def test_suppressed_recursive_provisioning_invalidates_outer_preflight(
        protected_tree, monkeypatch):
    case = protected_tree
    actual = provisioner._preflight
    nested = []
    def recursive(root_fd, **kwargs):
        manifest = actual(root_fd, **kwargs)
        with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError):
            provision(case)
        nested.append(True)
        return manifest
    monkeypatch.setattr(provisioner, "_preflight", recursive)
    refuses_without_chmod(case, monkeypatch)
    assert nested == [True]


def test_partial_chmod_failure_quarantines_root_and_closes_every_descriptor(
        protected_tree, monkeypatch):
    case = protected_tree
    actual = os.fchmod
    calls = []
    def failed(descriptor, mode):
        calls.append((descriptor, mode))
        if len(calls) == 3:
            raise OSError("authored permission mutation failure")
        return actual(descriptor, mode)
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", failed)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case)
    assert refused.value.mutation_started is True
    assert refused.value.quarantine_verified is True
    assert refused.value.quarantine_error_type is None
    assert refused.value.changed_entries >= 1
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700
    assert calls[-1][1] == 0o700


def test_failed_quarantine_is_reported_as_uncertain(protected_tree, monkeypatch):
    case = protected_tree
    actual = os.fchmod
    published = []
    def failed(descriptor, mode):
        if mode == 0o3770:
            actual(descriptor, mode)
            published.append(True)
            raise OSError("authored failure after root publication")
        if published and mode == 0o700:
            raise OSError("authored quarantine failure")
        return actual(descriptor, mode)
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", failed)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case)
    assert refused.value.mutation_started is True
    assert refused.value.quarantine_verified is False
    assert type(refused.value.quarantine_error_type) is str
    assert refused.value.quarantine_error_type
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o3770


def test_workspace_replacement_after_mutation_cannot_be_quarantined_as_original(
        protected_tree, monkeypatch):
    case = protected_tree
    actual = os.fchmod
    retained = case.outside / "original-workspace"
    changed = []
    def replaced(descriptor, mode):
        result = actual(descriptor, mode)
        if not changed:
            case.workspace.rename(retained)
            case.workspace.mkdir(mode=0o750)
            (case.workspace / "replacement-sentinel").write_bytes(b"replacement untouched")
            (case.workspace / "replacement-sentinel").chmod(0o600)
            changed.append(True)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", replaced)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case)
    assert refused.value.mutation_started is True
    assert refused.value.quarantine_verified is False
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o750
    sentinel = case.workspace / "replacement-sentinel"
    assert sentinel.read_bytes() == b"replacement untouched"
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o600
    assert stat.S_IMODE(retained.stat().st_mode) == 0o700


def test_chmod_that_applies_then_raises_remains_a_mutation_failure(
        protected_tree, monkeypatch):
    case = protected_tree
    actual = os.fchmod
    failed_once = []
    def applied_then_failed(descriptor, mode):
        result = actual(descriptor, mode)
        if not failed_once:
            failed_once.append(True)
            raise OSError("authored failure after a permission changed")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", applied_then_failed)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case)
    assert refused.value.mutation_started is True
    assert refused.value.changed_entries == 0
    assert refused.value.quarantine_verified is True
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_original_deadline_expiring_after_root_gate_quarantines_without_enablement(
        protected_tree, monkeypatch):
    case = protected_tree
    deadline = time.monotonic_ns() + 30_000_000_000
    actual = os.fchmod
    expired = []
    def expires_after_gate(descriptor, mode):
        result = actual(descriptor, mode)
        if not expired:
            expired.append(True)
            monkeypatch.setattr(provisioner.time, "monotonic_ns", lambda: deadline + 1)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(provisioner.os, "fchmod", expires_after_gate)
        with tracked_descriptors(patch):
            with pytest.raises(provisioner.OwnerWritableWorktreeProvisioningError) as refused:
                provision(case, absolute_deadline_ns=deadline)
    assert refused.value.mutation_started is True
    assert refused.value.quarantine_verified is True
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_actual_provisioning_precedes_first_census_and_native_continuation(
        tmp_path, monkeypatch, resource_parent):
    receipts = []
    def provision_native(allocated):
        # Explicit trusted fixture setup removes ambient checkout umask writes
        # before the real provisioning helper validates its protected input.
        _protect_control_tree(allocated.native.workspace)
        boundary = allocated.native.factory["boundary"]
        boundary.chmod(0o444)
        admitted = boundary_guard._owner_boundary
        def owner_boundary(path):
            value, digest = admitted(path)
            return {**value, "allowed_worktree_roots": [str(allocated.capture["worktree_root"])],
                "owner_private_paths": []}, digest
        monkeypatch.setattr(boundary_guard, "_owner_boundary", owner_boundary)
        receipts.append(provisioner.provision_owner_writable_worktree(boundary=boundary,
            workspace=allocated.native.workspace,
            absolute_deadline_ns=time.monotonic_ns() + 60_000_000_000))
    with delegated_case(tmp_path, monkeypatch, provision_workspace=provision_native) as case:
        assert len(receipts) == 1
        assert case.prior_worker_attempt._state.retired is True
        assert stat.S_IMODE(case.native.workspace.stat().st_mode) == 0o3770
        with prepared(case, resource_parent) as session:
            options, marker = run_options(session, tmp_path)
            result = session.run(**options)
            assert result.returncode == 0
            observed = json.loads(marker.read_bytes())
            assert observed["sha256"] == hashlib.sha256(
                session.candidate_prompt.provider_prompt).hexdigest()
            assert children(resource_parent) == {}
