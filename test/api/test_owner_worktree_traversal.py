"""Owner-authorized parent search publication while the workspace stays private.

Directory FDs, modes, inventories and changes are real. Deployment authority
and selected ACL/identity observations are authored host seams. These checks do
not establish a second UID, Git control readability or worker isolation.
"""
import hashlib
import inspect
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_worktree_traversal as traversal
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worktree_parent as parents
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
from test.api.test_owner_writable_worktree import snapshot, tracked_descriptors


@pytest.fixture
def traversal_case(tmp_path, pytestconfig, monkeypatch):
    # Only these freshly owned pytest containers are published by the fixture.
    # Account-wide private TMPDIR/home ancestors are never permission-normalized.
    basetemp = pytestconfig._tmp_path_factory.getbasetemp()
    assert tmp_path.parent == basetemp and basetemp.stat().st_uid == os.getuid()
    basetemp.chmod(0o750)
    tmp_path.chmod(0o750)
    anchor = tmp_path / "public-anchor"
    anchor.mkdir(mode=0o750)
    root = anchor / "worktrees"
    root.mkdir(mode=0o700)
    lane = root / "lane"
    lane.mkdir(mode=0o700)
    workspace = lane / "candidate"
    workspace.mkdir(mode=0o700)
    (workspace / ".git").write_bytes(b"gitdir: authored nonexecuted control\n")
    (workspace / ".git").chmod(0o600)
    (workspace / "module.py").write_bytes(b"SOURCE = 'unchanged'\n")
    (workspace / "module.py").chmod(0o600)
    owner = tmp_path / "private-owner"
    owner.mkdir(mode=0o700)
    boundary = owner / "boundary.json"
    boundary.write_bytes(b"authored host traversal boundary\n")
    boundary.chmod(0o444)
    private = owner / "state"
    private.mkdir(mode=0o700)
    (private / "sentinel").write_bytes(b"private state unchanged")
    (private / "sentinel").chmod(0o600)
    case = SimpleNamespace(anchor=anchor, root=root, lane=lane, workspace=workspace,
        owner=owner, boundary=boundary, private=private)
    case.policy = dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1,
        allowed_worktree_roots=[str(root)], owner_private_paths=[str(private)],
        worktree_traversal_anchor=str(anchor),
        worktree_traversal_directories=[str(root), str(lane)], worker_shared_group=os.getuid())
    def owner_boundary(path):
        assert Path(path) == case.boundary
        return dict(case.policy), hashlib.sha256(case.boundary.read_bytes()).hexdigest()
    monkeypatch.setattr(boundary_guard, "_owner_boundary", owner_boundary)
    return case


def publish(case, **changes):
    return traversal.provision_owner_worktree_traversal(**dict(boundary=case.boundary,
        workspace=case.workspace, absolute_deadline_ns=time.monotonic_ns() + 30_000_000_000) | changes)


def refuses_without_change(case, monkeypatch, **changes):
    public, private = snapshot(case.anchor), snapshot(case.owner)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("traversal preflight refusal changed a mode")
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", forbidden)
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case, **changes)
    assert calls == []
    assert error.value.mutation_attempted is False
    assert error.value.changed_directories == ()
    assert snapshot(case.anchor) == public and snapshot(case.owner) == private
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700
    return error.value


def test_real_parent_search_is_published_deepest_first_without_touching_workspace(traversal_case, monkeypatch):
    case = traversal_case
    workspace, private = snapshot(case.workspace), snapshot(case.owner)
    expected = {(case.root.stat().st_dev, case.root.stat().st_ino): str(case.root),
        (case.lane.stat().st_dev, case.lane.stat().st_ino): str(case.lane)}
    actual = os.fchmod
    calls = []
    def chmod(descriptor, mode):
        info = os.fstat(descriptor)
        calls.append((expected[(info.st_dev, info.st_ino)], mode))
        assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700
        return actual(descriptor, mode)
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", chmod)
        with tracked_descriptors(patch):
            receipt = publish(case)
    assert calls == [(str(case.lane), 0o710), (str(case.root), 0o710)]
    assert stat.S_IMODE(case.root.stat().st_mode) == stat.S_IMODE(case.lane.stat().st_mode) == 0o710
    assert snapshot(case.workspace) == workspace and snapshot(case.owner) == private
    assert receipt["completion_authority"] is receipt["execution_authority"] is False
    assert receipt["declared_group_search_bits_verified"] is True
    assert receipt["worker_group_membership_verified"] is False
    assert receipt["worker_traversal_qualified"] is False
    assert receipt["worker_isolation_qualified"] is False
    assert receipt["git_control_access_verified"] is False


@pytest.mark.parametrize("mode", [0o710, 0o750])
def test_selected_already_searchable_modes_are_reused_without_chmod(traversal_case, monkeypatch, mode):
    case = traversal_case
    case.root.chmod(mode)
    case.lane.chmod(mode)
    before = snapshot(case.anchor)
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", lambda *_args:
            pytest.fail("searchable selected directory was normalized"))
        with tracked_descriptors(patch):
            publish(case)
    assert snapshot(case.anchor) == before


@pytest.mark.parametrize("where,kind", [("root", "file"), ("root", "readable_directory"),
    ("lane", "file"), ("lane", "readable_directory")])
def test_owner_only_selected_directory_cannot_publish_an_unknown_sibling(
        traversal_case, monkeypatch, where, kind):
    parent = getattr(traversal_case, where)
    if kind == "file":
        (parent / "sibling").write_bytes(b"unselected sibling")
    else:
        (parent / "sibling").mkdir(mode=0o750)
    refuses_without_change(traversal_case, monkeypatch)


@pytest.mark.parametrize("field", ["worktree_traversal_anchor", "worktree_traversal_directories", "worker_shared_group"])
def test_each_trusted_optin_field_is_required_before_any_change(traversal_case, monkeypatch, field):
    del traversal_case.policy[field]
    refuses_without_change(traversal_case, monkeypatch)


@pytest.mark.parametrize("value", [True, "1000", "foreign"])
def test_worker_shared_group_must_be_the_exact_owner_integer(traversal_case, monkeypatch, value):
    traversal_case.policy["worker_shared_group"] = os.getuid() + 1 if value == "foreign" else value
    refuses_without_change(traversal_case, monkeypatch)


@pytest.mark.parametrize("scope", ["workspace", "private", "outside_anchor", "duplicate",
    "anchor_private_overlap", "selected_alias", "boundary_in_workspace"])
def test_selected_scope_must_be_exact_chain_and_disjoint_from_private_authority(
        traversal_case, monkeypatch, scope):
    case = traversal_case
    if scope == "workspace":
        case.policy["worktree_traversal_directories"] = [str(case.workspace)]
    elif scope == "private":
        case.policy["worktree_traversal_directories"] = [str(case.private)]
    elif scope == "outside_anchor":
        case.policy["worktree_traversal_directories"] = [str(case.anchor.parent)]
    elif scope == "duplicate":
        case.policy["worktree_traversal_directories"].append(str(case.root))
    elif scope == "anchor_private_overlap":
        case.policy["owner_private_paths"] = [str(case.anchor)]
    elif scope == "selected_alias":
        alias = case.anchor / "lane-alias"
        alias.symlink_to(case.lane, target_is_directory=True)
        case.policy["worktree_traversal_directories"] = [str(case.root), str(alias)]
    else:
        replacement = case.workspace / "boundary.json"
        replacement.write_bytes(case.boundary.read_bytes())
        replacement.chmod(0o444)
        case.boundary = replacement
    refuses_without_change(case, monkeypatch)


@pytest.mark.parametrize("mode", [0o750, 0o3770])
def test_workspace_must_remain_in_initial_exact_owner_only_mode(traversal_case, monkeypatch, mode):
    case = traversal_case
    case.workspace.chmod(mode)
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", lambda *_args:
            pytest.fail("already exposed workspace enabled parent traversal"))
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case)
    assert error.value.mutation_attempted is False
    assert stat.S_IMODE(case.workspace.stat().st_mode) == mode


@pytest.mark.parametrize("mode", [0o700, 0o775])
def test_unselected_unsafe_ancestor_is_not_implicitly_published(traversal_case, monkeypatch, mode):
    case = traversal_case
    case.anchor.chmod(mode)
    refuses_without_change(case, monkeypatch)


@pytest.mark.parametrize("field", ["st_uid", "st_gid"])
def test_selected_directory_requires_the_actual_owner_shared_group(traversal_case, monkeypatch, field):
    case = traversal_case
    expected = case.lane.stat()
    actual_stat, actual_fstat = os.stat, os.fstat
    def foreign(info):
        if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values[field] += 1
            return SimpleNamespace(**values)
        return info
    monkeypatch.setattr(traversal.os, "stat", lambda *args, **kwargs: foreign(actual_stat(*args, **kwargs)))
    monkeypatch.setattr(traversal.os, "fstat", lambda descriptor: foreign(actual_fstat(descriptor)))
    refuses_without_change(case, monkeypatch)


def test_group_execute_for_an_unrelated_group_does_not_establish_worker_search(traversal_case, monkeypatch):
    case = traversal_case
    expected = case.anchor.parent.stat()
    actual_stat, actual_fstat = os.stat, os.fstat
    def wrong_group(info):
        if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            # A protected root-owned test container is admissible to the owner
            # walk; its unrelated group-X cannot establish worker GID1000 search.
            values.update(st_uid=0, st_gid=os.getuid() + 1)
            return SimpleNamespace(**values)
        return info
    monkeypatch.setattr(traversal.os, "stat", lambda *args, **kwargs: wrong_group(actual_stat(*args, **kwargs)))
    monkeypatch.setattr(traversal.os, "fstat", lambda descriptor: wrong_group(actual_fstat(descriptor)))
    refuses_without_change(case, monkeypatch)


@pytest.mark.parametrize("name", ["system.posix_acl_access", "system.posix_acl_default"])
def test_acl_on_selected_directory_refuses_before_search_publication(traversal_case, monkeypatch, name):
    case = traversal_case
    expected = case.lane.stat()
    actual = os.listxattr
    def with_acl(descriptor):
        names = actual(descriptor)
        info = os.fstat(descriptor)
        return list(names) + [name] if (info.st_dev, info.st_ino) == (
            expected.st_dev, expected.st_ino) else names
    monkeypatch.setattr(traversal.os, "listxattr", with_acl)
    refuses_without_change(case, monkeypatch)


def test_selected_mount_identity_must_match_its_protected_parent(traversal_case, monkeypatch):
    case = traversal_case
    expected = case.lane.stat()
    actual = traversal._mount_identity
    def changed(descriptor):
        observed = actual(descriptor)
        info = os.fstat(descriptor)
        return (observed[0] + 1, observed[1]) if (info.st_dev, info.st_ino) == (
            expected.st_dev, expected.st_ino) else observed
    monkeypatch.setattr(traversal, "_mount_identity", changed)
    monkeypatch.setattr(parents, "_mount_identity", changed)
    refuses_without_change(case, monkeypatch)


def test_expired_original_deadline_refuses_before_any_search_change(traversal_case, monkeypatch):
    refuses_without_change(traversal_case, monkeypatch, absolute_deadline_ns=time.monotonic_ns() - 1)


def test_suppressed_recursive_preflight_invalidates_outer_publication(traversal_case, monkeypatch):
    case = traversal_case
    actual = traversal._preflight
    nested = []
    def recursive(**kwargs):
        plan = actual(**kwargs)
        with pytest.raises(traversal.OwnerWorktreeTraversalError):
            publish(case)
        nested.append(True)
        return plan
    monkeypatch.setattr(traversal, "_preflight", recursive)
    refuses_without_change(case, monkeypatch)
    assert nested == [True]


@pytest.mark.parametrize("fault", ["boundary_bytes", "selected_replacement"])
def test_actual_control_drift_after_preflight_refuses_before_fchmod(traversal_case, monkeypatch, fault):
    case = traversal_case
    actual = traversal._preflight
    calls = []
    retained = case.owner / "retained-lane"
    def changed(**kwargs):
        plan = actual(**kwargs)
        if fault == "boundary_bytes":
            case.boundary.chmod(0o600)
            case.boundary.write_bytes(b"changed authenticated boundary bytes\n")
            case.boundary.chmod(0o444)
        else:
            case.lane.rename(retained)
            case.lane.mkdir(mode=0o700)
            case.workspace.mkdir(mode=0o700)
        return plan
    monkeypatch.setattr(traversal, "_preflight", changed)
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", lambda *_args: calls.append(True))
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case)
    assert calls == [] and error.value.mutation_attempted is False
    assert stat.S_IMODE(case.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.lane.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700
    if fault == "boundary_bytes":
        assert case.boundary.read_bytes() == b"changed authenticated boundary bytes\n"
    else:
        assert (retained / "candidate/module.py").read_bytes() == b"SOURCE = 'unchanged'\n"


def test_partial_search_change_is_reported_without_rollback_or_workspace_enablement(traversal_case, monkeypatch):
    case = traversal_case
    actual = os.fchmod
    calls = []
    def failed(descriptor, mode):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("authored second traversal mutation failure")
        return actual(descriptor, mode)
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", failed)
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case)
    assert error.value.mutation_attempted is True
    assert error.value.changed_directories == (str(case.lane),)
    assert error.value.effects_complete is False
    assert stat.S_IMODE(case.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.lane.stat().st_mode) == 0o710
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_fchmod_that_applies_then_raises_retains_uncertain_parent_effects(traversal_case, monkeypatch):
    case = traversal_case
    actual = os.fchmod
    def applied(descriptor, mode):
        actual(descriptor, mode)
        raise OSError("authored failure after traversal mode changed")
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", applied)
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case)
    assert error.value.mutation_attempted is True
    assert error.value.changed_directories == ()
    assert error.value.effects_complete is False
    assert stat.S_IMODE(case.lane.stat().st_mode) == 0o710
    assert stat.S_IMODE(case.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_sibling_added_after_actual_change_prevents_completed_publication(traversal_case, monkeypatch):
    case = traversal_case
    actual = os.fchmod
    added = []
    def changed(descriptor, mode):
        result = actual(descriptor, mode)
        if not added:
            (case.root / "late-readable-sibling").mkdir(mode=0o750)
            added.append(True)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", changed)
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case)
    assert added == [True]
    assert error.value.mutation_attempted is True
    assert error.value.effects_complete is False
    assert stat.S_IMODE(case.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_deadline_after_first_change_keeps_outer_parent_and_workspace_private(traversal_case, monkeypatch):
    case = traversal_case
    deadline = time.monotonic_ns() + 30_000_000_000
    actual = os.fchmod
    changed = []
    def expired(descriptor, mode):
        result = actual(descriptor, mode)
        changed.append(True)
        monkeypatch.setattr(parents.time, "monotonic_ns", lambda: deadline + 1)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(traversal.os, "fchmod", expired)
        with tracked_descriptors(patch):
            with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
                publish(case, absolute_deadline_ns=deadline)
    assert changed == [True]
    assert error.value.mutation_attempted is True and error.value.effects_complete is False
    assert stat.S_IMODE(case.lane.stat().st_mode) == 0o710
    assert stat.S_IMODE(case.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700


def test_suppressed_reentry_after_final_direct_directory_open_refuses_before_fchmod(
        traversal_case, monkeypatch):
    case = traversal_case
    actual = traversal._open_directory
    injected = []
    def opened(plan, index, *, deadline):
        descriptor = actual(plan, index, deadline=deadline)
        caller = inspect.currentframe().f_back
        try:
            if (not injected and caller.f_code.co_name == "provision_owner_worktree_traversal"
                    and caller.f_locals["state"]["phase"] == "parent_search"):
                with pytest.raises(traversal.OwnerWorktreeTraversalError):
                    publish(case)
                injected.append(True)
        finally:
            del caller
        return descriptor
    monkeypatch.setattr(traversal, "_open_directory", opened)
    refuses_without_change(case, monkeypatch)
    assert injected == [True]


def test_deadline_expiring_after_final_closing_refuses_a_success_receipt(
        traversal_case, monkeypatch):
    case = traversal_case
    deadline = time.monotonic_ns() + 30_000_000_000
    actual = traversal._closing
    expired = []
    def closed(plan, *, deadline):
        result = actual(plan, deadline=deadline)
        caller = inspect.currentframe().f_back
        try:
            if (caller.f_code.co_name == "provision_owner_worktree_traversal"
                    and caller.f_locals["state"]["phase"] == "closing_search"):
                expired.append(True)
                monkeypatch.setattr(parents.time, "monotonic_ns", lambda: deadline + 1)
        finally:
            del caller
        return result
    monkeypatch.setattr(traversal, "_closing", closed)
    with tracked_descriptors(monkeypatch):
        with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
            publish(case, absolute_deadline_ns=deadline)
    assert expired == [True]
    assert error.value.mutation_attempted is True
    assert error.value.changed_directories == (str(case.lane), str(case.root))
    assert error.value.effects_complete is False
    assert error.value.phase == "closing_search"
    assert stat.S_IMODE(case.root.stat().st_mode) == stat.S_IMODE(case.lane.stat().st_mode) == 0o710
    assert stat.S_IMODE(case.workspace.stat().st_mode) == 0o700
