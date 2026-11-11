"""Preprovisioned writable linked-worktree custody and a real second child.

Native task claims, Portal projections, lifecycle files, Git control identity,
worker census, quantitative subleases and child births remain real. The shared
fixture authors its existing host deployment boundary and Quack transport. It
does not qualify a container, a different worker UID, permission provisioning,
external providers, or crash-safe ownership transfer.
"""
from contextlib import contextmanager
import hashlib
import json
import os
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_worktree_allocation as allocation
from ipfs_accelerate_py.agent_supervisor.runtime import native_writable_allocation as writable
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
from test.api.candidate_resource_test_support import lease_record, resource_parent
from test.api.test_current_source_worktree_allocation import _record_change
from test.api.test_native_candidate_continuation import (
    children, prepared, run_options,
)
from test.api.test_native_candidate_round_admission import _candidate_case, capture


REFUSAL = (ValueError, RuntimeError, TimeoutError, TypeError)


def provision(allocated):
    """Author deployment state before the first attempt captures its identity."""
    allocated.native.workspace.chmod(0o3770)
    (allocated.native.workspace / ".git").chmod(0o440)


def delegate(allocated, prior):
    allocated.strict_guard = allocated.guard
    allocated.first_prepared_attempt = prior
    return allocation.capture_native_delegated_source_worktree_allocation(
        first_worker_attempt=prior, **allocated.capture)


@contextmanager
def delegated_case(tmp_path, monkeypatch, *, provision_workspace=provision,
        capture_allocation=delegate):
    def provision_with_host_boundary(allocated):
        provision_workspace(allocated)
        # The inherited native fixture writes this host-authored boundary
        # using process umask. Provision its immutable mode before preparing
        # the first census; the delegated guard must never normalize it.
        allocated.native.factory["boundary"].chmod(0o444)
        admitted_boundary = boundary_guard._owner_boundary
        def owner_boundary(path):
            value, digest = admitted_boundary(path)
            return {**value, "allowed_worktree_roots": [str(allocated.capture["worktree_root"])]}, digest
        monkeypatch.setattr(boundary_guard, "_owner_boundary", owner_boundary)
    with _candidate_case(tmp_path, monkeypatch,
            provision_workspace=provision_with_host_boundary,
            capture_allocation=capture_allocation) as case:
        yield case


def binding_arguments(case, **changes):
    prior = json.loads(case.prior_worker_attempt._common)
    return dict(prior_worker_attempt=case.prior_worker_attempt,
        source_broker=case.source_broker,
        absolute_deadline_ns=prior["absolute_deadline_ns"]) | changes


def assert_current_refuses(case):
    with pytest.raises(REFUSAL):
        case.allocation.require_current()
    with pytest.raises(REFUSAL):
        capture(case)
    assert len(case.children) == 1


def marker_identity(info):
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode",
        "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


@pytest.mark.parametrize("code", [0, 7])
def test_real_delegated_continuation_keeps_root_marker_and_native_capacity(
        tmp_path, monkeypatch, resource_parent, code):
    with delegated_case(tmp_path, monkeypatch) as case:
        owner = resource_parent
        root, marker = case.native.workspace, case.native.workspace / ".git"
        root_before = root.stat()
        marker_before, marker_bytes = marker_identity(marker.stat()), marker.read_bytes()
        parent_before = lease_record(owner, owner.parent.lease_id)
        with prepared(case, owner) as session:
            prompt = session.candidate_prompt.provider_prompt
            options, observed = run_options(session, tmp_path, code=code)
            result = session.run(**options)
            assert result.returncode == code
            received = json.loads(observed.read_bytes())
            assert received["bytes"] == len(prompt)
            assert received["sha256"] == hashlib.sha256(prompt).hexdigest()
            assert session.metadata["observed_worker_children"] == 2
            assert session.metadata["resource_capacity_held"] is False
            assert children(owner) == {}
        parent_after = lease_record(owner, owner.parent.lease_id)
        renewable = {"heartbeat_at", "expires_at"}
        assert {key: value for key, value in parent_after.items() if key not in renewable} == {
            key: value for key, value in parent_before.items() if key not in renewable}
        assert all(parent_after[key] >= parent_before[key] for key in renewable)
        assert (root.stat().st_dev, root.stat().st_ino, root.stat().st_uid,
            root.stat().st_gid, stat.S_IMODE(root.stat().st_mode)) == (
                root_before.st_dev, root_before.st_ino, root_before.st_uid,
                root_before.st_gid, 0o3770)
        assert marker_identity(marker.stat()) == marker_before and marker.read_bytes() == marker_bytes
        assert case.native.source.get_task(case.native.task_cid).status == "in_progress"


def test_exact_delegated_capture_preserves_the_strict_allocation_contract(tmp_path, monkeypatch):
    with delegated_case(tmp_path, monkeypatch) as case:
        guard = case.allocation
        assert type(guard) is allocation.NativeDelegatedSourceWorktreeAllocation
        material = guard.require_current()
        assert material["schema"] == allocation.DELEGATED_SCHEMA
        assert material["task_cid"] == case.native.task_cid
        assert material["workspace"] == str(case.native.workspace)
        assert material["allocation_ownership_verified"] is True
        assert material["proof_authority"] is material["completion_authority"] is False
        delegation = material["writable_delegation"]
        assert delegation["first_worker_attempt_id"] == case.prior_worker_attempt.attempt_id
        assert delegation["current_workspace_policy_verified"] is True
        assert delegation["access_and_default_acl_absent"] is True
        assert delegation["permission_transition_verified"] is False
        assert delegation["nested_workspace_policy_verified"] is False
        assert delegation["worker_isolation_qualified"] is False
        guard.require_candidate_binding(**binding_arguments(case))
        with pytest.raises(REFUSAL):
            case.allocation_case.strict_guard.require_current()
        with pytest.raises(REFUSAL):
            allocation.capture_native_source_worktree_allocation(**case.allocation_case.capture)
        assert stat.S_IMODE(case.native.workspace.stat().st_mode) == 0o3770
        assert stat.S_IMODE((case.native.workspace / ".git").stat().st_mode) == 0o440


@pytest.mark.parametrize("mode", [0o700, 0o2770, 0o3777])
def test_capture_refuses_any_root_mode_outside_the_preprovisioned_contract(
        tmp_path, monkeypatch, mode):
    observed = []
    def changed(allocated):
        provision(allocated)
        allocated.native.workspace.chmod(mode)
    def refused(allocated, prior):
        observed.append(prior)
        return delegate(allocated, prior)
    with pytest.raises(REFUSAL):
        with delegated_case(tmp_path, monkeypatch,
                provision_workspace=changed, capture_allocation=refused):
            pytest.fail("unsafe delegated root was captured")
    assert len(observed) == 1 and observed[0]._state.process is None


@pytest.mark.parametrize("mode", [0o600, 0o660])
def test_capture_requires_the_existing_read_only_owner_git_marker(tmp_path, monkeypatch, mode):
    observed = []
    def changed(allocated):
        provision(allocated)
        (allocated.native.workspace / ".git").chmod(mode)
    def refused(allocated, prior):
        observed.append(prior)
        return delegate(allocated, prior)
    with pytest.raises(REFUSAL):
        with delegated_case(tmp_path, monkeypatch,
                provision_workspace=changed, capture_allocation=refused):
            pytest.fail("unsafe delegated Git marker was captured")
    assert len(observed) == 1 and observed[0]._state.process is None


@pytest.mark.parametrize("alleged", [None, {}, {"prepared": True}])
def test_capture_does_not_adopt_serialized_or_alleged_first_attempts(
        tmp_path, monkeypatch, alleged):
    observed = []
    def refused(allocated, prior):
        observed.append(prior)
        return allocation.capture_native_delegated_source_worktree_allocation(
            first_worker_attempt=alleged, **allocated.capture)
    with pytest.raises(REFUSAL):
        with delegated_case(tmp_path, monkeypatch, capture_allocation=refused):
            pytest.fail("alleged first attempt was adopted")
    assert len(observed) == 1 and observed[0]._state.process is None


def test_capture_refuses_a_hard_linked_git_marker(tmp_path, monkeypatch):
    observed = []
    def refused(allocated, prior):
        observed.append(prior)
        marker = allocated.native.workspace / ".git"
        other = tmp_path / "foreign-linked-worktree-marker"
        os.link(marker, other)
        try:
            return delegate(allocated, prior)
        finally:
            other.unlink()
    with pytest.raises(REFUSAL):
        with delegated_case(tmp_path, monkeypatch, capture_allocation=refused):
            pytest.fail("multiply linked delegated Git marker was accepted")
    assert len(observed) == 1 and observed[0]._state.process is None


def test_capture_cannot_rebase_after_the_actual_first_child_birth(tmp_path, monkeypatch):
    with delegated_case(tmp_path, monkeypatch) as case:
        assert case.prior_worker_attempt._state.spent is True
        with pytest.raises(REFUSAL):
            allocation.capture_native_delegated_source_worktree_allocation(
                first_worker_attempt=case.prior_worker_attempt, **case.allocation_case.capture)
        case.allocation.require_current()


@pytest.mark.parametrize("name", ["system.posix_acl_access", "system.posix_acl_default"])
def test_capture_refuses_an_acl_on_the_delegated_root(tmp_path, monkeypatch, name):
    observed = []
    actual_list = os.listxattr
    def refused(allocated, prior):
        observed.append(prior)
        identity = allocated.native.workspace.stat()
        def with_acl(descriptor, *args, **kwargs):
            result = actual_list(descriptor, *args, **kwargs)
            if type(descriptor) is int:
                info = os.fstat(descriptor)
                if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
                    return list(result) + [name]
            return result
        with monkeypatch.context() as patch:
            patch.setattr(writable.os, "listxattr", with_acl)
            return delegate(allocated, prior)
    with pytest.raises(REFUSAL):
        with delegated_case(tmp_path, monkeypatch, capture_allocation=refused):
            pytest.fail("delegated ACL was accepted")
    assert len(observed) == 1 and observed[0]._state.process is None


@pytest.mark.parametrize("changed", ["root_mode", "marker_mode", "marker_replacement",
    "canonical_mode", "portal_projection", "lifecycle", "grant"])
def test_current_delegation_refuses_actual_control_or_native_authority_drift(
        tmp_path, monkeypatch, changed):
    with delegated_case(tmp_path, monkeypatch) as case:
        if changed in {"root_mode", "marker_mode", "canonical_mode"}:
            path = {"root_mode": case.native.workspace,
                "marker_mode": case.native.workspace / ".git",
                "canonical_mode": case.native.canonical}[changed]
            original = stat.S_IMODE(path.stat().st_mode)
            path.chmod({"root_mode": 0o2770, "marker_mode": 0o660,
                "canonical_mode": 0o770}[changed])
            try:
                assert_current_refuses(case)
            finally:
                path.chmod(original)
        elif changed == "marker_replacement":
            path = case.native.workspace / ".git"
            retained = path.with_name(".git.original-delegation-test")
            raw = path.read_bytes()
            path.rename(retained)
            path.write_bytes(raw)
            path.chmod(0o440)
            try:
                assert_current_refuses(case)
            finally:
                path.unlink()
                retained.rename(path)
        elif changed == "portal_projection":
            path = case.allocation_case.paths.task_projection
            raw, mode = path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o600)
            path.write_bytes(raw + b"\nauthored delegated projection drift\n")
            path.chmod(mode)
            try:
                assert_current_refuses(case)
            finally:
                path.chmod(0o600)
                path.write_bytes(raw)
                path.chmod(mode)
        elif changed == "lifecycle":
            with _record_change(case.allocation_case,
                    lambda value: value.update(state="settling")):
                assert_current_refuses(case)
        else:
            case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
            assert_current_refuses(case)


def test_current_delegation_pins_mount_identity_after_capture(tmp_path, monkeypatch):
    with delegated_case(tmp_path, monkeypatch) as case:
        actual = writable._mount_identity
        def changed(descriptor):
            return (*actual(descriptor), "authored changed mount identity")
        monkeypatch.setattr(writable, "_mount_identity", changed)
        assert_current_refuses(case)


def test_current_delegation_rejects_a_byte_identical_replacement_boundary(tmp_path, monkeypatch):
    with delegated_case(tmp_path, monkeypatch) as case:
        path = case.boundary
        original = path.with_name(path.name + ".original-delegation-test")
        raw, mode = path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
        path.rename(original)
        path.write_bytes(raw)
        path.chmod(mode)
        try:
            assert_current_refuses(case)
        finally:
            path.unlink()
            original.rename(path)


@pytest.mark.parametrize("changed", ["prior", "deadline", "source_broker"])
def test_delegated_candidate_join_requires_the_original_first_attempt_broker_and_deadline(
        tmp_path, monkeypatch, changed):
    with delegated_case(tmp_path, monkeypatch) as case:
        substitutions = {"prior": dict(prior_worker_attempt=case.worker_attempt),
            "deadline": dict(absolute_deadline_ns=json.loads(
                case.prior_worker_attempt._common)["absolute_deadline_ns"] + 1),
            "source_broker": dict(source_broker={"claimed_current": True})}[changed]
        with pytest.raises(REFUSAL):
            case.allocation.require_candidate_binding(**binding_arguments(case, **substitutions))
        assert len(case.children) == 1


def test_delegation_keeps_native_source_binding_to_the_actual_linked_worktree(tmp_path, monkeypatch):
    observed = []
    def bound(allocated, prior):
        guard = delegate(allocated, prior)
        result = guard.bind_current_source_lookup(**allocated.source_binding)
        service = guard._lookup
        guard.require_source_binding(server=allocated.native.server, service=service)
        assert service._root == allocated.native.workspace
        assert getattr(service, "_native_source_worktree_allocation") is guard
        assert result["task_cid"] == allocated.native.task_cid
        observed.append((guard, service))
        return guard
    with delegated_case(tmp_path, monkeypatch, capture_allocation=bound) as case:
        assert len(observed) == 1 and observed[0][0] is case.allocation
        case.allocation.require_current()
        # First-child source changes cannot silently rebase the original native
        # reader, even while the delegated allocation itself remains current.
        with pytest.raises(REFUSAL):
            observed[0][1]._check_source()
