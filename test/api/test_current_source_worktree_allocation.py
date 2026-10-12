"""Real Portal projections and lifecycle custody for native source retrieval.

The native task, claim, grant, Portal projection, Git allocation and lifecycle
records are real. The inherited Quack transport and worker deployment metadata
are authored host seams. These tests do not qualify remote launch, bootstrap
provenance, a separate worker UID, model execution, or formal reconstruction.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle as lifecycle
from ipfs_accelerate_py.agent_supervisor.runtime import current_source_worktree_allocation as allocation
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import infer_board_namespace
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from test.api.test_current_source_task_admission import ALIAS, SOURCE, _native_case, _payload
from test.api.test_repository_source_inventory import _git


@dataclass(repr=False)
class AllocationCase:
    """Keep owner resources and credentials out of assertion representations."""

    native: object
    guard: object
    store: object
    record: object
    bridge: object
    paths: object
    portal_binding: dict
    portal_identity: dict
    target: str
    branch: str
    capture: dict
    source_binding: dict


def _protect_control_tree(root):
    """Trusted fixture provisions existing authority before its first capture."""
    root.chmod(0o700)
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(base) / name
            assert not path.is_symlink()
            path.chmod(0o700)
        for name in files:
            path = Path(base) / name
            assert path.is_file() and not path.is_symlink()
            path.chmod(path.stat().st_mode & 0o755 & ~0o022)


@contextmanager
def _allocation_case(tmp_path, monkeypatch):
    with _native_case(tmp_path, monkeypatch, defer_source_binding=True) as native:
        target = _git(native.canonical, "branch", "--show-current")
        branch = "implementation/current-source-allocation"
        attempt_root = tmp_path / "portal-attempts"
        namespace = infer_board_namespace(merge_target_branch=target,
            todo_path=Path(native.server.config.database_path), state_prefix="source-admission")

        def no_provider(*_args, **_kwargs):
            pytest.fail("allocation fixture unexpectedly dispatched a provider")

        bridge = DatabasePortalExecutionBridge(task_source=native.source,
            attempt_root=attempt_root, portal_factory=no_provider,
            repo_root=native.canonical, board_namespace=namespace,
            merge_target_branch=target, task_header_prefix="## " + ALIAS)
        attempt = replace(native.attempt, body={**dict(native.attempt.body),
            "control_binding": DatabaseImplementationDaemon._control_claim_binding(
                native.attempt, native.native_task)})
        paths, portal_binding = bridge._ensure_attempt_projection(attempt, native.native_task)
        portal_identity = bridge._prior_projection_identity(paths, portal_binding)
        store = lifecycle.WorktreeLifecycleStore(native.canonical)
        record = store.begin_preparing(task_id=ALIAS,
            canonical_task_cid=portal_identity["canonical_task_cid"],
            attempt=(1 << 52) | (native.attempt.attempt_number << 16) | 1,
            lane_id=str(paths.root) + ":0/1:" + str(os.getpid()),
            workspace_path=native.workspace, branch=branch, merge_target=target,
            state_dir=str(paths.root))
        session = None
        try:
            assert not native.workspace.exists()
            _git(native.canonical, "worktree", "add", "-q", "-b", branch,
                str(native.workspace), native.retained.request.expected_commit)
            record = store.mark_active(native.workspace, lease_id=record.lease_id,
                expected_fence=record.fence)
            # The real native creator follows process umask. This host fixture
            # provisions protected owner authority before capture, rather than
            # allowing the guard to normalize an untrusted path or mode.
            tmp_path.chmod(0o700)
            native.canonical.chmod(0o700)
            native.workspace.chmod(0o700)
            (native.workspace / ".git").chmod(0o600)
            _protect_control_tree(native.canonical / ".git")
            _protect_control_tree(attempt_root)
            capture = dict(task_admission=native.handle,
                canonical_repository=native.canonical, workspace=native.workspace,
                worktree_root=tmp_path, attempt_root=attempt_root,
                merge_target_branch=target, board_namespace=namespace)
            guard = allocation.capture_native_source_worktree_allocation(**capture)
            source_binding = dict(cas=native.retained.cas,
                expected_inventory_cid=native.retained.ref.inventory_cid,
                request=native.retained.request, limits=native.retained.limits,
                context_cid=cid_for_structured({"allocation_context": native.task_cid}),
                allowed_paths=(b"module.py".hex(),), max_response_bytes=4096,
                max_slice_bytes=512)
            native.binding = guard.bind_current_source_lookup(**source_binding)
            native.factory["worktree_admission"] = guard
            session = broker.start_source_lookup_worker_broker(**native.factory)
            native.session = session
            yield AllocationCase(native, guard, store, record, bridge, paths,
                dict(portal_binding), dict(portal_identity), target, branch, capture, source_binding)
        finally:
            if session is not None:
                session.close()
            current = store.load_workspace(native.workspace)
            if current is not None and not current.is_terminal:
                store.mark_terminal(native.workspace, lease_id=current.lease_id,
                    expected_fence=current.fence, reason="current-source-allocation-test-complete")
            if native.workspace.exists():
                _git(native.canonical, "worktree", "remove", "--force", str(native.workspace))
            _git(native.canonical, "branch", "-D", branch)


@pytest.fixture
def allocated(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch) as case:
        yield case


def _record_path(case):
    return case.store.workspace_path_for(case.native.workspace)


def _index_path(case):
    return case.store.task_index_path_for(canonical_task_cid=case.record.canonical_task_cid,
        task_id=case.record.task_id, attempt=case.record.attempt)


@contextmanager
def _record_change(case, change):
    """Retain authentic file bytes, with self-consistent forged digest/index."""
    record_path, index_path = _record_path(case), _index_path(case)
    before, index_before = record_path.read_bytes(), index_path.read_bytes()
    value = json.loads(before)
    change(value)
    record = lifecycle.WorkspaceLifecycleRecord.from_dict(value)
    record = replace(record, record_id=record.compute_record_id())
    record_path.write_text(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False))
    index_path.write_text(json.dumps(case.store._task_index_payload(record), sort_keys=True,
        allow_nan=False))
    try:
        yield
    finally:
        record_path.write_bytes(before)
        index_path.write_bytes(index_before)


def _assert_refused(case):
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.require_current()
    with pytest.raises((ValueError, RuntimeError)):
        case.native.session._read_source(_payload(case.native))
    assert case.native.session._native_response_bytes == 0


def test_actual_portal_allocation_joins_distinct_native_and_portal_identities(allocated):
    case, native = allocated, allocated.native
    material = case.guard.require_current()
    assert material["task_cid"] == native.task_cid
    assert material["portal_canonical_task_cid"] == case.portal_identity["canonical_task_cid"]
    assert native.task_cid != case.portal_identity["canonical_task_cid"]
    assert case.record.lease_id != native.attempt.lease_id
    assert material["workspace"] == str(native.workspace)
    assert material["branch"] == case.branch
    assert material["allocation_ownership_verified"] is True
    assert material["proof_authority"] is material["completion_authority"] is False
    assert native.binding["task_cid"] == native.task_cid
    assert native.binding["expected_task_body_cid"] == content_identity(dict(native.native_task.body))
    response = native.session._read_source(_payload(native))
    assert response["content"].encode() == SOURCE
    assert response["read_only"] is True


@pytest.mark.parametrize("field", ["task_id", "canonical_task_cid", "attempt", "branch",
    "workspace_path", "repo_root", "state_dir", "lane_id", "owner", "lease_id"])
def test_self_consistent_foreign_lifecycle_tuple_refuses(allocated, field):
    def change(value):
        if field == "owner":
            value[field]["start_time_ticks"] += 1
        elif field == "attempt":
            value[field] += 65536
        else:
            value[field] = "changed-unbound"
    with _record_change(allocated, change):
        _assert_refused(allocated)


@pytest.mark.parametrize("state", ["preparing", "settling", "terminal"])
def test_nonactive_lifecycle_state_refuses(allocated, state):
    with _record_change(allocated, lambda value: value.update(state=state)):
        _assert_refused(allocated)


def test_expired_lifecycle_lease_refuses_current_native_claim(allocated):
    with _record_change(allocated, lambda value: value.update(expires_at=time.time() - 1)):
        assert allocated.native.handle.require_current()["task_cid"] == allocated.native.task_cid
        _assert_refused(allocated)


def test_exact_lifecycle_index_required_with_authentic_workspace_record(allocated):
    path = _index_path(allocated)
    before = path.read_bytes()
    value = json.loads(before)
    value["workspace_path"] += "-foreign"
    path.write_text(json.dumps(value))
    try:
        _assert_refused(allocated)
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize("name", ["binding", "task_projection"])
def test_changed_portal_projection_or_binding_refuses(allocated, name):
    path = getattr(allocated.paths, name)
    before, mode = path.read_bytes(), path.stat().st_mode & 0o777
    path.chmod(0o600)
    path.write_bytes(before + b"\nchanged-unbound\n")
    path.chmod(mode)
    try:
        _assert_refused(allocated)
    finally:
        path.chmod(0o600)
        path.write_bytes(before)
        path.chmod(mode)


@pytest.mark.parametrize("name", ["binding", "task_projection", "git_marker"])
def test_byte_identical_immutable_control_file_replacement_refuses(allocated, name):
    path = (allocated.native.workspace / ".git" if name == "git_marker"
        else getattr(allocated.paths, name))
    original = path.with_name(path.name + ".original-test")
    replacement = path.with_name(path.name + ".replacement-test")
    before, mode = path.read_bytes(), path.stat().st_mode & 0o777
    path.rename(original)
    replacement.write_bytes(before)
    replacement.chmod(mode)
    replacement.rename(path)
    try:
        _assert_refused(allocated)
    finally:
        path.unlink()
        original.rename(path)


def test_real_lifecycle_renewal_keeps_allocation_identity_without_rebasing(allocated):
    before = allocated.guard.require_current()
    record = allocated.store.load_workspace(allocated.native.workspace)
    renewed = allocated.store.renew_lease(allocated.native.workspace,
        lease_id=record.lease_id, expected_fence=record.fence)
    assert renewed.fence == record.fence + 1
    assert renewed.expires_at >= record.expires_at
    assert allocated.guard.require_current() == before
    assert allocated.native.session._read_source(_payload(allocated.native))["content"].encode() == SOURCE


def test_lifecycle_fence_rollback_refuses_after_genuine_renewal(allocated):
    path, index_path = _record_path(allocated), _index_path(allocated)
    before, index_before = path.read_bytes(), index_path.read_bytes()
    record = allocated.store.load_workspace(allocated.native.workspace)
    renewed = allocated.store.renew_lease(allocated.native.workspace,
        lease_id=record.lease_id, expected_fence=record.fence)
    assert renewed.fence > record.fence
    allocated.guard.require_current()
    current, index_current = path.read_bytes(), index_path.read_bytes()
    path.write_bytes(before)
    index_path.write_bytes(index_before)
    try:
        _assert_refused(allocated)
    finally:
        path.write_bytes(current)
        index_path.write_bytes(index_current)


@pytest.mark.parametrize("substitution", ["omitted", "snapshot"])
def test_bound_allocation_cannot_be_omitted_or_replaced_before_proxy_grant(allocated, monkeypatch, substitution):
    factory = dict(allocated.native.factory)
    if substitution == "omitted":
        factory.pop("worktree_admission")
    else:
        factory["worktree_admission"] = allocated.guard.require_current()
    monkeypatch.setattr(allocated.native.server, "issue_typed_client_grant_record",
        lambda **_kwargs: pytest.fail("invalid allocation issued a new proxy grant"))
    with pytest.raises((ValueError, RuntimeError)):
        broker.start_source_lookup_worker_broker(**factory)
    assert allocated.native.session._read_source(_payload(allocated.native))["content"].encode() == SOURCE


def test_serialized_task_snapshot_and_unsealed_allocation_cannot_capture_authority(allocated):
    capture = dict(allocated.capture)
    capture["task_admission"] = allocated.native.handle.require_current()
    with pytest.raises((ValueError, RuntimeError)):
        allocation.capture_native_source_worktree_allocation(**capture)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        allocation.NativeSourceWorktreeAllocation(object(), **allocated.capture)


@pytest.mark.parametrize("field", ["proof_authority", "workspace"])
def test_cached_material_cannot_upgrade_authority_or_rebind_allocation(allocated, field):
    original = allocated.guard._material
    value = json.loads(original)
    value[field] = True if field == "proof_authority" else str(allocated.native.workspace) + "-foreign"
    value.pop("allocation_cid")
    value["allocation_cid"] = cid_for_structured(value)
    allocated.guard._material = canonical_json_bytes(value)
    try:
        _assert_refused(allocated)
    finally:
        allocated.guard._material = original


def test_refused_second_binding_cannot_adopt_or_remark_existing_reader(allocated):
    native = allocated.native
    original_service = native.server._command_gateway._current_source_lookup
    second = allocation.capture_native_source_worktree_allocation(**allocated.capture)
    with pytest.raises((ValueError, RuntimeError)):
        second.bind_current_source_lookup(**allocated.source_binding)
    assert native.server._command_gateway._current_source_lookup is original_service
    assert original_service._native_source_worktree_allocation is allocated.guard
    assert second._lookup is None
    assert native.session._read_source(_payload(native))["content"].encode() == SOURCE


def test_lifecycle_loss_after_native_reply_retains_incurred_backend_cost(allocated, monkeypatch):
    native = allocated.native
    path, index_path = _record_path(allocated), _index_path(allocated)
    before, index_before = path.read_bytes(), index_path.read_bytes()
    original_get = native.session._client.source_get

    def response_then_expire(payload):
        result = original_get(payload)
        value = json.loads(path.read_bytes())
        value["expires_at"] = time.time() - 1
        path.write_text(json.dumps(value))
        return result

    monkeypatch.setattr(native.session._client, "source_get", response_then_expire)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            native.session._read_source(_payload(native))
        assert native.session._native_response_bytes > 0
        assert native.session._native_wire_bytes > native.session._native_response_bytes
    finally:
        path.write_bytes(before)
        index_path.write_bytes(index_before)


def test_lifecycle_loss_at_closing_coordinator_callback_cannot_escape_detached_check(allocated, monkeypatch):
    native = allocated.native
    path = _record_path(allocated)
    before = path.read_bytes()
    state = {"remaining": None, "edited": False}
    original_clock = type(native.clock).__call__
    original_get = native.session._client.source_get

    def clock_with_late_lease_loss(clock):
        now = original_clock(clock)
        if clock is native.clock and state["remaining"] is not None:
            state["remaining"] -= 1
            if state["remaining"] == 0:
                value = json.loads(path.read_bytes())
                value["expires_at"] = time.time() - 1
                path.write_text(json.dumps(value))
                state["remaining"], state["edited"] = None, True
        return now

    def arm_after_read(payload):
        response = original_get(payload)
        # One allocation check performs the native task fence (two real clock
        # reads), then its own allocation fence (two more). Expire the lifecycle
        # on the closing fence's post-callback read, after allocation observation.
        state["remaining"] = 8
        return response

    monkeypatch.setattr(type(native.clock), "__call__", clock_with_late_lease_loss)
    monkeypatch.setattr(native.session._client, "source_get", arm_after_read)
    assert native.daemon.coordinator._clock_ms is native.clock
    try:
        with pytest.raises((ValueError, RuntimeError)):
            native.session._read_source(_payload(native))
        assert state["edited"] is True
        assert native.session._native_response_bytes > 0
        assert native.session._native_wire_bytes == native.session._native_response_bytes + 3
    finally:
        path.write_bytes(before)
