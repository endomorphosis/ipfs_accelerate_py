"""Read-only permission custody through real native claims and allocations.

The retained fixture has authored FakeQuackTransport/deployment seams. No
worker UID, child birth, container, model or target code is executed here.
"""
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_handoff as owner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_prelaunch as prelaunch
from test.api.test_current_source_task_admission import SOURCE, _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources
from test.api.test_native_nominated_source_allocation import bind, nominated_allocation  # noqa: F401

PROMPT = "Observe the original selected source; Unicode λ; no edits."
REFUSALS = (ValueError, RuntimeError, OSError)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True) as case:
        case.native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
        provision_authored_prelaunch_sources(case)
        assignment = prelaunch.capture_repository_world_prelaunch_assignment(
            task_admission=case.native.handle, worktree_admission=case.guard, prompt=PROMPT)
        yield case, assignment


def capture(prepared, **changes):
    case, assignment = prepared
    arguments = dict(assignment=assignment, worker_uid=os.geteuid() + 1,
                     worker_gid=case.native.workspace.stat().st_gid)
    return owner.capture_repository_world_worker_handoff(**(arguments | changes))


def protect_claim(case):
    coordinator = case.native.daemon.coordinator
    claim = coordinator.get_task_claim(case.native.attempt.claim_id)
    return coordinator, claim, coordinator.protect_task_claim(claim)


def test_only_allocated_root_mode_changes_while_strict_originals_remain_strict(prepared):
    case, assignment = prepared
    root = case.native.workspace
    paths = (case.native.canonical / "module.py", root / "module.py", root / ".git",
             case.paths.binding, case.paths.task_projection)
    before = [(p.stat(), p.read_bytes()) for p in paths]
    allocation = case.guard.require_current()
    gate = assignment.require_current()
    handoff = capture(prepared)
    try:
        first = handoff.require_current()
        assert first["state"] == "CAPTURED" and first["read_only_transition_verified"] is False
        current = handoff.apply_readonly_transition()
        assert current["state"] == "TRANSITIONED" and stat.S_IMODE(root.stat().st_mode) == 0o750
        assert current["source_continuity_verified"] is current["native_assignment_continuity_verified"] is True
        assert current["read_only_transition_verified"] is True
        assert current["allocation_cid"] == allocation["allocation_cid"]
        assert current["original_assignment_cid"] == gate["assignment_cid"]
        assert current["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
        assert current["timeout_nanoseconds"] == 90_000_000_000 and "timeout_seconds" not in current
        assert current["selected_source_count"] == 1 and current["selected_source_bytes"] == len(SOURCE)
        assert all(current[field] is False for field in owner.FALSE_FIELDS)
        assert all(owner._file_identity(p.stat()) == owner._file_identity(info) and p.read_bytes() == raw
                   for p, (info, raw) in zip(paths, before))
        assert handoff.require_current() == current
        with pytest.raises(ValueError):
            assignment.require_current()
        with pytest.raises(ValueError):
            case.guard.require_current()
        assert not root.stat().st_mode & 0o022
        wire = json.dumps(current)
        assert PROMPT not in wire and SOURCE.decode() not in wire
        assert str(root) not in wire and str(case.native.canonical) not in wire
    finally:
        closed = handoff.close()
    assert closed["state"] == "CLOSED" and closed["source_continuity_verified"] is False
    assert stat.S_IMODE(root.stat().st_mode) == 0o750  # No rollback represented as authority.


def test_public_membership_binds_original_native_objects_and_literal_prompt(prepared):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        expected = handoff.require_current()
        assert handoff.require_observer_assignment(prompt=PROMPT.encode(), worker_uid=expected["worker_uid"],
                                                   worker_gid=expected["worker_gid"]) == expected
        assert handoff.require_peer_assignment(server=case.native.server,
            canonical_repository=case.native.canonical, workspace=case.native.workspace,
            prompt=PROMPT, task_admission=case.native.handle, worktree_admission=case.guard,
            worker_uid=expected["worker_uid"], worker_gid=expected["worker_gid"]) == expected
        expected["world_dispatch_ready"] = True
        assert handoff.require_current()["world_dispatch_ready"] is False
    finally:
        handoff.close()


@pytest.mark.parametrize("field", ["prompt", "worker_uid", "worker_gid"])
def test_observer_membership_rejects_wrong_binding_without_worker_claim(prepared, field):
    handoff = capture(prepared)
    try:
        observed = handoff.require_current()
        args = dict(prompt=PROMPT, worker_uid=observed["worker_uid"], worker_gid=observed["worker_gid"])
        args[field] = "different" if field == "prompt" else observed[field] + 1
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="assignment_mismatch"):
            handoff.require_observer_assignment(**args)
    finally:
        handoff.close()


@pytest.mark.parametrize("field", ["server", "canonical_repository", "workspace", "task_admission", "worktree_admission"])
def test_peer_membership_refuses_foreign_original_assignment(prepared, field):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        observed = handoff.require_current()
        args = dict(server=case.native.server, canonical_repository=case.native.canonical,
            workspace=case.native.workspace, prompt=PROMPT, task_admission=case.native.handle,
            worktree_admission=case.guard, worker_uid=observed["worker_uid"], worker_gid=observed["worker_gid"])
        args[field] = Path("/foreign") if field in {"canonical_repository", "workspace"} else object()
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="peer_assignment_mismatch"):
            handoff.require_peer_assignment(**args)
    finally:
        handoff.close()


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 181, "90"])
def test_invalid_timeout_refuses_before_mutation(prepared, value):
    case, _assignment = prepared
    before = case.native.workspace.stat().st_mode
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="timeout_invalid"):
        capture(prepared, timeout_seconds=value)
    assert case.native.workspace.stat().st_mode == before


@pytest.mark.parametrize("field,value", [("worker_uid", True), ("worker_uid", 0), ("worker_uid", -1),
    ("worker_uid", 1 << 31), ("worker_gid", True), ("worker_gid", -1), ("worker_gid", 1 << 31)])
def test_invalid_worker_principal_refuses_without_authentic_birth_claim(prepared, field, value):
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError):
        capture(prepared, **{field: value})


def test_same_owner_uid_or_foreign_group_refuses(prepared):
    case, _assignment = prepared
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="principal_invalid"):
        capture(prepared, worker_uid=os.geteuid())
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="root_profile_required"):
        capture(prepared, worker_gid=case.native.workspace.stat().st_gid + 1)


@pytest.mark.parametrize("kind", ["serialized", "unsealed", "subclass"])
def test_assignment_must_be_original_issued_exact_type(prepared, kind):
    _case, assignment = prepared
    if kind == "serialized":
        replacement = assignment.require_current()
    else:
        cls = type(assignment) if kind == "unsealed" else type("AuthoredSubclass", (type(assignment),), {})
        replacement = object.__new__(cls)
    with pytest.raises(ValueError):
        capture(prepared, assignment=replacement)


def test_unissued_handoff_and_shallow_copy_are_not_live_authority(prepared):
    handoff = capture(prepared)
    try:
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError):
            owner.NativeRepositoryWorldWorkerHandoff.require_current(object.__new__(owner.NativeRepositoryWorldWorkerHandoff))
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="not_issued"):
            copy(handoff).require_current()
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="original_capture_required"):
            owner.NativeRepositoryWorldWorkerHandoff(object(), assignment=prepared[1],
                worker_uid=os.geteuid() + 1, worker_gid=prepared[0].native.workspace.stat().st_gid,
                timeout_seconds=90)
    finally:
        handoff.close()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("method", ["require_current", "apply_readonly_transition", "require_observer_assignment", "require_peer_assignment", "close"])
def test_empty_or_partially_populated_exact_object_has_named_closed_refusal(partial, method):
    handoff = object.__new__(owner.NativeRepositoryWorldWorkerHandoff)
    if partial:
        handoff._seal = owner._SEAL
    arguments = {}
    if method == "require_observer_assignment":
        arguments = dict(prompt=PROMPT, worker_uid=1001, worker_gid=1000)
    elif method == "require_peer_assignment":
        arguments = dict(server=object(), canonical_repository=Path("/canonical"), workspace=Path("/allocated"),
            prompt=PROMPT, task_admission=object(), worktree_admission=object(), worker_uid=1001, worker_gid=1000)
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError):
        getattr(handoff, method)(**arguments)


@pytest.mark.parametrize("field", ["_material", "_original_observation", "_allocation_refs", "_directory_refs", "_selected_refs"])
def test_byte_equal_private_witness_replacement_refuses(prepared, field):
    handoff = capture(prepared)
    before = getattr(handoff, field)
    try:
        replacement = bytes(bytearray(before)) if type(before) is bytes else tuple(list(before))
        assert before == replacement and before is not replacement
        setattr(handoff, field, replacement)
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="custody_replaced"):
            handoff.require_current()
    finally:
        setattr(handoff, field, before)
        handoff.close()


@pytest.mark.parametrize("method", ["require_current", "_checked", "_detached", "_retained", "apply_readonly_transition"])
def test_instance_methods_cannot_replace_native_fences(prepared, method):
    handoff = capture(prepared)
    try:
        with pytest.raises(AttributeError):
            setattr(handoff, method, lambda: {"source_continuity_verified": True})
    finally:
        handoff.close()


@pytest.mark.parametrize("change", ["revision", "body", "claim", "phase"])
def test_post_transition_exact_native_task_body_revision_claim_and_phase_remain_bound(prepared, change):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        native = case.native
        if change == "revision":
            with native.server._lock:
                native.server._connection.execute("UPDATE tasks SET revision = revision + 1 WHERE task_cid = ?", [native.task_cid])
        else:
            body = deepcopy(dict(native.native_task.body))
            if change == "body": body["unknown_extra"] = {"changed": True}
            elif change == "claim": body["completion_receipt"]["claim_id"] = "foreign"
            else: body["completion_receipt"]["attempt_execution_phase"] = "context"
            _update_body(native, body)
        with pytest.raises(REFUSALS):
            handoff.require_current()
    finally:
        handoff.close()


@pytest.mark.parametrize("transition", ["released", "expired", "revoked_grant"])
def test_post_transition_native_claim_or_grant_loss_refuses(prepared, transition):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        coordinator, claim, lease = protect_claim(case)
        if transition == "released": coordinator.release(lease, reason="authored handoff release")
        elif transition == "expired": coordinator.expire_task_claim(claim, now_ms=lease.expires_at_ms)
        else: case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
        with pytest.raises(REFUSALS):
            handoff.require_current()
    finally:
        handoff.close()


@pytest.mark.parametrize("mode", [0o700, 0o770, 0o3770, 0o755])
def test_post_transition_root_mode_drift_is_never_rebased(prepared, mode):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        case.native.workspace.chmod(mode)
        with pytest.raises(ValueError, match=r"^(handoff_allocated_root_custody_changed|prelaunch_source_root_not_owner_controlled)$"):
            handoff.require_current()
    finally:
        handoff.close()


@pytest.mark.parametrize("root", ["canonical", "workspace"])
@pytest.mark.parametrize("change", ["bytes", "mode"])
def test_post_transition_source_bytes_or_permission_change_refuses(prepared, root, change):
    case, _assignment = prepared
    handoff = capture(prepared)
    path = getattr(case.native, root) / "module.py"
    before, mode = path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
    try:
        handoff.apply_readonly_transition()
        if change == "bytes": path.write_bytes(before.replace(b"VALUE", b"OTHER"))
        else: path.chmod(mode | 0o020)
        with pytest.raises(REFUSALS):
            handoff.require_current()
    finally:
        if change == "bytes": path.write_bytes(before)
        else: path.chmod(mode)
        handoff.close()


@pytest.mark.parametrize("root", ["canonical", "workspace"])
def test_authored_selected_gid_metadata_fault_refuses_without_cross_uid_claim(prepared, monkeypatch, root):
    case, _assignment = prepared
    handoff = capture(prepared)
    target = getattr(case.native, root)
    original = owner._selected_metadata
    try:
        handoff.apply_readonly_transition()
        fired = []
        def changed_gid(path, paths):
            rows = original(path, paths)
            if path == target:
                fired.append(True)
                name, directories, file = rows[0]
                file = (*file[:3], file[3] + 1, *file[4:])
                rows = ((name, directories, file), *rows[1:])
            return rows
        with monkeypatch.context() as controlled:
            controlled.setattr(owner, "_selected_metadata", changed_gid)
            with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="selected_metadata_changed"):
                handoff.require_current()
        assert fired  # Authored negative metadata control, no OS UID/GID change.
    finally:
        handoff.close()


@pytest.mark.parametrize("name", ["binding", "task_projection", "git_marker"])
def test_identical_control_bytes_under_replaced_inode_refuse_post_transition(prepared, name):
    case, _assignment = prepared
    handoff = capture(prepared)
    path = case.native.workspace / ".git" if name == "git_marker" else getattr(case.paths, name)
    before, mode = path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
    original, replacement = path.with_name(path.name + ".handoff-original"), path.with_name(path.name + ".handoff-replacement")
    try:
        handoff.apply_readonly_transition()
        path.rename(original)
        replacement.write_bytes(before)
        replacement.chmod(mode)
        replacement.rename(path)
        with pytest.raises(REFUSALS):
            handoff.require_current()
    finally:
        if original.exists():
            path.rename(replacement)
            original.rename(path)
        handoff.close()


@pytest.mark.parametrize("field", ["lease_id", "state", "expires_at"])
def test_lifecycle_identity_or_live_expiry_drift_refuses_post_transition(prepared, field):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        value = time.time() - 1 if field == "expires_at" else "terminal" if field == "state" else "foreign"
        with _record_change(case, lambda row: row.update({field: value})):
            with pytest.raises(REFUSALS):
                handoff.require_current()
    finally:
        handoff.close()


def test_genuine_monotonic_lifecycle_and_native_renewals_preserve_handoff(prepared):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        before = handoff.apply_readonly_transition()
        record = case.store.load_workspace(case.native.workspace)
        renewed = case.store.renew_lease(case.native.workspace, lease_id=record.lease_id, expected_fence=record.fence)
        assert renewed.fence > record.fence
        coordinator, _claim, lease = protect_claim(case)
        assert coordinator.renew(lease, lease_ms=120000).lease_id == lease.lease_id
        assert handoff.require_current() == before
    finally:
        handoff.close()


@pytest.mark.parametrize("field", ["store_dir", "store_repo_root", "bridge_namespace", "bridge_source",
    "bridge_portal_factory", "source_intent", "intent_connection", "lookup_native_allocation"])
def test_mutated_original_allocation_deep_policy_cannot_retarget_detached_checks(prepared, monkeypatch, field):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        allocation = case.guard
        if field == "store_dir": target, name, value = allocation._store, "store_dir", allocation._store_path.parent
        elif field == "store_repo_root": target, name, value = allocation._store, "repo_root", case.native.workspace
        elif field == "bridge_namespace": target, name, value = allocation._bridge, "board_namespace", "foreign-namespace"
        elif field == "bridge_source": target, name, value = allocation._bridge, "task_source", object()
        elif field == "bridge_portal_factory": target, name, value = allocation._bridge, "portal_factory", lambda *_a, **_kw: None
        elif field == "source_intent": target, name, value = allocation._source, "_intent", object()
        elif field == "intent_connection": target, name, value = allocation._intent, "_bound_connection", object()
        else: target, name, value = handoff._source, "_native_source_worktree_allocation", object()
        with monkeypatch.context() as controlled:
            controlled.setattr(target, name, value)
            with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="deep_policy_changed"):
                handoff.require_current()
    finally:
        handoff.close()


def test_cas_callback_release_after_transition_is_closed_by_same_original_claim_fence(prepared, monkeypatch):
    case, _assignment = prepared
    handoff = capture(prepared)
    try:
        handoff.apply_readonly_transition()
        coordinator, _claim, lease = protect_claim(case)
        original, fired = case.native.retained.cas.get_bytes, []
        def release_after_read(cid):
            value = original(cid)
            if not fired:
                fired.append(True)
                coordinator.release(lease, reason="authored post-transition CAS release")
            return value
        with monkeypatch.context() as controlled:
            controlled.setattr(case.native.retained.cas, "get_bytes", release_after_read)
            with pytest.raises(REFUSALS):
                handoff.require_current()
        assert fired == [True]
    finally:
        handoff.close()


def test_permission_operation_partial_failure_poisoned_without_rollback_authority(prepared, monkeypatch):
    case, _assignment = prepared
    handoff = capture(prepared)
    actual = os.fchmod
    def changed_then_failed(fd, mode):
        actual(fd, mode)
        raise OSError("authored fchmod acknowledgement failure")
    try:
        with monkeypatch.context() as controlled:
            controlled.setattr(owner.os, "fchmod", changed_then_failed)
            with pytest.raises(OSError):
                handoff.apply_readonly_transition()
        assert stat.S_IMODE(case.native.workspace.stat().st_mode) == 0o750
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="not_current"):
            handoff.require_current()
        with pytest.raises(owner.RepositoryWorldWorkerHandoffError):
            handoff.apply_readonly_transition()
    finally:
        closed = handoff.close()
    assert closed["state"] == "RECOVERY_REQUIRED" and closed["read_only_transition_verified"] is False


def test_one_shot_transition_and_close_never_reset_modes_or_claim_live_continuity(prepared):
    case, _assignment = prepared
    handoff = capture(prepared)
    handoff.apply_readonly_transition()
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="already_consumed"):
        handoff.apply_readonly_transition()
    closed = handoff.close()
    assert closed["state"] == "CLOSED"
    assert closed["native_assignment_continuity_verified"] is closed["source_continuity_verified"] is False
    assert handoff.close() == closed
    with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="not_current"):
        handoff.require_current()
    assert stat.S_IMODE(case.native.workspace.stat().st_mode) == 0o750


def test_expiry_is_not_renewed_by_observation_or_native_lease_renewal(prepared, monkeypatch):
    handoff = capture(prepared)
    deadline = handoff._deadline
    try:
        with monkeypatch.context() as controlled:
            controlled.setattr(owner.time, "monotonic", lambda: deadline + 1)
            with pytest.raises(owner.RepositoryWorldWorkerHandoffError, match="expired"):
                handoff.require_current()
        assert handoff._deadline == deadline
    finally:
        handoff.close()


def test_genuine_original_nomination_survives_root_only_transition(nominated_allocation):
    case = nominated_allocation
    bind(case)
    provision_authored_prelaunch_sources(case)
    assignment = prelaunch.capture_repository_world_prelaunch_assignment(
        task_admission=case.native.handle, worktree_admission=case.guard, prompt=PROMPT)
    handoff = capture((case, assignment))
    try:
        before = assignment.require_current()["nominated_scope"]
        observed = handoff.apply_readonly_transition()
        assert observed["selected_source_count"] == 2 and observed["nominated_scope"] == before
        assert handoff.require_current() == observed
        assert observed["worker_process_verified"] is False
    finally:
        handoff.close()
