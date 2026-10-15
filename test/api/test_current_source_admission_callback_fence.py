"""Real owner claim/allocation callback fences; no providers or coding children.

The existing fixture authors only its Quack transport/deployment seams. Native
SQL claims, grants, Portal projection, lifecycle and Git controls remain real.
An external marker deliberately survives a post-callback refusal.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import threading
import time

import pytest

from test.api.test_current_source_task_admission import SOURCE, _update_body
from test.api.test_current_source_worktree_allocation import (
    _allocation_case, _index_path, _record_change, _record_path,
)


@pytest.fixture
def allocated(tmp_path, monkeypatch):
    # No source proxy is bound: this primitive reads allocation controls only.
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True) as case:
        yield case


def _run(case, callback, **kwargs):
    return case.guard.execute_with_current_claim(callback, **kwargs)


@contextmanager
def _grant_change(case, *, expired=False):
    gateway = case.native.server._command_gateway
    with gateway._grants_lock:
        original = dict(gateway._grants)
        revoked = set(gateway._revoked_grants)
        if expired:
            gateway._grants = {token: replace(grant, issued_at=1, expires_at=2)
                if grant.grant_id == case.native.daemon_grant_id else grant
                for token, grant in gateway._grants.items()}
        else:
            gateway._revoked_grants.add(case.native.daemon_grant_id)
    try:
        yield
    finally:
        with gateway._grants_lock:
            gateway._grants = original
            gateway._revoked_grants = revoked


@contextmanager
def _changed(case, change):
    native = case.native
    if change in {"revoked_grant", "expired_grant"}:
        with _grant_change(case, expired=change == "expired_grant"):
            yield
    elif change in {"task_body", "task_phase"}:
        before = deepcopy(dict(native.native_task.body))
        body = deepcopy(before)
        if change == "task_body": body["title"] = "different full task requirements"
        else: body["completion_receipt"]["attempt_execution_phase"] = "context"
        _update_body(native, body)
        try:
            yield
        finally:
            _update_body(native, before)
    elif change == "task_revision":
        with native.server._lock:
            native.server._connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_cid=?", [native.task_cid])
        try:
            yield
        finally:
            with native.server._lock:
                native.server._connection.execute("UPDATE tasks SET revision=revision-1 WHERE task_cid=?", [native.task_cid])
    elif change in {"lifecycle_expired", "lifecycle_owner", "lifecycle_state"}:
        def mutate(row):
            if change == "lifecycle_expired": row["expires_at"] = time.time() - 1
            elif change == "lifecycle_owner": row["owner"]["start_time_ticks"] += 1
            else: row["state"] = "settling"
        with _record_change(case, mutate):
            yield
    else:
        path = _index_path(case) if change == "lifecycle_index" else case.paths.task_projection
        original = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        path.chmod(0o600)
        if change == "lifecycle_index":
            row = json.loads(original); row["workspace_path"] += "-foreign"
            path.write_text(json.dumps(row))
        else:
            path.write_bytes(original + b"\nforeign projection\n")
        path.chmod(mode)
        try:
            yield
        finally:
            path.chmod(0o600); path.write_bytes(original); path.chmod(mode)


def test_callback_receives_actual_immutable_lease_and_all_owner_locks_remain_held(allocated):
    case = allocated
    protected = case.native.daemon._protect_attempt_write(case.native.attempt)
    seen = []
    result = object()
    def callback(lease):
        assert lease == protected
        assert lease.lease_id == case.native.attempt.lease_id
        assert lease.fencing_token == case.native.attempt.fencing_token
        assert lease.fence_epoch == case.native.attempt.fence_epoch
        with pytest.raises(FrozenInstanceError): lease.expires_at_ms = 0
        acquired = []
        def contender():
            for lock in (case.guard._lock, case.native.handle._lock, case.native.server._lock):
                available = lock.acquire(blocking=False)
                acquired.append(available)
                if available: lock.release()
        thread = threading.Thread(target=contender)
        thread.start(); thread.join(timeout=2)
        assert not thread.is_alive() and acquired == [False, False, False]
        seen.append(lease)
        return result
    assert _run(case, callback, minimum_remaining_ms=1000) is result
    assert seen == [protected]
    assert case.guard.require_current()["coding_process_verified"] is False


@pytest.mark.parametrize("change", ["revoked_grant", "expired_grant", "task_body", "task_phase",
    "task_revision", "lifecycle_expired", "lifecycle_owner", "lifecycle_state",
    "lifecycle_index", "projection"])
def test_stale_native_task_grant_and_allocation_refuse_before_callback(allocated, change):
    calls = []
    with _changed(allocated, change):
        with pytest.raises((ValueError, RuntimeError)):
            _run(allocated, lambda lease: calls.append(lease))
    assert calls == []


@pytest.mark.parametrize("change", ["revoked_grant", "expired_grant", "task_body", "task_phase",
    "task_revision", "lifecycle_expired", "lifecycle_owner", "lifecycle_state",
    "lifecycle_index", "projection"])
def test_post_callback_drift_refuses_but_preserves_external_effects(allocated, change):
    case = allocated
    marker = case.native.workspace / "callback-effect.txt"
    mutation = None
    calls = []
    def callback(lease):
        nonlocal mutation
        calls.append(lease)
        marker.write_text("already occurred")
        mutation = _changed(case, change)
        mutation.__enter__()
        return "cannot accept after drift"
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _run(case, callback)
        assert len(calls) == 1 and marker.read_text() == "already occurred"
    finally:
        if mutation is not None: mutation.__exit__(None, None, None)


def test_released_native_task_claim_refuses_before_callback(allocated):
    native = allocated.native
    lease = native.daemon._protect_attempt_write(native.attempt)
    native.daemon.coordinator.release(lease, reason="authored callback refusal")
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        _run(allocated, lambda protected: calls.append(protected))
    assert calls == []


@pytest.mark.parametrize("expire_before", [True, False])
def test_real_task_lease_expiry_is_checked_before_and_after_callback(allocated, monkeypatch, expire_before):
    native = allocated.native
    lease = native.daemon._protect_attempt_write(native.attempt)
    original_clock = type(native.clock).__call__
    future = [expire_before]
    def clock(value):
        return max(original_clock(value), lease.expires_at_ms + 1) if future[0] else original_clock(value)
    monkeypatch.setattr(type(native.clock), "__call__", clock)
    marker = native.workspace / "lease-effect.txt"
    calls = []
    def callback(protected):
        calls.append(protected)
        marker.write_text("effect remains after task expiry")
        future[0] = True
    with pytest.raises((ValueError, RuntimeError)):
        _run(allocated, callback)
    assert len(calls) == (0 if expire_before else 1)
    assert marker.exists() is (not expire_before)


@pytest.mark.parametrize("operation", ["callback", "allocation", "admission", "coordinator"])
@pytest.mark.parametrize("suppress", [False, True])
def test_callback_reentry_never_becomes_a_successful_outer_operation(allocated, operation, suppress):
    calls = []
    nested = []
    def callback(lease):
        calls.append(lease)
        def reenter():
            if operation == "callback":
                return _run(allocated, lambda _lease: nested.append("nested"))
            if operation == "allocation": return allocated.guard.require_current()
            if operation == "admission": return allocated.native.handle.require_current()
            return allocated.native.daemon.coordinator.protect_task_claim(allocated.native.handle._claim)
        if suppress:
            with pytest.raises((ValueError, RuntimeError)): reenter()
            return "suppressed refusal is not acceptance"
        return reenter()
    with pytest.raises((ValueError, RuntimeError)):
        _run(allocated, callback)
    assert len(calls) == 1 and nested == []
    assert allocated.guard.require_current()["execution_authority"] is False


def test_callback_exception_is_preserved_and_releases_callback_locks(allocated):
    error = LookupError("authored owner callback failure")
    marker = allocated.native.workspace / "raised-effect.txt"
    def callback(lease):
        marker.write_text("already occurred")
        raise error
    with pytest.raises(LookupError) as caught:
        _run(allocated, callback)
    assert caught.value is error and marker.exists()
    assert _run(allocated, lambda lease: "next independent owner operation") == "next independent owner operation"


def test_current_renewed_task_lease_is_passed_without_replacing_original_claim_identity(allocated):
    native = allocated.native
    original = native.daemon._protect_attempt_write(native.attempt)
    renewed = native.daemon.coordinator.renew(original, lease_ms=60000)
    assert renewed.lease_id == original.lease_id and renewed.fencing_token == original.fencing_token
    assert renewed.expires_at_ms != original.expires_at_ms
    seen = []
    assert _run(allocated, lambda lease: seen.append(lease) or "current lease", minimum_remaining_ms=1000) == "current lease"
    assert seen == [renewed]


def test_genuine_lifecycle_renewal_keeps_allocation_material_and_monotonic_custody(allocated):
    before = allocated.guard.require_current()
    current = allocated.store.load_workspace(allocated.native.workspace)
    renewed = allocated.store.renew_lease(allocated.native.workspace,
        lease_id=current.lease_id, expected_fence=current.fence)
    assert renewed.fence > current.fence
    assert _run(allocated, lambda lease: "same allocation") == "same allocation"
    assert allocated.guard.require_current() == before


def test_borrowed_native_sql_transaction_is_refused_before_callback_and_left_owned_by_caller(allocated):
    connection = allocated.native.server._connection
    calls = []
    with allocated.native.server._lock:
        connection.execute("BEGIN TRANSACTION")
    try:
        with pytest.raises((ValueError, RuntimeError), match="transaction"):
            _run(allocated, lambda lease: calls.append(lease))
        assert calls == [] and connection.in_transaction is True
    finally:
        with allocated.native.server._lock: connection.execute("ROLLBACK")
    assert _run(allocated, lambda lease: "clean transaction") == "clean transaction"


def test_callback_cannot_leave_a_borrowed_native_sql_transaction_for_its_postcheck(allocated):
    connection = allocated.native.server._connection
    marker = allocated.native.workspace / "transaction-effect.txt"
    def callback(lease):
        marker.write_text("already occurred")
        connection.execute("BEGIN TRANSACTION")
        return "unsettled native transaction"
    try:
        with pytest.raises((ValueError, RuntimeError), match="transaction"):
            _run(allocated, callback)
        assert connection.in_transaction is True and marker.exists()
    finally:
        with allocated.native.server._lock: connection.execute("ROLLBACK")


@pytest.mark.parametrize("minimum", [True, -1, "1000", 10 ** 100])
def test_invalid_minimum_ack_budget_refuses_before_owner_callback(allocated, minimum):
    calls = []
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _run(allocated, lambda lease: calls.append(lease), minimum_remaining_ms=minimum)
    assert calls == []


def test_actual_current_task_lease_must_have_full_minimum_ack_budget(allocated):
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        _run(allocated, lambda lease: calls.append(lease), minimum_remaining_ms=120001)
    assert calls == []


def test_source_edits_are_not_reinterpreted_as_allocation_or_proof_checks(allocated):
    source = allocated.native.workspace / "module.py"
    def callback(lease):
        source.write_bytes(SOURCE + b"# owner candidate edit\n")
        return "candidate changed"
    assert _run(allocated, callback) == "candidate changed"
    material = allocated.guard.require_current()
    assert material["allocation_ownership_verified"] is True
    assert material["proof_authority"] is material["execution_authority"] is material["completion_authority"] is False
    assert material["coding_process_verified"] is False
    assert source.read_bytes() != SOURCE


def test_original_read_only_workspace_permission_guard_is_preserved(allocated):
    root = allocated.native.workspace
    mode = root.stat().st_mode & 0o7777
    calls = []
    root.chmod(0o770)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _run(allocated, lambda lease: calls.append(lease))
        assert calls == []
    finally:
        root.chmod(mode)


def test_closing_coordinator_clock_cannot_hide_late_lifecycle_loss(allocated, monkeypatch):
    native = allocated.native
    path = _record_path(allocated)
    original = path.read_bytes()
    original_clock = type(native.clock).__call__
    armed = [False]
    changed = [False]
    def clock(value):
        now = original_clock(value)
        if value is native.clock and armed[0]:
            armed[0] = False; changed[0] = True
            row = json.loads(path.read_bytes()); row["expires_at"] = time.time() - 1
            path.write_text(json.dumps(row))
        return now
    monkeypatch.setattr(type(native.clock), "__call__", clock)
    def callback(lease):
        armed[0] = True
        return "the coordinator closing clock runs after allocation postchecks"
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _run(allocated, callback)
        assert changed[0] is True
    finally:
        path.write_bytes(original)
