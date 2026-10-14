"""Historical selected source joined to real native claim and Portal allocation.

Git, retained CAS, native claim/coordinator/grant, sealed Portal projection and
lifecycle records are real. Quack listener and deployment/UID metadata reuse the
existing authored host seams. No coding child, provider, target execution,
Terminal profile, reconstruction model or formal proof is dispatched here.
"""
from copy import deepcopy
from dataclasses import replace
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_worktree_allocation as allocation
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_scope as scopes
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from test.api.test_current_source_task_admission import SOURCE, _payload, _update_body
from test.api.test_current_source_worktree_allocation import (
    _allocation_case, _record_change, allocated,  # noqa: F401
)
from test.api.test_native_nominated_source_scope import SUPPORT, OTHER


@pytest.fixture
def nominated_allocation(tmp_path, monkeypatch):
    files = {b"module.py": SOURCE, b"instruction.md": SUPPORT, b"other.py": OTHER}
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True, source_files=files) as case:
        yield case


def nominated(case):
    native = case.native
    return scopes.nominate_source_scope(cas=native.retained.cas,
        expected_inventory_cid=native.retained.ref.inventory_cid,
        request=native.retained.request, limits=native.retained.limits,
        selected_paths=tuple(sorted(name.hex() for name in (b"module.py", b"instruction.md"))),
        required_paths=(b"instruction.md".hex(),), program_paths=(b"module.py".hex(),))


def arguments(case, scope=None):
    scope = nominated(case) if scope is None else scope
    return {**{key: value for key, value in case.source_binding.items() if key != "allowed_paths"},
        "scope": scope, "expected_scope_cid": scope.observation()["scope_cid"]}


def bind(case, **changes):
    case.native.binding = case.guard.bind_nominated_current_source_lookup(**(arguments(case) | changes))
    return case.native.binding


def assert_unbound(case):
    assert case.native.server._command_gateway._current_source_lookup is None
    assert case.guard._lookup is None and case.native.handle._source is None


def test_selected_scope_joins_distinct_native_portal_and_allocation_identities(nominated_allocation):
    case, native = nominated_allocation, nominated_allocation.native
    scope = nominated(case)
    historical = scope.observation()
    binding = bind(case, scope=scope, expected_scope_cid=historical["scope_cid"])
    observed = case.guard.require_nominated_source_scope()
    task_only = native.handle.require_nominated_source_scope()
    current_allocation = case.guard.require_current()
    assert observed["schema"] == "supervisor-native-nominated-source-worktree-allocation@1"
    assert observed["allocation_cid"] == current_allocation["allocation_cid"]
    assert observed["allocation_ownership_verified"] is True
    assert task_only["allocation_ownership_verified"] is False
    assert observed["task_cid"] == native.task_cid != case.portal_identity["canonical_task_cid"]
    assert case.record.lease_id != native.attempt.lease_id
    assert observed["source_scope_cid"] == historical["scope_cid"]
    assert observed["source_binding_cid"] == binding["binding_cid"]
    assert observed["frontier_cid"] == historical["frontier_cid"]
    assert (observed["selected_count"], observed["frontier_count"]) == (2, 1)
    assert historical["request"]["task_id"] != native.native_task.task_alias
    assert historical["task_admission_verified"] is False
    assert observed["native_source_binding_verified"] is observed["native_task_admission_verified"] is True
    assert all(observed[key] is False for key in (
        "bootstrap_verified", "compiled_context_verified", "coding_process_verified",
        "proof_authority", "execution_authority", "completion_authority", "omission_authority"))
    service = native.server._command_gateway._current_source_lookup
    assert service._native_source_task_admission is native.handle
    assert service._native_source_worktree_allocation is case.guard
    assert service._paths == tuple(row["raw_path_hex"] for row in historical["selected"])
    native.session = broker.start_source_lookup_worker_broker(**native.factory)
    result = native.session._read_source(_payload(native))
    assert result["content"].encode() == SOURCE and result["read_only"] is True
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native) | dict(raw_path_hex=b"other.py".hex(), source_cid=cid_for_bytes(OTHER)))
    assert native.daemon.provider_invocation_recorded(native.attempt.attempt_id,
        idempotency_key="provider:" + native.attempt.attempt_id) is None
    observed["allocation_cid"] = "changed-detached-observation"
    assert case.guard.require_nominated_source_scope()["allocation_cid"] == current_allocation["allocation_cid"]


@pytest.mark.parametrize("change", ["scope_cid", "inventory_cid", "request", "limits", "serialized_scope"])
def test_independent_pins_refuse_without_consuming_allocated_reader(nominated_allocation, change):
    case = nominated_allocation
    args = arguments(case)
    if change == "scope_cid": args["expected_scope_cid"] = cid_for_structured({"foreign_scope": True})
    elif change == "inventory_cid": args["expected_inventory_cid"] = cid_for_structured({"foreign_inventory": True})
    elif change == "request": args["request"] = replace(case.native.retained.request, task_id="foreign-capture")
    elif change == "limits": args["limits"] = replace(case.native.retained.limits, max_entries=case.native.retained.limits.max_entries - 1)
    else: args["scope"] = args["scope"].observation()
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**args)
    assert_unbound(case)
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.require_nominated_source_scope()
    bind(case)
    assert case.guard.require_nominated_source_scope()["allocation_ownership_verified"] is True


@pytest.mark.parametrize("change", ["revoke_grant", "released_lease", "task_body"])
def test_lost_native_claim_refuses_selected_allocation_binding(nominated_allocation, change):
    case, native = nominated_allocation, nominated_allocation.native
    args = arguments(case)
    if change == "revoke_grant": native.server._command_gateway.revoke_grant(native.daemon_grant_id)
    elif change == "released_lease":
        claim = native.daemon.coordinator.get_task_claim(native.attempt.claim_id)
        lease = native.daemon.coordinator.protect_task_claim(claim)
        native.daemon.coordinator.release(lease, reason="authored nominated allocation claim release")
    else:
        body = deepcopy(dict(native.native_task.body))
        body["title"] = "changed before selected allocation binding"
        _update_body(native, body)
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**args)
    assert_unbound(case)


@pytest.mark.parametrize("change", ["state", "owner", "portal_task", "lease"])
def test_authentic_record_with_foreign_lifecycle_tuple_refuses_binding(nominated_allocation, change):
    case = nominated_allocation
    args = arguments(case)
    def alter(value):
        if change == "state": value["state"] = "settling"
        elif change == "owner": value["owner"]["start_time_ticks"] += 1
        elif change == "portal_task": value["canonical_task_cid"] = "changed-portal-task"
        else: value["lease_id"] = "foreign-lifecycle-lease"
    with _record_change(case, alter):
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.bind_nominated_current_source_lookup(**args)
        assert_unbound(case)


@pytest.mark.parametrize("path", ["module.py", "other.py"])
def test_selected_or_omitted_allocated_original_drift_refuses_nomination(nominated_allocation, path):
    case = nominated_allocation
    args = arguments(case)
    (case.native.workspace / path).write_bytes(b"CHANGED = True\n")
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**args)
    assert_unbound(case)


@pytest.mark.parametrize("path", ["module.py", "other.py"])
def test_canonical_working_drift_refuses_broker_independent_source_join(nominated_allocation, monkeypatch, path):
    case, native = nominated_allocation, nominated_allocation.native
    (native.canonical / path).write_bytes(b"CHANGED = True\n")
    # The allocation proves Git/control ownership. Its nominated reader names
    # the clean allocated checkout; canonical original bytes are independently
    # joined by the broker before a proxy grant can be made available.
    bind(case)
    service = native.server._command_gateway._current_source_lookup
    monkeypatch.setattr(native.server, "issue_typed_client_grant_record",
        lambda **kwargs: pytest.fail("stale canonical source issued a proxy grant"))
    with pytest.raises((ValueError, RuntimeError)):
        broker.start_source_lookup_worker_broker(**native.factory)
    assert native.server._command_gateway._current_source_lookup is service
    assert service._native_source_worktree_allocation is case.guard


def test_corrupt_unselected_cas_source_refuses_allocated_cold_replay(nominated_allocation):
    case = nominated_allocation
    args = arguments(case)
    frame = next(case.native.retained.cas.source_root.rglob(cid_for_bytes(OTHER)))
    frame.write_bytes(b"corrupt unselected retained original")
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**args)
    assert_unbound(case)


@pytest.mark.parametrize("field", ["task_id", "max_entries"])
def test_cas_callback_input_mutation_cannot_publish_allocated_scope(nominated_allocation, monkeypatch, field):
    case = nominated_allocation
    args = arguments(case)
    original = case.native.retained.cas.get_bytes
    def changed(cid):
        value = original(cid)
        target = case.native.retained.request if field == "task_id" else case.native.retained.limits
        replacement = "changed-during-cas" if field == "task_id" else target.max_entries - 1
        object.__setattr__(target, field, replacement)
        return value
    monkeypatch.setattr(case.native.retained.cas, "get_bytes", changed)
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**args)
    assert_unbound(case)


def test_constructed_scope_cannot_replace_independent_allocated_cas_replay(nominated_allocation):
    case = nominated_allocation
    scope = nominated(case)
    value = scope.observation()
    value["selected"][0]["source_sha256"] = "0" * 64
    value["scope_cid"] = cid_for_structured({key: row for key, row in value.items() if key != "scope_cid"})
    forged = scopes.RepositorySourceScope(canonical_dag_json_bytes(value), scope._frontier_rows)
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**arguments(case, forged))
    assert_unbound(case)


def test_real_lifecycle_renewal_preserves_selected_allocation_join(nominated_allocation):
    case = nominated_allocation
    bind(case)
    observed = case.guard.require_nominated_source_scope()
    record = case.store.load_workspace(case.native.workspace)
    renewed = case.store.renew_lease(case.native.workspace, lease_id=record.lease_id, expected_fence=record.fence)
    assert renewed.fence == record.fence + 1
    assert case.guard.require_nominated_source_scope() == observed


def test_expired_allocation_refuses_observer_after_nomination_binding(nominated_allocation):
    case = nominated_allocation
    bind(case)
    with _record_change(case, lambda value: value.update(expires_at=time.time() - 1)):
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.require_nominated_source_scope()
    assert case.guard.require_nominated_source_scope()["allocation_ownership_verified"] is True


@pytest.mark.parametrize("path", ["module.py", "other.py"])
def test_allocated_source_drift_refuses_later_scope_observation(nominated_allocation, path):
    case = nominated_allocation
    bind(case)
    (case.native.workspace / path).write_bytes(b"CHANGED_AFTER_BIND = True\n")
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.require_nominated_source_scope()


@pytest.mark.parametrize("change", ["reader", "nomination", "allocation_marker", "task_body"])
def test_observer_refuses_replaced_join_or_native_task(nominated_allocation, monkeypatch, change):
    case, native = nominated_allocation, nominated_allocation.native
    bind(case)
    service = native.server._command_gateway._current_source_lookup
    with monkeypatch.context() as altered:
        if change == "reader": altered.setattr(native.server._command_gateway, "_current_source_lookup", object())
        elif change == "nomination": altered.setattr(case.guard, "_nominated_source_scope", object())
        elif change == "allocation_marker": altered.setattr(service, "_native_source_worktree_allocation", object())
        else:
            body = deepcopy(dict(native.native_task.body))
            body["title"] = "changed after selected allocation binding"
            _update_body(native, body)
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.require_nominated_source_scope()
    assert native.server._command_gateway._current_source_lookup is service


def test_legacy_allocation_reader_cannot_claim_a_nomination(allocated):
    with pytest.raises((ValueError, RuntimeError)):
        allocated.guard.require_nominated_source_scope()
    assert allocated.native.session._read_source(_payload(allocated.native))["content"].encode() == SOURCE


def test_serialized_allocation_cannot_supply_new_nomination_binding(nominated_allocation):
    case = nominated_allocation
    args = arguments(case)
    with pytest.raises((ValueError, RuntimeError, AttributeError, TypeError)):
        allocation.NativeSourceWorktreeAllocation.bind_nominated_current_source_lookup(
            case.guard.require_current(), **args)
    assert_unbound(case)
