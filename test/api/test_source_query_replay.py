"""Native source query replay on real owner/task/allocation resources.

Git/CAS, DuckDB/typed source connections, native task/claim/grants and Portal
allocation/lifecycle are real. Quack listener and deployment UID metadata
retain the existing authored host seams. These owner-side cases launch no
child, admit no worker UID and execute no target, provider, model or solver.
The checks are sequential and establish no filesystem writer lease.
"""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
from test.api.test_current_source_lookup import _client, _data
from test.api.test_current_source_task_admission import SOURCE, _payload
from test.api.test_current_source_worktree_allocation import _allocation_case
from test.api.test_native_nominated_source_allocation_custody import bind, source_files


DIAGNOSTIC = "native query CAS replay must release SERVER"


@pytest.fixture
def native_query(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files()) as case:
        bind(case)
        adapter = SimpleNamespace(server=case.native.server,
            gateway=case.native.server._command_gateway, task_cid=case.native.task_cid,
            request=case.native.retained.request, clients=[])
        client, _token, grant = _client(adapter)
        try:
            yield case, client, grant
        finally:
            for connection in adapter.clients:
                connection.close()


def test_native_query_cas_callback_reenters_native_fences_after_server_release(
        native_query, monkeypatch):
    case, client, grant = native_query
    service = case.guard._lookup
    original_lock = case.native.server._lock
    original_load = service._load
    original_get = case.native.retained.cas.get
    state = dict(active=False, server_held=False, reached=[], completed=[])

    def replay():
        state["active"] = True
        try:
            return original_load()
        finally:
            state["active"] = False

    def cas_get(cid):
        value = original_get(cid)
        if state["active"] and not state["reached"]:
            state["reached"].append(cid)
            state["server_held"] = original_lock._is_owned()
            # The gateway serializes callback failures. Retain this observed
            # condition so the caller also reports the diagnostic assertion,
            # before an unsafe COOR acquisition could deadlock the old path.
            assert not state["server_held"], DIAGNOSTIC
            assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
            assert case.guard.require_current()["allocation_ownership_verified"] is True
            state["completed"].append(cid)
        return value

    monkeypatch.setattr(service, "_load", replay)
    monkeypatch.setattr(case.native.retained.cas, "get", cas_get)
    try:
        response = client.source_get(_payload(case.native))
    except typed.TypedStateOwnerError:
        assert not state["server_held"], DIAGNOSTIC
        raise
    assert state["reached"] == state["completed"] and len(state["reached"]) == 1
    assert not state["active"] and not state["server_held"]
    assert _data(response) == SOURCE and response["ok"] is True
    result = response["result"]
    assert result["caller_grant_id"] == grant.grant_id
    assert result["response_bytes"] == len(canonical_json_bytes(response)) + 1
    assert result["response_wire_bytes"] == result["response_bytes"] + 3
    assert result["source_complete"] is True and result["next_continuation"] is None
    assert (result["proof_authority"] is result["completion_authority"]
            is result["edit_authority"] is result["omission_authority"] is False)
    assert service._continuations == {}
    assert case.native.server._command_gateway._current_source_lookup is service
    assert service is case.guard._lookup is case.native.handle._source
    assert case.native.daemon.provider_invocation_recorded(case.native.attempt.attempt_id,
        idempotency_key="provider:" + case.native.attempt.attempt_id) is None


# These imports and cases were appended after the retained unpatched control.
# Its historical one-case source snapshot remains an independent artifact.
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import sys
import threading
import time

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_lookup as lookup
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as inventories
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from test.api.test_current_source_lookup import (
    DEFAULT_SOURCE, _assert_response, _case as lookup_case, _payload as lookup_payload,
)


REPLAY_CHANGES = ("selected", "omitted", "store", "binding", "task", "grant", "connection", "lock")


@pytest.fixture
def query_case(tmp_path):
    with lookup_case(tmp_path) as case:
        yield case


def seed_cursor(case):
    payload = lookup_payload(case, maximum=1)
    response = case.client.source_get(payload)
    result, raw = _assert_response(case, payload, response)
    assert raw == DEFAULT_SOURCE[:1] and result["next_continuation"] is not None
    follow = lookup_payload(case, maximum=1, offset=result["end_offset"],
                            continuation=result["next_continuation"])
    return follow, case.service._continuations, dict(case.service._continuations)


def observe_replay(case, monkeypatch, action):
    original_load, original_get = case.service._load, case.cas.get
    original_lock = case.server._lock
    state = SimpleNamespace(active=False, completed=False, reached=[], actions=[])

    def replay():
        state.active = True
        try:
            result = original_load()
            state.completed = True
            return result
        finally:
            state.active = False

    def cas_get(cid):
        value = original_get(cid)
        if state.active and not state.reached:
            state.reached.append(cid)
            assert not original_lock._is_owned(), DIAGNOSTIC
            action()
            state.actions.append(cid)
        return value

    monkeypatch.setattr(case.service, "_load", replay)
    monkeypatch.setattr(case.cas, "get", cas_get)
    return state


def read_idle(case):
    assert case.gateway._current_source_read_active is None
    assert case.service._active_read is None
    assert not case.gateway._current_source_read_guard.locked()


def mutate_query(case, change, cleanup, replacement=None):
    if change in {"selected", "omitted"}:
        path = case.root / ("module.py" if change == "selected" else "other.py")
        before = path.read_bytes()
        cleanup.callback(path.write_bytes, before)
        path.write_bytes(before + b"# changed during native query replay\n")
    elif change == "task":
        before = canonical_json_bytes(case.body).decode()
        changed = canonical_json_bytes(dict(case.body, title="changed during native query replay")).decode()
        cleanup.callback(case.connection.execute,
            "UPDATE tasks SET body_json=? WHERE task_cid=?", [before, case.task_cid])
        case.connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?", [changed, case.task_cid])
    elif change == "grant":
        case.gateway.revoke_grant(case.grant.grant_id)
    elif change == "store":
        before = case.service._store
        cleanup.callback(setattr, case.service, "_store", before)
        case.service._store = replacement
    elif change == "binding":
        before = case.gateway._current_source_lookup
        cleanup.callback(setattr, case.gateway, "_current_source_lookup", before)
        case.gateway._current_source_lookup = replacement
    elif change == "connection":
        before = case.gateway._connection
        cleanup.callback(setattr, case.gateway, "_connection", before)
        case.gateway._connection = replacement
    else:
        assert change == "lock"
        before = case.gateway._transaction_lock
        cleanup.callback(setattr, case.gateway, "_transaction_lock", before)
        case.gateway._transaction_lock = replacement


@pytest.mark.parametrize("change", REPLAY_CHANGES)
def test_off_lock_replay_custody_loss_preserves_the_issued_cursor(query_case, monkeypatch, change):
    case = query_case
    follow, registry, before = seed_cursor(case)
    with ExitStack() as cleanup:
        replacement = None
        if change == "store":
            replacement = inventories._BoundCAS(case.cas, case.service._guard)
        elif change == "binding":
            replacement = lookup.CurrentSourceLookup(**case.config)
        elif change == "connection":
            replacement = open_duckdb_connection(case.root.parent / "replacement-query-connection.duckdb")
            cleanup.callback(replacement.close)
        elif change == "lock":
            replacement = threading.RLock()
        state = observe_replay(case, monkeypatch,
            lambda: mutate_query(case, change, cleanup, replacement))
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(follow)
        assert state.reached == state.actions and len(state.reached) == 1
        assert case.service._continuations is registry and registry == before
        read_idle(case)
    if change != "grant":
        # Restore only the caller's temporary mutation. The failed query must
        # leave its original issued cursor available for a genuine retry.
        result, raw = _assert_response(case, follow, case.client.source_get(follow))
        assert raw == DEFAULT_SOURCE[1:2]
        assert (case.grant.grant_id, follow["continuation"]) not in registry
        assert registry == {(case.grant.grant_id, result["next_continuation"]): case.grant.expires_at}
        read_idle(case)


def test_borrowed_transaction_started_during_replay_is_refused_and_left_to_its_caller(
        query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    entered, release = threading.Event(), threading.Event()

    def pause():
        entered.set()
        assert release.wait(timeout=2), "caller did not release retained replay"

    state = observe_replay(case, monkeypatch, pause)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(case.client.source_get, follow)
        try:
            assert entered.wait(timeout=2), "query did not reach retained replay"
            with case.server._lock:
                case.connection.execute("BEGIN")
            assert case.connection.in_transaction is True
            release.set()
            with pytest.raises(typed.TypedStateOwnerError):
                future.result(timeout=4)
            # This is a genuine transaction owned by the caller thread. The
            # failed source query cannot roll it back or consume its cursor.
            assert case.connection.in_transaction is True
            assert case.service._continuations is registry and registry == before
            assert state.reached == state.actions and len(state.reached) == 1
            read_idle(case)
        finally:
            release.set()
            if case.connection.in_transaction:
                case.connection.rollback()
    assert case.connection.in_transaction is False


@pytest.mark.parametrize("change", ("selected", "omitted", "task", "grant"))
def test_final_authorization_callback_mutation_refuses_before_cursor_publication(
        query_case, monkeypatch, change):
    case = query_case
    follow, registry, before = seed_cursor(case)
    original = case.gateway._authorize_source_grant
    reached = []
    with ExitStack() as cleanup:

        def authorize(grant, service):
            result = original(grant, service)
            staged = service._active_read
            if service is case.service and staged is not None and staged["phase"] == "staged" and not reached:
                reached.append(change)
                mutate_query(case, change, cleanup)
            return result

        monkeypatch.setattr(case.gateway, "_authorize_source_grant", authorize)
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(follow)
        assert reached == [change]
        assert case.service._continuations is registry and registry == before
        read_idle(case)


def assert_busy(error, elapsed):
    assert isinstance(error, typed.TypedStateOwnerRemoteError)
    assert error.error_type == "TypedStateOwnerProtocolError" and error.error_code == "protocol_denied"
    assert elapsed < 2, "overlapping source query did not promptly refuse"


def test_nested_query_on_another_real_connection_refuses_busy_and_outer_query_wins(query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    other, _token, other_grant = _client(case)
    nested = []

    def request_nested():
        started = time.monotonic()
        with pytest.raises(typed.TypedStateOwnerRemoteError) as refused:
            other.source_get(lookup_payload(case, maximum=1))
        assert_busy(refused.value, time.monotonic() - started)
        nested.append(True)
        assert registry == before

    state = observe_replay(case, monkeypatch, request_nested)
    result, raw = _assert_response(case, follow, case.client.source_get(follow))
    assert raw == DEFAULT_SOURCE[1:2] and nested == [True]
    assert state.reached == state.actions and len(state.reached) == 1
    assert (case.grant.grant_id, follow["continuation"]) not in registry
    assert registry == {(case.grant.grant_id, result["next_continuation"]): case.grant.expires_at}
    assert all(grant_id != other_grant.grant_id for grant_id, _cursor in registry)
    read_idle(case)


def test_concurrent_query_refuses_busy_before_cas_and_original_query_wins(query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    other, _token, other_grant = _client(case)
    entered, release = threading.Event(), threading.Event()

    def pause():
        entered.set()
        assert release.wait(timeout=2), "caller did not release retained replay"

    state = observe_replay(case, monkeypatch, pause)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(case.client.source_get, follow)
        try:
            assert entered.wait(timeout=2), "query did not reach retained replay"
            started = time.monotonic()
            with pytest.raises(typed.TypedStateOwnerRemoteError) as refused:
                other.source_get(lookup_payload(case, maximum=1))
            assert_busy(refused.value, time.monotonic() - started)
            assert registry == before
        finally:
            release.set()
        result, raw = _assert_response(case, follow, future.result(timeout=4))
    assert raw == DEFAULT_SOURCE[1:2]
    assert state.reached == state.actions and len(state.reached) == 1
    assert registry == {(case.grant.grant_id, result["next_continuation"]): case.grant.expires_at}
    assert all(grant_id != other_grant.grant_id for grant_id, _cursor in registry)
    read_idle(case)


def observe_source_clock(gateway, server_lock, monkeypatch, *, target, action):
    original_time = time.time
    state = SimpleNamespace(samples=0, reached=[], completed=[])

    def source_time():
        value = original_time()
        # Target only direct timestamps in the live source helper. Native
        # reentry legitimately checks daemon grants/lifecycle with their own
        # existing clocks; those clocks are outside this source-only contract.
        caller = sys._getframe(1)
        if (caller.f_code.co_name == "_execute_current_source_read"
                and caller.f_locals.get("self") is gateway
                and gateway._current_source_read_active is not None):
            state.samples += 1
            assert not server_lock._is_owned(), "source query clock must release SERVER"
            assert not gateway._grants_lock.locked(), "source query clock must release grant custody lock"
            assert not gateway._current_source_read_guard.locked()
            if state.samples == target:
                state.reached.append(target)
                action()
                state.completed.append(target)
        return value

    monkeypatch.setattr(time, "time", source_time)
    return state


def test_source_clock_callback_reenters_real_native_task_and_allocation_fences(native_query, monkeypatch):
    case, client, _grant = native_query
    gateway = case.native.server._command_gateway
    service = case.guard._lookup

    def reenter():
        assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
        assert case.guard.require_current()["allocation_ownership_verified"] is True

    state = observe_source_clock(gateway, case.native.server._lock, monkeypatch, target=1, action=reenter)
    response = client.source_get(_payload(case.native))
    assert _data(response) == SOURCE and response["ok"] is True
    assert state.samples == 3 and state.reached == state.completed == [1]
    assert response["result"]["response_bytes"] == len(canonical_json_bytes(response)) + 1
    assert response["result"]["response_wire_bytes"] == response["result"]["response_bytes"] + 3
    assert service._continuations == {} and service._active_read is None
    assert gateway._current_source_read_active is None
    assert service is case.guard._lookup is case.native.handle._source
    assert case.native.daemon.provider_invocation_recorded(case.native.attempt.attempt_id,
        idempotency_key="provider:" + case.native.attempt.attempt_id) is None


@pytest.mark.parametrize("change", ("admission_grant", "final_omitted", "final_task", "final_grant"))
def test_source_clock_callback_custody_loss_preserves_cursor_until_closing_refusal(
        query_case, monkeypatch, change):
    case = query_case
    follow, registry, before = seed_cursor(case)
    phase, mutation = change.split("_")
    target = 1 if phase == "admission" else 3
    with ExitStack() as cleanup:
        state = observe_source_clock(case.gateway, case.server._lock, monkeypatch, target=target,
            action=lambda: mutate_query(case, mutation, cleanup))
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(follow)
        assert state.samples == target and state.reached == state.completed == [target]
        assert case.service._continuations is registry and registry == before
        read_idle(case)


def test_expired_cursor_refusal_preserves_registry_and_fresh_read_can_issue_again(query_case):
    case = query_case
    follow, registry, _before = seed_cursor(case)
    key = case.grant.grant_id, follow["continuation"]
    registry[key] = int(time.time() * 1000) - 1
    expired = dict(registry)
    with pytest.raises(typed.TypedStateOwnerError):
        case.client.source_get(follow)
    assert case.service._continuations is registry and registry == expired
    read_idle(case)
    payload = lookup_payload(case, maximum=1)
    result, raw = _assert_response(case, payload, case.client.source_get(payload))
    assert raw == DEFAULT_SOURCE[:1]
    assert registry == {(case.grant.grant_id, result["next_continuation"]): case.grant.expires_at}
    assert registry[key] > expired[key]
    read_idle(case)


def test_legitimate_grant_renewal_before_query_uses_current_issued_record(query_case):
    case = query_case
    follow, registry, _before = seed_cursor(case)
    renewed = case.gateway.renew_grant(case.grant.grant_id, ttl_seconds=7200)
    assert renewed is not case.grant and renewed.grant_id == case.grant.grant_id
    assert renewed.expires_at > case.grant.expires_at
    result, raw = _assert_response(case, follow, case.client.source_get(follow))
    assert raw == DEFAULT_SOURCE[1:2]
    assert registry == {(renewed.grant_id, result["next_continuation"]): renewed.expires_at}
    read_idle(case)


def test_same_grant_renewal_during_cas_refuses_without_consuming_cursor(query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    renewed = []
    state = observe_replay(case, monkeypatch,
        lambda: renewed.append(case.gateway.renew_grant(case.grant.grant_id, ttl_seconds=7200)))
    with pytest.raises(typed.TypedStateOwnerError):
        case.client.source_get(follow)
    assert state.reached == state.actions and len(state.reached) == len(renewed) == 1
    assert renewed[0].grant_id == case.grant.grant_id and renewed[0] is not case.grant
    assert case.service._continuations is registry and registry == before
    read_idle(case)
    result, raw = _assert_response(case, follow, case.client.source_get(follow))
    assert raw == DEFAULT_SOURCE[1:2]
    assert registry == {(renewed[0].grant_id, result["next_continuation"]): renewed[0].expires_at}
    read_idle(case)


def test_admission_clock_registry_replacement_is_refused_before_preparation(query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    replacement = dict(registry)
    with ExitStack() as cleanup:
        cleanup.callback(setattr, case.service, "_continuations", registry)
        state = observe_source_clock(case.gateway, case.server._lock, monkeypatch, target=1,
            action=lambda: setattr(case.service, "_continuations", replacement))
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(follow)
        assert state.samples == 1 and state.reached == state.completed == [1]
        assert case.service._continuations is replacement and replacement == registry == before
        read_idle(case)


def test_redundant_cursor_field_cannot_skip_consuming_the_pinned_predecessor(query_case, monkeypatch):
    case = query_case
    follow, registry, _before = seed_cursor(case)
    state = observe_replay(case, monkeypatch,
        lambda: case.service._active_read.update(cursor=None))
    result, raw = _assert_response(case, follow, case.client.source_get(follow))
    assert raw == DEFAULT_SOURCE[1:2]
    assert state.reached == state.actions and len(state.reached) == 1
    assert (case.grant.grant_id, follow["continuation"]) not in registry
    assert registry == {(case.grant.grant_id, result["next_continuation"]): case.grant.expires_at}
    with pytest.raises(typed.TypedStateOwnerError):
        case.client.source_get(follow)
    read_idle(case)


def test_private_read_connection_substitution_refuses_before_native_sql_and_cursor_commit(query_case, monkeypatch):
    case = query_case
    follow, registry, before = seed_cursor(case)
    replacement = open_duckdb_connection(case.root.parent / "replacement-read-state-connection.duckdb")
    original_execute, sql_calls = replacement.execute, []

    def observed_execute(*args, **kwargs):
        sql_calls.append(True)
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(replacement, "execute", observed_execute)
    try:
        state = observe_replay(case, monkeypatch,
            lambda: case.service._active_read.update(connection=replacement))
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(follow)
        assert state.reached == state.actions and len(state.reached) == 1
        assert case.service._continuations is registry and registry == before
        assert case.gateway._connection is case.connection
        assert replacement.in_transaction is False and sql_calls == []
        read_idle(case)
    finally:
        replacement.close()
