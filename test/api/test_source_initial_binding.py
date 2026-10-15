"""Initial source binding on genuine owner/task/allocation resources.

The native DuckDB owner, task/claim/grants, Portal allocation/lifecycle and
Git/CAS are real. Existing listener/deployment UID metadata remains an authored
host seam. These owner-side cases create no broker, child or worker admission
and execute no source, provider, model or solver. Sequential checks establish
no filesystem writer lease or universal callback-safety qualification.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import stat
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_lookup as lookup
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as inventories
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from test.api.test_current_source_task_admission import _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_native_nominated_source_allocation_custody import (
    nominated_arguments, nomination, source_files,
)


DIAGNOSTIC = "initial source binding CAS replay must release SERVER"
ROUTES = ("allocation", "task", "server", "gateway")


@pytest.fixture
def unbound_allocation(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files()) as case:
        assert case.native.server._command_gateway._current_source_lookup is None
        assert case.native.handle._source is case.guard._lookup is None
        yield case


def initial_bind(case, route):
    native = case.native
    if route == "allocation":
        return case.guard.bind_current_source_lookup(**case.source_binding)
    if route == "task":
        return native.handle.bind_current_source_lookup(repository=native.workspace, **case.source_binding)
    current = native.handle.require_current()
    arguments = dict(case.source_binding, repository=native.workspace,
        task_cid=current["task_cid"], task_revision=current["task_revision"],
        expected_task_body_cid=current["expected_task_body_cid"])
    if route == "server":
        return native.server.bind_current_source_lookup(**arguments)
    assert route == "gateway"
    return native.server._command_gateway.bind_current_source_lookup(**arguments)


@pytest.mark.parametrize("route", ROUTES)
def test_initial_binding_cas_callback_reenters_native_fences_after_server_release(
        unbound_allocation, monkeypatch, route):
    case, native = unbound_allocation, unbound_allocation.native
    gateway = native.server._command_gateway
    original_lock, original_connection = native.server._lock, native.server._connection
    original_load, original_get = lookup.CurrentSourceLookup._load, native.retained.cas.get
    grants_before = frozenset(grant.grant_id for grant in gateway._grants.values())
    endpoints = native.factory["boundary"].parent / "worker-observations"
    endpoints_before = set(endpoints.iterdir())
    state = dict(active=False, loads=0, reached=[], completed=[])

    def replay(reader):
        targeted = reader._root == native.workspace
        if targeted:
            state["active"] = True
            state["loads"] += 1
        try:
            return original_load(reader)
        finally:
            if targeted:
                state["active"] = False

    def cas_get(cid):
        value = original_get(cid)
        if state["active"] and not state["reached"]:
            state["reached"].append(cid)
            # Diagnose every known SERVER recursion layer before acquiring
            # COOR through the original native handles, avoiding deadlock.
            assert not original_lock._is_owned(), DIAGNOSTIC
            assert native.handle.require_current()["task_cid"] == native.task_cid
            assert case.guard.require_current()["allocation_ownership_verified"] is True
            state["completed"].append(cid)
        return value

    monkeypatch.setattr(lookup.CurrentSourceLookup, "_load", replay)
    monkeypatch.setattr(native.retained.cas, "get", cas_get)
    binding = initial_bind(case, route)
    service = gateway._current_source_lookup
    assert type(service) is lookup.CurrentSourceLookup
    assert state["loads"] >= 1 and not state["active"]
    assert state["reached"] == state["completed"] and len(state["reached"]) == 1
    assert service._root == native.workspace and service._store.cas is native.retained.cas
    assert service._paths == case.source_binding["allowed_paths"]
    assert canonical_json_bytes(binding) == gateway._current_source_lookup_binding
    assert binding["task_cid"] == native.task_cid and binding["allowed_path_count"] == 1
    assert native.server._lock is gateway._transaction_lock is original_lock
    assert native.server._connection is gateway._connection is original_connection
    assert service._continuations == {} and service._active_read is None
    assert frozenset(grant.grant_id for grant in gateway._grants.values()) == grants_before
    assert set(endpoints.iterdir()) == endpoints_before
    if route in {"task", "allocation"}:
        assert native.handle._source is service
        assert service._native_source_task_admission is native.handle
    else:
        assert native.handle._source is None
        assert getattr(service, "_native_source_task_admission", None) is None
    if route == "allocation":
        assert case.guard._lookup is service
        assert service._native_source_worktree_allocation is case.guard
    else:
        assert case.guard._lookup is None
        assert getattr(service, "_native_source_worktree_allocation", None) is None
    assert native.session is None
    assert native.daemon.provider_invocation_recorded(native.attempt.attempt_id,
        idempotency_key="provider:" + native.attempt.attempt_id) is None


def observe_binding_replay(case, monkeypatch, action):
    """Gate a real CAS observer to the candidate's actual inventory replay."""
    native = case.native
    original_load, original_get = lookup.CurrentSourceLookup._load, native.retained.cas.get
    original_lock = native.server._lock
    state = SimpleNamespace(active=False, reader=None, reached=[], completed=[],
        original_load=original_load, original_get=original_get)

    def replay(reader):
        targeted = reader._root == native.workspace
        if targeted:
            state.active, state.reader = True, reader
        try:
            return original_load(reader)
        finally:
            if targeted:
                state.active = False

    def cas_get(cid):
        value = original_get(cid)
        if state.active and not state.reached:
            state.reached.append(cid)
            assert not original_lock._is_owned(), DIAGNOSTIC
            action(state.reader)
            state.completed.append(cid)
        return value

    monkeypatch.setattr(lookup.CurrentSourceLookup, "_load", replay)
    monkeypatch.setattr(native.retained.cas, "get", cas_get)
    return state


def assert_replayed_once(state):
    assert type(state.reader) is lookup.CurrentSourceLookup
    assert state.reached == state.completed and len(state.reached) == 1
    assert not state.active


def stop_observing_binding_replay(case, monkeypatch, state):
    monkeypatch.setattr(lookup.CurrentSourceLookup, "_load", state.original_load)
    monkeypatch.setattr(case.native.retained.cas, "get", state.original_get)


def assert_binding_idle(case):
    gateway = case.native.server._command_gateway
    assert gateway._current_source_bind_active is None
    assert gateway._current_source_read_active is None
    assert not gateway._current_source_read_guard.locked()
    assert case.native.handle._binding_in_progress is False
    assert case.guard._binding is False


def assert_unpublished(case):
    gateway = case.native.server._command_gateway
    assert all(value is None for value in (
        gateway._current_source_lookup, gateway._current_source_lookup_binding,
        gateway._current_source_lookup_owner, gateway._current_source_lookup_connection,
        gateway._current_source_lookup_publication))
    assert case.native.handle._source is case.guard._lookup is None
    assert case.native.handle._source_refs is case.guard._lookup_refs is None
    assert case.native.handle._nominated_source_scope is case.guard._nominated_source_scope is None
    assert_binding_idle(case)


def capture_no_delivery(case, monkeypatch):
    native, gateway = case.native, case.native.server._command_gateway
    endpoints = native.factory["boundary"].parent / "worker-observations"
    state = SimpleNamespace(grants=frozenset(grant.grant_id for grant in gateway._grants.values()),
        endpoints=endpoints, endpoint_names=set(endpoints.iterdir()), source_calls=[])
    original = gateway._execute_current_source_read

    def observed(*args, **kwargs):
        state.source_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(gateway, "_execute_current_source_read", observed)
    return state


def assert_no_delivery(case, state, *, revoked=()):
    gateway = case.native.server._command_gateway
    assert frozenset(grant.grant_id for grant in gateway._grants.values()) == state.grants.difference(revoked)
    assert set(state.endpoints.iterdir()) == state.endpoint_names
    assert state.source_calls == [] and case.native.session is None


REPLAY_CHANGES = (
    "selected", "omitted", "mode", "population", "request", "limits", "store",
    "task", "daemon_grant", "claim", "allocation", "connection", "lock", "owner_identity",
)


def mutate_binding(case, reader, change, cleanup, replacement=None):
    native, gateway = case.native, case.native.server._command_gateway
    if change in {"selected", "omitted"}:
        path = native.workspace / ("module.py" if change == "selected" else "unselected.py")
        before = path.read_bytes()
        cleanup.callback(path.write_bytes, before)
        path.write_bytes(before + b"# changed during initial binding\n")
    elif change == "mode":
        path = native.workspace / "module.py"
        before = stat.S_IMODE(path.stat().st_mode)
        cleanup.callback(path.chmod, before)
        path.chmod(before ^ stat.S_IXUSR)
    elif change == "population":
        path = native.workspace / "new-uninventoried.py"
        cleanup.callback(path.unlink, missing_ok=True)
        path.write_bytes(b"# Authored untracked source; never executed.\n")
    elif change in {"request", "limits"}:
        value = native.retained.request if change == "request" else native.retained.limits
        field = "task_id" if change == "request" else "max_entries"
        before = getattr(value, field)
        cleanup.callback(object.__setattr__, value, field, before)
        object.__setattr__(value, field, before + "-changed" if change == "request" else before - 1)
    elif change == "store":
        reader._store = replacement
    elif change == "task":
        before = deepcopy(dict(native.native_task.body))
        cleanup.callback(_update_body, native, before)
        _update_body(native, {**before, "title": "Changed during initial source binding"})
    elif change == "daemon_grant":
        gateway.revoke_grant(native.daemon_grant_id)
    elif change == "claim":
        coordinator = native.daemon.coordinator
        claim = coordinator.get_task_claim(native.attempt.claim_id)
        lease = coordinator.protect_task_claim(claim)
        coordinator.release(lease, reason="authored initial-binding callback release")
    elif change == "allocation":
        cleanup.enter_context(_record_change(case, lambda value: value.update(state="settling")))
    else:
        attribute = {"connection": "_connection", "lock": "_transaction_lock",
                     "owner_identity": "identity"}[change]
        before = getattr(gateway, attribute)
        cleanup.callback(setattr, gateway, attribute, before)
        setattr(gateway, attribute, replacement)


@pytest.mark.parametrize("change", REPLAY_CHANGES)
def test_binding_replay_custody_loss_refuses_without_native_consumption(
        unbound_allocation, monkeypatch, change):
    case, native = unbound_allocation, unbound_allocation.native
    gateway = native.server._command_gateway
    delivery = capture_no_delivery(case, monkeypatch)
    replacement_sql = []
    with ExitStack() as cleanup:
        replacement = None
        if change == "connection":
            # A distinct private file avoids opening the live owner's sealed
            # database a second time. A substituted handle must execute no SQL.
            replacement = open_duckdb_connection(native.workspace.parent / "replacement-initial-binding.duckdb")
            cleanup.callback(replacement.close)
            original_execute = replacement.execute

            def execute(*args, **kwargs):
                replacement_sql.append(True)
                return original_execute(*args, **kwargs)

            monkeypatch.setattr(replacement, "execute", execute)
        elif change == "lock":
            replacement = threading.RLock()
        elif change == "owner_identity":
            replacement = dict(gateway.identity)

        def mutate(reader):
            actual = inventories._BoundCAS(native.retained.cas, reader._guard) if change == "store" else replacement
            mutate_binding(case, reader, change, cleanup, actual)

        state = observe_binding_replay(case, monkeypatch, mutate)
        with pytest.raises((ValueError, RuntimeError)):
            initial_bind(case, "allocation")
        assert_replayed_once(state)
        assert_unpublished(case)
        revoked = (native.daemon_grant_id,) if change == "daemon_grant" else ()
        assert_no_delivery(case, delivery, revoked=revoked)
        assert replacement_sql == []
    assert_unpublished(case)
    if change in {"selected", "request", "store"}:
        # Restore only the caller's temporary mutation. The refused attempt
        # must not consume bind-once admission or prevent a genuine retry.
        stop_observing_binding_replay(case, monkeypatch, state)
        binding = initial_bind(case, "allocation")
        assert canonical_json_bytes(binding) == gateway._current_source_lookup_binding
        assert_exact_consumed(case, gateway._current_source_lookup,
            gateway._current_source_lookup_publication)
        assert_no_delivery(case, delivery)
    # Revocation and real coordinator release are intentionally not repaired.


def test_binding_candidate_cannot_publish_a_callback_invented_cursor_registry(
        unbound_allocation, monkeypatch):
    case = unbound_allocation
    delivery = capture_no_delivery(case, monkeypatch)
    key = ("authored-unissued-grant", "authored-unissued-cursor")
    state = observe_binding_replay(case, monkeypatch,
        lambda reader: reader._continuations.update({key: 1}))
    with pytest.raises((ValueError, RuntimeError)):
        initial_bind(case, "allocation")
    assert_replayed_once(state)
    assert state.reader._continuations == {key: 1}
    assert_unpublished(case)
    assert_no_delivery(case, delivery)


def test_transaction_started_during_binding_replay_remains_owned_by_the_caller(
        unbound_allocation, monkeypatch):
    case = unbound_allocation
    connection = case.native.server._connection
    delivery = capture_no_delivery(case, monkeypatch)
    state = observe_binding_replay(case, monkeypatch, lambda _reader: connection.execute("BEGIN"))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            initial_bind(case, "allocation")
        assert_replayed_once(state)
        assert connection.in_transaction is True
        assert_unpublished(case)
        assert_no_delivery(case, delivery)
    finally:
        if connection.in_transaction:
            connection.rollback()
    assert connection.in_transaction is False


def genuine_reader(case, *, scope=None):
    native = case.native
    current = native.handle.require_current()
    arguments = dict(case.source_binding, repository=native.workspace,
        task_cid=current["task_cid"], task_revision=current["task_revision"],
        expected_task_body_cid=current["expected_task_body_cid"])
    if scope is not None:
        arguments["allowed_paths"] = tuple(row["raw_path_hex"] for row in scope.observation()["selected"])
    return lookup.CurrentSourceLookup(**arguments)


def install_genuine_reader(gateway, reader):
    """Model another coherent reader without manufacturing its publication."""
    gateway._current_source_lookup = reader
    gateway._current_source_lookup_binding = canonical_json_bytes(reader.binding(gateway.identity))
    gateway._current_source_lookup_owner = canonical_json_bytes(dict(gateway.identity))
    gateway._current_source_lookup_connection = gateway._connection
    gateway._assert_current_source_binding(reader)


def assert_exact_consumed(case, service, publication):
    native, gateway = case.native, case.native.server._command_gateway
    assert type(service) is lookup.CurrentSourceLookup
    assert type(publication) is tuple and len(publication) == 8
    attempt, owner, connection, owner_bytes, retained, body, cas, root = publication
    assert attempt._published is gateway._current_source_lookup_publication is publication
    assert owner is gateway and connection is native.server._connection
    assert owner_bytes == canonical_json_bytes(dict(gateway.identity))
    assert retained is service and cas is native.retained.cas and root == native.workspace
    assert native.handle._source is case.guard._lookup is service
    assert native.handle._source_refs[0] is case.guard._lookup_refs[0] is service
    assert native.handle._source_refs[1] is case.guard._lookup_refs[1] is cas
    assert native.handle._source_refs[2] == case.guard._lookup_refs[2] == root
    assert native.handle._source_bytes == case.guard._lookup_bytes == body
    assert service._native_source_task_admission is native.handle
    assert service._native_source_worktree_allocation is case.guard
    assert_binding_idle(case)
    for bind in (
        lambda: initial_bind(case, "allocation"),
        lambda: initial_bind(case, "task"),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            bind()
    assert native.handle._source is case.guard._lookup is service


def test_own_publication_then_exception_retains_exact_consumed_native_reader(
        unbound_allocation, monkeypatch):
    case, native = unbound_allocation, unbound_allocation.native
    gateway, original = native.server._command_gateway, native.server.bind_current_source_lookup
    publications = []
    delivery = capture_no_delivery(case, monkeypatch)

    def publish_then_raise(**configuration):
        original(**configuration)
        publications.append(gateway._current_source_lookup_publication)
        raise RuntimeError("authored failure after own source publication")

    monkeypatch.setattr(native.server, "bind_current_source_lookup", publish_then_raise)
    with pytest.raises(RuntimeError, match="authored failure after own source publication"):
        initial_bind(case, "allocation")
    assert len(publications) == 1
    assert_exact_consumed(case, gateway._current_source_lookup, publications[0])
    assert len(publications) == 1
    assert_no_delivery(case, delivery)


@pytest.mark.parametrize("nominated", (False, True), ids=("legacy", "nominated"))
def test_postpublication_equal_reader_replacement_cannot_change_retained_native_joins(
        unbound_allocation, monkeypatch, nominated):
    case, native = unbound_allocation, unbound_allocation.native
    gateway = native.server._command_gateway
    scope = nomination(case) if nominated else None
    replacement = genuine_reader(case, scope=scope)
    original = native.server.bind_current_source_lookup
    publications = []
    delivery = capture_no_delivery(case, monkeypatch)

    def publish_replace_raise(**configuration):
        result = original(**configuration)
        publication = gateway._current_source_lookup_publication
        publications.append(publication)
        assert canonical_json_bytes(replacement.binding(gateway.identity)) == canonical_json_bytes(result)
        assert replacement is not publication[4]
        install_genuine_reader(gateway, replacement)
        raise RuntimeError("authored replacement after own source publication")

    monkeypatch.setattr(native.server, "bind_current_source_lookup", publish_replace_raise)
    with pytest.raises(RuntimeError, match="authored replacement after own source publication"):
        if nominated:
            case.guard.bind_nominated_current_source_lookup(**nominated_arguments(case, scope))
        else:
            initial_bind(case, "allocation")
    assert len(publications) == 1 and gateway._current_source_lookup is replacement
    original_reader = publications[0][4]
    assert_exact_consumed(case, original_reader, publications[0])
    assert len(publications) == 1
    assert getattr(replacement, "_native_source_task_admission", None) is None
    assert getattr(replacement, "_native_source_worktree_allocation", None) is None
    if nominated:
        task_join = native.handle._nominated_source_scope
        allocation_join = case.guard._nominated_source_scope
        assert task_join is native.handle._original_nominated_source_scope
        assert allocation_join is case.guard._original_nominated_source_scope
        assert task_join[0] is allocation_join[0] is scope
        assert allocation_join[2] is task_join
        assert task_join[4] is allocation_join[3] is original_reader
        assert task_join[5] == allocation_join[4] == publications[0][5]
    for handle in (native.handle, case.guard):
        with pytest.raises((ValueError, RuntimeError)):
            handle.require_source_binding(server=native.server, service=original_reader)
    assert_no_delivery(case, delivery)


def test_competing_genuine_reader_before_publication_is_never_adopted_or_cleared(
        unbound_allocation, monkeypatch):
    case, native = unbound_allocation, unbound_allocation.native
    gateway = native.server._command_gateway
    replacement = genuine_reader(case)
    delivery = capture_no_delivery(case, monkeypatch)
    state = observe_binding_replay(case, monkeypatch,
        lambda _reader: install_genuine_reader(gateway, replacement))
    with pytest.raises((ValueError, RuntimeError)):
        initial_bind(case, "allocation")
    assert_replayed_once(state)
    assert gateway._current_source_lookup is replacement
    assert gateway._current_source_lookup_publication is None
    assert native.handle._source is case.guard._lookup is None
    assert native.handle._source_refs is case.guard._lookup_refs is None
    assert getattr(replacement, "_native_source_task_admission", None) is None
    assert getattr(replacement, "_native_source_worktree_allocation", None) is None
    assert_binding_idle(case)
    with pytest.raises((ValueError, RuntimeError)):
        initial_bind(case, "allocation")
    assert gateway._current_source_lookup is replacement
    assert native.handle._source is case.guard._lookup is None
    assert_no_delivery(case, delivery)


def assert_stable_host_publication(case, binding):
    gateway = case.native.server._command_gateway
    reader = gateway._current_source_lookup
    assert type(reader) is lookup.CurrentSourceLookup
    assert canonical_json_bytes(binding) == gateway._current_source_lookup_binding
    assert gateway._current_source_lookup_publication[4] is reader
    assert reader._continuations == {} and reader._active_read is None
    assert case.native.handle._source is case.guard._lookup is None
    assert_binding_idle(case)


@pytest.mark.parametrize("route", ("gateway", "server"))
def test_nested_host_binding_refuses_busy_and_preserves_the_outer_publication(
        unbound_allocation, monkeypatch, route):
    case = unbound_allocation
    delivery = capture_no_delivery(case, monkeypatch)
    refused = []

    def nested(_reader):
        gateway = case.native.server._command_gateway
        assert gateway._current_source_bind_active is not None
        started = time.monotonic()
        with pytest.raises(typed.TypedStateOwnerProtocolError, match="busy"):
            initial_bind(case, route)
        assert time.monotonic() - started < 2, "nested host binding did not promptly refuse"
        refused.append(True)

    state = observe_binding_replay(case, monkeypatch, nested)
    binding = initial_bind(case, route)
    assert_replayed_once(state)
    assert refused == [True]
    assert_stable_host_publication(case, binding)
    assert_no_delivery(case, delivery)


@pytest.mark.parametrize("route", ("gateway", "server"))
def test_concurrent_host_binding_refuses_busy_and_preserves_the_outer_publication(
        unbound_allocation, monkeypatch, route):
    case = unbound_allocation
    delivery = capture_no_delivery(case, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def pause(_reader):
        entered.set()
        assert release.wait(timeout=8), "caller did not release initial binding replay"

    state = observe_binding_replay(case, monkeypatch, pause)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(initial_bind, case, route)
        try:
            assert entered.wait(timeout=5), "initial binding did not reach retained CAS replay"
            started = time.monotonic()
            with pytest.raises(typed.TypedStateOwnerProtocolError, match="busy"):
                initial_bind(case, route)
            assert time.monotonic() - started < 2, "concurrent host binding did not promptly refuse"
        finally:
            release.set()
        binding = future.result(timeout=8)
    assert_replayed_once(state)
    assert_stable_host_publication(case, binding)
    assert_no_delivery(case, delivery)


def test_preexisting_borrowed_transaction_refuses_before_native_sql_or_claim_fence(
        unbound_allocation, monkeypatch):
    case, native = unbound_allocation, unbound_allocation.native
    connection = native.server._connection
    entered, release = threading.Event(), threading.Event()
    sql, fences = [], []
    original_execute = connection.execute
    original_fence = native.daemon.coordinator.execute_with_task_claim_fence
    delivery = capture_no_delivery(case, monkeypatch)

    def transaction_owner():
        original_execute("BEGIN")
        entered.set()
        try:
            assert release.wait(timeout=8), "caller did not release its borrowed transaction"
        finally:
            connection.rollback()

    def observed_sql(*args, **kwargs):
        sql.append(True)
        # Fail diagnostically before waiting on another thread's transaction.
        assert not connection.in_transaction, "initial binding entered SQL inside a borrowed transaction"
        return original_execute(*args, **kwargs)

    def observed_fence(*args, **kwargs):
        fences.append(True)
        assert not connection.in_transaction, "initial binding entered COOR inside a borrowed transaction"
        return original_fence(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(transaction_owner)
        try:
            assert entered.wait(timeout=5), "caller did not establish its borrowed transaction"
            monkeypatch.setattr(connection, "execute", observed_sql)
            monkeypatch.setattr(native.daemon.coordinator, "execute_with_task_claim_fence", observed_fence)
            started = time.monotonic()
            with pytest.raises((ValueError, RuntimeError), match="inactive original transaction"):
                initial_bind(case, "allocation")
            assert time.monotonic() - started < 2
            assert connection.in_transaction is True and sql == fences == []
            assert_unpublished(case)
            assert_no_delivery(case, delivery)
        finally:
            release.set()
        future.result(timeout=5)
    assert connection.in_transaction is False


def observe_closing_binding_clock(case, monkeypatch, action):
    gateway, original_time = case.native.server._command_gateway, time.time
    original_lock = case.native.server._lock
    code = type(gateway).bind_current_source_lookup.__code__
    state = SimpleNamespace(samples=0, reached=[], completed=[])

    def binding_time():
        value = original_time()
        caller = sys._getframe(1)
        if (caller.f_code is code and caller.f_locals.get("self") is gateway
                and gateway._current_source_bind_active is not None):
            state.samples += 1
            assert not original_lock._is_owned(), "initial binding clock must release SERVER"
            assert not gateway._grants_lock.locked(), "initial binding clock must release grant custody lock"
            assert not gateway._current_source_read_guard.locked()
            state.reached.append(state.samples)
            action()
            state.completed.append(state.samples)
        return value

    monkeypatch.setattr(time, "time", binding_time)
    return state


def test_closing_binding_clock_callback_reenters_native_task_and_allocation_fences(
        unbound_allocation, monkeypatch):
    case, native = unbound_allocation, unbound_allocation.native
    delivery = capture_no_delivery(case, monkeypatch)

    def reenter():
        assert native.handle.require_current()["task_cid"] == native.task_cid
        assert case.guard.require_current()["allocation_ownership_verified"] is True

    state = observe_closing_binding_clock(case, monkeypatch, reenter)
    binding = initial_bind(case, "allocation")
    assert state.samples == 1 and state.reached == state.completed == [1]
    gateway = native.server._command_gateway
    assert canonical_json_bytes(binding) == gateway._current_source_lookup_binding
    assert_exact_consumed(case, gateway._current_source_lookup, gateway._current_source_lookup_publication)
    assert_no_delivery(case, delivery)


@pytest.mark.parametrize("change", ("daemon_grant", "allocation"))
def test_closing_binding_clock_custody_loss_refuses_before_any_publication(
        unbound_allocation, monkeypatch, change):
    case = unbound_allocation
    delivery = capture_no_delivery(case, monkeypatch)
    with ExitStack() as cleanup:
        state = observe_closing_binding_clock(case, monkeypatch,
            lambda: mutate_binding(case, None, change, cleanup))
        with pytest.raises((ValueError, RuntimeError)):
            initial_bind(case, "allocation")
        assert state.samples == 1 and state.reached == state.completed == [1]
        assert_unpublished(case)
        revoked = (case.native.daemon_grant_id,) if change == "daemon_grant" else ()
        assert_no_delivery(case, delivery, revoked=revoked)
