"""Canonical construction replays release SERVER before retained CAS callbacks.

Native claims/grants, task bodies, Portal allocations/lifecycle, Git and the
retained CAS are real. The inherited Quack listener and deployment UID adapter
remain authored host seams; the adapter reports the actual boundary-byte digest.
These owner-side cases admit no worker peer or child and execute no source,
model, solver or official verifier. They cover the two broker-owned canonical
loads, not arbitrary callbacks inside the native source.get gateway.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_lookup as native
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as boundary_guard
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from test.api.test_current_source_task_admission import SOURCE, _payload, _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_native_nominated_source_allocation_custody import (
    bind, source_files,
)


LOADS = ("first", "second")
CHANGES = (
    "canonical_selected", "canonical_omitted", "allocated_selected",
    "allocated_omitted", "task_body", "daemon_grant", "released_claim",
    "allocation_lifecycle", "source_binding", "source_store",
    "boundary_bytes", "parent_mode",
)


@pytest.fixture
def construction_case(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files()) as case:
        scope = bind(case)
        original_boundary = boundary_guard._owner_boundary

        def observed_boundary(path):
            # Retain the fixture's explicitly authored UID metadata while
            # reporting the digest of real bytes instead of a fixed fixture
            # digest that would conceal a callback's boundary-byte change.
            value, _digest = original_boundary(path)
            return value, hashlib.sha256(Path(path).read_bytes()).hexdigest()

        monkeypatch.setattr(boundary_guard, "_owner_boundary", observed_boundary)
        yield case, scope


def replacement_reader(case):
    """A genuine unbound reader on the same allocation and existing CAS."""
    source = case.guard._lookup
    binding = case.native.binding
    return native.CurrentSourceLookup(case.native.workspace, cas=case.native.retained.cas,
        expected_inventory_cid=source._inventory_cid, request=source._request,
        limits=source._limits, task_cid=binding["task_cid"],
        task_revision=binding["task_revision"],
        expected_task_body_cid=binding["expected_task_body_cid"],
        context_cid=binding["context_cid"], allowed_paths=source._paths,
        max_response_bytes=binding["max_response_bytes"],
        max_slice_bytes=binding["max_slice_bytes"])


def construction_callback(case, monkeypatch, *, load, action):
    """Observe the original class/CAS/grant/RPC methods without replacing types."""
    target = 1 if load == "first" else 2
    original_load = native.CurrentSourceLookup._load
    original_get = case.native.retained.cas.get
    original_issue = case.native.server.issue_typed_client_grant_record
    original_source_get = typed.TypedStateOwnerConnection.source_get
    state = SimpleNamespace(canonical_loads=0, target_active=False,
        target_started=False, reached=[], completed=[], proxy_grants=[], source_get_calls=[])

    def canonical_load(reader):
        targeted = False
        if reader._root == case.native.canonical:
            state.canonical_loads += 1
            targeted = state.canonical_loads == target
        if targeted:
            state.target_active = state.target_started = True
        try:
            return original_load(reader)
        finally:
            if targeted:
                state.target_active = False

    def cas_get(cid):
        value = original_get(cid)
        if state.target_active and not state.reached:
            state.reached.append(cid)
            # Fail diagnostically before acquiring COOR if the published
            # construction path still retains SERVER around this callback.
            assert not case.native.server._lock._is_owned(), (
                "broker construction replay must release SERVER")
            assert state.proxy_grants == [] and state.source_get_calls == []
            assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
            assert case.guard.require_current()["allocation_ownership_verified"] is True
            action()
            state.completed.append(cid)
        return value

    def issue(**kwargs):
        token, grant = original_issue(**kwargs)
        if kwargs["client_id"].startswith("source-worker-proxy:"):
            state.proxy_grants.append(grant.grant_id)
        return token, grant

    def source_get(connection, payload):
        state.source_get_calls.append(True)
        return original_source_get(connection, payload)

    monkeypatch.setattr(native.CurrentSourceLookup, "_load", canonical_load)
    monkeypatch.setattr(case.native.retained.cas, "get", cas_get)
    monkeypatch.setattr(case.native.server, "issue_typed_client_grant_record", issue)
    monkeypatch.setattr(typed.TypedStateOwnerConnection, "source_get", source_get)
    return state


def mutate_construction(case, change, replacement, cleanup):
    """Alter original retained resources; preserve real native handle objects."""
    owner = case.native
    if change.startswith(("canonical_", "allocated_")):
        checkout, selection = change.split("_")
        root = owner.canonical if checkout == "canonical" else owner.workspace
        path = root / ("module.py" if selection == "selected" else "unselected.py")
        before = path.read_bytes()
        cleanup.callback(path.write_bytes, before)
        path.write_bytes(before + b"# changed during broker construction replay\n")
    elif change == "task_body":
        before = deepcopy(dict(owner.native_task.body))
        cleanup.callback(_update_body, owner, before)
        _update_body(owner, before | {"title": "changed during broker construction replay"})
    elif change == "daemon_grant":
        owner.server._command_gateway.revoke_grant(owner.daemon_grant_id)
    elif change == "released_claim":
        claim = owner.daemon.coordinator.get_task_claim(owner.attempt.claim_id)
        lease = owner.daemon.coordinator.protect_task_claim(claim)
        owner.daemon.coordinator.release(lease, reason="authored broker construction custody loss")
    elif change == "allocation_lifecycle":
        cleanup.enter_context(_record_change(case, lambda value: value.update(state="settling")))
    elif change == "source_binding":
        gateway = owner.server._command_gateway
        original = gateway._current_source_lookup
        cleanup.callback(setattr, gateway, "_current_source_lookup", original)
        gateway._current_source_lookup = replacement
    elif change == "source_store":
        source = case.guard._lookup
        original = source._store
        cleanup.callback(setattr, source, "_store", original)
        source._store = replacement._store
    elif change == "boundary_bytes":
        path = owner.factory["boundary"]
        before = path.read_bytes()
        cleanup.callback(path.write_bytes, before)
        path.write_bytes(before + b"changed during broker construction replay\n")
    else:
        assert change == "parent_mode"
        parent = owner.factory["boundary"].parent / "worker-observations"
        mode = stat.S_IMODE(parent.stat().st_mode)
        cleanup.callback(parent.chmod, mode)
        parent.chmod(0o700)


def assert_retained_native_binding(case, service, nominations):
    owner = case.native
    assert case.guard._lookup is owner.handle._source is service
    assert service._native_source_task_admission is owner.handle
    assert service._native_source_worktree_allocation is case.guard
    assert case.guard._nominated_source_scope is nominations[0]
    assert owner.handle._nominated_source_scope is nominations[1]
    assert case.guard._lookup_bytes == owner.handle._source_bytes
    # Refused broker construction consumes no new binding opportunity and
    # cannot clear or recreate the existing bind-once native reader markers.
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_current_source_lookup(**case.source_binding)
    with pytest.raises((ValueError, RuntimeError)):
        owner.handle.bind_current_source_lookup(repository=owner.workspace, **case.source_binding)
    assert owner.daemon.provider_invocation_recorded(owner.attempt.attempt_id,
        idempotency_key="provider:" + owner.attempt.attempt_id) is None


@pytest.mark.parametrize("load", LOADS)
def test_construction_canonical_cas_callback_reenters_native_fences_after_server_release(
        construction_case, monkeypatch, load):
    case, scope = construction_case
    service = case.guard._lookup
    nominations = (case.guard._nominated_source_scope, case.native.handle._nominated_source_scope)
    state = construction_callback(case, monkeypatch, load=load, action=lambda: None)
    session = broker.start_source_lookup_worker_broker(**case.native.factory)
    case.native.session = session
    assert state.target_started and not state.target_active
    assert state.reached == state.completed and len(state.reached) == 1
    assert state.canonical_loads == 2
    assert state.proxy_grants == [session._original_grant_id]
    assert state.source_get_calls == []
    files = source_files()
    expected_refs = [dict(raw_path_hex=name.hex(), source_cid=cid_for_bytes(raw),
        size_bytes=len(raw), git_mode="100644")
        for name, raw in sorted(files.items()) if name != b"unselected.py"]
    assert session.descriptor["source_refs"] == expected_refs
    assert len(expected_refs) == scope.observation()["selected_count"] == 4
    assert len(scope.observation()["required_paths"]) == 3
    assert (session.descriptor["proof_authority"] is session.descriptor["execution_authority"]
            is session.descriptor["completion_authority"] is False)
    assert session._process is None and not session._registered.is_set()
    assert not session._closed and session._listener is not None and session._thread.is_alive()
    assert session.socket_path.exists() and session.descriptor_path.exists()
    assert case.native.server._command_gateway._current_source_lookup is service
    observed = session._read_source(_payload(case.native))
    assert observed["content"].encode("utf-8") == SOURCE and observed["read_only"] is True
    assert observed["proof_authority"] is observed["completion_authority"] is False
    assert state.source_get_calls == [True]
    assert session._native_response_bytes == observed["backend_response_bytes"] > 0
    assert session._native_wire_bytes == observed["backend_response_wire_bytes"]
    assert session._native_wire_bytes == session._native_response_bytes + 3
    assert_retained_native_binding(case, service, nominations)


@pytest.mark.parametrize("load", LOADS)
@pytest.mark.parametrize("change", CHANGES)
def test_construction_cas_custody_loss_refuses_before_proxy_grant_and_publication(
        construction_case, monkeypatch, load, change):
    case, _scope = construction_case
    service = case.guard._lookup
    nominations = (case.guard._nominated_source_scope, case.native.handle._nominated_source_scope)
    replacement = replacement_reader(case) if change in {"source_binding", "source_store"} else None
    parent = case.native.factory["boundary"].parent / "worker-observations"
    existing_paths = set(parent.iterdir())
    assert existing_paths == set()
    with ExitStack() as cleanup:
        state = construction_callback(case, monkeypatch, load=load,
            action=lambda: mutate_construction(case, change, replacement, cleanup))
        # Retain the genuine failed constructor allocation for cleanup checks;
        # its class, __init__ and native guard objects are not substituted.
        session = broker.SourceLookupWorkerBroker.__new__(broker.SourceLookupWorkerBroker)
        with pytest.raises((ValueError, RuntimeError)):
            session.__init__(**case.native.factory)
            case.native.session = session  # Preserve cleanup if construction wrongly succeeds.
        assert state.target_started and not state.target_active
        assert state.reached == state.completed and len(state.reached) == 1
        assert state.proxy_grants == state.source_get_calls == []
        assert session._grant is session._client is session._listener is session._thread is None
        assert session._grant_revoked is False and session._closed is True
        assert session._native_response_bytes == session._native_wire_bytes == 0
        assert session._process is None and not session._registered.is_set()
        assert set(parent.iterdir()) == existing_paths
        assert not session.socket_path.exists() and not session.descriptor_path.exists()
        assert not session.audit_path.exists()
        gateway = case.native.server._command_gateway
        assert gateway._current_source_lookup is (replacement if change == "source_binding" else service)
        assert_retained_native_binding(case, service, nominations)


@pytest.mark.parametrize("change", ("canonical_omitted", "source_proxy_grant"))
def test_custody_loss_after_real_proxy_client_open_refuses_before_endpoints(
        construction_case, monkeypatch, change):
    case, _scope = construction_case
    service = case.guard._lookup
    nominations = (case.guard._nominated_source_scope, case.native.handle._nominated_source_scope)
    gateway = case.native.server._command_gateway
    assert case.native.daemon_grant_id not in gateway._revoked_grants
    parent = case.native.factory["boundary"].parent / "worker-observations"
    existing_paths = set(parent.iterdir())
    assert existing_paths == set()
    original_init = typed.TypedStateOwnerConnection.__init__
    opened = []
    with ExitStack() as cleanup:
        state = construction_callback(case, monkeypatch, load="first", action=lambda: None)

        def proxy_open(connection, *args, **kwargs):
            result = original_init(connection, *args, **kwargs)
            if kwargs.get("client_id", "").startswith("source-worker-proxy:"):
                opened.append(connection)
                assert len(opened) == len(state.proxy_grants) == 1
                assert not case.native.server._lock._is_owned(), (
                    "source proxy client open must release SERVER")
                assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
                assert case.guard.require_current()["allocation_ownership_verified"] is True
                if change == "canonical_omitted":
                    mutate_construction(case, change, None, cleanup)
                else:
                    gateway.revoke_grant(state.proxy_grants[0])
            return result

        monkeypatch.setattr(typed.TypedStateOwnerConnection, "__init__", proxy_open)
        session = broker.SourceLookupWorkerBroker.__new__(broker.SourceLookupWorkerBroker)
        with pytest.raises((ValueError, RuntimeError)):
            session.__init__(**case.native.factory)
            case.native.session = session  # Preserve cleanup if construction wrongly succeeds.
        assert len(opened) == len(state.proxy_grants) == 1
        assert state.reached == state.completed and len(state.reached) == 1
        assert state.source_get_calls == []
        proxy_id = state.proxy_grants[0]
        assert session._grant.grant_id == session._original_grant_id == proxy_id
        assert session._client is opened[0] and opened[0]._closed is True
        assert opened[0]._socket.fileno() == -1
        assert session._grant_revoked is True and session._closed is True
        assert proxy_id in gateway._revoked_grants
        assert case.native.daemon_grant_id not in gateway._revoked_grants
        assert session._listener is session._thread is None
        assert session._native_response_bytes == session._native_wire_bytes == 0
        assert session._process is None and not session._registered.is_set()
        assert set(parent.iterdir()) == existing_paths
        assert not session.socket_path.exists() and not session.descriptor_path.exists()
        assert not session.audit_path.exists()
        assert gateway._current_source_lookup is service
        assert_retained_native_binding(case, service, nominations)
