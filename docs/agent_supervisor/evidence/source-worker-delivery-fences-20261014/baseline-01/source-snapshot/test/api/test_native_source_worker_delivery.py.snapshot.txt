"""Broker delivery CAS callbacks must release SERVER and close live custody.

The retained Git/CAS, native task/claim/grants, Portal allocation/lifecycle and
typed source.get owner connection are real. The fixture's Quack listener and
deployment UID metadata remain authored host seams. These owner-side reads do
not admit a separate worker UID, spawn a child or execute source or a model.
Callbacks target only the broker's canonical _selected load: native source.get
gateway callbacks are a separate, unchanged boundary. Sequential source checks
do not establish a filesystem writer lease.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test.api.test_current_source_task_admission import SOURCE, _payload, _update_body
from test.api.test_native_source_worker_spawn_custody import native_spawn  # noqa: F401


STAGES = ("prebackend", "postbackend")
CHANGES = (
    "canonical_selected", "canonical_omitted", "allocated_selected",
    "allocated_omitted", "task_body", "daemon_grant", "source_proxy_grant",
    "released_claim", "descriptor_bytes", "socket_mode",
)


def delivery_callback(case, session, monkeypatch, *, stage, action):
    """Observe genuine broker/CAS/RPC operations, replacing no native handle."""
    target = 1 if stage == "prebackend" else 2
    original_selected = session._selected
    original_load = session._canonical._load
    original_cas_get = case.native.retained.cas.get
    original_source_get = session._client.source_get
    state = SimpleNamespace(selected_calls=0, selected_active=False,
        load_active=False, target_load_started=False, reached=[], completed=[],
        backend_calls=[], backend_costs=[])

    def selected(payload):
        state.selected_calls += 1
        state.selected_active = state.selected_calls == target
        try:
            return original_selected(payload)
        finally:
            state.selected_active = False

    def canonical_load():
        targeted = state.selected_active
        if targeted:
            state.target_load_started = True
            state.load_active = True
        try:
            return original_load()
        finally:
            if targeted:
                state.load_active = False

    def cas_get(cid):
        value = original_cas_get(cid)
        if state.load_active and not state.reached:
            state.reached.append(cid)
            # Diagnose the old ordering before entering COOR. Attempting the
            # native fence while SERVER is retained can otherwise deadlock.
            assert not case.native.server._lock._is_owned(), (
                "broker-selected replay must release SERVER")
            assert len(state.backend_calls) == target - 1
            assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
            assert case.guard.require_current()["allocation_ownership_verified"] is True
            action()
            state.completed.append(cid)
        return value

    def source_get(payload):
        # This observer delegates to the original connection and gateway. CAS
        # reads there cannot activate the broker-only callback above.
        assert not state.selected_active and not state.load_active
        state.backend_calls.append(True)
        response = original_source_get(payload)
        state.backend_costs.append((response["result"]["response_bytes"],
                                    response["result"]["response_wire_bytes"]))
        return response

    monkeypatch.setattr(session, "_selected", selected)
    monkeypatch.setattr(session._canonical, "_load", canonical_load)
    monkeypatch.setattr(case.native.retained.cas, "get", cas_get)
    monkeypatch.setattr(session._client, "source_get", source_get)
    return state


def mutate_delivery_state(case, session, change):
    """Mutate the original retained resources; do not replace guarded handles."""
    native = case.native
    if change.startswith(("canonical_", "allocated_")):
        checkout, selection = change.split("_")
        root = native.canonical if checkout == "canonical" else native.workspace
        path = root / ("module.py" if selection == "selected" else "unselected.py")
        path.write_bytes(path.read_bytes() + b"# changed during broker delivery replay\n")
    elif change == "task_body":
        body = deepcopy(dict(native.native_task.body))
        body["title"] = "changed during broker delivery replay"
        _update_body(native, body)
    elif change in {"daemon_grant", "source_proxy_grant"}:
        grant_id = (native.daemon_grant_id if change == "daemon_grant"
                    else session._original_grant_id)
        native.server._command_gateway.revoke_grant(grant_id)
    elif change == "released_claim":
        claim = native.daemon.coordinator.get_task_claim(native.attempt.claim_id)
        lease = native.daemon.coordinator.protect_task_claim(claim)
        native.daemon.coordinator.release(lease, reason="authored broker delivery custody loss")
    elif change == "descriptor_bytes":
        original = session.descriptor_path.read_bytes()
        session.descriptor_path.chmod(0o600)
        session.descriptor_path.write_bytes(original + b"\n")
        session.descriptor_path.chmod(0o440)
    else:
        assert change == "socket_mode"
        session.socket_path.chmod(0o600)


def assert_original_handles(case, session):
    assert session._task_admission is case.native.handle
    assert session._worktree_admission is case.guard
    assert session._service is case.guard._lookup is case.native.handle._source
    assert session._cas is case.native.retained.cas
    assert case.native.daemon.provider_invocation_recorded(case.native.attempt.attempt_id,
        idempotency_key="provider:" + case.native.attempt.attempt_id) is None


@pytest.mark.parametrize("stage", STAGES)
def test_broker_selected_cas_callback_reenters_native_fences_after_server_release(
        native_spawn, monkeypatch, stage):
    case, session = native_spawn
    state = delivery_callback(case, session, monkeypatch, stage=stage, action=lambda: None)
    observed = session._read_source(_payload(case.native))
    assert state.target_load_started and not state.selected_active and not state.load_active
    assert state.reached == state.completed and len(state.reached) == 1
    assert state.selected_calls == 2 and len(state.backend_calls) == 1
    assert len(state.backend_costs) == 1
    assert observed["content"].encode("utf-8") == SOURCE
    assert observed["read_only"] is True
    assert observed["source_complete"] is True and observed["next_continuation"] is None
    assert (observed["proof_authority"] is observed["completion_authority"]
            is observed["edit_authority"] is observed["omission_authority"] is False)
    assert session._native_response_bytes == observed["backend_response_bytes"] == state.backend_costs[0][0]
    assert session._native_wire_bytes == observed["backend_response_wire_bytes"] == state.backend_costs[0][1]
    assert session._native_wire_bytes == session._native_response_bytes + 3
    assert session._records == [] and session._accepted == 0
    assert session._process is None and not session._registered.is_set()
    assert not session._closed
    assert_original_handles(case, session)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("change", CHANGES)
def test_broker_selected_cas_custody_loss_refuses_delivery_and_retains_backend_cost(
        native_spawn, monkeypatch, stage, change):
    case, session = native_spawn
    state = delivery_callback(case, session, monkeypatch, stage=stage,
        action=lambda: mutate_delivery_state(case, session, change))
    published = []
    with pytest.raises((ValueError, RuntimeError)):
        published.append(session._read_source(_payload(case.native)))
    assert published == []
    assert state.target_load_started and not state.selected_active and not state.load_active
    assert state.reached == state.completed and len(state.reached) == 1
    if stage == "prebackend":
        assert state.selected_calls == 1
        assert state.backend_calls == state.backend_costs == []
        assert session._native_response_bytes == session._native_wire_bytes == 0
    else:
        assert state.selected_calls == 2 and len(state.backend_calls) == 1
        assert len(state.backend_costs) == 1
        # The broker charges a validated native envelope before the closing
        # refusal. Failed delivery cannot erase bytes already incurred.
        assert session._native_response_bytes == state.backend_costs[0][0] > 0
        assert session._native_wire_bytes == state.backend_costs[0][1]
        assert session._native_wire_bytes == session._native_response_bytes + 3
    assert session._records == [] and session._accepted == 0
    assert session._process is None and not session._registered.is_set()
    assert_original_handles(case, session)
