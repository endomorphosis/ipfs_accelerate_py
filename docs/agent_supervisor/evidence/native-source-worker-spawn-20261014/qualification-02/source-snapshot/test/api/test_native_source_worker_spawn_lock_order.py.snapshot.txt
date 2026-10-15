"""Middle source replay must release SERVER before caller CAS callbacks.

These cases retain real native claim/Portal/allocation fences, Git/CAS and one
inert same-owner Python child. Listener and deployment metadata remain authored
host seams. No model, target source, solver or separate worker UID is exercised.
"""
import pytest

from test.api.test_current_source_task_admission import SOURCE, _payload
from test.api.test_native_source_worker_spawn_custody import (
    inert_child, native_spawn,  # noqa: F401
)


@pytest.mark.parametrize("reader_kind", ["canonical", "service"])
@pytest.mark.parametrize("stage", ["prepare", "postbirth"])
def test_middle_cas_callback_enters_native_fence_only_after_server_release(
        native_spawn, monkeypatch, reader_kind, stage):
    case, session = native_spawn
    if stage == "postbirth":
        assert session.require_before_spawn() is None

    original_canonical = session._canonical._load
    original_service = session._service._load
    original_get = case.native.retained.cas.get
    state = dict(canonical_completed=False, target_started=False, middle_active=False)
    reached, completed = [], []

    def target_load(original):
        state["target_started"] = True
        state["middle_active"] = True
        try:
            return original()
        finally:
            state["middle_active"] = False

    def canonical_load():
        try:
            if reader_kind == "canonical" and not state["target_started"]:
                return target_load(original_canonical)
            return original_canonical()
        finally:
            # The native nomination observer can load the allocated reader
            # earlier. Only its next load after the canonical middle replay
            # is the directly paired service replay under test.
            state["canonical_completed"] = True

    def service_load():
        if (reader_kind == "service" and state["canonical_completed"]
                and not state["target_started"]):
            return target_load(original_service)
        return original_service()

    def cas_callback(cid):
        value = original_get(cid)
        if state["middle_active"] and not reached:
            reached.append(cid)
            # Fail before entering COOR if the old placement still holds
            # SERVER. This makes the regression diagnostic without deadlock.
            assert not case.native.server._lock._is_owned()
            assert case.native.handle.require_current()["task_cid"] == case.native.task_cid
            assert case.guard.require_current()["allocation_ownership_verified"] is True
            completed.append(cid)
        return value

    monkeypatch.setattr(session._canonical, "_load", canonical_load)
    monkeypatch.setattr(session._service, "_load", service_load)
    monkeypatch.setattr(case.native.retained.cas, "get", cas_callback)

    if stage == "prepare":
        assert session.require_before_spawn() is None
        assert reached == completed and len(reached) == 1
        assert session._process is None and not session._registered.is_set()

    with inert_child(case.native.workspace) as process:
        session.note_spawned(process)
        assert state["target_started"] and not state["middle_active"]
        assert reached == completed and len(reached) == 1
        assert session._process is process and session._registered.is_set()
        assert not session._closed
        observed = session._read_source(_payload(case.native))
        assert observed["content"].encode() == SOURCE and observed["read_only"] is True
        assert observed["proof_authority"] is observed["completion_authority"] is False
        assert case.native.daemon.provider_invocation_recorded(case.native.attempt.attempt_id,
            idempotency_key="provider:" + case.native.attempt.attempt_id) is None
    assert process.poll() is not None
