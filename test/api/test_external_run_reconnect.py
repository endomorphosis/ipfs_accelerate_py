"""Deterministic tests for EAAEF-114 detach, reconnect and continuation cursors."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.api.external_run_handle import (
    EXTERNAL_RUN_HANDLE_INTERFACE,
    EXTERNAL_RUN_HANDLE_SCHEMA,
    ExternalRunHandle,
    deserialize_run_handle,
    serialize_run_handle,
)


OPERATOR = "principal:operator"
WORKER = "principal:worker"
SESSION = "session:reconnect"
REPO = "repo:reconnect"


def _start_request(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "session_id": SESSION,
        "repository_id": REPO,
        "objective_id": "objective:reconnect",
        "idempotency_key": "idem:reconnect-1",
    }
    values.update(changes)
    return values


def _open_handle(
    api: ExternalHandoffAPI | None = None, **changes: object
) -> ExternalRunHandle:
    return ExternalRunHandle.handoff(_start_request(**changes), api=api)


def test_serialize_deserialize_roundtrip() -> None:
    api = ExternalHandoffAPI()
    handle = _open_handle(api)
    handle.steer("narrow the owned-file patch")
    payload = handle.serialize()
    assert payload["schema"] == EXTERNAL_RUN_HANDLE_SCHEMA
    assert payload["interface"] == EXTERNAL_RUN_HANDLE_INTERFACE
    assert payload["run_id"] == handle.run_id
    assert payload["cursor"] == handle.cursor
    assert payload["authority_id"] == handle.authority_id
    assert payload["content_id"]
    assert payload["snapshot_id"] == payload["content_id"]

    restored = ExternalRunHandle.deserialize(payload, api=api)
    assert restored.run_id == handle.run_id
    assert restored.cursor == handle.cursor
    assert restored.authority_id == handle.authority_id
    assert restored.principal_id == handle.principal_id
    assert [event["content_id"] for event in restored.events] == [
        event["content_id"] for event in handle.events
    ]

    from_json = ExternalRunHandle.deserialize(json.dumps(payload), api=api)
    assert from_json.run_id == handle.run_id
    assert from_json.authority_id == handle.authority_id
    assert serialize_run_handle(handle)["content_id"] == payload["content_id"]
    assert deserialize_run_handle(payload, api=api).run_id == handle.run_id


def test_resume_from_cursor_returns_later_events_only() -> None:
    api = ExternalHandoffAPI()
    handle = _open_handle(api)
    origin = handle.cursor
    assert origin
    first = handle.steer("first continuation")
    second = handle.steer("second continuation")
    origin_ids = {origin}
    later = handle.resume_from(origin)
    later_ids = list(later.event_ids)
    assert origin not in later_ids
    assert first.cursor in later_ids
    assert second.cursor in later_ids
    assert later_ids[0] != origin
    assert set(origin_ids).isdisjoint(later_ids)
    assert list(later) == list(later.events)
    assert all(event["content_id"] != origin for event in later)

    empty = handle.resume_from(second.cursor)
    assert list(empty.event_ids) == []
    assert empty.cursor == second.cursor


def test_steer_and_cancel_reject_wrong_authority() -> None:
    api = ExternalHandoffAPI()
    handle = _open_handle(api)
    other = _open_handle(
        api, idempotency_key="idem:other", session_id="session:other"
    )
    assert handle.authority_id != other.authority_id

    with pytest.raises(ExternalHandoffAuthorityError, match="authority id") as steer_err:
        handle.steer("keep owned files only", authority_id=other.authority_id)
    assert steer_err.value.reason_code == "authority_mismatch"

    with pytest.raises(ExternalHandoffAuthorityError, match="authority id") as cancel_err:
        handle.cancel(authority_id=other.authority_id)
    assert cancel_err.value.reason_code == "authority_mismatch"

    forged = ExternalRunHandle(
        run_id=handle.run_id,
        cursor=handle.cursor,
        authority_id=other.authority_id,
        principal_id=OPERATOR,
        api=api,
    )
    with pytest.raises(ExternalHandoffAuthorityError) as forged_steer:
        forged.steer("forged steering")
    assert forged_steer.value.reason_code == "authority_mismatch"
    with pytest.raises(ExternalHandoffAuthorityError) as forged_cancel:
        forged.cancel()
    assert forged_cancel.value.reason_code == "authority_mismatch"

    receipt = handle.steer("authorized continuation")
    assert receipt.run_id == handle.run_id
    assert receipt.authority_id == handle.authority_id
    cancelled = handle.cancel()
    assert cancelled.run_status == "cancelled"
    assert cancelled.authority_id == handle.authority_id


def test_client_restart_reattaches_serialized_handle() -> None:
    api = ExternalHandoffAPI()
    handle = _open_handle(api)
    origin = handle.cursor
    handle.steer("before client restart")
    blob = handle.detach()
    assert handle.api is None

    reattached = ExternalRunHandle.deserialize(blob, api=api)
    later = reattached.resume_from(origin)
    assert origin not in later.event_ids
    assert later.event_ids
    steered = reattached.steer("after client restart")
    assert steered.run_id == handle.run_id
    assert steered.authority_id == handle.authority_id


def test_host_supervisor_restart_restores_run_and_cursor() -> None:
    host = ExternalHandoffAPI()
    handle = _open_handle(host)
    origin = handle.cursor
    handle.steer("before host restart")
    mid = handle.cursor
    handle.steer("still before host restart")
    blob = handle.serialize()

    restarted = ExternalHandoffAPI()
    restored = ExternalRunHandle.deserialize(blob, api=restarted)
    assert restored.run_id == handle.run_id
    assert restored.authority_id == handle.authority_id
    later = restored.resume_from(origin)
    assert origin not in later.event_ids
    assert mid in later.event_ids
    steered = restored.steer("after host restart")
    assert steered.run_id == handle.run_id
    assert steered.authority_id == handle.authority_id
    assert steered.reason_code == "steered"


def test_resume_from_unknown_cursor_fails_closed() -> None:
    handle = _open_handle()
    with pytest.raises(ExternalHandoffAPIError, match="cursor") as err:
        handle.resume_from("cursor:does-not-exist")
    assert err.value.reason_code == "unknown_cursor"
