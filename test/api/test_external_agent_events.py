"""Deterministic tests for EAAEF-130 typed external lifecycle events."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.observability.external_events import (
    CANONICAL_SUCCESS_KINDS,
    CONTRACT_VERSION,
    EXTERNAL_LIFECYCLE_EVENT_INTERFACE,
    EXTERNAL_LIFECYCLE_EVENT_SCHEMA,
    REQUIRED_IDENTITIES,
    SCHEMA_VERSION,
    TERMINAL_KINDS,
    ExternalLifecycleEvent,
    ExternalLifecycleEventStream,
    LifecycleEventKind,
    LifecycleIdentityError,
    LifecycleOrderError,
    LifecyclePrivacyError,
    decode_lifecycle_event,
    emit_lifecycle_event,
    validate_lifecycle_sequence,
)


FIXED_MS = 1_700_000_000_000
RUN_ID = "run:eaaef-130"
TASK_ID = "task:EAAEF-130"
ATTEMPT_ID = "attempt:1"
FENCE_TOKEN = "fence:token-1"
ARTIFACT_CID = "sha256:" + ("a" * 64)


def _identities() -> dict[str, str]:
    return {
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "attempt_id": ATTEMPT_ID,
        "fence_token": FENCE_TOKEN,
        "artifact_cid": ARTIFACT_CID,
    }


def _event(
    kind: LifecycleEventKind | str = LifecycleEventKind.HANDOFF_ACCEPTED,
    sequence: int = 0,
    **changes: object,
) -> ExternalLifecycleEvent:
    values: dict[str, object] = {
        "kind": kind,
        "sequence": sequence,
        "created_at_ms": FIXED_MS,
        **_identities(),
    }
    values.update(changes)
    return ExternalLifecycleEvent(**values)  # type: ignore[arg-type]


def _stream() -> ExternalLifecycleEventStream:
    return ExternalLifecycleEventStream(created_at_ms=FIXED_MS, **_identities())


def test_frozen_external_lifecycle_event_v1_schema() -> None:
    event = _event()
    payload = event.to_dict()
    assert EXTERNAL_LIFECYCLE_EVENT_INTERFACE == "ExternalLifecycleEvent@1"
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    assert event.schema == EXTERNAL_LIFECYCLE_EVENT_SCHEMA
    assert event.schema.endswith("@1")
    assert payload["interface"] == "ExternalLifecycleEvent@1"
    assert payload["schema"].endswith("@1")
    assert payload["contract_version"] == 1
    assert event.event_id == event.content_id
    with pytest.raises(FrozenInstanceError):
        event.kind = LifecycleEventKind.CLAIMED  # type: ignore[misc]


def test_canonical_handoff_to_terminal_order() -> None:
    stream = _stream()
    events = stream.emit_canonical_success_path()
    kinds = tuple(event.kind for event in events)
    assert kinds == CANONICAL_SUCCESS_KINDS
    assert [kind.value for kind in kinds] == [
        "handoff_accepted",
        "claimed",
        "leased",
        "launched",
        "checkpointed",
        "verified",
        "merge_proposed",
        "merge_accepted",
        "terminal_completed",
    ]
    assert events[-1].is_terminal
    assert TERMINAL_KINDS == {
        LifecycleEventKind.TERMINAL_COMPLETED,
        LifecycleEventKind.TERMINAL_CANCELLED,
        LifecycleEventKind.TERMINAL_FAILED,
        LifecycleEventKind.TERMINAL_QUARANTINED,
    }
    assert validate_lifecycle_sequence(events) == tuple(event.event_id for event in events)
    with pytest.raises(LifecycleOrderError, match="terminal"):
        stream.emit(LifecycleEventKind.CLAIMED)


def test_sequence_and_continuation_cursor_are_strictly_increasing() -> None:
    stream = _stream()
    events = stream.emit_canonical_success_path()
    sequences = [event.sequence for event in events]
    cursors = [event.continuation_cursor for event in events]
    assert sequences == list(range(len(events)))
    assert cursors == [sequence + 1 for sequence in sequences]
    assert cursors == sorted(set(cursors))
    assert stream.continuation_cursor == events[-1].continuation_cursor
    resumed = stream.after(events[3].continuation_cursor)
    assert [event.kind for event in resumed] == list(CANONICAL_SUCCESS_KINDS[4:])
    disordered = (_event(sequence=1), _event(kind=LifecycleEventKind.CLAIMED, sequence=1))
    with pytest.raises(LifecycleOrderError, match="strictly increasing"):
        validate_lifecycle_sequence(disordered)
    with pytest.raises(LifecycleOrderError, match="sequence \\+ 1"):
        _event(continuation_cursor=99)


def test_missing_identity_is_rejected() -> None:
    for field in REQUIRED_IDENTITIES:
        with pytest.raises(LifecycleIdentityError, match=field):
            _event(**{field: ""})
        with pytest.raises(LifecycleIdentityError, match=field):
            _event(**{field: None})
        payload = _event().to_dict()
        payload.pop(field)
        with pytest.raises(LifecycleIdentityError, match=field):
            ExternalLifecycleEvent.from_dict(payload)
    first = _event()
    second = _event(
        kind=LifecycleEventKind.CLAIMED,
        sequence=1,
        attempt_id="attempt:other",
    )
    with pytest.raises(LifecycleIdentityError, match="identities bound"):
        validate_lifecycle_sequence((first, second))


@pytest.mark.parametrize("field", list(REQUIRED_IDENTITIES))
def test_each_missing_identity_field_is_rejected(field: str) -> None:
    with pytest.raises(LifecycleIdentityError, match=field):
        emit_lifecycle_event(
            LifecycleEventKind.HANDOFF_ACCEPTED,
            sequence=0,
            **{**_identities(), field: ""},
        )


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "transcript",
        "transcript_body",
        "transcript_text",
        "body",
        "raw_transcript",
        "secret",
        "api_key",
        "access_token",
        "password",
        "chain_of_thought",
        "thinking",
    ),
)
def test_forbidden_transcript_and_secret_keys_are_rejected(forbidden_key: str) -> None:
    event = _event()
    payload = event.to_dict()
    assert forbidden_key not in payload
    payload[forbidden_key] = "exported chat"
    with pytest.raises(LifecyclePrivacyError, match="transcript|secret|chain-of-thought"):
        ExternalLifecycleEvent.from_dict(payload)
    nested = event.to_dict()
    nested["kind"] = {
        "value": "handoff_accepted",
        forbidden_key: "leaked",
    }
    with pytest.raises(LifecyclePrivacyError, match="transcript|secret|chain-of-thought"):
        ExternalLifecycleEvent.from_dict(nested)


def test_events_round_trip_and_keep_bound_identities() -> None:
    event = _event(kind=LifecycleEventKind.VERIFIED, sequence=5)
    restored = ExternalLifecycleEvent.from_json(event.to_json())
    assert restored == event
    assert restored.content_id == event.content_id
    assert decode_lifecycle_event(event.to_dict()) == event
    for field, value in _identities().items():
        assert getattr(event, field) == value
        assert event.to_dict()[field] == value


def test_unknown_kind_and_schema_are_rejected() -> None:
    with pytest.raises(Exception, match="kind must be one of"):
        _event(kind="accepted")
    payload = _event().to_dict()
    payload["schema"] = "ipfs_accelerate_py/agent-supervisor/external-lifecycle-event@2"
    with pytest.raises(Exception, match="unsupported schema"):
        ExternalLifecycleEvent.from_dict(payload)
    payload = _event().to_dict()
    payload["extra_field"] = "nope"
    with pytest.raises(Exception, match="unsupported fields"):
        ExternalLifecycleEvent.from_dict(payload)


def test_kind_order_rejects_regression_and_allows_terminal_failure() -> None:
    launched = _stream()
    launched.emit(LifecycleEventKind.HANDOFF_ACCEPTED)
    launched.emit(LifecycleEventKind.CLAIMED)
    launched.emit(LifecycleEventKind.LEASED)
    launched.emit(LifecycleEventKind.LAUNCHED)
    failed = launched.emit(LifecycleEventKind.TERMINAL_FAILED)
    assert failed.kind is LifecycleEventKind.TERMINAL_FAILED
    assert failed.sequence == 4
    assert failed.continuation_cursor == 5
    regression = (
        _event(),
        _event(kind=LifecycleEventKind.LAUNCHED, sequence=1),
        _event(kind=LifecycleEventKind.CLAIMED, sequence=2),
    )
    with pytest.raises(LifecycleOrderError, match="cannot follow"):
        validate_lifecycle_sequence(regression)
