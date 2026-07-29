from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CONTROL_CATALOG_VERSION,
    EVENT_CURSOR_SCHEMA,
    MUTATION_OPERATIONS,
    OPERATION_CATALOG_V2,
    OPERATION_CATALOG_V2_REQUIREMENT_ID,
    CapabilityDegradation,
    ControlBounds,
    ControlBoundsError,
    CursorReplayError,
    EventCursor,
    EventCursorError,
    Operation,
    OperationAuthority,
    OperationCatalog,
    PaginationKind,
    UnsupportedCapabilityError,
    UnsupportedCatalogVersionError,
    discover_control_catalog,
    get_operation_catalog,
    negotiate_catalog_version,
    replay_event_page,
)


EXPECTED_FAMILIES = {
    "capabilities",
    "health",
    "status",
    "metrics",
    "goals",
    "tasks",
    "bundles",
    "lanes",
    "events",
    "receipts",
    "caches",
    "objective",
    "refill",
    "plan",
    "lifecycle",
    "retry",
    "cancel",
    "quarantine",
    "artifact_query",
    "validation_replay",
}


def test_catalog_is_complete_canonical_immutable_and_side_effect_free() -> None:
    catalog = discover_control_catalog()

    assert catalog is OPERATION_CATALOG_V2
    assert get_operation_catalog() is catalog
    assert catalog.catalog_version == CONTROL_CATALOG_VERSION
    assert catalog.requirement_id == OPERATION_CATALOG_V2_REQUIREMENT_ID
    assert frozenset(catalog.operations) == frozenset(Operation)
    assert {item.family for item in catalog} == EXPECTED_FAMILIES
    assert OperationCatalog.from_json(catalog.to_json()) == catalog
    assert OperationCatalog.from_json(catalog.to_json()).catalog_id == (
        catalog.catalog_id
    )

    with pytest.raises(FrozenInstanceError):
        catalog.catalog_version = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        catalog.operation(Operation.STATUS).request_schema["title"] = "drift"  # type: ignore[index]

    # Discovery returns the already validated local value. It has no resolver,
    # provider loader, process launcher, filesystem root, or runtime factory.
    assert not any(
        token in discover_control_catalog.__code__.co_names
        for token in ("open", "Popen", "import_module", "resolve", "backend")
    )


def test_every_operation_declares_the_complete_v2_policy() -> None:
    for descriptor in OPERATION_CATALOG_V2:
        operation = descriptor.operation
        assert descriptor.request_schema["properties"]["operation"]["const"] == (
            operation.value
        )
        assert descriptor.result_schema["properties"]["operation"]["const"] == (
            operation.value
        )
        assert descriptor.authority is operation.authority
        assert descriptor.target_descriptor.required_selectors
        assert descriptor.roots
        assert isinstance(descriptor.bounds, ControlBounds)
        assert descriptor.pagination.kind in PaginationKind
        assert descriptor.backend_capability
        assert descriptor.degradation in CapabilityDegradation
        assert descriptor.audit_receipt_schema

        guarded = (
            descriptor.supports_dry_run,
            descriptor.requires_idempotency,
            descriptor.requires_authorization,
            descriptor.requires_lease,
            descriptor.requires_fencing,
        )
        if operation in MUTATION_OPERATIONS:
            assert descriptor.authority is OperationAuthority.MUTATION
            assert all(guarded)
        else:
            assert not any(guarded)

    events = OPERATION_CATALOG_V2.operation(Operation.EVENTS)
    assert events.pagination.kind is PaginationKind.EVENT_CURSOR
    assert events.pagination.cursor_schema == EVENT_CURSOR_SCHEMA
    assert events.uses_event_cursor
    assert all(
        not descriptor.uses_event_cursor
        for descriptor in OPERATION_CATALOG_V2
        if descriptor.operation is not Operation.EVENTS
    )


def test_catalog_negotiates_highest_mutual_version_and_rejects_unknown() -> None:
    negotiation = OPERATION_CATALOG_V2.negotiate((1, 2))

    assert negotiation.selected_version == CONTROL_CATALOG_VERSION
    assert negotiation.catalog_id == OPERATION_CATALOG_V2.catalog_id
    assert OPERATION_CATALOG_V2.negotiate_version((2,)) == 2
    assert negotiate_catalog_version((1, 2), (2, 3)) == 2

    with pytest.raises(UnsupportedCatalogVersionError, match="mutually"):
        OPERATION_CATALOG_V2.negotiate((1, 3))
    with pytest.raises(UnsupportedCatalogVersionError, match="unsupported"):
        get_operation_catalog(99)
    with pytest.raises(Exception, match="unknown operation"):
        OPERATION_CATALOG_V2.operation("execute_shell")


def test_backend_capability_support_and_declared_degradation_are_distinct() -> None:
    status = OPERATION_CATALOG_V2.operation(Operation.STATUS)
    pause = OPERATION_CATALOG_V2.operation(Operation.PAUSE)

    supported = OPERATION_CATALOG_V2.require_backend_capability(
        Operation.STATUS, (status.backend_capability,)
    )
    assert supported.supported
    assert not supported.degraded

    with pytest.raises(UnsupportedCapabilityError, match="requires backend"):
        OPERATION_CATALOG_V2.require_backend_capability(Operation.STATUS, ())
    degraded = OPERATION_CATALOG_V2.resolve_backend_capability(
        Operation.STATUS, ()
    )
    assert not degraded.supported
    assert degraded.degraded
    assert degraded.degradation is CapabilityDegradation.LOCAL_READ_ONLY

    with pytest.raises(UnsupportedCapabilityError, match="requires backend"):
        OPERATION_CATALOG_V2.resolve_backend_capability(Operation.PAUSE, ())
    assert pause.degradation is CapabilityDegradation.FAIL_CLOSED


def test_catalog_rejects_request_and_page_bounds_above_the_declaration() -> None:
    tasks = OPERATION_CATALOG_V2.operation(Operation.TASKS)
    tasks.validate_bounds(ControlBounds(), page_limit=tasks.pagination.max_limit)

    with pytest.raises(ControlBoundsError, match="page limit"):
        tasks.validate_bounds(
            ControlBounds(), page_limit=tasks.pagination.max_limit + 1
        )
    with pytest.raises(ControlBoundsError, match="max_items"):
        tasks.validate_bounds(
            ControlBounds(
                max_items=tasks.bounds.max_items + 1,
                max_paths=tasks.bounds.max_paths,
                max_effects=tasks.bounds.max_effects,
            )
        )


def test_event_cursor_round_trip_replays_exactly_once_across_pages() -> None:
    initial = EventCursor.initial("events:repository", snapshot_id="log:abc")
    events = tuple(
        {
            "sequence": sequence,
            "event_id": f"event:{sequence}",
            "type": "task_changed",
        }
        for sequence in range(1, 6)
    )

    assert EventCursor.from_json(initial.to_json()) == initial
    assert EventCursor.from_token(initial.to_token()) == initial

    first = replay_event_page(
        events,
        initial.to_token(),
        limit=2,
        snapshot_id="log:abc",
    )
    second = replay_event_page(
        events,
        first.next_cursor,
        limit=3,
        snapshot_id="log:abc",
    )

    assert [event["sequence"] for event in first.events] == [1, 2]
    assert [event["sequence"] for event in second.events] == [3, 4, 5]
    assert first.has_more
    assert not second.has_more
    assert second.next_cursor.position == 5
    assert second.next_cursor.last_event_id == "event:5"


def test_event_cursor_rejects_tampering_foreign_replay_gaps_and_duplicates() -> None:
    cursor = EventCursor(
        stream_id="events:repository",
        position=2,
        last_event_id="event:2",
        snapshot_id="log:abc",
    )
    forged = cursor.to_record()
    forged["position"] = 3
    with pytest.raises(EventCursorError, match="identity"):
        EventCursor.from_dict(forged)
    with pytest.raises(CursorReplayError, match="different stream"):
        cursor.assert_replayable(
            stream_id="events:other",
            earliest_position=1,
            latest_position=3,
        )
    with pytest.raises(CursorReplayError, match="gap"):
        replay_event_page(
            (
                {"sequence": 1, "event_id": "event:1"},
                {"sequence": 2, "event_id": "event:2"},
                {"sequence": 4, "event_id": "event:4"},
            ),
            cursor,
            stream_id="events:repository",
        )
    with pytest.raises(CursorReplayError, match="ordered and unique"):
        replay_event_page(
            (
                {"sequence": 3, "event_id": "event:3"},
                {"sequence": 3, "event_id": "event:duplicate"},
            ),
            cursor,
        )
    with pytest.raises(EventCursorError, match="malformed"):
        EventCursor.from_token("not+a+cursor")
