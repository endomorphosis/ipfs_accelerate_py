"""Tests for DatabaseEventLog@1 (DQP-013).

Evidence subset: monotonic sequences, duplicate IDs, cursor expiry, bounded
polling, coalescing, replay, redaction, retention, event/projection transaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CursorReplayError,
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    AUDIT_EVENT_TYPE,
    CONSUMER_CHECKPOINT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    ConsumerCheckpoint,
    DatabaseEventLog,
    DatabaseEventLogDuplicateError,
    DatabaseEventLogNotOpenError,
    DatabaseEventLogPayloadError,
    LogSeverity,
    open_database_event_log,
    redact_value,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database event log hermetic tests",
)


def _log(tmp_path: Path) -> DatabaseEventLog:
    return open_database_event_log(tmp_path / "control.duckdb")


def test_interface_identities(tmp_path: Path) -> None:
    log = _log(tmp_path)
    assert log.INTERFACE == DATABASE_EVENT_LOG_INTERFACE
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert ConsumerCheckpoint.INTERFACE == CONSUMER_CHECKPOINT_INTERFACE
    payload = log.to_dict()
    assert payload["interface"] == DATABASE_EVENT_LOG_INTERFACE
    assert payload["snapshot_id"]


def test_monotonic_sequences_and_immutable_event_ids(tmp_path: Path) -> None:
    log = _log(tmp_path)
    first = log.append_event("stream:tasks", "task.created", {"task": "T-1"})
    second = log.append_event("stream:tasks", "task.queued", {"task": "T-1"})
    third = log.append_event("stream:other", "metric.tick", {"n": 1})

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert third["sequence"] == 1
    assert first["global_sequence"] == 1
    assert second["global_sequence"] == 2
    assert third["global_sequence"] == 3
    assert first["event_id"].startswith("sha256:")
    assert first["event_id"] != second["event_id"]

    head = log.stream_head("stream:tasks")
    assert head.latest_sequence == 2
    assert head.last_event_id == second["event_id"]
    assert head.earliest_sequence == 1


def test_duplicate_ids_coalesce_exact_and_reject_conflicts(tmp_path: Path) -> None:
    log = _log(tmp_path)
    original = log.append_event(
        "stream:tasks",
        "task.created",
        {"task": "T-1"},
        event_id="evt:stable-1",
    )
    again = log.append_event(
        "stream:tasks",
        "task.created",
        {"task": "T-1"},
        event_id="evt:stable-1",
    )
    assert again["event_id"] == original["event_id"]
    assert again["sequence"] == original["sequence"]
    assert log.stream_head("stream:tasks").latest_sequence == 1

    with pytest.raises(DatabaseEventLogDuplicateError):
        log.append_event(
            "stream:tasks",
            "task.created",
            {"task": "T-DIFFERENT"},
            event_id="evt:stable-1",
        )


def test_bounded_polling_and_replay_without_loss_or_duplicates(
    tmp_path: Path,
) -> None:
    log = _log(tmp_path)
    for index in range(5):
        log.append_event("stream:tasks", "tick", {"n": index})

    cursor = log.initial_cursor("stream:tasks")
    first = log.poll(cursor, limit=2)
    assert len(first.events) == 2
    assert first.has_more is True
    assert [int(event["body"]["n"]) for event in first.events] == [0, 1]

    second = log.poll(first.next_cursor, limit=2)
    assert len(second.events) == 2
    assert [int(event["body"]["n"]) for event in second.events] == [2, 3]
    assert second.has_more is True

    third = log.poll(second.next_cursor, limit=2)
    assert len(third.events) == 1
    assert third.has_more is False
    assert int(third.events[0]["body"]["n"]) == 4

    # Resume from latest cursor yields empty page without advancing.
    empty = log.poll(log.latest_cursor("stream:tasks"), limit=10)
    assert empty.events == ()
    assert empty.has_more is False
    assert empty.next_cursor.position == 5


def test_consumer_checkpoint_resume_and_cursor_expiry(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for index in range(4):
        log.append_event("stream:tasks", "tick", {"n": index})

    page = log.poll(log.initial_cursor("stream:tasks"), limit=2)
    checkpoint = log.save_consumer_checkpoint("worker-a", page.next_cursor)
    assert checkpoint.consumer_id == "worker-a"
    assert checkpoint.position == 2

    loaded = log.load_consumer_checkpoint("worker-a", "stream:tasks")
    assert loaded is not None
    assert loaded.cursor.last_event_id == page.next_cursor.last_event_id

    resumed = log.poll(loaded.cursor, limit=10)
    assert [int(event["body"]["n"]) for event in resumed.events] == [2, 3]

    # Retention past the checkpoint expires it (cursor older than earliest-1).
    log.apply_retention("stream:tasks", retain_after_sequence=3)
    with pytest.raises(CursorReplayError):
        log.poll(loaded.cursor, limit=10)


def test_redaction_on_events_logs_and_metrics(tmp_path: Path) -> None:
    assert redact_value({"api_key": "secret", "ok": 1}) == {
        "api_key": "[REDACTED]",
        "ok": 1,
    }
    log = _log(tmp_path)
    event = log.append_event(
        "stream:secure",
        "secret.stored",
        {"password": "hunter2", "user": "alice"},
    )
    assert event["body"]["password"] == "[REDACTED]"
    assert event["body"]["user"] == "alice"

    structured = log.append_structured_log(
        severity=LogSeverity.INFO,
        component="auth",
        message="login",
        body={"token": "abc", "ip": "127.0.0.1"},
    )
    assert structured is not None
    assert structured["body"]["token"] == "[REDACTED]"
    assert structured["body"]["ip"] == "127.0.0.1"

    sample = log.append_metric_sample(
        "provider.calls",
        1000,
        labels={"authorization": "Bearer x", "route": "chat"},
    )
    assert sample["labels"]["authorization"] == "[REDACTED]"
    assert sample["labels"]["route"] == "chat"


def test_explicit_application_audit_not_quack_diagnostics(tmp_path: Path) -> None:
    log = _log(tmp_path)
    audit = log.append_audit(
        "lease.steal",
        actor="operator:alice",
        resource="task:T-1",
        outcome="denied",
        details={"reason": "policy"},
    )
    assert audit["type"] == AUDIT_EVENT_TYPE
    assert audit["body"]["action"] == "lease.steal"
    assert audit["body"]["actor"] == "operator:alice"
    page = log.poll(log.initial_cursor("audit"), limit=10)
    assert len(page.events) == 1
    assert page.events[0]["type"] == AUDIT_EVENT_TYPE


def test_recursive_logging_is_bounded(tmp_path: Path) -> None:
    log = _log(tmp_path)
    from ipfs_accelerate_py.agent_supervisor.runtime import database_event_log as mod

    # Manually exercise depth guard via module helpers.
    assert mod._enter_log() == 1
    assert mod._enter_log() == 2
    assert mod._enter_log() == 3
    # At depth 3 (> MAX_RECURSION_DEPTH), append_structured_log should no-op.
    dropped = log.append_structured_log(
        severity="info",
        component="bounded",
        message="should-drop",
    )
    assert dropped is None
    mod._exit_log()
    mod._exit_log()
    mod._exit_log()

    kept = log.append_structured_log(
        severity="info",
        component="bounded",
        message="should-keep",
    )
    assert kept is not None
    assert kept["message"] == "should-keep"


def test_event_projection_transaction_atomic(tmp_path: Path) -> None:
    log = _log(tmp_path)
    event = log.append_with_projection(
        "stream:tasks",
        "task.projected",
        {"task": "T-9"},
        projection_key="task-head",
        projection_value={"status": "ready", "task": "T-9"},
    )
    projection = log.get_projection("stream:tasks", "task-head")
    assert projection is not None
    assert projection["event_id"] == event["event_id"]
    assert projection["value"]["status"] == "ready"
    assert log.stream_head("stream:tasks").latest_sequence == 1


def test_integrity_checkpoint_and_export_jsonl_non_authoritative(
    tmp_path: Path,
) -> None:
    log = _log(tmp_path)
    log.append_event("stream:tasks", "a", {"n": 1})
    log.append_event("stream:tasks", "b", {"n": 2})
    checkpoint = log.create_integrity_checkpoint("stream:tasks")
    assert checkpoint.event_count == 2
    assert log.verify_integrity_checkpoint(checkpoint) is True
    assert log.verify_integrity_checkpoint(checkpoint.checkpoint_id) is True

    export_path = tmp_path / "export" / "events.jsonl"
    receipt = log.export_jsonl(export_path, stream_id="stream:tasks")
    assert receipt["authoritative"] is False
    assert receipt["event_count"] == 2
    assert export_path.is_file()
    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # Deleting the export has no authority effect.
    export_path.unlink()
    assert log.stream_head("stream:tasks").latest_sequence == 2
    page = log.poll(log.initial_cursor("stream:tasks"), limit=10)
    assert len(page.events) == 2


def test_retention_preserves_tip_and_advances_earliest(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for index in range(5):
        log.append_event("stream:tasks", "tick", {"n": index})
    head = log.apply_retention("stream:tasks", retain_after_sequence=3)
    assert head.earliest_sequence == 4
    assert head.latest_sequence == 5
    # After retention, events 1-3 are gone so replay from 0 has a gap.
    with pytest.raises(CursorReplayError):
        log.poll(log.initial_cursor("stream:tasks"), limit=10)

    # Cursor exactly one behind the retained floor remains resumeable.
    floor_cursor = EventCursor(
        stream_id="stream:tasks",
        position=3,
        last_event_id="retained-floor",
        snapshot_id=log.snapshot_id,
    )
    resumed = log.poll(floor_cursor, limit=10)
    assert [int(event["body"]["n"]) for event in resumed.events] == [3, 4]

    # Cursor older than the retained window expires.
    expired = EventCursor(
        stream_id="stream:tasks",
        position=1,
        last_event_id="gone",
        snapshot_id=log.snapshot_id,
    )
    with pytest.raises(CursorReplayError):
        log.poll(expired, limit=10)

    ids = log.list_event_ids("stream:tasks")
    assert len(ids) == 2
    live = EventCursor(
        stream_id="stream:tasks",
        position=4,
        last_event_id=ids[0],
        snapshot_id=log.snapshot_id,
    )
    page = log.poll(live, limit=10)
    assert len(page.events) == 1
    assert int(page.events[0]["body"]["n"]) == 4


def test_payload_bounds_and_batch_atomicity(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with pytest.raises(DatabaseEventLogPayloadError):
        log.append_event(
            "stream:tasks",
            "huge",
            {"blob": "x" * (300_000)},
        )

    # Batch is atomic: second invalid entry rolls back the first.
    with pytest.raises(DatabaseEventLogPayloadError):
        log.append_events(
            [
                {
                    "stream_id": "stream:tasks",
                    "event_type": "ok",
                    "body": {"n": 1},
                },
                {
                    "stream_id": "stream:tasks",
                    "event_type": "bad",
                    "body": {"blob": "y" * 300_000},
                },
            ]
        )
    assert log.stream_head("stream:tasks").latest_sequence == 0


def test_context_manager_close(tmp_path: Path) -> None:
    with open_database_event_log(tmp_path / "control.duckdb") as log:
        log.append_event("stream:x", "open", {})
    with pytest.raises(DatabaseEventLogNotOpenError):
        log.append_event("stream:x", "after-close", {})
