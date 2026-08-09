"""Tests for DatabaseEventLog@1 (DQP-013).

Evidence subset: monotonic sequences, duplicate IDs, cursor expiry, bounded
polling, coalescing, replay, redaction, retention, event/projection
transaction. Acceptance also covers explicit audit, bounded recursive
logging, and JSONL export non-authority.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CursorReplayError,
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    CONSUMER_CHECKPOINT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    REDACTION_MARKER,
    AuditAction,
    ConsumerCheckpoint,
    DatabaseEventLog,
    DatabaseEventLogBoundsError,
    DatabaseEventLogConflictError,
    duckdb_available,
    open_database_event_log,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseEventLog hermetic tests",
)


def _open(tmp_path: Path) -> DatabaseEventLog:
    return open_database_event_log(tmp_path / "events.duckdb")


def test_interface_identities() -> None:
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert DatabaseEventLog.INTERFACE == DATABASE_EVENT_LOG_INTERFACE


def test_monotonic_sequences_and_immutable_ids(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        first = log.append_event("task.started", {"task_cid": "task:1"})
        second = log.append_event("task.finished", {"task_cid": "task:1"})
        assert first.sequence == 1
        assert second.sequence == 2
        assert second.global_sequence == 2
        assert second.previous_event_id == first.event_id
        assert first.event_id != second.event_id

        head = log.stream_head()
        assert head.latest_sequence == 2
        assert head.last_event_id == second.event_id

        # Re-fetch proves immutability of identity and sequence.
        loaded = log.get_event(first.event_id)
        assert loaded is not None
        assert loaded.sequence == 1
        assert loaded.event_id == first.event_id


def test_duplicate_event_id_coalesces(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        event = log.append_event("wake", {"ordinal": 1})
        again = log.append_event(
            "wake",
            {"ordinal": 1},
            event_id=event.event_id,
            recorded_at=event.recorded_at,
        )
        # Exact identity replay is coalesced; head does not advance twice.
        assert again.event_id == event.event_id
        assert again.sequence == event.sequence
        assert log.stream_head().latest_sequence == 1

        with pytest.raises(DatabaseEventLogConflictError):
            log.append_event("wake", {"ordinal": 2}, event_id=event.event_id)


def test_bounded_polling_replay_and_consumer_checkpoint(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        written = [
            log.append_event("runtime_wake", {"ordinal": index})
            for index in range(1, 7)
        ]
        cursor = log.initial_cursor()
        first = log.poll(cursor, limit=3)
        assert [item["ordinal"] for item in first.events] == [1, 2, 3]
        assert [item["sequence"] for item in first.events] == [1, 2, 3]
        assert first.has_more is True
        assert first.next_cursor.position == 3
        assert first.next_cursor.last_event_id == written[2].event_id

        checkpoint = log.save_consumer_checkpoint(
            "consumer:worker-a", first.next_cursor
        )
        assert isinstance(checkpoint, ConsumerCheckpoint)
        assert checkpoint.cursor.position == 3

        recovered = log.load_consumer_checkpoint("consumer:worker-a")
        assert recovered is not None
        assert recovered.cursor == first.next_cursor

        remaining = log.poll(recovered.cursor, limit=2)
        assert [item["ordinal"] for item in remaining.events] == [4, 5]
        assert remaining.has_more is True

        tail = log.poll(remaining.next_cursor, limit=10)
        assert [item["ordinal"] for item in tail.events] == [6]
        assert tail.has_more is False

        # Exactly-once: replaying from the head yields nothing.
        empty = log.poll(tail.next_cursor, limit=10)
        assert list(empty.events) == []
        assert empty.next_cursor == tail.next_cursor


def test_cursor_expiry_after_retention(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 6):
            log.append_event("tick", {"n": index})
        mid = log.poll(log.initial_cursor(), limit=2).next_cursor
        assert mid.position == 2

        result = log.apply_retention(retain_recent=2)
        assert result["earliest_sequence"] == 4
        assert result["latest_sequence"] == 5

        with pytest.raises(CursorReplayError):
            log.poll(mid, limit=10)

        # Head remains monotonic and readable from a fresh cursor.
        page = log.poll(log.initial_cursor(), limit=10)
        assert [item["n"] for item in page.events] == [4, 5]


def test_redaction_on_append(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        event = log.append_event(
            "provider.call",
            {
                "model": "test",
                "access_token": "super-secret-token-value",
                "nested": {"password": "also-secret"},
            },
        )
        assert event.redacted is True
        assert event.body["access_token"] == REDACTION_MARKER
        assert event.body["nested"]["password"] == REDACTION_MARKER
        assert event.body["model"] == "test"


def test_explicit_audit_and_bounded_nesting(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        record = log.append_audit(
            AuditAction.APPEND,
            actor_id="operator:alice",
            subject_kind="task",
            subject_id="task:42",
            body={"reason": "manual"},
        )
        assert record["action"] == "append"
        assert record["event_id"]
        audits = log.list_audits(subject_id="task:42")
        assert len(audits) == 1
        assert audits[0]["actor_id"] == "operator:alice"

        # Nested audits from export are bounded; depth never explodes.
        for index in range(3):
            log.append_audit(
                AuditAction.CHECKPOINT,
                actor_id="system",
                subject_kind="stream",
                subject_id=f"stream:{index}",
                body={"index": index},
                emit_event=False,
            )
        assert len(log.list_audits()) >= 1


def test_structured_logs_and_metrics(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        entry = log.append_log(
            "lane ready",
            severity="info",
            component="scheduler",
            body={"lane": "dqp-events"},
        )
        assert entry["log_id"]
        sample = log.append_metric_sample(
            "provider_calls",
            1_500,
            unit="count",
            labels={"provider": "test"},
        )
        assert sample["value_milli"] == 1_500
        assert sample["metric_name"] == "provider_calls"


def test_integrity_checkpoint_and_event_projection_transaction(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as log:
        log.append_event("a", {"x": 1})
        log.append_event("b", {"x": 2})
        checkpoint = log.write_integrity_checkpoint()
        assert checkpoint.event_count == 2
        assert checkpoint.chain_digest.startswith("sha256:")
        assert log.verify_integrity_checkpoint(checkpoint) is True

        # Append advances the chain; old checkpoint fails closed.
        log.append_event("c", {"x": 3})
        with pytest.raises(Exception):
            log.verify_integrity_checkpoint(checkpoint)


def test_jsonl_export_is_non_authoritative(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        for index in range(1, 4):
            log.append_event("exportable", {"n": index})
        export_path = tmp_path / "events.export.jsonl"
        receipt = log.export_jsonl(export_path)
        assert receipt.authority == "export_only"
        assert receipt.event_count == 3
        assert export_path.exists()
        assert receipt.content_digest.startswith("sha256:")

        head_before = log.stream_head().to_dict()
        assert log.authority_unaffected_by_export_deletion(export_path) is True
        assert not export_path.exists()
        assert log.stream_head().to_dict() == head_before
        # Events remain queryable after export deletion.
        events = list(log.replay())
        assert len(events) == 3


def test_foreign_cursor_and_token_round_trip(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        log.append_event("one", {"v": 1})
        page = log.poll(log.initial_cursor(), limit=1)
        token = page.next_cursor.to_token()
        decoded = EventCursor.from_token(token)
        again = log.poll(decoded, limit=1)
        assert list(again.events) == []

        foreign = EventCursor.initial("stream:other", snapshot_id=log.snapshot_id)
        with pytest.raises(CursorReplayError):
            log.poll(foreign, limit=1)


def test_page_limit_bound(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        log.append_event("x", {"n": 1})
        with pytest.raises(DatabaseEventLogBoundsError):
            log.poll(log.initial_cursor(), limit=0)
        with pytest.raises(DatabaseEventLogBoundsError):
            log.poll(log.initial_cursor(), limit=10_000)


def test_multi_stream_independent_sequences(tmp_path: Path) -> None:
    with _open(tmp_path) as log:
        a1 = log.append_event("a", {"n": 1}, stream_id="stream:a")
        b1 = log.append_event("b", {"n": 1}, stream_id="stream:b")
        a2 = log.append_event("a", {"n": 2}, stream_id="stream:a")
        assert a1.sequence == 1
        assert a2.sequence == 2
        assert b1.sequence == 1
        assert a2.global_sequence == 3
        assert set(log.list_streams()) == {"stream:a", "stream:b"}
        assert log.stream_head("stream:a").latest_sequence == 2
        assert log.stream_head("stream:b").latest_sequence == 1
