"""Tests for DuckDB-authoritative event/audit/log/metric store (DQP-013).

Acceptance:

* Event IDs and per-stream sequences are immutable
* Polling resumes without loss or duplicate effects
* Application audit is explicit rather than inferred from Quack diagnostics
* Recursive logging is bounded
* Exported JSONL deletion has no authority effect
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CursorReplayError,
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    APPLICATION_AUDIT_COMPONENT,
    APPLICATION_AUDIT_EVENT_TYPE,
    CONSUMER_CHECKPOINT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    MAX_RECURSIVE_LOG_DEPTH,
    ApplicationAuditRecord,
    ConsumerCheckpoint,
    DatabaseEventLog,
    DatabaseEventLogAuthorityError,
    DatabaseEventLogBoundsError,
    DatabaseEventLogImmutabilityError,
    DatabaseEventLogNotOpenError,
    LogSeverity,
    open_database_event_log,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    StateAuthorityClass,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database event-log hermetic tests",
)


def _open_log(tmp_path: Path, **kwargs) -> DatabaseEventLog:
    return open_database_event_log(tmp_path / "control.duckdb", **kwargs)


def test_interface_identities() -> None:
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_CURSOR_INTERFACE == "EventCursor@1"
    assert CONSUMER_CHECKPOINT_INTERFACE == "ConsumerCheckpoint@1"
    assert DatabaseEventLog.INTERFACE == DATABASE_EVENT_LOG_INTERFACE
    assert ConsumerCheckpoint.INTERFACE == CONSUMER_CHECKPOINT_INTERFACE


def test_event_ids_and_sequences_are_immutable(tmp_path: Path) -> None:
    with _open_log(tmp_path) as store:
        first = store.append_event("task.changed", {"task_id": "T-1"})
        second = store.append_event("task.changed", {"task_id": "T-2"})
        assert first["sequence"] == 1
        assert second["sequence"] == 2
        assert first["event_id"] != second["event_id"]
        assert first["event_id"].startswith("sha256:")
        assert second["previous_event_id"] == first["event_id"]

        head = store.stream_head()
        assert head.latest_sequence == 2
        assert head.last_event_id == second["event_id"]

        # Direct mutation of durable rows must not be offered by the API; prove
        # the stored identity is still the content-addressed envelope.
        with open_duckdb_connection(store.database_path) as connection:
            row = connection.execute(
                "SELECT event_id, sequence, body_json FROM domain_events "
                "WHERE sequence = 1 LIMIT 1"
            ).fetchone()
        body = json.loads(str(row[2]))
        assert body["event_id"] == first["event_id"]
        assert body["sequence"] == 1

        # Forcing a conflicting identity at an occupied sequence fails closed.
        with open_duckdb_connection(store.database_path) as connection:
            with pytest.raises(Exception):
                connection.execute(
                    """
                    INSERT INTO domain_events (
                        event_id, stream_id, sequence, global_sequence,
                        event_type, task_cid, attempt_id, session_id,
                        recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, '', '', '', ?, ?)
                    """,
                    [
                        "forged:event",
                        first["stream_id"],
                        1,
                        99,
                        "forged",
                        "1970-01-01T00:00:00Z",
                        "{}",
                    ],
                )


def test_polling_resumes_without_loss_or_duplicate_effects(tmp_path: Path) -> None:
    with _open_log(tmp_path) as store:
        written = [
            store.append_event("tick", {"n": index})
            for index in range(1, 6)
        ]
        cursor = store.initial_cursor()
        first = store.poll(cursor, limit=2)
        assert [event["sequence"] for event in first.events] == [1, 2]
        assert first.has_more
        assert first.next_cursor.position == 2
        assert first.next_cursor.last_event_id == written[1]["event_id"]

        checkpoint = store.save_checkpoint("consumer:worker-a", first.next_cursor)
        assert checkpoint.checkpoint_id
        recovered = store.load_checkpoint("consumer:worker-a")
        assert recovered is not None
        assert recovered.cursor == first.next_cursor

        # Simulate restart: reopen store, load checkpoint, continue.
        store.close()

    with DatabaseEventLog(
        tmp_path / "control.duckdb", install_schema=False
    ) as restarted:
        loaded = restarted.load_checkpoint("consumer:worker-a")
        assert loaded is not None
        remaining = restarted.poll(loaded.cursor, limit=10)
        assert [event["sequence"] for event in remaining.events] == [3, 4, 5]
        assert not remaining.has_more

        # Replaying from the final cursor yields no events (no duplicate effects).
        empty = restarted.poll(remaining.next_cursor, limit=10)
        assert list(empty.events) == []
        assert empty.next_cursor == remaining.next_cursor

        # Checkpoint no-op on identical cursor still round-trips.
        again = restarted.save_checkpoint("consumer:worker-a", remaining.next_cursor)
        assert again.cursor == remaining.next_cursor


def test_cursor_rejects_foreign_stream_and_anchor_mismatch(tmp_path: Path) -> None:
    with _open_log(tmp_path) as store:
        event = store.append_event("alpha", {"x": 1})
        foreign = EventCursor.initial("stream:other")
        with pytest.raises(CursorReplayError, match="different stream"):
            store.poll(foreign)
        forged = EventCursor(
            stream_id=event["stream_id"],
            snapshot_id=event["snapshot_id"],
            position=1,
            last_event_id="sha256:" + ("00" * 32),
        )
        with pytest.raises(CursorReplayError, match="anchor"):
            store.poll(forged)


def test_application_audit_is_explicit_not_quack_diagnostics(
    tmp_path: Path,
) -> None:
    with _open_log(tmp_path) as store:
        # Seed Quack-style diagnostics that must never become audit authority.
        with open_duckdb_connection(store.database_path) as connection:
            connection.execute(
                """
                INSERT INTO quack_query_telemetry (
                    sample_id, session_id, server_id, observed_at,
                    latency_us, row_count, status, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "telemetry:1",
                    "session:q",
                    "server:1",
                    "1970-01-01T00:00:00Z",
                    12,
                    3,
                    "ok",
                    json.dumps({"sql": "SELECT 1"}),
                ],
            )

        assert store.list_audit() == ()
        assert store.quack_diagnostics_are_not_audit() is True

        audit = store.append_audit(
            "task.claimed",
            actor_id="daemon:impl",
            subject_id="task:T-9",
            task_cid="task:T-9",
            body={"reason": "lease_granted"},
        )
        assert isinstance(audit, ApplicationAuditRecord)
        assert audit.action == "task.claimed"
        assert audit.event_id.startswith("sha256:")
        listed = store.list_audit()
        assert len(listed) == 1
        assert listed[0].audit_id == audit.audit_id
        assert listed[0].to_dict()["source"] == "application_explicit"

        # Audit projects as an explicit domain event, not as telemetry.
        page = store.poll(store.initial_cursor(), limit=10)
        assert any(
            event.get("type") == APPLICATION_AUDIT_EVENT_TYPE
            for event in page.events
        )
        with open_duckdb_connection(store.database_path) as connection:
            logs = connection.execute(
                "SELECT component FROM structured_logs WHERE component = ?",
                [APPLICATION_AUDIT_COMPONENT],
            ).fetchall()
            telemetry = connection.execute(
                "SELECT COUNT(*) FROM quack_query_telemetry"
            ).fetchone()
        assert logs
        assert int(telemetry[0]) == 1
        # Telemetry row count does not inflate application audit.
        assert len(store.list_audit()) == 1


def test_recursive_logging_is_bounded(tmp_path: Path) -> None:
    import ipfs_accelerate_py.agent_supervisor.runtime.database_event_log as mod

    with _open_log(tmp_path, max_recursive_log_depth=MAX_RECURSIVE_LOG_DEPTH) as store:
        depth_hits: list[int | None] = []

        def nested_logger(level: int) -> None:
            # Hold an outer frame so append_log sees re-entrant depth, matching
            # a log handler that logs about logging.
            outer = mod._enter_logging()
            try:
                assert outer == level
                result = store.append_log(
                    f"level-{level}",
                    severity=LogSeverity.DEBUG,
                    component="recursive-test",
                    body={"level": level},
                )
                depth_hits.append(None if result is None else level)
                if result is not None and level < 5:
                    nested_logger(level + 1)
            finally:
                mod._exit_logging()

        nested_logger(1)
        # Outer frame depth D means append_log sees D+1. With max=2, only the
        # first call (outer=1, inner=2) is accepted; outer=2 yields inner=3 drop.
        accepted = [item for item in depth_hits if item is not None]
        dropped = [item for item in depth_hits if item is None]
        assert accepted == [1]
        assert dropped == [None]

        with open_duckdb_connection(store.database_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM structured_logs WHERE component = ?",
                ["recursive-test"],
            ).fetchone()
        assert int(count[0]) == 1

        # Explicit audit re-entry beyond the bound fails closed.
        mod._tls.depth = MAX_RECURSIVE_LOG_DEPTH
        try:
            with pytest.raises(DatabaseEventLogBoundsError, match="recursive"):
                store.append_audit("should-fail", subject_id="x")
        finally:
            mod._tls.depth = 0


def test_exported_jsonl_deletion_has_no_authority_effect(tmp_path: Path) -> None:
    with _open_log(tmp_path) as store:
        for index in range(3):
            store.append_event("export.sample", {"n": index})
        export_path = tmp_path / "events.export.jsonl"
        receipt = store.export_jsonl(export_path)
        assert receipt.authority_class is StateAuthorityClass.EXPORT
        assert receipt.to_dict()["authoritative"] is False
        assert receipt.to_dict()["is_export_only"] is True
        assert export_path.is_file()
        lines = export_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

        proof = store.delete_export_has_no_authority_effect(export_path)
        assert not export_path.exists()
        assert proof["authoritative_effect"] is False
        assert proof["latest_sequence"] == 3
        assert proof["event_count"] == 3

        # Polling still returns the full authoritative population.
        page = store.poll(store.initial_cursor(), limit=10)
        assert [event["sequence"] for event in page.events] == [1, 2, 3]


def test_metrics_traces_integrity_and_retention(tmp_path: Path) -> None:
    with _open_log(tmp_path) as store:
        sample = store.record_metric(
            "tasks.completed",
            1_000,
            unit="count",
            labels={"lane": "1"},
        )
        assert sample["metric_name"] == "tasks.completed"
        assert sample["sample_id"]
        # Second sample with same labels/time is idempotent by content id.
        again = store.record_metric(
            "tasks.completed",
            1_000,
            unit="count",
            labels={"lane": "1"},
            observed_at=sample["observed_at"],
        )
        assert again["sample_id"] == sample["sample_id"]

        trace = store.append_trace(
            "phase-enter",
            component="daemon",
            trace_id="trace:1",
            span_id="span:1",
            body={"phase": "implement"},
        )
        assert trace is not None
        assert trace["severity"] == "trace"

        for index in range(5):
            store.append_event("bulk", {"n": index})
        integrity = store.write_integrity_checkpoint()
        assert integrity.population_digest.startswith("sha256:")
        assert integrity.latest_sequence == 5

        retained = store.apply_retention(retain_recent=2)
        assert retained["deleted"] == 3
        assert retained["earliest_sequence"] == 4
        head = store.stream_head()
        assert head.earliest_sequence == 4
        assert head.latest_sequence == 5

        # Cursor older than retained window expires rather than silently skipping.
        stale = EventCursor(
            stream_id=head.stream_id,
            snapshot_id=head.snapshot_id,
            position=1,
            last_event_id="sha256:" + ("ab" * 32),
        )
        with pytest.raises(CursorReplayError):
            store.poll(stale)

        page = store.poll(store.initial_cursor(), limit=10)
        assert [event["sequence"] for event in page.events] == [4, 5]


def test_closed_store_and_multi_stream(tmp_path: Path) -> None:
    store = DatabaseEventLog(tmp_path / "control.duckdb")
    with pytest.raises(DatabaseEventLogNotOpenError):
        store.append_event("x", {})
    store.open()
    a = store.append_event("a", {"k": 1}, stream_id="stream:a")
    b = store.append_event("b", {"k": 1}, stream_id="stream:b")
    assert a["sequence"] == 1
    assert b["sequence"] == 1
    assert a["stream_id"] != b["stream_id"]
    page_a = store.poll(store.initial_cursor("stream:a"), stream_id="stream:a")
    page_b = store.poll(store.initial_cursor("stream:b"), stream_id="stream:b")
    assert len(page_a.events) == 1
    assert len(page_b.events) == 1
    store.close()
    with pytest.raises(DatabaseEventLogNotOpenError):
        store.poll(store.initial_cursor())


def test_consumer_checkpoint_identity_and_export_receipt_authority(
    tmp_path: Path,
) -> None:
    with _open_log(tmp_path) as store:
        event = store.append_event("once", {"v": 1})
        cursor = store.latest_cursor()
        checkpoint = ConsumerCheckpoint(
            consumer_id="consumer:identity",
            cursor=cursor,
        )
        assert checkpoint.to_dict()["interface"] == CONSUMER_CHECKPOINT_INTERFACE
        with pytest.raises(DatabaseEventLogImmutabilityError):
            ConsumerCheckpoint(
                consumer_id="consumer:identity",
                cursor=cursor,
                checkpoint_id="forged",
            )
        receipt = store.export_jsonl(tmp_path / "out.jsonl")
        with pytest.raises(DatabaseEventLogAuthorityError):
            type(receipt)(
                export_id=receipt.export_id,
                stream_id=receipt.stream_id,
                destination=receipt.destination,
                artifact_digest=receipt.artifact_digest,
                event_count=receipt.event_count,
                earliest_sequence=receipt.earliest_sequence,
                latest_sequence=receipt.latest_sequence,
                authority_class=StateAuthorityClass.AUTHORITATIVE,
                recorded_at=receipt.recorded_at,
            )
        assert store.to_dict()["authority_class"] == (
            StateAuthorityClass.AUTHORITATIVE.value
        )
        assert store.to_dict()["jsonl_authority_class"] == (
            StateAuthorityClass.EXPORT.value
        )
        assert event["sequence"] == 1
