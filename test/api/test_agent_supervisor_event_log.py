"""Regression coverage for the JSONL event-log surface (DQP-013 companion).

JSONL remains a durable compatibility and export surface. After DQP-013 the
control-plane authority for events, audit, metrics, and consumer cursors is
``DatabaseEventLog``; these tests prove the existing JSONL helpers still:

* mint immutable event IDs and per-stream sequences
* support gapless cursor polling without duplicate effects
* persist consumer checkpoints for restart resume
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    CursorReplayError,
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    initial_event_cursor,
    latest_event_cursor,
    read_event_cursor_checkpoint,
    read_jsonl_event_page,
    write_event_cursor_checkpoint,
)


def test_jsonl_event_ids_and_sequences_are_immutable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(path, "task.changed", {"task_id": "T-1"})
    second = append_jsonl_event(path, "task.changed", {"task_id": "T-2"})

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert first["event_id"].startswith("sha256:")
    assert second["event_id"] != first["event_id"]
    assert second["previous_event_id"] == first["event_id"]
    assert first["stream_id"] == second["stream_id"]
    assert first["snapshot_id"] == second["snapshot_id"]

    # Re-reading the durable line must preserve identity fields.
    page = read_jsonl_event_page(path, initial_event_cursor(path), limit=10)
    assert [event["event_id"] for event in page.events] == [
        first["event_id"],
        second["event_id"],
    ]
    assert latest_event_cursor(path).position == 2
    assert latest_event_cursor(path).last_event_id == second["event_id"]


def test_jsonl_polling_resumes_without_loss_or_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    checkpoint_path = tmp_path / "consumer.cursor.json"
    written = [
        append_jsonl_event(path, "tick", {"n": index}) for index in range(1, 6)
    ]

    cursor = initial_event_cursor(path)
    first = read_jsonl_event_page(path, cursor, limit=2)
    assert [event["sequence"] for event in first.events] == [1, 2]
    assert first.has_more
    assert first.next_cursor.last_event_id == written[1]["event_id"]

    assert write_event_cursor_checkpoint(checkpoint_path, first.next_cursor)
    # Identical checkpoint is a no-op.
    assert not write_event_cursor_checkpoint(checkpoint_path, first.next_cursor)

    recovered = read_event_cursor_checkpoint(checkpoint_path)
    assert recovered == first.next_cursor

    remaining = read_jsonl_event_page(path, recovered, limit=10)
    assert [event["sequence"] for event in remaining.events] == [3, 4, 5]
    assert not remaining.has_more

    empty = read_jsonl_event_page(path, remaining.next_cursor, limit=10)
    assert list(empty.events) == []
    assert empty.next_cursor == remaining.next_cursor


def test_jsonl_cursor_rejects_foreign_stream_and_bad_anchor(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    event = append_jsonl_event(path, "alpha", {"x": 1})
    foreign = EventCursor.initial("events:other-stream")
    with pytest.raises(CursorReplayError, match="different stream"):
        read_jsonl_event_page(path, foreign)
    forged = EventCursor(
        stream_id=event["stream_id"],
        snapshot_id=event["snapshot_id"],
        position=1,
        last_event_id="sha256:" + ("00" * 32),
    )
    with pytest.raises(CursorReplayError, match="anchor"):
        read_jsonl_event_page(path, forged)


def test_jsonl_export_deletion_does_not_affect_independent_log(
    tmp_path: Path,
) -> None:
    """Deleting a copied JSONL export must not rewrite the authoritative file.

    Under DQP-013 database authority, JSONL is export-only. This regression
    keeps the same non-authority property for the legacy file path: a derived
    copy can be deleted without touching the source log's event IDs.
    """

    source = tmp_path / "events.jsonl"
    export = tmp_path / "events.export.jsonl"
    first = append_jsonl_event(source, "sample", {"n": 1})
    second = append_jsonl_event(source, "sample", {"n": 2})
    export.write_bytes(source.read_bytes())
    assert export.exists()
    export.unlink()
    assert not export.exists()

    page = read_jsonl_event_page(source, initial_event_cursor(source), limit=10)
    assert [event["event_id"] for event in page.events] == [
        first["event_id"],
        second["event_id"],
    ]
    assert [event["sequence"] for event in page.events] == [1, 2]
