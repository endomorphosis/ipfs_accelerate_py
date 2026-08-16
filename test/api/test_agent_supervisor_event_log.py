"""Regression tests for the legacy JSONL event log (DQP-013 companion).

Keeps the existing file-backed event API stable while DatabaseEventLog becomes
authority. Covers append, monotonic sequences, bounded page replay, and cursor
checkpoints so validation can prove the JSONL surface remains coherent as an
export/compat adapter.
"""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    EventCursor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    initial_event_cursor,
    latest_event_cursor,
    read_event_cursor_checkpoint,
    read_jsonl_event_page,
    read_jsonl_events,
    write_event_cursor_checkpoint,
)


def test_append_is_monotonic_and_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(path, "task.created", {"task_id": "T-1"})
    second = append_jsonl_event(path, "task.queued", {"task_id": "T-1"})

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert first["event_id"].startswith("sha256:")
    assert first["event_id"] != second["event_id"]
    assert first["previous_event_id"] == ""
    assert second["previous_event_id"] == first["event_id"]
    assert first["stream_id"] == second["stream_id"]

    events = read_jsonl_events(path)
    assert len(events) == 2
    assert events[0]["event_id"] == first["event_id"]


def test_bounded_page_replay_resumes_without_loss(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    for index in range(5):
        append_jsonl_event(path, "tick", {"n": index})

    cursor = initial_event_cursor(path)
    first = read_jsonl_event_page(path, cursor, limit=2)
    assert len(first.events) == 2
    assert first.has_more is True
    assert [int(event["n"]) for event in first.events] == [0, 1]

    second = read_jsonl_event_page(path, first.next_cursor, limit=10)
    assert [int(event["n"]) for event in second.events] == [2, 3, 4]
    assert second.has_more is False

    tip = latest_event_cursor(path)
    empty = read_jsonl_event_page(path, tip, limit=10)
    assert empty.events == ()
    assert empty.next_cursor.position == tip.position


def test_cursor_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl_event(path, "started", {"ok": True})
    cursor = latest_event_cursor(path)
    checkpoint_path = tmp_path / "cursor.json"

    assert write_event_cursor_checkpoint(checkpoint_path, cursor) is True
    assert write_event_cursor_checkpoint(checkpoint_path, cursor) is False

    loaded = read_event_cursor_checkpoint(checkpoint_path)
    assert isinstance(loaded, EventCursor)
    assert loaded.position == cursor.position
    assert loaded.last_event_id == cursor.last_event_id
    assert loaded.stream_id == cursor.stream_id
