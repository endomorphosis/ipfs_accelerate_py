from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
    TaskboardMaterializationEntry,
    TaskboardStore,
    commit_taskboard_materialization,
    preview_taskboard_materialization,
)
from test.api.test_agent_supervisor_prompt_plan_admission import _admit


def _materialized_store(tmp_path: Path) -> TaskboardStore:
    source = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="AUTO",
    )
    source.materialize(_admit()[-1])
    return source.store


def test_snapshot_query_get_and_ready_set_are_bounded(tmp_path: Path) -> None:
    store = _materialized_store(tmp_path)
    snapshot = store.snapshot()
    task = snapshot.tasks[0]

    assert snapshot.task_count == 1
    assert snapshot.byte_count == store.path.stat().st_size
    assert store.query(status="todo", limit=1) == (task,)
    assert store.query(task_cids=(task.task_cid,), limit=1) == (task,)
    assert store.get(task.task_id) == task
    assert store.get(task.task_cid) == task
    assert store.get("missing") is None
    assert store.ready_set(limit=1) == (task,)

    with pytest.raises(ValueError, match="limit"):
        store.query(limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.ready_set(limit=1025)
    with pytest.raises(ValueError, match="offset"):
        store.query(offset=-1)


def test_status_cas_is_revision_fenced_and_emits_replayable_event(
    tmp_path: Path,
) -> None:
    store = _materialized_store(tmp_path)
    initial = store.snapshot()
    task = initial.tasks[0]
    cursor = store.event_cursor()

    no_op = store.compare_and_swap_status(
        task.task_id,
        expected_status="todo",
        new_status="todo",
        expected_revision=initial.revision,
    )
    assert not no_op.changed
    assert no_op.board_revision == initial.revision

    changed = store.compare_and_swap_status(
        task.task_cid,
        expected_status=("todo", "ready"),
        new_status="completed",
        expected_revision=initial.revision,
        event_payload={"reason": "validated"},
    )
    assert changed.changed
    assert changed.task.status == "completed"
    assert changed.task.task_cid == task.task_cid
    assert changed.board_revision != initial.revision
    assert changed.event["reason"] == "validated"

    page = store.events(cursor, limit=1)
    assert len(page.events) == 1
    assert page.events[0]["event_id"] == changed.event["event_id"]
    assert page.events[0]["previous_status"] == "todo"
    assert page.events[0]["status"] == "completed"

    with pytest.raises(ValueError, match="stale taskboard revision"):
        store.compare_and_swap_status(
            task.task_id,
            expected_status="todo",
            new_status="blocked",
            expected_revision=initial.revision,
        )


def test_integrity_and_configured_paths_fail_closed(tmp_path: Path) -> None:
    store = _materialized_store(tmp_path)
    report = store.check_integrity().require_valid()

    assert report.task_count == 1
    assert report.board_revision == store.snapshot().revision

    store.path.write_bytes(b"\xff")
    corrupt = store.check_integrity()
    assert not corrupt.valid
    assert corrupt.reason_codes
    with pytest.raises(ValueError, match="integrity failed"):
        corrupt.require_valid()

    with pytest.raises(ValueError, match="escapes"):
        TaskboardStore(tmp_path.parent / "escape.md", root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        TaskboardStore(
            tmp_path / "tasks.md",
            root=tmp_path,
            journal_path=tmp_path.parent / "escape.json",
        )


def test_materialization_journal_replay_is_an_exact_no_op(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    journal = tmp_path / "tasks.materialization.json"
    entry = TaskboardMaterializationEntry(
        task_id="AUTO-001",
        goal_id="GOAL-001",
        rendered_block=(
            "## AUTO-001 Demonstrate journal replay\n\n"
            "- Status: todo\n"
            "- Goal id: GOAL-001"
        ),
    )
    preview = preview_taskboard_materialization("", (entry,))

    first = commit_taskboard_materialization(
        board,
        journal,
        preview,
        epoch_id="epoch-1",
        expected_board_revision=preview.base_board_revision,
    )
    replay = commit_taskboard_materialization(
        board,
        journal,
        preview,
        epoch_id="epoch-1",
        expected_board_revision=preview.base_board_revision,
    )

    assert first.committed
    assert first.changed
    assert first.board_write_count == 1
    assert replay.committed
    assert replay.resumed
    assert not replay.changed
    assert replay.write_count == 0
    assert board.read_text(encoding="utf-8").count("## AUTO-001 ") == 1
