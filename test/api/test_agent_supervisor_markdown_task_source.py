from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import taskboard_store
from ipfs_accelerate_py.agent_supervisor.markdown_task_source import (
    MARKDOWN_TASK_SOURCE_SCHEMA,
    MarkdownTaskSource,
    MarkdownTaskSourceConflict,
    MarkdownTaskSourceError,
    MarkdownTaskSourceIntegrityError,
    parse_markdown_task_source,
    project_admitted_plan,
)
from ipfs_accelerate_py.agent_supervisor.taskboard_store import TaskboardStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)
from test.api.test_agent_supervisor_prompt_plan_admission import _admit


def _admission():
    return _admit()[-1]


def _rewrite_first_marker(text: str, mutate) -> str:
    prefix = "<!-- agent-supervisor-task-source:v1:"
    start = text.index(prefix) + len(prefix)
    end = text.index(" -->", start)
    token = text[start:end]
    padding = "=" * ((4 - len(token) % 4) % 4)
    payload = json.loads(base64.urlsafe_b64decode(token + padding))
    mutate(payload)
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    replacement = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return text[:start] + replacement + text[end:]


def test_projection_is_byte_stable_lossless_and_supervisor_compatible(
    tmp_path: Path,
) -> None:
    admission = _admission()
    first = project_admitted_plan(
        admission,
        task_prefix="AUTO",
        board_namespace="prompt-board",
    )
    second = project_admitted_plan(
        admission,
        task_prefix="AUTO",
        board_namespace="prompt-board",
    )

    assert first == second
    assert first.rendered_text == second.rendered_text
    assert "- Task CID: " in first.rendered_text
    assert f"- Plan root: {admission.plan_root_cid}" in first.rendered_text
    assert f"- Schema: {MARKDOWN_TASK_SOURCE_SCHEMA}" in first.rendered_text
    assert "- Revision: 1" in first.rendered_text
    assert "- Resource class: " in first.rendered_text
    assert "- Conflict policy: " in first.rendered_text
    assert "- Outputs: " in first.rendered_text
    assert "- Validation: " in first.rendered_text
    assert "- Acceptance: " in first.rendered_text

    board = tmp_path / "tasks.md"
    board.write_text(first.rendered_text, encoding="utf-8")
    parsed = parse_markdown_task_source(first.rendered_text, path=board)
    daemon_tasks = parse_task_file(board, task_header_prefix="## AUTO-")

    assert parsed.plan_root == admission.plan_root_cid
    assert parsed.task_cids == first.task_cids
    assert parsed.projection_id == first.projection_id
    assert daemon_tasks[0].task_id == first.task_ids[0]
    assert daemon_tasks[0].metadata["task cid"] == first.task_cids[0]
    marker = parsed.tasks[0].metadata
    task_record = marker["task_record"]
    assert task_record["outputs"]
    assert task_record["validations"]
    assert task_record["acceptance"]
    assert task_record["resource_class"]
    assert marker["goal_records"]


def test_mutable_status_is_not_part_of_task_cid_and_cas_is_revision_fenced(
    tmp_path: Path,
) -> None:
    admission = _admission()
    task = admission.admitted_graph.tasks[0]
    changed_status = replace(
        task,
        status="completed",
        created_at_ms=123,
        updated_at_ms=456,
    )
    assert changed_status.task_cid == task.task_cid

    source = MarkdownTaskSource(tmp_path / "tasks.md", root=tmp_path)
    result = source.materialize(admission)
    snapshot = result.snapshot
    cas = source.compare_and_swap_status(
        snapshot.task_ids[0],
        expected_status="todo",
        new_status="completed",
        expected_revision=snapshot.revision,
    )

    assert cas.changed
    assert cas.task.task_cid == task.task_cid
    assert cas.task.status == "completed"
    with pytest.raises(ValueError, match="stale taskboard revision"):
        source.compare_and_swap_status(
            snapshot.task_ids[0],
            expected_status="todo",
            new_status="blocked",
            expected_revision=snapshot.revision,
        )


def test_materialization_replay_is_a_true_noop_and_store_queries_are_bounded(
    tmp_path: Path,
) -> None:
    admission = _admission()
    source = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="AUTO",
    )

    first = source.materialize(admission)
    replay = source.materialize(admission)
    snapshot = source.snapshot()

    assert first.committed and first.changed
    assert replay.committed and replay.no_op
    assert replay.write_count == 0
    assert source.query(limit=1) == snapshot.tasks[:1]
    assert source.get(snapshot.task_cids[0]) == snapshot.tasks[0]
    assert source.ready_set(limit=1) == snapshot.tasks[:1]
    assert source.check_integrity().valid
    with pytest.raises(ValueError, match="limit"):
        source.query(limit=10_000)


def test_rejected_plan_alias_conflict_path_escape_and_stale_revision_fail_closed(
    tmp_path: Path,
) -> None:
    rejected = _admit(security_decision="deny")[-1]
    with pytest.raises(MarkdownTaskSourceError, match="admitted"):
        project_admitted_plan(rejected)

    admission = _admission()
    task_cid = admission.task_cids[0]
    with pytest.raises(MarkdownTaskSourceError, match="task alias"):
        project_admitted_plan(admission, aliases={task_cid: "bad alias"})
    with pytest.raises(ValueError, match="escapes"):
        TaskboardStore(tmp_path.parent / "escape.md", root=tmp_path)

    source = MarkdownTaskSource(tmp_path / "tasks.md", root=tmp_path)
    with pytest.raises(MarkdownTaskSourceConflict, match="stale"):
        source.materialize(
            admission,
            expected_board_revision="taskboard:sha256:" + ("0" * 64),
        )


def test_partial_render_population_drift_duplicate_and_cycle_are_rejected(
    tmp_path: Path,
) -> None:
    projection = project_admitted_plan(_admission(), task_prefix="AUTO")
    text = projection.rendered_text

    partial = text.replace(" -->", "", 1)
    with pytest.raises(MarkdownTaskSourceIntegrityError, match="partial"):
        parse_markdown_task_source(partial)

    output_drift = text.replace("- Outputs: ", "- Outputs: foreign.py, ", 1)
    with pytest.raises(MarkdownTaskSourceIntegrityError, match="drifted"):
        parse_markdown_task_source(output_drift)

    duplicate = text + "\n" + text
    with pytest.raises(
        (MarkdownTaskSourceIntegrityError, ValueError),
        match="duplicate|population",
    ):
        parse_markdown_task_source(duplicate)

    drifted = _rewrite_first_marker(
        text,
        lambda payload: payload["task_population_cids"].append(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
    )
    with pytest.raises(MarkdownTaskSourceIntegrityError, match="population"):
        parse_markdown_task_source(drifted)

    cycled = _rewrite_first_marker(
        text,
        lambda payload: (
            payload["dependency_aliases"].append(payload["task_alias"]),
            payload["dependency_task_cids"].append(payload["task_cid"]),
        ),
    )
    with pytest.raises(ValueError, match="itself|cycle|drift"):
        parse_markdown_task_source(cycled)

    board = tmp_path / "tasks.md"
    board.write_text(partial, encoding="utf-8")
    assert not TaskboardStore(board, root=tmp_path).check_integrity().valid


def test_existing_population_drift_is_not_appended_or_overwritten(
    tmp_path: Path,
) -> None:
    source = MarkdownTaskSource(tmp_path / "tasks.md", root=tmp_path)
    admission = _admission()
    source.materialize(admission)
    before = source.path.read_bytes()
    changed_projection = source.project(
        admission,
        aliases={admission.task_cids[0]: "OTHER-001"},
    )

    with pytest.raises(MarkdownTaskSourceConflict, match="population|plan root"):
        source.materialize(changed_projection)
    assert source.path.read_bytes() == before


def test_interrupted_existing_journal_recovers_without_duplicate_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="AUTO",
    )
    projection = source.project(_admission())
    original_atomic_write = taskboard_store._atomic_write
    calls = 0

    def interrupt_committed_journal(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected interruption after board publication")
        original_atomic_write(path, payload)

    monkeypatch.setattr(
        taskboard_store,
        "_atomic_write",
        interrupt_committed_journal,
    )
    with pytest.raises(OSError, match="injected interruption"):
        source.materialize(projection)
    monkeypatch.setattr(taskboard_store, "_atomic_write", original_atomic_write)

    restarted = MarkdownTaskSource(
        source.path,
        root=tmp_path,
        journal_path=source.journal_path,
        task_prefix="AUTO",
    )
    recovered = restarted.materialize(projection)
    persisted = source.path.read_text(encoding="utf-8")

    assert recovered.committed
    assert recovered.resumed
    assert recovered.transaction is not None
    assert recovered.transaction.board_write_count == 0
    assert persisted.count(f"## {projection.task_ids[0]} ") == 1
    assert source.snapshot().task_cids == projection.task_cids


def test_events_watch_and_integrity_are_bounded(tmp_path: Path) -> None:
    source = MarkdownTaskSource(tmp_path / "tasks.md", root=tmp_path)
    source.materialize(_admission())
    cursor = source.store.event_cursor()
    event = source.append_event(
        "task_observed",
        {"schema": "test/event@1", "task_id": source.snapshot().task_ids[0]},
    )
    watched = source.watch(
        revision=source.snapshot().revision,
        cursor=cursor,
        timeout=0,
        event_limit=1,
    )

    assert event["sequence"] == 1
    assert watched.changed
    assert len(watched.events) == 1
    assert watched.events[0]["event_id"] == event["event_id"]
    assert watched.snapshot.task_count == 1
