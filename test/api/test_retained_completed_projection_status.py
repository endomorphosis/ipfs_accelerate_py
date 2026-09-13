"""Retained local status evidence never substitutes for current task authority."""

import copy
from datetime import datetime, timezone
from types import SimpleNamespace
import time

import pytest

from test.api.test_append_quarantine_recovery import native_append_quarantine


@pytest.fixture
def retained(tmp_path, monkeypatch):
    fixture = native_append_quarantine(
        tmp_path, monkeypatch, attempt_inside_repository=True
    )
    daemon = fixture.daemon
    alias = fixture.request.task_id
    updated = daemon._mark_tasks_completed_in_todo(
        [alias],
        primary_task_id=alias,
        completion_reason="merged_status_repair",
        expected_task_cids={alias: fixture.request.canonical_task_id},
    )
    assert (
        updated["updated"] is True
        and updated["commit_result"]["reason"] == "no_changes"
    )
    daemon._record_event(
        "task_completed",
        {
            "task_id": alias,
            "reason": "task_became_completed",
            "completion_receipt_repair": False,
        },
    )
    daemon._record_event(
        "daemon_pass",
        {
            "active_task_id": "",
            "completed_count": 1,
            "ready_count": 0,
            "selection_idle_reason": "database_pending_merge_reconciliation",
        },
    )
    before = fixture.paths.events.read_bytes()
    todo = daemon._mark_task_completed_in_todo(
        alias, expected_task_cids={alias: fixture.request.canonical_task_id}
    )
    assert todo["reason"] == "already_completed" and todo["updated"] is False
    assert fixture.paths.events.read_bytes() == before
    projection = SimpleNamespace(paths=fixture.paths, binding=fixture.binding)
    events = daemon._iter_merge_lifecycle_events()
    request = daemon.merge_queue.get(fixture.request.request_id)
    bindings = fixture.bridge._append_recovery_event_bindings(
        request, projection, events=events
    )
    assert bindings is not None
    return SimpleNamespace(
        fixture=fixture,
        events=events,
        todo=todo,
        projection=projection,
        source=bindings["source"],
        finish=bindings["finish"],
        started=time.time(),
    )


def observe(row):
    return row.fixture.bridge._retained_append_existing_completion_status(
        row.events,
        source_event=row.source,
        finish_event=row.finish,
        todo=row.todo,
        projection=row.projection,
        settlement_started_at=row.started,
    )


def test_ignored_local_completion_is_retained_without_event_or_task_write(retained):
    before = retained.fixture.paths.events.read_bytes()
    selected = observe(retained)
    assert selected is not None and selected[1]["type"] == "todo_status_updated"
    assert retained.fixture.paths.events.read_bytes() == before
    assert retained.fixture.record.status == "blocked"
    assert (
        retained.fixture.daemon.merge_queue.get(
            retained.fixture.request.request_id
        ).status
        == "quarantined"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_status",
        "duplicate_status",
        "reconciled_status",
        "member_identity",
        "projection_path",
        "changed_commit",
        "updated_integer",
        "committed_integer",
        "future_status",
        "naive_status",
        "unknown_reason",
        "duplicate_completion",
        "future_completion",
        "completion_repair_integer",
        "foreign_namespace",
        "inserted_status",
        "changed_settlement_todo",
    ],
)
def test_changed_or_ambiguous_retained_observation_is_rejected(retained, fault):
    row = retained
    row.events = copy.deepcopy(row.events)
    status = next(e for e in row.events if e["type"] == "todo_status_updated")
    completion = next(e for e in row.events if e["type"] == "task_completed")
    if fault == "missing_status":
        row.events.remove(status)
    elif fault == "duplicate_status":
        row.events.append(copy.deepcopy(status))
    elif fault == "reconciled_status":
        status["type"] = "todo_status_reconciled"
    elif fault == "member_identity":
        status["completion_receipts"][0]["canonical_task_cid"] = "foreign"
    elif fault == "projection_path":
        status["path"] += ".foreign"
    elif fault == "changed_commit":
        status["commit_result"]["reason"] = "committed"
    elif fault == "updated_integer":
        status["updated"] = 1
    elif fault == "committed_integer":
        status["commit_result"]["committed"] = 0
    elif fault == "future_status":
        status["timestamp"] = datetime.fromtimestamp(
            row.started + 60, timezone.utc
        ).isoformat()
    elif fault == "naive_status":
        status["timestamp"] = datetime.now().isoformat()
    elif fault == "unknown_reason":
        status["completion_reason"] = "manual_override"
    elif fault == "duplicate_completion":
        row.events.append(copy.deepcopy(completion))
    elif fault == "future_completion":
        completion["timestamp"] = datetime.fromtimestamp(
            row.started + 60, timezone.utc
        ).isoformat()
    elif fault == "completion_repair_integer":
        completion["completion_receipt_repair"] = 0
    elif fault == "foreign_namespace":
        status["board_namespace"] = "foreign"
    elif fault == "inserted_status":
        status["inserted_status_task_ids"] = [status["task_id"]]
    elif fault == "changed_settlement_todo":
        row.todo["reason"] = "updated"
    before = row.fixture.paths.events.read_bytes()
    assert observe(row) is None
    assert row.fixture.paths.events.read_bytes() == before
    assert row.fixture.record.status == "blocked"
