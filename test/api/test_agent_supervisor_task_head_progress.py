"""Resume head qualification replays events without granting task authority."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import INTENT_EVENT_SCHEMA
from ipfs_accelerate_py.agent_supervisor.task_sources.task_head_progress import (
    TaskHeadProgressError, verify_task_head_progress,
)


def _event(offset, before, after, revision, *, completed=False):
    kind = "intent.completion_recorded" if completed else "intent.task_status_changed"
    payload = {
        "task_cid": "task:a", "task_alias": "A", "goal_cid": "goal:a",
        "previous_status": before, "status": after, "revision": revision,
    }
    if completed:
        payload.update(completion_receipt_cid="receipt:a", evidence_digest="evidence:a")
    event = {
        "stream_id": "stream:intent", "sequence": 30 + offset,
        "global_sequence": 342 + offset, "event_type": kind,
        "task_cid": "task:a", "recorded_at": "2026-09-09T16:00:00Z",
        "body": {
            "schema": INTENT_EVENT_SCHEMA, "event_type": kind,
            "subject_id": "task:a", "body": payload,
            "recorded_at": "2026-09-09T16:00:00Z", "owner_id": "test-owner",
        },
    }
    _rehash(event)
    return event


def _rehash(event):
    event["event_id"] = content_identity({
        key: event[key] for key in ("stream_id", "sequence", "global_sequence", "event_type", "body")
    })


def _case():
    head = dict(task_cid="task:a", task_alias="A", goal_cid="goal:a", status="blocked", revision=24)
    return dict(
        anchor_heads=[head], observed_heads=[{**head, "status": "completed", "revision": 27}],
        anchor_cursor=342, anchor_stream_sequence=30, observed_cursor=345,
        events=[
            _event(1, "blocked", "retrying", 25),
            _event(2, "retrying", "in_progress", 26),
            _event(3, "in_progress", "completed", 27, completed=True),
        ],
    )


def test_bootstrap_anchor_allows_only_replayed_progress_without_mutating_inputs():
    case = _case()
    original = deepcopy(case)
    result = verify_task_head_progress(**case)
    assert case == original
    assert result["anchor_cursor"] == 342
    assert result["observed_cursor"] == 345
    assert result["launch_authority"] is False
    assert result["task_completion_authority"] is False
    assert result["progress_cid"]


def test_exact_anchor_is_valid_without_events():
    case = _case()
    case.update(observed_heads=deepcopy(case["anchor_heads"]), observed_cursor=342, events=[])
    assert verify_task_head_progress(**case)["event_ids"] == []


@pytest.mark.parametrize("change", [
    "gap", "duplicate", "truncated", "ahead", "head", "alias", "goal",
    "event_hash", "previous_status", "revision", "boolean_revision", "stream",
    "definition_event", "task_index", "foreign_subject", "completion_binding",
    "duplicate_head", "empty_heads", "stream_gap", "boolean_cursor",
])
def test_resume_rejects_unproven_or_foreign_head_changes(change):
    case = _case()
    event = case["events"][1]
    payload = event["body"]["body"]
    if change == "gap": case["events"].pop(0)
    elif change == "duplicate": case["events"][1] = deepcopy(case["events"][0])
    elif change == "truncated": case["events"].pop()
    elif change == "ahead": case["observed_cursor"] = 341
    elif change == "head": case["observed_heads"][0]["revision"] = 28
    elif change == "alias": payload["task_alias"] = "B"
    elif change == "goal": payload["goal_cid"] = "goal:b"
    elif change == "event_hash": event["event_id"] = "forged"
    elif change == "previous_status": payload["previous_status"] = "completed"
    elif change == "revision": payload["revision"] = 28
    elif change == "boolean_revision": payload["revision"] = True
    elif change == "stream": event["stream_id"] = "stream:foreign"
    elif change == "definition_event":
        event["event_type"] = event["body"]["event_type"] = "intent.task_acceptance_set"
    elif change == "task_index": event["task_cid"] = "task:b"
    elif change == "foreign_subject": event["body"]["subject_id"] = "task:b"
    elif change == "completion_binding":
        event = case["events"][-1]
        del event["body"]["body"]["completion_receipt_cid"]
    elif change == "duplicate_head": case["anchor_heads"] *= 2
    elif change == "empty_heads": case["observed_heads"] = []
    elif change == "stream_gap": event["sequence"] += 1
    elif change == "boolean_cursor": case["observed_cursor"] = True
    if change != "event_hash": _rehash(event)
    with pytest.raises(TaskHeadProgressError):
        verify_task_head_progress(**case)


def test_validation_event_does_not_advance_task_revision():
    case = _case()
    event = case["events"][-1]
    event["event_type"] = event["body"]["event_type"] = "intent.validation_recorded"
    event["body"]["subject_id"] = "validation:a"
    event["body"]["body"] = {"task_cid": "task:a", "outcome": "passed", "result_id": "validation:a"}
    _rehash(event)
    case["observed_heads"][0].update(status="in_progress", revision=26)
    assert verify_task_head_progress(**case)["observed_cursor"] == 345


def test_native_repository_status_and_evidence_events_replay(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import open_intent_repository

    with open_intent_repository(tmp_path / "test-control.duckdb") as repo:
        repo.upsert_goal(goal_cid="goal:a", goal_alias="G", title="Test goal")
        repo.upsert_task(task_cid="task:a", task_alias="A", goal_cid="goal:a")
        keys = ("task_cid", "task_alias", "goal_cid", "status", "revision")
        anchor = {key: repo.get_task("task:a")[key] for key in keys}
        cursor = repo.event_watermark()
        sequence = repo.list_events(limit=512)[-1]["sequence"]
        repo.cas_task_status(task_cid="task:a", expected_revision=anchor["revision"], new_status="in_progress")
        repo.record_validation_result(task_cid="task:a", outcome="passed", evidence_digest="sha256:" + "ab" * 32)
        repo.record_evidence(task_cid="task:a", evidence_kind="validation", digest="sha256:" + "cd" * 32)
        observed = {key: repo.get_task("task:a")[key] for key in keys}
        before = repo.snapshot().projection_cid
        result = verify_task_head_progress(
            anchor_heads=[anchor], observed_heads=[observed],
            anchor_cursor=cursor, anchor_stream_sequence=sequence,
            observed_cursor=repo.event_watermark(),
            events=repo.list_events(after_global_sequence=cursor, limit=512),
        )
        assert len(result["event_ids"]) == 3
        assert repo.snapshot().projection_cid == before
