"""Restart checks advance from real events, never from a mutable pin cache."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import open_intent_repository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_head_progress import TaskHeadProgressError


@pytest.fixture
def program(tmp_path):
    path = tmp_path / "restart.duckdb"
    with open_intent_repository(path) as repo:
        repo.upsert_goal(goal_cid="goal:a", goal_alias="G", title="Goal")
        repo.upsert_plan(plan_cid="plan:a", goal_cid="goal:a", plan_alias="P")
        repo.upsert_task(task_cid="task:a", task_alias="A", goal_cid="goal:a", plan_cid="plan:a")
    source = DatabaseTaskSource(path, install_schema=False, plan_root_cid="plan:a")
    task = source.intent.get_task("task:a")
    anchor = {key: task[key] for key in ("task_cid", "task_alias", "goal_cid", "status", "revision")}
    cursor = source.snapshot().event_cursor
    try:
        yield source, anchor, cursor
    finally:
        source.close()


def check(source, anchor, cursor):
    return source.restart_check_pin(anchor_heads=[anchor], anchor_cursor=cursor)


def test_native_source_automatically_moves_check_and_preserves_anchor_on_reopen(program):
    source, anchor, cursor = program
    original = check(source, anchor, cursor)
    assert not original["advanced"]
    source.compare_and_set_status("A", anchor["revision"], "in_progress")
    source.record_validation_result(task_cid="task:a", outcome="passed", evidence_digest="sha256:" + "ab" * 32)
    source.compare_and_set_status("A", anchor["revision"] + 1, "completed", receipt={"validator": "test"}, evidence_digests=["sha256:" + "ab" * 32])
    before = source.snapshot().to_dict()
    pin = check(source, anchor, cursor)
    assert pin["advanced"] and pin["event_cursor"] == cursor + 3
    assert pin["anchor_cursor"] == original["anchor_cursor"] == cursor
    assert pin["progress"]["observed_heads"][0]["status"] == "completed"
    assert all(pin[key] is False for key in (
        "launch_authority", "completion_authority", "source_change_authority", "effect_settlement_authority",
    ))
    assert source.snapshot().to_dict() == before
    # A new client reconstructs the same pin without a process-local/JSON cache.
    with DatabaseTaskSource(source.database_path, install_schema=False, plan_root_cid="plan:a") as reopened:
        assert check(reopened, anchor, cursor) == pin


@pytest.mark.parametrize("change", [
    "DELETE FROM domain_events WHERE global_sequence=(SELECT MAX(global_sequence)-1 FROM domain_events)",
    "UPDATE tasks SET revision=revision+1",
    "UPDATE tasks SET body_json='{}'",
    "DELETE FROM task_revisions WHERE revision=2",
    "UPDATE task_revisions SET recorded_at='changed' WHERE revision=2",
    "DELETE FROM completion_receipts",
    "UPDATE completion_receipts SET evidence_digest='forged'",
    "UPDATE completion_receipts SET task_cid='foreign'",
    "UPDATE completion_receipts SET body_json=json_merge_patch(body_json, '{\"schema\":\"foreign@999\"}')",
    "UPDATE completion_receipts SET body_json=json_merge_patch(body_json, '{\"extra\":true}')",
])
def test_corrupt_history_or_receipts_cannot_move_restart_check(program, change):
    source, anchor, cursor = program
    source.compare_and_set_status("A", anchor["revision"], "in_progress")
    source.record_validation_result(task_cid="task:a", outcome="passed", evidence_digest="sha256:" + "ab" * 32)
    source.compare_and_set_status("A", anchor["revision"] + 1, "completed", receipt={"validator": "test"}, evidence_digests=["sha256:" + "ab" * 32])
    with source.intent._connection(write=True) as connection:
        connection.execute(change)
    with pytest.raises(TaskHeadProgressError):
        check(source, anchor, cursor)


def test_definition_change_requires_native_requalification(program):
    source, anchor, cursor = program
    source.intent.upsert_task(task_cid="task:b", task_alias="B", goal_cid="goal:a")
    with pytest.raises(TaskHeadProgressError):
        check(source, anchor, cursor)


def test_event_append_can_follow_transition_timestamp_without_invalidating_pin(program):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity

    source, anchor, cursor = program
    source.compare_and_set_status("A", anchor["revision"], "in_progress")
    # The event wrapper is stamped independently after the revision is stored.
    event = dict(source.intent.list_events(after_global_sequence=cursor)[0])
    event["body"] = dict(event["body"])
    later = (datetime.fromisoformat(event["recorded_at"].replace("Z", "+00:00")) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    event["recorded_at"] = event["body"]["recorded_at"] = later
    event_id = content_identity({key: event[key] for key in (
        "stream_id", "sequence", "global_sequence", "event_type", "body",
    )})
    with source.intent._connection(write=True) as connection:
        connection.execute("UPDATE domain_events SET event_id=?,recorded_at=?,body_json=? WHERE global_sequence=?", [
            event_id, event["recorded_at"], json.dumps(event["body"]), cursor + 1,
        ])
        # Encoding differences do not change the contents of a task revision.
        row = connection.execute("SELECT body_json FROM tasks WHERE task_cid='task:a'").fetchone()
        connection.execute("UPDATE tasks SET body_json=?", [json.dumps(json.loads(row[0]), indent=2)])
    assert check(source, anchor, cursor)["event_cursor"] == cursor + 1


def test_quack_executes_closed_snapshot_on_server_without_streaming_joins(program):
    from ipfs_accelerate_py.agent_supervisor.task_sources.restart_check_pin import inspect_restart_check_pin

    source, anchor, cursor = program
    observed = []
    with source.intent._connection(write=False) as connection:
        class Quack:
            _quack_uri = "quack:127.0.0.1:24070"
            _quack_mutation_token = "private-test-token"
            def execute(self, sql, parameters):
                assert sql == "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)"
                uri, query, token = parameters
                assert uri == self._quack_uri and token == self._quack_mutation_token
                assert token not in query and "?" not in query
                observed.append(query)
                return connection.execute(query)
        pin = inspect_restart_check_pin(Quack(), anchor_heads=[anchor], anchor_cursor=cursor)
    assert pin["event_cursor"] == cursor and len(observed) == 1


def test_quack_failure_does_not_fall_back_or_expose_credential(program):
    from ipfs_accelerate_py.agent_supervisor.task_sources.restart_check_pin import inspect_restart_check_pin

    _, anchor, cursor = program
    class FailedQuack:
        _quack_uri = "quack:127.0.0.1:24070"
        _quack_mutation_token = "private-test-token"
        def execute(self, *args):
            raise RuntimeError(self._quack_mutation_token)
    with pytest.raises(TaskHeadProgressError, match="Quack snapshot query unavailable") as caught:
        inspect_restart_check_pin(FailedQuack(), anchor_heads=[anchor], anchor_cursor=cursor)
    assert "private-test-token" not in str(caught.value)


def test_restart_races_refresh_automatically_but_integrity_failures_do_not():
    from ipfs_accelerate_py.agent_supervisor.task_sources.restart_check_pin import (
        RestartCheckPinChanged, refresh_restart_check,
    )

    calls = []
    def native_check():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise RestartCheckPinChanged("current event cursor advanced")
        return {"valid": True, "pin": 350}
    assert refresh_restart_check(native_check) == {"valid": True, "pin": 350}
    assert calls == [1, 2, 3]
    calls.clear()
    def corrupt():
        calls.append(1)
        raise TaskHeadProgressError("missing completion receipt")
    with pytest.raises(TaskHeadProgressError, match="missing completion"):
        refresh_restart_check(corrupt)
    assert calls == [1]
    calls.clear()
    def racing():
        calls.append(1)
        raise RestartCheckPinChanged("current event cursor advanced")
    with pytest.raises(RestartCheckPinChanged):
        refresh_restart_check(racing)
    assert len(calls) == 3


def test_observation_uses_one_read_statement(program):
    from ipfs_accelerate_py.agent_supervisor.task_sources.restart_check_pin import inspect_restart_check_pin

    source, anchor, cursor = program
    statements = []
    with source.intent._connection(write=False) as connection:
        class Recorded:
            def execute(self, sql, parameters):
                statements.append(sql)
                return connection.execute(sql, parameters)
        assert not inspect_restart_check_pin(Recorded(), anchor_heads=[anchor], anchor_cursor=cursor)["advanced"]
    assert len(statements) == 1


def test_sawm_native_restart_check_uses_current_proven_heads(program, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("sawm_restart_pin_test", root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source, anchor, cursor = program
    source.compare_and_set_status("A", anchor["revision"], "in_progress")
    authority = {"expected_task_heads": {"A": anchor}}
    population = {"taskboard": [{"task_id": "A", "task_cid": "task:a", "goal_cid": "goal:a"}], "plan_root_cid": "plan:a"}
    monkeypatch.setattr(module, "_M70_TARGET_EVENT_WATERMARK", cursor)
    pin = module._m70_restart_check_pin(source, population, authority)
    assert pin["event_cursor"] == cursor + 1
    assert authority["expected_task_heads"]["A"]["revision"] == 1
    # Keep the board-specific counts while exercising its real head verifier.
    real_snapshot = source.snapshot
    def snapshot():
        current = real_snapshot()
        body = current.to_dict()
        body.update(task_count=45, goal_count=29, dependency_count=136)
        return SimpleNamespace(**body, to_dict=lambda: body)
    monkeypatch.setattr(source, "snapshot", snapshot)
    materializer = SimpleNamespace(
        _validated_m70_live_preflight_contract=lambda _: {"bind_store_report_to_live_event_digest": True},
        _M70_TARGET_PLAN_REVISION=1, MigrationRequired=RuntimeError,
    )
    statuses, revisions, _ = module._verify_m70_live_head_task_projection(
        source, population, materializer, authority=authority,
        expected_projection_cid=source.snapshot().projection_cid,
    )
    assert statuses == {"A": "in_progress"} and revisions == {"A": 2}
