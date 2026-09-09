"""Retained running attempts remain barriers until native terminalization."""

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


@pytest.mark.parametrize(
    "identity_case",
    [
        "missing_process_id",
        "missing_record",
        "malformed_birth",
        "permission_denied",
        "closed_process",
        "dead_process",
        "reused_pid",
        "live_process",
    ],
)
def test_retained_extra_gate_blocks_dispatch_with_process_uncertainty(
    tmp_path, monkeypatch, identity_case
):
    calls = []
    daemon = _open_daemon(tmp_path, session="owner", provider_calls=calls, clock_ms=lambda: 1000)
    try:
        population = _population(2)
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt.task_alias == "DQP-T001"
        monkeypatch.setattr(
            daemon, "_task_alias_is_extra_gate", lambda task: task.task_cid == attempt.task_cid
        )
        attempt = daemon.commit_phase(attempt, "context")
        original_process_id = daemon.process_instance_id
        original_record = dict(daemon._database_process_instance_record(original_process_id))
        daemon.process_instance_id = "process:successor-observer"
        key = daemon._process_instance_metadata_key(original_process_id)
        connection = daemon._require_connection()
        body = dict(attempt.body)
        if identity_case != "missing_process_id":
            body["process_instance_id"] = original_process_id
        connection.execute(
            "UPDATE database_task_attempts SET body_json = ? WHERE attempt_id = ?",
            [json.dumps(body), attempt.attempt_id],
        )
        if identity_case == "missing_record":
            connection.execute("DELETE FROM daemon_execution_metadata WHERE key = ?", [key])
        else:
            record = dict(original_record)
            birth = dict(record["process_birth"])
            if identity_case == "malformed_birth":
                birth.pop("start_time_ticks")
            elif identity_case == "closed_process":
                record.update(state="closed", closed_at_ms=1000)
            elif identity_case in {"permission_denied", "dead_process"}:
                birth["pid"] = 987654321
                native_kill = module.os.kill

                def observed_kill(pid, sig):
                    if pid == 987654321:
                        if identity_case == "permission_denied":
                            raise PermissionError("process inspection denied")
                        raise ProcessLookupError("process absent")
                    return native_kill(pid, sig)

                monkeypatch.setattr(module.os, "kill", observed_kill)
            elif identity_case == "reused_pid":
                birth["start_time_ticks"] += 1
            record["process_birth"] = birth
            connection.execute(
                "UPDATE daemon_execution_metadata SET value = ? WHERE key = ?",
                [json.dumps(record), key],
            )
        daemon.task_prefix = "DQP-T002"
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = True
        daemon._database_portal_reconciliation_result = {"blocked": False}
        before = daemon.task_source.get(attempt.task_cid)
        before_attempt = daemon.get_attempt(attempt.attempt_id)
        before_claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert daemon.list_running_attempts() == []
        assert len(daemon.list_running_attempts(apply_selection=False)) == 1
        for _ in range(2):
            result = daemon.run_once()
            assert calls == []
            assert (
                result["selection_idle_reason"] == "database_portal_owner_attempt_outside_selection"
            )
            assert result["database_portal_reconciliation"]["safe_to_restart"] is False
            assert result["write_count"] == 0
            assert calls == []
            assert daemon.task_source.get(attempt.task_cid) == before
            assert daemon.get_attempt(attempt.attempt_id) == before_attempt
            assert daemon.coordinator.get_task_claim(attempt.claim_id) == before_claim
    finally:
        daemon.close()


def test_native_expiry_clears_retained_barrier_before_later_dispatch(tmp_path, monkeypatch):
    clock = {"now": 1000}
    calls = []
    daemon = _open_daemon(
        tmp_path,
        session="owner",
        provider_calls=calls,
        lease_ms=5000,
        clock_ms=lambda: clock["now"],
    )
    try:
        population = _population(2)
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        monkeypatch.setattr(
            daemon, "_task_alias_is_extra_gate", lambda task: task.task_cid == attempt.task_cid
        )
        attempt = daemon.commit_phase(attempt, "context")
        daemon.task_prefix = "DQP-T002"
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = True
        daemon._database_portal_reconciliation_result = {"blocked": False}
        clock["now"] = 7000
        result = daemon.run_once()
        assert result["selection_idle_reason"] == "database_expired_attempts_reconciled"
        assert calls == []
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
        assert daemon.list_running_attempts(apply_selection=False) == []
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
        result = daemon.run_once()
        assert (
            result.get("selection_idle_reason") != "database_portal_owner_attempt_outside_selection"
        )
        assert calls == ["task:cid:002"]
    finally:
        daemon.close()
