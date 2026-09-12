from __future__ import annotations

import copy
import json
import logging

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    compact_daemon_pass_result,
    daemon_pass_is_idle,
    log_daemon_pass_result,
)


def idle_with_recovery(**overrides):
    return {
        "unchanged": True,
        "write_count": 0,
        "active_task_id": "",
        "selection_idle_reason": "no_ready_tasks",
        "terminal_portal_reconciliations": [{
            "task_cid": "sha256:task032",
            "attempt_id": "attempt:failed032",
            "status": "blocked",
            "changed": False,
            "reason": "historical_fingerprint_unavailable",
            "operator_review_required": True,
            "provider_dispatched": True,
            "attempt_consumed": False,
        }],
        **overrides,
    }


def test_protected_recovery_refusal_survives_idle_heartbeat(caplog):
    result = idle_with_recovery()
    before = copy.deepcopy(result)
    logger = logging.getLogger("test-recovery-heartbeat")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_daemon_pass_result(logger, "Pass: %s", result, emit_idle_info=True)
        log_daemon_pass_result(logger, "Pass: %s", result, emit_idle_info=False)
    assert len(caplog.records) == 1
    summary = caplog.records[0].args["recovery_observations"]
    assert summary["entries"][0]["reason"] == "historical_fingerprint_unavailable"
    assert summary["entries"][0]["stage"] == "terminal_portal_reconciliations"
    assert summary["entries"][0]["task_cid"] == "sha256:task032"
    assert summary["entries"][0]["provider_dispatched"] is True
    assert summary["entries"][0]["attempt_consumed"] is False
    assert summary["retry_authority"] is False
    assert result == before and daemon_pass_is_idle(result)


def test_callback_and_retry_deferrals_are_visible_without_nested_evidence():
    result = idle_with_recovery(
        unknown_callback_reopens=[{
            "changed": False, "status": "unknown", "reason": "callback_outcome_unknown",
            "attempt_id": "attempt:unknown", "provider_dispatched": True,
            "recovery_receipt": {"private_details": "do not copy"},
        }],
        terminal_retry_reconciliations=[{
            "changed": False, "status": "blocked", "reason": "verification_recovery_lifecycle_changed",
            "task_alias": "DOEP-041", "recovery_deferred": True,
            "error_type": "OwnershipError", "error_id": "sha256:error",
            "error": "unbounded exception text", "task_body": {"data": "not a heartbeat"},
        }],
    )
    compact = compact_daemon_pass_result(result)
    entries = compact["recovery_observations"]["entries"]
    assert [row["stage"] for row in entries] == [
        "unknown_callback_reopens", "terminal_retry_reconciliations", "terminal_portal_reconciliations"]
    assert entries[1]["error_type"] == "OwnershipError"
    assert entries[1]["recovery_deferred"] is True
    rendered = json.dumps(compact)
    for excluded in ("private_details", "do not copy", "unbounded exception text", "not a heartbeat"):
        assert excluded not in rendered


def test_recovery_summary_has_bounded_rows_and_text():
    row = {"changed": False, "status": "blocked", "reason": "x" * 100000,
           "attempt_id": "y" * 100000, "recovery_receipt": {"data": "z" * 100000}}
    result = idle_with_recovery(terminal_portal_reconciliations=[row] * 1000)
    summary = compact_daemon_pass_result(result)["recovery_observations"]
    assert len(summary["entries"]) == 16
    assert summary["truncated"] is True
    assert all(len(entry["reason"]) == 512 for entry in summary["entries"])
    assert len(json.dumps(summary)) < 20000


@pytest.mark.parametrize("malformed", [None, {}, "not outcomes", 1])
def test_malformed_stage_does_not_crash_idle_logging(malformed):
    result = idle_with_recovery(terminal_portal_reconciliations=malformed)
    assert "recovery_observations" not in compact_daemon_pass_result(result)


def test_ordinary_noop_and_durable_change_are_not_misreported_as_rejected_recovery():
    result = idle_with_recovery(terminal_portal_reconciliations=[
        None, "bad row", {"changed": False, "status": "completed"},
        {"changed": False, "status": []}, {"changed": False, "status": {}},
        {"changed": True, "status": "retrying", "reason": "recovered"},
    ])
    assert "recovery_observations" not in compact_daemon_pass_result(result)


def test_changed_pass_keeps_full_result(caplog):
    result = idle_with_recovery(unchanged=False, write_count=1, implementation_result={"recovered": True})
    logger = logging.getLogger("test-changed-recovery-heartbeat")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_daemon_pass_result(logger, "Pass: %s", result, emit_idle_info=False)
    assert caplog.records[0].args == result
    assert not daemon_pass_is_idle(result)
