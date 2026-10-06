"""Bounded native observations preserve failure decisions and exclude bodies."""
import json
import linecache
import os
import traceback
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import bridge_failure_diagnostics as diagnostics
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeError


def _envelope():
    return {"schema": "router-implementation-error@2", "error_type": "ValueError", "diagnostic": {
        "schema": "router-implementation-failure-diagnostic@1", "phase": "argument_validation",
        "exceptions": [{"exception_type": "ValueError", "frames": []}], "chain_truncated": False,
        "completion_authority": False, "automatic_retry_admitted": False,
        "settlement_authority": False, "provider_dispatch_observed": None}}


def _observe():
    return diagnostics.observe_bridge_failure(DatabasePortalBridgeError("portal_provider_failed"), {
        "schema": "database-native-provider-failure@1", "callback_state": "failed_outcome_settled",
        "provider_effect_state": "failed_provider_exited", "native_exit": {
            "schema": "database-native-provider-exit@1", "returncode": 1,
            "reaped": True, "process_group_absent": True, "subreaper_children_absent": True,
            "lifecycle_finalized": True}}, phase="terminal_failure")


def test_no_traceback_source_or_exception_message_reads(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("traceback source access is forbidden")
    monkeypatch.setattr(linecache, "getline", forbidden)
    monkeypatch.setattr(linecache, "getlines", forbidden)
    monkeypatch.setattr(traceback, "extract_tb", forbidden)
    secret = "private model text must never be retained"
    failure = DatabasePortalBridgeError(secret)
    failure.__cause__ = failure
    result = diagnostics.observe_bridge_failure(failure, {}, phase="unknown_callback")
    assert result["reason_code"] == "other"
    assert result["chain_truncated"] is True
    assert len(result["exceptions"]) == 1
    assert secret not in json.dumps(result)
    assert diagnostics.validate_bridge_failure_diagnostic(result) == result


def test_deep_traceback_walk_is_bounded_without_source_reads():
    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        raise DatabasePortalBridgeError("private")
    try:
        recurse(300)
    except DatabasePortalBridgeError as failure:
        result = diagnostics.observe_bridge_failure(failure, {}, phase="unknown_callback")
    assert result["chain_truncated"] is True
    assert result["exceptions"] == [{"exception_type": "DatabasePortalBridgeError", "frames": []}]


def test_exception_formatting_hooks_are_not_executed():
    class UntrustedFormatting(DatabasePortalBridgeError):
        def __str__(self):
            raise AssertionError("exception formatting is not diagnostic authority")
    failure = UntrustedFormatting("x" * 1_000_000)
    result = diagnostics.observe_bridge_failure(failure, {}, phase="unknown_callback")
    assert result["reason_code"] == "other"
    assert result["exceptions"][0]["exception_type"] == "other"
    assert diagnostics.validate_bridge_failure_diagnostic(result) == result


@pytest.mark.parametrize("mutation", ["authority", "type", "frame", "extra", "callback", "bool-returncode", "absent-custody"])
def test_closed_projection_rejects_malformed_and_injected_fields(mutation):
    value = _observe()
    if mutation == "authority":
        value["settlement_authority"] = True
    elif mutation == "type":
        value["exceptions"][0]["exception_type"] = "private_dynamic_class"
    elif mutation == "frame":
        value["exceptions"][0]["frames"] = [{"file": "/private/source.py", "line": 1}]
    elif mutation == "extra":
        value["exceptions"][0]["message"] = "private"
    elif mutation == "callback":
        value["callback"]["state"] = []
    elif mutation == "bool-returncode":
        value["callback"]["native_exit"]["returncode"] = True
    else:
        value["callback"]["native_exit"]["present"] = False
    assert diagnostics.validate_bridge_failure_diagnostic(value) is None


@pytest.mark.parametrize("kind", ["single", "duplicate", "nested", "malformed", "conflict", "trailing-partial", "oversize"])
def test_child_json_is_bounded_observation_only(tmp_path, kind):
    path = tmp_path / "native.log"
    row = json.dumps(_envelope()) + "\n"
    expected = {"single": "observed", "duplicate": "ambiguous", "nested": "missing",
                "malformed": "invalid", "conflict": "ambiguous", "trailing-partial": "invalid",
                "oversize": "invalid"}[kind]
    raw = {"single": row, "duplicate": row * 2,
           "nested": json.dumps({"model_text": _envelope()}) + "\n",
           "malformed": '{"schema":"router-implementation-error@2",broken}\n',
           "conflict": row + '{"schema":"router-implementation-error@2",broken}\n',
           "trailing-partial": row.rstrip("\n"),
           "oversize": '{"schema":"router-implementation-error@2","body":"' + "x" * 17000 + '"}\n'}[kind]
    with path.open("w") as writer:
        writer.write(raw)
        result = diagnostics.child_report_from_log(path, writer)
    assert result["status"] == expected
    assert result["observation_only"] is True
    assert result["scope"] == "bounded_native_log_tail"
    if expected == "observed":
        # Even a model could print this row; it cannot establish dispatch.
        assert result["diagnostic"]["diagnostic"]["provider_dispatch_observed"] is None
        assert result["diagnostic"]["diagnostic"]["settlement_authority"] is False
    else:
        assert result["diagnostic"] is None


@pytest.mark.parametrize("change", ["replacement", "symlink"])
def test_child_log_must_match_exact_open_writer_inode(tmp_path, change):
    path = tmp_path / "native.log"
    with path.open("w") as writer:
        writer.write(json.dumps(_envelope()) + "\n")
        writer.flush()
        path.rename(tmp_path / "original")
        if change == "symlink":
            path.symlink_to(tmp_path / "original")
        else:
            path.write_text(json.dumps(_envelope()) + "\n")
        assert diagnostics.child_report_from_log(path, writer)["status"] == "unavailable"


@pytest.mark.parametrize("pooled", [False, True])
def test_actual_native_pre_receipt_failure_retains_closed_diagnostics(tmp_path, monkeypatch, pooled):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=pooled)
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "router_timeout.py").write_text(
        "import sys\nsys.path.insert(0," + repr(str(root)) + ")\n"
        "from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner\n"
        "sys.argv=['runner','--model','fixture','--timeout','0']\nraise SystemExit(runner.main())\n")
    try:
        result = daemon.run_once()
        observed = result["implementation_result"]["bridge_failure_diagnostic"]
        assert diagnostics.validate_bridge_failure_diagnostic(observed) == observed
        assert observed["phase"] == "terminal_failure"
        assert observed["reason_code"] == "portal_provider_failed"
        assert observed["callback"]["state"] == "failed_outcome_settled"
        assert observed["callback"]["native_exit"]["subreaper_children_absent"] is True
        child = observed["child_reported_router_failure"]
        assert child["status"] == "observed"
        assert child["diagnostic"]["diagnostic"]["phase"] == "argument_validation"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "failed"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        assert daemon.claim_next() is None
        events = daemon._require_connection().execute(
            "SELECT body_json FROM daemon_execution_events WHERE event_type = ?", [diagnostics.EVENT]).fetchall()
        assert len(events) == 1 and json.loads(events[0][0]) == observed
        logs = list(bridge._paths(attempt).root.rglob("*.log"))
        assert not any("router-implementation-invocation@1" in path.read_text() for path in logs)
    finally:
        daemon.close()


def test_diagnostic_failure_never_changes_native_settlement(tmp_path, monkeypatch):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("diagnostic unavailable")
    monkeypatch.setattr(diagnostics, "observe_bridge_failure", unavailable)
    try:
        result = daemon.run_once()
        assert result["implementation_result"]["bridge_failure_diagnostic"] is None
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "failed"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
    finally:
        daemon.close()


def test_unknown_callback_retains_custody_with_closed_diagnostic(tmp_path, monkeypatch):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch)
    def unavailable(*_args, **_kwargs):
        raise DatabasePortalBridgeError("private pre-dispatch failure body")
    bridge.portal_factory = unavailable
    try:
        result = daemon.run_once()
        implementation = result["implementation_result"]
        assert implementation["reason"] == "provider_callback_outcome_unknown"
        observed = implementation["bridge_failure_diagnostic"]
        assert diagnostics.validate_bridge_failure_diagnostic(observed) == observed
        assert observed["phase"] == "unknown_callback"
        assert observed["reason_code"] == "other"
        assert observed["callback"]["state"] == "started_outcome_unknown"
        assert observed["callback"]["native_exit"]["present"] is False
        assert "private pre-dispatch" not in json.dumps(observed)
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        assert portals == []
    finally:
        daemon.close()
