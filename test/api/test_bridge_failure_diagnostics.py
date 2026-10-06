"""Bounded native observations preserve failure decisions and exclude bodies."""
import hashlib
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


def test_legacy_bridge_diagnostic_remains_readable_without_custody_observation():
    legacy = _observe()
    legacy["schema"] = diagnostics.LEGACY_SCHEMA
    legacy.pop("native_provider_custody_observation")
    assert diagnostics.validate_bridge_failure_diagnostic(legacy) == legacy
    assert "native_provider_custody_observation" not in legacy


@pytest.mark.parametrize("mutation", ["legacy-extra", "current-missing", "unknown-version"])
def test_bridge_diagnostic_versions_keep_separate_closed_shapes(mutation):
    value = _observe()
    if mutation == "legacy-extra":
        value["schema"] = diagnostics.LEGACY_SCHEMA
    elif mutation == "current-missing":
        value.pop("native_provider_custody_observation")
    else:
        value["schema"] = "database-bridge-failure-diagnostic@3"
    assert diagnostics.validate_bridge_failure_diagnostic(value) is None


def test_untrusted_custody_payload_is_not_exported_or_promoted():
    private = {"reason_code": "private command and source text", "settlement_authority": True}
    failure = DatabasePortalBridgeError("portal_provider_failed", result={
        "native_provider_custody_observation": private})
    observed = diagnostics.observe_bridge_failure(failure, {}, phase="unknown_callback")
    assert observed["schema"] == diagnostics.SCHEMA
    assert observed["native_provider_custody_observation"] is None
    assert "private" not in json.dumps(observed)
    assert observed["callback"]["native_exit"]["present"] is False
    assert observed["settlement_authority"] is False
    assert diagnostics.validate_bridge_failure_diagnostic(observed) == observed
    observed["native_provider_custody_observation"] = private
    assert diagnostics.validate_bridge_failure_diagnostic(observed) is None


def test_native_gate_denial_survives_projection_without_granting_custody(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_provider_custody_observation import (
        append_native_provider_custody_check,
    )
    custody = append_native_provider_custody_check(None, "issuance", "lifecycle_not_finalized")
    failure = DatabasePortalBridgeError("portal_provider_failed", result={
        "native_provider_custody_observation": custody})
    observed = diagnostics.observe_bridge_failure(failure, {}, phase="unknown_callback")
    assert observed["native_provider_custody_observation"] == custody
    assert observed["callback"]["native_exit"]["present"] is False
    assert observed["settlement_authority"] is False
    assert diagnostics.validate_bridge_failure_diagnostic(observed) == observed
    diagnostics.write_bridge_failure_observation(tmp_path, task_cid="task:fixture",
        attempt_id="attempt:fixture", diagnostic=observed)
    envelope = json.loads((tmp_path / diagnostics.OBSERVATION_FILENAME).read_bytes())
    assert diagnostics.validate_bridge_failure_observation(envelope) == envelope
    assert envelope["diagnostic"]["native_provider_custody_observation"] == custody
    custody["checks"][0]["reason_code"] = "private mutation"
    assert "private" not in json.dumps(observed)


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
    daemon.execution_state_dir = tmp_path / "native-state"
    assert daemon.events_path is None
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
        sidecar = daemon.execution_state_dir / diagnostics.OBSERVATION_FILENAME
        envelope = json.loads(sidecar.read_bytes())
        assert diagnostics.validate_bridge_failure_observation(envelope) == envelope
        assert envelope["diagnostic"] == observed
        assert envelope["task_cid_sha256"] == hashlib.sha256(attempt.task_cid.encode()).hexdigest()
        assert envelope["attempt_id_sha256"] == hashlib.sha256(attempt.attempt_id.encode()).hexdigest()
        assert sidecar.stat().st_size <= diagnostics.OBSERVATION_MAX_BYTES
        assert list(daemon.execution_state_dir.iterdir()) == [sidecar]
        logs = list(bridge._paths(attempt).root.rglob("*.log"))
        assert not any("router-implementation-invocation@1" in path.read_text() for path in logs)
    finally:
        daemon.close()


@pytest.mark.parametrize("broken", ["observe_bridge_failure", "write_bridge_failure_observation"])
def test_diagnostic_failure_never_changes_native_settlement(tmp_path, monkeypatch, broken):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("diagnostic unavailable")
    monkeypatch.setattr(diagnostics, broken, unavailable)
    try:
        result = daemon.run_once()
        observed = result["implementation_result"]["bridge_failure_diagnostic"]
        assert (observed is None) == (broken == "observe_bridge_failure")
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "failed"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
    finally:
        daemon.close()


@pytest.mark.parametrize("mutation", ["extra", "raw_identity", "bool_hash", "authority", "diagnostic"])
def test_sidecar_validator_rejects_injected_binding_or_authority(tmp_path, mutation):
    diagnostics.write_bridge_failure_observation(tmp_path, task_cid="private:task",
        attempt_id="private:attempt", diagnostic=_observe())
    value = json.loads((tmp_path / diagnostics.OBSERVATION_FILENAME).read_bytes())
    if mutation == "extra":
        value["message"] = "private body"
    elif mutation == "raw_identity":
        value["task_cid_sha256"] = "private:task"
    elif mutation == "bool_hash":
        value["attempt_id_sha256"] = True
    elif mutation == "authority":
        value["observation_only"] = False
    else:
        value["diagnostic"]["settlement_authority"] = True
    assert diagnostics.validate_bridge_failure_observation(value) is None


def test_sidecar_replacement_is_atomic_and_never_follows_destination_symlink(tmp_path, monkeypatch):
    target = tmp_path / "external"
    target.write_text("unchanged")
    state = tmp_path / "state"
    state.mkdir()
    sidecar = state / diagnostics.OBSERVATION_FILENAME
    sidecar.symlink_to(target)
    diagnostics.write_bridge_failure_observation(state, task_cid="task:first",
        attempt_id="attempt:first", diagnostic=_observe())
    assert not sidecar.is_symlink() and target.read_text() == "unchanged"
    first = sidecar.read_bytes()
    assert b"task:first" not in first and b"attempt:first" not in first
    def interrupted(*_args):
        raise OSError("authored atomic replacement interruption")
    monkeypatch.setattr(diagnostics.os, "replace", interrupted)
    with pytest.raises(OSError):
        diagnostics.write_bridge_failure_observation(state, task_cid="task:second",
            attempt_id="attempt:second", diagnostic=_observe())
    assert sidecar.read_bytes() == first
    assert list(state.iterdir()) == [sidecar]


def test_sidecar_requires_explicit_state_directory_without_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    diagnostics.write_bridge_failure_observation(None, task_cid="task",
        attempt_id="attempt", diagnostic=_observe())
    assert list(tmp_path.iterdir()) == []


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
