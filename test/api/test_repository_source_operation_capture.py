"""Exact append witnesses with authored MCP sessions and real bounded files.

These tests create no scheduler, native index, provider or worker. An authored
checkpoint and session exercise the existing MCP handle, while audit writes
and witness hashes are real. They establish correlation mechanics only.
"""
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp


@pytest.fixture
def audit(tmp_path):
    value = mcp._Audit(tmp_path / "operations.jsonl")
    try:
        yield value
    finally:
        value.close()


def _raw(audit):
    return Path(os.readlink("/proc/self/fd/" + str(audit.fd))).read_bytes()


def _reference(raw):
    return dict(sequence=json.loads(raw)["sequence"], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


@pytest.fixture
def server(audit):
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def status_summary(self):
            return {"generation": 1, "source_current": True}
    value = mcp.RepositoryMCPServer.__new__(mcp.RepositoryMCPServer)
    value.audit = audit
    value.checkpoint = lambda: None  # Authored; no resource state is opened.
    value.initialization_attempted = value.ready = False
    value.session = value.selection_audit = None
    value.session_factory = Session
    value._resource_options = {}
    value.config = {"workspace": "/authored/workspace", "state_dir": "/authored/state", "representation": "original"}
    return value


def _initialize(server):
    response = server.handle(dict(jsonrpc="2.0", id=1, method="initialize", params={
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {}}))
    server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
    return response


def _status(server):
    return server.handle(dict(jsonrpc="2.0", id=2, method="tools/call",
                              params={"name": "repository_status", "arguments": {}}))


def test_reference_commits_exact_line_including_newline_and_preserves_native_schema(audit):
    with audit.capture() as capture:
        audit.record("tool", tool="repository_status", returned_bytes=3)
    raw = _raw(audit)
    assert raw.endswith(b"\n") and capture == {"status": "complete", "operations": [_reference(raw)]}
    assert json.loads(raw)["schema"] == "repository-source-mcp-audit@1"
    assert capture["operations"][0]["sha256"] != hashlib.sha256(raw[:-1]).hexdigest()
    assert "tool" not in capture["operations"][0] and "content" not in json.dumps(capture)


def test_scope_does_not_infer_previous_or_subsequent_operations_from_sequence(audit):
    audit.record("initialize")
    with audit.capture() as capture:
        audit.record("tool", tool="repository_status")
    audit.record("tool", tool="repository_status")
    lines = _raw(audit).splitlines(keepends=True)
    assert capture == {"status": "complete", "operations": [_reference(lines[1])]}
    assert capture["operations"][0]["sequence"] == 2


def test_passed_witness_is_the_same_object_and_can_establish_real_zero(audit):
    witness = {"status": "incomplete", "operations": []}
    with audit.capture(witness) as captured:
        assert captured is witness
    assert witness == {"status": "complete", "operations": []} and _raw(audit) == b""


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "id": 1, "method": "ping"},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "id": 1, "method": "unknown"},
    {"jsonrpc": "1.0", "id": 1, "method": "ping"},
])
def test_native_protocol_paths_can_complete_without_audited_operations(server, message):
    with server.capture_audit_operations() as capture:
        server.handle(message)
    assert capture == {"status": "complete", "operations": []}
    assert _raw(server.audit) == b""


def test_initialization_and_each_tool_have_separate_exact_witnesses(server):
    with server.capture_audit_operations() as initialized:
        assert "result" in _initialize(server)
    with server.capture_audit_operations() as called:
        assert _status(server)["result"]["isError"] is False
    lines = _raw(server.audit).splitlines(keepends=True)
    assert initialized == {"status": "complete", "operations": [_reference(lines[0])]}
    assert called == {"status": "complete", "operations": [_reference(lines[1])]}
    assert server.audit._capture is None


@pytest.mark.parametrize("failure", ["before", "partial", "zero"])
def test_append_failure_is_incomplete_even_if_caller_catches_it(audit, monkeypatch, failure):
    original = mcp.os.write
    calls = []
    def write(fd, raw):
        calls.append(len(raw))
        if failure == "partial" and len(calls) == 1:
            return original(fd, raw[:7])
        if failure == "zero": return 0
        raise OSError("authored audit append failure")
    monkeypatch.setattr(mcp.os, "write", write)
    with audit.capture() as capture:
        with pytest.raises(OSError):
            audit.record("initialize")
    assert capture == {"status": "incomplete", "operations": []}
    assert len(_raw(audit)) == (7 if failure == "partial" else 0)
    assert audit._capture is None


def test_partial_writes_publish_only_after_the_entire_line(audit, monkeypatch):
    original = mcp.os.write
    witness = {"status": "incomplete", "operations": []}
    calls = []
    def write(fd, raw):
        assert witness["operations"] == []
        calls.append(len(raw))
        return original(fd, raw[:13])
    monkeypatch.setattr(mcp.os, "write", write)
    with audit.capture(witness):
        audit.record("initialize")
    assert len(calls) > 1 and witness["operations"] == [_reference(_raw(audit))]
    assert witness["status"] == "complete"


def test_complete_write_followed_by_exception_cannot_invent_a_published_witness(audit, monkeypatch):
    original = mcp.os.write
    def uncertain(fd, raw):
        original(fd, raw)
        raise OSError("authored interruption after complete write")
    monkeypatch.setattr(mcp.os, "write", uncertain)
    with audit.capture() as capture:
        with pytest.raises(OSError):
            audit.record("initialize")
    assert _raw(audit).endswith(b"\n") and len(_raw(audit).splitlines()) == 1
    assert capture == {"status": "incomplete", "operations": []}


@pytest.mark.parametrize("error", [ValueError("authored witness failure"), OSError("authored witness failure"), MemoryError()])
def test_ordinary_witness_failure_does_not_change_successful_mcp_reply(server, monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(server.audit, "_publish_capture", unavailable)
    with server.capture_audit_operations() as capture:
        reply = _initialize(server)
    assert "result" in reply and server.ready is True
    assert capture == {"status": "incomplete", "operations": []}
    assert len(_raw(server.audit).splitlines()) == 1 and server.audit._capture is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_control_flow_interruption_during_publication_propagates_incomplete(server, monkeypatch, error):
    def interrupted(*args, **kwargs):
        raise error
    monkeypatch.setattr(server.audit, "_publish_capture", interrupted)
    with pytest.raises(type(error)):
        with server.capture_audit_operations() as capture:
            _initialize(server)
    assert capture == {"status": "incomplete", "operations": []}
    assert len(_raw(server.audit).splitlines()) == 1 and server.audit._capture is None
    with server.capture_audit_operations() as after:
        server.handle({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert after == {"status": "complete", "operations": []}


def test_publication_failure_after_reference_append_keeps_it_but_not_completeness(audit, monkeypatch):
    original = audit._publish_capture
    def failed_after_append(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("authored publication completion failed")
    monkeypatch.setattr(audit, "_publish_capture", failed_after_append)
    with audit.capture() as capture:
        audit.record("initialize")
    assert capture == {"status": "incomplete", "operations": [_reference(_raw(audit))]}


@pytest.mark.parametrize("error", [RuntimeError("authored handler failure"), KeyboardInterrupt(), SystemExit(2)])
def test_handler_failure_after_audit_commit_preserves_reference_without_completeness(audit, error):
    with pytest.raises(type(error)):
        with audit.capture() as capture:
            audit.record("initialize")
            raise error
    assert capture == {"status": "incomplete", "operations": [_reference(_raw(audit))]}
    assert audit._capture is None


def test_capture_limit_is_incomplete_without_rejecting_already_committed_operation(audit):
    with audit.capture() as capture:
        audit.record("initialize")
        audit.record("tool", tool="repository_status")
    lines = _raw(audit).splitlines(keepends=True)
    assert len(lines) == 2
    assert capture == {"status": "incomplete", "operations": [_reference(lines[0])]}


def test_nested_scope_is_refused_without_replacing_outer_capture(audit):
    with audit.capture() as capture:
        with pytest.raises(ValueError, match="already_active"):
            with audit.capture():
                pytest.fail("nested scope entered")
        audit.record("initialize")
    assert capture == {"status": "complete", "operations": [_reference(_raw(audit))]}
    assert audit._capture is None
    with audit.capture() as later:
        audit.record("tool", tool="repository_status")
    assert later == {"status": "complete", "operations": [_reference(_raw(audit).splitlines(keepends=True)[1])]}


@pytest.mark.parametrize("active", [False, True])
def test_foreign_thread_cannot_enter_capture(audit, active):
    errors = []
    def attempt():
        try:
            with audit.capture():
                errors.append("entered")
        except ValueError as error:
            errors.append(str(error))
    def run():
        worker = threading.Thread(target=attempt)
        worker.start()
        worker.join(2)
        assert not worker.is_alive()
    if active:
        with audit.capture() as capture:
            run()
        assert capture == {"status": "complete", "operations": []}
    else:
        run()
    assert errors == ["audit_capture_thread_differs"] and audit._capture is None


def test_foreign_thread_append_is_refused_and_capture_is_conservatively_incomplete(audit):
    errors = []
    def append():
        try:
            audit.record("initialize")
        except ValueError as error:
            errors.append(str(error))
    with audit.capture() as capture:
        worker = threading.Thread(target=append)
        worker.start()
        worker.join(2)
        assert not worker.is_alive()
    assert errors == ["audit_capture_thread_differs"]
    assert capture == {"status": "incomplete", "operations": []} and _raw(audit) == b""


@pytest.mark.parametrize("witness", [False, [], {}, {"status": "not_entered", "operations": []},
    {"status": "incomplete", "operations": [1]}, {"status": "incomplete", "operations": [], "extra": True}])
def test_malformed_capture_input_is_refused_before_state_changes(audit, witness):
    with pytest.raises(ValueError, match="invalid_audit_capture"):
        with audit.capture(witness):
            pytest.fail("invalid capture entered")
    assert audit._capture is None and _raw(audit) == b""


def test_scope_identity_change_cannot_report_complete(audit):
    with audit.capture() as capture:
        audit._capture = None  # Explicit authored corruption of the in-process scope.
    assert capture == {"status": "incomplete", "operations": []} and audit._capture is None


def test_validation_failure_caught_inside_scope_does_not_establish_zero(audit):
    with audit.capture() as capture:
        with pytest.raises(ValueError, match="invalid_read_audit"):
            audit.record("initialize", read_observation={})
    assert capture == {"status": "incomplete", "operations": []} and _raw(audit) == b""
