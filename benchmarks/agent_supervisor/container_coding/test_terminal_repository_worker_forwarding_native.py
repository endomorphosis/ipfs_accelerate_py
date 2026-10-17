"""Native owner -> worker stdio -> copied forwarding evidence, without providers.

Native indexing, admitted leases, owner peer/PID/socket checks, framing, FIFO
writes and retained audit bytes are real. Container, worker UID, allocated
worktree, router and task identities use explicitly authored existing fixtures.
A worker-reported witness remains untrusted even when these local tests also
observe the pipe bytes; no production receipt or model ingestion is established.
"""
import copy
import hashlib
import json
import os
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as bindings
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio
from benchmarks.agent_supervisor.container_coding import terminal_repository_worker_forwarding as forwarding
from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_native_observation import (
    _row, _wait_for_response_accounting,
)
from test.api.test_repository_source_owner_broker import owner, short_service_directory
from test.api.test_repository_source_parent_admission import native_parent
from test.api.test_repository_source_owner_roundtrip import (
    authored_worker_descriptor, _message, _tool, _output_pipe,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bridge(owner, messages):
    payload = b"".join(protocol.canonical_json(message) + b"\n" for message in messages)
    observer = stdio.ForwardingObservation()
    input_fd, sender_fd = os.pipe()
    with os.fdopen(input_fd, "rb", buffering=0) as incoming:
        try:
            assert len(payload) <= os.fpathconf(sender_fd, "PC_PIPE_BUF")
            assert os.write(sender_fd, payload) == len(payload)
        finally:
            os.close(sender_fd)
        with _output_pipe() as (outgoing, received):
            code = stdio.serve(owner.descriptor_path, owner.descriptor_sha256,
                workspace=owner.workspace, incoming=incoming, outgoing=outgoing,
                deadline_monotonic=time.monotonic() + 20, observation=observer)
        stdout = bytes(received)
    receipt = observer.receipt()
    assert not stdout or stdout.endswith(b"\n")
    assert len(stdout) <= protocol.MAX_RESPONSE_FRAME_BYTES
    return code, stdout, receipt


@pytest.mark.parametrize("outcome", ["read", "failed_read", "stdout_checkpoint_failure"])
def test_native_forwarding_copies_join_without_independent_worker_authority(
        owner, native_parent, authored_worker_descriptor, monkeypatch, tmp_path, request, outcome):
    binding = bindings.create_binding(workspace=owner.workspace,
        requested_policy="repository-source-raw@1", descriptor_path=owner.descriptor_path,
        descriptor_sha256=owner.descriptor_sha256)
    binding["usage"] = bindings.unavailable_usage()
    row = _row(binding)
    original_row = copy.deepcopy(row)
    verified = []
    verify_peer = protocol._base._verify_peer

    def real_peer(channel, descriptor):
        result = verify_peer(channel, descriptor)
        verified.append(descriptor["owner_pid"])
        return result

    monkeypatch.setattr(protocol._base, "_verify_peer", real_peer)
    if outcome == "stdout_checkpoint_failure":
        native_write, native_checkpoint = stdio._Stdio.write, stdio._Stdio.checkpoint
        phase = {"target": False, "checkpoints": 0}

        def traced_write(stream, raw):
            phase["target"] = json.loads(raw).get("id") == 3
            phase["checkpoints"] = 0
            try:
                return native_write(stream, raw)
            finally:
                phase["target"] = False

        def fail_final_checkpoint(stream):
            if phase["target"]:
                phase["checkpoints"] += 1
                if phase["checkpoints"] == 2:
                    raise protocol.RepositorySourceOwnerProtocolError("cancelled")
            return native_checkpoint(stream)

        monkeypatch.setattr(stdio._Stdio, "write", traced_write)
        monkeypatch.setattr(stdio._Stdio, "checkpoint", fail_final_checkpoint)
    messages = [
        _message("initialize", 1, {"protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "authored-forwarding-fixture", "version": "1"}}),
        _message("notifications/initialized", None),
        _tool(2, "repository_status"),
        _tool(3, "repository_read", generation=2 if outcome == "failed_read" else 1, path="calc.py"),
    ]
    code, stdout, receipt = _bridge(owner, messages)
    assert code == (2 if outcome == "stdout_checkpoint_failure" else 0)
    assert receipt["complete"] is True and receipt["returncode"] == code
    assert receipt["owner_exchange_attempts"] == receipt["observed_responses"] == 4
    assert receipt["owner_exchange_unconfirmed"] == 0
    events = receipt["events"]
    assert len(events) == 4 and [event["exchange_ordinal"] for event in events] == [1, 2, 3, 4]
    assert events[1]["output_kind"] == events[1]["forwarding_status"] == "no_output"
    assert events[1]["stdout_sha256"] is None and events[1]["stdout_bytes"] == 0
    for event, line in zip((events[0], events[2], events[3]), stdout.splitlines(keepends=True)):
        assert event["output_kind"] == "owner_message"
        assert event["stdout_sha256"] == _sha(line) and event["stdout_bytes"] == len(line)
        expected = "unconfirmed" if event is events[3] and outcome == "stdout_checkpoint_failure" else "write_completed"
        assert event["forwarding_status"] == expected
    replies = [protocol.strict_json_loads(line) for line in stdout.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3]
    assert json.loads(replies[1]["result"]["content"][0]["text"])["generation"] == 1
    assert replies[-1]["result"]["isError"] is (outcome == "failed_read")
    if outcome != "failed_read":
        assert "value + 1" in json.loads(replies[-1]["result"]["content"][0]["text"])["content"]
    assert len(verified) == 8  # Real peer verification before and after each owner exchange.
    _wait_for_response_accounting(owner, 4)
    audit = owner.close()
    assert audit["cleanup_complete"] and audit["accounting_complete"]
    assert audit["schema"] == "repository-source-owner-session@2"
    assert len(audit["requests"]) == 4
    for event, record in zip(events, audit["requests"]):
        assert event["request_sha256"] == record["request_sha256"]
        assert event["owner_response_sha256"] == record["response_sha256"]
        assert event["owner_response_bytes"] == record["response_bytes"]
    assert not native_parent.released
    operations_path = owner.audit_path.parent / audit["retained_mcp_audit"]["name"]
    originals = {owner.audit_path: owner.audit_path.read_bytes(), operations_path: operations_path.read_bytes()}
    assert _sha(originals[operations_path]) == audit["retained_mcp_audit"]["sha256"]
    receipt_raw = protocol.canonical_json(receipt)
    assert b"value + 1" not in receipt_raw and str(owner.workspace).encode() not in receipt_raw
    copied = tmp_path / "copied-forwarding-evidence"
    copied.mkdir(mode=0o700)
    artifacts = {"owner-audit.json": originals[owner.audit_path],
                 "owner-operations.jsonl": originals[operations_path], "worker-forwarding.json": receipt_raw}
    for name, raw in artifacts.items():
        path = copied / name
        path.write_bytes(raw)
        path.chmod(0o400)
        assert path.read_bytes() == raw
    result = forwarding.summarize_worker_forwarding(
        audit_raw=(copied / "owner-audit.json").read_bytes(),
        operations_raw=(copied / "owner-operations.jsonl").read_bytes(), row=row,
        forwarding_receipts=[(copied / "worker-forwarding.json").read_bytes()])
    assert result["status"] == "observed", result
    counts = result["counts"]
    failed_output = int(outcome == "stdout_checkpoint_failure")
    assert counts["reported_received_operations"] == 3
    assert counts["reported_received_tool_results"] == 2
    assert counts["reported_received_successful_reads"] == int(outcome != "failed_read")
    assert counts["reported_completed_stdout_operations"] == 3 - failed_output
    assert counts["reported_completed_stdout_tool_results"] == 2 - failed_output
    assert counts["reported_completed_stdout_successful_reads"] == int(outcome == "read")
    assert counts["unconfirmed_stdout_operations"] == failed_output
    assert result["worker_reported_only"] is True
    for key in ("independent_worker_receipt_verified", "provider_ingestion_verified",
                "output_payloads_independently_verified", "token_savings_measured", "proof_authority",
                "execution_authority", "completion_authority", "production_route_qualified"):
        assert result[key] is False
    assert row == original_row
    assert {path: path.read_bytes() for path in originals} == originals
    assert not native_parent.released
    request.node.user_properties.append(("worker_forwarding_observation", json.dumps(result, sort_keys=True)))
    request.node.user_properties.append(("worker_stdout_capture", json.dumps(dict(
        sha256=_sha(stdout), bytes=len(stdout), response_count=len(replies),
        line_sha256=[_sha(line) for line in stdout.splitlines(keepends=True)],
        content_retained=False, fixture_only=True), sort_keys=True)))
    request.node.user_properties.append(("evidence_boundary",
        "New admitted native fixture; authored worker/container/router/task identities; real FIFO capture; "
        "copied owner evidence and worker-reported receipt; no production handoff or provider ingestion"))
