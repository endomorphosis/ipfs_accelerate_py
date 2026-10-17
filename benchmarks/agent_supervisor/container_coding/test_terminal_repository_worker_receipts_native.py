"""Native stdio publication -> worker snapshot -> copied owner evidence.

Native leases/indexing, owner peer/PID checks, JSON framing, real FIFO output,
receipt files and bounded capture are exercised through their existing APIs.
Worker/container isolation, router/task identity and cleanup report fields are
authored fixtures. These tests do not qualify production handoff, independent
worker authentication, helper quiescence or provider/model ingestion.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_worker_receipts as receipts_api
from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as observation
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


@pytest.fixture
def authored_outbox_worker_uid(owner, monkeypatch):
    """Author only outbox UID identity for the existing same-process fixture.

    The live owner descriptor names a distinct worker UID, while this fixture
    executes its stdio bridge in the test process. Project the target's worker
    UID after real creation; map only that field back during the original
    directory check. All actual OS identities, file checks, paths, descriptors,
    devices, inodes, modes, links, sizes and times remain unchanged. This is not
    a different-UID filesystem qualification.
    """
    owner_uid, worker_uid = os.getuid(), owner.descriptor["worker_uid"]
    assert owner_uid != worker_uid
    create, check = receipts_api.create_outbox, receipts_api._check_directory

    def projected_create(**kwargs):
        handle = create(**kwargs)
        assert handle.target["uid"] == owner_uid
        handle._target = {**handle._target, "uid": worker_uid}
        return handle

    def projected_check(fd, target):
        assert target["uid"] in {owner_uid, worker_uid}
        check(fd, {**target, "uid": owner_uid})

    monkeypatch.setattr(receipts_api, "create_outbox", projected_create)
    monkeypatch.setattr(receipts_api, "_check_directory", projected_check)


def _bridge_main(binding, messages, monkeypatch):
    """Use the bound helper arguments and real pipes, without a new process."""
    payload = b"".join(protocol.canonical_json(message) + b"\n" for message in messages)
    input_fd, sender_fd = os.pipe()
    diagnostic = io.StringIO()
    with os.fdopen(input_fd, "rb", buffering=0) as incoming:
        try:
            assert len(payload) <= os.fpathconf(sender_fd, "PC_PIPE_BUF")
            assert os.write(sender_fd, payload) == len(payload)
        finally:
            os.close(sender_fd)
        with _output_pipe() as (outgoing, received):
            # Replace this module's sys reference only; pytest's own streams
            # and the native owner/service thread keep their actual modules.
            with monkeypatch.context() as local:
                local.setattr(stdio, "sys", SimpleNamespace(
                    stdin=SimpleNamespace(buffer=incoming),
                    stdout=SimpleNamespace(buffer=outgoing), stderr=diagnostic))
                arguments = binding["config"]["settings"]["args"]
                assert arguments[:2] == ["-I", "-B"]
                assert arguments[2] == binding["helper_script"]
                code = stdio.main(arguments[3:])
        stdout = bytes(received)
    assert not stdout or stdout.endswith(b"\n")
    assert len(stdout) <= protocol.MAX_RESPONSE_FRAME_BYTES
    return code, stdout, diagnostic.getvalue()


def _report(row):
    report = dict(repository_source_policy="repository-source-raw@1", provider_invocations=[row],
        worker_cleanup_returncode=0, repository_source_dispatch={"helper_adoption_guaranteed": False},
        usage={"total_tokens": None})
    report["public_instruction"] = {key: row["public_instruction"][key] for key in (
        "artifact", "context_cid", "manifest_cid", "task_cid", "source_path", "source_sha256", "source_bytes",
        "completion_authority", "scope_expansion_authority")}
    report["public_instruction"]["sha256"] = row["public_instruction"]["artifact_sha256"]
    return report


@pytest.mark.parametrize("case", ["published", "missing", "partial", "report_tamper"])
def test_native_published_worker_receipts_remain_reports_after_owner_collection(
        owner, native_parent, authored_worker_descriptor, authored_outbox_worker_uid, monkeypatch, request, case):
    handle = receipts_api.create_outbox(workspace=owner.workspace,
        session_id=owner.descriptor["session_id"], descriptor_sha256=owner.descriptor_sha256,
        invocation_id="a" * 32)
    target = copy.deepcopy(handle.target)
    directory = Path(target["directory"])
    captured = None
    try:
        assert not directory.is_relative_to(owner.workspace)
        binding = binding_api.create_binding(workspace=owner.workspace,
            requested_policy="repository-source-raw@1", descriptor_path=owner.descriptor_path,
            descriptor_sha256=owner.descriptor_sha256, worker_forwarding=target)
        binding["usage"] = binding_api.unavailable_usage()
        row = _row(binding)
        verified = []
        verify_peer = protocol._base._verify_peer

        def real_peer(channel, descriptor):
            result = verify_peer(channel, descriptor)
            verified.append(descriptor["owner_pid"])
            return result

        monkeypatch.setattr(protocol._base, "_verify_peer", real_peer)
        messages = [
            _message("initialize", 1, {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "authored-receipt-fixture", "version": "1"}}),
            _message("notifications/initialized", None),
            _tool(2, "repository_status"),
            _tool(3, "repository_read", generation=1, path="calc.py"),
        ]
        code, stdout, diagnostic = _bridge_main(binding, messages, monkeypatch)
        assert code == 0 and diagnostic == ""
        assert len(verified) == 8
        slot = directory / "receipt-00.json"
        raw = slot.read_bytes()
        witness = protocol.strict_json_loads(raw)
        assert raw == protocol.canonical_json(witness)
        assert witness["complete"] is True and witness["returncode"] == 0
        assert witness["observed_responses"] == witness["owner_exchange_attempts"] == 4
        assert witness["owner_exchange_unconfirmed"] == 0
        events = witness["events"]
        assert events[1]["output_kind"] == events[1]["forwarding_status"] == "no_output"
        assert events[1]["stdout_sha256"] is None and events[1]["stdout_bytes"] == 0
        lines = stdout.splitlines(keepends=True)
        assert len(lines) == 3
        for event, line in zip((events[0], events[2], events[3]), lines):
            assert event["forwarding_status"] == "write_completed"
            assert event["stdout_sha256"] == _sha(line) and event["stdout_bytes"] == len(line)
        replies = [protocol.strict_json_loads(line) for line in lines]
        assert [reply["id"] for reply in replies] == [1, 2, 3]
        assert replies[-1]["result"]["isError"] is False
        assert "value + 1" in json.loads(replies[-1]["result"]["content"][0]["text"])["content"]
        assert b"value + 1" not in raw and str(owner.workspace).encode() not in raw

        if case == "missing":
            slot.unlink()
        elif case == "partial":
            slot.chmod(0o600)
            slot.write_bytes(raw[:len(raw) // 2])
        captured = handle.collect_and_close()
        assert not directory.exists()
        if case in {"partial", "missing"}:
            assert captured["status"] == "unavailable"
            assert captured["receipt_count"] is None
        else:
            assert captured["status"] == "captured" and captured["cleanup_complete"] is True
            assert captured["receipt_count"] == 1
            entry = captured["receipts"][0]
            assert entry == dict(receipt=witness, sha256=_sha(raw), bytes=len(raw), slot=0)
            assert receipts_api.captured_receipt_bytes(captured, target) == [raw]
        row["repository_source_worker_forwarding"] = copy.deepcopy(captured)
        if case == "report_tamper":
            # Change only the copied report; the retained-byte hash still binds
            # the original receipt, so owner collection must reject this copy.
            row["repository_source_worker_forwarding"]["receipts"][0]["receipt"]["events"][3]["stdout_sha256"] = "0" * 64
        report = _report(row)
        original_report = copy.deepcopy(report)
        scope = dict(task_cid="authored-task-cid", task_alias="authored-task", baseline_commit="d" * 40,
            outputs=[dict(path="calc.py", effect="modify", baseline_sha256="e" * 64)])

        _wait_for_response_accounting(owner, 4)
        audit = owner.close()
        assert audit["cleanup_complete"] and audit["accounting_complete"]
        assert audit["schema"] == "repository-source-owner-session@2"
        for event, record in zip(events, audit["requests"]):
            assert event["request_sha256"] == record["request_sha256"]
            assert event["owner_response_sha256"] == record["response_sha256"]
            assert event["owner_response_bytes"] == record["response_bytes"]
        operations = owner.audit_path.parent / audit["retained_mcp_audit"]["name"]
        originals = {path: path.read_bytes() for path in (owner.audit_path, operations)}
        assert _sha(originals[operations]) == audit["retained_mcp_audit"]["sha256"]
        # Qualify the actual evidence-directory protection separately from the
        # explicitly authored container/worker fixture boundary.
        with tempfile.TemporaryDirectory(prefix=".worker-receipts-observation-", dir=Path.home()) as copied:
            parent = Path(copied) / "worker-observations"
            parent.mkdir(mode=0o750)
            for path, contents in originals.items():
                retained = parent / path.name
                retained.write_bytes(contents)
                retained.chmod(0o400)
            result = observation.collect(parent=parent, scope=scope, report=report)
        assert result["status"] == "observed", result
        assert result["prepared_mcp_counts"]["tool_calls"] == 2
        assert result["prepared_mcp_counts"]["successful_reads"] == 1
        assert result["operation_correlation"]["counts"]["completed_socket_send_operations"] == 3
        assert result["operation_correlation"]["counts"]["completed_socket_send_successful_reads"] == 1
        forwarded = result["worker_forwarding_observation"]
        assert forwarded["session_count"] == 1
        if case == "published":
            assert forwarded["status"] == "observed", forwarded
            assert forwarded["known_session_count"] == 1 and forwarded["unknown_session_count"] == 0
            assert forwarded["counts"] == dict(reported_received_operations=3,
                reported_received_tool_results=2, reported_received_successful_reads=1,
                reported_completed_stdout_operations=3, reported_completed_stdout_tool_results=2,
                reported_completed_stdout_successful_reads=1, unconfirmed_stdout_operations=0)
        else:
            assert forwarded["status"] == "unavailable" and forwarded["counts"] is None
            assert forwarded["known_session_count"] == 0 and forwarded["unknown_session_count"] == 1
        assert forwarded["worker_reported_only"] is True
        for key in ("independent_worker_receipt_verified", "provider_ingestion_verified",
                    "output_payloads_independently_verified", "token_savings_measured", "proof_authority",
                    "execution_authority", "completion_authority", "production_route_qualified"):
            assert forwarded[key] is False
        assert result["delivered_tool_calls"] is result["delivered_reads"] is result["model_consumed_tokens"] is None
        assert result["worker_cleanup_independently_verified"] is False
        assert report == original_report
        assert {path: path.read_bytes() for path in originals} == originals
        assert not native_parent.released
        assert "value + 1" not in json.dumps(result) and str(owner.workspace) not in json.dumps(result)
        request.node.user_properties.append(("worker_receipt_collection", json.dumps(result, sort_keys=True)))
        request.node.user_properties.append(("worker_receipt_stdout_capture", json.dumps(dict(
            case=case, receipt_sha256=_sha(raw), receipt_bytes=len(raw), stdout_sha256=_sha(stdout),
            stdout_bytes=len(stdout), response_count=len(replies),
            line_sha256=[_sha(line) for line in lines], content_retained=False, fixture_only=True), sort_keys=True)))
        request.node.user_properties.append(("evidence_boundary",
            "New admitted native fixture; authored worker/container/router/task/cleanup identities "
            "and outbox UID projection; actual outbox FD/device/inode/mode/byte checks; "
            "real stdio publication and bounded worker snapshot; copied sealed owner evidence; "
            "no production handoff, helper quiescence, independent worker receipt or provider ingestion"))
    finally:
        if captured is None:
            handle.collect_and_close()
