"""Native owner correlation with authored identity and failure boundaries.

These tests require the existing admitted runner and borrowed native parent.
The source service, audit writes and socket traffic are real. Container/peer
identities come from the existing authored fixture; injected failures and large
response padding establish no production deployment or provider consumption.
"""
import hashlib
import json
import socket
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from test.api.test_repository_source_owner_broker import (
    owner, native_parent, short_service_directory, exchange, initialize, message, tool,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _request(owner, method, *, identity=1, params=None):
    return protocol.make_request(owner.descriptor["session_id"],
                                 message(method, identity=identity, params=params))


def _call(owner, name, **arguments):
    return _request(owner, "tools/call", params={"name": name, "arguments": arguments})


def _closed(owner, *, expected):
    # Receipt of socket bytes precedes the owner's final send cancellation
    # check. Wait for its actual accounting boundary before a normal close can
    # set the stop signal. Injected send failures still retain their real status.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with owner._accounting_lock:
            if (owner._inflight is None and owner._accepted == expected
                    and len(owner._records) == expected):
                break
        time.sleep(0.005)
    else:
        raise AssertionError("owner response accounting did not complete before fixture close")
    audit = owner.close()
    assert audit["schema"] == "repository-source-owner-session@2"
    assert audit["service_stopped"] and audit["cleanup_complete"] and audit["accounting_complete"]
    retained = audit["retained_mcp_audit"]
    raw = (owner.audit_path.parent / retained["name"]).read_bytes()
    assert retained["sha256"] == _sha(raw) and retained["bytes"] == len(raw)
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    witnesses = [{"sequence": row["sequence"], "sha256": _sha(line), "bytes": len(line)}
                 for row, line in zip(rows, lines)]
    assert all(line.endswith(b"\n") for line in lines)
    observed = []
    for record in audit["requests"]:
        capture = record["operation_capture"]
        assert set(capture) == {"status", "operations"}
        assert capture["status"] in {"not_entered", "complete", "incomplete"}
        assert len(capture["operations"]) <= 1
        if capture["status"] == "not_entered":
            assert capture["operations"] == []
        for witness in capture["operations"]:
            assert witness == witnesses[witness["sequence"] - 1]
            observed.append(witness)
    # These scenarios have intact audit capture even when response transmission
    # fails. The assertion compares the actual retained rows, not request order.
    assert observed == witnesses
    assert audit["cross_uid_deployment_qualified"] is False
    assert audit["provider_calls"] == 0 and audit["token_savings_measured"] is False
    assert b"value + 1" not in raw
    return audit, rows


def _matched(record, request):
    assert record["request_sha256"] == _sha(protocol.canonical_json(request))
    assert record["operation_capture"]["status"] == "complete"
    assert len(record["operation_capture"]["operations"]) == 1


def test_native_initialize_read_and_tool_error_bind_exact_retained_rows(owner, native_parent):
    initialize(owner)
    generation = tool(owner, "repository_status")["generation"]
    read = _call(owner, "repository_read", generation=generation, path="calc.py")
    reply = exchange(owner, None, request=read)
    assert reply["ok"] and not reply["message"]["result"]["isError"]
    missing = _call(owner, "repository_read", generation=generation, path="missing.py")
    failure = exchange(owner, None, request=missing)
    # Outer transport success is distinct from native MCP tool success.
    assert failure["ok"] and failure["message"]["result"]["isError"]
    audit, rows = _closed(owner, expected=5)
    assert [row["event"] for row in rows] == ["initialize", "tool", "tool", "tool"]
    assert [row["success"] for row in rows] == [True, True, True, False]
    assert audit["requests"][0]["operation_capture"]["operations"][0]["sequence"] == 1
    for record, request in zip(audit["requests"][-2:], (read, missing)):
        _matched(record, request)
        assert record["delivered"] and record["status"] == "ok"
        assert record["prepared_response_sha256"] == record["response_sha256"]
        assert record["prepared_response_bytes"] == record["response_bytes"]
    assert not native_parent.released


def test_ping_list_and_notifications_have_complete_empty_capture(owner):
    initialize(owner)
    requests = [_request(owner, "ping"), _request(owner, "tools/list"),
                _request(owner, "notifications/initialized", identity=None), _request(owner, "ping")]
    for request in requests:
        assert exchange(owner, None, request=request)["ok"]
    audit, rows = _closed(owner, expected=6)
    assert len(rows) == 1 and rows[0]["event"] == "initialize"
    for record in audit["requests"][1:]:
        assert record["operation_capture"] == {"status": "complete", "operations": []}
        assert record["delivered"]
    assert [record["request_sha256"] for record in audit["requests"][-4:]] == [
        _sha(protocol.canonical_json(request)) for request in requests]


def test_replayed_request_has_no_second_operation_witness(owner):
    initialize(owner)
    request = _call(owner, "repository_status")
    assert exchange(owner, None, request=request)["ok"]
    refusal = exchange(owner, None, request=request)
    assert refusal["ok"] is False and refusal["reason_code"] == "invalid_request"
    audit, rows = _closed(owner, expected=4)
    first, replay = audit["requests"][-2:]
    _matched(first, request)
    assert first["request_sha256"] == replay["request_sha256"]
    assert replay["ordinal"] == first["ordinal"] + 1
    assert replay["operation_capture"] == {"status": "not_entered", "operations": []}
    assert replay["delivered"] and len(rows) == 2


def test_malformed_request_cannot_inherit_next_handler_capture(owner):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner.socket_path))
        protocol.send_frame(channel, {"unrecognized": True}, deadline_monotonic=time.monotonic() + 2)
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.receive_frame(channel, deadline_monotonic=time.monotonic() + 2)
    assert exchange(owner, message("ping"))["ok"]
    audit, rows = _closed(owner, expected=2)
    malformed, ping = audit["requests"]
    assert malformed["request_sha256"] is None and malformed["delivered"] is False
    assert malformed["operation_capture"] == {"status": "not_entered", "operations": []}
    assert ping["operation_capture"] == {"status": "complete", "operations": []}
    assert rows == []


def test_real_budget_substitution_preserves_operation_without_claiming_its_response_sent(owner, monkeypatch):
    initialize(owner)
    original = owner._server.session.status_summary

    def padded_status():
        # Authored payload padding exercises the genuine cumulative byte limit;
        # source files and the broker's reservation counter remain untouched.
        return {**original(), "authored_padding": "x" * 300000}

    monkeypatch.setattr(owner._server.session, "status_summary", padded_status)
    first = _call(owner, "repository_status")
    second = _call(owner, "repository_status")
    assert exchange(owner, None, request=first)["ok"]
    refusal = exchange(owner, None, request=second)
    assert refusal["ok"] is False and refusal["reason_code"] == "budget_exhausted"
    audit, rows = _closed(owner, expected=4)
    record = audit["requests"][-1]
    _matched(record, second)
    assert record["delivered"] and record["status"] == "budget_exhausted"
    assert record["prepared_response_sha256"] != record["response_sha256"]
    assert record["prepared_response_bytes"] > record["response_bytes"]
    assert len(rows) == 3 and rows[-1]["success"] is True
    assert audit["response_bytes"] == sum(row["response_bytes"] for row in audit["requests"])
    assert audit["response_bytes"] <= broker.MAX_TOTAL_RESPONSE_BYTES


@pytest.mark.parametrize("failure", ["context", "peer", "cancellation"])
def test_prepared_operation_survives_failure_after_handler_without_response(owner, monkeypatch, failure):
    initialize(owner)
    original = owner._server.handle

    def handled(request):
        reply = original(request)
        if failure == "context":
            owner._boundary_sha256 = "c" * 64
        elif failure == "peer":
            def refused(channel):
                raise ValueError("authored post-handler peer refusal")
            monkeypatch.setattr(owner, "_peer", refused)
        else:
            owner._stop.set()
        return reply

    monkeypatch.setattr(owner._server, "handle", handled)
    request = _call(owner, "repository_status")
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        exchange(owner, None, request=request)
    audit, rows = _closed(owner, expected=3)
    record = audit["requests"][-1]
    _matched(record, request)
    assert record["prepared_response_sha256"] is None and record["response_sha256"] is None
    assert record["delivered"] is False and record["response_bytes"] == 0
    assert len(rows) == 2 and rows[-1]["success"] is True


@pytest.mark.parametrize("failure", ["before_write", "partial_write", "after_full_write"])
def test_failed_frame_send_retains_exact_prepared_operation(owner, monkeypatch, failure):
    initialize(owner)
    original = protocol.send_frame

    def failed_send(channel, value, **kwargs):
        if value.get("schema") != protocol.RESPONSE_SCHEMA:
            return original(channel, value, **kwargs)
        if failure == "partial_write":
            raw = protocol.canonical_json(value)
            channel.sendall(len(raw).to_bytes(4, "big") + raw[:8])
        elif failure == "after_full_write":
            original(channel, value, **kwargs)
        raise protocol.RepositorySourceOwnerProtocolError(
            "cancelled" if failure == "after_full_write" else "transport_error")

    monkeypatch.setattr(protocol, "send_frame", failed_send)
    request = _call(owner, "repository_status")
    if failure == "after_full_write":
        # The client can receive bytes even when the owner's send never returns
        # successfully. Its observation must remain conservative in this case.
        assert exchange(owner, None, request=request)["ok"]
    else:
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            exchange(owner, None, request=request)
    audit, rows = _closed(owner, expected=3)
    record = audit["requests"][-1]
    _matched(record, request)
    assert record["delivered"] is False
    assert record["prepared_response_sha256"] == record["response_sha256"]
    assert record["prepared_response_bytes"] == record["response_bytes"] > 0
    assert record["status"] == ("cancelled" if failure == "after_full_write" else "transport_error")
    assert len(rows) == 2


def test_consecutive_handlers_do_not_reuse_operation_capture(owner):
    initialize(owner)
    generation = tool(owner, "repository_status")["generation"]
    first = _call(owner, "repository_read", generation=generation, path="calc.py")
    gap = _request(owner, "ping")
    second = _call(owner, "repository_read", generation=generation, path="calc.py")
    for request in (first, gap, second):
        assert exchange(owner, None, request=request)["ok"]
    audit, rows = _closed(owner, expected=6)
    one, empty, two = audit["requests"][-3:]
    _matched(one, first)
    _matched(two, second)
    assert empty["operation_capture"] == {"status": "complete", "operations": []}
    left, right = one["operation_capture"]["operations"][0], two["operation_capture"]["operations"][0]
    assert left["sequence"] + 1 == right["sequence"]
    assert left["sha256"] != right["sha256"]
    assert len(rows) == 4
