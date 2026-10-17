"""Authored wire contracts and real bounded socket/pipe I/O; no native allocator."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol


@pytest.fixture
def descriptor():
    now = time.monotonic_ns()
    return dict(schema=protocol.DESCRIPTOR_SCHEMA, session_id="1" * 32,
        owner_uid=1001, worker_uid=1002, owner_pid=1234, owner_start_time_ticks=100,
        workspace="/allocated/repository", boundary_manifest_sha256="a" * 64,
        socket_path="/protected/owner/source.sock", socket_device=1, socket_inode=2,
        issued_monotonic_ns=now, expires_monotonic_ns=now + 60_000_000_000,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False,
        representation="compact", config_sha256="b" * 64, helper_sha256="c" * 64, read_only=True)


def request(message=None):
    return protocol.make_request("1" * 32,
        {"jsonrpc": "2.0", "id": 7, "method": "ping"} if message is None else message,
        request_id="2" * 32)


def response(req=None, message=None):
    req = request() if req is None else req
    return dict(schema=protocol.RESPONSE_SCHEMA, session_id=req["session_id"],
        request_id=req["request_id"], ok=True,
        message={"jsonrpc": "2.0", "id": 7, "result": {}} if message is None else message,
        reason_code=None)


def test_descriptor_has_only_public_identity_and_no_authority(descriptor):
    assert protocol.validate_descriptor(descriptor) is descriptor
    assert not {"parent_token", "index_path", "state_dir", "task_id", "task_cid"} & set(descriptor)
    assert protocol.DESCRIPTOR_FLAG == "--repository-source-owner-descriptor"
    assert protocol.DIGEST_FLAG == "--repository-source-owner-descriptor-sha256"


@pytest.mark.parametrize("key,value", [
    ("owner_uid", 1002), ("worker_uid", 0), ("owner_pid", True),
    ("read_only", False), ("read_only", 1), ("representation", "minified"),
    ("config_sha256", "B" * 64), ("helper_sha256", "c" * 63),
    ("boundary_manifest_sha256", None), ("proof_authority", True),
    ("execution_authority", True), ("completion_authority", 0),
    ("max_requests", 65), ("max_total_response_bytes", 524289),
    ("workspace", "/allocated/../private"), ("socket_path", "relative.sock"),
])
def test_bad_descriptor_refused(descriptor, key, value):
    descriptor[key] = value
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="peer_refused"):
        protocol.validate_descriptor(descriptor)


@pytest.mark.parametrize("field", ["parent_token", "state_dir", "task_id", "context_receipt_sha256"])
def test_extra_descriptor_authority_or_private_fields_refused(descriptor, field):
    descriptor[field] = "not-a-capability"
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_descriptor(descriptor)


def test_descriptor_loader_uses_existing_owner_checks(monkeypatch, descriptor):
    seen = []
    def load(path, digest, *, workspace, validator):
        seen.append((path, digest, workspace, validator))
        return validator(descriptor)
    monkeypatch.setattr(protocol._base, "load_descriptor", load)
    assert protocol.load_descriptor("/owner/descriptor.json", "a" * 64,
                                    workspace="/allocated/repository") == descriptor
    assert seen == [("/owner/descriptor.json", "a" * 64, "/allocated/repository", protocol.validate_descriptor)]


def test_real_kernel_identity():
    facts = protocol.process_identity(os.getpid())
    assert facts["pid"] == os.getpid() and facts["uid"] == os.geteuid()
    assert facts["start_time_ticks"] > 0


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
    b'{"a":1e999}', b'{"a":"\\ud800"}', b'\xff', b'{"a":', b'{}{}',
    b'[' * 22 + b'0' + b']' * 22])
def test_strict_json_refuses_ambiguous_or_unbounded_values(raw):
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.strict_json_loads(raw)


@pytest.mark.parametrize("value", [{1: "implicit-key"}, {"a": (1, 2)}, {"a": float("nan")},
    {"a": object()}, {"a": "\ud800"}])
def test_canonical_json_never_silently_coerces(value):
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.canonical_json(value)


def test_unicode_byte_limits_are_bytes():
    raw = protocol.canonical_json({"source": "é" * 10})
    assert len(raw) > len(raw.decode())
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.strict_json_loads(raw, max_bytes=len(raw) - 1)
    assert protocol.strict_json_loads(raw, max_bytes=len(raw)) == {"source": "é" * 10}


def test_json_node_limit_and_cycles_closed():
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.canonical_json([None] * protocol.MAX_JSON_NODES)
    value = []
    value.append(value)
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.canonical_json(value)


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
    {"jsonrpc": "2.0", "id": "ping", "method": "ping", "params": {}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": {}}},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "repository_status"}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "repository_read", "arguments": {"generation": 1, "path": "src/code.py", "representation": "compact"}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "repository_refresh", "arguments": {"generation": 1}}},
])
def test_bounded_mcp_subset_admitted(message):
    assert request(message)["message"] == message


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "id": 1, "method": "resources/read"},
    {"jsonrpc": "2.0", "method": "ping"},
    {"jsonrpc": "2.0", "id": True, "method": "ping"},
    {"jsonrpc": "2.0", "id": None, "method": "ping"},
    {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"command": "ignored"}},
    {"jsonrpc": "2.0", "id": 1, "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {"extra": True}},
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": "unbounded"}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "repository_replace_symbol", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "repository_status", "arguments": {"workspace": "/private"}}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "repository_read", "arguments": {"generation": True, "path": "source.py"}}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "repository_read", "arguments": {"generation": 1, "path": "../private"}}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "repository_context_query", "arguments": {"generation": 1, "query": "test", "max_bytes": 65537}}},
])
def test_unsupported_or_mutating_mcp_requests_closed(message):
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        request(message)


def test_request_envelope_is_closed_and_size_bounded():
    value = request()
    value["parent_token"] = "not-a-token"
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_request(value)
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {
        "_meta": {"payload": "x" * protocol.MAX_MESSAGE_REQUEST_BYTES}}}
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        request(message)


def test_response_matches_request_and_descriptor(descriptor):
    value = response()
    assert protocol.validate_response(value, request=request(), descriptor=descriptor) == value


@pytest.mark.parametrize("field,value", [("schema", "other"), ("session_id", "3" * 32),
    ("request_id", "3" * 32), ("ok", 1), ("reason_code", "arbitrary"), ("message", None)])
def test_response_mismatch_refused(field, value):
    current = response()
    current[field] = value
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(current, request=request())


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "id": "7", "result": {}},
    {"jsonrpc": "2.0", "id": 7, "result": {}, "extra": 1},
    {"jsonrpc": "2.0", "id": 7, "result": [], "error": {}},
    {"jsonrpc": "2.0", "id": 7, "error": {"code": True, "message": "error"}},
])
def test_invalid_nested_jsonrpc_response_closed(message):
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(response(message=message), request=request())


def test_notifications_have_only_null_success_response():
    req = request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    value = response(req)
    value["message"] = None
    assert protocol.validate_response(value, request=req) == value
    value["message"] = {"jsonrpc": "2.0", "id": 7, "result": {}}
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(value, request=req)


def test_refusals_are_closed_and_do_not_carry_source():
    value = response()
    value.update(ok=False, message=None, reason_code="budget_exhausted")
    assert protocol.validate_response(value, request=request()) == value
    value["message"] = {"source": "should-not-deliver"}
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(value, request=request())


def test_tools_list_cannot_advertise_write_or_duplicate_tools():
    req = request({"jsonrpc": "2.0", "id": 7, "method": "tools/list"})
    for tools in ([{"name": "repository_replace_symbol"}],
                  [{"name": "repository_status"}, {"name": "repository_status"}]):
        value = response(req, {"jsonrpc": "2.0", "id": 7, "result": {"tools": tools}})
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.validate_response(value, request=req)


def test_response_aggregate_budget_is_not_individual_large_frame_budget(descriptor):
    value = response(message={"jsonrpc": "2.0", "id": 7, "result": {"text": "x" * 524288}})
    assert protocol.validate_response(value, request=request()) == value
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(value, request=request(), descriptor=descriptor)
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        protocol.validate_response(response(), request=request(), max_bytes=32)


@contextmanager
def transport(kind):
    if kind == "socket":
        receiver, sender = socket.socketpair()
    else:
        read_fd, write_fd = os.pipe()
        receiver, sender = os.fdopen(read_fd, "rb", buffering=0), os.fdopen(write_fd, "wb", buffering=0)
    try:
        yield receiver, sender
    finally:
        receiver.close()
        sender.close()


def start_reader(channel, *, deadline=None, cancel=None, max_bytes=protocol.MAX_FRAME_BYTES):
    outcomes = []
    def read():
        try:
            outcomes.append(protocol.receive_frame(channel,
                deadline_monotonic=time.monotonic() + 1 if deadline is None else deadline,
                cancel_event=cancel, max_bytes=max_bytes))
        except BaseException as exc:
            outcomes.append(exc)
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return thread, outcomes


def join(thread):
    thread.join(3)
    assert not thread.is_alive(), "frame operation exceeded its absolute deadline"


@pytest.mark.parametrize("kind", ["socket", "pipe"])
def test_actual_large_frames_roundtrip_without_truncation_and_restore_modes(kind):
    value = {"source": "é" * 40000}
    with transport(kind) as (receiver, sender):
        thread, outcomes = start_reader(receiver)
        try:
            count = protocol.send_frame(sender, value, deadline_monotonic=time.monotonic() + 1)
        finally:
            join(thread)
        assert outcomes == [value]
        assert count == len(protocol.canonical_json(value))
        assert os.get_blocking(receiver.fileno()) and os.get_blocking(sender.fileno())


@pytest.mark.parametrize("kind", ["socket", "pipe"])
@pytest.mark.parametrize("partial", [b"", b"\x00\x00", b"\x00\x00\x00\x10{"])
def test_idle_or_partial_frames_observe_cancellation(kind, partial):
    signal = threading.Event()
    with transport(kind) as (receiver, sender):
        if partial:
            os.write(sender.fileno(), partial)
        thread, outcomes = start_reader(receiver, cancel=signal)
        signal.set()
        join(thread)
        assert len(outcomes) == 1 and isinstance(outcomes[0], protocol.RepositorySourceOwnerProtocolError)
        assert outcomes[0].reason_code == "cancelled"
        assert os.get_blocking(receiver.fileno())


@pytest.mark.parametrize("kind", ["socket", "pipe"])
def test_idle_deadline_ends_without_another_request(kind):
    with transport(kind) as (receiver, _sender):
        deadline = time.monotonic() + .05
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="deadline_exceeded"):
            protocol.receive_frame(receiver, deadline_monotonic=deadline)
        assert time.monotonic() < deadline + 1
        assert os.get_blocking(receiver.fileno())


def test_partial_traffic_does_not_restart_absolute_deadline():
    with transport("socket") as (receiver, sender):
        deadline = time.monotonic() + .15
        thread, outcomes = start_reader(receiver, deadline=deadline)
        os.write(sender.fileno(), (1000).to_bytes(4, "big"))
        for _ in range(8):
            if not thread.is_alive():
                break
            os.write(sender.fileno(), b" ")
            time.sleep(.03)
        join(thread)
        assert len(outcomes) == 1 and outcomes[0].reason_code == "deadline_exceeded"


@pytest.mark.parametrize("kind", ["socket", "pipe"])
def test_blocked_output_cancels_without_background_writer(kind):
    signal = threading.Event()
    with transport(kind) as (_receiver, sender):
        os.set_blocking(sender.fileno(), False)
        try:
            while True:
                os.write(sender.fileno(), b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(sender.fileno(), True)
        timer = threading.Timer(.05, signal.set)
        timer.start()
        try:
            with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="cancelled"):
                protocol.send_frame(sender, {"value": 1}, deadline_monotonic=time.monotonic() + 1,
                                    cancel_event=signal)
        finally:
            timer.join(1)
        assert os.get_blocking(sender.fileno())


def test_oversized_send_never_writes_a_truncated_frame():
    with transport("socket") as (receiver, sender):
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.send_frame(sender, {"source": "x" * 100}, max_bytes=16,
                                deadline_monotonic=time.monotonic() + 1)
        receiver.setblocking(False)
        with pytest.raises(BlockingIOError):
            receiver.recv(1)


@pytest.mark.parametrize("raw", [b'{ "a":1}', b'{"a":1,"a":2}', b'[]', b'{"b":1,"a":2}'])
def test_frame_requires_canonical_object_bytes(raw):
    with transport("socket") as (receiver, sender):
        os.write(sender.fileno(), len(raw).to_bytes(4, "big") + raw)
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.receive_frame(receiver, deadline_monotonic=time.monotonic() + 1)


@pytest.mark.parametrize("length", [0, 1, protocol.MAX_FRAME_BYTES + 1, (1 << 32) - 1])
def test_oversized_header_refused_before_body(length):
    with transport("socket") as (receiver, sender):
        os.write(sender.fileno(), length.to_bytes(4, "big"))
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="invalid_request"):
            protocol.receive_frame(receiver, deadline_monotonic=time.monotonic() + 1)


def test_incomplete_frame_closed():
    with transport("socket") as (receiver, sender):
        os.write(sender.fileno(), b"\x00\x00")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="transport_error"):
            protocol.receive_frame(receiver, deadline_monotonic=time.monotonic() + 1)


@pytest.mark.parametrize("value", [None, True, "10", float("nan"), float("inf"), 10**400])
def test_invalid_deadline_never_touches_channel(value):
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="deadline_exceeded"):
        protocol.receive_frame(object(), deadline_monotonic=value)


def test_deadline_caps_each_frame_at_two_seconds(monkeypatch):
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: 10.0))
    assert protocol._deadline(1000.0) == 12.0
    assert protocol._deadline(10.5) == 10.5


def test_cancelled_expired_exchange_prefers_cancellation():
    signal = threading.Event()
    signal.set()
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="cancelled"):
        protocol.receive_frame(object(), deadline_monotonic=time.monotonic() - 1, cancel_event=signal)


def test_regular_files_are_not_protocol_channels(tmp_path):
    path = tmp_path / "not-a-pipe"
    path.write_bytes(b"{}")
    with path.open("rb") as stream:
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="transport_error"):
            protocol.receive_frame(stream, deadline_monotonic=time.monotonic() + 1)


def test_direct_isolated_import_never_loads_the_accelerator_package():
    source = str(Path(protocol.__file__).resolve())
    script = (
        "import importlib.util,sys;"
        "s=importlib.util.spec_from_file_location('owner_wire',sys.argv[1]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert 'ipfs_accelerate_py' not in sys.modules;"
        "assert m.make_request('1'*32,{'jsonrpc':'2.0','id':1,'method':'ping'})['schema']==m.REQUEST_SCHEMA"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, source],
                            capture_output=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b""


def test_delayed_response_start_does_not_spend_active_frame_budget(monkeypatch):
    monkeypatch.setattr(protocol, "MAX_FRAME_SECONDS", .05)
    with transport("socket") as (receiver, sender):
        value = {"result": "ready"}
        raw = protocol.canonical_json(value)
        deadline = time.monotonic() + 1
        timer = threading.Timer(.15, lambda: os.write(sender.fileno(), len(raw).to_bytes(4, "big") + raw))
        timer.start()
        try:
            protocol._wait_for_response_start(receiver, deadline_monotonic=deadline)
            assert protocol.receive_frame(receiver, deadline_monotonic=deadline) == value
        finally:
            timer.join(2)


def test_response_start_wait_observes_cancellation():
    signal = threading.Event()
    with transport("socket") as (receiver, _sender):
        timer = threading.Timer(.05, signal.set)
        timer.start()
        try:
            with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="cancelled"):
                protocol._wait_for_response_start(receiver, deadline_monotonic=time.monotonic() + 1,
                                                 cancel_event=signal)
        finally:
            timer.join(2)
        assert os.get_blocking(receiver.fileno())


def test_response_start_wait_observes_caller_deadline():
    with transport("socket") as (receiver, _sender):
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError, match="deadline_exceeded"):
            protocol._wait_for_response_start(receiver, deadline_monotonic=time.monotonic() + .05)


def test_response_start_budget_is_separate_and_bounded(monkeypatch):
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: 10.0))
    assert protocol._deadline(1000.0, max_seconds=protocol.MAX_RPC_SECONDS) == 40.0
    assert protocol._deadline(10.5, max_seconds=protocol.MAX_RPC_SECONDS) == 10.5


def test_maximum_response_frame_roundtrips_exact_bytes():
    overhead = len(protocol.canonical_json({"source": ""}))
    value = {"source": "x" * (protocol.MAX_RESPONSE_FRAME_BYTES - overhead)}
    with transport("socket") as (receiver, sender):
        thread, outcomes = start_reader(receiver, deadline=time.monotonic() + 2)
        try:
            count = protocol.send_frame(sender, value, deadline_monotonic=time.monotonic() + 2,
                                        max_bytes=protocol.MAX_RESPONSE_FRAME_BYTES)
        finally:
            join(thread)
        assert count == protocol.MAX_RESPONSE_FRAME_BYTES and outcomes == [value]
