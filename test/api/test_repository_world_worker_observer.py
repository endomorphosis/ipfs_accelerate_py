"""Finite protocol tests; authored owner/peer seams are not UID qualification.

These tests use real sockets and Popen/process observations for refusal paths.
Positive protocol units explicitly replace the native handoff and kernel peer
reader with authored metadata. Only root's separate container experiment can
qualify the actual 1000-to-1001 worker observation.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_observer as observer
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff


PROMPT = "Observe assignment metadata only. λ"
ARGV = (sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read()")


@pytest.fixture
def authored_unit_owner(monkeypatch):
    """A declared protocol seam, never a real native source/claim assertion."""
    birth = observer.process_identity(os.getpid())
    value = dict(schema="repository-world-readonly-worker-handoff@1", profile="selected_source_observation@1",
        state="TRANSITIONED", handoff_id="authored-handoff", original_assignment_cid="authored-assignment",
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), prompt_bytes=len(PROMPT.encode()),
        owner_pid=birth["pid"], owner_uid=birth["uid"], owner_start_time_ticks=birth["start_time_ticks"],
        boot_id=observer._boot_id(), worker_uid=os.getuid() + 1, worker_gid=os.getgid(),
        source_continuity_verified=True, native_assignment_continuity_verified=True,
        read_only_transition_verified=True, worker_process_verified=False, timeout_seconds=90.0,
        world_dispatch_ready=False, compiled_context_verified=False, proof_authority=False,
        execution_authority=False, completion_authority=False, provider_launch=False, omission_authority=False)
    handle = object.__new__(NativeRepositoryWorldWorkerHandoff)
    monkeypatch.setattr(NativeRepositoryWorldWorkerHandoff, "require_current", lambda _self: deepcopy(value))
    def observed(_self, *, prompt, worker_uid, worker_gid):
        assert _self is handle
        assert hashlib.sha256(prompt.encode() if type(prompt) is str else prompt).hexdigest() == value["prompt_sha256"]
        assert (worker_uid, worker_gid) == (value["worker_uid"], value["worker_gid"])
        return deepcopy(value)
    monkeypatch.setattr(NativeRepositoryWorldWorkerHandoff, "require_observer_assignment", observed)
    return handle, value


@pytest.fixture
def authored_broker(authored_unit_owner):
    handle, value = authored_unit_owner
    service = observer.RepositoryWorldWorkerObserver(handoff=handle, prompt=PROMPT,
        expected_worker_argv=ARGV, timeout_seconds=30)
    try:
        yield service, value
    finally:
        receipt = service.close()
        assert receipt["service_stopped"] is True
        assert all(receipt[k] is False for k in observer.FALSE_FIELDS)


@contextmanager
def child(argv=ARGV):
    process = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)


def request(service, **changes):
    return dict(schema=observer.REQUEST_SCHEMA, operation="observe_assignment",
        session_id=service.descriptor["session_id"], request_id=os.urandom(16).hex(),
        descriptor_sha256=service.descriptor_sha256) | changes


def raw_exchange(service, value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(10)
        channel.connect("\0" + service.descriptor["endpoint_name"])
        observer._send(channel, observer._wire(value, maximum=observer.MAX_REQUEST_BYTES))
        return observer._decode(observer._recv(channel, maximum=observer.MAX_FRAME_BYTES), maximum=observer.MAX_FRAME_BYTES)


def author_peer(monkeypatch, service):
    """Author only a protocol peer projection; no actual UID success claim."""
    peer = dict(pid=os.getpid(), uid=service.descriptor["worker_uid"], gid=os.getgid(),
        start_time_ticks=observer.process_identity(os.getpid())["start_time_ticks"],
        root_pid=service._root_child["pid"], root_start_time_ticks=service._root_child["start_time_ticks"],
        groups_sha256="a" * 64, group_count=1)
    monkeypatch.setattr(observer.RepositoryWorldWorkerObserver, "_peer", lambda _self, _channel: deepcopy(peer))
    return peer


def test_abstract_endpoint_descriptor_has_no_source_command_or_credentials(authored_broker):
    service, _value = authored_broker
    observed = service.descriptor
    assert observer.validate_worker_observer_descriptor(observed,
        expected_sha256=service.descriptor_sha256) == observed
    wire = observer._wire(observed)
    assert PROMPT.encode() not in wire and ARGV[-1].encode() not in wire
    assert observed["worker_uid"] != observed["owner_uid"]
    assert all(observed[k] is False for k in observer.FALSE_FIELDS)
    observed["handoff_id"] = "changed-copy"
    assert service.descriptor["handoff_id"] != observed["handoff_id"]


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf, float(1 << 64)])
def test_finite_numeric_wire_refuses_nonfinite_or_unbounded(number):
    with pytest.raises(ValueError):
        observer._wire({"value": number})


def test_finite_handoff_timeout_float_preserved_exactly():
    raw = observer._wire({"timeout_seconds": 90.0})
    assert observer._decode(raw, maximum=observer.MAX_FRAME_BYTES)["timeout_seconds"] == 90.0


@pytest.mark.parametrize("change", ["extra", "hash", "same_uid", "bool_counter", "expiry", "endpoint", "authority"])
def test_rehashed_descriptor_shape_and_principal_forgery_refused(authored_broker, change):
    service, _value = authored_broker
    descriptor = service.descriptor
    if change == "extra": descriptor["sql"] = "unavailable"
    elif change == "same_uid": descriptor["worker_uid"] = descriptor["owner_uid"]
    elif change == "bool_counter": descriptor["max_requests"] = True
    elif change == "expiry": descriptor["expires_monotonic_ns"] = descriptor["issued_monotonic_ns"]
    elif change == "endpoint": descriptor["endpoint_name"] = "/tmp/untrusted.sock"
    elif change == "authority": descriptor["execution_authority"] = True
    pin = "0" * 64 if change == "hash" else observer._sha(observer._wire(descriptor))
    with pytest.raises(ValueError):
        observer.validate_worker_observer_descriptor(descriptor, expected_sha256=pin)


def test_actual_socket_unregistered_worker_has_no_assignment(authored_broker):
    service, _value = authored_broker
    result = observer.observe_repository_world_worker_assignment(descriptor=service.descriptor,
        expected_descriptor_sha256=service.descriptor_sha256)
    assert result["reason_code"] == "worker_not_registered" and result["assignment"] is None
    assert result["worker_peer_authenticated"] is False


def test_actual_same_uid_peer_refused_after_genuine_child_registration(authored_broker):
    service, _value = authored_broker
    with child() as process:
        service.note_spawned(process)
        result = observer.observe_repository_world_worker_assignment(descriptor=service.descriptor,
            expected_descriptor_sha256=service.descriptor_sha256)
        assert result["reason_code"] == "peer_refused" and result["peer"] is None
        assert service._peer_binding is None


@pytest.mark.parametrize("bad", [None, object(), {"pid": 1}])
def test_json_or_fake_popen_never_registers_worker(authored_broker, bad):
    with pytest.raises(ValueError):
        authored_broker[0].note_spawned(bad)


def test_actual_wrong_literal_child_command_refused(authored_broker):
    service, _value = authored_broker
    with child((sys.executable, "-I", "-c", "import sys; sys.stdin.readline()")) as process:
        with pytest.raises(ValueError):
            service.note_spawned(process)
    assert service._process is None


def test_actual_second_or_exited_child_cannot_register(authored_broker):
    service, _value = authored_broker
    with child() as first, child() as second:
        service.note_spawned(first)
        with pytest.raises(ValueError):
            service.note_spawned(second)
        first.terminate()
        first.wait(timeout=2)
        result = raw_exchange(service, request(service))
        assert result["ok"] is False and result["assignment"] is None


@pytest.mark.parametrize("change", ["operation", "extra", "schema", "session", "pin", "request_id"])
def test_closed_operations_fail_before_owner_observation(authored_broker, monkeypatch, change):
    service, _value = authored_broker
    value = request(service)
    if change == "extra": value["sql"] = "unavailable"
    elif change == "operation": value["operation"] = "source.get"
    elif change == "schema": value["schema"] = "future"
    elif change == "session": value["session_id"] = "0" * 32
    elif change == "pin": value["descriptor_sha256"] = "0" * 64
    else: value["request_id"] = "bad"
    def forbidden(_self):
        pytest.fail("invalid protocol must not reobserve native owner")
    monkeypatch.setattr(observer.RepositoryWorldWorkerObserver, "_handoff_current", forbidden)
    result = raw_exchange(service, value)
    assert result["reason_code"] == "invalid_request" and result["assignment"] is None


def test_authored_positive_protocol_and_replay_do_not_qualify_actual_uid(authored_broker, monkeypatch):
    service, _value = authored_broker
    with child() as process:
        service.note_spawned(process)
        peer = author_peer(monkeypatch, service)
        value = request(service)
        result = raw_exchange(service, value)
        assert result["ok"] is True and result["peer"] == peer
        assert result["assignment"]["worker_process_verified"] is False
        assert all(result[k] is False for k in observer.FALSE_FIELDS)
        replay = raw_exchange(service, value)
        assert replay["reason_code"] == "replayed_request"
        audit = service.close()
        assert audit["sent_response_wire_bytes"] == audit["sent_response_bytes"] + 4 * audit["sent_responses"]


def test_authored_native_change_during_send_preparation_refuses(authored_broker, monkeypatch):
    service, value = authored_broker
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        original = observer._wire
        def changed(item, **kwargs):
            raw = original(item, **kwargs)
            if type(item) is dict and item.get("schema") == observer.RESPONSE_SCHEMA and item.get("ok"):
                value["native_assignment_continuity_verified"] = False
            return raw
        monkeypatch.setattr(observer, "_wire", changed)
        response = raw_exchange(service, request(service))
        assert response["reason_code"] == "observation_refused"
        assert response["assignment"] is None and service._peer_binding is None


def test_expiry_refuses_without_reobserving_owner(authored_broker, monkeypatch):
    service, _value = authored_broker
    monkeypatch.setattr(observer.time, "monotonic_ns", lambda: service.descriptor["expires_monotonic_ns"] + 1)
    result = service._answer(request(service), None)
    assert result["reason_code"] == "session_expired" and result["assignment"] is None


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b' {"a":1}', b'{"a":true}\n'])
def test_noncanonical_or_duplicate_input_is_refused(raw):
    with pytest.raises(ValueError):
        observer._decode(raw, maximum=observer.MAX_REQUEST_BYTES)


def test_oversized_frame_rejected_before_body_receive():
    first, second = socket.socketpair()
    try:
        first.sendall((observer.MAX_REQUEST_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(ValueError):
            observer._recv(second, maximum=observer.MAX_REQUEST_BYTES)
    finally:
        first.close()
        second.close()


def test_complete_request_population_is_bounded(authored_broker):
    service, _value = authored_broker
    for _index in range(observer.MAX_REQUESTS):
        assert raw_exchange(service, request(service))["ok"] is False
    audit = service.close()
    assert audit["accepted_connections"] == observer.MAX_REQUESTS
    assert audit["sent_response_wire_bytes"] <= observer.MAX_TOTAL_RESPONSE_BYTES


def test_closed_client_helper_reports_no_exception_content(monkeypatch):
    class Streams:
        def __init__(self, raw): self.buffer = io.BytesIO(raw)
    output = Streams(b"")
    monkeypatch.setattr(observer.sys, "stdin", Streams(b'{"sql":"private-text"}\n'))
    monkeypatch.setattr(observer.sys, "stdout", output)
    assert observer.worker_observer_client_main() == 1
    value = json.loads(output.buffer.getvalue())
    assert value["reason_code"] == "client_refused"
    assert b"private-text" not in output.buffer.getvalue()
    assert all(value[k] is False for k in observer.FALSE_FIELDS)
