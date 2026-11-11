"""Bounded delivery protocol tests; authored seams do not qualify worker UID.

Real socket/Popen negatives exercise refusal. Positive units explicitly replace
the native binding and peer projection; actual cross-UID delivery is qualified
by the separate owner/container experiment, never by these fixtures.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_observer as metadata
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_worker_packet_delivery as delivery
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_packet import NativeRepositoryWorldCompiledPacketBinding


TASK = '{"body":"Original native assignment — λ"}'
PACKET = "Compiled packet guidance.\n" + json.dumps({"body": "Café § 12. " + "x" * 12000}, ensure_ascii=False)
ARGV = (sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read()")


@pytest.fixture
def authored_protocol_owners(monkeypatch):
    birth = metadata.process_identity(os.getpid())
    handoff = object.__new__(NativeRepositoryWorldWorkerHandoff)
    handoff_info = dict(schema="repository-world-readonly-worker-handoff@1", profile="selected_source_observation@1",
        state="TRANSITIONED", handoff_id="authored-handoff", original_assignment_cid="authored-assignment",
        prompt_sha256=delivery._sha(TASK.encode()), prompt_bytes=len(TASK.encode()), owner_pid=birth["pid"],
        owner_uid=birth["uid"], owner_start_time_ticks=birth["start_time_ticks"], boot_id=metadata._boot_id(),
        worker_uid=os.getuid() + 1, worker_gid=os.getgid(), source_continuity_verified=True,
        native_assignment_continuity_verified=True, read_only_transition_verified=True, worker_process_verified=False,
        timeout_nanoseconds=90_000_000_000, world_dispatch_ready=False, compiled_context_verified=False,
        proof_authority=False, execution_authority=False, completion_authority=False, provider_launch=False, omission_authority=False)
    monkeypatch.setattr(NativeRepositoryWorldWorkerHandoff, "require_current", lambda _self: deepcopy(handoff_info))
    def observe(_self, *, prompt, worker_uid, worker_gid):
        assert _self is handoff and prompt == TASK
        assert (worker_uid, worker_gid) == (handoff_info["worker_uid"], handoff_info["worker_gid"])
        return deepcopy(handoff_info)
    monkeypatch.setattr(NativeRepositoryWorldWorkerHandoff, "require_observer_assignment", observe)
    binding = object.__new__(NativeRepositoryWorldCompiledPacketBinding)
    info = dict(schema="repository-world-compiled-worker-packet-binding@1", profile="native_structural_readonly_packet@1",
        binding_id="authored-binding", packet_id="authored-packet", context_cid="authored-context",
        prompt_sha256=delivery._sha(PACKET.encode()), prompt_bytes=len(PACKET.encode()), handoff_id="authored-handoff",
        original_assignment_cid="authored-assignment", task_cid="authored-native-task", task_revision=1,
        expected_task_body_cid="authored-native-body", native_task_relations_cid="authored-native-relations",
        structural_compiler_context_current=True, native_task_projection_current=True, native_assignment_continuity_verified=True,
        compiler_signature_verified=False, semantic_alignment_verified=False, world_dispatch_ready=False,
        provider_launch=False, provider_ingestion_verified=False, proof_authority=False, execution_authority=False,
        completion_authority=False, omission_authority=False, worker_edit_authority=False)
    state = {"information": info, "text": PACKET, "current": True, "calls": 0}
    def material(_self, *, handoff: object, original_prompt):
        assert _self is binding and handoff is owners[0] and original_prompt == TASK
        state["calls"] += 1
        if not state["current"]:
            raise ValueError("authored_native_assignment_changed")
        return deepcopy(state["information"]), state["text"]
    owners = handoff, binding
    monkeypatch.setattr(NativeRepositoryWorldCompiledPacketBinding, "delivery_material", material)
    return owners, handoff_info, state


@pytest.fixture
def authored_service(authored_protocol_owners):
    (handoff, binding), handoff_info, state = authored_protocol_owners
    observer = metadata.RepositoryWorldWorkerObserver(handoff=handoff, prompt=TASK, expected_worker_argv=ARGV,
        timeout_seconds=30)
    service = delivery.RepositoryWorldWorkerPacketDelivery(observer=observer, binding=binding, timeout_seconds=30)
    try:
        yield service, observer, handoff_info, state
    finally:
        assert service.close()["service_stopped"] is True
        assert observer.close()["service_stopped"] is True


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


def author_peer(monkeypatch, service):
    peer = dict(pid=os.getpid(), uid=service.descriptor["worker_uid"], gid=os.getgid(),
        start_time_ticks=metadata.process_identity(os.getpid())["start_time_ticks"],
        root_pid=service._root_child["pid"], root_start_time_ticks=service._root_child["start_time_ticks"],
        groups_sha256="a" * 64, group_count=1)
    monkeypatch.setattr(metadata.RepositoryWorldWorkerObserver, "_peer", lambda _self, _channel: deepcopy(peer))
    return peer


def request(service, **changes):
    return dict(schema=delivery.REQUEST_SCHEMA, operation="get_packet", session_id=service.descriptor["session_id"],
        request_id=os.urandom(16).hex(), descriptor_sha256=service.descriptor_sha256) | changes


def acknowledge(service, get_request_id, **changes):
    return request(service, operation="ack_packet", get_request_id=get_request_id,
        **{k: service.descriptor[k] for k in ("prompt_sha256", "prompt_bytes", "packet_id", "context_cid")}) | changes


def exchange(service, value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(10)
        channel.connect("\0" + service.descriptor["endpoint_name"])
        delivery._send(channel, delivery._wire(value, maximum=delivery.MAX_REQUEST_BYTES), maximum=delivery.MAX_REQUEST_BYTES)
        return delivery._decode(delivery._recv(channel, maximum=delivery.MAX_FRAME_BYTES), maximum=delivery.MAX_FRAME_BYTES)


def test_packet_endpoint_explicit_and_metadata_default_unchanged(authored_service):
    service, observer, _handoff, _state = authored_service
    descriptor = service.descriptor
    assert descriptor["schema"] == delivery.DESCRIPTOR_SCHEMA
    assert observer.descriptor["schema"] == metadata.DESCRIPTOR_SCHEMA
    assert observer.descriptor["max_frame_bytes"] == 8192
    assert descriptor["max_frame_bytes"] > len(PACKET.encode()) > metadata.MAX_FRAME_BYTES
    assert delivery.validate_worker_packet_descriptor(descriptor, expected_sha256=service.descriptor_sha256) == descriptor
    assert descriptor["original_prompt_sha256"] != descriptor["prompt_sha256"]
    assert PACKET.encode() not in delivery._wire(descriptor) and TASK.encode() not in delivery._wire(descriptor)
    assert all(descriptor[k] is False for k in delivery.FALSE_FIELDS)
    descriptor["binding_id"] = "changed-copy"
    assert service.descriptor["binding_id"] == "authored-binding"


@pytest.mark.parametrize("change", ["extra", "pin", "uid", "bool", "expiry", "endpoint", "authority", "packet_bound"])
def test_rehashed_descriptor_forgery_rejected(authored_service, change):
    service = authored_service[0]
    value = service.descriptor
    if change == "extra": value["sql"] = "private"
    elif change == "uid": value["worker_uid"] = value["owner_uid"]
    elif change == "bool": value["max_requests"] = True
    elif change == "expiry": value["expires_monotonic_ns"] = value["issued_monotonic_ns"]
    elif change == "endpoint": value["endpoint_name"] = "/tmp/private.socket"
    elif change == "authority": value["execution_authority"] = True
    elif change == "packet_bound": value["prompt_bytes"] = delivery.MAX_PACKET_BYTES + 1
    pin = "0" * 64 if change == "pin" else delivery._sha(delivery._wire(value))
    with pytest.raises(ValueError):
        delivery.validate_worker_packet_descriptor(value, expected_sha256=pin)


def test_actual_unregistered_and_same_uid_socket_never_receives_packet(authored_service):
    service = authored_service[0]
    first = exchange(service, request(service))
    assert first["reason_code"] == "worker_not_registered" and first["packet_text"] is None
    with child() as process:
        service.note_spawned(process)
        second = exchange(service, request(service))
        assert second["reason_code"] == "peer_refused" and second["packet_text"] is None
    assert service.close()["sent_packet_bytes"] == 0


@pytest.mark.parametrize("value", [None, {}, object()])
def test_fake_process_never_registers(authored_service, value):
    with pytest.raises(ValueError):
        authored_service[0].note_spawned(value)


def test_wrong_command_and_second_actual_child_rejected(authored_service):
    service = authored_service[0]
    with child((sys.executable, "-I", "-c", "import sys; sys.stdin.readline()")) as wrong:
        with pytest.raises(ValueError): service.note_spawned(wrong)
    with child() as first, child() as second:
        service.note_spawned(first)
        with pytest.raises(ValueError): service.note_spawned(second)


@pytest.mark.parametrize("change", ["operation", "extra", "schema", "session", "pin", "request_id"])
def test_unknown_operations_fail_before_binding_read(authored_service, change):
    service, _observer, _handoff, state = authored_service
    value = request(service)
    if change == "extra": value["sql"] = "private"
    elif change == "operation": value["operation"] = "model.invoke"
    elif change == "schema": value["schema"] = "future"
    elif change == "session": value["session_id"] = "0" * 32
    elif change == "pin": value["descriptor_sha256"] = "0" * 64
    else: value["request_id"] = "bad"
    before = state["calls"]
    result = exchange(service, value)
    assert result["reason_code"] == "invalid_request" and result["packet_text"] is None
    assert state["calls"] == before


def test_authored_packet_ack_exact_bytes_one_delivery_no_authority(authored_service, monkeypatch):
    service = authored_service[0]
    with child() as process:
        service.note_spawned(process)
        peer = author_peer(monkeypatch, service)
        value = request(service)
        result = exchange(service, value)
        assert result["ok"] is True and result["packet_text"] == PACKET and result["peer"] == peer
        assert result["compiled_packet_membership_verified"] is True
        assert all(result[k] is False for k in delivery.FALSE_FIELDS)
        replay = exchange(service, value)
        assert replay["reason_code"] == "replayed_request" and replay["packet_text"] is None
        ack = exchange(service, acknowledge(service, value["request_id"]))
        assert ack["ok"] is True and ack["acknowledgement_recorded"] is True and ack["packet_text"] is None
        assert exchange(service, request(service))["reason_code"] == "packet_already_delivered"
        audit = service.close()
        assert audit["sent_packet_responses"] == audit["acknowledged_packets"] == 1
        assert audit["sent_packet_bytes"] == len(PACKET.encode())
        assert audit["sent_response_wire_bytes"] == audit["sent_response_bytes"] + 4 * audit["sent_responses"]
        assert audit["raw_packet_exported"] is False and all(audit[k] is False for k in delivery.FALSE_FIELDS)


@pytest.mark.parametrize("change", ["get_request_id", "prompt_sha256", "prompt_bytes", "packet_id", "context_cid", "bool_bytes"])
def test_acknowledgement_wrong_join_never_qualifies(authored_service, monkeypatch, change):
    service = authored_service[0]
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        get = request(service)
        assert exchange(service, get)["ok"] is True
        value = acknowledge(service, get["request_id"])
        if change == "bool_bytes": value["prompt_bytes"] = True
        elif change == "prompt_bytes": value[change] += 1
        else: value[change] = "wrong"
        result = exchange(service, value)
        assert result["reason_code"] == "acknowledgement_refused" and result["packet_text"] is None
        assert service.close()["acknowledged_packets"] == 0


def test_ack_before_get_and_duplicate_ack_refused(authored_service, monkeypatch):
    service = authored_service[0]
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        assert exchange(service, acknowledge(service, "0" * 32))["reason_code"] == "acknowledgement_refused"
        get = request(service)
        assert exchange(service, get)["ok"]
        assert exchange(service, acknowledge(service, get["request_id"]))["ok"]
        assert exchange(service, acknowledge(service, get["request_id"]))["reason_code"] == "acknowledgement_refused"


@pytest.mark.parametrize("stage", ["get", "ack", "second_get"])
def test_native_projection_change_blocks_delivery_and_ack(authored_service, monkeypatch, stage):
    service, _observer, _handoff, state = authored_service
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        get = request(service)
        if stage != "get": assert exchange(service, get)["ok"]
        if stage == "second_get": assert exchange(service, acknowledge(service, get["request_id"]))["ok"]
        state["current"] = False
        value = acknowledge(service, get["request_id"]) if stage == "ack" else request(service)
        result = exchange(service, value)
        assert result["reason_code"] == "delivery_refused" and result["packet_text"] is None


@pytest.mark.parametrize("change", ["body_cid", "relations_cid", "packet_bytes", "binding_pointer", "observer_pointer", "command"])
def test_original_identity_mutation_refused(authored_service, monkeypatch, change):
    service, observer, _handoff, state = authored_service
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        if change == "body_cid": state["information"]["expected_task_body_cid"] = "changed"
        elif change == "relations_cid": state["information"]["native_task_relations_cid"] = "changed"
        elif change == "packet_bytes": state["text"] += "changed"
        elif change == "binding_pointer": service._binding = object.__new__(NativeRepositoryWorldCompiledPacketBinding)
        elif change == "observer_pointer": service._observer = object.__new__(metadata.RepositoryWorldWorkerObserver)
        else: observer._worker_command_line = b"changed\0"
        result = exchange(service, request(service))
        assert result["reason_code"] == ("peer_refused" if change == "observer_pointer" else "delivery_refused")
        assert result["packet_text"] is None


def test_closing_native_change_after_response_serialization_exports_no_packet(authored_service, monkeypatch):
    service, _observer, _handoff, state = authored_service
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        original = delivery._wire
        def changed(value, **kwargs):
            raw = original(value, **kwargs)
            if type(value) is dict and value.get("schema") == delivery.RESPONSE_SCHEMA and value.get("ok"):
                state["current"] = False
            return raw
        monkeypatch.setattr(delivery, "_wire", changed)
        result = exchange(service, request(service))
        assert result["reason_code"] == "delivery_refused" and result["packet_text"] is None
        assert service.close()["sent_packet_bytes"] == 0


def test_different_authenticated_birth_cannot_acknowledge(authored_service, monkeypatch):
    service = authored_service[0]
    with child() as process:
        service.note_spawned(process)
        peer = author_peer(monkeypatch, service)
        get = request(service)
        assert exchange(service, get)["ok"]
        peer["start_time_ticks"] += 1
        monkeypatch.setattr(metadata.RepositoryWorldWorkerObserver, "_peer", lambda _self, _channel: deepcopy(peer))
        assert exchange(service, acknowledge(service, get["request_id"]))["reason_code"] == "peer_refused"


def test_exited_original_child_and_closed_metadata_owner_refuse(authored_service, monkeypatch):
    service, observer, _handoff, _state = authored_service
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        process.terminate()
        process.wait(timeout=2)
        assert exchange(service, request(service))["reason_code"] == "peer_refused"
    observer.close()
    with pytest.raises(ValueError): service._material()


def test_packet_cap_checked_before_service_start(authored_protocol_owners):
    (handoff, binding), _handoff_info, state = authored_protocol_owners
    state["text"] = "λ" * (delivery.MAX_PACKET_BYTES // 2 + 1)
    state["information"].update(prompt_sha256=delivery._sha(state["text"].encode()), prompt_bytes=len(state["text"].encode()))
    observer = metadata.RepositoryWorldWorkerObserver(handoff=handoff, prompt=TASK, expected_worker_argv=ARGV, timeout_seconds=30)
    try:
        with pytest.raises(ValueError): delivery.RepositoryWorldWorkerPacketDelivery(observer=observer, binding=binding)
    finally:
        observer.close()


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b' {"a":1}', b'{"a":true}\n',
    b'{"a":' + b'[' * 17 + b'0' + b']' * 17 + b'}'])
def test_bounded_canonical_decode_rejects_unsupported_json(raw):
    with pytest.raises(ValueError): delivery._decode(raw, maximum=delivery.MAX_REQUEST_BYTES)


def test_oversized_frame_refuses_without_reading_payload():
    first, second = socket.socketpair()
    try:
        first.sendall((delivery.MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(ValueError): delivery._recv(second, maximum=delivery.MAX_FRAME_BYTES)
    finally:
        first.close()
        second.close()


def test_closed_client_helper_never_prints_packet_or_exception(monkeypatch):
    class Stream:
        def __init__(self, raw): self.buffer = io.BytesIO(raw)
    output = Stream(b"")
    monkeypatch.setattr(delivery.sys, "stdin", Stream(b'{"sql":"private-input"}\n'))
    monkeypatch.setattr(delivery.sys, "stdout", output)
    assert delivery.worker_packet_client_main() == 1
    value = json.loads(output.buffer.getvalue())
    assert value["reason_code"] == "client_refused" and b"private-input" not in output.buffer.getvalue()
    assert all(value[k] is False for k in delivery.FALSE_FIELDS)


def authored_client_transport(monkeypatch, service, information, *, change=None):
    """A fake transport/owner only for independent client grammar tests."""
    descriptor = service.descriptor
    descriptor.update(owner_uid=os.getuid() + 1, worker_uid=os.getuid())
    pin = delivery._sha(delivery._wire(descriptor))
    request_id = "a" * 32
    peer = dict(pid=os.getpid(), uid=os.getuid(), gid=os.getgid(),
        start_time_ticks=metadata.process_identity(os.getpid())["start_time_ticks"], root_pid=os.getpid(),
        root_start_time_ticks=metadata.process_identity(os.getpid())["start_time_ticks"], groups_sha256="b" * 64, group_count=1)
    response = service._response(request_id, "get_packet", information=deepcopy(information),
        text=PACKET, peer=peer)
    if change == "payload": response["packet_text"] += "tampered"
    elif change == "byte_count": response["prompt_bytes"] += 1
    elif change == "bool_bytes": response["prompt_bytes"] = True
    elif change == "hash": response["prompt_sha256"] = "0" * 64
    elif change == "binding": response["binding"]["expected_task_body_cid"] = "changed"
    elif change == "peer": response["peer"]["start_time_ticks"] += 1
    elif change == "authority": response["provider_ingestion_verified"] = True
    elif change == "extra": response["raw_sql"] = "private"
    elif change == "ack": response["acknowledgement_recorded"] = True
    elif change == "wrong_operation": response["operation"] = "ack_packet"
    raw = delivery._wire(response, maximum=delivery.MAX_FRAME_BYTES)
    class Channel:
        def __init__(self): self.buffer = io.BytesIO(struct.pack("!I", len(raw)) + raw)
        def __enter__(self): return self
        def __exit__(self, *_arguments): pass
        def settimeout(self, _seconds): pass
        def connect(self, endpoint): assert endpoint == "\0" + descriptor["endpoint_name"]
        def getsockopt(self, *_arguments):
            return struct.pack("3i", descriptor["owner_pid"], descriptor["owner_uid"], descriptor["owner_gid"])
        def sendall(self, _raw): pass
        def recv(self, count): return self.buffer.read(count)
    monkeypatch.setattr(delivery.socket, "socket", lambda *_arguments: Channel())
    request_value = dict(schema=delivery.REQUEST_SCHEMA, operation="get_packet", session_id=descriptor["session_id"],
        request_id=request_id, descriptor_sha256=pin)
    return descriptor, pin, request_value


@pytest.mark.parametrize("change", ["payload", "byte_count", "bool_bytes", "hash", "binding", "peer", "authority", "extra", "ack", "wrong_operation"])
def test_client_independently_rejects_corrupted_received_packet(authored_service, monkeypatch, change):
    service, _observer, _handoff, state = authored_service
    # Shut down real services before replacing the socket constructor.
    service.close()
    _observer.close()
    descriptor, pin, get = authored_client_transport(monkeypatch, service, state["information"], change=change)
    with pytest.raises(ValueError): delivery._client_exchange(descriptor, pin, get, 1)


def test_client_independent_exact_unicode_payload_identity(authored_service, monkeypatch):
    service, observer, _handoff, state = authored_service
    service.close()
    observer.close()
    descriptor, pin, get = authored_client_transport(monkeypatch, service, state["information"])
    result = delivery._client_exchange(descriptor, pin, get, 1)
    assert result["packet_text"] == PACKET and result["prompt_bytes"] == len(PACKET.encode("utf-8"))


def test_client_ack_uses_received_bytes_and_withholds_private_packet_on_refusal(authored_service, monkeypatch):
    service, observer, _handoff, _state = authored_service
    descriptor = service.descriptor
    calls = []
    def transport(_descriptor, _pin, value, _timeout):
        calls.append(deepcopy(value))
        if value["operation"] == "get_packet":
            return dict(ok=True, reason_code=None, packet_text=PACKET, packet_id=descriptor["packet_id"],
                context_cid=descriptor["context_cid"], peer={"authored": True})
        assert value["prompt_sha256"] == hashlib.sha256(PACKET.encode()).hexdigest()
        assert value["prompt_bytes"] == len(PACKET.encode()) and value["get_request_id"] == calls[0]["request_id"]
        return dict(ok=False, reason_code="delivery_refused")
    monkeypatch.setattr(delivery, "_client_exchange", transport)
    result = delivery.fetch_repository_world_worker_packet(descriptor=descriptor, expected_descriptor_sha256=service.descriptor_sha256)
    assert result["packet_text"] is None
    assert result["receipt"]["status"] == "received_unacknowledged"
    assert result["receipt"]["packet_byte_identity_verified"] is True
    assert result["receipt"]["acknowledgement_recorded"] is False


def test_response_wire_budget_refuses_without_partial_packet(authored_service, monkeypatch):
    service = authored_service[0]
    with child() as process:
        service.note_spawned(process)
        author_peer(monkeypatch, service)
        service._wire_bytes = delivery.MAX_TOTAL_RESPONSE_WIRE_BYTES - delivery.MAX_METADATA_BYTES
        response = exchange(service, request(service))
        assert response["reason_code"] == "response_budget_exhausted" and response["packet_text"] is None
        assert service.close()["sent_packet_responses"] == 0


def test_invalid_json_cardinality_refused_before_decoder(monkeypatch):
    def forbidden(*_arguments, **_keywords): pytest.fail("must refuse before JSON allocation")
    monkeypatch.setattr(delivery.json, "loads", forbidden)
    raw = b'{"a":[' + b','.join([b'0'] * 4097) + b']}'
    with pytest.raises(ValueError): delivery._decode(raw, maximum=delivery.MAX_FRAME_BYTES)


def test_unissued_empty_observer_never_starts_packet_endpoint(authored_protocol_owners):
    _owners, _handoff_info, _state = authored_protocol_owners
    with pytest.raises(ValueError):
        delivery.RepositoryWorldWorkerPacketDelivery(observer=object.__new__(metadata.RepositoryWorldWorkerObserver), binding=_owners[1])


def test_success_helper_stdout_receipts_only_and_two_command_bound(authored_service, monkeypatch):
    service, observer, _handoff, _state = authored_service
    inputs = delivery._wire(dict(descriptor=service.descriptor, expected_descriptor_sha256=service.descriptor_sha256))
    service.close()
    observer.close()
    class Stream:
        def __init__(self, raw): self.buffer = io.BytesIO(raw)
    output, calls = Stream(b""), []
    monkeypatch.setattr(delivery.sys, "stdin", Stream(inputs + b"\nget_packet\nget_packet\nget_packet\n"))
    monkeypatch.setattr(delivery.sys, "stdout", output)
    def fetch(**keywords):
        calls.append(keywords)
        return dict(packet_text=PACKET, receipt=dict(schema=delivery.RECEIPT_SCHEMA, status="received_and_acknowledged",
            prompt_sha256=delivery._sha(PACKET.encode()), prompt_bytes=len(PACKET.encode()), raw_packet_exported=False,
            raw_packet_export_scope="receipt_and_stdout_only", **dict.fromkeys(delivery.FALSE_FIELDS, False)))
    monkeypatch.setattr(delivery, "fetch_repository_world_worker_packet", fetch)
    assert delivery.worker_packet_client_main() == 0 and len(calls) == 2
    lines = output.buffer.getvalue().splitlines()
    assert len(lines) == 2 and all(json.loads(line)["schema"] == delivery.RECEIPT_SCHEMA for line in lines)
    assert PACKET.encode() not in output.buffer.getvalue() and TASK.encode() not in output.buffer.getvalue()
