"""Owner request acknowledgments never become observations or task authority."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
import socket
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_observation_request as request


def scope():
    return request.observation_scope(program_id="aseh-test", owner_identity={
        "store_id": "test-only", "generation": 5,
    }, source_head="a" * 40, source_tree="b" * 40, launch_admission_id="sha256:test")


def exchange(server, *, available=True):
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(request.request_observation, server.scope)
        deadline = time.monotonic() + 2
        while not result.done() and time.monotonic() < deadline:
            server.poll(observer_available=available)
            time.sleep(.001)
        return result.result(timeout=1)


def raw_exchange(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.settimeout(1)
        client.connect(request._address(server.scope))
        client.sendall(payload)
        server.poll(observer_available=True)
        return request._receive(client)


def packet(server):
    return {"schema": request.SCHEMA, "scope": copy.deepcopy(server.scope),
            "requester_birth": request._birth(os.getpid()), "nonce": "c" * 32}


def test_exact_kernel_peer_request_coalesces_without_sampling_or_authority():
    server = request.OwnerObservationRequests(scope())
    try:
        for _ in range(3):
            result = exchange(server)
            assert result["accepted"] is True
            assert result["reason"] == "queued_for_next_sample"
            assert server.pending.is_set()
            assert result["authenticated_observation"] is False
            assert result["completion_authority"] is False
            assert result["mutation_authority"] is False
        server.sample_started()
        assert not server.pending.is_set()
    finally:
        server.close()


def test_unavailable_monitor_cannot_accept_or_clear_failure():
    server = request.OwnerObservationRequests(scope())
    try:
        result = exchange(server, available=False)
        assert result["accepted"] is False
        assert result["reason"] == "observer_unavailable"
        assert not server.pending.is_set()
    finally:
        server.close()


@pytest.mark.parametrize("field", ["program_id", "owner_identity", "source_head",
                                  "source_tree", "launch_admission_id", "process_birth"])
def test_exact_owner_scope_mismatch_never_enqueues(field):
    server = request.OwnerObservationRequests(scope())
    try:
        payload = packet(server)
        payload["scope"][field] = "foreign"
        result = raw_exchange(server, request._encoded(payload))
        assert result["accepted"] is False
        assert result["reason"] == "request_rejected"
        assert not server.pending.is_set()
    finally:
        server.close()


@pytest.mark.parametrize("change", ["unknown_field", "query", "replayed_birth", "bool_generation"])
def test_unknown_queries_birth_reuse_and_type_confusion_rejected(change):
    server = request.OwnerObservationRequests(scope())
    try:
        payload = packet(server)
        if change == "unknown_field": payload["grant"] = "never-export"
        if change == "query": payload["query"] = "SELECT secret"
        if change == "replayed_birth": payload["requester_birth"]["start_time_ticks"] += 1
        if change == "bool_generation": payload["scope"]["owner_identity"]["generation"] = True
        result = raw_exchange(server, request._encoded(payload))
        assert result["accepted"] is False
        assert b"never-export" not in request._encoded(result)
        assert b"secret" not in request._encoded(result)
        assert not server.pending.is_set()
    finally:
        server.close()


def test_foreign_uid_rejected_even_with_exact_scope(monkeypatch):
    server = request.OwnerObservationRequests(scope())
    try:
        monkeypatch.setattr(request, "_peer", lambda c: (os.getpid(), os.getuid() + 1))
        assert raw_exchange(server, request._encoded(packet(server)))["accepted"] is False
        assert not server.pending.is_set()
    finally:
        server.close()


@pytest.mark.parametrize("payload", [b"broken secret payload", b"x" * (request.MAX_PACKET + 1), b"[" * 2000 + b"0" + b"]" * 2000])
def test_malformed_or_oversized_packet_has_bounded_redacted_reply(payload):
    server = request.OwnerObservationRequests(scope())
    try:
        result = raw_exchange(server, payload)
        assert result["reason"] == "request_rejected"
        assert len(request._encoded(result)) < 512
        assert b"secret" not in request._encoded(result)
    finally:
        server.close()


def test_silent_client_is_bounded_and_does_not_stall_owner():
    server = request.OwnerObservationRequests(scope())
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.connect(request._address(server.scope))
            started = time.monotonic()
            server.poll(observer_available=True)
            assert time.monotonic() - started < .5
            assert not server.pending.is_set()
    finally:
        server.close()


def test_listener_requires_current_birth_and_closed_route_is_unavailable():
    binding = scope()
    binding["process_birth"]["start_time_ticks"] += 1
    with pytest.raises(ValueError, match="admitted owner process"):
        request.OwnerObservationRequests(binding)
    server = request.OwnerObservationRequests(scope())
    server.close()
    with pytest.raises(OSError):
        request.request_observation(server.scope)


def test_listener_close_error_cannot_escape_native_cleanup():
    server = request.OwnerObservationRequests(scope())
    server.close()
    class ClosedSocket:
        def close(self):
            raise OSError("already closed")
    server.listener = ClosedSocket()
    server.close()


@pytest.mark.parametrize("mismatch", ["uid", "birth"])
def test_client_refuses_wrong_receiver_before_request_send(monkeypatch, mismatch):
    server = request.OwnerObservationRequests(scope())
    try:
        if mismatch == "uid":
            monkeypatch.setattr(request, "_peer", lambda c: (os.getpid(), os.getuid() + 1))
        else:
            actual = request._birth
            def changed_birth(pid):
                value = actual(pid)
                value["start_time_ticks"] += 1
                return value
            monkeypatch.setattr(request, "_birth", changed_birth)
        with pytest.raises(ValueError, match="receiver birth differs"):
            request.request_observation(server.scope)
        assert not server.pending.is_set()
    finally:
        server.close()
