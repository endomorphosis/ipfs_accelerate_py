"""Real AF_UNIX transport clocks only; no native owner or provider authority.

Import this suite through the real package after the exact candidate patch is
materialized. Inert JSON, same-process peer identity, and helper success here do
not attest a current database claim, native lease, paid call, or model result.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import threading
import time
import uuid

import pytest
import ipfs_accelerate_py.cli_runtime.grok_production_owner as owner
import test.api.test_grok_production_owner as existing_owner_cases


@contextmanager
def actual_pair():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        yield left, right
    finally:
        left.close()
        right.close()


@contextmanager
def actual_writer(channel, action):
    stopped = threading.Event()
    failures = []

    def run():
        try:
            action(channel, stopped)
        except OSError as error:
            # Closing the exact owned socket is how the reader cancels a
            # remaining real writer; unrelated errors are retained below.
            if not stopped.is_set():
                failures.append(error)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run, name="actual-private-rpc-writer")
    thread.start()
    try:
        yield stopped
    finally:
        stopped.set()
        try:
            channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        thread.join(timeout=2)
        assert not thread.is_alive(), "actual writer thread did not close"
        assert not failures, [type(error).__name__ for error in failures]


def test_real_fragmented_frame_finishes_before_one_original_deadline():
    value = {"fixture": "fragmented", "native_authority": False}
    body = owner.canonical(value)

    def write(channel, stopped):
        for start in range(0, len(body), 7):
            if stopped.wait(.005):
                return
            channel.sendall(body[start:start + 7])

    with actual_pair() as (reader, writer):
        original_deadline = time.monotonic() + 1.0
        with actual_writer(writer, write):
            assert owner.receive(reader, deadline_monotonic=original_deadline) == value
            assert 0 < reader.gettimeout() < 1.0
            assert time.monotonic() < original_deadline


def test_real_drip_feed_cannot_renew_receive_deadline():
    delivered = []

    def drip(channel, stopped):
        while not stopped.wait(.025):
            channel.sendall(b" ")
            delivered.append(time.monotonic())

    with actual_pair() as (reader, writer):
        # A longer ambient socket timeout must not become a new time grant.
        reader.settimeout(9)
        original_deadline = time.monotonic() + .18
        with actual_writer(writer, drip):
            with pytest.raises((TimeoutError, owner.ProductionOwnerRefusal)):
                owner.receive(reader, deadline_monotonic=original_deadline)
        assert delivered, "the test must actually deliver fragments"
        assert time.monotonic() >= original_deadline


def test_actual_blocked_sendall_uses_original_deadline():
    with actual_pair() as (sender, unread):
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        sender.settimeout(9)
        value = {"fixture": "x" * 7900}
        assert len(owner.canonical(value)) <= owner.MAX_FRAME
        original_deadline = time.monotonic() + .15
        with pytest.raises((TimeoutError, owner.ProductionOwnerRefusal)):
            owner.send(sender, value, deadline_monotonic=original_deadline)
        assert time.monotonic() >= original_deadline
        # Bytes may have reached the real unread socket; they are not a result
        # or permission to resend the operation under a fresh deadline.
        assert unread.recv(8192, socket.MSG_DONTWAIT)


@pytest.mark.parametrize("operation", ["connect", "send", "receive"])
def test_expired_absolute_clock_refuses_before_actual_socket_IO(operation):
    with actual_pair() as (client, other):
        if operation == "receive":
            other.sendall(b'{"fixture":true}\n')
        with pytest.raises(owner.ProductionOwnerRefusal, match="owner_rpc_absolute_deadline_expired"):
            if operation == "connect":
                owner._connect_with_deadline(client, "\0unused-" + uuid.uuid4().hex,
                    time.monotonic() - .01)
            elif operation == "send":
                owner.send(client, {"fixture": True}, deadline_monotonic=time.monotonic() - .01)
            else:
                owner.receive(client, deadline_monotonic=time.monotonic() - .01)
        if operation == "receive":
            assert client.recv(8192, socket.MSG_DONTWAIT) == b'{"fixture":true}\n'
        else:
            with pytest.raises(BlockingIOError):
                other.recv(8192, socket.MSG_DONTWAIT)


def test_actual_EOF_without_complete_frame_still_refuses():
    with actual_pair() as (reader, writer):
        writer.sendall(b'{"fixture":')
        writer.shutdown(socket.SHUT_WR)
        with pytest.raises(owner.ProductionOwnerRefusal, match="owner_unexpected_EOF"):
            owner.receive(reader, deadline_monotonic=time.monotonic() + 1.0)


def test_actual_connect_send_receive_share_one_clock():
    # Abstract AF_UNIX needs no filesystem alias or external endpoint. This
    # repository's Linux/proc owner protocol already requires Linux semantics.
    endpoint = "\0grok-private-rpc-" + uuid.uuid4().hex
    received = []
    original_deadline = time.monotonic() + .20
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(endpoint)
        listener.listen(1)
        listener.settimeout(1)

        def answer(server, stopped):
            channel, _ = server.accept()
            with channel:
                channel.settimeout(1)
                raw = bytearray()
                while b"\n" not in raw:
                    part = channel.recv(1024)
                    assert part
                    raw.extend(part)
                received.append(bytes(raw))
                # The real connection and request have already consumed part
                # of the original clock. A response gets no renewed budget.
                stopped.wait(max(0, original_deadline + .05 - time.monotonic()))
                if not stopped.is_set():
                    channel.sendall(owner.canonical({"fixture": "late"}))

        with actual_writer(listener, answer):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                owner._connect_with_deadline(client, endpoint, original_deadline)
                assert owner.peer(client)["pid"] == os.getpid()
                owner.send(client, {"fixture": "request"}, deadline_monotonic=original_deadline)
                with pytest.raises((TimeoutError, owner.ProductionOwnerRefusal)):
                    owner.receive(client, deadline_monotonic=original_deadline)
        assert received == [owner.canonical({"fixture": "request"})]
        assert time.monotonic() >= original_deadline


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf")])
def test_invalid_deadline_is_rejected_without_sending(deadline):
    with actual_pair() as (client, other):
        with pytest.raises(owner.ProductionOwnerRefusal, match="owner_rpc_absolute_deadline_invalid"):
            owner.send(client, {"fixture": True}, deadline_monotonic=deadline)
        with pytest.raises(BlockingIOError):
            other.recv(8192, socket.MSG_DONTWAIT)


@pytest.mark.parametrize("slow_stage", ["final_socket_identity", "response_digest"])
def test_real_ACK_slow_final_validation_refuses_before_receipt_commit(tmp_path, monkeypatch, slow_stage):
    """Retain actual observations/digests; only their test latency is injected.

    The existing fixture forks a genuine setsid client, observes its real birth
    and kernel peer, sends a real ACK, and returns the actual client result.
    Its context is explicitly inert protocol metadata, never a native grant.
    """
    root = Path(owner.__file__).resolve(strict=True).parents[2]
    expected = root / 'test/api/test_grok_production_owner.py'
    assert Path(existing_owner_cases.__file__).resolve(strict=True) == expected
    assert existing_owner_cases.__spec__.origin == str(expected)
    assert existing_owner_cases.core is owner
    parent_pid = os.getpid()
    genuine_socket_identity = owner.socket_identity
    genuine_digest = owner.digest
    child_socket_reads = 0

    def slow_actual_socket_identity(path):
        nonlocal child_socket_reads
        actual = genuine_socket_identity(path)
        if os.getpid() != parent_pid:
            child_socket_reads += 1
            if slow_stage == 'final_socket_identity' and child_socket_reads == 2:
                time.sleep(1.10)
        return actual

    def slow_actual_response_digest(value):
        actual = genuine_digest(value)
        if (os.getpid() != parent_pid and slow_stage == 'response_digest' and type(value) is dict
                and value.get('schema') == owner.RPC_SCHEMA and value.get('status') == 'current'):
            time.sleep(1.10)
        return actual

    monkeypatch.setattr(owner, 'socket_identity', slow_actual_socket_identity)
    monkeypatch.setattr(owner, 'digest', slow_actual_response_digest)
    _, requests, result = existing_owner_cases.socket_case(
        tmp_path, stages=('worker_v4_dispatch',))
    assert len(requests) == 1
    assert result['successful'] is False
    assert result['error'] == 'owner_rpc_absolute_deadline_expired'
    assert result['latched'] == 'owner_client_failure_latched'
    assert result['receipts'] == []
    assert result['native_authority_claimed'] is False
