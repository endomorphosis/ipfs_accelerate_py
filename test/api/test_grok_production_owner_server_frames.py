"""Unexecuted private frame tests; no SDK/claim/owner/native authority.

Only actual socketpair I/O and stdlib time/thread controls are exercised. No
ProductionOwnerServer or ExistingProductionOwner is constructed or replaced.
"""
import socket
import threading
import time

import pytest

import ipfs_accelerate_py.cli_runtime.grok_production_owner as core
import ipfs_accelerate_py.cli_runtime.grok_production_owner_server as server


def test_complete_frame_preserves_canonical_value():
    left, right = socket.socketpair()
    try:
        value = {'schema': 'private-frame-only', 'sequence': 1}
        left.sendall(core.canonical(value))
        assert server._receive_frame(core, right, work_deadline_monotonic=time.monotonic()+10) == value
    finally:
        left.close(); right.close()


def test_fragmented_frame_completes_within_one_absolute_budget():
    left, right = socket.socketpair()
    value = {'schema': 'private-fragments', 'sequence': 1}
    raw = core.canonical(value)
    def writer():
        left.sendall(raw[:3]); time.sleep(.01); left.sendall(raw[3:9]); time.sleep(.01); left.sendall(raw[9:])
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert server._receive_frame(core, right, work_deadline_monotonic=time.monotonic()+10) == value
    finally:
        thread.join(1); left.close(); right.close()
        assert not thread.is_alive()


def test_drip_bytes_do_not_extend_original_absolute_frame_deadline():
    left, right = socket.socketpair()
    stop = threading.Event()
    def writer():
        try:
            while not stop.is_set():
                left.sendall(b'x')
                stop.wait(.02)
        except OSError:
            pass
    thread = threading.Thread(target=writer)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(server.ProductionServerRefusal,
                           match='^production_owner_frame_absolute_deadline_expired$'):
            server._receive_frame(core, right, work_deadline_monotonic=started+3.15)
        assert time.monotonic()-started < .9
    finally:
        stop.set(); thread.join(1); left.close(); right.close()
        assert not thread.is_alive()


def test_expired_original_work_margin_refuses_before_receive():
    left, right = socket.socketpair()
    try:
        with pytest.raises(server.ProductionServerRefusal,
                           match='^production_owner_frame_absolute_deadline_expired$'):
            server._receive_frame(core, right, work_deadline_monotonic=time.monotonic()+2)
    finally:
        left.close(); right.close()


@pytest.mark.parametrize('kind', ['duplicate', 'multiple', 'oversize', 'eof'])
def test_invalid_or_partial_frames_remain_closed(kind):
    left, right = socket.socketpair()
    values = {'duplicate': b'{"a":1,"a":2}\n', 'multiple': b'{}\n{}\n',
              'oversize': b'x'*(core.MAX_FRAME+1), 'eof': b'{"partial":'}
    try:
        left.sendall(values[kind]); left.shutdown(socket.SHUT_WR)
        with pytest.raises((server.ProductionServerRefusal, core.ProductionOwnerRefusal)):
            server._receive_frame(core, right, work_deadline_monotonic=time.monotonic()+10)
    finally:
        left.close(); right.close()


def test_successful_send_preserves_one_canonical_frame():
    left, right = socket.socketpair()
    try:
        value = {'schema': 'private-send-only', 'sequence': 1}
        server._send_frame(core, left, value, work_deadline_monotonic=time.monotonic()+10)
        right.settimeout(1)
        assert right.recv(core.MAX_FRAME+1) == core.canonical(value)
    finally:
        left.close(); right.close()


def test_expired_original_work_margin_refuses_before_send():
    left, right = socket.socketpair()
    try:
        with pytest.raises(server.ProductionServerRefusal,
                           match='^production_owner_frame_absolute_deadline_expired$'):
            server._send_frame(core, left, {'not': 'sent'}, work_deadline_monotonic=time.monotonic()+2)
        right.setblocking(False)
        with pytest.raises(BlockingIOError):
            right.recv(1)
    finally:
        left.close(); right.close()
