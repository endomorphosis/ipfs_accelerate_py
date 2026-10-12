"""Wire-level controls, independent of the native owner/source integration."""
import socket
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.current_source_lookup import MAX_RESPONSE_BYTES
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes


def test_source_frame_ceiling_rejects_header_before_waiting_for_body():
    client, peer = socket.socketpair()
    try:
        client.settimeout(0.2)
        peer.sendall((MAX_RESPONSE_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(typed.TypedStateOwnerProtocolError, match="frame size"):
            typed._receive_frame(client, maximum_bytes=MAX_RESPONSE_BYTES)
    finally:
        client.close()
        peer.close()


@pytest.mark.parametrize("raw", [b'{"ok": true}', b'{"ok":true,"ok":true}', b'{"count":1.0}'])
def test_source_frames_refuse_noncanonical_received_bytes(raw):
    client, peer = socket.socketpair()
    try:
        peer.sendall(len(raw).to_bytes(4, "big") + raw)
        with pytest.raises(typed.TypedStateOwnerProtocolError):
            typed._receive_frame(client, maximum_bytes=MAX_RESPONSE_BYTES, require_canonical=True)
    finally:
        client.close()
        peer.close()


def test_legacy_frame_reader_retains_whitespace_tolerance():
    client, peer = socket.socketpair()
    try:
        raw = b' {"ok": true} \n'
        peer.sendall(len(raw).to_bytes(4, "big") + raw)
        assert typed._receive_frame(client) == {"ok": True}
    finally:
        client.close()
        peer.close()


@pytest.mark.parametrize("noncanonical", [False, True])
def test_source_request_poisons_stream_after_invalid_wire_response(noncanonical):
    client_socket, peer = socket.socketpair()
    client_socket.settimeout(1)
    client = object.__new__(typed.TypedStateOwnerConnection)
    client._socket = client_socket
    client._request_lock = threading.RLock()
    client._closed = False
    client._active = False
    client._prepared_command = None
    client._request_index = 0
    client._local_completion_deadline = None
    errors = []

    def serve():
        try:
            request = typed._receive_frame(peer)
            assert request["action"] == "source.get"
            if noncanonical:
                reply = dict(schema=typed.TYPED_STATE_OWNER_SCHEMA,
                             request_id=request["request_id"], ok=True, result={})
                raw = b" " + canonical_json_bytes(reply)
                peer.sendall(len(raw).to_bytes(4, "big") + raw)
            else:
                peer.sendall((MAX_RESPONSE_BYTES + 1).to_bytes(4, "big"))
        except BaseException as exc:
            errors.append(exc)
        finally:
            peer.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with pytest.raises(typed.TypedStateOwnerProtocolError):
            client._request("source.get", payload={})
        assert client._closed is True
        assert client_socket.fileno() == -1
        with pytest.raises(typed.TypedStateOwnerProtocolError, match="closed"):
            client._request("source.get", payload={})
    finally:
        client_socket.close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert not errors
