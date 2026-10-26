"""Native owner-to-stdio roundtrips with explicitly authored isolation guards.

The imported owner fixture authors the unavailable container manifest, allocated
Git worktree and server-side worker ancestry. The loader below authors only the
same-process worker's descriptor-access prerequisite, retaining the actual
descriptor bytes, hash and schema. Client SO_PEERCRED/PID-birth verification,
socket identity, framing, native MCP/index work and child leases remain real.
These tests do not qualify cross-UID deployment or automatic Terminal wiring.
Native work requires the enclosing admitted runner's actual borrowed parent.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio
from test.api.test_repository_source_owner_broker import owner, short_service_directory
from test.api.test_repository_source_parent_admission import native_parent


@pytest.fixture
def authored_worker_descriptor(owner, monkeypatch):
    """Preserve hash/schema binding while declaring absent cross-UID isolation."""
    observed = {"loads": [], "loaded": threading.Event()}

    def load(path, digest, *, workspace=None):
        path = Path(path)
        raw = path.read_bytes()
        value = protocol.validate_descriptor(protocol.strict_json_loads(raw))
        if (path != owner.descriptor_path or digest != owner.descriptor_sha256
                or hashlib.sha256(raw).hexdigest() != digest
                or raw != protocol.canonical_json(value)
                or value != owner.descriptor
                or Path(workspace) != owner.workspace
                or value["workspace"] != str(owner.workspace)
                or not value["issued_monotonic_ns"] <= time.monotonic_ns() < value["expires_monotonic_ns"]):
            raise protocol.RepositorySourceOwnerProtocolError("peer_refused")
        # Only the worker UID / protected-parent / current-directory access
        # prerequisites are authored. request_message still authenticates the
        # real owner process twice and checks the real socket inode and mode.
        observed["loads"].append((path, digest))
        observed["loaded"].set()
        return value

    monkeypatch.setattr(protocol, "load_descriptor", load)
    return observed


def _message(method, identity, params=None):
    value = {"jsonrpc": "2.0", "method": method}
    if identity is not None:
        value["id"] = identity
    if params is not None:
        value["params"] = params
    return value


def _tool(identity, name, **arguments):
    return _message("tools/call", identity, {"name": name, "arguments": arguments})


@contextmanager
def _output_pipe():
    """Drain an actual bounded FIFO concurrently; no drainer survives cleanup."""
    reader_fd, writer_fd = os.pipe()
    received, failures = bytearray(), []
    stopped = threading.Event()

    def drain():
        try:
            with os.fdopen(reader_fd, "rb", buffering=0) as reader:
                os.set_blocking(reader.fileno(), False)
                while not stopped.is_set():
                    if not select.select([reader], [], [], .1)[0]:
                        continue
                    try:
                        part = os.read(reader.fileno(), 65536)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not part:
                        return
                    received.extend(part)
                    if len(received) > protocol.MAX_RESPONSE_FRAME_BYTES:
                        raise AssertionError("authored stdio output exceeded bounded drain")
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=drain, name="authored-owner-stdio-drainer")
    with os.fdopen(writer_fd, "wb", buffering=0) as outgoing:
        worker.start()
        try:
            yield outgoing, received
        finally:
            outgoing.close()  # EOF lets the drainer retain all already-written bytes.
            worker.join(2)
            if worker.is_alive():
                stopped.set()
                worker.join(1)
            assert not worker.is_alive(), "authored stdio drainer did not stop"
            assert not failures, failures


def _relay(owner, messages, *, digest=None, cancel_event=None):
    # Finite inputs fit a single atomic write into an empty FIFO. Output is
    # drained concurrently so native replies cannot deadlock on pipe capacity.
    payload = b"".join(protocol.canonical_json(value) + b"\n" for value in messages)
    input_fd, writer_fd = os.pipe()
    with os.fdopen(input_fd, "rb", buffering=0) as incoming:
        try:
            assert len(payload) <= os.fpathconf(writer_fd, "PC_PIPE_BUF")
            if payload:
                assert os.write(writer_fd, payload) == len(payload)
        finally:
            os.close(writer_fd)
        with _output_pipe() as (outgoing, received):
            result = stdio.serve(owner.descriptor_path, owner.descriptor_sha256 if digest is None else digest,
                workspace=owner.workspace, incoming=incoming, outgoing=outgoing,
                deadline_monotonic=time.monotonic() + 20, cancel_event=cancel_event)
        raw = bytes(received)
    assert len(raw) <= protocol.MAX_RESPONSE_FRAME_BYTES
    assert not raw or raw.endswith(b"\n")
    return result, [protocol.strict_json_loads(line) for line in raw.splitlines()]


def _payload(response):
    assert "result" in response, response
    result = response["result"]
    assert result["isError"] is False, result
    assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
    return json.loads(result["content"][0]["text"])


def test_stdio_native_read_query_refresh_and_exact_retained_audit(
        owner, native_parent, authored_worker_descriptor, monkeypatch):
    source = owner.workspace / "calc.py"
    original = source.read_bytes(), stat.S_IMODE(source.stat().st_mode)
    native_child = owner._native
    assert native_child.parent_lease_id == native_parent.lease_id
    assert native_child.owner_pid == os.getpid()
    assert native_child._scheduler.state_path == native_parent._scheduler.state_path
    assert not native_child.released

    verified, sockets = [], []
    verify_peer = protocol._base._verify_peer
    socket_identity = protocol._base._socket_identity

    def real_peer(channel, descriptor):
        result = verify_peer(channel, descriptor)
        verified.append(descriptor["owner_pid"])
        return result

    def real_socket(descriptor, parent, group):
        result = socket_identity(descriptor, parent, group)
        sockets.append((descriptor["socket_device"], descriptor["socket_inode"]))
        return result

    monkeypatch.setattr(protocol._base, "_verify_peer", real_peer)
    monkeypatch.setattr(protocol._base, "_socket_identity", real_socket)
    started = [
        _message("initialize", 1, {"protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "authored-owner-stdio-fixture", "version": "1"}}),
        _message("notifications/initialized", None),
        _tool(2, "repository_status"),
    ]
    code, replies = _relay(owner, started)
    assert code == 0 and [reply["id"] for reply in replies] == [1, 2]
    assert "result" in replies[0]
    generation = _payload(replies[1])["generation"]
    operations = [
        _message("tools/list", 3),
        _tool(4, "repository_read", generation=generation, path="calc.py"),
        _tool(5, "repository_context_query", generation=generation, query="add", max_bytes=4096),
        _tool(6, "repository_refresh", generation=generation),
        _tool(7, "repository_read", generation=generation + 1, path="calc.py"),
        _tool(8, "repository_replace_symbol", generation=generation + 1, path="calc.py"),
    ]
    code, replies = _relay(owner, operations)
    assert code == 0 and [reply["id"] for reply in replies] == [3, 4, 5, 6, 7, 8]
    listed = {tool["name"] for tool in replies[0]["result"]["tools"]}
    assert listed == protocol.READ_ONLY_TOOL_NAMES
    read_before, query, refreshed, read_after = map(_payload, replies[1:5])
    assert read_before["content"].encode() == read_after["content"].encode() == original[0]
    assert read_before["generation"] == generation
    assert read_after["generation"] == refreshed["generation"] == generation + 1
    assert query["generation"] == generation and query["candidates"]
    assert replies[-1]["error"] == {"code": -32600, "message": "invalid_request"}
    assert (source.read_bytes(), stat.S_IMODE(source.stat().st_mode)) == original

    # The edit failed in the worker adapter; eight actual owner exchanges remain.
    expected_requests = len(started) + len(operations) - 1
    assert len(verified) == expected_requests * 2
    assert set(verified) == {os.getpid()}
    assert sockets == [(owner.descriptor["socket_device"], owner.descriptor["socket_inode"])] * expected_requests
    assert len(authored_worker_descriptor["loads"]) == expected_requests + 2
    private_path = Path(owner._binding["invocation_dir"])
    result = owner.close()
    assert result["accepted_requests"] == expected_requests
    assert len(result["requests"]) == expected_requests
    assert all(row["delivered"] for row in result["requests"])
    assert result["accounting_complete"] and result["cleanup_complete"]
    assert result["native_child_acquired"] and result["native_child_released"]
    assert native_child.released and not native_parent.released
    assert not private_path.exists() and not owner.socket_path.exists() and not owner.descriptor_path.exists()
    usage = result["prepared_mcp_usage"]
    assert usage["initializations"] == 1 and usage["tool_calls"] == 5
    assert usage["successful_reads"] == 2 and usage["refreshes"] == 1
    assert usage["canonical_edits"] == usage["uncertain_edits"] == usage["errors"] == 0
    observed = usage["read_observation"]
    assert observed["identified_reads"] == 2 and observed["complete"]
    assert observed["same_generation_repeat_reads"] == 0
    assert observed["cross_generation_repeat_reads"] == 1

    retained = result["retained_mcp_audit"]
    raw = (owner.audit_path.parent / retained["name"]).read_bytes()
    assert retained["content_free"] and len(raw) == retained["bytes"]
    assert hashlib.sha256(raw).hexdigest() == retained["sha256"]
    assert original[0] not in raw and b"value + 1" not in raw
    rows = [json.loads(line) for line in raw.splitlines()]
    reads = [row["read_observation"] for row in rows if row["tool"] == "repository_read"]
    assert len(reads) == 2 and [row["generation"] for row in reads] == [generation, generation + 1]
    source_sha = hashlib.sha256(original[0]).hexdigest()
    identity_sha = hashlib.sha256(protocol.canonical_json(dict(
        path="calc.py", source_sha256=source_sha, scope="file", symbol_id=None))).hexdigest()
    for row in reads:
        assert row["source_identity_sha256"] == identity_sha
        assert row["content_sha256"] == source_sha and row["content_bytes"] == len(original[0])
        assert row["scope"] == "file" and row["representation"] == "original" and row["encoding"] == "utf-8"
    public = json.dumps(result)
    assert "lease_key" not in public and "parent_token" not in public and str(private_path) not in public
    assert result["cross_uid_deployment_qualified"] is False
    assert result["automatic_terminal_wiring"] is False
    assert result["token_savings_measured"] is False


def test_stdio_eof_preserves_owner_custody_until_explicit_close(owner, native_parent, authored_worker_descriptor):
    assert _relay(owner, []) == (0, [])
    assert owner._accepted == 0 and not owner._native.released
    assert owner.socket_path.exists() and owner.descriptor_path.exists()
    result = owner.close()
    assert result["cleanup_complete"] and result["native_child_released"]
    assert not native_parent.released


def test_stdio_idle_cancel_preserves_owner_custody(owner, native_parent, authored_worker_descriptor, monkeypatch):
    cancel, waiting = threading.Event(), threading.Event()
    result = []
    input_fd, writer_fd = os.pipe()
    with os.fdopen(input_fd, "rb", buffering=0) as incoming, _output_pipe() as (outgoing, received):
        ready = protocol._ready

        def real_ready(descriptor, *, writing, deadline, cancel_event=None):
            if descriptor == incoming.fileno() and not writing:
                waiting.set()
            return ready(descriptor, writing=writing, deadline=deadline, cancel_event=cancel_event)

        monkeypatch.setattr(protocol, "_ready", real_ready)

        def run():
            result.append(stdio.serve(owner.descriptor_path, owner.descriptor_sha256,
                workspace=owner.workspace, incoming=incoming, outgoing=outgoing,
                deadline_monotonic=time.monotonic() + 5, cancel_event=cancel))

        worker = threading.Thread(target=run, name="authored-owner-stdio-client")
        worker.start()
        try:
            assert waiting.wait(2), "adapter did not reach its actual idle FIFO wait"
            assert len(authored_worker_descriptor["loads"]) == 1
            cancel.set()
            worker.join(2)
            assert not worker.is_alive() and result == [2]
        finally:
            cancel.set()
            os.close(writer_fd)
            worker.join(3)
        assert not worker.is_alive()
    assert received == b""
    assert owner._accepted == 0 and not owner._native.released
    result = owner.close()
    assert result["cleanup_complete"] and result["native_child_released"]
    assert not native_parent.released


def test_stdio_closed_owner_refuses_without_new_request(owner, native_parent, authored_worker_descriptor):
    result = owner.close()
    assert result["cleanup_complete"] and result["native_child_released"]
    assert _relay(owner, [_message("ping", 1)]) == (2, [])
    assert owner._accepted == 0 and not native_parent.released


def test_stdio_authored_loader_retains_actual_descriptor_digest_binding(owner, authored_worker_descriptor):
    assert _relay(owner, [_message("ping", 1)], digest="0" * 64) == (2, [])
    assert not authored_worker_descriptor["loads"] and owner._accepted == 0
