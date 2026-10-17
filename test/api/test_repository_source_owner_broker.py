"""Explicit native resources with authored container/peer fixture boundaries.

These checks never claim real cross-UID deployment. Source capture, native
indexing, socket framing, leases, cancellation and lifecycle are real; only the
unavailable root-owned container manifest/Git allocation/peer are authored.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import stat
import tempfile
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from test.api.test_repository_source_parent_admission import native_parent


def policy_digest(parent):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import policy_sha256
    return policy_sha256(parent._scheduler.config.persisted_dict())


@pytest.fixture
def short_service_directory():
    # AF_UNIX paths must fit Linux's 108-byte sockaddr field even when the
    # admitted wrapper's pytest directory has a long evidence-package prefix.
    with tempfile.TemporaryDirectory(prefix="rm-") as directory:
        yield Path(directory)


@pytest.fixture
def owner(tmp_path, monkeypatch, native_parent, short_service_directory):
    repository, workspace = tmp_path / "repository", tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    source = workspace / "calc.py"
    source.write_text("# Authored source; never executed.\ndef add(value):\n    return value + 1\n")
    source.chmod(0o640)
    original = source.read_bytes(), stat.S_IMODE(source.stat().st_mode)
    boundary = short_service_directory / "boundary.json"
    boundary.write_text("authored container identity; not deployed")
    (short_service_directory / "worker-observations").mkdir(mode=0o750)
    monkeypatch.setattr(broker.guard, "_workspace", lambda *args: (repository, workspace))
    monkeypatch.setattr(broker.guard, "_owner_boundary", lambda path: (
        dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1,
             allowed_worktree_roots=[str(tmp_path)]), "b" * 64))
    monkeypatch.setattr(broker.guard, "_protected_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker.OwnerRepositoryMCPBroker, "_peer", lambda self, channel: None)
    value = broker.OwnerRepositoryMCPBroker(repository=repository, workspace=workspace,
        boundary=boundary, parent_lease=native_parent,
        expected_policy_sha256=policy_digest(native_parent), deadline_monotonic=time.monotonic() + 60)
    try:
        assert value._ready.wait(15), "native admission did not return within its bounded wait"
        assert value._service_error is None, value._service_error
        yield value
    finally:
        result = value.close()
        assert result["cleanup_complete"], result
        assert not native_parent.released
        assert (source.read_bytes(), stat.S_IMODE(source.stat().st_mode)) == original


def message(method, *, identity=1, params=None):
    value = dict(jsonrpc="2.0", method=method)
    if identity is not None:
        value["id"] = identity
    if params is not None:
        value["params"] = params
    return value


def exchange(owner, value, *, request=None):
    request = protocol.make_request(owner.descriptor["session_id"], value) if request is None else request
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner.socket_path))
        protocol.send_frame(channel, request, deadline_monotonic=time.monotonic() + 2)
        # Use the lower-level framing API with an explicit twelve-second
        # native-response wait followed by a two-second complete frame bound.
        assert select.select([channel], [], [], 12)[0], "bounded native response wait expired"
        reply = protocol.receive_frame(channel, deadline_monotonic=time.monotonic() + 2)
    return protocol.validate_response(reply, request=request, descriptor=owner.descriptor)


def initialize(owner):
    result = exchange(owner, message("initialize", params={"protocolVersion": "2025-06-18",
        "capabilities": {}, "clientInfo": {"name": "authored-fixture", "version": "1"}}))
    assert "result" in result["message"], result
    assert exchange(owner, message("notifications/initialized", identity=None))["message"] is None


def tool(owner, name, **arguments):
    result = exchange(owner, message("tools/call", params={"name": name, "arguments": arguments}))
    assert result["ok"], result
    content = result["message"]["result"]
    assert not content["isError"], content
    return json.loads(content["content"][0]["text"])


def test_native_read_query_refresh_and_preserved_audit(owner, native_parent):
    initialize(owner)
    assert owner._native.parent_lease_id == native_parent.lease_id
    assert owner._native.owner_pid == os.getpid()
    assert owner._native._scheduler.state_path == native_parent._scheduler.state_path
    listing = exchange(owner, message("tools/list"))["message"]["result"]["tools"]
    assert "repository_replace_symbol" not in {row["name"] for row in listing}
    status = tool(owner, "repository_status")
    generation = status["generation"]
    assert "value + 1" in tool(owner, "repository_read", generation=generation, path="calc.py")["content"]
    query = tool(owner, "repository_context_query", generation=generation, query="add", max_bytes=4096)
    assert query["generation"] == generation and query["candidates"]
    assert tool(owner, "repository_refresh", generation=generation)["generation"] == generation + 1
    private_path = Path(owner._binding["invocation_dir"])
    result = owner.close()
    assert result["native_child_released"] and result["cleanup_complete"]
    assert not private_path.exists()
    assert result["prepared_mcp_usage"]["successful_reads"] == 1
    retained = result["retained_mcp_audit"]
    raw = (owner.audit_path.parent / retained["name"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == retained["sha256"]
    assert b"read_observation" in raw and b"value + 1" not in raw
    public = json.dumps(result)
    assert "lease_key" not in public and str(private_path) not in public
    assert result["cross_uid_deployment_qualified"] is False


def test_descriptor_expires_no_later_than_parent_deadline(owner):
    assert owner.descriptor["expires_monotonic_ns"] <= int(owner.deadline_monotonic * 1e9)
    assert owner.descriptor["read_only"] is True
    assert "parent_token" not in owner.descriptor and "state_dir" not in owner.descriptor


def test_public_descriptor_expiry_cancels_native_work_and_frame_signal(owner):
    owner.descriptor["expires_monotonic_ns"] = time.monotonic_ns() - 1
    assert time.monotonic() < owner.deadline_monotonic
    assert owner._signal.is_set()
    assert owner._server._resource_signal.is_set()
    assert owner.close()["cleanup_complete"]


def test_replayed_outer_request_is_not_dispatched_twice(owner, monkeypatch):
    calls = []
    original = broker.mcp.RepositoryMCPServer.handle
    def observed(self, request):
        calls.append(request)
        return original(self, request)
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", observed)
    request = protocol.make_request(owner.descriptor["session_id"], message("ping"))
    assert exchange(owner, None, request=request)["ok"]
    assert exchange(owner, None, request=request)["reason_code"] == "invalid_request"
    assert len(calls) == 1


def test_edit_is_refused_before_server_dispatch(owner, monkeypatch):
    def forbidden(*args):
        pytest.fail("read-only service dispatched a canonical edit")
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", forbidden)
    request = dict(schema=protocol.REQUEST_SCHEMA, session_id=owner.descriptor["session_id"],
        request_id="e" * 32, message=message("tools/call", params={"name": "repository_replace_symbol", "arguments": {}}))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner.socket_path))
        protocol.send_frame(channel, request, deadline_monotonic=time.monotonic() + 2)
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.receive_frame(channel, deadline_monotonic=time.monotonic() + 2)
    result = owner.close()
    assert result["accepted_requests"] == 1 and not result["requests"][0]["delivered"]
    assert result["prepared_mcp_usage"]["tool_calls"] == 0


def test_exhausted_budget_refuses_before_native_handler(owner, monkeypatch):
    owner._budget_exhausted = True
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle",
                        lambda *args: pytest.fail("exhausted response budget dispatched native work"))
    result = exchange(owner, message("ping"))
    assert result["reason_code"] == "budget_exhausted"


def test_peer_refusal_never_dispatches_native_handler(owner, monkeypatch):
    def refused(channel):
        raise ValueError("authored wrong peer")
    monkeypatch.setattr(owner, "_peer", refused)
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle",
                        lambda *args: pytest.fail("refused peer dispatched native work"))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner.socket_path))
        with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
            protocol.receive_frame(channel, deadline_monotonic=time.monotonic() + 2)
    result = owner.close()
    assert result["requests"][0]["status"] == "peer_refused"


def test_malformed_frames_and_notifications_share_attempt_limit(owner):
    for _ in range(4):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(2)
            channel.connect(str(owner.socket_path))
            channel.shutdown(socket.SHUT_WR)
            assert channel.recv(1) == b""
    for _ in range(broker.MAX_REQUESTS - 4):
        assert exchange(owner, message("notifications/initialized", identity=None))["ok"]
    assert owner._done.wait(3)
    result = owner.close()
    assert result["accepted_requests"] == len(result["requests"]) == broker.MAX_REQUESTS
    assert result["accounting_complete"] and result["cleanup_complete"]
    assert result["prepared_mcp_usage"]["tool_calls"] == 0


def test_blocked_handler_retains_child_and_later_close_finishes(owner, monkeypatch, native_parent):
    entered, finish = threading.Event(), threading.Event()
    original = broker.mcp.RepositoryMCPServer.handle
    closed_threads, handle_threads = [], []
    native_close = broker.mcp.RepositoryMCPServer.close
    def blocked(self, request):
        handle_threads.append(threading.get_ident())
        entered.set()
        assert finish.wait(10), "authored handler release was not signaled"
        return original(self, request)
    def closed(self):
        closed_threads.append(threading.get_ident())
        return native_close(self)
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", blocked)
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "close", closed)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(str(owner.socket_path))
        protocol.send_frame(channel, protocol.make_request(owner.descriptor["session_id"], message("ping")),
                            deadline_monotonic=time.monotonic() + 2)
        try:
            assert entered.wait(2)
            pending = owner.close(timeout_seconds=.01)
            assert pending["cleanup_pending"] and not pending["native_child_released"]
            assert not pending["service_stopped"] and not owner.audit_path.exists()
            assert not owner._native.released and not native_parent.released
            assert Path(owner._binding["invocation_dir"]).exists()
            assert not owner.socket_path.exists() and not owner.descriptor_path.exists()
            assert not closed_threads
        finally:
            finish.set()
        result = owner.close()
    assert result["cleanup_complete"] and closed_threads == handle_threads


def test_boundary_change_during_handler_prevents_delivery(owner, monkeypatch):
    original = broker.mcp.RepositoryMCPServer.handle
    def changed(self, request):
        result = original(self, request)
        owner._boundary_sha256 = "c" * 64
        return result
    monkeypatch.setattr(broker.mcp.RepositoryMCPServer, "handle", changed)
    with pytest.raises(protocol.RepositorySourceOwnerProtocolError):
        exchange(owner, message("ping"))
    result = owner.close()
    assert not result["requests"][0]["delivered"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 4, 10**400])
def test_invalid_close_wait_rejected_without_stopping_service(owner, value):
    with pytest.raises(ValueError):
        owner.close(timeout_seconds=value)
    assert not owner._stop.is_set()


def test_invalid_parent_rejected_before_container_or_state_access(tmp_path):
    with pytest.raises(ValueError):
        broker.OwnerRepositoryMCPBroker(repository=tmp_path, workspace=tmp_path,
            boundary=tmp_path / "missing", parent_lease=object(), expected_policy_sha256="a" * 64,
            deadline_monotonic=time.monotonic() + 30)
    assert list(tmp_path.iterdir()) == []
