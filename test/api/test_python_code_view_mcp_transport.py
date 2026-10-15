"""Bounded live MCP transport over actual OS pipes and task-owned native Git."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_mcp_transport import (
    PROTOCOL_VERSION,
    PythonCodeViewMCPTransport,
    PythonCodeViewMCPTransportError,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_session_broker import (
    PythonCodeViewSessionBroker,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_wire_worktree import (
    build_python_code_view_wire_edit,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_worktree import (
    apply_code_view_patch,
)
from test.api.test_python_code_view_session_broker import (  # noqa: F401 -- real Git fixture
    SID,
    SOURCE,
    tree,
)

_MISSING = object()


def _open(tree, *, editable=False):  # noqa: F811 -- fixture registration
    worktree, scope, _, _ = tree
    broker = PythonCodeViewSessionBroker(worktree=worktree, scope=scope, session_id=SID,
        lease_id=worktree.lease_id, fence=worktree.fence, ttl_seconds=300)
    return broker, PythonCodeViewMCPTransport(broker=broker, session_id=SID, editable=editable)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _request(identity, method, params=_MISSING):
    value = {"jsonrpc": "2.0", "id": identity, "method": method}
    if params is not _MISSING:
        value["params"] = params
    return value


async def _pipe_exchange(transport, value, frames=None):
    """Send must really write all response bytes through a live kernel pipe."""
    raw = value if type(value) is bytes else _encode(value)
    reader, writer = os.pipe()

    def read_all():
        chunks = []
        while chunk := os.read(reader, 8192):
            chunks.append(chunk)
        return b"".join(chunks)

    def write_all(data):
        pending = memoryview(data)
        while pending:
            sent = os.write(writer, pending)
            assert sent > 0
            pending = pending[sent:]

    async def send(data):
        assert type(data) is bytes and data.endswith(b"\n") and data.count(b"\n") == 1
        await asyncio.to_thread(write_all, data)

    receiving = asyncio.create_task(asyncio.to_thread(read_all))
    try:
        await transport.exchange(raw, send)
    finally:
        os.close(writer)
        delivered = await receiving
        os.close(reader)
    if frames is not None:
        frames.append({"request_utf8": raw.decode("utf-8"),
                       "response_utf8": delivered.decode("utf-8")})
    if not delivered:
        return None
    assert delivered.endswith(b"\n") and delivered.count(b"\n") == 1
    result = json.loads(delivered)
    assert result["jsonrpc"] == "2.0"
    _public(result)
    return result


class _Client:
    def __init__(self, transport, frames=None):
        self.transport, self.frames, self.sequence = transport, frames, 0

    async def call(self, method, params=_MISSING):
        self.sequence += 1
        return await _pipe_exchange(self.transport, _request(self.sequence, method, params), self.frames)

    async def ready(self, protocol=PROTOCOL_VERSION):
        response = await self.call("initialize", {"protocolVersion": protocol,
            "capabilities": {}, "clientInfo": {"name": "real-pipe-fixture", "version": "1"}})
        assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert response["result"]["capabilities"] == {"tools": {"listChanged": False}}
        assert await _pipe_exchange(self.transport,
            {"jsonrpc": "2.0", "method": "notifications/initialized"}, self.frames) is None

    async def tool(self, name, **arguments):
        response = await self.call("tools/call", {"name": "python_code_view." + name,
            "arguments": {"session_id": SID, **arguments}})
        assert "error" not in response, response
        result = response["result"]
        assert set(result) == {"content", "isError"} and type(result["isError"]) is bool
        assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
        body = json.loads(result["content"][0]["text"])
        assert result["isError"] is bool(body.get("is_error", False))
        _public(body)
        return body


def _public(value):
    pending = [value]
    forbidden = {"original_source", "visible_sources", "prepared", "native_edit", "patch_text", "token_map", "source_span"}
    while pending:
        item = pending.pop()
        if type(item) is dict:
            assert not forbidden.intersection(item)
            for key, child in item.items():
                if key.endswith("_authority"):
                    assert child is False
                if key == "lean_admits":
                    assert child == 0
                pending.append(child)
        elif type(item) is list:
            pending.extend(item)


def _wire(response):
    legend, state = response["legend"], response["state"]
    local = [alias for alias in legend["functions"][0][2] if alias[2] == 1][-1]
    index, slot = next((i, s) for i, s in enumerate(legend["slots"]) if s[1:3] == [0, 1])
    return build_python_code_view_wire_edit(view_id=state["view_id"], slot_id=slot[0],
        before_sha256=state["hashes"][index], expression=f"({local[1]} * 3) + 1")


async def _delivered(client):
    response = await client.tool("read_view", path="calc.py")
    native = await client.tool("deliver_native_source", view_id=response["state"]["view_id"])
    assert native["source"] == SOURCE
    await client.tool("acknowledge_native_source", delivery_id=native["delivery_id"],
                      view_id=response["state"]["view_id"])
    return response, native


def test_closed_initialization_and_real_private_discovery(tree):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready(protocol="2099-01-01")
        response = await client.call("tools/list")
        tools = response["result"]["tools"]
        assert len(tools) == 6 and all(tool["name"].startswith("python_code_view.") for tool in tools)
        assert not any("apply" in tool["name"] or "consume" in tool["name"] for tool in tools)
        assert not transport._visible and not broker._views
        assert transport._manager.runtime_router is None
        assert SID not in json.dumps(tools) and str(tree[0].worktree_path) not in json.dumps(tools)
        await transport.aclose()
        assert transport._closed and broker._closed and not transport._visible
    asyncio.run(scenario())


def test_tools_cannot_dispatch_before_initialized_notification(tree):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        response = await client.call("tools/call", {"name": "python_code_view.read_view",
            "arguments": {"session_id": SID, "path": "calc.py"}})
        assert response["error"]["code"] == -32000 and not broker._views
        initialized = await client.call("initialize", {"protocolVersion": PROTOCOL_VERSION,
            "capabilities": {}, "clientInfo": {"name": "fixture", "version": "1"}})
        assert "result" in initialized
        response = await client.call("tools/list")
        assert response["error"]["code"] == -32000
        await transport.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("params", [
    {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"roots": {}}, "clientInfo": {"name": "x", "version": "1"}},
    {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}, "private": "/secret/owner"},
])
def test_initialize_closed_parameters_do_not_disclose_private_values(tree, params):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        response = await _pipe_exchange(transport, _request(1, "initialize", params))
        assert response["error"]["code"] == -32602 and "/secret/owner" not in json.dumps(response)
        assert not broker._views and not transport._visible
        await transport.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("raw", [
    b'{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/list"}',
    b'\xff/private/secret',
    b" " * (96 * 1024 + 1),
])
def test_bounded_duplicate_free_json_parser_refuses_without_dispatch(tree, raw):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        response = await _pipe_exchange(transport, raw)
        assert response["error"]["code"] in {-32700, -32600}
        assert "/private/secret" not in json.dumps(response) and not broker._views
        await transport.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("value", [
    {"jsonrpc": "2.0", "id": True, "method": "tools/list"},
    [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}],
])
def test_non_request_envelopes_never_reach_tools(tree, value):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        response = await _pipe_exchange(transport, value)
        assert response["error"]["code"] == -32600 and not broker._views
        await transport.aclose()
    asyncio.run(scenario())


def test_repeated_request_identity_is_refused_instead_of_cached(tree):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        request = _request("same-id", "tools/call", {"name": "python_code_view.read_view",
            "arguments": {"session_id": SID, "path": "calc.py"}})
        first = await _pipe_exchange(transport, request)
        assert "result" in first and broker._views
        repeated = await _pipe_exchange(transport, request)
        assert repeated["error"]["code"] in {-32600, -32000}
        await transport.aclose()
    asyncio.run(scenario())


def test_new_identity_reexecutes_identical_tool_arguments_after_source_changes(tree):  # noqa: F811
    _, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        before = await client.tool("read_view", path="calc.py")
        path = tree[0].worktree_path / "calc.py"
        changed = SOURCE.replace("return intermediate_quantity * 2", "return intermediate_quantity * 4")
        path.write_bytes(changed.encode())
        after = await client.tool("read_view", path="calc.py")
        assert after["state"]["view_id"] != before["state"]["view_id"]
        native = await client.tool("deliver_native_source", view_id=after["state"]["view_id"])
        assert native["source"] == changed and transport._visible == {"calc.py": changed}
        assert path.read_bytes() == changed.encode()
        await transport.aclose()
    asyncio.run(scenario())


def test_actual_frame_budget_closes_custody_at_the_bound(tree):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready()  # initialize request and initialized notification
        for _ in range(254):
            assert "result" in await client.call("tools/list")
        with pytest.raises(PythonCodeViewMCPTransportError):
            await client.call("tools/list")
        assert transport._closed and broker._closed and not transport._visible
    asyncio.run(scenario())


@pytest.mark.parametrize("arguments", [
    {"path": "calc.py", "__use_result_cache": True},
    {"path": "calc.py", "visible_sources": {"calc.py": "/private/forged"}},
])
def test_model_cache_controls_and_visibility_cannot_bypass_live_dispatch(tree, arguments):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        result = await client.tool("read_view", **arguments)
        assert result["is_error"] is True and not broker._views and not transport._visible
        assert "/private/forged" not in json.dumps(result)
        await transport.aclose()
    asyncio.run(scenario())


def test_os_pipe_tools_prepare_then_controller_handoff_applies_once(tree, record_property):  # noqa: F811
    broker, transport = _open(tree, editable=True)
    frames = []
    async def scenario():
        client = _Client(transport, frames)
        await client.ready()
        discovery = await client.call("tools/list")
        assert len(discovery["result"]["tools"]) == 7
        response = await client.tool("read_view", path="calc.py")
        request = _wire(response)
        first = await client.tool("propose_edit", wire_request=request)
        assert first["is_error"] is True and not transport._visible
        native = await client.tool("deliver_native_source", view_id=response["state"]["view_id"])
        assert native["source"] == SOURCE and transport._visible == {"calc.py": SOURCE}
        before_ack = await client.tool("propose_edit", wire_request=request)
        assert before_ack["is_error"] is True, "completed send does not acknowledge model receipt"
        await client.tool("acknowledge_native_source", delivery_id=native["delivery_id"],
                          view_id=response["state"]["view_id"])
        path = tree[0].worktree_path / "calc.py"
        before_proposal_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        proposal = await client.tool("propose_edit", wire_request=request)
        assert proposal["candidate_only"] is True
        assert path.read_bytes() == SOURCE.encode()
        handoff = await transport.consume_candidate(proposal["candidate_id"])
        assert handoff.prepared.patch_text and handoff.visible_sources == {"calc.py": SOURCE}
        with pytest.raises(TypeError):
            handoff.visible_sources["calc.py"] = "changed"
        assert path.read_bytes() == SOURCE.encode(), "controller handoff must not apply"
        after_handoff_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert transport._closed and broker._closed and not transport._visible
        with pytest.raises(PythonCodeViewMCPTransportError):
            await transport.consume_candidate(proposal["candidate_id"])
        worktree, scope, _, _ = tree
        result = apply_code_view_patch(worktree=worktree, scope=scope, prepared=handoff.prepared,
            expected_view_id=handoff.prepared.view.view_id, lease_id=worktree.lease_id,
            fence=worktree.fence, visible_sources=handoff.visible_sources)
        assert result.applied, result.to_dict()
        expected = SOURCE.replace("return intermediate_quantity * 2", "return (intermediate_quantity * 3) + 1")
        assert path.read_bytes() == expected.encode()
        assert (tree[2] / "calc.py").read_bytes() == SOURCE.encode()
        execution = subprocess.run([sys.executable, "-B", "-c",
            "from calc import adjusted_total; assert adjusted_total(4,3)==22; assert adjusted_total(-5,4)==-2; assert adjusted_total(0,0)==1"],
            cwd=worktree.worktree_path, text=True, capture_output=True)
        assert execution.returncode == 0, execution.stderr
        final_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        record_property("python_code_view_mcp_transport_probe", json.dumps({
            "schema": "supervisor-python-code-view-mcp-transport-probe@1",
            "transport": "actual-os-pipe", "frames": frames,
            "framed_utf8_bytes": sum(len(frame["request_utf8"].encode()) + len(frame["response_utf8"].encode()) for frame in frames),
            "native_preimage_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "before_proposal_sha256": before_proposal_sha256,
            "after_handoff_sha256": after_handoff_sha256,
            "observed_final_native_sha256": final_sha256,
            "patch_sha256": hashlib.sha256(handoff.prepared.patch_text.encode()).hexdigest(),
            "controller_events": [{"operation": "native_apply", "applied": result.applied,
                                   "native_before_sha256": after_handoff_sha256,
                                   "native_after_sha256": final_sha256}],
            "native_fixture_execution_passed": True,
            "model_tools_apply_source": False, "model_input_tokens": None,
            "native_apply_count": 1, "provider_calls": 0, "downloads": 0, "lean_admits": 0,
        }, ensure_ascii=False, sort_keys=True))
    asyncio.run(scenario())


def test_native_response_send_failure_releases_all_private_custody(tree):  # noqa: F811
    broker, transport = _open(tree, editable=True)
    secret = "/private/transport/secret"
    async def scenario():
        client = _Client(transport)
        await client.ready()
        response = await client.tool("read_view", path="calc.py")
        async def failed_send(_data):
            raise OSError(secret)
        with pytest.raises(PythonCodeViewMCPTransportError) as captured:
            await transport.exchange(_encode(_request(99, "tools/call", {
                "name": "python_code_view.deliver_native_source", "arguments": {
                    "session_id": SID, "view_id": response["state"]["view_id"]}})), failed_send)
        assert secret not in str(captured.value)
        assert transport._closed and broker._closed and not transport._visible
        assert not broker._views and not broker._native
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode()
    asyncio.run(scenario())


def test_fresh_source_drift_refuses_controller_candidate_handoff(tree):  # noqa: F811
    _, transport = _open(tree, editable=True)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        response, _ = await _delivered(client)
        proposal = await client.tool("propose_edit", wire_request=_wire(response))
        assert proposal["candidate_only"] is True
        path = tree[0].worktree_path / "calc.py"
        path.write_bytes((SOURCE + "# concurrent edit\n").encode())
        with pytest.raises(PythonCodeViewMCPTransportError):
            await transport.consume_candidate(proposal["candidate_id"])
        assert path.read_bytes() == (SOURCE + "# concurrent edit\n").encode()
        await transport.aclose()
    asyncio.run(scenario())


def test_slow_send_refuses_overlap_and_never_grants_visibility_early(tree):  # noqa: F811
    broker, transport = _open(tree, editable=True)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        response = await client.tool("read_view", path="calc.py")
        entered, release = asyncio.Event(), asyncio.Event()
        async def stalled_send(_data):
            entered.set()
            await release.wait()
        pending = asyncio.create_task(transport.exchange(_encode(_request(99, "tools/call", {
            "name": "python_code_view.deliver_native_source", "arguments": {
                "session_id": SID, "view_id": response["state"]["view_id"]}})), stalled_send))
        await asyncio.wait_for(entered.wait(), 5)
        try:
            assert not transport._visible
            with pytest.raises(PythonCodeViewMCPTransportError):
                await transport.consume_candidate("not-a-candidate")
            with pytest.raises(PythonCodeViewMCPTransportError):
                await _pipe_exchange(transport, _request(100, "tools/list"))
        finally:
            release.set()
            await pending
        assert transport._visible == {"calc.py": SOURCE} and not broker._closed
        await transport.aclose()
    asyncio.run(scenario())


def test_controller_close_is_idempotent_and_closed_tool_frames_have_no_body(tree):  # noqa: F811
    broker, transport = _open(tree, editable=True)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        await _delivered(client)
        assert transport._visible
        await transport.aclose()
        await transport.aclose()
        assert transport._closed and broker._closed and not transport._visible
        response = await client.call("tools/list")
        assert response["error"]["code"] == -32000
        assert "source" not in json.dumps(response) and str(tree[0].worktree_path) not in json.dumps(response)
    asyncio.run(scenario())


def test_cancelled_send_preserves_cancellation_and_closes_session(tree):  # noqa: F811
    broker, transport = _open(tree, editable=True)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        response = await client.tool("read_view", path="calc.py")
        entered, blocked = asyncio.Event(), asyncio.Event()
        async def cancelled_send(_data):
            entered.set()
            await blocked.wait()
        pending = asyncio.create_task(transport.exchange(_encode(_request(99, "tools/call", {
            "name": "python_code_view.deliver_native_source", "arguments": {
                "session_id": SID, "view_id": response["state"]["view_id"]}})), cancelled_send))
        await asyncio.wait_for(entered.wait(), 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert transport._closed and broker._closed and not transport._visible
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode()
    asyncio.run(scenario())


def test_foreign_thread_event_loop_cannot_take_channel_custody(tree):  # noqa: F811
    broker, transport = _open(tree)
    async def scenario():
        client = _Client(transport)
        await client.ready()
        with pytest.raises(PythonCodeViewMCPTransportError):
            await asyncio.to_thread(lambda: asyncio.run(transport.aclose()))
        assert not transport._closed and not broker._closed
        assert "result" in await client.call("tools/list")
        await transport.aclose()
    asyncio.run(scenario())
