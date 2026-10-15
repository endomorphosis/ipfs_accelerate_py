"""Real dedicated MCP dispatch with native custody and controller-only apply."""
from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
import threading

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    python_code_view_session_broker as broker_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_mcp_tools import (
    PythonCodeViewMCPTools,
    register_python_code_view_tools,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_session_broker import (
    PythonCodeViewSessionBroker,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_wire_worktree import (
    build_python_code_view_wire_edit,
)
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ipfs_accelerate_py.mcp_server.tool_metadata import get_tool_metadata
from ipfs_accelerate_py.mcp_server.tool_registry import ToolRegistry, get_global_registry
from test.api.test_python_code_view_session_broker import (  # noqa: F401 -- actual Git fixture
    SID,
    SOURCE,
    tree,
)

CATEGORY = "python_code_view"
READ_TOOLS = {
    "read_view", "acknowledge_legend", "acknowledge_read", "deliver_native_source",
    "acknowledge_native_source", "close_session",
}


def _open(tree, *, editable=True, resolver=None):  # noqa: F811 -- fixture registration
    worktree, scope, _, _ = tree
    broker = PythonCodeViewSessionBroker(worktree=worktree, scope=scope, session_id=SID,
        lease_id=worktree.lease_id, fence=worktree.fence, ttl_seconds=300)
    visible, calls = {}, []

    def transport_visibility(request):
        calls.append(request)
        return dict(visible)

    adapter = PythonCodeViewMCPTools(broker=broker, session_id=SID,
        visibility_resolver=(resolver or transport_visibility) if editable else None)
    manager = HierarchicalToolManager(registry=ToolRegistry())
    names = register_python_code_view_tools(manager, adapter)
    return broker, adapter, manager, visible, calls, names


async def _call(manager, name, **arguments):
    return await manager.dispatch(CATEGORY, name, {"session_id": SID, **arguments})


def _wire(response, *, increment=1):
    legend, state = response["legend"], response["state"]
    local = [a for a in legend["functions"][0][2] if a[2] == 1][-1]
    index, slot = next((i, s) for i, s in enumerate(legend["slots"]) if s[1:3] == [0, 1])
    return build_python_code_view_wire_edit(view_id=state["view_id"], slot_id=slot[0],
        before_sha256=state["hashes"][index], expression=f"({local[1]} * 3) + {increment}")


async def _delivered(manager, visible):
    response = await _call(manager, "read_view", path="calc.py")
    native = await _call(manager, "deliver_native_source", view_id=response["state"]["view_id"])
    # The controller records source from the actual returned tool body, never
    # from private artifacts, original fixtures, compact views or a file guess.
    visible["calc.py"] = native["source"]
    ack = await _call(manager, "acknowledge_native_source", delivery_id=native["delivery_id"],
                      view_id=response["state"]["view_id"])
    assert ack["delivery_id"] == native["delivery_id"]
    return response, native


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


def test_real_registration_is_task_private_and_discovery_is_independent(tree):  # noqa: F811
    broker, adapter, manager, _, _, names = _open(tree, editable=False)
    assert set(names) == {f"{CATEGORY}.{name}" for name in READ_TOOLS}
    assert {v["name"] for v in manager.list_tools(CATEGORY)} == READ_TOOLS
    assert not broker._views and not broker._native and not broker._known_native
    descriptors = adapter.descriptors()
    descriptors[0]["input_schema"]["properties"].clear()
    assert all(d["input_schema"]["properties"] for d in adapter.descriptors())
    for name in READ_TOOLS:
        schema = manager.get_tool_schema(CATEGORY, name)["input_schema"]
        assert schema["additionalProperties"] is False
        assert manager.registry.get_tool(f"{CATEGORY}.{name}").get_schema()["input_schema"] == schema
        metadata = get_tool_metadata(f"{CATEGORY}.{name}")
        assert metadata._func is None and metadata.mcp_schema == schema
        assert SID not in json.dumps(metadata.to_dict())
        assert str(tree[0].worktree_path) not in json.dumps(metadata.to_dict())
        assert not {"visible_sources", "prepared", "worktree", "scope", "fence", "lease_id"}.intersection(schema["properties"])
        assert inspect.iscoroutinefunction(manager._categories[CATEGORY][name].func)
    assert manager.runtime_router is None
    adapter.close()


def test_registration_refuses_process_global_registry(tree):  # noqa: F811
    worktree, scope, _, _ = tree
    broker = PythonCodeViewSessionBroker(worktree=worktree, scope=scope, session_id=SID,
        lease_id=worktree.lease_id, fence=worktree.fence)
    adapter = PythonCodeViewMCPTools(broker=broker, session_id=SID)
    registry = get_global_registry()
    manager = HierarchicalToolManager(registry=registry)
    before = tuple(registry.get_all_tools())
    with pytest.raises(ValueError):
        register_python_code_view_tools(manager, adapter)
    assert tuple(registry.get_all_tools()) == before
    adapter.close()


@pytest.mark.parametrize("fault", ["category", "registry", "loader", "router"])
def test_registration_rejects_non_private_manager_without_overwriting(tree, fault):  # noqa: F811
    worktree, scope, _, _ = tree
    broker = PythonCodeViewSessionBroker(worktree=worktree, scope=scope, session_id=SID,
        lease_id=worktree.lease_id, fence=worktree.fence)
    adapter = PythonCodeViewMCPTools(broker=broker, session_id=SID)
    manager = HierarchicalToolManager(registry=ToolRegistry())
    if fault == "category":
        manager.register_tool("other", "existing", lambda: {"existing": True})
    elif fault == "registry":
        manager.registry.register_function(lambda: {}, tool_name="existing")
    elif fault == "loader":
        manager.register_category_loader("other", lambda value: None)
    else:
        manager.runtime_router = object()
    before = (manager.list_categories(), [v.name for v in manager.registry.get_all_tools()])
    with pytest.raises(ValueError):
        register_python_code_view_tools(manager, adapter)
    assert before == (manager.list_categories(), [v.name for v in manager.registry.get_all_tools()])
    adapter.close()


def test_repeat_registration_does_not_replace_live_bindings(tree):  # noqa: F811
    _, adapter, manager, _, _, names = _open(tree)
    with pytest.raises(ValueError):
        register_python_code_view_tools(manager, adapter)
    assert {v.name for v in manager.registry.get_all_tools()} == set(names)
    adapter.close()


@pytest.mark.parametrize("arguments", [
    {"path": "calc.py", "worktree": "/private/owner"},
    {"path": "calc.py", "visible_sources": {"calc.py": SOURCE}},
    {"path": "calc.py", "__use_result_cache": True},
    {"path": "calc.py", "session_id": True},
    {"path": "calc.py", "session_id": "ANOTHER-SESSION"},
    {"path": False},
    {"path": "λ" * 1025},
])
def test_closed_requests_refuse_before_source_delivery(tree, arguments):  # noqa: F811
    broker, adapter, manager, _, calls, _ = _open(tree)
    async def scenario():
        result = await _call(manager, "read_view", **arguments)
        assert result["is_error"] is True
        assert result["error_code"] in {"invalid_request", "operation_refused"}
        assert not broker._views and not broker._native and calls == []
        _public(result)
        assert "/private/owner" not in json.dumps(result)
    asyncio.run(scenario())
    adapter.close()


def test_proposal_without_actual_native_ack_cannot_write(tree):  # noqa: F811
    _, adapter, manager, visible, _, _ = _open(tree)
    visible["calc.py"] = SOURCE  # Merely knowing private bytes cannot grant native delivery.
    async def scenario():
        response = await _call(manager, "read_view", path="calc.py")
        result = await _call(manager, "propose_edit", wire_request=_wire(response))
        assert result["is_error"] is True
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode()
        _public(result)
    asyncio.run(scenario())
    adapter.close()


def test_sensitive_controller_failure_is_not_echoed(tree):  # noqa: F811
    secret = "/private/owner/controller_secret"
    def failed(_request):
        raise RuntimeError(secret)
    _, adapter, manager, visible, _, _ = _open(tree, resolver=failed)
    async def scenario():
        response, _ = await _delivered(manager, visible)
        result = await _call(manager, "propose_edit", wire_request=_wire(response))
        assert result["is_error"] is True and secret not in json.dumps(result)
        _public(result)
    asyncio.run(scenario())
    adapter.close()


def test_actual_mcp_flow_prepares_only_and_controller_applies_exactly_once(tree, record_property):  # noqa: F811
    broker, adapter, manager, visible, calls, _ = _open(tree)
    frames = []
    async def exchange(name, **arguments):
        request = {"session_id": SID, **arguments}
        frames.append({"role": "assistant", "name": name, "content": request})
        result = await manager.dispatch(CATEGORY, name, request)
        frames.append({"role": "tool", "name": name, "content": result})
        _public(result)
        return result
    async def scenario():
        frames.append({"role": "tool", "name": "discovery", "content": adapter.descriptors()})
        response = await exchange("read_view", path="calc.py")
        await exchange("acknowledge_legend", legend_id=response["state"]["legend_id"], read_id=response["read_id"])
        await exchange("acknowledge_read", view_id=response["state"]["view_id"], read_id=response["read_id"])
        reference = await exchange("read_view", path="calc.py", known_read_id=response["read_id"])
        assert reference["view_id"] == response["state"]["view_id"] and "state" not in reference
        native = await exchange("deliver_native_source", view_id=reference["view_id"])
        visible["calc.py"] = native["source"]
        await exchange("acknowledge_native_source", delivery_id=native["delivery_id"], view_id=reference["view_id"])
        reused = await exchange("deliver_native_source", view_id=reference["view_id"], known_delivery_id=native["delivery_id"])
        assert "source" not in reused and reused["delivery_id"] == native["delivery_id"]
        request = _wire(response)
        proposal = await exchange("propose_edit", wire_request=request)
        assert proposal["candidate_only"] is True and calls == [request]
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode(), "model tools must not apply"
        prepared = adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
        with pytest.raises(ValueError):
            adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
        result = broker.apply_edit(prepared=prepared, visible_sources=dict(visible), session_id=SID)
        assert result.applied, result.to_dict()
        frames.append({"role": "controller", "name": "native_apply", "content": result.to_dict()})
        after = await exchange("read_view", path="calc.py")
        assert after["state"]["view_id"] != response["state"]["view_id"]
        expected = SOURCE.replace("return intermediate_quantity * 2", "return (intermediate_quantity * 3) + 1")
        assert (tree[0].worktree_path / "calc.py").read_bytes() == expected.encode()
        assert (tree[2] / "calc.py").read_bytes() == SOURCE.encode()
        execution = subprocess.run([sys.executable, "-B", "-c",
            "from calc import adjusted_total; assert adjusted_total(4,3)==22; assert adjusted_total(-5,4)==-2; assert adjusted_total(0,0)==1"],
            cwd=tree[0].worktree_path, text=True, capture_output=True)
        assert execution.returncode == 0, execution.stderr
        closed = await exchange("close_session")
        assert closed["lean_admits"] == 0
        denied = await exchange("read_view", path="calc.py")
        assert denied["is_error"] is True
    asyncio.run(scenario())
    rendered = json.dumps(frames, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record_property("python_code_view_mcp_probe", json.dumps({
        "schema": "supervisor-python-code-view-mcp-probe@1", "actual_manager_dispatch": True,
        "model_tools_apply_source": False, "native_fixture_execution_passed": True,
        "framed_transcript_bytes": len(rendered.encode()), "frames": frames,
        "model_input_tokens": None, "provider_calls": 0, "downloads": 0, "lean_admits": 0,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def test_candidate_recheck_rejects_stale_source_without_consumption(tree):  # noqa: F811
    _, adapter, manager, visible, _, _ = _open(tree)
    async def scenario():
        response, _ = await _delivered(manager, visible)
        proposal = await _call(manager, "propose_edit", wire_request=_wire(response))
        changed = SOURCE + "# external edit\n"
        (tree[0].worktree_path / "calc.py").write_bytes(changed.encode())
        with pytest.raises(ValueError):
            adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
        assert (tree[0].worktree_path / "calc.py").read_bytes() == changed.encode()
    asyncio.run(scenario())
    adapter.close()


def test_candidate_capacity_and_consumed_handle_never_revives(tree):  # noqa: F811
    _, adapter, manager, visible, _, _ = _open(tree)
    async def scenario():
        response, _ = await _delivered(manager, visible)
        proposals = [await _call(manager, "propose_edit", wire_request=_wire(response, increment=i)) for i in range(4)]
        assert len({p["candidate_id"] for p in proposals}) == 4
        refused = await _call(manager, "propose_edit", wire_request=_wire(response, increment=9))
        assert refused["is_error"] is True
        old = proposals[0]["candidate_id"]
        adapter.consume_candidate(old, visible_sources=dict(visible))
        recreated = await _call(manager, "propose_edit", wire_request=_wire(response, increment=0))
        assert recreated["candidate_id"] != old
        with pytest.raises(ValueError):
            adapter.consume_candidate(old, visible_sources=dict(visible))
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode()
    asyncio.run(scenario())
    adapter.close()


def test_expired_session_refuses_proposals_and_clears_controller_callback(tree, monkeypatch):  # noqa: F811
    broker, adapter, manager, visible, calls, _ = _open(tree)
    async def scenario():
        response, _ = await _delivered(manager, visible)
        proposal = await _call(manager, "propose_edit", wire_request=_wire(response))
        assert "candidate_id" in proposal and len(calls) == 1
        monkeypatch.setattr(broker_module, "_monotonic", lambda: broker._deadline + 1)
        result = await _call(manager, "propose_edit", wire_request=_wire(response))
        assert result["is_error"] is True and len(calls) == 1
        assert broker._closed and not broker._views
        assert adapter._closed and adapter._broker is None and adapter._resolver is None
        assert not adapter._candidates
        with pytest.raises(ValueError):
            adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
    asyncio.run(scenario())
    adapter.close()


def test_consumption_rechecks_caller_visibility_and_keeps_valid_candidate(tree):  # noqa: F811
    _, adapter, manager, visible, _, _ = _open(tree)
    async def scenario():
        response, _ = await _delivered(manager, visible)
        proposal = await _call(manager, "propose_edit", wire_request=_wire(response))
        with pytest.raises(ValueError):
            adapter.consume_candidate(proposal["candidate_id"], visible_sources={"calc.py": response["state"]["source"]})
        assert (tree[0].worktree_path / "calc.py").read_bytes() == SOURCE.encode()
        prepared = adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
        assert prepared.view.view_id == response["state"]["view_id"]
    asyncio.run(scenario())
    adapter.close()


def test_actual_fence_advance_clears_adapter_candidate_and_callback_custody(tree):  # noqa: F811
    broker, adapter, manager, visible, calls, _ = _open(tree)
    renewed = None
    async def scenario():
        nonlocal renewed
        response, _ = await _delivered(manager, visible)
        proposal = await _call(manager, "propose_edit", wire_request=_wire(response))
        worktree = tree[0]
        renewed = worktree.lifecycle_store.renew_lease(worktree.worktree_path,
            lease_id=worktree.lease_id, expected_fence=worktree.fence)
        result = await _call(manager, "read_view", path="calc.py")
        assert result["is_error"] is True and len(calls) == 1
        assert broker._closed
        assert adapter._closed and adapter._broker is None and adapter._resolver is None
        assert not adapter._candidates
        with pytest.raises(ValueError):
            adapter.consume_candidate(proposal["candidate_id"], visible_sources=dict(visible))
    try:
        asyncio.run(scenario())
    finally:
        if renewed is not None:
            tree[0].fence = renewed.fence
        adapter.close()


def test_explicit_async_wrapper_keeps_event_loop_responsive_during_source_read(tree, monkeypatch):  # noqa: F811
    broker, adapter, manager, _, _, _ = _open(tree, editable=False)
    entered, release = threading.Event(), threading.Event()
    original = broker.read_view
    event_loop_thread = threading.get_ident()
    worker_threads = []
    def observed(**arguments):
        worker_threads.append(threading.get_ident())
        entered.set()
        if not release.wait(2):
            raise AssertionError("event loop cannot release a blocking inline read")
        return original(**arguments)
    monkeypatch.setattr(broker, "read_view", observed)
    async def scenario():
        task = asyncio.create_task(_call(manager, "read_view", path="calc.py"))
        try:
            await asyncio.wait_for(asyncio.to_thread(entered.wait), 1)
            assert not task.done()
            release.set()
            response = await asyncio.wait_for(task, 10)
            assert "state" in response
            assert worker_threads and all(t != event_loop_thread for t in worker_threads)
        finally:
            release.set()
    asyncio.run(scenario())
    adapter.close()
