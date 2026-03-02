"""Tests for MCP++ tool adapter delegation.

These tests verify that mcplusplus_module tool registration functions preserve
legacy tool names while delegating behavior to canonical mcp_server adapters.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest


class _DummyMCP:
    """Minimal MCP-like registrar used by adapter tests."""

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def tool(self):
        def _decorator(func):
            self.tools[func.__name__] = {"func": func}
            return func

        return _decorator


@pytest.mark.trio
async def test_taskqueue_status_delegates_to_canonical(monkeypatch):
    """`p2p_taskqueue_status` should pass through args to canonical adapter."""
    from ipfs_accelerate_py.mcplusplus_module.tools import taskqueue_tools

    captured: Dict[str, Any] = {}

    async def _fake_status(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "delegated": True}

    monkeypatch.setattr(taskqueue_tools.canonical, "p2p_taskqueue_status", _fake_status)

    mcp = _DummyMCP()
    taskqueue_tools.register_p2p_taskqueue_tools(mcp)

    result = await mcp.tools["p2p_taskqueue_status"]["func"](
        remote_multiaddr="/ip4/127.0.0.1/tcp/9000/p2p/peer",
        peer_id="peer-abc",
        timeout_s=2.5,
        detail=True,
    )

    assert result["ok"] is True
    assert result["delegated"] is True
    assert captured == {
        "remote_multiaddr": "/ip4/127.0.0.1/tcp/9000/p2p/peer",
        "peer_id": "peer-abc",
        "timeout_s": 2.5,
        "detail": True,
    }


@pytest.mark.trio
async def test_taskqueue_submit_docker_hub_delegates_kwargs(monkeypatch):
    """Docker-hub wrapper should forward all canonical arguments and kwargs."""
    from ipfs_accelerate_py.mcplusplus_module.tools import taskqueue_tools

    captured: Dict[str, Any] = {}

    async def _fake_submit_docker_hub(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "task_id": "delegated-task"}

    monkeypatch.setattr(
        taskqueue_tools.canonical,
        "p2p_taskqueue_submit_docker_hub",
        _fake_submit_docker_hub,
    )

    mcp = _DummyMCP()
    taskqueue_tools.register_p2p_taskqueue_tools(mcp)

    result = await mcp.tools["p2p_taskqueue_submit_docker_hub"]["func"](
        image="python:3.12",
        command=["python", "-V"],
        environment={"MODE": "test"},
        remote_peer_id="peer-xyz",
        extra_flag=True,
    )

    assert result == {"ok": True, "task_id": "delegated-task"}
    assert captured["image"] == "python:3.12"
    assert captured["command"] == ["python", "-V"]
    assert captured["environment"] == {"MODE": "test"}
    assert captured["remote_peer_id"] == "peer-xyz"
    assert captured["extra_flag"] is True


@pytest.mark.trio
async def test_workflow_core_tools_delegate_to_canonical(monkeypatch):
    """Workflow status/submit/get-next tools should call canonical adapters."""
    from ipfs_accelerate_py.mcplusplus_module.tools import workflow_tools

    captured: Dict[str, Any] = {}

    async def _fake_status():
        captured["status_called"] = True
        return {"success": True, "status": {"queue_size": 3}}

    async def _fake_schedule(**kwargs):
        captured["schedule_kwargs"] = kwargs
        return {"success": True, "workflow_id": kwargs.get("workflow_id")}

    async def _fake_next():
        captured["next_called"] = True
        return {"success": True, "workflow": {"workflow_id": "wf-1"}, "message": "ok"}

    monkeypatch.setattr(workflow_tools.canonical, "get_p2p_scheduler_status", _fake_status)
    monkeypatch.setattr(workflow_tools.canonical, "schedule_p2p_workflow", _fake_schedule)
    monkeypatch.setattr(workflow_tools.canonical, "get_next_p2p_workflow", _fake_next)

    mcp = _DummyMCP()
    workflow_tools.register_p2p_workflow_tools(mcp)

    status = await mcp.tools["p2p_scheduler_status"]["func"]()
    submit = await mcp.tools["p2p_submit_task"]["func"](
        task_id="task-1",
        workflow_id="wf-1",
        name="test",
        tags=["p2p-only"],
        priority=7,
    )
    next_task = await mcp.tools["p2p_get_next_task"]["func"]()

    assert captured["status_called"] is True
    assert captured["next_called"] is True
    assert status["success"] is True
    assert status["tool"] == "p2p_scheduler_status"
    assert submit["success"] is True
    assert submit["workflow_id"] == "wf-1"
    assert captured["schedule_kwargs"]["workflow_id"] == "wf-1"
    assert captured["schedule_kwargs"]["metadata"] == {"task_id": "task-1"}
    assert next_task["task"] == {"workflow_id": "wf-1"}


def test_tools_resolver_prefers_explicit_modules(monkeypatch):
    """Tools resolver should prefer explicit module imports when available."""
    from ipfs_accelerate_py.mcplusplus_module import tools

    calls = []

    def _taskqueue(_mcp):
        return None

    def _workflow(_mcp):
        return None

    def _fake_import_module(name: str):
        calls.append(name)
        if name.endswith("tools.taskqueue_tools"):
            return SimpleNamespace(register_p2p_taskqueue_tools=_taskqueue)
        if name.endswith("tools.workflow_tools"):
            return SimpleNamespace(register_p2p_workflow_tools=_workflow)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(tools, "import_module", _fake_import_module)

    taskqueue_registrar, workflow_registrar = tools._resolve_p2p_registrars()
    assert taskqueue_registrar is _taskqueue
    assert workflow_registrar is _workflow
    assert calls == [
        "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools",
        "ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools",
    ]


def test_tools_resolver_falls_back_to_package_symbols(monkeypatch):
    """Tools resolver should return package symbols if imports fail."""
    from ipfs_accelerate_py.mcplusplus_module import tools

    def _fake_import_module(_name: str):
        raise ImportError("simulated failure")

    monkeypatch.setattr(tools, "import_module", _fake_import_module)

    taskqueue_registrar, workflow_registrar = tools._resolve_p2p_registrars()
    assert taskqueue_registrar is tools.register_p2p_taskqueue_tools
    assert workflow_registrar is tools.register_p2p_workflow_tools


def test_tools_resolver_partial_import_failure_uses_package_symbols(monkeypatch):
    """Any explicit-module import failure should fall back to package symbols."""
    from ipfs_accelerate_py.mcplusplus_module import tools

    calls = []

    def _fake_import_module(name: str):
        calls.append(name)
        if name.endswith("tools.taskqueue_tools"):
            return SimpleNamespace(register_p2p_taskqueue_tools=lambda _mcp: None)
        if name.endswith("tools.workflow_tools"):
            raise ImportError("simulated workflow import failure")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(tools, "import_module", _fake_import_module)

    taskqueue_registrar, workflow_registrar = tools._resolve_p2p_registrars()
    assert taskqueue_registrar is tools.register_p2p_taskqueue_tools
    assert workflow_registrar is tools.register_p2p_workflow_tools
    assert calls == [
        "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools",
        "ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools",
    ]


def test_register_all_p2p_tools_uses_resolver(monkeypatch):
    """Aggregate registration should call registrar callables from resolver."""
    from ipfs_accelerate_py.mcplusplus_module import tools

    calls = []

    def _taskqueue(_mcp):
        calls.append("taskqueue")

    def _workflow(_mcp):
        calls.append("workflow")

    monkeypatch.setattr(tools, "_resolve_p2p_registrars", lambda: (_taskqueue, _workflow))
    tools.register_all_p2p_tools(object())

    assert calls == ["taskqueue", "workflow"]


def test_taskqueue_adapter_resolver_fallback(monkeypatch):
    """Taskqueue module resolver should return fallback adapter on import failure."""
    from ipfs_accelerate_py.mcplusplus_module.tools import taskqueue_tools

    def _fake_import_module(_name: str):
        raise ImportError("simulated failure")

    monkeypatch.setattr(taskqueue_tools, "import_module", _fake_import_module)
    adapter = taskqueue_tools._resolve_canonical_p2p_adapter()

    assert hasattr(adapter, "p2p_taskqueue_status")
    assert hasattr(adapter, "list_peers")


def test_workflow_adapter_resolver_fallback(monkeypatch):
    """Workflow module resolver should return fallback adapter on import failure."""
    from ipfs_accelerate_py.mcplusplus_module.tools import workflow_tools

    def _fake_import_module(_name: str):
        raise ImportError("simulated failure")

    monkeypatch.setattr(workflow_tools, "import_module", _fake_import_module)
    adapter = workflow_tools._resolve_canonical_workflow_adapter()

    assert hasattr(adapter, "get_p2p_scheduler_status")
    assert hasattr(adapter, "schedule_p2p_workflow")
    assert hasattr(adapter, "get_next_p2p_workflow")


def test_p2p_missing_dependency_stub_contract():
    """P2P compatibility stub should be falsy and raise clear runtime errors."""
    import ipfs_accelerate_py.mcplusplus_module as mcplusplus_module
    from ipfs_accelerate_py.mcplusplus_module import p2p

    assert p2p._missing_dependency_stub is mcplusplus_module._missing_dependency_stub

    stub = p2p._missing_dependency_stub("ExampleSymbol")
    assert not stub
    assert "ExampleSymbol" in repr(stub)

    with pytest.raises(RuntimeError, match="ExampleSymbol is unavailable"):
        stub()

    with pytest.raises(RuntimeError, match="ExampleSymbol is unavailable"):
        _ = stub.some_attribute


def test_mcplusplus_module_missing_dependency_stub_contract():
    """Top-level MCP++ compatibility stub should be falsy and explicit."""
    import ipfs_accelerate_py.mcplusplus_module as mcplusplus_module

    stub = mcplusplus_module._missing_dependency_stub("TopLevelSymbol")
    assert not stub
    assert "TopLevelSymbol" in repr(stub)

    with pytest.raises(RuntimeError, match="TopLevelSymbol is unavailable"):
        stub()

    with pytest.raises(RuntimeError, match="TopLevelSymbol is unavailable"):
        _ = stub.some_attribute


def test_workflow_module_optional_dependency_contract():
    """Workflow module should expose explicit stubs when scheduler deps are absent."""
    from ipfs_accelerate_py.mcplusplus_module.p2p import workflow

    if workflow.HAVE_P2P_SCHEDULER:
        assert workflow.P2PWorkflowScheduler is not None
        assert workflow.P2PTask is not None
        assert workflow.WorkflowTag is not None
        assert workflow.MerkleClock is not None
        return

    assert workflow.P2PWorkflowScheduler is not None
    assert workflow.P2PTask is not None
    assert workflow.WorkflowTag is not None
    assert workflow.MerkleClock is not None
    assert not workflow.P2PWorkflowScheduler

    with pytest.raises(RuntimeError, match="P2PWorkflowScheduler is unavailable"):
        workflow.P2PWorkflowScheduler()
