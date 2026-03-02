"""Tests for MCP++ tool adapter delegation.

These tests verify that mcplusplus_module tool registration functions preserve
legacy tool names while delegating behavior to canonical mcp_server adapters.
"""

from __future__ import annotations

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
