#!/usr/bin/env python3
"""Compatibility adapter tests for UNI-013 P2P service/registry surfaces."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.p2p_mcp_registry_adapter import (
    P2PMCPRegistryAdapter,
    RUNTIME_FASTAPI,
    RUNTIME_TRIO,
)
from ipfs_accelerate_py.mcp_server.p2p_service_manager import P2PServiceManager


class _DummyRuntime:
    def __init__(self) -> None:
        self.running = False
        self.last_error = ""

    def start(self, **_kwargs):
        self.running = True
        return types.SimpleNamespace(started=types.SimpleNamespace(wait=lambda timeout=0.0: True))

    def stop(self, timeout_s: float = 0.0) -> bool:
        _ = timeout_s
        self.running = False
        return True


class _StartRaisesRuntime:
    def __init__(self) -> None:
        self.running = False
        self.last_error = ""

    def start(self, **_kwargs):
        raise RuntimeError("boom")


class _NotRunningRuntime:
    def __init__(self) -> None:
        self.running = False
        self.last_error = ""

    def start(self, **_kwargs):
        return types.SimpleNamespace(started=types.SimpleNamespace(wait=lambda timeout=0.0: True))


class _Host:
    def __init__(self) -> None:
        async def trio_tool(**_kwargs):
            return {"ok": True}

        trio_tool._mcp_runtime = "trio"  # type: ignore[attr-defined]

        self.tools = {
            "echo": {
                "function": lambda value="": {"value": value},
                "description": "echo",
                "input_schema": {"type": "object"},
            },
            "trio.echo": {
                "function": trio_tool,
                "description": "trio echo",
                "input_schema": {"type": "object"},
            },
        }

    async def validate_p2p_message(self, msg: dict) -> bool:
        return bool(msg.get("ok"))


class TestMCPServerUNI013P2PAdapters(unittest.TestCase):
    def test_service_manager_pool_lifecycle(self) -> None:
        mgr = P2PServiceManager(enabled=True)
        self.assertTrue(mgr.release_connection("peer-a", object()))
        self.assertIsNotNone(mgr.acquire_connection("peer-a"))
        stats = mgr.get_pool_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)

    def test_service_manager_start_state_stop_contract(self) -> None:
        mgr = P2PServiceManager(enabled=True)
        dummy_runtime = _DummyRuntime()

        bundle = types.SimpleNamespace(peer_registry=object(), peer_bootstrap=object())

        with patch(
            "ipfs_accelerate_py.p2p_tasks.runtime.TaskQueueP2PServiceRuntime",
            return_value=dummy_runtime,
        ):
            with patch(
                "ipfs_accelerate_py.mcp_server.mcplusplus.create_peer_service_bundle",
                return_value=bundle,
            ):
                started = mgr.start(accelerate_instance=object())
                self.assertTrue(started)

            with patch(
                "ipfs_accelerate_py.p2p_tasks.service.get_local_service_state",
                return_value={
                    "running": True,
                    "peer_id": "peer-1",
                    "listen_port": 7000,
                    "started_at": 123.0,
                },
            ):
                with patch("ipfs_accelerate_py.p2p_tasks.service.list_known_peers", return_value=[{"peer_id": "p2"}]):
                    state = mgr.state()

            self.assertTrue(state.running)
            self.assertEqual(state.peer_id, "peer-1")
            self.assertEqual(state.connected_peers, 1)
            self.assertIs(mgr.get_peer_registry(), bundle.peer_registry)
            self.assertIs(mgr.get_peer_bootstrap(), bundle.peer_bootstrap)
            self.assertTrue(mgr.has_advanced_features())
            self.assertTrue(state.peer_registry_available)
            self.assertTrue(state.bootstrap_available)
            self.assertTrue(mgr.get_capabilities()["bootstrap"])
            self.assertTrue(mgr.stop())

    def test_service_manager_start_restores_env_when_start_raises(self) -> None:
        mgr = P2PServiceManager(enabled=True)
        os.environ.pop("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE", None)

        with patch(
            "ipfs_accelerate_py.p2p_tasks.runtime.TaskQueueP2PServiceRuntime",
            return_value=_StartRaisesRuntime(),
        ):
            started = mgr.start(accelerate_instance=object())

        self.assertFalse(started)
        self.assertNotIn("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE", os.environ)

    def test_service_manager_start_restores_env_when_runtime_not_running(self) -> None:
        mgr = P2PServiceManager(enabled=True)
        os.environ.pop("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE", None)

        with patch(
            "ipfs_accelerate_py.p2p_tasks.runtime.TaskQueueP2PServiceRuntime",
            return_value=_NotRunningRuntime(),
        ):
            with patch(
                "ipfs_accelerate_py.mcp_server.mcplusplus.create_peer_service_bundle",
                return_value=types.SimpleNamespace(peer_registry=object(), peer_bootstrap=object()),
            ):
                started = mgr.start(accelerate_instance=object())

        self.assertFalse(started)
        self.assertNotIn("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE", os.environ)
        self.assertIsNone(mgr.get_peer_registry())
        self.assertIsNone(mgr.get_peer_bootstrap())
        self.assertFalse(mgr.has_advanced_features())

    def test_registry_adapter_runtime_metadata_and_filtering(self) -> None:
        adapter = P2PMCPRegistryAdapter(_Host())
        tools = adapter.tools
        self.assertIn("echo", tools)
        self.assertIn("trio.echo", tools)
        self.assertEqual(tools["echo"]["runtime"], RUNTIME_FASTAPI)
        self.assertEqual(tools["trio.echo"]["runtime"], RUNTIME_TRIO)
        self.assertEqual(len(adapter.get_trio_tools()), 1)

    def test_registry_adapter_validate_message_contract(self) -> None:
        async def _run() -> None:
            adapter = P2PMCPRegistryAdapter(_Host())
            self.assertTrue(await adapter.validate_p2p_message({"ok": True}))
            self.assertFalse(await adapter.validate_p2p_message({"ok": False}))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
