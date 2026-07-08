#!/usr/bin/env python3
"""Tests for canonical trio adapter facades under mcp_server."""

import unittest
from unittest.mock import AsyncMock

import anyio

from ipfs_accelerate_py.mcp_server.trio_adapter import (
    TRIO_AVAILABLE,
    TrioMCPServerAdapter,
    TrioServerConfig,
)


class _DummyServer:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class TestTrioAdapterFacade(unittest.TestCase):
    def test_trio_available_flag_is_bool(self) -> None:
        self.assertIsInstance(TRIO_AVAILABLE, bool)

    def test_start_and_stop_contract(self) -> None:
        async def _run() -> None:
            created = {}

            def _factory(**kwargs):
                created.update(kwargs)
                return _DummyServer()

            serve_fn = AsyncMock(return_value={"ok": True})
            adapter = TrioMCPServerAdapter(
                TrioServerConfig(name="trio-test", description="desc"),
                server_factory=_factory,
                serve_fn=serve_fn,
            )

            result = await adapter.start()
            self.assertEqual(result, {"ok": True})
            self.assertTrue(adapter.running)
            self.assertEqual(created.get("name"), "trio-test")
            self.assertEqual(created.get("description"), "desc")
            serve_fn.assert_awaited_once()

            server = adapter.server
            self.assertIsNotNone(server)
            assert server is not None
            await adapter.stop()
            self.assertFalse(adapter.running)
            self.assertTrue(server.stopped)

        anyio.run(_run)

    def test_start_resets_state_on_sync_serve_failure(self) -> None:
        async def _run() -> None:
            def _factory(**_kwargs):
                return _DummyServer()

            def _serve(_server, _config):
                raise RuntimeError("serve failed")

            adapter = TrioMCPServerAdapter(
                TrioServerConfig(name="trio-test", description="desc"),
                server_factory=_factory,
                serve_fn=_serve,
            )

            with self.assertRaisesRegex(RuntimeError, "serve failed"):
                await adapter.start()

            self.assertFalse(adapter.running)
            self.assertIsNone(adapter.server)

        anyio.run(_run)

    def test_start_resets_state_on_async_serve_failure(self) -> None:
        async def _run() -> None:
            def _factory(**_kwargs):
                return _DummyServer()

            async def _serve(_server, _config):
                raise RuntimeError("async serve failed")

            adapter = TrioMCPServerAdapter(
                TrioServerConfig(name="trio-test", description="desc"),
                server_factory=_factory,
                serve_fn=_serve,
            )

            with self.assertRaisesRegex(RuntimeError, "async serve failed"):
                await adapter.start()

            self.assertFalse(adapter.running)
            self.assertIsNone(adapter.server)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
