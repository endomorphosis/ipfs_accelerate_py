#!/usr/bin/env python3
"""UNI-141 bespoke_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.bespoke_tools.native_bespoke_tools import (
    cache_stats,
    register_native_bespoke_tools,
    system_health,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI141BespokeTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_bespoke_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        cache_schema = by_name["cache_stats"]["input_schema"]
        self.assertEqual(cache_schema["properties"]["namespace"]["minLength"], 1)

    def test_cache_stats_validation(self) -> None:
        async def _run() -> None:
            result = await cache_stats(namespace="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("namespace", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            health_result = await system_health()
            self.assertIn(health_result.get("status"), ["success", "error"])

            cache_result = await cache_stats(namespace="primary")
            self.assertIn(cache_result.get("status"), ["success", "error"])
            self.assertEqual(cache_result.get("namespace"), "primary")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
