#!/usr/bin/env python3
"""UNI-162 cache tools parity-expansion tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools import manage_cache


class TestMCPServerUNI162CacheTools(unittest.TestCase):
    def test_manage_cache_configure_rejects_invalid_configuration(self) -> None:
        async def _run() -> None:
            result = await manage_cache(action="configure", configuration="invalid")
            self.assertEqual(result.get("success"), False)
            self.assertIn("configuration must be an object", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_cache_warm_up_rejects_invalid_keys(self) -> None:
        async def _run() -> None:
            result = await manage_cache(action="warm_up", configuration={"keys": ["ok", "   "]})
            self.assertEqual(result.get("success"), False)
            self.assertIn("configuration.keys must be a list of non-empty strings", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_cache_analyze_returns_deterministic_analysis(self) -> None:
        async def _run() -> None:
            result = await manage_cache(action="analyze")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("operation"), "analyze")
            self.assertIn("analysis", result)
            self.assertIn("cache_health", result["analysis"])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
