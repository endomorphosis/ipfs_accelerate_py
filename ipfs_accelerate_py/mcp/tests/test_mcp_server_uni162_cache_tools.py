#!/usr/bin/env python3
"""UNI-162 cache tools parity-expansion tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools import (
    cache_get,
    get_cached_embeddings,
    manage_cache,
    optimize_cache,
)


class TestMCPServerUNI162CacheTools(unittest.TestCase):
    def test_cache_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        class _ContradictoryManager:
            def get(self, key: str, namespace: str = "default") -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"get failed for {namespace}:{key}",
                }

            def get_stats(self, namespace: str | None = None) -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"stats failed for {namespace or 'all'}",
                }

            def optimize(self, **_: object) -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": "optimize failed",
                }

            def get_cached_embeddings(self, text: str, model: str = "default") -> dict:
                return {
                    "status": "success",
                    "success": False,
                    "error": f"embedding miss for {model}:{text}",
                }

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools._get_cache_manager",
                return_value=_ContradictoryManager(),
            ):
                cached = await cache_get(key="demo")
                managed = await manage_cache(action="stats")
                optimized = await optimize_cache(cache_type="embeddings")
                embedded = await get_cached_embeddings(text="hello", model="demo")

            self.assertEqual(cached.get("status"), "error")
            self.assertEqual(cached.get("success"), False)
            self.assertIn("get failed", str(cached.get("error", "")))

            self.assertEqual(managed.get("status"), "error")
            self.assertEqual(managed.get("success"), False)
            self.assertIn("stats failed", str(managed.get("error", "")))

            self.assertEqual(optimized.get("status"), "error")
            self.assertEqual(optimized.get("success"), False)
            self.assertEqual(optimized.get("error"), "optimize failed")

            self.assertEqual(embedded.get("status"), "error")
            self.assertEqual(embedded.get("success"), False)
            self.assertIn("embedding miss", str(embedded.get("error", "")))

        anyio.run(_run)

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
