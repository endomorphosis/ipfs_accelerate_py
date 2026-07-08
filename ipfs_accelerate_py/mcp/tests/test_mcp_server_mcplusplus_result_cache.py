#!/usr/bin/env python3
"""Unit tests for unified MCP++ result cache primitive."""

import tempfile
import unittest
from pathlib import Path

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.result_cache import (
    DiskCacheBackend,
    MemoryCacheBackend,
    ResultCache,
)


class TestResultCachePrimitive(unittest.TestCase):
    """Validate memory and disk cache behavior."""

    def test_memory_cache_put_get_invalidate(self) -> None:
        async def _run() -> None:
            cache = ResultCache(backend=MemoryCacheBackend(max_size=10), default_ttl=5.0)
            await cache.put("task-a", {"value": 123})
            value = await cache.get("task-a")
            self.assertEqual(value, {"value": 123})
            self.assertTrue(await cache.invalidate("task-a"))
            self.assertIsNone(await cache.get("task-a"))

        anyio.run(_run)

    def test_memory_cache_ttl_expiration(self) -> None:
        async def _run() -> None:
            cache = ResultCache(backend=MemoryCacheBackend(max_size=10), default_ttl=1.0)
            await cache.put("task-ttl", "temp", ttl=0.1)
            self.assertEqual(await cache.get("task-ttl"), "temp")
            await anyio.sleep(0.2)
            self.assertIsNone(await cache.get("task-ttl"))

        anyio.run(_run)

    def test_disk_cache_roundtrip(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory(prefix="mcp_cache_test_") as tmp:
                backend = DiskCacheBackend(Path(tmp), max_size=5)
                cache = ResultCache(backend=backend, default_ttl=5.0)
                await cache.put("task-disk", {"ok": True})
                value = await cache.get("task-disk")
                self.assertEqual(value, {"ok": True})
                stats = await cache.get_stats()
                self.assertEqual(stats["backend"], "disk")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
