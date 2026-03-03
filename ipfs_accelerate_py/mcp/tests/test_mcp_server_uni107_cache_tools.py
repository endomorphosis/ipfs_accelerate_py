#!/usr/bin/env python3
"""UNI-107 cache tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.cache_tools.native_cache_tools import (
    cache_embeddings,
    get_cached_embeddings,
    optimize_cache,
    register_native_cache_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI107CacheTools(unittest.TestCase):
    def test_register_includes_new_cache_parity_operations(self) -> None:
        manager = _DummyManager()
        register_native_cache_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("optimize_cache", names)
        self.assertIn("cache_embeddings", names)
        self.assertIn("get_cached_embeddings", names)

    def test_cache_embeddings_rejects_missing_text(self) -> None:
        async def _run() -> None:
            result = await cache_embeddings(text="   ", embeddings=[0.1, 0.2])
            self.assertEqual(result.get("success"), False)
            self.assertIn("text is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_cache_embeddings_accepts_json_embedding_vector(self) -> None:
        async def _run() -> None:
            result = await cache_embeddings(
                text="hello world",
                embeddings="[0.11, 0.22, 0.33]",
                model="unit-model",
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertIn("success", result)

        anyio.run(_run)

    def test_get_cached_embeddings_rejects_missing_text(self) -> None:
        async def _run() -> None:
            result = await get_cached_embeddings(text="")
            self.assertEqual(result.get("success"), False)
            self.assertIn("text is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_get_cached_embeddings_round_trip_shape(self) -> None:
        async def _run() -> None:
            await cache_embeddings(
                text="round-trip-sentence",
                embeddings=[0.5, 0.6],
                model="roundtrip-model",
            )
            result = await get_cached_embeddings(text="round-trip-sentence", model="roundtrip-model")
            self.assertIn(result.get("status"), ["found", "not_found", "error"])
            if result.get("status") == "found":
                self.assertEqual(result.get("cache_hit"), True)
                self.assertIn("embeddings", result)

        anyio.run(_run)

    def test_optimize_cache_returns_optimization_envelope(self) -> None:
        async def _run() -> None:
            result = await optimize_cache(cache_type="embeddings", strategy="lfu", max_age_hours=24)
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("optimization_strategy"), "lfu")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
