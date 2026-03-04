#!/usr/bin/env python3
"""UNI-124 sparse embedding tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools.native_sparse_embedding_tools import (
    generate_sparse_embedding,
    manage_sparse_models,
    register_native_sparse_embedding_tools,
    sparse_search,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI124SparseEmbeddingTools(unittest.TestCase):
    def test_register_includes_sparse_embedding_tools(self) -> None:
        manager = _DummyManager()
        register_native_sparse_embedding_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("generate_sparse_embedding", names)
        self.assertIn("sparse_search", names)
        self.assertIn("manage_sparse_models", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_sparse_embedding_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        generate_schema = by_name["generate_sparse_embedding"]["input_schema"]
        self.assertIn("splade", generate_schema["properties"]["model"].get("enum", []))

        search_schema = by_name["sparse_search"]["input_schema"]
        self.assertEqual(search_schema["properties"]["top_k"].get("minimum"), 1)

    def test_generate_rejects_empty_text(self) -> None:
        async def _run() -> None:
            result = await generate_sparse_embedding(text="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("text is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_search_rejects_invalid_model(self) -> None:
        async def _run() -> None:
            result = await sparse_search(
                query="hello",
                collection_name="docs",
                model="bert",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("model must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await manage_sparse_models(action="delete")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_configure_requires_model_and_config(self) -> None:
        async def _run() -> None:
            result = await manage_sparse_models(action="configure")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("model_name and config object are required", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await generate_sparse_embedding(text="hello world", model="splade", top_k=5)
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("model"), "splade")
            self.assertEqual(result.get("top_k"), 5)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
