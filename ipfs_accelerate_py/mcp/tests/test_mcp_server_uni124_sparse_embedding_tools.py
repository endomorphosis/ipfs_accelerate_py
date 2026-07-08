#!/usr/bin/env python3
"""UNI-124 sparse embedding tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools.native_sparse_embedding_tools import (
    generate_sparse_embedding,
    index_sparse_collection,
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

    def test_generate_and_search_defaults_with_minimal_payloads(self) -> None:
        async def _minimal_generate(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_search(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools.native_sparse_embedding_tools._API",
                {
                    "generate_sparse_embedding": _minimal_generate,
                    "index_sparse_collection": None,
                    "sparse_search": _minimal_search,
                    "manage_sparse_models": None,
                },
            ):
                generated = await generate_sparse_embedding(text="hello", model="splade", top_k=7)
                searched = await sparse_search(query="hello", collection_name="docs", model="splade", top_k=3)

            self.assertEqual(generated.get("status"), "success")
            self.assertEqual(generated.get("text"), "hello")
            self.assertEqual(generated.get("model"), "splade")
            self.assertEqual(generated.get("top_k"), 7)
            self.assertEqual(generated.get("sparse_embedding"), {})

            self.assertEqual(searched.get("status"), "success")
            self.assertEqual(searched.get("query"), "hello")
            self.assertEqual(searched.get("collection_name"), "docs")
            self.assertEqual(searched.get("model"), "splade")
            self.assertEqual(searched.get("top_k"), 3)
            self.assertEqual(searched.get("results"), [])
            self.assertEqual(searched.get("total_found"), 0)
            self.assertEqual(searched.get("has_more"), False)

        anyio.run(_run)

    def test_index_and_manage_defaults_with_minimal_payloads(self) -> None:
        async def _minimal_index(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_manage(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools.native_sparse_embedding_tools._API",
                {
                    "generate_sparse_embedding": None,
                    "index_sparse_collection": _minimal_index,
                    "sparse_search": None,
                    "manage_sparse_models": _minimal_manage,
                },
            ):
                indexed = await index_sparse_collection(collection_name="docs", dataset="sample")
                listed = await manage_sparse_models(action="list")
                stats = await manage_sparse_models(action="stats")

            self.assertEqual(indexed.get("status"), "success")
            self.assertEqual(indexed.get("collection_name"), "docs")
            self.assertEqual(indexed.get("dataset"), "sample")
            self.assertEqual(indexed.get("split"), "train")
            self.assertEqual(indexed.get("column"), "text")
            self.assertEqual(indexed.get("batch_size"), 100)
            self.assertEqual(indexed.get("total_documents"), 0)
            self.assertEqual(indexed.get("results"), {})

            self.assertEqual(listed.get("status"), "success")
            self.assertEqual(listed.get("action"), "list")
            self.assertEqual(listed.get("models"), [])

            self.assertEqual(stats.get("status"), "success")
            self.assertEqual(stats.get("action"), "stats")
            self.assertEqual(stats.get("stats"), {})

        anyio.run(_run)

    def test_sparse_embedding_wrappers_infer_error_status_from_contradictory_delegate_payload(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools.native_sparse_embedding_tools._API",
                {
                    "generate_sparse_embedding": _contradictory_failure,
                    "index_sparse_collection": _contradictory_failure,
                    "sparse_search": _contradictory_failure,
                    "manage_sparse_models": _contradictory_failure,
                },
            ):
                generated = await generate_sparse_embedding(text="hello")
                indexed = await index_sparse_collection(collection_name="docs", dataset="sample")
                searched = await sparse_search(query="hello", collection_name="docs")
                managed = await manage_sparse_models(action="list")

            self.assertEqual(generated.get("status"), "error")
            self.assertEqual(generated.get("error"), "delegate failed")

            self.assertEqual(indexed.get("status"), "error")
            self.assertEqual(indexed.get("error"), "delegate failed")

            self.assertEqual(searched.get("status"), "error")
            self.assertEqual(searched.get("error"), "delegate failed")

            self.assertEqual(managed.get("status"), "error")
            self.assertEqual(managed.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
