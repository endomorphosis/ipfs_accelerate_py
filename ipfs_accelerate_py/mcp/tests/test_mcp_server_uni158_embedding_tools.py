#!/usr/bin/env python3
"""UNI-158 deterministic parity tests for native embedding tools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.embedding_tools import native_embedding_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI158EmbeddingTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_embedding_tools.register_native_embedding_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        single_props = schemas["generate_embedding"]["properties"]
        self.assertEqual(single_props["batch_size"].get("minimum"), 1)

        from_file_props = schemas["generate_embeddings_from_file"]["properties"]
        self.assertIn("json", from_file_props["output_format"].get("enum", []))

        generate_props = schemas["generate_embeddings"]["properties"]
        self.assertEqual(generate_props["texts"].get("minItems"), 1)

        shard_props = schemas["shard_embeddings"]["properties"]
        self.assertEqual(shard_props["shard_count"].get("minimum"), 1)

    def test_generate_embeddings_validates_inputs_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("embedding boom")

        async def _run() -> None:
            invalid_texts = await native_embedding_tools.generate_embeddings(texts=["ok", ""])
            self.assertEqual(invalid_texts.get("status"), "error")
            self.assertIn("non-empty strings", str(invalid_texts.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"generate_embeddings": _boom}, clear=False):
                result = await native_embedding_tools.generate_embeddings(texts=["hello"])
                self.assertEqual(result.get("status"), "error")
                self.assertIn("generate_embeddings failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_generate_embedding_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("single embedding boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            invalid_text = await native_embedding_tools.generate_embedding(text="   ")
            self.assertEqual(invalid_text.get("status"), "error")
            self.assertIn("text must be a non-empty string", str(invalid_text.get("error", "")))

            invalid_batch = await native_embedding_tools.generate_embedding(text="hello", batch_size=0)
            self.assertEqual(invalid_batch.get("status"), "error")
            self.assertIn("batch_size must be a positive integer", str(invalid_batch.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"generate_embedding": _minimal}, clear=False):
                result = await native_embedding_tools.generate_embedding(text="hello", batch_size=8)
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("batch_size"), 8)
                self.assertEqual(result.get("embedding"), [])

            with patch.dict(native_embedding_tools._API, {"generate_embedding": _boom}, clear=False):
                failed = await native_embedding_tools.generate_embedding(text="hello")
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("generate_embedding failed", str(failed.get("error", "")))

        anyio.run(_run)

    def test_generate_embeddings_from_file_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("file embedding boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            invalid_path = await native_embedding_tools.generate_embeddings_from_file(file_path="  ")
            self.assertEqual(invalid_path.get("status"), "error")
            self.assertIn("file_path must be a non-empty string", str(invalid_path.get("error", "")))

            invalid_format = await native_embedding_tools.generate_embeddings_from_file(
                file_path="input.txt",
                output_format="csv",
            )
            self.assertEqual(invalid_format.get("status"), "error")
            self.assertIn("output_format must be one of", str(invalid_format.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"generate_embeddings_from_file": _minimal}, clear=False):
                result = await native_embedding_tools.generate_embeddings_from_file(
                    file_path="input.txt",
                    output_format="json",
                    batch_size=16,
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("file_path"), "input.txt")
                self.assertEqual(result.get("batch_size"), 16)
                self.assertEqual(result.get("embeddings"), [])

            with patch.dict(native_embedding_tools._API, {"generate_embeddings_from_file": _boom}, clear=False):
                failed = await native_embedding_tools.generate_embeddings_from_file(file_path="input.txt")
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("generate_embeddings_from_file failed", str(failed.get("error", "")))

        anyio.run(_run)

    def test_chunk_text_and_shard_validate_contracts(self) -> None:
        async def _run() -> None:
            invalid_overlap = await native_embedding_tools.chunk_text_for_embeddings(
                text="abc def ghi",
                chunk_size=5,
                chunk_overlap=5,
            )
            self.assertEqual(invalid_overlap.get("status"), "error")
            self.assertIn("smaller than chunk_size", str(invalid_overlap.get("error", "")))

            invalid_shard_count = await native_embedding_tools.shard_embeddings(
                embeddings=[[0.1, 0.2]],
                shard_count=0,
            )
            self.assertEqual(invalid_shard_count.get("status"), "error")
            self.assertIn("shard_count must be a positive integer", str(invalid_shard_count.get("error", "")))

        anyio.run(_run)

    def test_manage_endpoints_validates_context_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("endpoint boom")

        async def _run() -> None:
            invalid_context = await native_embedding_tools.manage_embedding_endpoints(
                action="list",
                model="all-MiniLM",
                context_length="bad",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_context.get("status"), "error")
            self.assertIn("context_length must be an integer", str(invalid_context.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"manage_endpoints": _boom}, clear=False):
                result = await native_embedding_tools.manage_embedding_endpoints(
                    action="list",
                    model="all-MiniLM",
                )
                self.assertEqual(result.get("status"), "error")
                self.assertIn("manage_embedding_endpoints failed", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
