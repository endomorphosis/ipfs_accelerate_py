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

        semantic_props = schemas["semantic_search"]["properties"]
        self.assertEqual(semantic_props["top_k"].get("maximum"), 1000)

        hybrid_props = schemas["hybrid_search"]["properties"]
        self.assertEqual(hybrid_props["top_k"].get("minimum"), 1)

        filter_props = schemas["search_with_filters"]["properties"]
        self.assertIn("semantic", filter_props["search_method"].get("enum", []))

        multimodal_props = schemas["multi_modal_search"]["properties"]
        self.assertEqual(multimodal_props["top_k"].get("maximum"), 1000)

        generate_props = schemas["generate_embeddings"]["properties"]
        self.assertEqual(generate_props["texts"].get("minItems"), 1)

        shard_props = schemas["shard_embeddings"]["properties"]
        self.assertEqual(shard_props["shard_count"].get("minimum"), 1)

    def test_generate_embeddings_validates_inputs_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("embedding boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "embedding upstream failed"}

        async def _run() -> None:
            invalid_texts = await native_embedding_tools.generate_embeddings(texts=["ok", ""])
            self.assertEqual(invalid_texts.get("status"), "error")
            self.assertIn("non-empty strings", str(invalid_texts.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"generate_embeddings": _minimal}, clear=False):
                result = await native_embedding_tools.generate_embeddings(
                    texts=["hello", "world"],
                    model_name="all-MiniLM",
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("model_name"), "all-MiniLM")
                self.assertEqual(result.get("embeddings"), [])
                self.assertEqual(result.get("count"), 2)
                self.assertEqual(result.get("dimension"), 0)

            with patch.dict(native_embedding_tools._API, {"generate_embeddings": _boom}, clear=False):
                result = await native_embedding_tools.generate_embeddings(texts=["hello"])
                self.assertEqual(result.get("status"), "error")
                self.assertIn("generate_embeddings failed", str(result.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"generate_embeddings": _contradictory_failure}, clear=False):
                result = await native_embedding_tools.generate_embeddings(texts=["hello"])
                self.assertEqual(result.get("status"), "error")
                self.assertEqual(result.get("error"), "embedding upstream failed")

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

    def test_semantic_search_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("semantic boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "semantic upstream failed"}

        async def _run() -> None:
            invalid_query = await native_embedding_tools.semantic_search(
                query=" ",
                vector_store_id="vs-1",
            )
            self.assertEqual(invalid_query.get("status"), "error")
            self.assertIn("query must be a non-empty string", str(invalid_query.get("error", "")))

            invalid_top_k = await native_embedding_tools.semantic_search(
                query="hello",
                vector_store_id="vs-1",
                top_k=0,
            )
            self.assertEqual(invalid_top_k.get("status"), "error")
            self.assertIn("top_k must be between 1 and 1000", str(invalid_top_k.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"semantic_search": _minimal}, clear=False):
                result = await native_embedding_tools.semantic_search(
                    query="hello",
                    vector_store_id="vs-1",
                    include_metadata=False,
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("query"), "hello")
                self.assertEqual(result.get("vector_store_id"), "vs-1")
                self.assertEqual(result.get("include_metadata"), False)
                self.assertEqual(result.get("results"), [])
                self.assertEqual(result.get("total_results"), 0)

            with patch.dict(native_embedding_tools._API, {"semantic_search": _boom}, clear=False):
                failed = await native_embedding_tools.semantic_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("semantic_search failed", str(failed.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"semantic_search": _contradictory_failure}, clear=False):
                failed = await native_embedding_tools.semantic_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(failed.get("status"), "error")
                self.assertEqual(failed.get("error"), "semantic upstream failed")

        anyio.run(_run)

    def test_hybrid_search_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("hybrid boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            invalid_top_k = await native_embedding_tools.hybrid_search(
                query="hello",
                vector_store_id="vs-1",
                top_k=0,
            )
            self.assertEqual(invalid_top_k.get("status"), "error")
            self.assertIn("top_k must be >= 1", str(invalid_top_k.get("error", "")))

            invalid_weight = await native_embedding_tools.hybrid_search(
                query="hello",
                vector_store_id="vs-1",
                lexical_weight="bad",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_weight.get("status"), "error")
            self.assertIn("lexical_weight must be a number", str(invalid_weight.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"hybrid_search": _minimal}, clear=False):
                result = await native_embedding_tools.hybrid_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("query"), "hello")
                self.assertEqual(result.get("vector_store_id"), "vs-1")
                self.assertEqual(result.get("results"), [])

            with patch.dict(native_embedding_tools._API, {"hybrid_search": _boom}, clear=False):
                failed = await native_embedding_tools.hybrid_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("hybrid_search failed", str(failed.get("error", "")))

        anyio.run(_run)

    def test_search_with_filters_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("filter search boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            invalid_filters = await native_embedding_tools.search_with_filters(
                query="hello",
                vector_store_id="vs-1",
                filters=[],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_filters.get("status"), "error")
            self.assertIn("filters must be an object", str(invalid_filters.get("error", "")))

            invalid_method = await native_embedding_tools.search_with_filters(
                query="hello",
                vector_store_id="vs-1",
                filters={"category": "tech"},
                search_method="vector",
            )
            self.assertEqual(invalid_method.get("status"), "error")
            self.assertIn("search_method must be one of", str(invalid_method.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"search_with_filters": _minimal}, clear=False):
                result = await native_embedding_tools.search_with_filters(
                    query="hello",
                    vector_store_id="vs-1",
                    filters={"category": "tech"},
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("query"), "hello")
                self.assertEqual(result.get("vector_store_id"), "vs-1")
                self.assertEqual(result.get("results"), [])

            with patch.dict(native_embedding_tools._API, {"search_with_filters": _boom}, clear=False):
                failed = await native_embedding_tools.search_with_filters(
                    query="hello",
                    vector_store_id="vs-1",
                    filters={"category": "tech"},
                )
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("search_with_filters failed", str(failed.get("error", "")))

        anyio.run(_run)

    def test_multi_modal_search_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("multimodal boom")

        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            invalid_inputs = await native_embedding_tools.multi_modal_search(
                vector_store_id="vs-1",
            )
            self.assertEqual(invalid_inputs.get("status"), "error")
            self.assertIn("either query or image_query must be provided", str(invalid_inputs.get("error", "")))

            invalid_weights = await native_embedding_tools.multi_modal_search(
                query="hello",
                vector_store_id="vs-1",
                modality_weights="bad",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_weights.get("status"), "error")
            self.assertIn("modality_weights must be an object", str(invalid_weights.get("error", "")))

            with patch.dict(native_embedding_tools._API, {"multi_modal_search": _minimal}, clear=False):
                result = await native_embedding_tools.multi_modal_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("text_query"), "hello")
                self.assertEqual(result.get("vector_store_id"), "vs-1")
                self.assertEqual(result.get("results"), [])

            with patch.dict(native_embedding_tools._API, {"multi_modal_search": _boom}, clear=False):
                failed = await native_embedding_tools.multi_modal_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("multi_modal_search failed", str(failed.get("error", "")))

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

    def test_embedding_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "embedding delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_embedding_tools._API,
                {
                    "generate_embedding": _contradictory_failure,
                    "generate_embeddings_from_file": _contradictory_failure,
                    "hybrid_search": _contradictory_failure,
                    "search_with_filters": _contradictory_failure,
                    "multi_modal_search": _contradictory_failure,
                    "shard_embeddings": _contradictory_failure,
                    "chunk_text": _contradictory_failure,
                    "manage_endpoints": _contradictory_failure,
                },
                clear=False,
            ):
                single = await native_embedding_tools.generate_embedding(text="hello")
                self.assertEqual(single.get("status"), "error")

                self.assertEqual(single.get("error"), "embedding delegate failed")

                from_file = await native_embedding_tools.generate_embeddings_from_file(file_path="input.txt")
                self.assertEqual(from_file.get("status"), "error")
                self.assertEqual(from_file.get("error"), "embedding delegate failed")

                hybrid = await native_embedding_tools.hybrid_search(query="hello", vector_store_id="vs-1")
                self.assertEqual(hybrid.get("status"), "error")
                self.assertEqual(hybrid.get("error"), "embedding delegate failed")

                filtered = await native_embedding_tools.search_with_filters(
                    query="hello",
                    vector_store_id="vs-1",
                    filters={"category": "tech"},
                )
                self.assertEqual(filtered.get("status"), "error")
                self.assertEqual(filtered.get("error"), "embedding delegate failed")

                multimodal = await native_embedding_tools.multi_modal_search(
                    query="hello",
                    vector_store_id="vs-1",
                )
                self.assertEqual(multimodal.get("status"), "error")
                self.assertEqual(multimodal.get("error"), "embedding delegate failed")

                sharded = await native_embedding_tools.shard_embeddings(
                    embeddings=[[0.1, 0.2]],
                    shard_count=1,
                )
                self.assertEqual(sharded.get("status"), "error")
                self.assertEqual(sharded.get("error"), "embedding delegate failed")

                chunked = await native_embedding_tools.chunk_text_for_embeddings(text="hello world")
                self.assertEqual(chunked.get("status"), "error")
                self.assertEqual(chunked.get("error"), "embedding delegate failed")

                endpoints = await native_embedding_tools.manage_embedding_endpoints(
                    action="list",
                    model="all-MiniLM",
                )
                self.assertEqual(endpoints.get("status"), "error")
                self.assertEqual(endpoints.get("error"), "embedding delegate failed")

        anyio.run(_run)

    def test_shard_chunk_and_endpoint_success_defaults_are_preserved(self) -> None:
        async def _minimal(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch.dict(native_embedding_tools._API, {"shard_embeddings": _minimal}, clear=False):
                shard_result = await native_embedding_tools.shard_embeddings(
                    embeddings=[[0.1, 0.2], [0.3, 0.4]],
                    shard_count=2,
                    strategy="balanced",
                )
                self.assertEqual(shard_result.get("status"), "success")
                self.assertEqual(shard_result.get("shard_count"), 2)
                self.assertEqual(shard_result.get("total_embeddings"), 2)
                self.assertEqual(shard_result.get("strategy"), "balanced")
                self.assertEqual(shard_result.get("shards"), [])

            with patch.dict(native_embedding_tools._API, {"chunk_text": _minimal}, clear=False):
                chunk_result = await native_embedding_tools.chunk_text_for_embeddings(text="hello world")
                self.assertEqual(chunk_result.get("status"), "success")
                self.assertEqual(chunk_result.get("original_length"), len("hello world"))
                self.assertEqual(chunk_result.get("chunks"), [])
                self.assertEqual(chunk_result.get("chunk_count"), 0)

            with patch.dict(native_embedding_tools._API, {"manage_endpoints": _minimal}, clear=False):
                list_result = await native_embedding_tools.manage_embedding_endpoints(
                    action="list",
                    model="all-MiniLM",
                )
                self.assertEqual(list_result.get("status"), "success")
                self.assertEqual(list_result.get("action"), "list")
                self.assertEqual(list_result.get("model"), "all-MiniLM")
                self.assertEqual(list_result.get("endpoints"), [])

                test_result = await native_embedding_tools.manage_embedding_endpoints(
                    action="test",
                    model="all-MiniLM",
                    endpoint="https://example.invalid/embed",
                )
                self.assertEqual(test_result.get("status"), "success")
                self.assertEqual(test_result.get("action"), "test")
                self.assertEqual(test_result.get("endpoint"), "https://example.invalid/embed")
                self.assertEqual(test_result.get("available"), False)
        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
