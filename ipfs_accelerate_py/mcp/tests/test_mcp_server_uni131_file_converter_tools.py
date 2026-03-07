#!/usr/bin/env python3
"""UNI-131 file converter tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.file_converter_tools import native_file_converter_tools as nfc


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI131FileConverterTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        nfc.register_native_file_converter_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        batch_schema = by_name["batch_convert_tool"]["input_schema"]
        self.assertEqual(batch_schema["properties"]["input_paths"].get("minItems"), 1)
        self.assertEqual(batch_schema["properties"]["max_concurrent"].get("minimum"), 1)

        convert_schema = by_name["convert_file_tool"]["input_schema"]
        self.assertEqual(convert_schema["properties"]["input_path"].get("minLength"), 1)
        self.assertEqual(convert_schema["properties"]["backend"].get("minLength"), 1)
        self.assertEqual(convert_schema["properties"]["output_format"].get("minLength"), 1)

        embedding_schema = by_name["generate_embeddings_tool"]["input_schema"]
        self.assertIn("faiss", embedding_schema["properties"]["vector_store"].get("enum", []))

        archive_schema = by_name["extract_archive_tool"]["input_schema"]
        self.assertEqual(archive_schema["properties"]["max_depth"].get("minimum"), 0)

        download_schema = by_name["download_url_tool"]["input_schema"]
        self.assertEqual(download_schema["properties"]["timeout"].get("minimum"), 1)
        self.assertEqual(download_schema["properties"]["max_size_mb"].get("minimum"), 1)

    def test_convert_file_tool_rejects_blank_input_path(self) -> None:
        async def _run() -> None:
            result = await nfc.convert_file_tool(input_path="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("input_path is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_download_url_tool_rejects_invalid_limits(self) -> None:
        async def _run() -> None:
            timeout_error = await nfc.download_url_tool(url="https://example.com", timeout=0)
            self.assertEqual(timeout_error.get("status"), "error")
            self.assertIn("timeout must be an integer >= 1", str(timeout_error.get("error", "")))

            size_error = await nfc.download_url_tool(url="https://example.com", max_size_mb=0)
            self.assertEqual(size_error.get("status"), "error")
            self.assertIn("max_size_mb must be an integer >= 1", str(size_error.get("error", "")))

        anyio.run(_run)

    def test_batch_convert_tool_rejects_invalid_inputs(self) -> None:
        async def _run() -> None:
            empty_list = await nfc.batch_convert_tool(input_paths=[])
            self.assertEqual(empty_list.get("status"), "error")
            self.assertIn("input_paths must be a non-empty list of strings", str(empty_list.get("error", "")))

            blank_value = await nfc.batch_convert_tool(input_paths=["ok.txt", "   "])
            self.assertEqual(blank_value.get("status"), "error")
            self.assertIn("input_paths must be a non-empty list of strings", str(blank_value.get("error", "")))

            bad_concurrency = await nfc.batch_convert_tool(input_paths=["ok.txt"], max_concurrent=0)
            self.assertEqual(bad_concurrency.get("status"), "error")
            self.assertIn("max_concurrent must be an integer >= 1", str(bad_concurrency.get("error", "")))

        anyio.run(_run)

    def test_generate_summary_tool_rejects_blank_llm_model(self) -> None:
        async def _run() -> None:
            result = await nfc.generate_summary_tool(input_path="/tmp/example.txt", llm_model="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("llm_model must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_generate_embeddings_tool_rejects_invalid_vector_store(self) -> None:
        async def _run() -> None:
            result = await nfc.generate_embeddings_tool(
                input_path="/tmp/example.txt",
                vector_store="sqlite",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("vector_store must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_extract_archive_tool_rejects_invalid_depth(self) -> None:
        async def _run() -> None:
            result = await nfc.extract_archive_tool(archive_path="archive.zip", max_depth=-1)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_depth must be an integer >= 0", str(result.get("error", "")))

        anyio.run(_run)

    def test_file_info_tool_normalizes_non_dict_delegate_payload(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"file_info_tool": lambda **_: "ready"}):
                result = await nfc.file_info_tool(input_path="/tmp/example.txt")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("tool"), "file_info_tool")
            self.assertEqual(result.get("input_path"), "/tmp/example.txt")
            self.assertEqual(result.get("result"), "ready")

        anyio.run(_run)

    def test_convert_file_tool_wraps_delegate_exception(self) -> None:
        async def _run() -> None:
            def _boom(**_: object) -> dict:
                raise RuntimeError("backend exploded")

            with patch.dict(nfc._API, {"convert_file_tool": _boom}):
                result = await nfc.convert_file_tool(input_path="/tmp/example.txt")

            self.assertEqual(result.get("status"), "error")
            self.assertIn("convert_file_tool failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_convert_file_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"convert_file_tool": lambda **_: {"status": "success"}}):
                result = await nfc.convert_file_tool(input_path="/tmp/example.txt")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "convert_file_tool")
            self.assertEqual(result.get("input_path"), "/tmp/example.txt")

        anyio.run(_run)

    def test_download_url_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"download_url_tool": lambda **_: {"status": "success"}}):
                result = await nfc.download_url_tool(url="https://example.com", timeout=15, max_size_mb=25)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "download_url_tool")
            self.assertEqual(result.get("url"), "https://example.com")
            self.assertEqual(result.get("timeout"), 15)
            self.assertEqual(result.get("max_size_mb"), 25)

        anyio.run(_run)

    def test_file_info_tool_error_only_payload_infers_error_status(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"file_info_tool": lambda **_: {"error": "unavailable"}}):
                result = await nfc.file_info_tool(input_path="/tmp/example.txt")

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("tool"), "file_info_tool")
            self.assertEqual(result.get("input_path"), "/tmp/example.txt")
            self.assertIn("unavailable", str(result.get("error", "")))

        anyio.run(_run)

    def test_batch_convert_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"batch_convert_tool": lambda **_: {"status": "success"}}):
                result = await nfc.batch_convert_tool(input_paths=["/tmp/example.txt"], max_concurrent=2)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "batch_convert_tool")
            self.assertEqual(result.get("input_paths"), ["/tmp/example.txt"])
            self.assertEqual(result.get("max_concurrent"), 2)

        anyio.run(_run)

    def test_generate_summary_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"generate_summary_tool": lambda **_: {"status": "success"}}):
                result = await nfc.generate_summary_tool(input_path="/tmp/example.txt")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "generate_summary_tool")
            self.assertEqual(result.get("input_path"), "/tmp/example.txt")

        anyio.run(_run)

    def test_generate_embeddings_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"generate_embeddings_tool": lambda **_: {"status": "success"}}):
                result = await nfc.generate_embeddings_tool(
                    input_path="/tmp/example.txt",
                    vector_store="qdrant",
                    enable_ipfs=True,
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "generate_embeddings_tool")
            self.assertEqual(result.get("vector_store"), "qdrant")
            self.assertEqual(result.get("enable_ipfs"), True)

        anyio.run(_run)

    def test_extract_archive_tool_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"extract_archive_tool": lambda **_: {"status": "success"}}):
                result = await nfc.extract_archive_tool(archive_path="archive.zip", recursive=False)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tool"), "extract_archive_tool")
            self.assertEqual(result.get("archive_path"), "archive.zip")
            self.assertEqual(result.get("recursive"), False)

        anyio.run(_run)

    def test_extract_knowledge_graph_tool_error_only_payload_infers_error_status(self) -> None:
        async def _run() -> None:
            with patch.dict(nfc._API, {"extract_knowledge_graph_tool": lambda **_: {"error": "graph unavailable"}}):
                result = await nfc.extract_knowledge_graph_tool(input_path="/tmp/example.txt", enable_ipfs=True)

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("tool"), "extract_knowledge_graph_tool")
            self.assertEqual(result.get("enable_ipfs"), True)
            self.assertIn("graph unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
