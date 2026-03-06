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

        convert_schema = by_name["convert_file_tool"]["input_schema"]
        self.assertEqual(convert_schema["properties"]["input_path"].get("minLength"), 1)
        self.assertEqual(convert_schema["properties"]["backend"].get("minLength"), 1)
        self.assertEqual(convert_schema["properties"]["output_format"].get("minLength"), 1)

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


if __name__ == "__main__":
    unittest.main()
