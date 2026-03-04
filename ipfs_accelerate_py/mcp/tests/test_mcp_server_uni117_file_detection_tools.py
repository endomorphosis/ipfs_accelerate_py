#!/usr/bin/env python3
"""UNI-117 file-detection tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.file_detection_tools.native_file_detection_tools import (
    analyze_detection_accuracy,
    batch_detect_file_types,
    detect_file_type,
    register_native_file_detection_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI117FileDetectionTools(unittest.TestCase):
    def test_register_includes_file_detection_tools(self) -> None:
        manager = _DummyManager()
        register_native_file_detection_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("detect_file_type", names)
        self.assertIn("batch_detect_file_types", names)
        self.assertIn("analyze_detection_accuracy", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_file_detection_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        detect_schema = by_name["detect_file_type"]["input_schema"]
        strategy = detect_schema["properties"]["strategy"]
        self.assertIn("accurate", strategy.get("enum", []))

        batch_schema = by_name["batch_detect_file_types"]["input_schema"]
        self.assertEqual(batch_schema["properties"]["pattern"].get("default"), "*")

    def test_detect_file_type_rejects_invalid_method(self) -> None:
        async def _run() -> None:
            result = await detect_file_type(file_path="/tmp/x.txt", methods=["bad"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("methods entries must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_batch_detect_file_types_requires_directory_or_file_paths(self) -> None:
        async def _run() -> None:
            result = await batch_detect_file_types()
            self.assertEqual(result.get("status"), "error")
            self.assertIn("either directory or file_paths must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_batch_detect_file_types_rejects_invalid_strategy(self) -> None:
        async def _run() -> None:
            result = await batch_detect_file_types(file_paths=["/tmp/a.txt"], strategy="slow")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("strategy must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_analyze_detection_accuracy_rejects_empty_directory(self) -> None:
        async def _run() -> None:
            result = await analyze_detection_accuracy(directory=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("directory is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_detect_file_type_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await detect_file_type(file_path="/tmp/x.txt", strategy="accurate")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertIn("file_path", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
