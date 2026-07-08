#!/usr/bin/env python3
"""UNI-117 file-detection tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.file_detection_tools import native_file_detection_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI117FileDetectionTools(unittest.TestCase):
    def test_register_includes_file_detection_tools(self) -> None:
        manager = _DummyManager()
        native_file_detection_tools.register_native_file_detection_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("detect_file_type", names)
        self.assertIn("batch_detect_file_types", names)
        self.assertIn("analyze_detection_accuracy", names)
        self.assertIn("generate_detection_report", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        native_file_detection_tools.register_native_file_detection_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        detect_schema = by_name["detect_file_type"]["input_schema"]
        strategy = detect_schema["properties"]["strategy"]
        self.assertIn("accurate", strategy.get("enum", []))

        batch_schema = by_name["batch_detect_file_types"]["input_schema"]
        self.assertEqual(batch_schema["properties"]["pattern"].get("default"), "*")

        report_schema = by_name["generate_detection_report"]["input_schema"]
        self.assertEqual(report_schema["properties"]["top_mime_types"].get("maximum"), 50)
        self.assertEqual(report_schema["properties"]["include_examples"].get("default"), True)

    def test_detect_file_type_rejects_invalid_method(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.detect_file_type(
                file_path="/tmp/x.txt",
                methods=["bad"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("methods entries must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_batch_detect_file_types_requires_directory_or_file_paths(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.batch_detect_file_types()
            self.assertEqual(result.get("status"), "error")
            self.assertIn("either directory or file_paths must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_batch_detect_file_types_rejects_invalid_strategy(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.batch_detect_file_types(
                file_paths=["/tmp/a.txt"],
                strategy="slow",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("strategy must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_analyze_detection_accuracy_rejects_empty_directory(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.analyze_detection_accuracy(directory=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("directory is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_detect_file_type_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.detect_file_type(
                file_path="/tmp/x.txt",
                strategy="accurate",
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertIn("file_path", result)

        anyio.run(_run)

    def test_generate_detection_report_rejects_empty_results(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.generate_detection_report(results={})
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty object", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_detection_report_rejects_invalid_top_mime_types(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.generate_detection_report(
                results={"/tmp/a.txt": {"mime_type": "text/plain", "confidence": 0.8}},
                top_mime_types=0,
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("between 1 and 50", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_detection_report_success_shape(self) -> None:
        async def _run() -> None:
            result = await native_file_detection_tools.generate_detection_report(
                results={
                    "/tmp/a.txt": {"mime_type": "text/plain", "confidence": 0.9},
                    "/tmp/b.pdf": {"mime_type": "application/pdf", "confidence": 0.8},
                    "/tmp/c.unknown": {"error": "not detected"},
                }
            )
            self.assertEqual(result.get("status"), "success")
            report = result.get("report") or {}
            self.assertEqual(report.get("total_files"), 3)
            self.assertEqual(report.get("successful"), 2)
            self.assertEqual(report.get("failed"), 1)
            self.assertIn("common_mime_types", report)

        anyio.run(_run)

    def test_file_detection_wrappers_infer_error_status_from_contradictory_delegate_payload(self) -> None:
        def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_file_detection_tools._API,
                {
                    "detect_file_type": _contradictory_failure,
                    "batch_detect_file_types": _contradictory_failure,
                    "analyze_detection_accuracy": _contradictory_failure,
                },
                clear=False,
            ):
                detected = await native_file_detection_tools.detect_file_type(file_path="/tmp/x.txt")
                batched = await native_file_detection_tools.batch_detect_file_types(file_paths=["/tmp/x.txt"])
                analyzed = await native_file_detection_tools.analyze_detection_accuracy(directory="/tmp")

            self.assertEqual(detected.get("status"), "error")
            self.assertEqual(detected.get("error"), "delegate failed")

            self.assertEqual(batched.get("status"), "error")
            self.assertEqual(batched.get("error"), "delegate failed")

            self.assertEqual(analyzed.get("status"), "error")
            self.assertEqual(analyzed.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
