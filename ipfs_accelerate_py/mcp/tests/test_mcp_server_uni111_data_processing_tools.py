#!/usr/bin/env python3
"""UNI-111 data processing tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.data_processing_tools import native_data_processing_tools
from ipfs_accelerate_py.mcp_server.tools.data_processing_tools.native_data_processing_tools import (
    chunk_text,
    convert_format,
    register_native_data_processing_tools,
    transform_data,
    validate_data,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI111DataProcessingTools(unittest.TestCase):
    def test_register_includes_data_processing_tools(self) -> None:
        manager = _DummyManager()
        register_native_data_processing_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("chunk_text", names)
        self.assertIn("transform_data", names)
        self.assertIn("convert_format", names)
        self.assertIn("validate_data", names)

    def test_chunk_text_rejects_invalid_overlap_contract(self) -> None:
        async def _run() -> None:
            result = await chunk_text(text="abcdef", chunk_size=4, overlap=4)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("smaller than chunk_size", str(result.get("message", "")))

        anyio.run(_run)

    def test_chunk_text_rejects_unknown_strategy(self) -> None:
        async def _run() -> None:
            result = await chunk_text(text="abcdef", strategy="windowed")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("strategy must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_transform_data_requires_transformation(self) -> None:
        async def _run() -> None:
            result = await transform_data(data={"x": 1}, transformation=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("transformation is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_transform_data_rejects_non_object_parameters(self) -> None:
        async def _run() -> None:
            result = await transform_data(data={"x": 1}, transformation="normalize", parameters=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("parameters must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_convert_format_requires_formats(self) -> None:
        async def _run() -> None:
            result = await convert_format(data={"x": 1}, source_format="", target_format="json")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("source_format and target_format are required", str(result.get("message", "")))

        anyio.run(_run)

    def test_convert_format_rejects_non_object_options(self) -> None:
        async def _run() -> None:
            result = await convert_format(data={"x": 1}, source_format="json", target_format="json", options="bad")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("options must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_validate_data_requires_validation_type(self) -> None:
        async def _run() -> None:
            result = await validate_data(data={"x": 1}, validation_type="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("validation_type is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_validate_data_rejects_non_array_rules(self) -> None:
        async def _run() -> None:
            result = await validate_data(data={"x": 1}, validation_type="schema", rules={"bad": True})  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("rules must be an array", str(result.get("message", "")))

        anyio.run(_run)

    def test_validate_data_rejects_non_object_rules_entries(self) -> None:
        async def _run() -> None:
            result = await validate_data(data={"x": 1}, validation_type="schema", rules=[{"ok": True}, "bad"])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("rules entries must be objects", str(result.get("message", "")))

        anyio.run(_run)

    def test_data_processing_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_data_processing_tools._API,
                {
                    "chunk_text": _contradictory_failure,
                    "transform_data": _contradictory_failure,
                    "convert_format": _contradictory_failure,
                    "validate_data": _contradictory_failure,
                },
                clear=False,
            ):
                chunked = await chunk_text(text="abcdef")
                transformed = await transform_data(data={"x": 1}, transformation="normalize")
                converted = await convert_format(data={"x": 1}, source_format="json", target_format="yaml")
                validated = await validate_data(data={"x": 1}, validation_type="schema")

            self.assertEqual(chunked.get("status"), "error")
            self.assertEqual(chunked.get("error"), "delegate failed")

            self.assertEqual(transformed.get("status"), "error")
            self.assertEqual(transformed.get("error"), "delegate failed")

            self.assertEqual(converted.get("status"), "error")
            self.assertEqual(converted.get("error"), "delegate failed")

            self.assertEqual(validated.get("status"), "error")
            self.assertEqual(validated.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
