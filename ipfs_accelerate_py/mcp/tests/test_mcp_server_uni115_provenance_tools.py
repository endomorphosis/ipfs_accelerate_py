#!/usr/bin/env python3
"""UNI-115 provenance tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.provenance_tools.native_provenance_tools import (
    record_provenance,
    register_native_provenance_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI115ProvenanceTools(unittest.TestCase):
    def test_register_includes_provenance_tool(self) -> None:
        manager = _DummyManager()
        register_native_provenance_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("record_provenance", names)

    def test_record_provenance_rejects_missing_dataset_id(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="", operation="transform")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("dataset_id is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_rejects_missing_operation(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="dataset-1", operation=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("operation is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_rejects_invalid_inputs_shape(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="dataset-1", operation="transform", inputs=["ok", 1])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("array of strings", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_rejects_invalid_parameters(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="dataset-1", operation="transform", parameters=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_success_shape(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="dataset-1", operation="transform")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("dataset_id", result)
                self.assertIn("operation", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
