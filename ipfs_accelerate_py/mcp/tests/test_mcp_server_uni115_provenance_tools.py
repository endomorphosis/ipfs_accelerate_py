#!/usr/bin/env python3
"""UNI-115 provenance tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.provenance_tools.native_provenance_tools import (
    record_provenance_batch,
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
        self.assertIn("record_provenance_batch", names)

    def test_register_schema_contract(self) -> None:
        manager = _DummyManager()
        register_native_provenance_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        schema = by_name["record_provenance"]["input_schema"]
        timestamp = schema["properties"]["timestamp"]
        self.assertEqual(timestamp.get("format"), "date-time")

        batch_schema = by_name["record_provenance_batch"]["input_schema"]
        records = batch_schema["properties"]["records"]
        self.assertEqual(records.get("type"), "array")
        self.assertEqual(records.get("minItems"), 1)
        self.assertEqual(
            records["items"]["properties"]["timestamp"].get("format"),
            "date-time",
        )

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

    def test_record_provenance_rejects_invalid_timestamp(self) -> None:
        async def _run() -> None:
            result = await record_provenance(
                dataset_id="dataset-1",
                operation="transform",
                timestamp="not-a-timestamp",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("valid ISO-8601", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_rejects_empty_tag(self) -> None:
        async def _run() -> None:
            result = await record_provenance(
                dataset_id="dataset-1",
                operation="transform",
                tags=["ok", " "],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("cannot contain empty strings", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_success_shape(self) -> None:
        async def _run() -> None:
            result = await record_provenance(dataset_id="dataset-1", operation="transform")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("dataset_id", result)
                self.assertIn("operation", result)

        anyio.run(_run)

    def test_record_provenance_batch_rejects_empty_records(self) -> None:
        async def _run() -> None:
            result = await record_provenance_batch(records=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty array", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_provenance_batch_fail_fast_stops_on_error(self) -> None:
        async def _run() -> None:
            result = await record_provenance_batch(
                records=[
                    {"dataset_id": "dataset-1", "operation": "transform"},
                    {"dataset_id": "", "operation": "ingest"},
                    {"dataset_id": "dataset-3", "operation": "publish"},
                ],
                fail_fast=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("processed"), 2)
            self.assertEqual(result.get("requested"), 3)
            self.assertEqual(result.get("error_count"), 1)

        anyio.run(_run)

    def test_record_provenance_batch_success_shape(self) -> None:
        async def _run() -> None:
            result = await record_provenance_batch(
                records=[
                    {"dataset_id": "dataset-1", "operation": "transform"},
                    {"dataset_id": "dataset-2", "operation": "index"},
                ]
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("processed"), 2)
            self.assertEqual(result.get("requested"), 2)
            self.assertIn("success_count", result)
            self.assertIn("results", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
