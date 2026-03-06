#!/usr/bin/env python3
"""UNI-178 provenance verification/reporting parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.provenance_tools.native_provenance_tools import (
    generate_provenance_report,
    register_native_provenance_tools,
    verify_provenance_records,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI178ProvenanceTools(unittest.TestCase):
    def test_register_includes_verification_and_report_tools(self) -> None:
        manager = _DummyManager()
        register_native_provenance_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("verify_provenance_records", names)
        self.assertIn("generate_provenance_report", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_provenance_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        verify_schema = by_name["verify_provenance_records"]["input_schema"]
        verify_records = verify_schema["properties"]["records"]
        self.assertEqual(verify_records.get("type"), "array")
        self.assertEqual(verify_records.get("minItems"), 1)

        report_schema = by_name["generate_provenance_report"]["input_schema"]
        report_records = report_schema["properties"]["records"]
        self.assertEqual(report_records.get("type"), "array")
        self.assertEqual(report_records.get("minItems"), 1)

    def test_verify_provenance_records_rejects_empty_records(self) -> None:
        async def _run() -> None:
            result = await verify_provenance_records(records=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty array", str(result.get("message", "")))

        anyio.run(_run)

    def test_verify_provenance_records_detects_invalid_records(self) -> None:
        async def _run() -> None:
            result = await verify_provenance_records(
                records=[
                    {"status": "success", "dataset_id": "dataset-1", "operation": "transform"},
                    {"status": "error", "dataset_id": "dataset-2", "operation": "publish", "message": "boom"},
                    {"status": "success", "dataset_id": "", "operation": "index"},
                ]
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("verified_count"), 1)
            self.assertEqual(result.get("failed_count"), 2)
            self.assertFalse(bool(result.get("all_valid")))

        anyio.run(_run)

    def test_generate_provenance_report_rejects_empty_records(self) -> None:
        async def _run() -> None:
            result = await generate_provenance_report(records=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty array", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_provenance_report_aggregate_shape(self) -> None:
        async def _run() -> None:
            result = await generate_provenance_report(
                records=[
                    {"status": "success", "dataset_id": "dataset-1", "operation": "transform"},
                    {"status": "error", "dataset_id": "dataset-2", "operation": "transform", "message": "bad hash"},
                    {"status": "success", "dataset_id": "dataset-3", "operation": "publish"},
                ],
                include_errors=True,
                aggregate_by_operation=True,
            )
            self.assertEqual(result.get("status"), "success")
            report = result.get("report") or {}
            self.assertEqual(report.get("success_count"), 2)
            self.assertEqual(report.get("error_count"), 1)
            by_operation = report.get("by_operation") or {}
            self.assertEqual(by_operation.get("transform"), 2)
            self.assertEqual(by_operation.get("publish"), 1)
            self.assertIn("error_samples", report)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
