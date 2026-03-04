#!/usr/bin/env python3
"""UNI-118 audit tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.audit_tools.native_audit_tools import (
    generate_audit_report,
    record_audit_event,
    register_native_audit_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI118AuditTools(unittest.TestCase):
    def test_register_includes_audit_tools(self) -> None:
        manager = _DummyManager()
        register_native_audit_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("record_audit_event", names)
        self.assertIn("generate_audit_report", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_audit_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        event_schema = by_name["record_audit_event"]["input_schema"]
        self.assertIn("critical", event_schema["properties"]["severity"].get("enum", []))

        report_schema = by_name["generate_audit_report"]["input_schema"]
        self.assertIn("comprehensive", report_schema["properties"]["report_type"].get("enum", []))
        self.assertEqual(report_schema["properties"]["start_time"].get("format"), "date-time")

    def test_record_audit_event_rejects_empty_action(self) -> None:
        async def _run() -> None:
            result = await record_audit_event(action="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_record_audit_event_rejects_invalid_severity(self) -> None:
        async def _run() -> None:
            result = await record_audit_event(action="dataset.read", severity="fatal")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("severity must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_audit_report_rejects_invalid_report_type(self) -> None:
        async def _run() -> None:
            result = await generate_audit_report(report_type="summary")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("report_type must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_audit_report_rejects_invalid_start_time(self) -> None:
        async def _run() -> None:
            result = await generate_audit_report(report_type="security", start_time="not-a-date")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("start_time must be a valid ISO-8601 datetime", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_audit_report_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await generate_audit_report(report_type="compliance", output_format="json")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("report_type"), "compliance")
            self.assertEqual(result.get("output_format"), "json")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
