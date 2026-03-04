#!/usr/bin/env python3
"""UNI-120 email tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools import (
    email_analyze_export,
    email_search_export,
    email_test_connection,
    register_native_email_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI120EmailTools(unittest.TestCase):
    def test_register_includes_email_tools(self) -> None:
        manager = _DummyManager()
        register_native_email_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("email_test_connection", names)
        self.assertIn("email_list_folders", names)
        self.assertIn("email_analyze_export", names)
        self.assertIn("email_search_export", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_email_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        conn_schema = by_name["email_test_connection"]["input_schema"]
        self.assertEqual(conn_schema["properties"]["timeout"].get("minimum"), 1)

        search_schema = by_name["email_search_export"]["input_schema"]
        self.assertIn("all", search_schema["properties"]["field"].get("enum", []))

    def test_email_test_connection_rejects_invalid_protocol(self) -> None:
        async def _run() -> None:
            result = await email_test_connection(protocol="smtp")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("protocol must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_test_connection_rejects_bad_timeout(self) -> None:
        async def _run() -> None:
            result = await email_test_connection(protocol="imap", timeout=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout must be an integer >= 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_analyze_export_requires_file_path(self) -> None:
        async def _run() -> None:
            result = await email_analyze_export()
            self.assertEqual(result.get("status"), "error")
            self.assertIn("file_path is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_search_export_requires_query(self) -> None:
        async def _run() -> None:
            result = await email_search_export(file_path="/tmp/email.json", query="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_search_export_rejects_invalid_field(self) -> None:
        async def _run() -> None:
            result = await email_search_export(
                file_path="/tmp/email.json",
                query="invoice",
                field="header",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("field must be one of", str(result.get("message", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
