#!/usr/bin/env python3
"""UNI-120 email tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools import (
    email_analyze_export,
    email_list_folders,
    email_parse_eml,
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
        self.assertIn("email_parse_eml", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_email_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        conn_schema = by_name["email_test_connection"]["input_schema"]
        self.assertEqual(conn_schema["properties"]["timeout"].get("minimum"), 1)

        search_schema = by_name["email_search_export"]["input_schema"]
        self.assertIn("all", search_schema["properties"]["field"].get("enum", []))

        parse_schema = by_name["email_parse_eml"]["input_schema"]
        self.assertEqual(parse_schema["properties"]["include_attachments"].get("default"), True)

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

    def test_email_parse_eml_requires_file_path(self) -> None:
        async def _run() -> None:
            result = await email_parse_eml(file_path="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("file_path is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_parse_eml_rejects_non_boolean_include_attachments(self) -> None:
        async def _run() -> None:
            result = await email_parse_eml(  # type: ignore[arg-type]
                file_path="/tmp/mail.eml",
                include_attachments="yes",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be a boolean", str(result.get("message", "")))

        anyio.run(_run)

    def test_email_test_connection_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_test_connection(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools._API",
                {
                    "email_test_connection": _minimal_test_connection,
                    "email_list_folders": None,
                    "email_analyze_export": None,
                    "email_search_export": None,
                },
            ):
                result = await email_test_connection(protocol="imap", server="mail.example", use_ssl=False, timeout=10)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("protocol"), "imap")
            self.assertEqual(result.get("server"), "mail.example")
            self.assertEqual(result.get("use_ssl"), False)
            self.assertEqual(result.get("timeout"), 10)

        anyio.run(_run)

    def test_email_list_folders_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_list_folders(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools._API",
                {
                    "email_test_connection": None,
                    "email_list_folders": _minimal_list_folders,
                    "email_analyze_export": None,
                    "email_search_export": None,
                },
            ):
                result = await email_list_folders(server="mail.example", timeout=12)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("server"), "mail.example")
            self.assertEqual(result.get("folders"), [])
            self.assertEqual(result.get("folder_count"), 0)
            self.assertEqual(result.get("timeout"), 12)

        anyio.run(_run)

    def test_email_export_success_defaults_with_minimal_payloads(self) -> None:
        async def _minimal_analyze(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_search(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools._API",
                {
                    "email_test_connection": None,
                    "email_list_folders": None,
                    "email_analyze_export": _minimal_analyze,
                    "email_search_export": _minimal_search,
                },
            ):
                analyze_result = await email_analyze_export(file_path="/tmp/email.json")
                search_result = await email_search_export(file_path="/tmp/email.json", query="invoice", field="subject")

            self.assertEqual(analyze_result.get("status"), "success")
            self.assertEqual(analyze_result.get("file_path"), "/tmp/email.json")
            self.assertEqual(analyze_result.get("analysis"), {})

            self.assertEqual(search_result.get("status"), "success")
            self.assertEqual(search_result.get("file_path"), "/tmp/email.json")
            self.assertEqual(search_result.get("query"), "invoice")
            self.assertEqual(search_result.get("field"), "subject")
            self.assertEqual(search_result.get("results"), [])
            self.assertEqual(search_result.get("match_count"), 0)

        anyio.run(_run)

    def test_email_parse_eml_success_defaults_with_minimal_payload(self) -> None:
        async def _minimal_parse(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools._API",
                {
                    "email_test_connection": None,
                    "email_list_folders": None,
                    "email_analyze_export": None,
                    "email_search_export": None,
                    "email_parse_eml": _minimal_parse,
                },
            ):
                result = await email_parse_eml(file_path="/tmp/mail.eml", include_attachments=False)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("file_path"), "/tmp/mail.eml")
            self.assertEqual(result.get("include_attachments"), False)
            self.assertEqual(result.get("email"), {})

        anyio.run(_run)

    def test_email_wrappers_infer_error_status_from_contradictory_delegate_payload(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.email_tools.native_email_tools._API",
                {
                    "email_test_connection": _contradictory_failure,
                    "email_list_folders": _contradictory_failure,
                    "email_analyze_export": _contradictory_failure,
                    "email_search_export": _contradictory_failure,
                    "email_parse_eml": _contradictory_failure,
                },
            ):
                tested = await email_test_connection(protocol="imap", server="mail.example")
                listed = await email_list_folders(server="mail.example")
                analyzed = await email_analyze_export(file_path="/tmp/email.json")
                searched = await email_search_export(file_path="/tmp/email.json", query="invoice")
                parsed = await email_parse_eml(file_path="/tmp/mail.eml")

            self.assertEqual(tested.get("status"), "error")
            self.assertEqual(tested.get("error"), "delegate failed")

            self.assertEqual(listed.get("status"), "error")
            self.assertEqual(listed.get("error"), "delegate failed")

            self.assertEqual(analyzed.get("status"), "error")
            self.assertEqual(analyzed.get("error"), "delegate failed")

            self.assertEqual(searched.get("status"), "error")
            self.assertEqual(searched.get("error"), "delegate failed")

            self.assertEqual(parsed.get("status"), "error")
            self.assertEqual(parsed.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
