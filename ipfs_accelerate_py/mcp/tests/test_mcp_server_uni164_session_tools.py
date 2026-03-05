#!/usr/bin/env python3
"""UNI-164 deterministic cleanup-option parity tests for native session tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.session_tools.native_session_tools import (
    manage_session,
    register_native_session_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI164SessionTools(unittest.TestCase):
    def test_registration_schema_includes_cleanup_option_contracts(self) -> None:
        manager = _DummyManager()
        register_native_session_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_props = schemas["manage_session"]["properties"]
        cleanup_options = manage_props["cleanup_options"]
        cleanup_props = cleanup_options["properties"]

        self.assertEqual(cleanup_props["max_age_hours"].get("minimum"), 1)
        self.assertEqual(cleanup_props["dry_run"].get("type"), "boolean")

    def test_manage_session_cleanup_rejects_invalid_cleanup_options_shape(self) -> None:
        async def _run() -> None:
            result = await manage_session(
                action="cleanup",
                cleanup_options=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "INVALID_CLEANUP_OPTIONS")
            self.assertIn("cleanup_options must be an object", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_session_cleanup_rejects_non_boolean_dry_run(self) -> None:
        async def _run() -> None:
            result = await manage_session(
                action="cleanup",
                cleanup_options={"dry_run": "yes"},  # type: ignore[dict-item]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "INVALID_CLEANUP_OPTIONS")
            self.assertIn("cleanup_options.dry_run must be a boolean", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_session_cleanup_rejects_non_positive_max_age_hours(self) -> None:
        async def _run() -> None:
            result = await manage_session(
                action="cleanup",
                cleanup_options={"max_age_hours": 0},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "INVALID_CLEANUP_OPTIONS")
            self.assertIn("max_age_hours must be a positive integer", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_session_cleanup_success_contains_cleanup_report(self) -> None:
        async def _run() -> None:
            result = await manage_session(
                action="cleanup",
                cleanup_options={"max_age_hours": 12, "dry_run": True},
            )
            self.assertEqual(result.get("status"), "success")
            report = result.get("cleanup_report")
            self.assertIsInstance(report, dict)
            self.assertEqual(report.get("max_age_hours"), 12)
            self.assertEqual(report.get("dry_run"), True)
            self.assertEqual(report.get("expired_session_count"), 0)
            self.assertEqual(result.get("cleaned_up"), 0)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
