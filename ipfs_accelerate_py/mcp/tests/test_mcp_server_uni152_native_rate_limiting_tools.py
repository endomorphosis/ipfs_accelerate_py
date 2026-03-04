#!/usr/bin/env python3
"""UNI-152 native rate_limiting category parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.rate_limiting.native_rate_limiting_tools import (
    check_rate_limit,
    configure_rate_limits,
    manage_rate_limits,
    register_native_rate_limiting_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI152NativeRateLimitingTools(unittest.TestCase):
    def test_register_schema_contracts_are_hardened(self) -> None:
        manager = _DummyManager()
        register_native_rate_limiting_tools(manager)

        by_name = {call["name"]: call for call in manager.calls}
        self.assertIn("configure_rate_limits", by_name)
        self.assertIn("check_rate_limit", by_name)
        self.assertIn("manage_rate_limits", by_name)

        check_props = by_name["check_rate_limit"]["input_schema"]["properties"]
        self.assertEqual(check_props["limit_name"].get("minLength"), 1)
        self.assertEqual(check_props["identifier"].get("minLength"), 1)

        manage_props = by_name["manage_rate_limits"]["input_schema"]["properties"]
        self.assertIn("stats", manage_props["action"].get("enum", []))

    def test_configure_rejects_invalid_flags_and_entry_shape(self) -> None:
        async def _run() -> None:
            invalid_flag = await configure_rate_limits(
                limits=[{"name": "api", "requests_per_second": 10}],
                apply_immediately="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_flag.get("status"), "error")
            self.assertIn("apply_immediately must be a boolean", str(invalid_flag.get("error", "")))

            invalid_entry = await configure_rate_limits(
                limits=[{"name": "api", "requests_per_second": 10}, "bad"],  # type: ignore[list-item]
            )
            self.assertEqual(invalid_entry.get("status"), "error")
            self.assertGreater(len(invalid_entry.get("errors", [])), 0)
            self.assertIn("limits[1] must be an object", " ".join(invalid_entry.get("errors", [])))

        anyio.run(_run)

    def test_check_rejects_blank_identifier_and_bad_metadata(self) -> None:
        async def _run() -> None:
            invalid_identifier = await check_rate_limit(limit_name="api", identifier="   ")
            self.assertEqual(invalid_identifier.get("status"), "error")
            self.assertIn("identifier must be a non-empty string", str(invalid_identifier.get("error", "")))

            invalid_metadata = await check_rate_limit(
                limit_name="api",
                identifier="client-a",
                request_metadata="bad",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_metadata.get("status"), "error")
            self.assertIn(
                "request_metadata must be an object or null",
                str(invalid_metadata.get("error", "")),
            )

        anyio.run(_run)

    def test_manage_rejects_blank_stats_limit_and_unknown_action(self) -> None:
        async def _run() -> None:
            invalid_stats = await manage_rate_limits(action="stats", limit_name="   ")
            self.assertEqual(invalid_stats.get("status"), "error")
            self.assertIn(
                "limit_name must be a non-empty string when provided for stats",
                str(invalid_stats.get("error", "")),
            )

            invalid_action = await manage_rate_limits(action="bad-action")
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertIn("Unknown action", str(invalid_action.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
