#!/usr/bin/env python3
"""UNI-151 native rate_limiting_tools category parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.rate_limiting_tools.native_rate_limiting_tools_category import (
    check_rate_limit,
    configure_rate_limits,
    manage_rate_limits,
    register_native_rate_limiting_tools_category,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI151RateLimitingToolsCategory(unittest.TestCase):
    def test_register_schema_contracts_are_hardened(self) -> None:
        manager = _DummyManager()
        register_native_rate_limiting_tools_category(manager)

        by_name = {call["name"]: call for call in manager.calls}
        self.assertIn("configure_rate_limits", by_name)
        self.assertIn("check_rate_limit", by_name)
        self.assertIn("manage_rate_limits", by_name)

        check_props = by_name["check_rate_limit"]["input_schema"]["properties"]
        self.assertEqual(check_props["limit_name"].get("minLength"), 1)
        self.assertEqual(check_props["identifier"].get("minLength"), 1)

        manage_props = by_name["manage_rate_limits"]["input_schema"]["properties"]
        self.assertIn("update", manage_props["action"].get("enum", []))

    def test_configure_rate_limits_validation_envelope(self) -> None:
        async def _run() -> None:
            invalid_flag = await configure_rate_limits(
                limits=[{"name": "api", "requests": 10}],
                apply_immediately="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_flag.get("status"), "error")
            self.assertIn("apply_immediately must be a boolean", str(invalid_flag.get("error", "")))

            invalid_item = await configure_rate_limits(
                limits=[{"name": "api"}, "bad"],  # type: ignore[list-item]
            )
            self.assertEqual(invalid_item.get("status"), "error")
            self.assertIn("limits[1] must be an object", str(invalid_item.get("error", "")))

        anyio.run(_run)

    def test_check_rate_limit_validation_envelope(self) -> None:
        async def _run() -> None:
            invalid_identifier = await check_rate_limit(limit_name="api", identifier="   ")
            self.assertEqual(invalid_identifier.get("status"), "error")
            self.assertIn(
                "identifier must be a non-empty string",
                str(invalid_identifier.get("error", "")),
            )

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

    def test_manage_rate_limits_stats_rejects_blank_limit_name(self) -> None:
        async def _run() -> None:
            result = await manage_rate_limits(action="stats", limit_name="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "limit_name must be a non-empty string when provided for stats",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
