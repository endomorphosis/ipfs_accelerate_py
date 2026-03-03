#!/usr/bin/env python3
"""UNI-108 rate-limiting tools parity tests."""

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


class TestMCPServerUNI108RateLimitingTools(unittest.TestCase):
    def test_register_includes_rate_limiting_tools(self) -> None:
        manager = _DummyManager()
        register_native_rate_limiting_tools_category(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("configure_rate_limits", names)
        self.assertIn("check_rate_limit", names)
        self.assertIn("manage_rate_limits", names)

    def test_configure_rate_limits_rejects_non_list_limits(self) -> None:
        async def _run() -> None:
            result = await configure_rate_limits(limits={"name": "api"})  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("limits must be a list", " ".join(result.get("errors", [])))

        anyio.run(_run)

    def test_check_rate_limit_rejects_missing_limit_name(self) -> None:
        async def _run() -> None:
            result = await check_rate_limit(limit_name="  ")
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("allowed"), False)
            self.assertIn("limit_name is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_rate_limits_requires_action(self) -> None:
        async def _run() -> None:
            result = await manage_rate_limits(action="")
            self.assertEqual(result.get("error"), "action is required")
            self.assertIn("valid_actions", result)

        anyio.run(_run)

    def test_manage_rate_limits_requires_limit_name_for_enable(self) -> None:
        async def _run() -> None:
            result = await manage_rate_limits(action="enable", limit_name=" ")
            self.assertIn("limit_name required for enable action", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_rate_limits_rejects_unknown_action(self) -> None:
        async def _run() -> None:
            result = await manage_rate_limits(action="bad-action")
            self.assertIn("Unknown action", str(result.get("error", "")))
            self.assertIn("valid_actions", result)

        anyio.run(_run)

    def test_manage_rate_limits_list_shape(self) -> None:
        async def _run() -> None:
            result = await manage_rate_limits(action="list")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("action"), "list")
            if result.get("status") == "success":
                self.assertIn("limits", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
