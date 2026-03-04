#!/usr/bin/env python3
"""UNI-119 alert tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.alert_tools.native_alert_tools import (
    evaluate_alert_rules,
    list_alert_rules,
    register_native_alert_tools,
    send_discord_message,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI119AlertTools(unittest.TestCase):
    def test_register_includes_alert_tools(self) -> None:
        manager = _DummyManager()
        register_native_alert_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("send_discord_message", names)
        self.assertIn("evaluate_alert_rules", names)
        self.assertIn("list_alert_rules", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_alert_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}
        self.assertEqual(by_name["list_alert_rules"]["input_schema"]["properties"]["enabled_only"].get("default"), False)

    def test_send_discord_message_rejects_empty_text(self) -> None:
        async def _run() -> None:
            result = await send_discord_message(text="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("text is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_evaluate_alert_rules_rejects_bad_event_shape(self) -> None:
        async def _run() -> None:
            result = await evaluate_alert_rules(event=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("event must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_list_alert_rules_rejects_non_boolean_enabled_only(self) -> None:
        async def _run() -> None:
            result = await list_alert_rules(enabled_only="yes")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("enabled_only must be a boolean", str(result.get("message", "")))

        anyio.run(_run)

    def test_send_discord_message_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await send_discord_message(text="hello", role_names=["ops"])
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("text"), "hello")

        anyio.run(_run)

    def test_evaluate_alert_rules_success_envelope_shape(self) -> None:
        async def _run() -> None:
            event = {"severity": "warning", "message": "test"}
            result = await evaluate_alert_rules(event=event)
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("event"), event)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
