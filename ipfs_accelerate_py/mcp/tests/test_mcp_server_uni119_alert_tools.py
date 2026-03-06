#!/usr/bin/env python3
"""UNI-119 alert tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.alert_tools.native_alert_tools import (
    evaluate_alert_rules,
    list_alert_rules,
    remove_alert_rule,
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
        self.assertIn("remove_alert_rule", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_alert_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}
        self.assertEqual(by_name["list_alert_rules"]["input_schema"]["properties"]["enabled_only"].get("default"), False)
        self.assertEqual(by_name["remove_alert_rule"]["input_schema"]["required"], ["rule_id"])

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

    def test_remove_alert_rule_rejects_empty_rule_id(self) -> None:
        async def _run() -> None:
            result = await remove_alert_rule(rule_id=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("rule_id is required", str(result.get("message", "")))

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

    def test_send_discord_message_success_defaults_with_minimal_backend_payload(self) -> None:
        async def _minimal_send(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.alert_tools.native_alert_tools._API",
                {
                    "send_discord_message": _minimal_send,
                    "evaluate_alert_rules": None,
                    "list_alert_rules": None,
                },
            ):
                result = await send_discord_message(
                    text="hello",
                    role_names=["ops"],
                    channel_id="alerts",
                    thread_id="thread-1",
                )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("text"), "hello")
            self.assertEqual(result.get("role_names"), ["ops"])
            self.assertEqual(result.get("channel_id"), "alerts")
            self.assertEqual(result.get("thread_id"), "thread-1")

        anyio.run(_run)

    def test_evaluate_and_list_defaults_with_minimal_backend_payloads(self) -> None:
        async def _minimal_evaluate(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_list(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.alert_tools.native_alert_tools._API",
                {
                    "send_discord_message": None,
                    "evaluate_alert_rules": _minimal_evaluate,
                    "list_alert_rules": _minimal_list,
                },
            ):
                event = {"severity": "warning", "message": "test"}
                eval_result = await evaluate_alert_rules(event=event, rule_ids=["rule-1"])
                list_result = await list_alert_rules(enabled_only=True)

            self.assertEqual(eval_result.get("status"), "success")
            self.assertEqual(eval_result.get("event"), event)
            self.assertEqual(eval_result.get("rule_ids"), ["rule-1"])
            self.assertEqual(eval_result.get("results"), [])
            self.assertEqual(eval_result.get("triggered_rules"), 0)

            self.assertEqual(list_result.get("status"), "success")
            self.assertEqual(list_result.get("enabled_only"), True)
            self.assertEqual(list_result.get("rules"), [])
            self.assertEqual(list_result.get("count"), 0)

        anyio.run(_run)

    def test_remove_alert_rule_success_defaults_with_minimal_backend_payload(self) -> None:
        def _minimal_remove(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.alert_tools.native_alert_tools._API",
                {
                    "send_discord_message": None,
                    "evaluate_alert_rules": None,
                    "list_alert_rules": None,
                    "remove_alert_rule": _minimal_remove,
                },
            ):
                result = await remove_alert_rule(rule_id="rule-1")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("rule_id"), "rule-1")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
