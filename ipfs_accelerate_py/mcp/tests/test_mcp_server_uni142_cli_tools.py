#!/usr/bin/env python3
"""UNI-142 cli parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.cli.native_cli_tools import (
    execute_command,
    register_native_cli_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI142CliTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_cli_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        schema = by_name["execute_command"]["input_schema"]
        props = schema["properties"]

        self.assertEqual(props["command"]["minLength"], 1)
        self.assertEqual(props["timeout_seconds"]["minimum"], 1)

    def test_execute_command_validation(self) -> None:
        async def _run() -> None:
            result = await execute_command(command="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("command", str(result.get("error", "")))

            result = await execute_command(command="echo", args=[""])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("args", str(result.get("error", "")))

            result = await execute_command(command="echo", timeout_seconds=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout_seconds", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            result = await execute_command(
                command="echo",
                args=["hello"],
                timeout_seconds=3,
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("command"), "echo")
            self.assertEqual(result.get("args"), ["hello"])
            self.assertEqual(result.get("timeout_seconds"), 3)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
