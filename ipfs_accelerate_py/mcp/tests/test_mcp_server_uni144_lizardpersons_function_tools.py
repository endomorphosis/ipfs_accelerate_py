#!/usr/bin/env python3
"""UNI-144 lizardpersons_function_tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.lizardpersons_function_tools.native_lizardpersons_function_tools import (
    get_current_time,
    register_native_lizardpersons_function_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI144LizardpersonsFunctionTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_lizardpersons_function_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        schema = by_name["get_current_time"]["input_schema"]
        props = schema["properties"]

        self.assertEqual(props["format_type"]["enum"], ["iso", "human", "timestamp"])
        self.assertEqual(props["format_type"]["default"], "iso")
        self.assertEqual(props["check_if_within_working_hours"]["default"], False)

    def test_get_current_time_validation(self) -> None:
        async def _run() -> None:
            result = await get_current_time(format_type="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("format_type", str(result.get("error", "")))

            result = await get_current_time(format_type="custom")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("one of", str(result.get("error", "")))

            result = await get_current_time(
                format_type="iso",
                check_if_within_working_hours="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("check_if_within_working_hours", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            result = await get_current_time(format_type="iso")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("format_type"), "iso")

        anyio.run(_run)

    def test_get_current_time_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.lizardpersons_function_tools.native_lizardpersons_function_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await get_current_time(format_type="human", check_if_within_working_hours=True)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("value"), "")
            self.assertEqual(result.get("format_type"), "human")
            self.assertEqual(result.get("check_if_within_working_hours"), True)

        anyio.run(_run)

    def test_get_current_time_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.lizardpersons_function_tools.native_lizardpersons_function_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"error": "clock unavailable"}

                mock_api.__getitem__.return_value = _impl
                result = await get_current_time(format_type="iso")

            self.assertEqual(result.get("status"), "error")
            self.assertIn("clock unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
