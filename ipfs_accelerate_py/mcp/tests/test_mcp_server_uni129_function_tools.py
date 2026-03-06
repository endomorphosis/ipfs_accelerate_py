#!/usr/bin/env python3
"""UNI-129 function tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.functions.native_function_tools import (
    execute_python_snippet,
    register_native_function_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI129FunctionTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_function_tools(manager)
        self.assertEqual(len(manager.calls), 1)

        schema = manager.calls[0]["input_schema"]
        self.assertEqual(schema["properties"]["timeout_seconds"].get("minimum"), 1)
        self.assertEqual(schema["properties"]["timeout_seconds"].get("default"), 30)

    def test_execute_rejects_empty_code(self) -> None:
        async def _run() -> None:
            result = await execute_python_snippet(code="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("code must be a non-empty string", str(result.get("message", "")))

        anyio.run(_run)

    def test_execute_rejects_invalid_timeout(self) -> None:
        async def _run() -> None:
            result = await execute_python_snippet(code="print('x')", timeout_seconds=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout_seconds must be an integer >= 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_execute_rejects_invalid_context_shape(self) -> None:
        async def _run() -> None:
            result = await execute_python_snippet(code="print('x')", context=["bad"])  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("context must be an object", str(result.get("message", "")))

        anyio.run(_run)

    def test_execute_normalizes_delegate_payload(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.functions.native_function_tools._API"
            ) as mock_api:
                async def _impl(**kwargs):
                    return {"message": "ok", "echo": kwargs.get("code")}

                mock_api.__getitem__.return_value = _impl

                result = await execute_python_snippet(code="print('ok')", timeout_seconds=5)
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("timeout_seconds"), 5)
                self.assertEqual(result.get("echo"), "print('ok')")

        anyio.run(_run)

    def test_execute_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.functions.native_function_tools._API"
            ) as mock_api:
                async def _impl(**kwargs):
                    _ = kwargs
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl

                result = await execute_python_snippet(code="print('ok')", timeout_seconds=7)
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("timeout_seconds"), 7)
                self.assertEqual(result.get("message"), "Execution request processed")
                self.assertEqual(result.get("execution_time_ms"), 0)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
