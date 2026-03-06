#!/usr/bin/env python3
"""UNI-132 development tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools import (
    codebase_search,
    register_native_development_tools,
    vscode_cli_status,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI132DevelopmentTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_development_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        search_schema = by_name["codebase_search"]["input_schema"]
        props = search_schema["properties"]
        self.assertEqual(props["context"].get("minimum"), 0)
        self.assertIn("json", props["format"].get("enum", []))

        status_schema = by_name["vscode_cli_status"]["input_schema"]
        anyof = status_schema["properties"]["install_dir"]["anyOf"]
        self.assertEqual(anyof[0].get("minLength"), 1)

    def test_codebase_search_rejects_blank_pattern(self) -> None:
        async def _run() -> None:
            result = await codebase_search(pattern="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("pattern is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_codebase_search_rejects_invalid_context(self) -> None:
        async def _run() -> None:
            result = await codebase_search(pattern="foo", context=-1)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("context must be an integer >= 0", str(result.get("error", "")))

        anyio.run(_run)

    def test_vscode_cli_status_rejects_blank_install_dir(self) -> None:
        async def _run() -> None:
            result = await vscode_cli_status(install_dir="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("install_dir must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_codebase_search_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await codebase_search(pattern="README", path=".")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("pattern"), "README")
            self.assertEqual(result.get("path"), ".")

        anyio.run(_run)

    def test_codebase_search_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await codebase_search(pattern="README", path="src")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("pattern"), "README")
            self.assertEqual(result.get("path"), "src")
            self.assertEqual(result.get("result", {}).get("matches"), [])
            self.assertEqual(result.get("result", {}).get("summary"), {})

        anyio.run(_run)

    def test_vscode_cli_status_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await vscode_cli_status(install_dir="/opt/code")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("installed"), False)
            self.assertEqual(result.get("install_dir"), "/opt/code")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
