#!/usr/bin/env python3
"""UNI-132 development tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

import ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools as native_development_tools

from ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools import (
    codebase_search,
    documentation_generator,
    lint_python_codebase,
    register_native_development_tools,
    run_comprehensive_tests,
    test_generator,
    vscode_cli_execute,
    vscode_cli_status,
    vscode_cli_tunnel_login,
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

        self.assertIn("documentation_generator", by_name)
        self.assertIn("lint_python_codebase", by_name)
        self.assertIn("run_comprehensive_tests", by_name)
        self.assertIn("test_generator", by_name)
        self.assertIn("vscode_cli_execute", by_name)

        search_schema = by_name["codebase_search"]["input_schema"]
        props = search_schema["properties"]
        self.assertEqual(props["context"].get("minimum"), 0)
        self.assertIn("json", props["format"].get("enum", []))

        docs_schema = by_name["documentation_generator"]["input_schema"]
        self.assertIn("html", docs_schema["properties"]["format_type"].get("enum", []))

        execute_schema = by_name["vscode_cli_execute"]["input_schema"]
        self.assertEqual(execute_schema["properties"]["timeout"].get("maximum"), 300)

    def test_codebase_search_rejects_blank_pattern(self) -> None:
        async def _run() -> None:
            result = await codebase_search(pattern="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("pattern is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_documentation_generator_rejects_invalid_format_type(self) -> None:
        async def _run() -> None:
            result = await documentation_generator(input_path="src", format_type="pdf")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("format_type must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_lint_python_codebase_rejects_invalid_patterns(self) -> None:
        async def _run() -> None:
            result = await lint_python_codebase(patterns="*.py")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("patterns must be a list", str(result.get("error", "")))

        anyio.run(_run)

    def test_run_comprehensive_tests_rejects_invalid_framework(self) -> None:
        async def _run() -> None:
            result = await run_comprehensive_tests(test_framework="nose")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("test_framework must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_test_generator_rejects_missing_specification(self) -> None:
        async def _run() -> None:
            result = await test_generator(name="sample", test_specification=None)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("test_specification is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_vscode_cli_execute_rejects_invalid_timeout(self) -> None:
        async def _run() -> None:
            result = await vscode_cli_execute(command=["--version"], timeout=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout must be an integer between 1 and 300", str(result.get("error", "")))

        anyio.run(_run)

    def test_vscode_cli_tunnel_login_rejects_invalid_provider(self) -> None:
        async def _run() -> None:
            result = await vscode_cli_tunnel_login(provider="gitlab")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("provider must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_vscode_cli_status_rejects_blank_install_dir(self) -> None:
        async def _run() -> None:
            result = await vscode_cli_status(install_dir="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("install_dir must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _run() -> None:
            async def _contradictory_failure(**_: object) -> dict:
                return {"status": "success", "success": False, "error": "delegate failure"}

            with patch.dict(
                native_development_tools._API,
                {
                    "codebase_search": _contradictory_failure,
                    "documentation_generator": _contradictory_failure,
                    "run_comprehensive_tests": _contradictory_failure,
                    "vscode_cli_execute": _contradictory_failure,
                    "vscode_cli_status": _contradictory_failure,
                },
                clear=False,
            ):
                searched = await codebase_search(pattern="README", path="src")
                self.assertEqual(searched.get("status"), "error")
                self.assertEqual(searched.get("success"), False)
                self.assertEqual(searched.get("error"), "delegate failure")

                documented = await documentation_generator(input_path="src", output_path="docs")
                self.assertEqual(documented.get("status"), "error")
                self.assertEqual(documented.get("success"), False)
                self.assertEqual(documented.get("error"), "delegate failure")

                tested = await run_comprehensive_tests(path=".")
                self.assertEqual(tested.get("status"), "error")
                self.assertEqual(tested.get("success"), False)
                self.assertEqual(tested.get("error"), "delegate failure")

                executed = await vscode_cli_execute(command=["--version"], timeout=30)
                self.assertEqual(executed.get("status"), "error")
                self.assertEqual(executed.get("success"), False)
                self.assertEqual(executed.get("error"), "delegate failure")

                status = await vscode_cli_status(install_dir="/opt/code")
                self.assertEqual(status.get("status"), "error")
                self.assertEqual(status.get("success"), False)
                self.assertEqual(status.get("error"), "delegate failure")

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

    def test_documentation_generator_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await documentation_generator(input_path="src", output_path="docs")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("input_path"), "src")
            self.assertEqual(result.get("output_path"), "docs")
            self.assertEqual(result.get("result", {}).get("files_generated"), [])
            self.assertEqual(result.get("result", {}).get("format_type"), "markdown")

        anyio.run(_run)

    def test_run_comprehensive_tests_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await run_comprehensive_tests(path=".")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("path"), ".")
            self.assertEqual(result.get("suite_results"), {})
            self.assertEqual(
                result.get("summary"),
                {"total_passed": 0, "total_failed": 0, "total_skipped": 0},
            )

        anyio.run(_run)

    def test_vscode_cli_execute_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.development_tools.native_development_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await vscode_cli_execute(command=["--version"], timeout=30)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("command"), ["--version"])
            self.assertEqual(result.get("returncode"), 0)
            self.assertEqual(result.get("stdout"), "")
            self.assertEqual(result.get("stderr"), "")

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
