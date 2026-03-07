#!/usr/bin/env python3
"""UNI-140 software_engineering_tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools import (
    analyze_github_actions,
    coordinate_auto_healing,
    detect_error_patterns,
    parse_kubernetes_logs,
    parse_systemd_logs,
    parse_workflow_logs,
    register_native_software_engineering_tools,
    scrape_repository,
    search_repositories,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI140SoftwareEngineeringTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_software_engineering_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        self.assertIn("analyze_github_actions", by_name)
        self.assertIn("parse_systemd_logs", by_name)
        self.assertIn("detect_error_patterns", by_name)
        self.assertIn("coordinate_auto_healing", by_name)

        scrape_schema = by_name["scrape_repository"]["input_schema"]
        self.assertEqual(scrape_schema["properties"]["max_items"]["minimum"], 1)

        actions_schema = by_name["analyze_github_actions"]["input_schema"]
        self.assertEqual(actions_schema["properties"]["max_runs"]["minimum"], 1)

        systemd_schema = by_name["parse_systemd_logs"]["input_schema"]
        self.assertIn("warning", systemd_schema["properties"]["priority_filter"]["enum"])

    def test_scrape_repository_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_repository(repository_url="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("repository_url", str(result.get("error", "")))

            result = await scrape_repository(repository_url="https://example.com/repo")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("github.com", str(result.get("error", "")))

            result = await scrape_repository(
                repository_url="https://github.com/example/repo",
                max_items=0,
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_items", str(result.get("error", "")))

        anyio.run(_run)

    def test_analyze_github_actions_validation(self) -> None:
        async def _run() -> None:
            result = await analyze_github_actions(repository_url="https://github.com/example/repo", max_runs=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_runs", str(result.get("error", "")))

        anyio.run(_run)

    def test_parse_systemd_logs_validation(self) -> None:
        async def _run() -> None:
            result = await parse_systemd_logs(log_content="svc log", priority_filter="panic")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("priority_filter must be null or one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_detect_error_patterns_validation(self) -> None:
        async def _run() -> None:
            result = await detect_error_patterns(error_logs="timeout")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("error_logs must be a list of non-empty strings", str(result.get("error", "")))

        anyio.run(_run)

    def test_coordinate_auto_healing_validation(self) -> None:
        async def _run() -> None:
            result = await coordinate_auto_healing(error_report={})
            self.assertEqual(result.get("status"), "error")
            self.assertIn("error_report must be a non-empty object", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            scrape_result = await scrape_repository(
                repository_url="https://github.com/example/repo",
                max_items=1,
            )
            self.assertIn(scrape_result.get("status"), ["success", "error"])
            self.assertEqual(scrape_result.get("repository_url"), "https://github.com/example/repo")

            search_result = await search_repositories(query="mcp", max_results=1)
            self.assertIn(search_result.get("status"), ["success", "error"])
            self.assertEqual(search_result.get("query"), "mcp")
            self.assertEqual(search_result.get("max_results"), 1)

        anyio.run(_run)

    def test_parse_workflow_logs_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await parse_workflow_logs(log_content="error: boom")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("errors"), [])
            self.assertEqual(result.get("warnings"), [])
            self.assertEqual(result.get("patterns"), [])
            self.assertEqual(result.get("statistics"), {"total_lines": 0, "error_lines": 0, "warning_lines": 0})

        anyio.run(_run)

    def test_parse_kubernetes_logs_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await parse_kubernetes_logs(log_content="2026-01-01T00:00:00.000Z INFO [api] ok")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("entries"), [])
            self.assertEqual(result.get("errors"), [])
            self.assertEqual(result.get("recommendations"), [])

        anyio.run(_run)

    def test_coordinate_auto_healing_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await coordinate_auto_healing(error_report={"success": True, "patterns": []})

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("healing_actions"), [])
            self.assertEqual(result.get("executed"), False)
            self.assertEqual(result.get("results"), [])
            self.assertEqual(result.get("recommendations"), [])

        anyio.run(_run)

    def test_search_repositories_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.software_engineering_tools.native_software_engineering_tools._API"
            ) as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"error": "github unavailable"}

                mock_api.__getitem__.return_value = _impl
                result = await search_repositories(query="mcp")

            self.assertEqual(result.get("status"), "error")
            self.assertIn("github unavailable", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
