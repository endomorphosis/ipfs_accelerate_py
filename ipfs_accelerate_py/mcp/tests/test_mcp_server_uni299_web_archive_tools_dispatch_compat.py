#!/usr/bin/env python3
"""UNI-299 web-archive tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.web_archive_tools import native_web_archive_tools


class TestMCPServerUNI299WebArchiveToolsDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_web_archive_tools_dispatch_infers_error_status_from_contradictory_delegate_payloads(
        self, mock_wrapper
    ) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failure"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_web_archive_tools._API,
                {
                    "create_warc": _contradictory_failure,
                    "archive_to_wayback": _contradictory_failure,
                    "search_common_crawl": _contradictory_failure,
                    "get_common_crawl_content": _contradictory_failure,
                    "search_github_repositories": _contradictory_failure,
                    "unified_search": _contradictory_failure,
                    "unified_fetch": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="web-archive-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                created = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "create_warc",
                        {"url": "https://example.com"},
                    )
                )
                self.assertEqual(created.get("status"), "error")
                self.assertEqual(created.get("success"), False)
                self.assertEqual(created.get("error"), "delegate failure")

                archived = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "archive_to_wayback",
                        {"url": "https://example.com"},
                    )
                )
                self.assertEqual(archived.get("status"), "error")
                self.assertEqual(archived.get("success"), False)
                self.assertEqual(archived.get("error"), "delegate failure")

                searched = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "search_common_crawl",
                        {"domain": "example.com", "limit": 10},
                    )
                )
                self.assertEqual(searched.get("status"), "error")
                self.assertEqual(searched.get("success"), False)
                self.assertEqual(searched.get("error"), "delegate failure")

                content = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "get_common_crawl_content",
                        {"url": "https://example.com", "timestamp": "20240101000000"},
                    )
                )
                self.assertEqual(content.get("status"), "error")
                self.assertEqual(content.get("success"), False)
                self.assertEqual(content.get("error"), "delegate failure")

                repositories = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "search_github_repositories",
                        {"query": "ipfs", "order": "desc", "per_page": 5, "page": 1},
                    )
                )
                self.assertEqual(repositories.get("status"), "error")
                self.assertEqual(repositories.get("success"), False)
                self.assertEqual(repositories.get("error"), "delegate failure")

                unified_search_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "unified_search",
                        {"query": "ipfs", "max_results": 5, "mode": "balanced"},
                    )
                )
                self.assertEqual(unified_search_result.get("status"), "error")
                self.assertEqual(unified_search_result.get("success"), False)
                self.assertEqual(unified_search_result.get("error"), "delegate failure")

                unified_fetch_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "web_archive_tools",
                        "unified_fetch",
                        {"url": "https://example.com", "mode": "balanced"},
                    )
                )
                self.assertEqual(unified_fetch_result.get("status"), "error")
                self.assertEqual(unified_fetch_result.get("success"), False)
                self.assertEqual(unified_fetch_result.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
