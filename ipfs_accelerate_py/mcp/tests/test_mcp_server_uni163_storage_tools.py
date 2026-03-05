#!/usr/bin/env python3
"""UNI-163 storage tools reporting parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class TestMCPServerUNI163StorageTools(unittest.TestCase):
    def test_manage_collections_stats_rejects_invalid_report_format(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="stats",
                report_format="xml",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("report_format must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_collections_stats_rejects_invalid_include_breakdown_type(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="stats",
                include_breakdown="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("include_breakdown must be a boolean", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_collections_stats_returns_storage_report_envelope(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="stats",
                report_format="summary",
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertIn("storage_report", result)
            report = result["storage_report"]
            self.assertEqual(report.get("report_format"), "summary")
            self.assertIn("summary", report)
            self.assertIn("breakdown", report)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
