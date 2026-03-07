#!/usr/bin/env python3
"""UNI-232 backend alias include-breakdown validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI232StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_non_boolean_include_breakdown(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(include_breakdown="yes")
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "include_breakdown must be a boolean",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_backend_alias_includes_breakdown_when_enabled(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            breakdown = result.get("breakdown") or {}
            self.assertEqual(breakdown.get("available_count"), 1)
            self.assertEqual(breakdown.get("unavailable_count"), 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
