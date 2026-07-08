#!/usr/bin/env python3
"""UNI-255 backend alias backend-count passthrough parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI255StorageTools(unittest.TestCase):
    def test_backend_alias_surfaces_backend_report_backend_count(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
            )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 2)
            self.assertEqual(
                result.get("backend_count"),
                (result.get("backend_report") or {}).get("backend_count"),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()