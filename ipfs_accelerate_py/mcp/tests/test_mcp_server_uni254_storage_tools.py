#!/usr/bin/env python3
"""UNI-254 backend alias generated-at passthrough parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI254StorageTools(unittest.TestCase):
    def test_backend_alias_surfaces_backend_report_generated_at(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
            )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 2)
            self.assertIsInstance(result.get("generated_at"), str)
            self.assertEqual(
                result.get("generated_at"),
                (result.get("backend_report") or {}).get("generated_at"),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()