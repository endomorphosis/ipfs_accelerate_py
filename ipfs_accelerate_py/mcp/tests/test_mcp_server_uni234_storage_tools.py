#!/usr/bin/env python3
"""UNI-234 backend alias list-entry validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI234StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_empty_backend_type_entry(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(backend_types=["memory", ""])
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "backend_types must be an array of non-empty strings",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_backend_alias_rejects_empty_unavailable_backend_entry(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(unavailable_backends=["ipfs", " "])
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_backends must be an array of non-empty strings",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
