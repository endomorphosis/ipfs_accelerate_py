#!/usr/bin/env python3
"""UNI-221 storage collection get-alias not-found envelope parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_collection,
)


class TestMCPServerUNI221StorageTools(unittest.TestCase):
    def test_get_collection_alias_returns_not_found_envelope_for_missing_collection(self) -> None:
        async def _run() -> None:
            result = await get_storage_collection(collection_name="uni221-missing")
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("collection_name"), "uni221-missing")
            self.assertEqual(result.get("found"), False)
            self.assertIn("not found", str(result.get("error", "")).lower())

        anyio.run(_run)

    def test_get_collection_alias_returns_found_on_success(self) -> None:
        async def _run() -> None:
            # default collection is available in fallback storage manager
            result = await get_storage_collection(collection_name="default")
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("collection_name"), "default")
            self.assertEqual(result.get("found"), True)
            self.assertIn("collection", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
