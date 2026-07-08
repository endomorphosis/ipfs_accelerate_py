#!/usr/bin/env python3
"""UNI-245 backend alias availability-filter default normalization tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI245StorageTools(unittest.TestCase):
    def test_backend_alias_normalizes_empty_availability_filter_to_all(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                availability_filter="",
            )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("availability_filter"), "all")
            self.assertEqual(result.get("backend_count"), 2)
            backends = result.get("backends") or []
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual(set(by_type.keys()), {"memory", "ipfs"})
            self.assertEqual((by_type.get("memory") or {}).get("available"), True)
            self.assertEqual((by_type.get("ipfs") or {}).get("available"), False)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()