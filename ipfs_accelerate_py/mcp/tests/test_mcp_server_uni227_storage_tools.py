#!/usr/bin/env python3
"""UNI-227 backend alias availability-filter normalization parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI227StorageTools(unittest.TestCase):
    def test_backend_alias_normalizes_availability_filter_casing_and_whitespace(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                availability_filter=" UnAvAiLaBlE ",
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("availability_filter"), "unavailable")
            self.assertEqual(result.get("backend_count"), 1)
            backends = result.get("backends") or []
            self.assertEqual(len(backends), 1)
            self.assertEqual((backends[0] or {}).get("storage_type"), "ipfs")
            self.assertEqual((backends[0] or {}).get("available"), False)

        anyio.run(_run)

    def test_backend_alias_normalizes_available_filter(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                availability_filter=" AvAiLaBlE ",
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("availability_filter"), "available")
            self.assertEqual(result.get("backend_count"), 1)
            backends = result.get("backends") or []
            self.assertEqual(len(backends), 1)
            self.assertEqual((backends[0] or {}).get("storage_type"), "memory")
            self.assertEqual((backends[0] or {}).get("available"), True)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
