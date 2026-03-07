#!/usr/bin/env python3
"""UNI-242 backend alias list normalization parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI242StorageTools(unittest.TestCase):
    def test_backend_alias_normalizes_backend_types_casing_and_whitespace(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=[" Memory ", " IPFS "],
                unavailable_backends=["ipfs"],
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 2)
            backends = result.get("backends") or []
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual(set(by_type.keys()), {"memory", "ipfs"})
            self.assertEqual((by_type.get("memory") or {}).get("available"), True)
            self.assertEqual((by_type.get("ipfs") or {}).get("available"), False)

        anyio.run(_run)

    def test_backend_alias_normalizes_unavailable_backends_casing_and_whitespace(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=[" IpFs "],
                availability_filter="unavailable",
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 1)
            backends = result.get("backends") or []
            self.assertEqual(len(backends), 1)
            self.assertEqual((backends[0] or {}).get("storage_type"), "ipfs")
            self.assertEqual((backends[0] or {}).get("available"), False)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()