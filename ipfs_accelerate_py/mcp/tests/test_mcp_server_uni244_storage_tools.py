#!/usr/bin/env python3
"""UNI-244 backend alias unavailable-reason exposure parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI244StorageTools(unittest.TestCase):
    def test_backend_alias_only_exposes_unavailable_reason_for_unavailable_backends(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                unavailable_reasons={
                    "memory": "should not appear",
                    "ipfs": "dial timeout",
                },
            )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 2)
            backends = result.get("backends") or []
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual((by_type.get("ipfs") or {}).get("unavailable_reason"), "dial timeout")
            self.assertIsNone((by_type.get("memory") or {}).get("unavailable_reason"))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()