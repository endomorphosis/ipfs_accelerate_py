#!/usr/bin/env python3
"""UNI-252 backend alias default capability omission parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI252StorageTools(unittest.TestCase):
    def test_backend_alias_omits_capabilities_when_not_requested(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
            )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("backend_count"), 2)
            backends = result.get("backends") or []
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual(set(by_type.keys()), {"memory", "ipfs"})
            self.assertNotIn("capabilities", by_type.get("memory") or {})
            self.assertNotIn("capabilities", by_type.get("ipfs") or {})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()