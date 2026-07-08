#!/usr/bin/env python3
"""UNI-228 backend alias include-capabilities validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI228StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_non_boolean_include_capabilities(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(include_capabilities="yes")
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "include_capabilities must be a boolean",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_backend_alias_includes_capabilities_when_enabled(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory"],
                include_capabilities=True,
            )
            self.assertEqual(result.get("status"), "success")
            backends = result.get("backends") or []
            self.assertEqual(len(backends), 1)
            self.assertIn("capabilities", backends[0] or {})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
