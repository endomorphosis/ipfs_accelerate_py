#!/usr/bin/env python3
"""UNI-225 backend alias unknown-backend validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI225StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_unknown_backend_type(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(backend_types=["tape"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "backend_types contains unknown storage backends",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_backend_alias_rejects_unknown_unavailable_reason_backend(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(unavailable_reasons={"tape": "unsupported"})
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_reasons contains unknown storage backends",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
