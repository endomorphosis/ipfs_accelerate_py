#!/usr/bin/env python3
"""UNI-226 backend alias unavailable-backends validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI226StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_unknown_unavailable_backend(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(unavailable_backends=["tape"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_backends contains unknown storage backends",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_backend_alias_reports_invalid_unavailable_backends_list(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(unavailable_backends=["ipfs", "tape", "legacy"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_backends contains unknown storage backends",
                str(result.get("error", "")),
            )
            self.assertEqual(result.get("invalid_backends"), ["legacy", "tape"])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
