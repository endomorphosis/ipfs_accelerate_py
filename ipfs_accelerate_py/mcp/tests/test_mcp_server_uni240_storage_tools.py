#!/usr/bin/env python3
"""UNI-240 backend alias unavailable-reasons value-type parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
)


class TestMCPServerUNI240StorageTools(unittest.TestCase):
    def test_backend_alias_rejects_non_string_unavailable_reason_value(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["ipfs"],
                unavailable_reasons={"ipfs": 1},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_reasons must be an object with non-empty string keys/values",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()