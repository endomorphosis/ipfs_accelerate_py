#!/usr/bin/env python3
"""Tests for canonical MCP server DID key manager."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ipfs_accelerate_py.mcp_server import did_key_manager as did_mod


class TestDIDKeyManager(unittest.TestCase):
    """Validate canonical DID key manager behavior."""

    def test_singleton_respects_key_file_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "did_key.json"
            mgr = did_mod.get_did_key_manager(key_file=key_file)
            self.assertEqual(mgr.key_file, key_file)

            cached = did_mod.get_did_key_manager()
            self.assertIs(cached, mgr)

    def test_stub_mode_contract_without_ucan(self) -> None:
        if did_mod._UCAN_AVAILABLE:
            self.skipTest("py-ucan is installed; stub-mode contract test is not applicable")

        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "did_key.json"
            mgr = did_mod.DIDKeyManager(key_file=key_file)
            self.assertEqual(mgr.did, "did:key:stub-ucan-not-installed")
            with self.assertRaises(RuntimeError):
                mgr.export_secret_b64()


if __name__ == "__main__":
    unittest.main()
