#!/usr/bin/env python3
"""UNI-207 deferred module adapter import-contract tests."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server import enterprise_api, grpc_transport, nl_ucan_policy


class TestMCPServerUNI207DeferredModuleAdapters(unittest.TestCase):
    def test_enterprise_adapter_import_contract(self) -> None:
        self.assertTrue(hasattr(enterprise_api, "__all__"))
        self.assertGreaterEqual(len(getattr(enterprise_api, "__all__", [])), 1)

    def test_grpc_adapter_import_contract(self) -> None:
        self.assertTrue(hasattr(grpc_transport, "__all__"))
        self.assertGreaterEqual(len(getattr(grpc_transport, "__all__", [])), 1)

    def test_nl_ucan_adapter_import_contract(self) -> None:
        self.assertTrue(hasattr(nl_ucan_policy, "__all__"))
        self.assertGreaterEqual(len(getattr(nl_ucan_policy, "__all__", [])), 1)


if __name__ == "__main__":
    unittest.main()
