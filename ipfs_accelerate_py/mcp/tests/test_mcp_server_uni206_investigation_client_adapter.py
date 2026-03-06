#!/usr/bin/env python3
"""UNI-206 investigation client adapter parity tests."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server import investigation_mcp_client


class TestMCPServerUNI206InvestigationClientAdapter(unittest.TestCase):
    def test_adapter_exposes_expected_symbols(self) -> None:
        self.assertTrue(hasattr(investigation_mcp_client, "InvestigationMCPClient"))
        self.assertTrue(hasattr(investigation_mcp_client, "InvestigationMCPClientError"))
        self.assertTrue(hasattr(investigation_mcp_client, "create_investigation_mcp_client"))

    def test_factory_returns_client_instance(self) -> None:
        client = investigation_mcp_client.create_investigation_mcp_client()
        self.assertIsInstance(client, investigation_mcp_client.InvestigationMCPClient)


if __name__ == "__main__":
    unittest.main()
