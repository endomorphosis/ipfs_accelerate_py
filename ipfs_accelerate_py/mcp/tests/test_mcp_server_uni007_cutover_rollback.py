#!/usr/bin/env python3
"""UNI-007 cutover dry-run and rollback verification tests."""

import os
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp.server import create_mcp_server


class _DummyLegacyServer:
    def __init__(self):
        self.mcp = None


class _DummyUnifiedServer:
    pass


class TestUNI007CutoverRollback(unittest.TestCase):
    """Validate cutover dry-run and explicit rollback controls."""

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_cutover_dry_run_validates_unified_then_stays_legacy(self, mock_unified_create, mock_wrapper):
        mock_unified_create.return_value = _DummyUnifiedServer()
        mock_wrapper.return_value = _DummyLegacyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "1",
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
            },
            clear=False,
        ):
            server = create_mcp_server(name="cutover-dry-run")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_called_once()

        status = getattr(server, "_unified_cutover_dry_run", {})
        self.assertTrue(status.get("enabled"))
        self.assertTrue(status.get("ok"))
        self.assertEqual(status.get("error"), "")

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_cutover_dry_run_failure_records_error_and_falls_back(self, mock_unified_create, mock_wrapper):
        mock_unified_create.side_effect = RuntimeError("dry-run-failure")
        mock_wrapper.return_value = _DummyLegacyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="cutover-dry-run-failure")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_called_once()

        status = getattr(server, "_unified_cutover_dry_run", {})
        self.assertTrue(status.get("enabled"))
        self.assertFalse(status.get("ok"))
        self.assertIn("dry-run-failure", str(status.get("error") or ""))

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_force_legacy_rollback_overrides_bridge_flag(self, mock_unified_create, mock_wrapper):
        mock_wrapper.return_value = _DummyLegacyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            server = create_mcp_server(name="force-rollback")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_not_called()
        mock_wrapper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
