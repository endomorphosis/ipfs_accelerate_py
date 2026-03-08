#!/usr/bin/env python3
"""UNI-007 cutover dry-run and rollback verification tests."""

import os
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp.server import (
    _reset_mcp_facade_telemetry,
    create_mcp_server,
    get_mcp_facade_telemetry,
)


class _DummyLegacyServer:
    def __init__(self):
        self.mcp = None


class _DummyUnifiedServer:
    pass


class TestUNI007CutoverRollback(unittest.TestCase):
    """Validate cutover dry-run and explicit rollback controls."""

    def setUp(self) -> None:
        _reset_mcp_facade_telemetry()

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

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("bridge_requested"))
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertFalse(telemetry.get("bridge_active"))
        self.assertTrue(telemetry.get("cutover_dry_run"))
        self.assertTrue(telemetry.get("dry_run_ok"))
        self.assertEqual(telemetry.get("reason"), "dry_run_legacy_fallback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("dry_run_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 0)

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

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertEqual(telemetry.get("reason"), "dry_run_failure_fallback")
        self.assertIn("dry-run-failure", str(telemetry.get("bridge_error") or ""))

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("bridge_failure_calls"), 1)

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

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_unified_bridge_path_records_facade_telemetry(self, mock_unified_create, mock_wrapper):
        mock_unified_create.return_value = _DummyUnifiedServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            server = create_mcp_server(name="unified-bridge")

        self.assertIs(server, mock_unified_create.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_not_called()

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("bridge_requested"))
        self.assertTrue(telemetry.get("bridge_active"))
        self.assertFalse(telemetry.get("used_legacy_wrapper"))
        self.assertEqual(telemetry.get("selected_runtime"), "unified")
        self.assertEqual(telemetry.get("reason"), "unified_bridge")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 0)


if __name__ == "__main__":
    unittest.main()
