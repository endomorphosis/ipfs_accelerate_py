#!/usr/bin/env python3
"""Focused CLI startup regression tests for MCP cutover behavior."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from ipfs_accelerate_py.mcp import cli
from ipfs_accelerate_py.mcp.server import _reset_mcp_facade_telemetry, get_mcp_facade_telemetry


class _DummyLegacyServer:
    def __init__(self) -> None:
        self.run = MagicMock()
        self.mcp = None


class TestMCPCLI(unittest.TestCase):
    def setUp(self) -> None:
        _reset_mcp_facade_telemetry()

    def test_main_starts_server_with_parsed_host_port(self) -> None:
        accelerate = object()
        mcp_server = MagicMock()
        cache = MagicMock()
        cache.get_stats.return_value = {}

        argv = [
            "mcp-cli",
            "--host",
            "127.0.0.1",
            "--port",
            "9010",
            "--no-p2p-service",
            "--no-p2p-task-worker",
        ]

        with patch.object(sys, "argv", argv):
            with patch("ipfs_accelerate_py.ipfs_accelerate_py", return_value=accelerate):
                with patch("ipfs_accelerate_py.github_cli.cache.get_global_cache", return_value=cache):
                    with patch("ipfs_accelerate_py.mcp.server.create_mcp_server", return_value=mcp_server) as mock_create:
                        cli.main()

        mock_create.assert_called_once_with(accelerate_instance=accelerate)
        mcp_server.run.assert_called_once_with(host="127.0.0.1", port=9010)

    def test_main_dev_mode_enables_reload(self) -> None:
        accelerate = object()
        mcp_server = MagicMock()
        cache = MagicMock()
        cache.get_stats.return_value = {}

        argv = [
            "mcp-cli",
            "--host",
            "0.0.0.0",
            "--port",
            "9020",
            "--dev",
            "--no-p2p-service",
            "--no-p2p-task-worker",
        ]

        with patch.object(sys, "argv", argv):
            with patch("ipfs_accelerate_py.ipfs_accelerate_py", return_value=accelerate):
                with patch("ipfs_accelerate_py.github_cli.cache.get_global_cache", return_value=cache):
                    with patch("ipfs_accelerate_py.mcp.server.create_mcp_server", return_value=mcp_server) as mock_create:
                        cli.main()

        mock_create.assert_called_once_with(accelerate_instance=accelerate)
        mcp_server.run.assert_called_once_with(host="0.0.0.0", port=9020, reload=True)

    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_main_tracks_d2_bridge_disable_override_telemetry(self, mock_unified_create: MagicMock) -> None:
        accelerate = object()
        cache = MagicMock()
        cache.get_stats.return_value = {}

        argv = [
            "mcp-cli",
            "--host",
            "127.0.0.1",
            "--port",
            "9012",
            "--no-p2p-service",
            "--no-p2p-task-worker",
        ]

        class _DummyUnifiedServer:
            def __init__(self) -> None:
                self.run = MagicMock()
                self.mcp = None

        unified_server = _DummyUnifiedServer()
        mock_unified_create.return_value = unified_server

        with patch.object(sys, "argv", argv):
            with patch.dict(
                "os.environ",
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "0",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                with patch("ipfs_accelerate_py.ipfs_accelerate_py", return_value=accelerate):
                    with patch("ipfs_accelerate_py.github_cli.cache.get_global_cache", return_value=cache):
                        cli.main()

        mock_unified_create.assert_called_once()
        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 1)
        self.assertEqual(counts.get("bridge_disable_ignored_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 0)

        telemetry = getattr(unified_server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("bridge_active"))
        self.assertTrue(telemetry.get("bridge_disable_ignored"))
        self.assertFalse(telemetry.get("force_legacy_rollback"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "unified_bridge")
        unified_server.run.assert_called_once_with(host="127.0.0.1", port=9012)

    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_main_preserves_d2_legacy_fallback_telemetry(self, mock_unified_create: MagicMock) -> None:
        accelerate = object()
        cache = MagicMock()
        cache.get_stats.return_value = {}
        created_servers: list[_DummyLegacyServer] = []

        argv = [
            "mcp-cli",
            "--host",
            "127.0.0.1",
            "--port",
            "9011",
            "--no-p2p-service",
            "--no-p2p-task-worker",
        ]

        def _make_legacy_server(*args, **kwargs) -> _DummyLegacyServer:
            del args, kwargs
            server = _DummyLegacyServer()
            created_servers.append(server)
            return server

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.object(sys, "argv", argv):
                with patch.dict(
                    "os.environ",
                    {
                        "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                        "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                        "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                    },
                    clear=False,
                ):
                    with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", side_effect=_make_legacy_server):
                        with patch("ipfs_accelerate_py.ipfs_accelerate_py", return_value=accelerate):
                            with patch("ipfs_accelerate_py.github_cli.cache.get_global_cache", return_value=cache):
                                cli.main()

        mock_unified_create.assert_not_called()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))
        self.assertEqual(len(created_servers), 1)

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

        created_server = created_servers[0]
        telemetry = getattr(created_server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")
        created_server.run.assert_called_once_with(host="127.0.0.1", port=9011)


if __name__ == "__main__":
    unittest.main()
