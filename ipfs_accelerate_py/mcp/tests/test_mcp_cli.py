#!/usr/bin/env python3
"""Focused CLI startup regression tests for MCP cutover behavior."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from ipfs_accelerate_py.mcp import cli


class TestMCPCLI(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
