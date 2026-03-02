#!/usr/bin/env python3
"""Tests for canonical mcp_server standalone server facades."""

import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp_server import standalone_server


class TestMCPServerStandaloneFacade(unittest.TestCase):
    @patch("ipfs_accelerate_py.mcp.standalone.run_server")
    def test_run_server_delegates_to_legacy_standalone(self, mock_run) -> None:
        standalone_server.run_server(
            host="127.0.0.1",
            port=9901,
            name="demo",
            description="demo-desc",
            verbose=True,
        )

        mock_run.assert_called_once_with(
            host="127.0.0.1",
            port=9901,
            name="demo",
            description="demo-desc",
            verbose=True,
        )

    @patch("ipfs_accelerate_py.mcp_server.standalone_server.run_canonical_fastapi_server")
    def test_run_fastapi_server_uses_canonical_fastapi_facade(self, mock_run_fastapi) -> None:
        standalone_server.run_fastapi_server(
            host="0.0.0.0",
            port=8899,
            mount_path="/mcp",
            name="demo",
            description="demo-desc",
            verbose=False,
        )

        self.assertEqual(mock_run_fastapi.call_count, 1)
        cfg = mock_run_fastapi.call_args.args[0]
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertEqual(cfg.port, 8899)
        self.assertEqual(cfg.mount_path, "/mcp")
        self.assertEqual(cfg.name, "demo")
        self.assertEqual(cfg.description, "demo-desc")
        self.assertFalse(cfg.verbose)


if __name__ == "__main__":
    unittest.main()
