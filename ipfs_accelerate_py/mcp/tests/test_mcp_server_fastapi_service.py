#!/usr/bin/env python3
"""Tests for canonical mcp_server FastAPI facade modules."""

import os
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp_server.fastapi_config import UnifiedFastAPIConfig
from ipfs_accelerate_py.mcp_server.fastapi_service import create_fastapi_app, run_fastapi_server


class TestUnifiedFastAPIConfig(unittest.TestCase):
    def test_from_env_parses_expected_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_HOST": "127.0.0.1",
                "IPFS_MCP_PORT": "9012",
                "IPFS_MCP_MOUNT_PATH": "/api/mcp",
                "IPFS_MCP_NAME": "demo",
                "IPFS_MCP_DESCRIPTION": "demo-desc",
                "IPFS_MCP_VERBOSE": "true",
            },
            clear=False,
        ):
            cfg = UnifiedFastAPIConfig.from_env()

        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 9012)
        self.assertEqual(cfg.mount_path, "/api/mcp")
        self.assertEqual(cfg.name, "demo")
        self.assertEqual(cfg.description, "demo-desc")
        self.assertTrue(cfg.verbose)


class TestUnifiedFastAPIServiceFacade(unittest.TestCase):
    @patch("ipfs_accelerate_py.mcp.integration.create_standalone_app")
    def test_create_fastapi_app_delegates_to_integration(self, mock_create) -> None:
        mock_create.return_value = {"app": "ok"}
        cfg = UnifiedFastAPIConfig(name="svc", description="svc-desc", mount_path="/mcp")

        app = create_fastapi_app(cfg)

        self.assertEqual(app, {"app": "ok"})
        mock_create.assert_called_once_with(name="svc", description="svc-desc", mount_path="/mcp")

    @patch("ipfs_accelerate_py.mcp.integration.run_standalone_app")
    @patch("ipfs_accelerate_py.mcp.integration.create_standalone_app")
    def test_run_fastapi_server_delegates_to_runner(self, mock_create, mock_run) -> None:
        mock_create.return_value = {"app": "ok"}
        cfg = UnifiedFastAPIConfig(host="0.0.0.0", port=8899, verbose=True)

        run_fastapi_server(cfg)

        mock_create.assert_called_once()
        mock_run.assert_called_once_with({"app": "ok"}, host="0.0.0.0", port=8899, verbose=True)


if __name__ == "__main__":
    unittest.main()
