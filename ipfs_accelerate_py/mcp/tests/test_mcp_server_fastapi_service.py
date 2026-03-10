#!/usr/bin/env python3
"""Tests for canonical mcp_server FastAPI facade modules."""

import os
import unittest
from unittest.mock import patch

import ipfs_accelerate_py.mcp_server.fastapi_service as fastapi_service_module
from ipfs_accelerate_py.mcp_server.fastapi_config import UnifiedFastAPIConfig
from ipfs_accelerate_py.mcp_server.fastapi_service import (
    create_fastapi_app,
    get_fastapi_app,
    get_fastapi_config,
    run_fastapi_server,
)


class _EnvResetMixin:
    def setUp(self) -> None:
        fastapi_service_module._DEFAULT_CONFIG = None
        fastapi_service_module._DEFAULT_APP = None


class TestUnifiedFastAPIConfig(_EnvResetMixin, unittest.TestCase):
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

    def test_from_env_parses_legacy_fallback_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_HOST": "",
                "IPFS_MCP_PORT": "",
                "IPFS_MCP_MOUNT_PATH": "",
                "IPFS_MCP_NAME": "",
                "IPFS_MCP_DESCRIPTION": "",
                "IPFS_MCP_VERBOSE": "",
                "HOST": "10.0.0.4",
                "PORT": "9911",
                "MOUNT_PATH": "/legacy-mcp",
                "APP_NAME": "legacy-app",
                "APP_DESCRIPTION": "legacy-desc",
                "DEBUG": "1",
            },
            clear=False,
        ):
            cfg = UnifiedFastAPIConfig.from_env()

        self.assertEqual(cfg.host, "10.0.0.4")
        self.assertEqual(cfg.port, 9911)
        self.assertEqual(cfg.mount_path, "/legacy-mcp")
        self.assertEqual(cfg.name, "legacy-app")
        self.assertEqual(cfg.description, "legacy-desc")
        self.assertTrue(cfg.verbose)


class TestUnifiedFastAPIServiceFacade(unittest.TestCase):
    @patch("ipfs_accelerate_py.mcp.integration.create_standalone_app")
    def test_create_fastapi_app_delegates_to_integration(self, mock_create) -> None:
        mock_create.return_value = {"app": "ok"}
        cfg = UnifiedFastAPIConfig(name="svc", description="svc-desc", mount_path="/mcp", verbose=True)

        app = create_fastapi_app(cfg)

        self.assertEqual(app, {"app": "ok"})
        mock_create.assert_called_once_with(
            name="svc",
            description="svc-desc",
            mount_path="/mcp",
            verbose=True,
        )

    @patch("ipfs_accelerate_py.mcp.integration.run_standalone_app")
    @patch("ipfs_accelerate_py.mcp.integration.create_standalone_app")
    def test_run_fastapi_server_delegates_to_runner(self, mock_create, mock_run) -> None:
        mock_create.return_value = {"app": "ok"}
        cfg = UnifiedFastAPIConfig(host="0.0.0.0", port=8899, verbose=True)

        run_fastapi_server(cfg)

        mock_create.assert_called_once()
        mock_run.assert_called_once_with({"app": "ok"}, host="0.0.0.0", port=8899, verbose=True)

    @patch("ipfs_accelerate_py.mcp.integration.create_standalone_app")
    def test_get_fastapi_config_and_app_are_lazy_cached_compat_exports(self, mock_create) -> None:
        mock_create.return_value = {"app": "cached"}

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_HOST": "127.0.0.8",
                "IPFS_MCP_PORT": "9123",
                "IPFS_MCP_MOUNT_PATH": "/cached-mcp",
                "IPFS_MCP_NAME": "cached-name",
                "IPFS_MCP_DESCRIPTION": "cached-desc",
                "IPFS_MCP_VERBOSE": "yes",
            },
            clear=False,
        ):
            settings = get_fastapi_config()
            app = get_fastapi_app()
            module_settings = fastapi_service_module.settings
            module_app = fastapi_service_module.app

        self.assertIs(settings, module_settings)
        self.assertIs(app, module_app)
        self.assertEqual(settings.host, "127.0.0.8")
        self.assertEqual(settings.port, 9123)
        self.assertEqual(settings.mount_path, "/cached-mcp")
        self.assertEqual(settings.name, "cached-name")
        self.assertEqual(settings.description, "cached-desc")
        self.assertTrue(settings.verbose)
        self.assertEqual(app, {"app": "cached"})
        mock_create.assert_called_once_with(
            name="cached-name",
            description="cached-desc",
            mount_path="/cached-mcp",
            verbose=True,
        )


if __name__ == "__main__":
    unittest.main()
