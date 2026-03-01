#!/usr/bin/env python3
"""Process-level transport helper tests for MCP integration module."""

import types
import unittest
from unittest.mock import MagicMock, patch

from ipfs_accelerate_py.mcp.integration import create_standalone_app, run_standalone_app


class _DummyServer:
    def __init__(self):
        self.app = object()


class TestMCPTransportProcessLevel(unittest.TestCase):
    """Validate standalone app creation and uvicorn run wiring."""

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server", return_value=_DummyServer())
    def test_create_standalone_app_mounts_mcp_server(self, _mock_create_server: MagicMock) -> None:
        app = create_standalone_app(mount_path="/mcp", name="demo", description="demo server")

        # Works with both real FastAPI app and fallback app.
        self.assertTrue(hasattr(app, "_mcp_server"))
        mounted_server = getattr(app, "_mcp_server")
        self.assertIsNotNone(mounted_server)

        if hasattr(app, "mounts"):
            # Fallback app path.
            self.assertEqual(app.mounts[0]["path"], "/mcp")

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server", return_value=_DummyServer())
    def test_create_standalone_app_uses_fallback_when_fastapi_missing(self, _mock_create_server: MagicMock) -> None:
        with patch.dict("sys.modules", {"fastapi": None}):
            app = create_standalone_app(mount_path="/mcp", name="demo", description="demo server")

        self.assertTrue(hasattr(app, "mounts"))
        self.assertEqual(app.mounts[0]["path"], "/mcp")
        self.assertEqual(app.mounts[0]["name"], "mcp_server")

    def test_run_standalone_app_invokes_uvicorn(self) -> None:
        mock_run = MagicMock()
        fake_uvicorn = types.SimpleNamespace(run=mock_run)

        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn}):
            run_standalone_app(app=object(), host="127.0.0.1", port=8899, verbose=True)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 8899)
        self.assertEqual(kwargs["log_level"], "debug")


if __name__ == "__main__":
    unittest.main()
