#!/usr/bin/env python3
"""Process-level transport helper tests for MCP integration module."""

import os
import types
import unittest
from unittest.mock import MagicMock, patch

from ipfs_accelerate_py.mcp.fastapi_integration import integrate_mcp_with_fastapi
from ipfs_accelerate_py.mcp.integration import create_standalone_app, run_standalone_app
from ipfs_accelerate_py.mcp.integration import initialize_mcp_server
from ipfs_accelerate_py.mcp.server import _reset_mcp_facade_telemetry, get_mcp_facade_telemetry


class _DummyServer:
    def __init__(self):
        self.app = object()
        self.tools = {}
        self.mcp = None

    def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema,
            "execution_context": execution_context,
            "tags": tags,
        }


class _DummyApp:
    def __init__(self) -> None:
        self.mount = MagicMock()


class _DummyModelServer:
    def __init__(self, accelerate_instance=None) -> None:
        self.resources = {}
        if accelerate_instance is not None:
            self.resources["ipfs_accelerate_py"] = accelerate_instance


class TestMCPTransportProcessLevel(unittest.TestCase):
    """Validate standalone app creation and uvicorn run wiring."""

    def setUp(self) -> None:
        _reset_mcp_facade_telemetry()

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

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", side_effect=lambda *args, **kwargs: _DummyServer())
    def test_create_standalone_app_preserves_additive_profile_metadata(self, _mock_wrapper: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            app = create_standalone_app(mount_path="/mcp", name="demo", description="demo server")

        mounted_server = getattr(app, "_mcp_server")
        expected_profiles = [
            "mcp++/profile-a-idl",
            "mcp++/profile-b-cid-artifacts",
            "mcp++/profile-c-ucan",
            "mcp++/profile-d-temporal-policy",
            "mcp++/profile-e-mcp-p2p",
        ]

        self.assertEqual(getattr(mounted_server, "_unified_supported_profiles", []), expected_profiles)
        self.assertEqual(
            getattr(mounted_server, "_unified_profile_negotiation", {}).get("profiles"),
            expected_profiles,
        )
        self.assertTrue(getattr(mounted_server, "_unified_profile_negotiation", {}).get("supports_profile_negotiation"))
        self.assertEqual(
            getattr(mounted_server, "_unified_profile_negotiation", {}).get("mode"),
            "optional_additive",
        )
        self.assertEqual(
            (getattr(mounted_server, "_unified_server_context_snapshot", {}) or {}).get("profile_negotiation"),
            getattr(mounted_server, "_unified_profile_negotiation", {}),
        )

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", side_effect=lambda *args, **kwargs: _DummyServer())
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_create_standalone_app_preserves_d2_legacy_fallback_telemetry(self, mock_unified_create: MagicMock, _mock_wrapper: MagicMock) -> None:
        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                app = create_standalone_app(mount_path="/mcp", name="demo", description="demo server")

        mounted_server = getattr(app, "_mcp_server")
        telemetry = getattr(mounted_server, "_mcp_facade_telemetry", {})

        mock_unified_create.assert_not_called()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

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

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server", return_value=_DummyServer())
    def test_initialize_mcp_server_mounts_blank_internal_prefix(self, mock_create_server: MagicMock) -> None:
        app = _DummyApp()
        accelerate = object()

        server = initialize_mcp_server(app, accelerate, mount_path="/api/mcp")

        self.assertIs(server, mock_create_server.return_value)
        mock_create_server.assert_called_once_with(accelerate_instance=accelerate, mount_path="")
        app.mount.assert_called_once_with("/api/mcp", server.app, name="mcp_server")

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", side_effect=lambda *args, **kwargs: _DummyServer())
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_initialize_mcp_server_preserves_d2_legacy_fallback_telemetry(self, mock_unified_create: MagicMock, _mock_wrapper: MagicMock) -> None:
        app = _DummyApp()
        accelerate = object()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                server = initialize_mcp_server(app, accelerate, mount_path="/api/mcp")

        telemetry = getattr(server, "_mcp_facade_telemetry", {})

        mock_unified_create.assert_not_called()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")
        app.mount.assert_called_once_with("/api/mcp", server.app, name="mcp_server")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

    @patch("ipfs_accelerate_py.mcp.fastapi_integration.create_mcp_server", return_value=_DummyServer())
    def test_integrate_mcp_with_fastapi_mounts_and_records_server(self, mock_create_server: MagicMock) -> None:
        app = _DummyApp()
        accelerate = object()
        model_server = _DummyModelServer(accelerate)

        integrate_mcp_with_fastapi(app, model_server)

        mock_create_server.assert_called_once_with(accelerate_instance=accelerate, mount_path="")
        app.mount.assert_called_once_with("/mcp", mock_create_server.return_value.app, name="mcp_server")
        self.assertIs(model_server.resources["mcp_server"], mock_create_server.return_value)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", side_effect=lambda *args, **kwargs: _DummyServer())
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_integrate_mcp_with_fastapi_preserves_d2_legacy_fallback_telemetry(self, mock_unified_create: MagicMock, _mock_wrapper: MagicMock) -> None:
        app = _DummyApp()
        accelerate = object()
        model_server = _DummyModelServer(accelerate)

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                integrate_mcp_with_fastapi(app, model_server)

        server = model_server.resources["mcp_server"]
        telemetry = getattr(server, "_mcp_facade_telemetry", {})

        mock_unified_create.assert_not_called()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")
        app.mount.assert_called_once_with("/mcp", server.app, name="mcp_server")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

    @patch("ipfs_accelerate_py.mcp.fastapi_integration.create_mcp_server")
    def test_integrate_mcp_with_fastapi_without_accelerate_is_noop(self, mock_create_server: MagicMock) -> None:
        app = _DummyApp()
        model_server = _DummyModelServer()

        integrate_mcp_with_fastapi(app, model_server)

        mock_create_server.assert_not_called()
        app.mount.assert_not_called()
        self.assertNotIn("mcp_server", model_server.resources)


if __name__ == "__main__":
    unittest.main()
