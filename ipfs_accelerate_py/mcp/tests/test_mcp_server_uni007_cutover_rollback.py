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
    def test_default_startup_path_now_uses_unified_runtime(self, mock_unified_create, mock_wrapper):
        mock_unified_create.return_value = _DummyUnifiedServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "",
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            os.environ.pop("IPFS_MCP_ENABLE_UNIFIED_BRIDGE", None)
            server = create_mcp_server(name="default-unified")

        self.assertIs(server, mock_unified_create.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_not_called()

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("bridge_defaulted"))
        self.assertTrue(telemetry.get("bridge_active"))
        self.assertEqual(telemetry.get("selected_runtime"), "unified")
        self.assertEqual(telemetry.get("reason"), "unified_default")

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_cutover_dry_run_validates_unified_then_stays_legacy(self, mock_unified_create, mock_wrapper):
        mock_unified_create.return_value = _DummyUnifiedServer()
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
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
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))

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
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("reason"), "dry_run_legacy_fallback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("dry_run_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 0)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("dry_run_legacy_fallback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_cutover_dry_run_failure_records_error_and_falls_back(self, mock_unified_create, mock_wrapper):
        mock_unified_create.side_effect = RuntimeError("dry-run-failure")
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
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
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))

        status = getattr(server, "_unified_cutover_dry_run", {})
        self.assertTrue(status.get("enabled"))
        self.assertFalse(status.get("ok"))
        self.assertIn("dry-run-failure", str(status.get("error") or ""))

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertEqual(telemetry.get("reason"), "dry_run_failure_fallback")
        self.assertIn("dry-run-failure", str(telemetry.get("bridge_error") or ""))

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("bridge_failure_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("dry_run_failure_fallback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_bridge_error_fallback_records_reason_and_warning(self, mock_unified_create, mock_wrapper):
        mock_unified_create.side_effect = RuntimeError("bridge-boom")
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                server = create_mcp_server(name="bridge-error-fallback")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_called_once()
        self.assertTrue(any("bridge unavailable" in line.lower() for line in captured.output))
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertFalse(telemetry.get("bridge_active"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("reason"), "bridge_error_fallback")
        self.assertIn("bridge-boom", str(telemetry.get("bridge_error") or ""))

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("bridge_failure_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("bridge_error_fallback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_force_legacy_rollback_overrides_bridge_flag(self, mock_unified_create, mock_wrapper):
        mock_wrapper.return_value = _DummyLegacyServer()

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
                server = create_mcp_server(name="force-rollback")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_not_called()
        mock_wrapper.assert_called_once()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("deprecation_warning_emitted"))
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_force_rollback_takes_precedence_over_cutover_dry_run(self, mock_unified_create, mock_wrapper):
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "1",
                },
                clear=False,
            ):
                server = create_mcp_server(name="force-rollback-with-dry-run")

        self.assertIs(server, mock_wrapper.return_value)
        mock_unified_create.assert_not_called()
        mock_wrapper.assert_called_once()
        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))

        status = getattr(server, "_unified_cutover_dry_run", {})
        self.assertTrue(status.get("enabled"))
        self.assertFalse(status.get("ok"))
        self.assertEqual(status.get("error"), "")

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertTrue(telemetry.get("bridge_requested"))
        self.assertTrue(telemetry.get("used_legacy_wrapper"))
        self.assertTrue(telemetry.get("force_legacy_rollback"))
        self.assertTrue(telemetry.get("cutover_dry_run"))
        self.assertFalse(telemetry.get("dry_run_ok"))
        self.assertFalse(telemetry.get("bridge_active"))
        self.assertEqual(telemetry.get("bridge_error"), "")
        self.assertEqual(telemetry.get("reason"), "force_legacy_rollback")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("rollback_calls"), 1)
        self.assertEqual(counts.get("dry_run_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 0)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 1)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_legacy_rollback_warning_is_deduplicated_until_reset(self, mock_unified_create, mock_wrapper):
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as captured:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                first_server = create_mcp_server(name="rollback-warning-first")

        self.assertTrue(any("D2 opt-in only" in line for line in captured.output))
        self.assertTrue(getattr(first_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            second_server = create_mcp_server(name="rollback-warning-second")

        self.assertFalse(getattr(second_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))

        _reset_mcp_facade_telemetry()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as after_reset_logs:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                third_server = create_mcp_server(name="rollback-warning-third")

        self.assertTrue(any("D2 opt-in only" in line for line in after_reset_logs.output))
        self.assertTrue(getattr(third_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))
        self.assertEqual(get_mcp_facade_telemetry().get("facade_calls"), 1)
        self.assertEqual(get_mcp_facade_telemetry().get("warning_emissions"), 1)
        self.assertEqual((get_mcp_facade_telemetry().get("reason_counts") or {}).get("force_legacy_rollback"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_bridge_disable_flag_no_longer_bypasses_unified_default_in_d2(self, mock_unified_create, mock_wrapper):
        mock_unified_create.return_value = _DummyUnifiedServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "0",
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            server = create_mcp_server(name="bridge-disable-ignored")

        self.assertIs(server, mock_unified_create.return_value)
        mock_unified_create.assert_called_once()
        mock_wrapper.assert_not_called()

        telemetry = getattr(server, "_mcp_facade_telemetry", {})
        self.assertFalse(telemetry.get("bridge_defaulted"))
        self.assertFalse(telemetry.get("bridge_requested"))
        self.assertTrue(telemetry.get("bridge_disable_ignored"))
        self.assertTrue(telemetry.get("bridge_active"))
        self.assertEqual(telemetry.get("selected_runtime"), "unified")
        self.assertEqual(telemetry.get("deprecation_phase"), "D2_opt_in_only")
        self.assertEqual(telemetry.get("reason"), "unified_bridge")

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 1)
        self.assertEqual(counts.get("bridge_disable_ignored_calls"), 1)
        self.assertEqual(counts.get("unified_bridge_calls"), 1)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 0)
        self.assertEqual(counts.get("warning_emissions"), 0)
        self.assertEqual((counts.get("reason_counts") or {}).get("unified_bridge"), 1)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_legacy_warning_is_deduplicated_per_reason(self, mock_unified_create, mock_wrapper):
        mock_wrapper.return_value = _DummyLegacyServer()

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as first_logs:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                first_server = create_mcp_server(name="legacy-warning-by-reason-first")

        self.assertTrue(any("D2 opt-in only" in line for line in first_logs.output))
        self.assertEqual(getattr(first_server, "_mcp_facade_telemetry", {}).get("reason"), "force_legacy_rollback")
        self.assertTrue(getattr(first_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "1",
                "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
            },
            clear=False,
        ):
            second_server = create_mcp_server(name="legacy-warning-by-reason-second")

        self.assertEqual(getattr(second_server, "_mcp_facade_telemetry", {}).get("reason"), "force_legacy_rollback")
        self.assertFalse(getattr(second_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))

        mock_unified_create.side_effect = RuntimeError("bridge-reason-change")

        with self.assertLogs("ipfs_accelerate_mcp.server", level="WARNING") as third_logs:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_FORCE_LEGACY_ROLLBACK": "0",
                    "IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN": "0",
                },
                clear=False,
            ):
                third_server = create_mcp_server(name="legacy-warning-by-reason-third")

        self.assertTrue(any("D2 opt-in only" in line for line in third_logs.output))
        self.assertEqual(getattr(third_server, "_mcp_facade_telemetry", {}).get("reason"), "bridge_error_fallback")
        self.assertTrue(getattr(third_server, "_mcp_facade_telemetry", {}).get("deprecation_warning_emitted"))

        counts = get_mcp_facade_telemetry()
        self.assertEqual(counts.get("facade_calls"), 3)
        self.assertEqual(counts.get("legacy_wrapper_calls"), 3)
        self.assertEqual(counts.get("bridge_failure_calls"), 1)
        self.assertEqual(counts.get("warning_emissions"), 2)
        self.assertEqual((counts.get("reason_counts") or {}).get("force_legacy_rollback"), 2)
        self.assertEqual((counts.get("reason_counts") or {}).get("bridge_error_fallback"), 1)

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
        self.assertEqual(counts.get("warning_emissions"), 0)
        self.assertEqual((counts.get("reason_counts") or {}).get("unified_bridge"), 1)


if __name__ == "__main__":
    unittest.main()
