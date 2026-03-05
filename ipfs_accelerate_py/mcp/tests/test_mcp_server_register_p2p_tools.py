#!/usr/bin/env python3
"""Tests for canonical P2P tool registration helpers."""

import types
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp_server import register_p2p_tools


class TestRegisterP2PToolsFacade(unittest.TestCase):
    def test_discover_reports_available_and_import_error(self) -> None:
        available_module = types.SimpleNamespace(register_native_p2p_tools=lambda _m: None)

        def _import(path: str):
            if path == "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools":
                return available_module
            raise ImportError("missing")

        with patch("importlib.import_module", side_effect=_import):
            records = register_p2p_tools.discover_p2p_tool_modules()

        self.assertEqual(len(records), len(register_p2p_tools.P2P_TOOL_MODULES))
        self.assertEqual(records[0]["status"], "available")
        self.assertEqual(records[1]["status"], "import_error")

    def test_register_invokes_available_registrars(self) -> None:
        calls = []

        def _reg_a(manager):
            calls.append(("a", manager))

        def _reg_b(manager):
            calls.append(("b", manager))

        modules = {
            "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools": types.SimpleNamespace(
                register_native_p2p_tools=_reg_a
            ),
            "ipfs_accelerate_py.mcp_server.tools.p2p_tools.native_p2p_tools": types.SimpleNamespace(
                register_native_p2p_tools_category=_reg_b
            ),
        }

        def _import(path: str):
            if path in modules:
                return modules[path]
            raise ImportError("missing")

        manager = object()
        with patch("importlib.import_module", side_effect=_import):
            summary = register_p2p_tools.register_p2p_category_loaders(manager)

        self.assertEqual(summary["loaded"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("a", manager))
        self.assertEqual(calls[1], ("b", manager))

    def test_register_uses_single_resolver_path(self) -> None:
        calls = []

        def _reg(manager):
            calls.append(manager)

        with patch(
            "ipfs_accelerate_py.mcp_server.register_p2p_tools._resolve_p2p_registrars",
            return_value=[
                {
                    "module": "m",
                    "registrar": "r",
                    "status": "available",
                    "callable": _reg,
                }
            ],
        ) as mock_resolve:
            summary = register_p2p_tools.register_p2p_category_loaders(object())

        mock_resolve.assert_called_once_with()
        self.assertEqual(summary["loaded"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["total"], len(register_p2p_tools.P2P_TOOL_MODULES))
        self.assertEqual(len(calls), 1)

    def test_register_continues_when_registrar_raises(self) -> None:
        calls = []

        def _reg_ok(manager):
            calls.append(("ok", manager))

        def _reg_fail(_manager):
            raise RuntimeError("boom")

        manager = object()
        with patch(
            "ipfs_accelerate_py.mcp_server.register_p2p_tools._resolve_p2p_registrars",
            return_value=[
                {"module": "m1", "registrar": "r1", "status": "available", "callable": _reg_fail},
                {"module": "m2", "registrar": "r2", "status": "available", "callable": _reg_ok},
            ],
        ):
            summary = register_p2p_tools.register_p2p_category_loaders(manager)

        self.assertEqual(summary["loaded"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total"], len(register_p2p_tools.P2P_TOOL_MODULES))
        self.assertEqual(calls, [("ok", manager)])


if __name__ == "__main__":
    unittest.main()
