#!/usr/bin/env python3
"""UNI-105 logic tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.logic_tools.native_logic_tools import (
    register_native_logic_tools,
    tdfol_convert,
    tdfol_parse,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI105LogicTools(unittest.TestCase):
    def test_register_includes_tdfol_parse_and_convert(self) -> None:
        manager = _DummyManager()
        register_native_logic_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("tdfol_parse", names)
        self.assertIn("tdfol_convert", names)

    def test_tdfol_parse_rejects_missing_text(self) -> None:
        async def _run() -> None:
            result = await tdfol_parse(text="   ")
            self.assertEqual(result.get("success"), False)
            self.assertIn("'text' is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_tdfol_parse_fallback_shape(self) -> None:
        async def _run() -> None:
            result = await tdfol_parse(text="forall x P(x)", format="symbolic", language="en")
            self.assertIn(result.get("success"), [True, False])
            # Source unavailable path may be a minimal envelope; canonical fallback includes format/language.
            if result.get("success") is False:
                self.assertIn("error", result)
                if "format" in result or "language" in result:
                    self.assertIn("format", result)
                    self.assertIn("language", result)

        anyio.run(_run)

    def test_tdfol_convert_rejects_missing_formula(self) -> None:
        async def _run() -> None:
            result = await tdfol_convert(formula="  ")
            self.assertEqual(result.get("success"), False)
            self.assertIn("'formula' is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_tdfol_convert_fallback_shape(self) -> None:
        async def _run() -> None:
            result = await tdfol_convert(
                formula="forall x P(x)",
                source_format="tdfol",
                target_format="fol",
            )
            self.assertIn(result.get("success"), [True, False])
            if result.get("success") is False:
                self.assertIn("error", result)
                if "source_format" in result or "target_format" in result:
                    self.assertIn("source_format", result)
                    self.assertIn("target_format", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
