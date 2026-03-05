#!/usr/bin/env python3
"""UNI-161 deterministic logic tools parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.logic_tools import native_logic_tools as logic_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI161LogicTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        logic_tools.register_native_logic_tools(manager)

        by_name = {call["name"]: call for call in manager.calls}

        parse_props = by_name["tdfol_parse"]["input_schema"]["properties"]
        self.assertEqual(parse_props["format"].get("minLength"), 1)
        self.assertEqual(parse_props["language"].get("minLength"), 1)

        prove_props = by_name["tdfol_prove"]["input_schema"]["properties"]
        self.assertEqual(prove_props["timeout_ms"].get("minimum"), 1)
        self.assertEqual(prove_props["max_depth"].get("minimum"), 1)

    def test_tdfol_parse_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_format = await logic_tools.tdfol_parse(text="forall x P(x)", format="")
            self.assertEqual(invalid_format.get("success"), False)
            self.assertIn("'format' must be a non-empty string", str(invalid_format.get("error", "")))

            with patch.dict(
                logic_tools._API,
                {
                    "tdfol_parse": lambda **_: (_ for _ in ()).throw(RuntimeError("parse exploded")),
                },
                clear=False,
            ):
                wrapped = await logic_tools.tdfol_parse(text="forall x P(x)")
            self.assertEqual(wrapped.get("success"), False)
            self.assertIn("tdfol_parse failed", str(wrapped.get("error", "")))

        anyio.run(_run)

    def test_tdfol_convert_validation(self) -> None:
        async def _run() -> None:
            invalid_source = await logic_tools.tdfol_convert(
                formula="forall x P(x)",
                source_format="",
            )
            self.assertEqual(invalid_source.get("success"), False)
            self.assertIn("'source_format' must be a non-empty string", str(invalid_source.get("error", "")))

            invalid_target = await logic_tools.tdfol_convert(
                formula="forall x P(x)",
                target_format="",
            )
            self.assertEqual(invalid_target.get("success"), False)
            self.assertIn("'target_format' must be a non-empty string", str(invalid_target.get("error", "")))

        anyio.run(_run)

    def test_tdfol_prove_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_axioms = await logic_tools.tdfol_prove(
                formula="forall x P(x)",
                axioms=["A", ""],
            )
            self.assertEqual(invalid_axioms.get("success"), False)
            self.assertIn("'axioms' must be a list of non-empty strings", str(invalid_axioms.get("error", "")))

            invalid_timeout = await logic_tools.tdfol_prove(
                formula="forall x P(x)",
                timeout_ms=0,
            )
            self.assertEqual(invalid_timeout.get("success"), False)
            self.assertIn("'timeout_ms' must be an integer greater than or equal to 1", str(invalid_timeout.get("error", "")))

            with patch.dict(
                logic_tools._API,
                {
                    "tdfol_prove": lambda **_: (_ for _ in ()).throw(RuntimeError("prove exploded")),
                },
                clear=False,
            ):
                wrapped = await logic_tools.tdfol_prove(formula="forall x P(x)")
            self.assertEqual(wrapped.get("success"), False)
            self.assertIn("tdfol_prove failed", str(wrapped.get("error", "")))
            self.assertEqual(wrapped.get("formula"), "forall x P(x)")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
