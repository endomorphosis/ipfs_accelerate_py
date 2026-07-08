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

        kb_export_props = by_name["tdfol_kb_export"]["input_schema"]["properties"]
        self.assertIn("json", kb_export_props["export_format"].get("enum", []))

        cec_prove_props = by_name["cec_prove"]["input_schema"]["properties"]
        self.assertEqual(cec_prove_props["timeout"].get("minimum"), 1)

        cec_parse_props = by_name["cec_parse"]["input_schema"]["properties"]
        self.assertEqual(cec_parse_props["language"].get("minLength"), 1)
        self.assertEqual(cec_parse_props["domain"].get("default"), "general")

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
            self.assertIn(
                "'timeout_ms' must be an integer greater than or equal to 1",
                str(invalid_timeout.get("error", "")),
            )

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

    def test_tdfol_kb_tools_validate_and_normalize(self) -> None:
        async def _run() -> None:
            invalid_axiom = await logic_tools.tdfol_kb_add_axiom(formula="")
            self.assertEqual(invalid_axiom.get("success"), False)
            self.assertIn("'formula' is required", str(invalid_axiom.get("error", "")))

            invalid_export = await logic_tools.tdfol_kb_export(export_format="xml")
            self.assertEqual(invalid_export.get("success"), False)
            self.assertIn("'export_format' must be one of", str(invalid_export.get("error", "")))

            added_axiom = await logic_tools.tdfol_kb_add_axiom(formula="forall x P(x)")
            self.assertEqual(added_axiom.get("success"), True)

            added_theorem = await logic_tools.tdfol_kb_add_theorem(formula="forall x Q(x)")
            self.assertEqual(added_theorem.get("success"), True)

            queried = await logic_tools.tdfol_kb_query()
            self.assertEqual(queried.get("success"), True)
            self.assertIn("stats", queried)

            exported = await logic_tools.tdfol_kb_export(export_format="json")
            self.assertEqual(exported.get("success"), True)
            self.assertEqual(exported.get("format"), "json")

        anyio.run(_run)

    def test_cec_tools_validate_and_wrap_exceptions(self) -> None:
        async def _run() -> None:
            invalid_goal = await logic_tools.cec_prove(goal="", timeout=30)
            self.assertEqual(invalid_goal.get("success"), False)
            self.assertIn("'goal' is required", str(invalid_goal.get("error", "")))

            invalid_timeout = await logic_tools.cec_prove(goal="P(a)", timeout=0)
            self.assertEqual(invalid_timeout.get("success"), False)
            self.assertIn(
                "'timeout' must be an integer greater than or equal to 1",
                str(invalid_timeout.get("error", "")),
            )

            invalid_parse = await logic_tools.cec_parse(text="acts(agent)", language="")
            self.assertEqual(invalid_parse.get("success"), False)
            self.assertIn("'language' must be a non-empty string", str(invalid_parse.get("error", "")))

            invalid_validate = await logic_tools.cec_validate_formula(formula="")
            self.assertEqual(invalid_validate.get("success"), False)
            self.assertEqual(invalid_validate.get("valid"), False)

            analyzed = await logic_tools.cec_analyze_formula(formula="K(agent,P)")
            self.assertIn(analyzed.get("success"), [True, False])
            self.assertEqual(analyzed.get("formula"), "K(agent,P)")

            complexity = await logic_tools.cec_formula_complexity(formula="K(agent,P)")
            self.assertIn(complexity.get("success"), [True, False])
            self.assertEqual(complexity.get("formula"), "K(agent,P)")

            with patch.dict(
                logic_tools._API,
                {
                    "cec_prove": lambda **_: (_ for _ in ()).throw(RuntimeError("cec exploded")),
                },
                clear=False,
            ):
                wrapped = await logic_tools.cec_prove(goal="P(a)")
            self.assertEqual(wrapped.get("success"), False)
            self.assertIn("cec_prove failed", str(wrapped.get("error", "")))

        anyio.run(_run)

    def test_logic_tools_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                logic_tools._API,
                {
                    "tdfol_parse": _contradictory_failure,
                    "tdfol_prove": _contradictory_failure,
                    "cec_prove": _contradictory_failure,
                },
                clear=False,
            ):
                parsed = await logic_tools.tdfol_parse(text="forall x P(x)")
                proved = await logic_tools.tdfol_prove(formula="forall x P(x)")
                cec_proved = await logic_tools.cec_prove(goal="P(a)")

            self.assertEqual(parsed.get("status"), "error")
            self.assertEqual(parsed.get("success"), False)
            self.assertEqual(parsed.get("error"), "delegate failed")

            self.assertEqual(proved.get("status"), "error")
            self.assertEqual(proved.get("success"), False)
            self.assertEqual(proved.get("error"), "delegate failed")

            self.assertEqual(cec_proved.get("status"), "error")
            self.assertEqual(cec_proved.get("success"), False)
            self.assertEqual(cec_proved.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
