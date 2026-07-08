#!/usr/bin/env python3
"""UNI-146 lizardperson_argparse_programs parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.lizardperson_argparse_programs import (
    native_lizardperson_argparse_programs as argparse_mod,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI146LizardpersonArgparsePrograms(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        argparse_mod.register_native_lizardperson_argparse_programs(manager)
        by_name = {call["name"]: call for call in manager.calls}

        schema = by_name["municipal_bluebook_validator_info"]["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], [])

        invoke_schema = by_name["municipal_bluebook_validator_invoke"]["input_schema"]
        self.assertEqual(invoke_schema["type"], "object")
        self.assertEqual(invoke_schema["properties"]["allow_execution"]["default"], False)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            result = await argparse_mod.municipal_bluebook_validator_info()
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(
                result.get("entrypoint"),
                "municipal_bluebook_citation_validator.main",
            )
            if result.get("status") == "success":
                self.assertIn("callable", result)
                self.assertIn("fallback", result)

        anyio.run(_run)

    def test_error_envelope_shape_on_api_failure(self) -> None:
        async def _run() -> None:
            class _ExplodingAPI:
                def get(self, *_args, **_kwargs):
                    raise RuntimeError("boom")

            with patch.object(argparse_mod, "_API", new=_ExplodingAPI()):
                result = await argparse_mod.municipal_bluebook_validator_info()

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(
                result.get("entrypoint"),
                "municipal_bluebook_citation_validator.main",
            )
            self.assertIn("boom", str(result.get("error", "")))

        anyio.run(_run)

    def test_invoke_tool_dry_run_and_validation(self) -> None:
        async def _run() -> None:
            invalid = await argparse_mod.municipal_bluebook_validator_invoke(argv=["", "ok"])
            self.assertEqual(invalid.get("status"), "error")
            self.assertIn("non-empty strings", str(invalid.get("error", "")))

            dry_run = await argparse_mod.municipal_bluebook_validator_invoke(
                argv=["--citation-dir", "./citations"],
                allow_execution=False,
            )
            self.assertEqual(dry_run.get("status"), "success")
            self.assertIs(dry_run.get("invoked"), False)
            self.assertIs(dry_run.get("dry_run"), True)

        anyio.run(_run)

    def test_invoke_tool_executes_entrypoint_when_enabled(self) -> None:
        async def _run() -> None:
            class _Entry:
                def __init__(self) -> None:
                    self.calls: list[list[str]] = []

                def __call__(self, argv):
                    self.calls.append(list(argv))
                    return 3

            entry = _Entry()
            with patch.object(argparse_mod, "_API", new={"validator_main": entry}):
                result = await argparse_mod.municipal_bluebook_validator_invoke(
                    argv=["--sample-size", "5"],
                    allow_execution=True,
                )

            self.assertEqual(result.get("status"), "success")
            self.assertIs(result.get("invoked"), True)
            self.assertIs(result.get("dry_run"), False)
            self.assertEqual(result.get("exit_code"), 3)
            self.assertEqual(entry.calls, [["--sample-size", "5"]])

        anyio.run(_run)

    def test_info_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch.object(argparse_mod, "_API", new={}):
                result = await argparse_mod.municipal_bluebook_validator_info()

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("entrypoint"), "municipal_bluebook_citation_validator.main")
            self.assertEqual(result.get("callable"), False)

        anyio.run(_run)

    def test_invoke_error_envelope_has_success_false(self) -> None:
        async def _run() -> None:
            with patch.object(argparse_mod, "_API", new={}):
                result = await argparse_mod.municipal_bluebook_validator_invoke(argv=["--sample-size", "5"])

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("success"), False)
            self.assertIn("validator entrypoint unavailable", str(result.get("error", "")))

        anyio.run(_run)

    def test_argparse_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _run() -> None:
            class _Entry:
                def __call__(self, _argv):
                    return {"status": "success", "success": False, "error": "delegate failure"}

            with patch.object(argparse_mod, "_API", new={"validator_main": _Entry()}):
                info_result = await argparse_mod.municipal_bluebook_validator_info()
                invoke_result = await argparse_mod.municipal_bluebook_validator_invoke(
                    argv=["--sample-size", "5"],
                    allow_execution=True,
                )

            self.assertEqual(info_result.get("status"), "success")
            self.assertEqual(invoke_result.get("status"), "error")
            self.assertEqual(invoke_result.get("error"), "delegate failure")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
