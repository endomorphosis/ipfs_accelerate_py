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


if __name__ == "__main__":
    unittest.main()
