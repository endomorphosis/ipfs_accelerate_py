#!/usr/bin/env python3
"""UNI-157 deterministic parity tests for native dataset tools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.dataset_tools import native_dataset_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI157DatasetTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_dataset_tools.register_native_dataset_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        load_props = schemas["load_dataset"]["properties"]
        self.assertEqual(load_props["source"].get("minLength"), 1)

        text_to_fol_props = schemas["text_to_fol"]["properties"]
        self.assertEqual(text_to_fol_props["confidence_threshold"].get("minimum"), 0)
        self.assertEqual(text_to_fol_props["confidence_threshold"].get("maximum"), 1)

    def test_load_and_save_validate_and_wrap_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("dataset boom")

        async def _run() -> None:
            invalid_source = await native_dataset_tools.load_dataset(source="   ")
            self.assertEqual(invalid_source.get("status"), "error")
            self.assertIn("source must be a non-empty string", str(invalid_source.get("error", "")))

            invalid_dataset = await native_dataset_tools.save_dataset(dataset_data="   ")
            self.assertEqual(invalid_dataset.get("status"), "error")
            self.assertIn("dataset_data must be non-empty", str(invalid_dataset.get("error", "")))

            with patch.dict(native_dataset_tools._API, {"load_dataset": _boom}, clear=False):
                result = await native_dataset_tools.load_dataset(source="dataset://id")
                self.assertEqual(result.get("status"), "error")
                self.assertIn("load_dataset failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_text_to_fol_validates_threshold_and_predicates(self) -> None:
        async def _run() -> None:
            invalid_predicates = await native_dataset_tools.text_to_fol(
                text_input="All humans are mortal",
                domain_predicates=["Human", ""],
            )
            self.assertEqual(invalid_predicates.get("status"), "error")
            self.assertIn("domain_predicates must be an array of non-empty strings", str(invalid_predicates.get("error", "")))

            invalid_threshold = await native_dataset_tools.text_to_fol(
                text_input="All humans are mortal",
                confidence_threshold=1.5,
            )
            self.assertEqual(invalid_threshold.get("status"), "error")
            self.assertIn("confidence_threshold must be between 0 and 1", str(invalid_threshold.get("error", "")))

        anyio.run(_run)

    def test_legal_text_to_deontic_validates_flags_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("deontic boom")

        async def _run() -> None:
            invalid_flag = await native_dataset_tools.legal_text_to_deontic(
                text_input="Drivers must stop at red lights",
                extract_obligations="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_flag.get("status"), "error")
            self.assertIn("extract_obligations must be a boolean", str(invalid_flag.get("error", "")))

            with patch.dict(native_dataset_tools._API, {"legal_text_to_deontic": _boom}, clear=False):
                result = await native_dataset_tools.legal_text_to_deontic(
                    text_input="Drivers must stop at red lights",
                )
                self.assertEqual(result.get("status"), "error")
                self.assertIn("legal_text_to_deontic failed", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
