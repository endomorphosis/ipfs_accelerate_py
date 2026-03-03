#!/usr/bin/env python3
"""UNI-103 dataset_tools logic-conversion parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.dataset_tools.native_dataset_tools import (
    legal_text_to_deontic,
    register_native_dataset_tools,
    text_to_fol,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI103DatasetLogicTools(unittest.TestCase):
    def test_register_includes_logic_conversion_tools(self) -> None:
        manager = _DummyManager()
        register_native_dataset_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("text_to_fol", names)
        self.assertIn("legal_text_to_deontic", names)

    def test_text_to_fol_fallback_rejects_empty_input(self) -> None:
        async def _run() -> None:
            result = await text_to_fol(text_input="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("text_input must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_text_to_fol_fallback_success_shape(self) -> None:
        async def _run() -> None:
            result = await text_to_fol(
                text_input="All humans are mortal",
                domain_predicates=["Human", "Mortal"],
                confidence_threshold=0.8,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertIn("fol", result)
            self.assertEqual(result.get("domain_predicates"), ["Human", "Mortal"])

        anyio.run(_run)

    def test_legal_text_to_deontic_fallback_success_shape(self) -> None:
        async def _run() -> None:
            result = await legal_text_to_deontic(
                text_input="Drivers must stop at red lights",
                jurisdiction="us",
                document_type="regulation",
            )
            self.assertEqual(result.get("status"), "success")
            self.assertIn("deontic", result)
            self.assertEqual(result.get("document_type"), "regulation")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
