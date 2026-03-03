#!/usr/bin/env python3
"""UNI-105 PDF tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.pdf_tools.native_pdf_tools import (
    pdf_batch_process,
    register_native_pdf_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI105PdfTools(unittest.TestCase):
    def test_register_includes_pdf_batch_process(self) -> None:
        manager = _DummyManager()
        register_native_pdf_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("pdf_batch_process", names)

    def test_pdf_batch_process_rejects_missing_sources(self) -> None:
        async def _run() -> None:
            result = await pdf_batch_process(pdf_sources=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("pdf_sources must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_pdf_batch_process_fallback_shape(self) -> None:
        async def _run() -> None:
            result = await pdf_batch_process(pdf_sources=["a.pdf", "b.pdf"])
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("pdf_sources"), ["a.pdf", "b.pdf"])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
