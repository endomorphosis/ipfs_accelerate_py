#!/usr/bin/env python3
"""UNI-160 deterministic PDF tools parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.pdf_tools import native_pdf_tools as pdf_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI160PdfTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        pdf_tools.register_native_pdf_tools(manager)

        by_name = {call["name"]: call for call in manager.calls}

        query_props = by_name["pdf_query_corpus"]["input_schema"]["properties"]
        self.assertEqual(query_props["max_documents"].get("minimum"), 1)
        self.assertEqual(query_props["confidence_threshold"].get("minimum"), 0.0)
        self.assertEqual(query_props["confidence_threshold"].get("maximum"), 1.0)

        batch_props = by_name["pdf_batch_process"]["input_schema"]["properties"]
        self.assertEqual(batch_props["pdf_sources"].get("minItems"), 1)
        self.assertEqual(batch_props["batch_size"].get("minimum"), 1)
        self.assertEqual(batch_props["parallel_workers"].get("minimum"), 1)

    def test_pdf_query_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid = await pdf_tools.pdf_query_corpus("", max_documents=1)
            self.assertEqual(invalid.get("status"), "error")
            self.assertIn("query must be a non-empty string", str(invalid.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_query_corpus": lambda **_: (_ for _ in ()).throw(RuntimeError("query backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_query_corpus("find docs")
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_query_corpus failed", str(wrapped.get("error", "")))

        anyio.run(_run)

    def test_pdf_extract_entities_validation(self) -> None:
        async def _run() -> None:
            invalid_source = await pdf_tools.pdf_extract_entities(pdf_source="")
            self.assertEqual(invalid_source.get("status"), "error")
            self.assertIn("pdf_source must be a non-empty string", str(invalid_source.get("error", "")))

            invalid_entities = await pdf_tools.pdf_extract_entities(
                pdf_source="file.pdf",
                entity_types=["ORG", ""],
            )
            self.assertEqual(invalid_entities.get("status"), "error")
            self.assertIn("entity_types must be a list of non-empty strings", str(invalid_entities.get("error", "")))

            invalid_patterns = await pdf_tools.pdf_extract_entities(
                pdf_source="file.pdf",
                custom_patterns=["bad"],
            )
            self.assertEqual(invalid_patterns.get("status"), "error")
            self.assertIn("custom_patterns must be an object", str(invalid_patterns.get("error", "")))

        anyio.run(_run)

    def test_pdf_batch_process_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_sources = await pdf_tools.pdf_batch_process(pdf_sources=[""])
            self.assertEqual(invalid_sources.get("status"), "error")
            self.assertIn("pdf_sources entries must be non-empty strings or objects", str(invalid_sources.get("error", "")))

            invalid_callback = await pdf_tools.pdf_batch_process(
                pdf_sources=["a.pdf"],
                progress_callback="",
            )
            self.assertEqual(invalid_callback.get("status"), "error")
            self.assertIn("progress_callback must be a non-empty string", str(invalid_callback.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_batch_process": lambda **_: (_ for _ in ()).throw(RuntimeError("batch backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_batch_process(pdf_sources=["a.pdf"])
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_batch_process failed", str(wrapped.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
