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

        relationship_props = by_name["pdf_analyze_relationships"]["input_schema"]["properties"]
        self.assertEqual(relationship_props["document_id"].get("minLength"), 1)
        self.assertEqual(relationship_props["min_confidence"].get("maximum"), 1.0)

        cross_props = by_name["pdf_cross_document_analysis"]["input_schema"]["properties"]
        self.assertEqual(cross_props["document_ids"].get("minItems"), 1)
        self.assertEqual(cross_props["similarity_threshold"].get("maximum"), 1.0)

        optimize_props = by_name["pdf_optimize_for_llm"]["input_schema"]["properties"]
        self.assertEqual(optimize_props["max_chunk_size"].get("minimum"), 1)
        self.assertEqual(optimize_props["overlap_size"].get("minimum"), 0)

        graph_props = by_name["pdf_query_knowledge_graph"]["input_schema"]["properties"]
        self.assertIn("sparql", graph_props["query_type"].get("enum", []))
        self.assertEqual(graph_props["max_results"].get("minimum"), 1)

        batch_props = by_name["pdf_batch_process"]["input_schema"]["properties"]
        self.assertEqual(batch_props["pdf_sources"].get("minItems"), 1)
        self.assertEqual(batch_props["batch_size"].get("minimum"), 1)
        self.assertEqual(batch_props["parallel_workers"].get("minimum"), 1)

    def test_pdf_analyze_relationships_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_document = await pdf_tools.pdf_analyze_relationships("")
            self.assertEqual(invalid_document.get("status"), "error")
            self.assertIn("document_id must be a non-empty string", str(invalid_document.get("error", "")))

            invalid_types = await pdf_tools.pdf_analyze_relationships(
                "doc-1",
                relationship_types=["SIGNED_BY", ""],
            )
            self.assertEqual(invalid_types.get("status"), "error")
            self.assertIn("relationship_types must be a list of non-empty strings", str(invalid_types.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_analyze_relationships": lambda **_: (_ for _ in ()).throw(RuntimeError("relationships backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_analyze_relationships("doc-1")
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_analyze_relationships failed", str(wrapped.get("error", "")))

        anyio.run(_run)

    def test_pdf_cross_document_analysis_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_docs = await pdf_tools.pdf_cross_document_analysis(document_ids=[])
            self.assertEqual(invalid_docs.get("status"), "error")
            self.assertIn("document_ids must be provided as a non-empty array", str(invalid_docs.get("error", "")))

            invalid_types = await pdf_tools.pdf_cross_document_analysis(
                document_ids=["doc-1", "doc-2"],
                analysis_types=["entities", ""],
            )
            self.assertEqual(invalid_types.get("status"), "error")
            self.assertIn("analysis_types must be a list of non-empty strings", str(invalid_types.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_cross_document_analysis": lambda **_: (_ for _ in ()).throw(RuntimeError("cross-document backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_cross_document_analysis(document_ids=["doc-1", "doc-2"])
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_cross_document_analysis failed", str(wrapped.get("error", "")))

        anyio.run(_run)

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

    def test_pdf_query_knowledge_graph_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_graph = await pdf_tools.pdf_query_knowledge_graph(graph_id="", query="MATCH (n) RETURN n")
            self.assertEqual(invalid_graph.get("status"), "error")
            self.assertIn("graph_id must be a non-empty string", str(invalid_graph.get("error", "")))

            invalid_type = await pdf_tools.pdf_query_knowledge_graph(
                graph_id="graph-1",
                query="MATCH (n) RETURN n",
                query_type="sql",
            )
            self.assertEqual(invalid_type.get("status"), "error")
            self.assertIn("query_type must be one of", str(invalid_type.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_query_knowledge_graph": lambda **_: (_ for _ in ()).throw(RuntimeError("graph backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_query_knowledge_graph(
                    graph_id="graph-1",
                    query="MATCH (n) RETURN n",
                )
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_query_knowledge_graph failed", str(wrapped.get("error", "")))

        anyio.run(_run)

    def test_pdf_optimize_for_llm_validation_and_exception_envelope(self) -> None:
        async def _run() -> None:
            invalid_source = await pdf_tools.pdf_optimize_for_llm(pdf_source=None)
            self.assertEqual(invalid_source.get("status"), "error")
            self.assertIn("pdf_source must be provided", str(invalid_source.get("error", "")))

            invalid_overlap = await pdf_tools.pdf_optimize_for_llm(
                pdf_source="file.pdf",
                max_chunk_size=100,
                overlap_size=200,
            )
            self.assertEqual(invalid_overlap.get("status"), "error")
            self.assertIn("overlap_size must be less than or equal to max_chunk_size", str(invalid_overlap.get("error", "")))

            with patch.dict(
                pdf_tools._API,
                {
                    "pdf_optimize_for_llm": lambda **_: (_ for _ in ()).throw(RuntimeError("optimize backend exploded")),
                },
                clear=False,
            ):
                wrapped = await pdf_tools.pdf_optimize_for_llm(pdf_source="file.pdf")
            self.assertEqual(wrapped.get("status"), "error")
            self.assertIn("pdf_optimize_for_llm failed", str(wrapped.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
