#!/usr/bin/env python3
"""UNI-223 PDF import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.pdf_tools import (
    pdf_analyze_relationships,
    pdf_batch_process,
    pdf_cross_document_analysis,
    pdf_extract_entities,
    pdf_ingest_to_graphrag,
    pdf_optimize_for_llm,
    pdf_query_corpus,
    pdf_query_knowledge_graph,
)
from ipfs_accelerate_py.mcp_server.tools.pdf_tools import native_pdf_tools


def test_pdf_package_exports_supported_native_functions() -> None:
    assert pdf_ingest_to_graphrag is native_pdf_tools.pdf_ingest_to_graphrag
    assert pdf_query_corpus is native_pdf_tools.pdf_query_corpus
    assert pdf_query_knowledge_graph is native_pdf_tools.pdf_query_knowledge_graph
    assert pdf_analyze_relationships is native_pdf_tools.pdf_analyze_relationships
    assert pdf_batch_process is native_pdf_tools.pdf_batch_process
    assert pdf_extract_entities is native_pdf_tools.pdf_extract_entities
    assert pdf_optimize_for_llm is native_pdf_tools.pdf_optimize_for_llm
    assert pdf_cross_document_analysis is native_pdf_tools.pdf_cross_document_analysis