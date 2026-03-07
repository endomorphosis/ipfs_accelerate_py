#!/usr/bin/env python3
"""UNI-193 file converter import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.file_converter_tools import (
    batch_convert_tool,
    convert_file_tool,
    download_url_tool,
    extract_archive_tool,
    extract_knowledge_graph_tool,
    file_info_tool,
    generate_embeddings_tool,
    generate_summary_tool,
)
from ipfs_accelerate_py.mcp_server.tools.file_converter_tools import native_file_converter_tools


def test_file_converter_package_exports_supported_source_compatible_functions() -> None:
    assert batch_convert_tool is native_file_converter_tools.batch_convert_tool
    assert convert_file_tool is native_file_converter_tools.convert_file_tool
    assert extract_knowledge_graph_tool is native_file_converter_tools.extract_knowledge_graph_tool
    assert generate_summary_tool is native_file_converter_tools.generate_summary_tool
    assert generate_embeddings_tool is native_file_converter_tools.generate_embeddings_tool
    assert extract_archive_tool is native_file_converter_tools.extract_archive_tool
    assert file_info_tool is native_file_converter_tools.file_info_tool
    assert download_url_tool is native_file_converter_tools.download_url_tool
