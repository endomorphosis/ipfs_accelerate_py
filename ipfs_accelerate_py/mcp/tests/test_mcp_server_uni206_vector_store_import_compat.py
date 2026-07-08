#!/usr/bin/env python3
"""UNI-206 vector store import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.vector_store_tools import (
    enhanced_vector_index,
    enhanced_vector_search,
    enhanced_vector_storage,
    vector_index,
    vector_metadata,
    vector_retrieval,
)
from ipfs_accelerate_py.mcp_server.tools.vector_store_tools import native_vector_store_tools


def test_vector_store_package_exports_supported_native_functions() -> None:
    assert vector_index is native_vector_store_tools.vector_index
    assert vector_retrieval is native_vector_store_tools.vector_retrieval
    assert vector_metadata is native_vector_store_tools.vector_metadata
    assert enhanced_vector_index is native_vector_store_tools.enhanced_vector_index
    assert enhanced_vector_search is native_vector_store_tools.enhanced_vector_search
    assert enhanced_vector_storage is native_vector_store_tools.enhanced_vector_storage
