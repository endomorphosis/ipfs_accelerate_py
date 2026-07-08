#!/usr/bin/env python3
"""UNI-235 sparse embedding import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools import (
    generate_sparse_embedding,
    index_sparse_collection,
    manage_sparse_models,
    sparse_search,
)
from ipfs_accelerate_py.mcp_server.tools.sparse_embedding_tools import native_sparse_embedding_tools


def test_sparse_embedding_package_exports_source_compatible_functions() -> None:
    assert generate_sparse_embedding is native_sparse_embedding_tools.generate_sparse_embedding
    assert index_sparse_collection is native_sparse_embedding_tools.index_sparse_collection
    assert sparse_search is native_sparse_embedding_tools.sparse_search
    assert manage_sparse_models is native_sparse_embedding_tools.manage_sparse_models