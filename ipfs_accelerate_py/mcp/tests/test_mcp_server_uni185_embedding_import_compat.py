#!/usr/bin/env python3
"""UNI-185 embedding import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.embedding_tools import (
    create_embeddings,
    index_dataset,
    search_embeddings,
)
from ipfs_accelerate_py.mcp_server.tools.embedding_tools import native_embedding_tools


def test_embedding_package_exports_source_compatible_aliases() -> None:
    assert create_embeddings is native_embedding_tools.generate_embedding
    assert index_dataset is native_embedding_tools.generate_embeddings
    assert search_embeddings is native_embedding_tools.generate_embeddings_from_file


def test_embedding_native_module_exposes_source_compatible_aliases() -> None:
    assert native_embedding_tools.create_embeddings is native_embedding_tools.generate_embedding
    assert native_embedding_tools.index_dataset is native_embedding_tools.generate_embeddings
    assert search_embeddings is native_embedding_tools.search_embeddings