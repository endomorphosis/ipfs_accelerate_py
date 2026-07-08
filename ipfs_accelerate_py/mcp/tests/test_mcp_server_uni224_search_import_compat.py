#!/usr/bin/env python3
"""UNI-224 search import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.search_tools import (
    faceted_search,
    semantic_search,
    similarity_search,
)
from ipfs_accelerate_py.mcp_server.tools.search_tools import native_search_tools


def test_search_package_exports_supported_native_functions() -> None:
    assert semantic_search is native_search_tools.semantic_search
    assert similarity_search is native_search_tools.similarity_search
    assert faceted_search is native_search_tools.faceted_search