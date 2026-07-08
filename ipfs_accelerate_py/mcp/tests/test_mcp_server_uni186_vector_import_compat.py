#!/usr/bin/env python3
"""UNI-186 vector import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.vector_tools import (
    create_vector_index,
    search_vector_index,
)
from ipfs_accelerate_py.mcp_server.tools.vector_tools import native_vector_tools


def test_vector_package_exports_source_compatible_functions() -> None:
    assert create_vector_index is native_vector_tools.create_vector_index
    assert search_vector_index is native_vector_tools.search_vector_index