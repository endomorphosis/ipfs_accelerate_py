#!/usr/bin/env python3
"""UNI-189 storage import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.storage_tools import (
    manage_collections,
    query_storage,
    retrieve_data,
    store_data,
)
from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


def test_storage_package_exports_source_compatible_functions() -> None:
    assert store_data is native_storage_tools.store_data
    assert retrieve_data is native_storage_tools.retrieve_data
    assert manage_collections is native_storage_tools.manage_collections
    assert query_storage is native_storage_tools.query_storage