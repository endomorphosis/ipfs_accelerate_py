#!/usr/bin/env python3
"""UNI-203 index management import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.index_management_tools import (
    load_index,
    manage_index_configuration,
    manage_shards,
    monitor_index_status,
    orchestrate_index_lifecycle,
)
from ipfs_accelerate_py.mcp_server.tools.index_management_tools import native_index_management_tools


def test_index_management_package_exports_supported_native_functions() -> None:
    assert load_index is native_index_management_tools.load_index
    assert manage_shards is native_index_management_tools.manage_shards
    assert monitor_index_status is native_index_management_tools.monitor_index_status
    assert manage_index_configuration is native_index_management_tools.manage_index_configuration
    assert orchestrate_index_lifecycle is native_index_management_tools.orchestrate_index_lifecycle
