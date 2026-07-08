#!/usr/bin/env python3
"""UNI-231 bespoke import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.bespoke_tools import (
    cache_stats,
    create_vector_store,
    delete_index,
    execute_workflow,
    list_indices,
    system_health,
    system_status,
)
from ipfs_accelerate_py.mcp_server.tools.bespoke_tools import native_bespoke_tools


def test_bespoke_package_exports_supported_native_functions() -> None:
    assert system_health is native_bespoke_tools.system_health
    assert system_status is native_bespoke_tools.system_status
    assert cache_stats is native_bespoke_tools.cache_stats
    assert execute_workflow is native_bespoke_tools.execute_workflow
    assert list_indices is native_bespoke_tools.list_indices
    assert delete_index is native_bespoke_tools.delete_index
    assert create_vector_store is native_bespoke_tools.create_vector_store