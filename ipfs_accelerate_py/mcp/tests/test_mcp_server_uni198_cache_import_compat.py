#!/usr/bin/env python3
"""UNI-198 cache import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.cache_tools import (
    cache_clear,
    cache_delete,
    cache_embeddings,
    cache_get,
    cache_set,
    cache_stats,
    get_cache_stats,
    get_cached_embeddings,
    manage_cache,
    monitor_cache,
    optimize_cache,
)
from ipfs_accelerate_py.mcp_server.tools.cache_tools import native_cache_tools


def test_cache_package_exports_supported_native_functions() -> None:
    assert cache_get is native_cache_tools.cache_get
    assert cache_set is native_cache_tools.cache_set
    assert cache_delete is native_cache_tools.cache_delete
    assert cache_clear is native_cache_tools.cache_clear
    assert cache_stats is native_cache_tools.cache_stats
    assert manage_cache is native_cache_tools.manage_cache
    assert optimize_cache is native_cache_tools.optimize_cache
    assert cache_embeddings is native_cache_tools.cache_embeddings
    assert get_cached_embeddings is native_cache_tools.get_cached_embeddings
    assert get_cache_stats is native_cache_tools.get_cache_stats
    assert monitor_cache is native_cache_tools.monitor_cache
