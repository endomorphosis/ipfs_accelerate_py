#!/usr/bin/env python3
"""UNI-205 IPFS cluster import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.ipfs_cluster_tools import (
    manage_ipfs_cluster,
    manage_ipfs_content,
)
from ipfs_accelerate_py.mcp_server.tools.ipfs_cluster_tools import native_ipfs_cluster_tools


def test_ipfs_cluster_package_exports_supported_native_functions() -> None:
    assert manage_ipfs_cluster is native_ipfs_cluster_tools.manage_ipfs_cluster
    assert manage_ipfs_content is native_ipfs_cluster_tools.manage_ipfs_content
