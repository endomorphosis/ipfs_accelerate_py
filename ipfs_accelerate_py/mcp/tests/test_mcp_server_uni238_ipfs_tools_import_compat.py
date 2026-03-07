#!/usr/bin/env python3
"""UNI-238 ipfs-tools package import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.ipfs_tools import get_from_ipfs, pin_to_ipfs
from ipfs_accelerate_py.mcp_server.tools.ipfs_tools import native_ipfs_tools_category


def test_ipfs_tools_package_exports_source_compatible_functions() -> None:
    assert pin_to_ipfs is native_ipfs_tools_category.pin_to_ipfs
    assert get_from_ipfs is native_ipfs_tools_category.get_from_ipfs