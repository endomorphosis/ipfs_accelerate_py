#!/usr/bin/env python3
"""Import compatibility checks for the unified legacy_mcp_tools package."""

from ipfs_accelerate_py.mcp_server.tools.legacy_mcp_tools import legacy_tools_inventory
from ipfs_accelerate_py.mcp_server.tools.legacy_mcp_tools.native_legacy_mcp_tools import (
    legacy_tools_inventory as native_legacy_tools_inventory,
)


def test_legacy_mcp_tools_package_exports_native_functions() -> None:
    assert legacy_tools_inventory is native_legacy_tools_inventory