#!/usr/bin/env python3
"""Import compatibility checks for the unified idl package."""

from ipfs_accelerate_py.mcp_server.tools.idl import load_idl_tools, register_native_idl_tools
from ipfs_accelerate_py.mcp_server.tools.idl import native_idl_tools


def test_idl_package_exports_native_functions() -> None:
    assert load_idl_tools is native_idl_tools.load_idl_tools
    assert register_native_idl_tools is native_idl_tools.register_native_idl_tools
