#!/usr/bin/env python3
"""Import compatibility checks for the unified lizardpersons_function_tools package."""

from ipfs_accelerate_py.mcp_server.tools.lizardpersons_function_tools import get_current_time
from ipfs_accelerate_py.mcp_server.tools.lizardpersons_function_tools.native_lizardpersons_function_tools import (
    get_current_time as native_get_current_time,
)


def test_lizardpersons_function_tools_package_exports_native_functions() -> None:
    assert get_current_time is native_get_current_time