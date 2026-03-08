#!/usr/bin/env python3
"""Import compatibility checks for the unified rate_limiting_tools package."""

from ipfs_accelerate_py.mcp_server.tools.rate_limiting_tools import (
    check_rate_limit,
    configure_rate_limits,
    manage_rate_limits,
)
from ipfs_accelerate_py.mcp_server.tools.rate_limiting_tools.native_rate_limiting_tools_category import (
    check_rate_limit as native_check_rate_limit,
    configure_rate_limits as native_configure_rate_limits,
    manage_rate_limits as native_manage_rate_limits,
)


def test_rate_limiting_tools_package_exports_native_functions() -> None:
    assert configure_rate_limits is native_configure_rate_limits
    assert check_rate_limit is native_check_rate_limit
    assert manage_rate_limits is native_manage_rate_limits