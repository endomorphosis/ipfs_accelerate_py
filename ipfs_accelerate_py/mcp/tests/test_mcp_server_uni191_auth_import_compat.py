#!/usr/bin/env python3
"""UNI-191 auth import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.auth_tools import (
    authenticate_user,
    get_user_info,
    validate_token,
)
from ipfs_accelerate_py.mcp_server.tools.auth_tools import native_auth_tools


def test_auth_package_exports_source_compatible_functions() -> None:
    assert authenticate_user is native_auth_tools.authenticate_user
    assert validate_token is native_auth_tools.validate_token
    assert get_user_info is native_auth_tools.get_user_info