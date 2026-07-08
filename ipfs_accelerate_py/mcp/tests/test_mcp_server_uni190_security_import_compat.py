#!/usr/bin/env python3
"""UNI-190 security import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.security_tools import check_access_permission
from ipfs_accelerate_py.mcp_server.tools.security_tools import native_security_tools


def test_security_package_exports_source_compatible_function() -> None:
    assert check_access_permission is native_security_tools.check_access_permission