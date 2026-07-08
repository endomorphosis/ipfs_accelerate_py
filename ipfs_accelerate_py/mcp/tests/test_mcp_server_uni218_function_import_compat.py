#!/usr/bin/env python3
"""UNI-218 function import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.functions import execute_python_snippet
from ipfs_accelerate_py.mcp_server.tools.functions import native_function_tools


def test_function_package_exports_supported_native_functions() -> None:
    assert execute_python_snippet is native_function_tools.execute_python_snippet