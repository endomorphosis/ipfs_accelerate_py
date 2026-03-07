#!/usr/bin/env python3
"""UNI-192 email import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.email_tools import (
    email_analyze_export,
    email_list_folders,
    email_parse_eml,
    email_search_export,
    email_test_connection,
)
from ipfs_accelerate_py.mcp_server.tools.email_tools import native_email_tools


def test_email_package_exports_supported_source_compatible_functions() -> None:
    assert email_test_connection is native_email_tools.email_test_connection
    assert email_list_folders is native_email_tools.email_list_folders
    assert email_analyze_export is native_email_tools.email_analyze_export
    assert email_search_export is native_email_tools.email_search_export
    assert email_parse_eml is native_email_tools.email_parse_eml