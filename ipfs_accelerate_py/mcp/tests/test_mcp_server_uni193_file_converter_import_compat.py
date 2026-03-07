#!/usr/bin/env python3
"""UNI-193 file converter import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.file_converter_tools import (
    convert_file_tool,
    download_url_tool,
    file_info_tool,
)
from ipfs_accelerate_py.mcp_server.tools.file_converter_tools import native_file_converter_tools


def test_file_converter_package_exports_supported_source_compatible_functions() -> None:
    assert convert_file_tool is native_file_converter_tools.convert_file_tool
    assert file_info_tool is native_file_converter_tools.file_info_tool
    assert download_url_tool is native_file_converter_tools.download_url_tool
