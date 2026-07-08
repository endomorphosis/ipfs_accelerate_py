#!/usr/bin/env python3
"""UNI-196 data processing import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.data_processing_tools import (
    chunk_text,
    convert_format,
    transform_data,
    validate_data,
)
from ipfs_accelerate_py.mcp_server.tools.data_processing_tools import native_data_processing_tools


def test_data_processing_package_exports_supported_native_functions() -> None:
    assert chunk_text is native_data_processing_tools.chunk_text
    assert transform_data is native_data_processing_tools.transform_data
    assert convert_format is native_data_processing_tools.convert_format
    assert validate_data is native_data_processing_tools.validate_data
