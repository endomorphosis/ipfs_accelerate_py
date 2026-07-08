#!/usr/bin/env python3
"""UNI-217 file detection import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.file_detection_tools import (
    analyze_detection_accuracy,
    batch_detect_file_types,
    detect_file_type,
    generate_detection_report,
)
from ipfs_accelerate_py.mcp_server.tools.file_detection_tools import native_file_detection_tools


def test_file_detection_package_exports_supported_native_functions() -> None:
    assert detect_file_type is native_file_detection_tools.detect_file_type
    assert batch_detect_file_types is native_file_detection_tools.batch_detect_file_types
    assert analyze_detection_accuracy is native_file_detection_tools.analyze_detection_accuracy
    assert generate_detection_report is native_file_detection_tools.generate_detection_report