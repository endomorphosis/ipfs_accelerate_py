#!/usr/bin/env python3
"""UNI-220 analysis import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.analysis_tools import (
    analyze_data_distribution,
    cluster_analysis,
    detect_outliers,
    dimensionality_reduction,
    quality_assessment,
)
from ipfs_accelerate_py.mcp_server.tools.analysis_tools import native_analysis_tools


def test_analysis_package_exports_supported_native_functions() -> None:
    assert analyze_data_distribution is native_analysis_tools.analyze_data_distribution
    assert cluster_analysis is native_analysis_tools.cluster_analysis
    assert quality_assessment is native_analysis_tools.quality_assessment
    assert detect_outliers is native_analysis_tools.detect_outliers
    assert dimensionality_reduction is native_analysis_tools.dimensionality_reduction