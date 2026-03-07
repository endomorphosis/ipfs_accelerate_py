#!/usr/bin/env python3
"""UNI-197 dashboard import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.dashboard_tools import (
    check_tdfol_performance_regression,
    compare_tdfol_strategies,
    export_tdfol_statistics,
    generate_tdfol_dashboard,
    get_tdfol_metrics,
    get_tdfol_profiler_report,
    profile_tdfol_operation,
    reset_tdfol_metrics,
)
from ipfs_accelerate_py.mcp_server.tools.dashboard_tools import native_dashboard_tools


def test_dashboard_package_exports_supported_native_functions() -> None:
    assert get_tdfol_metrics is native_dashboard_tools.get_tdfol_metrics
    assert profile_tdfol_operation is native_dashboard_tools.profile_tdfol_operation
    assert generate_tdfol_dashboard is native_dashboard_tools.generate_tdfol_dashboard
    assert export_tdfol_statistics is native_dashboard_tools.export_tdfol_statistics
    assert get_tdfol_profiler_report is native_dashboard_tools.get_tdfol_profiler_report
    assert compare_tdfol_strategies is native_dashboard_tools.compare_tdfol_strategies
    assert check_tdfol_performance_regression is native_dashboard_tools.check_tdfol_performance_regression
    assert reset_tdfol_metrics is native_dashboard_tools.reset_tdfol_metrics
